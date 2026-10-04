using System.Collections;
using System.Reflection;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;

namespace CombatSolverCli;

// The execution cursor belongs here, never to the search DLL's mutable selector.
// Occurrences in solver 0.44 count (id, upgrade), not full-state-key matches.
internal sealed class ChoiceTransaction : IDisposable
{
    internal sealed record Token(string Id, int Upgrade, string Key, int SourceOccurrence, int OptionOccurrence);
    internal sealed record Plan(string Source, string Effect, Token[] Cards);
    internal sealed record Snapshot(CardModel Card, string Id, int Upgrade, string Key);
    internal sealed class Request(long id, string entry, string source, string effect, int min, int max, bool cancelable, long? parentId)
    {
        internal readonly long? ParentId = parentId;
        internal readonly bool Cancelable = cancelable;
        internal bool Cancelled;
        internal readonly long Id = id;
        internal readonly string Entry = entry, Source = source, Effect = effect;
        internal readonly int Min = min, Max = max;
        internal Snapshot[]? Options;
        internal CardModel[]? Expected;
        internal int? PlanIndex;
        internal bool Completed, Explicit, NotOpened;
        internal readonly HashSet<CardModel> Moved = new();
    }

    private readonly object gate = new();
    private readonly Plan[] plan;
    private readonly Func<CardModel, string> key;
    internal readonly Player Player;
    private readonly List<Request> requests = new();
    private int reserved;
    private Exception? failure;
    private bool closed;
    private static long sequence;
    internal readonly long Id = Interlocked.Increment(ref sequence);

    private static object Read(object obj, string name) => obj.GetType().GetProperty(name,
        BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(obj)!;

    internal ChoiceTransaction(IEnumerable<object> choices, Player player, Func<CardModel, string> key)
    {
        Player = player; this.key = key;
        plan = choices.Select(c => new Plan(Read(c, "SourcePile").ToString()!, Read(c, "Effect").ToString()!,
            ((IEnumerable)Read(c, "Cards")).Cast<object>().Select(t => new Token(
                (string)Read(t, "CardId"), (int)Read(t, "UpgradeLevel"), (string)Read(t, "StateKey"),
                (int)Read(t, "SourceOccurrence"), (int)Read(t, "OptionOccurrence"))).ToArray())).ToArray();
    }

    internal Request BeginRequest(string entry, string source, string effect, int min, int max, bool cancelable = false, long? parentId = null)
    {
        lock (gate)
        {
            Check();
            var request = new Request(requests.Count + 1, entry, source, effect, min, max, cancelable, parentId);
            requests.Add(request);
            return request;
        }
    }

    internal void OpenRequest(Request request, IEnumerable<CardModel> candidates)
    {
        lock (gate)
        {
            Check();
            if (!requests.Contains(request)) throw Fail(request, "foreign request");
            if (request.Options != null) return; // Multiple native Count reads of the same materialized list.
            request.Options = candidates.Select(Snap).ToArray();
            // Native zero-candidate requests need no solver choice. Preserve the event,
            // but do not consume the following nonempty request's plan.
            if (request.Options.Length == 0 && (reserved == plan.Length || plan[reserved].Cards.Length != 0))
            { request.Expected = []; return; }
            if (reserved == plan.Length) throw Fail(request, "unexpected native request; plan exhausted");
            var expected = plan[reserved];
            if (request.Source != "unknown" && expected.Source != request.Source)
                throw Fail(request, $"source expected={expected.Source} actual={request.Source}");
            if (request.Effect != "unknown" && expected.Effect != request.Effect)
                throw Fail(request, $"effect expected={expected.Effect} actual={request.Effect}");
            bool sourceBacked = request.Source is not ("unknown" or "None");
            var source = !sourceBacked ? request.Options
                : Enum.Parse<PileType>(request.Source).GetPile(Player).Cards.Select(Snap).ToArray();
            var selected = new List<CardModel>();
            foreach (var token in expected.Cards)
            {
                bool Identity(Snapshot s) => s.Id == token.Id && s.Upgrade == token.Upgrade;
                var equivalents = request.Options.Where(Identity).ToArray();
                var original = source.Where(Identity).Skip(token.SourceOccurrence).FirstOrDefault();
                // Solver options may be shuffled while the native command filters
                // that subset in pile order. SourceOccurrence is the identity in
                // the complete pile; OptionOccurrence is a different coordinate.
                // Only generated choices (no source pile) use the option ordinal
                // to identify the instance. Never substitute another equal card.
                var option = sourceBacked
                    ? request.Options.FirstOrDefault(s => ReferenceEquals(s.Card, original?.Card))
                    : equivalents.Skip(token.OptionOccurrence).FirstOrDefault();
                if (token.SourceOccurrence < 0 || token.OptionOccurrence < 0 || token.OptionOccurrence >= equivalents.Length
                    || option == null || original == null || (!sourceBacked && !ReferenceEquals(option.Card, original.Card))
                    || (!string.IsNullOrEmpty(token.Key) && option.Key != token.Key))
                    throw Fail(request, $"target mismatch: {token.Id} source_occurrence={token.SourceOccurrence} option_occurrence={token.OptionOccurrence}; expected_key={token.Key}; actual_key={option?.Key}");
                if (selected.Contains(option.Card)) throw Fail(request, "duplicate planned target");
                selected.Add(option.Card);
            }
            // Native auto-select permits fewer than Min when there are insufficient options.
            if ((selected.Count < Math.Min(request.Min, request.Options.Length) && !(selected.Count == 0 && request.Cancelable)) || selected.Count > request.Max)
                throw Fail(request, $"selection count {selected.Count} outside {request.Min}..{request.Max}");
            request.Expected = selected.ToArray();
            request.PlanIndex = reserved++;
        }
    }

    internal CardModel[] Select(IEnumerable<CardModel> options, int min, int max)
    {
        lock (gate)
        {
            Check();
            var cards = options.ToArray();
            var matches = requests.Where(r => !r.Completed && r.Options != null
                && r.Options.Length == cards.Length && r.Options.All(s => cards.Contains(s.Card))).ToArray();
            if (matches.Length != 1) throw Fail(null, "explicit surface has no unique native request");
            var request = matches[0];
            if ((request.Expected!.Length < min && !(request.Expected.Length == 0 && request.Cancelable)) || request.Expected.Length > max)
                throw Fail(request, "explicit selection violates UI cardinality");
            request.Explicit = true;
            request.Cancelled = request.Expected.Length == 0 && min > 0 && request.Cancelable;
            return request.Expected.ToArray();
        }
    }

    internal void CompleteRequest(Request request, IEnumerable<CardModel> actual, bool singleCardResult = false)
    {
        lock (gate)
        {
            Check();
            if (!requests.Contains(request)) throw Fail(request, "foreign result");
            if (request.Completed) throw Fail(request, "duplicate result");
            var cards = actual.ToArray();
            // Scalar native APIs log [null] for no selection, while their Task
            // returns null. Decode both at this shared boundary, never filter
            // nulls out of arbitrary multi-card results.
            if (singleCardResult && cards.Length == 1 && cards[0] == null) cards = [];
            if (cards.Any(c => c == null) || singleCardResult && cards.Length > 1)
                throw Fail(request, "invalid result shape: actual=" + string.Join(",", cards.Select(c => c?.Id.Entry ?? "<null>")));
            if (request.Options == null)
            {
                // E.g. combat ended before native code opened its selection.
                if (cards.Length != 0) throw Fail(request, "result without candidate snapshot");
                request.NotOpened = true;
                request.Options = []; request.Expected = [];
            }
            if (!request.Expected!.SequenceEqual(cards))
                throw Fail(request, "result mismatch: expected=" + string.Join(",", request.Expected!.Select(c => c.Id.Entry))
                    + " actual=" + string.Join(",", cards.Select(c => c.Id.Entry)));
            request.Completed = true;
            Console.Error.WriteLine($"[CombatSolverCli] choice_verified transaction={Id} request={request.Id} entry={request.Entry} status={(request.NotOpened ? "not_opened" : request.Cancelled ? "cancelled" : request.Options.Length == 0 ? "no_candidates" : cards.Length == 0 ? "empty" : "selected")} mode={(request.Explicit ? "explicit" : "automatic")} cards={string.Join(",", cards.Select(c => c.Id.Entry))}");
        }
    }

    internal void ObserveMove(CardModel card, string destination, int index)
    {
        lock (gate)
        {
            Check();
            foreach (var request in requests.Where(r => r.Completed && r.PlanIndex != null && r.Expected!.Contains(card)))
            {
                if (!request.Moved.Add(card)) continue;
                var effect = plan[request.PlanIndex!.Value].Effect;
                var expected = effect switch { "Discard" => "Discard", "Exhaust" => "Exhaust",
                    "MoveToDrawTop" or "MoveToDrawBottom" => "Draw", "MoveToHand" => "Hand", _ => null };
                if (expected != null && (destination != expected || effect == "MoveToDrawTop" && index != 0))
                    throw Fail(request, $"effect {effect} contradicts native movement to {destination}[{index}]");
            }
        }
    }

    internal void Fault(Exception error) { lock (gate) failure ??= error; }
    internal void Check()
    {
        lock (gate)
        {
            if (failure != null) throw new InvalidOperationException($"choice_transaction_failed transaction={Id}", failure);
            if (closed) throw new InvalidOperationException("choice_transaction_closed");
        }
    }
    private Snapshot Snap(CardModel card) => new(card, card.Id.Entry, card.CurrentUpgradeLevel, key(card));
    private Exception Fail(Request? request, string message)
    {
        var error = new InvalidOperationException($"choice_transaction_mismatch transaction={Id} request={request?.Id} entry={request?.Entry}: {message}");
        failure ??= error;
        return error;
    }
    internal void FinishAction(bool combatEnded = false)
    {
        lock (gate)
        {
            Check();
            if ((!combatEnded && reserved != plan.Length) || requests.Any(r => !r.Completed))
                throw Fail(null, $"unfinished: planned={plan.Length} reserved={reserved} pending={requests.Count(r => !r.Completed)}");
            if (reserved != plan.Length)
                Console.Error.WriteLine($"[CombatSolverCli] choice_action_cancelled_by_combat_end transaction={Id} unrequested={plan.Length - reserved} requests={requests.Count}");
            else
                Console.Error.WriteLine($"[CombatSolverCli] choice_action_verified transaction={Id} requests={requests.Count}");
            closed = true;
        }
    }
    public void Dispose() { lock (gate) closed = true; }
}
