using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading;
using HarmonyLib;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Odds;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Spire.PublicHistory;

// Hooks observe public rule inputs and results. In particular they do not read
// AbstractOdds.CurrentValue, private AI moves, insertion indices or RNG state.
public static class NativeHistory
{
    private sealed class RunHistory
    {
        public readonly OddsHistory Odds;
        public bool? PreviousPointHadShop;
        public RunHistory(RunState run, bool fresh)
        {
            Odds = new(fresh, run.CurrentActIndex, run.UnlockState.NumberOfRuns == 0,
                run.Modifiers.Any(m => m.Id.Entry == "DEADLY_EVENTS"));
            PreviousPointHadShop = fresh ? false : null;
        }
    }
    private sealed class CombatHistory
    {
        public CombatHistory() { }
        public bool ObservedStart;
        public Dictionary<Creature, List<Dictionary<string, object?>>> Previous = new();
        public Dictionary<Creature, List<Dictionary<string, object?>>>? Pending;
    }
    private sealed record AddScope(CardPile Pile, Placement Position);
    private static readonly object Gate = new();
    private static readonly ConditionalWeakTable<RunState, RunHistory> Runs = new();
    private static readonly ConditionalWeakTable<CombatState, CombatHistory> Combats = new();
    private static readonly ConditionalWeakTable<CardPile, DrawKnowledge<CardModel>> Draws = new();
    private static readonly AsyncLocal<AddScope?> Adding = new();
    private static readonly AsyncLocal<bool> Shuffling = new();
    private static bool _installed;

    public static void Install(Harmony harmony)
    {
        if (_installed) return;
        void Patch(Type type, string name, string? prefix = null, string? postfix = null, Type[]? args = null)
        {
            var method = AccessTools.Method(type, name, args) ?? throw new MissingMethodException(type.FullName, name);
            harmony.Patch(method, prefix == null ? null : new HarmonyMethod(typeof(NativeHistory), prefix),
                postfix == null ? null : new HarmonyMethod(typeof(NativeHistory), postfix));
        }
        foreach (string name in new[] { "SetUpNewSingleplayer", "SetUpNewMultiplayer", "SetUpTest" })
            Patch(typeof(RunManager), name, prefix: nameof(FreshRun));
        foreach (string name in new[] { "SetUpSavedSingleplayer", "SetUpSavedMultiplayer" })
            Patch(typeof(RunManager), name, prefix: nameof(LoadedRun));
        Patch(typeof(RunManager), "SetActInternal", prefix: nameof(EnterAct));
        Patch(typeof(RunState), "AppendToMapPointHistory", postfix: nameof(PointEntered));
        Patch(typeof(RunState), "PushRoom", postfix: nameof(RoomEntered));
        Patch(typeof(PotionRewardOdds), "Roll", postfix: nameof(PotionRolled));
        Patch(typeof(UnknownMapPointOdds), "Roll", postfix: nameof(QuestionRolled));
        Patch(typeof(CardPileCmd), "Add", nameof(BeginAdd), nameof(EndAdd),
            new[] { typeof(IEnumerable<CardModel>), typeof(CardPile), typeof(CardPilePosition),
                typeof(AbstractModel), typeof(bool), typeof(bool) });
        Patch(typeof(CardPileCmd), "Shuffle", nameof(BeginShuffle), nameof(EndShuffle));
        Patch(typeof(CardPile), "AddInternal", postfix: nameof(CardAdded));
        Patch(typeof(CardPile), "RemoveInternal", prefix: nameof(CardRemoving));
        foreach (string name in new[] { "RandomizeOrderInternal", "MoveToTopInternal", "MoveToBottomInternal" })
            Patch(typeof(CardPile), name, postfix: nameof(PileRandomized));
        CombatManager.Instance.CombatSetUp += CombatStarted;
        CombatManager.Instance.TurnStarted += TurnStarted;
        CombatManager.Instance.TurnEnded += TurnEnded;
        _installed = true;
    }

    private static RunHistory Track(RunState run) => Runs.GetValue(run, r => new(r, false));
    private static void FreshRun(RunState state) { lock (Gate) { Runs.Remove(state); Runs.Add(state, new(state, true)); } }
    private static void LoadedRun(RunState state) { lock (Gate) { Runs.Remove(state); Runs.Add(state, new(state, false)); } }
    private static void EnterAct(RunManager __instance, int actIndex)
    { lock (Gate) { if (__instance.DebugOnlyGetState() is RunState run) Track(run).Odds.EnterAct(actIndex); } }
    private static void PointEntered(RunState __instance, RoomType initialRoomType)
    { lock (Gate) Track(__instance).PreviousPointHadShop = initialRoomType == RoomType.Shop; }
    private static void RoomEntered(RunState __instance, AbstractRoom room)
    {
        // Includes an event's nested merchant. Reading the hidden history of a
        // loaded save would incorrectly turn an unobserved previous room into fact.
        if (room.RoomType == RoomType.Shop) lock (Gate) Track(__instance).PreviousPointHadShop = true;
    }

    private static bool HasRelic(Player p, string id) => p.Relics.Any(r => r.Id.Entry == id);
    private static bool ForcedPotion(Player p, RoomType type) => HasRelic(p, "WHITE_BEAST_STATUE")
        && type is RoomType.Monster or RoomType.Elite or RoomType.Boss;
    private static void PotionRolled(Player player, RoomType roomType, bool __result)
    {
        lock (Gate)
            if (player.RunState is RunState run && run.Players.Count == 1)
                Track(run).Odds.PotionGenerated(__result, ForcedPotion(player, roomType));
    }

    private static HashSet<string> Eligible(RunState run, IEnumerable<RoomType> blacklist)
    {
        var eligible = OddsHistory.RoomTypes.Except(blacklist.Select(t => t.ToString())).ToHashSet();
        if (run.Players.Any(p => HasRelic(p, "JUZU_BRACELET"))) eligible.Remove("Monster");
        // The golden path is displayed on the map. Its public layout identifies
        // the active effect without consulting GoldenCompass.GoldenPathAct.
        bool goldenMap = run.Map?.GetType().Name == "GoldenPathActMap"
            && run.Players.Any(p => HasRelic(p, "GOLDEN_COMPASS"));
        bool lantern = run.CurrentActIndex == 2 && run.Players.Any(p => p.Deck.Cards.Any(c => c.Id.Entry == "LANTERN_KEY"));
        if (goldenMap || lantern) return new() { "Event" };
        return eligible;
    }
    private static void QuestionRolled(IEnumerable<RoomType> blacklist, IRunState runState, RoomType __result)
    {
        lock (Gate)
            if (runState is RunState run)
                Track(run).Odds.QuestionRoomGenerated(__result.ToString(), Eligible(run, blacklist));
    }

    private static void BeginAdd(CardPile newPile, CardPilePosition position, out AddScope? __state)
    {
        __state = Adding.Value;
        Adding.Value = new(newPile, Shuffling.Value ? Placement.Unknown : position switch
        { CardPilePosition.Top => Placement.Top, CardPilePosition.Bottom => Placement.Bottom, _ => Placement.Unknown });
    }
    private static void EndAdd(AddScope? __state) => Adding.Value = __state;
    private static void BeginShuffle(Player player, out bool __state)
    {
        __state = Shuffling.Value;
        Shuffling.Value = true;
        if (player.PlayerCombatState is { } pcs) PileRandomized(pcs.DrawPile);
    }
    private static void EndShuffle(bool __state) => Shuffling.Value = __state;
    // AsyncLocal scopes flow through the original Tasks, including paused card
    // selections. Postfix restores the caller immediately; no synchronous wait.
    private static void CardAdded(CardPile __instance, CardModel card)
    {
        if (__instance.Type != PileType.Draw) return;
        lock (Gate)
            Draws.GetOrCreateValue(__instance).Added(card,
                Adding.Value is { } scope && ReferenceEquals(scope.Pile, __instance) ? scope.Position : Placement.Unknown,
                __instance.Cards.Count - 1);
    }
    private static void CardRemoving(CardPile __instance, CardModel card)
    {
        if (__instance.Type != PileType.Draw || !__instance.Cards.Contains(card)) return;
        lock (Gate) Draws.GetOrCreateValue(__instance).Removed(card, __instance.Cards.Count);
    }
    private static void PileRandomized(CardPile __instance)
    {
        if (__instance.Type == PileType.Draw)
            lock (Gate) Draws.GetOrCreateValue(__instance).Invalidate(__instance.Cards.Count);
    }

    private static void CombatStarted(CombatState state)
    {
        lock (Gate) { Combats.Remove(state); Combats.Add(state, new() { ObservedStart = true }); }
    }
    private static void TurnStarted(CombatState state)
    {
        lock (Gate)
        {
            var history = Combats.GetOrCreateValue(state);
            if (state.CurrentSide == CombatSide.Enemy)
                history.Pending = state.Enemies.ToDictionary(c => c, c => VisibleIntents(c, state));
        }
    }
    private static void TurnEnded(CombatState state)
    {
        // SwitchSides fires this before the next hand draw. A draw-triggered
        // selection already needs the previous enemy turn's visible intent.
        lock (Gate)
        {
            var history = Combats.GetOrCreateValue(state);
            if (state.CurrentSide == CombatSide.Player && history.Pending is { } pending)
            { history.Previous = pending; history.Pending = null; }
        }
    }
    private static List<Dictionary<string, object?>> VisibleIntents(Creature creature, CombatState state)
    {
        var result = new List<Dictionary<string, object?>>();
        foreach (var intent in creature.Monster?.NextMove?.Intents ?? [])
        {
            var row = new Dictionary<string, object?> { ["intent"] = intent.IntentType.ToString() };
            if (intent is AttackIntent attack)
            {
                var targets = state.PlayerCreatures.ToList();
                row["damage"] = attack.GetSingleDamage(targets, creature);
                row["hits"] = attack.Repeats;
                row["total_damage"] = attack.GetTotalDamage(targets, creature);
            }
            result.Add(row);
        }
        return result;
    }

    public static List<Dictionary<string, object?>> Capture(RunState run, Func<object, string?> reference)
    {
        lock (Gate)
        {
            var memory = new List<Dictionary<string, object?>>();
            var history = Track(run);
            var player = run.Players[0];
            memory.Add(new() { ["entity_type"] = "history_rule", ["content_id"] = "question_previous_shop",
                ["enabled"] = new { value = history.PreviousPointHadShop, known = history.PreviousPointHadShop.HasValue, applicable = true } });
            foreach (var type in new[] { RoomType.Monster, RoomType.Elite, RoomType.Boss, RoomType.Event })
                memory.Add(Probability("potion_drop", type.ToString(),
                    history.Odds.PotionProbability(type == RoomType.Elite, ForcedPotion(player, type))));
            // Shop eligibility depends on the chosen node's children and the
            // previous room. Export both conditional distributions, never pretend
            // that a single unconditional distribution applies to every branch.
            foreach (bool blocked in new[] { false, true })
                foreach (var pair in history.Odds.RoomProbabilities(Eligible(run, blocked ? new[] { RoomType.Shop } : Array.Empty<RoomType>())))
                    memory.Add(Probability("question_room", (blocked ? "shop_blocked:" : "shop_allowed:") + pair.Key, pair.Value));
            var combat = player.Creature.CombatState as CombatState;
            if (!CombatManager.Instance.IsInProgress || combat == null) return memory;
            var previous = Combats.GetOrCreateValue(combat);
            foreach (var enemy in combat.Enemies)
            {
                string? owner = reference(enemy);
                if (owner == null) continue;
                if (previous.Previous.TryGetValue(enemy, out var intents))
                {
                    foreach (var intent in intents)
                        memory.Add(new(intent) { ["entity_type"] = "previous_intent", ["owner_ref"] = owner,
                            ["known"] = true, ["applicable"] = true });
                    if (intents.Count > 0) continue;
                }
                memory.Add(new() { ["entity_type"] = "previous_intent", ["owner_ref"] = owner,
                    ["known"] = previous.ObservedStart || previous.Previous.ContainsKey(enemy),
                    ["applicable"] = !previous.ObservedStart && !previous.Previous.ContainsKey(enemy) });
            }
            if (player.PlayerCombatState is { } pcs)
                foreach (var (card, position) in Draws.GetOrCreateValue(pcs.DrawPile).KnownPositions(pcs.DrawPile.Cards.Count))
                    if (reference(card) is string owner)
                        memory.Add(new() { ["entity_type"] = "known_draw_position", ["owner_ref"] = owner,
                            ["position"] = position, ["known"] = true });
            return memory;
        }
    }

    private static Dictionary<string, object?> Probability(string kind, string scope, float? value) => new()
    {
        ["entity_type"] = "history_probability", ["content_id"] = kind, ["scope"] = scope,
        ["probability"] = new { value, known = value.HasValue, applicable = true },
    };
}
