using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Rewards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;

namespace RunRecorder;

internal static class SelectionCapture
{
    private static readonly int MainThread = Environment.CurrentManagedThreadId;
    private static RunState? _run;

    // Selection tracking belongs to both features. Turning off journal writes
    // must not erase the offer that the controller is currently choosing from.
    internal static void Safe(Action action)
    {
        if (Environment.CurrentManagedThreadId != MainThread) return;
        try { action(); }
        catch (Exception ex) { Godot.GD.PushError("[RunRecorder selection] " + ex); }
    }

    internal static bool Prepare(Player? player)
    {
        var manager = RunManager.Instance;
        var run = manager?.DebugOnlyGetState();
        if (Environment.CurrentManagedThreadId != MainThread || player == null || run == null ||
            manager!.NetService == null || manager.IsCleaningUp ||
            !manager.IsSingleplayerOrFakeMultiplayer || run.Players.Count != 1 || run.Players[0] != player)
            return false;
        if (run != _run) { Reset(); _run = run; }
        return true;
    }

    private static void Capture(SelectionScope scope, SelectionOffer offer)
    {
        offer = offer with { Metadata = offer.Metadata ?? scope.Metadata };
        scope.Offer = offer;
        scope.CapturedUtc = DateTimeOffset.UtcNow;
        Recorder.CaptureSelection(scope, offer);
    }

    internal static void Tick()
    {
        if (_run != null && RunManager.Instance?.DebugOnlyGetState() != _run) Reset();
        Prune();
    }

    internal static SelectionScope? Active => Choices.Values
        .Where(s => s.Offer != null && s.Task?.IsCompleted != true)
        .OrderByDescending(s => s.CapturedUtc).FirstOrDefault();
	private static readonly AsyncLocal<SelectionScope?> Current = new AsyncLocal<SelectionScope>();

	private static readonly Dictionary<(ulong, uint), SelectionScope> Choices = new Dictionary<(ulong, uint), SelectionScope>();

	private static ConditionalWeakTable<object, SelectionScope> RewardScreens = new ConditionalWeakTable<object, SelectionScope>();
    private static ConditionalWeakTable<object, SelectionScope> CombatPileScreens = new();

    internal static void BindCombatPileScreen(object screen)
    {
        if (Current.Value is not { } scope) return;
        CombatPileScreens.Remove(screen);
        CombatPileScreens.Add(screen, scope);
        RefreshCombatPileScreen(screen);
    }

    internal static void RefreshCombatPileScreen(object screen)
    {
        if (!CombatPileScreens.TryGetValue(screen, out var scope) || scope.Task?.IsCompleted == true ||
            Snapshot.Read(Snapshot.Read(screen, "_completionSource"), "Task") is Task { IsCompleted: true } ||
            Snapshot.Read(screen, "_pile") is not CardPile pile ||
            Snapshot.Read(screen, "_prefs") is not CardSelectorPrefs prefs) return;
        var filter = Snapshot.Read(screen, "_filter") as Func<CardModel, bool>;
        IEnumerable<CardModel> cards = pile.Cards.Where(card => filter == null || filter(card));
        // Draw-pile selections expose the selectable cards, never their draw order.
        if (pile.Type == PileType.Draw) cards = cards.OrderBy(card => card.Rarity).ThenBy(card => card.Id);
        var offered = cards.ToArray();
        var offer = new SelectionOffer(offered, Math.Min(prefs.MinSelect, offered.Length),
            Math.Min(prefs.MaxSelect, offered.Length), prefs.Cancelable)
        {
            Metadata = SelectionMetadata.Derive(scope.Source, prefs, pile.Type, scope.Metadata)
        };
        var previous = scope.Offer;
        if (previous != null && previous.Min == offer.Min && previous.Max == offer.Max &&
            previous.Cancelable == offer.Cancelable && previous.Cards.SequenceEqual(offered) &&
            previous.Metadata == offer.Metadata) return;
        Capture(scope, offer);
    }

	internal static void BindRewardScreen(object? screen)
	{
		if (screen != null)
		{
			SelectionScope value = Current.Value;
			if (value != null)
			{
				RewardScreens.Remove(screen);
				RewardScreens.Add(screen, value);
			}
		}
	}

	internal static void RefreshRewardScreen(object screen, CardCreationResult[] cards, CardRewardAlternative[] alternatives)
	{
		if (!RewardScreens.TryGetValue(screen, out SelectionScope value))
		{
			return;
		}
		Task? task = value.Task;
		if (task == null || !task.IsCompleted)
		{
			Capture(value, new SelectionOffer(cards.Select((CardCreationResult c) => c.Card).ToArray(), 1, 1,
                Cancelable: alternatives.Any(a => a.AfterSelected == PostAlternateCardRewardAction.EndSelectionAndDoNotCompleteReward),
                null, alternatives));
		}
	}

	internal static SelectionScope? Enter(MethodBase method, object[] args, object? source = null)
	{
		Player player = args.OfType<Player>().FirstOrDefault() ?? (source as Reward)?.Player;
		if (!Prepare(player))
		{
			return null;
		}
		SelectionScope selectionScope = new SelectionScope
		{
			Player = player,
			Source = method.Name,
			Parent = Current.Value,
            Metadata = SelectionMetadata.From(method, args, Current.Value?.Metadata)
		};
		Current.Value = selectionScope;
		return selectionScope;
	}

	internal static void Exit(SelectionScope? scope, object? result)
	{
		if (scope != null)
		{
			scope.Task = result as Task;
			if (Current.Value == scope)
			{
				Current.Value = scope.Parent;
			}
		}
	}

	internal static void Reserved(Player player, uint id)
	{
		SelectionScope value = Current.Value;
		if (value != null && value.Player == player)
		{
			Choices[(player.NetId, id)] = value;
		}
	}

	internal static void ObserveScreen(MethodBase method, object[] args)
	{
		if (Current.Value == null)
		{
			return;
		}
		IReadOnlyList<IReadOnlyList<CardModel>> readOnlyList = args.OfType<IReadOnlyList<IReadOnlyList<CardModel>>>().FirstOrDefault();
		if (readOnlyList != null)
		{
			Observe(new SelectionOffer(Array.Empty<CardModel>(), 1, 1, Cancelable: false, readOnlyList.Select((IReadOnlyList<CardModel> b) => b.ToArray()).ToArray()));
			return;
		}
		CardModel[] array = args.OfType<IReadOnlyList<CardModel>>().FirstOrDefault()?.ToArray() ?? args.OfType<IReadOnlyList<CardCreationResult>>().FirstOrDefault()?.Select((CardCreationResult c) => c.Card).ToArray();
		if (array != null)
		{
			if (method.DeclaringType?.Name == "NCardRewardSelectionScreen")
			{
                var alternatives = args.OfType<IReadOnlyList<CardRewardAlternative>>().FirstOrDefault()?.ToArray()
                    ?? Array.Empty<CardRewardAlternative>();
				Observe(new SelectionOffer(array, 1, 1,
                    Cancelable: alternatives.Any(a => a.AfterSelected == PostAlternateCardRewardAction.EndSelectionAndDoNotCompleteReward),
                    null, alternatives));
				return;
			}
			CardSelectorPrefs[] array2 = args.OfType<CardSelectorPrefs>().ToArray();
			bool cancelable = ((array2.Length != 0) ? array2[0].Cancelable : args.OfType<bool>().FirstOrDefault());
			Observe(new SelectionOffer(array, (array2.Length == 0) ? 1 : array2[0].MinSelect, (array2.Length == 0) ? 1 : array2[0].MaxSelect, cancelable)
            {
                Metadata = SelectionMetadata.Derive(Current.Value.Source,
                    array2.Length == 0 ? null : array2[0], previous: Current.Value.Metadata)
            });
		}
	}

	internal static Task<IEnumerable<CardModel>> GetSelectedCards(ICardSelector selector, IEnumerable<CardModel> cards, int min, int max)
	{
		CardModel[] array = cards.ToArray();
		Observe(new SelectionOffer(array, min, max, min == 0));
		return selector.GetSelectedCards(array, min, max);
	}

	internal static Task<IEnumerable<CardModel>> GetSelectedCardsWithPrefs(ICardSelector selector, IEnumerable<CardModel> cards, int min, int max, CardSelectorPrefs prefs)
	{
		CardModel[] array = cards.ToArray();
		Observe(new SelectionOffer(array, prefs.MinSelect, prefs.MaxSelect, prefs.Cancelable)
        {
            Metadata = SelectionMetadata.Derive(Current.Value?.Source ?? "unknown", prefs,
                previous: Current.Value?.Metadata)
        });
		return selector.GetSelectedCards(array, min, max);
	}

	internal static Task<IEnumerable<CardModel>> GetSelectedCardsWithSkip(ICardSelector selector, IEnumerable<CardModel> cards, int min, int max, bool canSkip)
	{
		CardModel[] array = cards.ToArray();
		Observe(new SelectionOffer(array, 1, 1, canSkip));
		return selector.GetSelectedCards(array, min, max);
	}

	internal static void Observe(SelectionOffer offer)
	{
		SelectionScope value = Current.Value;
		if (value != null && !(value.Offer != null))
		{
			Capture(value, offer);
		}
	}

	internal static void ObserveHand(NPlayerHand hand)
	{
		SelectionScope value = Current.Value;
		if (value == null || (object)value.Offer != null || !hand.IsInCardSelection || !(Snapshot.Read(hand, "_prefs") is CardSelectorPrefs cardSelectorPrefs))
		{
			return;
		}
		object obj = Snapshot.Read(hand.PeekButton, "IsPeeking");
		if ((!(obj is bool) || !(bool)obj) && Snapshot.Read(hand, "Holders") is IEnumerable<NHandCardHolder> source)
		{
			Observe(new SelectionOffer((from h in source
				where h.Visible && h.CardModel != null
				select h.CardModel).ToArray(), cardSelectorPrefs.MinSelect, cardSelectorPrefs.MaxSelect, cardSelectorPrefs.Cancelable)
            {
                Metadata = SelectionMetadata.Derive("FromHand", cardSelectorPrefs, previous: value.Metadata)
            });
		}
	}

    internal static void ObserveHandEntry(CardSelectorPrefs prefs, Func<CardModel, bool>? filter)
    {
        var scope = Current.Value;
        if (scope == null || scope.Offer != null || scope.Player.PlayerCombatState == null) return;
        // Capture the engine's offered hand before another mod can submit/close
        // the UI. Re-reading Holders on submission would already contain the answer.
        var cards = scope.Player.PlayerCombatState.Hand.Cards
            .Where(card => filter == null || filter(card)).ToArray();
        Observe(new SelectionOffer(cards, prefs.MinSelect, prefs.MaxSelect, prefs.Cancelable)
        {
            Metadata = SelectionMetadata.Derive("FromHand", prefs, previous: scope.Metadata)
        });
    }

	internal static SelectionScope? Consume(Player player, uint id)
	{
		Choices.Remove((player.NetId, id), out SelectionScope value);
		return value;
	}

	internal static void Prune()
	{
		KeyValuePair<(ulong, uint), SelectionScope>[] array = Choices.Where<KeyValuePair<(ulong, uint), SelectionScope>>((KeyValuePair<(ulong, uint), SelectionScope> p) => p.Value.Task?.IsCompleted ?? false).ToArray();
		foreach (KeyValuePair<(ulong, uint), SelectionScope> keyValuePair in array)
		{
			Choices.Remove(keyValuePair.Key);
		}
	}

	internal static void Reset()
	{
		_run = null;
		Choices.Clear();
		RewardScreens = new ConditionalWeakTable<object, SelectionScope>();
        CombatPileScreens = new ConditionalWeakTable<object, SelectionScope>();
		Current.Value = null;
	}
}
