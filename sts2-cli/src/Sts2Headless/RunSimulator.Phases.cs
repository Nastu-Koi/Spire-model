using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

public partial class RunSimulator
{
    // Opening the map is a view change. EnterMapCoord performs the actual room exit.
    private bool _protocolMapVisible;
    private readonly HashSet<RestSiteRoom> _protocolCompletedRestSites = new(ReferenceEqualityComparer.Instance);

    private Dictionary<string, object?> PublishSnapshot(string phase, PublicSnapshot snapshot,
        List<CandidateBinding> bindings) => PublishDecision(phase, snapshot.Entities, bindings,
            relations: snapshot.Relations);

    private bool CanActInCombat() => CombatManager.Instance.IsInProgress && IsPlayPhase()
        && !CombatManager.Instance.PlayerActionsDisabled && !HasPendingInteraction();

    private Dictionary<string, object?> PublishCombatBoundary()
    {
        if (!CanActInCombat()) return NonDecisionBoundary("waiting");
        var snapshot = BuildPublicSnapshot();
        var player = _runState!.Players[0];
        var bindings = new List<CandidateBinding>();
        foreach (var card in player.PlayerCombatState!.Hand.Cards)
        {
            var source = snapshot.Cards[card];
            foreach (var target in new Creature?[] { null }.Concat(snapshot.Creatures.Keys))
            {
                if (!card.CanPlayTargeting(target)) continue;
                string? targetRef = target == null ? null : snapshot.Creatures[target];
                bindings.Add(new(Candidate("PLAY_CARD", $"play:{source}:{targetRef ?? "none"}", source, targetRef),
                    () => CanActInCombat() && player.PlayerCombatState.Hand.Cards.Contains(card)
                        && card.CanPlayTargeting(target),
                    () => RunManager.Instance.ActionQueueSet.EnqueueWithoutSynchronizing(new PlayCardAction(card, target))));
            }
        }
        bindings.Add(new(Candidate("END_TURN", "end_turn"), CanActInCombat, () =>
        {
            _turnStarted.Reset();
            _combatEnded.Reset();
            YieldPatches.SuppressYield = true;
            try
            {
                PlayerCmd.EndTurn(player, canBackOut: false);
                for (int i = 0; i < 1000; i++)
                {
                    _syncCtx.Pump();
                    if (HasPendingInteraction() || _turnStarted.IsSet || _combatEnded.IsSet
                        || player.Creature.IsDead || !CombatManager.Instance.IsInProgress) break;
                    Thread.Sleep(1);
                }
            }
            finally { YieldPatches.SuppressYield = false; }
        }));
        AddPotionCandidates(snapshot, bindings);
        return PublishSnapshot("combat", snapshot, bindings);
    }

    private void AddPotionCandidates(PublicSnapshot snapshot, List<CandidateBinding> bindings)
    {
        var player = _runState!.Players[0];
        foreach (var (potion, source) in snapshot.Potions)
        {
            bool Present() => player.CanUseOrRemovePotions && player.Potions.Contains(potion);
            bool Usable() => Present() && (potion.Usage == PotionUsage.AnyTime
                || potion.Usage == PotionUsage.CombatOnly && CanActInCombat())
                && potion.PassesCustomUsabilityCheck;
            if (!Present()) continue;
            bindings.Add(new(Candidate("DISCARD_POTION", "discard:" + source, source), Present,
                () => _pendingOperation.Start("discard potion", () => PotionCmd.Discard(potion))));
            if (!Usable()) continue;
            foreach (var target in new Creature?[] { null }.Concat(snapshot.Creatures.Keys))
            {
                if (!potion.IsValidTarget(target)) continue;
                string? targetRef = target == null ? null : snapshot.Creatures[target];
                bindings.Add(new(Candidate("USE_POTION", $"use:{source}:{targetRef ?? "none"}", source, targetRef),
                    () => Usable() && potion.IsValidTarget(target),
                    () => RunManager.Instance.ActionQueueSet.EnqueueWithoutSynchronizing(
                        new UsePotionAction(potion, target, CombatManager.Instance.IsInProgress))));
            }
        }
    }

    private IReadOnlyList<MapPoint> ProtocolTravelablePoints()
    {
        var state = _runState!;
        var map = state.Map;
        if (!Hook.ShouldProceedToNextMapPoint(state)) return [];
        if (state.VisitedMapCoords.Count == 0) return [map.StartingMapPoint];
        var last = state.VisitedMapCoords.Last();
        if (map.SecondBossMapPoint is { } second && last == map.BossMapPoint.coord) return [second];
        if (last.row == map.GetRowCount() - 1) return [map.BossMapPoint];
        var point = map.GetPoint(last) ?? throw new InvalidOperationException("Current map coordinate is absent.");
        return MapTravel.GetTravelablePointsFrom(state, point).ToList();
    }

    private Dictionary<string, object?> PublishMapBoundary()
    {
        var snapshot = BuildPublicSnapshot();
        var bindings = new List<CandidateBinding>();
        foreach (var point in ProtocolTravelablePoints())
        {
            var reference = snapshot.Map[point.coord];
            bindings.Add(new(Candidate("MOVE_TO_NODE", "move:" + reference, reference),
                () => ProtocolTravelablePoints().Any(p => p.coord == point.coord), () =>
                {
                    _protocolMapVisible = false;
                    _pendingOperation.Start("map travel", () => RunManager.Instance.EnterMapCoord(point.coord));
                }));
        }
        AddPotionCandidates(snapshot, bindings);
        return PublishSnapshot("map", snapshot, bindings);
    }

    private Dictionary<string, object?> PublishEventBoundary()
    {
        var localEvent = RunManager.Instance.EventSynchronizer.GetLocalEvent();
        if (localEvent == null) return ProtocolError("missing_local_event");
        if (localEvent is FakeMerchant fake) return PublishMerchantBoundary(_runState!.CurrentRoom!, fake.Inventory);
        if (localEvent.IsFinished) return OpenProtocolMap();
        var snapshot = BuildPublicSnapshot();
        snapshot.Entities.Add(new() { ["entity_type"] = "event", ["content_id"] = localEvent.Id.ToString() });
        var bindings = new List<CandidateBinding>();
        foreach (var (option, index) in localEvent.CurrentOptions.Select((o, i) => (o, i)))
        {
            var reference = $"option:{index}";
            snapshot.Entities.Add(new() { ["entity_type"] = "event_option", ["ref"] = reference,
                ["content_id"] = option.TextKey, ["enabled"] = !option.IsLocked,
                ["effect_coverage"] = "opaque", ["semantic_program"] = OpaqueProgram(option.TextKey) });
            AddDisplayedVariables(snapshot, reference, option.Title, option.Description);
            if (option.IsLocked) continue;
            bool abandon = localEvent is Trial && option.TextKey == "TRIAL.pages.REJECT.options.DOUBLE_DOWN";
            bindings.Add(new(Candidate(abandon ? "ABANDON_RUN" : "CHOOSE_EVENT_OPTION", "event:" + reference, reference),
                () => ReferenceEquals(RunManager.Instance.EventSynchronizer.GetLocalEvent(), localEvent)
                    && !localEvent.IsFinished && localEvent.CurrentOptions.Contains(option) && !option.IsLocked,
                () =>
                {
                    if (abandon) RunManager.Instance.Abandon();
                    else _pendingOperation.Start("event option", option.Chosen);
                }));
        }
        if (bindings.Count == 0) return ProtocolError("unsupported_event_view:" + localEvent.Id.Entry);
        AddPotionCandidates(snapshot, bindings);
        return PublishSnapshot("event", snapshot, bindings);
    }

    private Dictionary<string, object?> PublishRestBoundary(RestSiteRoom room)
    {
        if (room.Options.Count == 0) return OpenProtocolMap();
        var snapshot = BuildPublicSnapshot();
        var bindings = new List<CandidateBinding>();
        foreach (var (option, index) in room.Options.Select((o, i) => (o, i)))
        {
            var reference = $"rest:{index}";
            snapshot.Entities.Add(new() { ["entity_type"] = "rest_option", ["ref"] = reference,
                ["content_id"] = option.OptionId, ["enabled"] = option.IsEnabled,
                ["effect_coverage"] = "opaque", ["semantic_program"] = OpaqueProgram(option.OptionId) });
            AddDisplayedVariables(snapshot, reference, option.Title, option.Description);
            if (!option.IsEnabled) continue;
            bindings.Add(new(Candidate("CHOOSE_REST_OPTION", reference, reference),
                () => ReferenceEquals(_runState!.CurrentRoom, room) && room.Options.Contains(option) && option.IsEnabled,
                () => _pendingOperation.Start("rest option", async () =>
                {
                    if (await RunManager.Instance.RestSiteSynchronizer.ChooseLocalOption(index))
                        _protocolCompletedRestSites.Add(room);
                })));
        }
        // The native view enables Proceed only after a choice (or no options).
        if (_protocolCompletedRestSites.Contains(room))
            bindings.Add(new(Candidate("LEAVE_ROOM", "leave_rest"), () => ReferenceEquals(_runState!.CurrentRoom, room),
                () => _protocolMapVisible = true));
        AddPotionCandidates(snapshot, bindings);
        return PublishSnapshot("rest_site", snapshot, bindings);
    }

    private Dictionary<string, object?> OpenProtocolMap()
    {
        _protocolMapVisible = true;
        return PublishMapBoundary();
    }
}
