using System.Threading.Channels;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Rewards;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

public partial class RunSimulator
{
    private sealed class ProtocolRewardMenu(RewardsSet set)
    {
        public readonly RewardsSet Set = set;
        public readonly Channel<Func<Task>> Commands = Channel.CreateUnbounded<Func<Task>>();
        public volatile bool Busy;
        public void Submit(Func<Task> command)
        {
            if (Busy) throw new InvalidOperationException("Reward selection already in progress.");
            Busy = true;
            if (!Commands.Writer.TryWrite(command)) throw new InvalidOperationException("Reward menu closed.");
        }
    }

    private volatile ProtocolRewardMenu? _protocolRewards;
    private readonly HashSet<CombatRoom> _protocolRewardsOffered = new(ReferenceEqualityComparer.Instance);
    private bool _protocolEnabled;
    private bool HasProtocolRewardMenu => _protocolRewards is { Busy: false };

    private async Task SelectProtocolRewards(RewardsSet set)
    {
        var parent = _protocolRewards;
        var menu = new ProtocolRewardMenu(set);
        _protocolRewards = menu;
        try
        {
            var synchronizer = RunManager.Instance.RewardsSetSynchronizer;
            while (!synchronizer.IsRewardsSetCompleted(set))
            {
                menu.Busy = false;
                _pendingOperation.NotifyProgress();
                var command = await menu.Commands.Reader.ReadAsync();
                await command();
            }
        }
        finally
        {
            menu.Commands.Writer.TryComplete();
            _protocolRewards = parent;
            _pendingOperation.NotifyProgress();
        }
    }

    private bool CanTakeReward(Reward reward)
    {
        if (reward.SuccessfullySelected) return false;
        if (reward is PotionReward { Potion: { } potion })
            return reward.Player.HasOpenPotionSlots && Hook.ShouldProcurePotion(_runState!,
                reward.Player.Creature.CombatState, potion, reward.Player);
        return true;
    }

    private Dictionary<string, object?> PublishRewardsBoundary(ProtocolRewardMenu menu)
    {
        var snapshot = BuildPublicSnapshot();
        var bindings = new List<CandidateBinding>();
        foreach (var (reward, index) in menu.Set.Rewards.Select((r, i) => (r, i)))
        {
            if (reward.SuccessfullySelected) continue;
            string reference = $"reward:{index}";
            var entity = new Dictionary<string, object?> { ["entity_type"] = "reward", ["ref"] = reference,
                ["content_id"] = reward.GetType().Name, ["effect_coverage"] = "opaque" };
            if (reward is GoldReward gold) entity["gold"] = gold.Amount;
            if (reward is RelicReward relic) entity["content_id"] = relic.Relic?.Id.ToString();
            if (reward is PotionReward potion) entity["content_id"] = potion.Potion?.Id.ToString();
            entity["semantic_program"] = OpaqueProgram((string?)entity["content_id"] ?? "reward");
            snapshot.Entities.Add(entity);
            bool Legal() => ReferenceEquals(_protocolRewards, menu) && !menu.Busy && CanTakeReward(reward);
            if (!Legal()) continue;
            if (reward is CardReward cards)
            {
                // Expanding the native offer is free inspection. Only the final card
                // or effectful alternative becomes an action; opening a panel is not.
                foreach (var card in cards.Cards)
                {
                    var cardRef = snapshot.AddCard(card, "reward");
                    snapshot.Relations.Add(new { source = reference, target = cardRef, role = "offers" });
                    bindings.Add(new(Candidate("TAKE_CARD_REWARD", $"take:{reference}:{cardRef}", cardRef),
                        () => Legal() && cards.Cards.Contains(card), () => menu.Submit(async () =>
                        {
                            _cardSelector.PreselectedRewardCard = card;
                            await RunManager.Instance.RewardsSetSynchronizer.SelectLocalReward(reward);
                        })));
                }
                foreach (var alternate in CardRewardAlternative.Generate(cards))
                {
                    if (alternate.AfterSelected == PostAlternateCardRewardAction.EndSelectionAndDoNotCompleteReward) continue;
                    var altRef = $"{reference}:alternative:{alternate.OptionId}";
                    snapshot.Entities.Add(new() { ["entity_type"] = "reward_alternative", ["ref"] = altRef,
                        ["content_id"] = alternate.OptionId, ["effect_coverage"] = "opaque",
                        ["semantic_program"] = OpaqueProgram(alternate.OptionId) });
                    bindings.Add(new(Candidate("CHOOSE_REWARD_ALTERNATIVE", altRef, altRef), Legal,
                        () => menu.Submit(async () =>
                        {
                            _cardSelector.PreselectedRewardAlternative = alternate.OptionId;
                            await RunManager.Instance.RewardsSetSynchronizer.SelectLocalReward(reward);
                        })));
                }
            }
            else bindings.Add(new(Candidate("TAKE_REWARD", "take:" + reference, reference), Legal,
                () => menu.Submit(async () =>
                {
                    if (!await RunManager.Instance.RewardsSetSynchronizer.SelectLocalReward(reward))
                        throw new InvalidOperationException("A validated reward was refused.");
                })));
        }
        if (!menu.Set.DisallowSkipping)
            bindings.Add(new(Candidate("LEAVE_REWARDS", "leave_rewards"),
                () => ReferenceEquals(_protocolRewards, menu) && !menu.Busy && !menu.Set.DisallowSkipping,
                () => menu.Submit(() =>
                {
                    RunManager.Instance.RewardsSetSynchronizer.SkipLocalRewardsSet();
                    return Task.CompletedTask;
                })));
        // Discarding a held potion can make a full-slot potion reward legal. It must
        // use the menu continuation because an event may already own PendingOperation.
        foreach (var (potion, reference) in snapshot.Potions)
            if (potion.Owner.CanUseOrRemovePotions)
                bindings.Add(new(Candidate("DISCARD_POTION", "discard:" + reference, reference),
                    () => !menu.Busy && potion.Owner.CanUseOrRemovePotions && potion.Owner.Potions.Contains(potion),
                    () => menu.Submit(() => PotionCmd.Discard(potion))));
        return PublishSnapshot("rewards", snapshot, bindings);
    }

    private Dictionary<string, object?> PublishCardRewardBoundary()
    {
        var cards = _cardSelector.PendingRewardCards!;
        var alternatives = _cardSelector.PendingRewardAlternatives!;
        var snapshot = BuildPublicSnapshot();
        var bindings = new List<CandidateBinding>();
        foreach (var (result, index) in cards.Select((c, i) => (c, i)))
        {
            var reference = snapshot.AddCard(result.Card, "reward");
            bindings.Add(new(Candidate("TAKE_CARD_REWARD", "take:" + reference, reference),
                () => ReferenceEquals(_cardSelector.PendingRewardCards, cards), () => _cardSelector.ResolveReward(index)));
        }
        foreach (var (alternate, index) in alternatives.Select((a, i) => (a, i)))
        {
            var reference = $"reward_alternative:{index}";
            snapshot.Entities.Add(new() { ["entity_type"] = "reward_alternative", ["ref"] = reference,
                ["content_id"] = alternate.OptionId, ["effect_coverage"] = "opaque",
                ["semantic_program"] = OpaqueProgram(alternate.OptionId) });
            bindings.Add(new(Candidate("CHOOSE_REWARD_ALTERNATIVE", reference, reference),
                () => ReferenceEquals(_cardSelector.PendingRewardAlternatives, alternatives),
                () => _cardSelector.ResolveRewardAlternative(index)));
        }
        return PublishSnapshot("card_reward", snapshot, bindings);
    }

    private Dictionary<string, object?> PublishPostCombatBoundary(CombatRoom room)
    {
        if (!room.IsPreFinished) return NonDecisionBoundary("waiting");
        if (_protocolRewardsOffered.Add(room))
        {
            _pendingOperation.Start("combat rewards", async () =>
            {
                var player = _runState!.Players[0];
                var set = room.Encounter.ShouldGiveRewards
                    ? await RewardsCmd.GenerateForRoomEnd(player, room)
                    : new RewardsSet(player).EmptyForRoom(room);
                await Hook.BeforeCombatRewardOffered(set, _runState, room);
                await set.Offer();
            });
            WaitForPendingOperation();
            return PublishProtocolBoundary();
        }
        if (room.RoomType == RoomType.Boss && (_runState!.Map.SecondBossMapPoint == null
            || _runState.CurrentMapCoord == _runState.Map.SecondBossMapPoint.coord))
        {
            if (_runState.CurrentActIndex >= 2) return ProtocolError("unconfirmed_final_victory");
            _pendingOperation.Start("next act", () => RunManager.Instance.EnterNextAct());
            WaitForPendingOperation();
            return PublishProtocolBoundary();
        }
        _pendingOperation.Start("leave combat rewards", RunManager.Instance.ProceedFromTerminalRewardsScreen,
            () => _protocolMapVisible = ReferenceEquals(_runState!.CurrentRoom, room));
        WaitForPendingOperation();
        return PublishProtocolBoundary();
    }
}
