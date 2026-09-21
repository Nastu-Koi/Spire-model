using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.TreasureRelicPicking;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

public partial class RunSimulator
{
    private readonly HashSet<TreasureRoom> _protocolOpenedChests = new(ReferenceEqualityComparer.Instance);
    private readonly HashSet<TreasureRoom> _protocolFinishedChests = new(ReferenceEqualityComparer.Instance);

    private Dictionary<string, object?> PublishMerchantBoundary(AbstractRoom room, MerchantInventory? inventory = null)
    {
        if (_protocolClosedMerchants.Contains(room)) return OpenProtocolMap();
        var snapshot = BuildPublicSnapshot();
        inventory ??= ((MerchantRoom)room).GetLocalInventory();
        var player = _runState!.Players[0];
        var bindings = new List<CandidateBinding>();
        foreach (var (entry, index) in inventory.AllEntries.Select((e, i) => (e, i)))
        {
            var reference = $"merchant:{index}";
            var entity = new Dictionary<string, object?> { ["entity_type"] = "shop_item", ["ref"] = reference,
                ["content_id"] = entry.GetType().Name, ["price"] = entry.Cost, ["sold_out"] = !entry.IsStocked };
            switch (entry)
            {
                case MerchantCardEntry { CreationResult: { } result }:
                    var cardRef = snapshot.AddCard(result.Card, "shop");
                    snapshot.Relations.Add(new { source = reference, target = cardRef, role = "offers" });
                    entity["content_id"] = result.Card.Id.ToString();
                    break;
                case MerchantRelicEntry { Model: { } relic }: entity["content_id"] = relic.Id.ToString(); break;
                case MerchantPotionEntry { Model: { } potion }: entity["content_id"] = potion.Id.ToString(); break;
            }
            entity["effect_coverage"] = "opaque";
            entity["semantic_program"] = OpaqueProgram((string)entity["content_id"]!);
            snapshot.Entities.Add(entity);
            bool Legal() => ReferenceEquals(_runState.CurrentRoom, room) && entry.IsStocked && entry.EnoughGold
                && (entry is not MerchantPotionEntry { Model: { } potion } || (player.HasOpenPotionSlots
                    && Hook.ShouldProcurePotion(_runState, player.Creature.CombatState, potion, player)))
                && (entry is not MerchantCardRemovalEntry || player.Deck.Cards.Any(c => c.IsRemovable));
            if (!Legal()) continue;
            bindings.Add(new(Candidate("BUY_ITEM", "buy:" + reference, reference), Legal,
                () => _pendingOperation.Start("merchant purchase", async () =>
                {
                    if (entry is MerchantCardRemovalEntry removal)
                    {
                        if (await removal.OnTryPurchaseWrapper(inventory)) removal.SetUsed();
                        // Canceling the native removal selector intentionally buys nothing.
                    }
                    else if (!await entry.OnTryPurchaseWrapper(inventory))
                        throw new InvalidOperationException("A validated purchase was refused.");
                })));
        }
        bindings.Add(new(Candidate("LEAVE_ROOM", "leave_shop"), () => ReferenceEquals(_runState.CurrentRoom, room),
            () => _protocolMapVisible = true));
        AddPotionCandidates(snapshot, bindings);
        return PublishSnapshot("shop", snapshot, bindings);
    }

    private Dictionary<string, object?> PublishTreasureBoundary(TreasureRoom room)
    {
        if (_protocolFinishedChests.Contains(room)) return OpenProtocolMap();
        var snapshot = BuildPublicSnapshot();
        var bindings = new List<CandidateBinding>();
        var synchronizer = RunManager.Instance.TreasureRoomRelicSynchronizer;
        if (!_protocolOpenedChests.Contains(room))
        {
            bindings.Add(new(Candidate("OPEN_CHEST", "open_chest"),
                () => ReferenceEquals(_runState!.CurrentRoom, room) && !_protocolOpenedChests.Contains(room),
                () =>
                {
                    _protocolOpenedChests.Add(room);
                    _pendingOperation.Start("open treasure chest", async () =>
                    {
                        await room.DoNormalRewards();
                        await room.DoExtraRewardsIfNeeded();
                    });
                }));
        }
        else if (synchronizer.CurrentRelics is { Count: > 0 } relics)
        {
            foreach (var (relic, index) in relics.Select((r, i) => (r, i)))
            {
                var reference = $"treasure:{index}";
                snapshot.Entities.Add(new() { ["entity_type"] = "relic", ["ref"] = reference,
                    ["content_id"] = relic.Id.ToString(), ["zone"] = "treasure",
                    ["effect_coverage"] = "opaque", ["semantic_program"] = OpaqueProgram(relic.Id.ToString()) });
                bindings.Add(new(Candidate("TAKE_TREASURE_RELIC", reference, reference),
                    () => ReferenceEquals(synchronizer.CurrentRelics, relics) && !_protocolFinishedChests.Contains(room),
                    () => _pendingOperation.Start("treasure relic", async () =>
                    {
                        var awarded = new TaskCompletionSource<List<RelicPickingResult>>(TaskCreationOptions.RunContinuationsAsynchronously);
                        void Award(List<RelicPickingResult> results) => awarded.TrySetResult(results);
                        synchronizer.RelicsAwarded += Award;
                        try
                        {
                            synchronizer.PickRelicLocally(index);
                            var results = await awarded.Task;
                            foreach (var result in results)
                                if (result.player != null) await RelicCmd.Obtain(result.relic.ToMutable(), result.player);
                            _protocolFinishedChests.Add(room);
                        }
                        finally { synchronizer.RelicsAwarded -= Award; }
                    })));
            }
        }
        else return OpenProtocolMap();
        bindings.Add(new(Candidate("LEAVE_ROOM", "leave_treasure"), () => ReferenceEquals(_runState!.CurrentRoom, room),
            () =>
            {
                synchronizer.SkipRelicLocally();
                _protocolFinishedChests.Add(room);
                _protocolMapVisible = true;
            }));
        AddPotionCandidates(snapshot, bindings);
        return PublishSnapshot("treasure", snapshot, bindings);
    }
}
