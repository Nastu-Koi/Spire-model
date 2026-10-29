using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace SpireTakeover;

// The live controller must use the same UI lifecycle as a human click. Calling
// the synchronizer directly can consume a reward while leaving its button alive.
internal static class LiveUi
{
    // The shop whose inventory the policy closed with leave_shop: its Continue is next.
    // Any other shop room with a closed inventory has yet to be opened.
    private static NMerchantRoom? _leftShop;

    internal static void LeftShop(NMerchantRoom room) => _leftShop = room;
    internal static IEnumerable<Node> Nodes(Node? root)
    {
        if (root == null || !GodotObject.IsInstanceValid(root)) yield break;
        yield return root;
        foreach (var child in root.GetChildren())
            foreach (var node in Nodes(child)) yield return node;
    }

    internal static bool CanClick(NClickableControl? button) => button != null &&
        GodotObject.IsInstanceValid(button) && button.IsInsideTree() && button.IsVisibleInTree() &&
        button.IsEnabled && button.MouseFilter != Control.MouseFilterEnum.Ignore &&
        button.Modulate.A > .95f && button.SelfModulate.A > .95f &&
        (button is not NRestSiteButton || Snapshot.Read(button, "_isUnclickable") is not true);

    internal static void Click(NClickableControl button)
    {
        if (!CanClick(button)) throw new InvalidOperationException("Game button is not ready: " + button.GetType().Name);
        // ForceClick invokes OnRelease AND Released, including native cleanup.
        button.ForceClick();
    }

    internal static bool Handles(string command) => command is "take_reward" or "skip_rewards" or
        "open_chest" or "pick_relic" or "choose_event_option" or "choose_rest_option" or "leave_rest_site" or "leave_shop";

    internal static NClickableControl? Resolve(string command, JsonElement args, Snapshot snapshot)
    {
        if (NMapScreen.Instance?.IsOpen == true) return null;
        var overlay = NOverlayStack.Instance?.Peek();
        if (command is "take_reward" or "skip_rewards")
        {
            if (overlay is not NRewardsScreen screen) return null;
            if (command == "take_reward")
            {
                var reward = snapshot.Resolve(args.GetProperty("reward_instance_id").GetInt64());
                return Nodes(screen).OfType<NRewardButton>().FirstOrDefault(b => b.Reward == reward && CanClick(b));
            }
            if (Snapshot.Read(screen, "_rewardsSet") is not RewardsSet set ||
                set.Id != args.GetProperty("reward_set_id").GetUInt32()) return null;
            return Nodes(screen).OfType<NProceedButton>().FirstOrDefault(CanClick);
        }
        if (overlay != null) return null;
        var nodes = Nodes(NGame.Instance).ToArray();
        switch (command)
        {
            case "open_chest":
                var closed = nodes.OfType<NTreasureRoom>().FirstOrDefault(r => r.IsVisibleInTree());
                return closed == null ? null : ChestButton(closed);
            case "pick_relic":
                var room = nodes.OfType<NTreasureRoom>().FirstOrDefault(r => r.IsVisibleInTree());
                if (room == null || !TreasureReady(room)) return null;
                var index = args.GetProperty("index");
                if (index.ValueKind == JsonValueKind.Null)
                    return room.ProceedButton.IsSkip && CanClick(room.ProceedButton) ? room.ProceedButton : null;
                return Nodes(room).OfType<NTreasureRoomRelicHolder>()
                    .FirstOrDefault(h => h.Index == index.GetInt32() && CanClick(h));
            case "choose_event_option":
                var option = snapshot.Resolve(args.GetProperty("option_instance_id").GetInt64());
                return nodes.OfType<NEventOptionButton>().FirstOrDefault(b => b.Option == option && CanClick(b));
            case "choose_rest_option":
                var options = RunManager.Instance.RestSiteSynchronizer.GetLocalOptions();
                var selected = args.GetProperty("index").GetInt32();
                if (options == null || selected < 0 || selected >= options.Count) return null;
                return nodes.OfType<NRestSiteButton>().FirstOrDefault(b => b.Option == options[selected] && CanClick(b));
            case "leave_rest_site":
                return nodes.OfType<NRestSiteRoom>().Select(r => r.ProceedButton).FirstOrDefault(CanClick);
            case "leave_shop":
                // Leaving closes the open inventory as its Back does; Continue then opens the map.
                var shop = NMerchantRoom.Instance;
                return shop != null && ShopReady(shop) ? Snapshot.Read(shop.Inventory, "_backButton") as NClickableControl : null;
            default: return null;
        }
    }

    // The open inventory once its slide-in has finished: what a human sees and clicks.
    internal static bool ShopReady(NMerchantRoom room)
    {
        if (!GodotObject.IsInstanceValid(room) || !room.IsVisibleInTree() || !room.Inventory.IsOpen) return false;
        if (Snapshot.Read(room.Inventory, "_inventoryTween") is Tween tween &&
            GodotObject.IsInstanceValid(tween) && tween.IsRunning()) return false;
        return CanClick(Snapshot.Read(room.Inventory, "_backButton") as NClickableControl);
    }

    // Opens the shop as a click on the merchant does (MerchantOpened -> OpenInventory).
    // The merchant is not checked with CanClick: skin mods (Merchant2CuteII) leave it
    // failing the opacity or mouse-filter test although a human can click it.
    internal static bool OpenShop(RunState run)
    {
        if (NMapScreen.Instance?.IsOpen == true || NOverlayStack.Instance?.Peek() != null ||
            run.CurrentRoom is not MerchantRoom || NMerchantRoom.Instance is not { } shop ||
            !GodotObject.IsInstanceValid(shop) || !shop.IsVisibleInTree() || shop.Inventory.IsOpen ||
            ReferenceEquals(_leftShop, shop)) return false;
        var merchant = shop.MerchantButton;
        if (!GodotObject.IsInstanceValid(merchant) || !merchant.IsVisibleInTree() || !merchant.IsEnabled ||
            merchant.IsLocalPlayerDead) return false;
        shop.OpenInventory();
        return true;
    }

    internal static NClickableControl? ChestButton(NTreasureRoom room) =>
        Snapshot.Read(room, "_hasChestBeenOpened") is true ? null
            : Nodes(room).OfType<NClickableControl>().FirstOrDefault(b => b.GetType().Name == "NTreasureButton" && CanClick(b));

    internal static bool TreasureReady(NTreasureRoom room) =>
        Snapshot.Read(room, "_isRelicCollectionOpen") is true &&
        Nodes(room).OfType<NTreasureRoomRelicHolder>().Any(CanClick);

    internal static bool SelectionReady(SelectionScope scope)
    {
        if (DateTimeOffset.UtcNow - scope.CapturedUtc < TimeSpan.FromMilliseconds(400)) return false;
        var overlay = NOverlayStack.Instance?.Peek();
        var nodes = overlay is Node top ? new[] { top } : Nodes(NGame.Instance).Where(n => n.GetType().Name == "NPlayerHand");
        return nodes.Any(node => node is Control control && control.IsVisibleInTree() &&
            Snapshot.Read(Snapshot.Read(node, "_completionSource") ?? Snapshot.Read(node, "_selectionCompletionSource"), "Task")
                is Task { IsCompleted: false });
    }

    internal static bool CompleteSelectionTask<T>(object node, T value)
    {
        var source = Snapshot.Read(node, "_completionSource") ?? Snapshot.Read(node, "_selectionCompletionSource");
        if (source is not TaskCompletionSource<T> completion || completion.Task.IsCompleted) return false;
        // Returning a discard/draw-pile choice can move those cards immediately.
        // Match native CompleteSelection: unsubscribe BEFORE releasing the task,
        // so the old screen cannot auto-complete again as its pile changes.
        if (node is NCombatPileCardSelectScreen)
            typeof(NCombatPileCardSelectScreen).GetMethod("UnsubscribeFromPile",
                BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(node, null);
        if (!completion.TrySetResult(value)) throw new InvalidOperationException("Selection already completed");
        return true;
    }

    internal static NClickableControl? PresentationButton(RunState run)
    {
        if (NMapScreen.Instance?.IsOpen == true) return null;
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is NRewardsScreen screen)
        {
            var rewards = Snapshot.ActiveRewardsSet(run, null);
            if (!screen.IsComplete && rewards?.Rewards.Any(r => !r.SuccessfullySelected) == true) return null;
            return Nodes(screen).OfType<NProceedButton>().FirstOrDefault(CanClick);
        }
        // The finished sphere stays on the stack once its rewards close. Its Proceed is
        // enabled only after those rewards (OnMinigameFinished), so it cannot skip them.
        if (overlay is NCrystalSphereScreen sphere)
            return Nodes(sphere).OfType<NProceedButton>().FirstOrDefault(CanClick);
        if (overlay != null) return null;
        var nodes = Nodes(NGame.Instance).ToArray();
        var dialogue = nodes.OfType<NAncientDialogueHitbox>().FirstOrDefault(CanClick);
        if (dialogue != null) return dialogue;
        var eventProceed = nodes.OfType<NEventOptionButton>().FirstOrDefault(b => b.Option.IsProceed && CanClick(b));
        if (eventProceed != null) return eventProceed;
        // A Skip choice, opening a chest and leaving a shop belong to the policy. A shop's
        // Continue is enabled from the moment the room is entered: the inventory is opened
        // first (OpenShop), and Continue pressed only after leave_shop closed it.
        if (run.CurrentRoom is MerchantRoom)
        {
            var shop = NMerchantRoom.Instance;
            if (shop == null || !GodotObject.IsInstanceValid(shop) || shop.Inventory.IsOpen ||
                !ReferenceEquals(_leftShop, shop)) return null;
            return CanClick(shop.ProceedButton) ? shop.ProceedButton : null;
        }
        // So is leaving a rest site that keeps options after a choice (Miniature Tent):
        // its Proceed is enabled as soon as the first one is done.
        if (run.CurrentRoom is RestSiteRoom && RunManager.Instance.RestSiteSynchronizer.GetLocalOptions().Count > 0) return null;
        return nodes.OfType<NProceedButton>().FirstOrDefault(b => !b.IsSkip && CanClick(b));
    }
}
