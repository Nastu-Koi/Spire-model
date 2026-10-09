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
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace RunRecorder;

// The live controller must use the same UI lifecycle as a human click. Calling
// the synchronizer directly can consume a reward while leaving its button alive.
internal static class LiveUi
{
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
        "open_chest" or "pick_relic" or "choose_event_option" or "choose_rest_option" or "leave_shop";

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
            case "leave_shop":
                return nodes.OfType<NMerchantRoom>().Select(r => r.ProceedButton).FirstOrDefault(CanClick);
            default: return null;
        }
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
        if (overlay != null) return null;
        var nodes = Nodes(NGame.Instance).ToArray();
        var dialogue = nodes.OfType<NAncientDialogueHitbox>().FirstOrDefault(CanClick);
        if (dialogue != null) return dialogue;
        var eventProceed = nodes.OfType<NEventOptionButton>().FirstOrDefault(b => b.Option.IsProceed && CanClick(b));
        if (eventProceed != null) return eventProceed;
        // A Skip choice, opening a chest and leaving a shop belong to the policy. A shop's
        // Continue is enabled from the moment the room is entered: pressing it here would
        // walk past the merchant before the shop was ever observed.
        if (run.CurrentRoom is MerchantRoom) return null;
        return nodes.OfType<NProceedButton>().FirstOrDefault(b => !b.IsSkip && CanClick(b));
    }
}
