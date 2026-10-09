using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Rewards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;

namespace RunRecorder;

// Inspecting already-generated card rewards is free. Execute still enters the
// native reward button lifecycle, so it removes the button only after success.
internal static class LiveRewards
{
    private sealed record PendingChoice(RunState Run, CardReward Reward, CardModel? Card,
        string? Alternative, DateTime Deadline)
    {
        internal NCardRewardSelectionScreen? Screen;
        internal Task? NativeTask;
        internal Task? RoundTask;
        internal bool Opening = true;
    }
    private static PendingChoice? _pending;

    internal static void Reset() => _pending = null;

    internal static void Track(Task task)
    {
        // The last patched task returned during ForceClick is the outer native
        // NRewardButton.GetReward task. Its completion also covers human Skip,
        // including a screen opened and closed entirely between timer ticks.
        if (_pending is { Opening: true } pending) pending.NativeTask = task;
    }

    internal static void RewardRoundStarted(object screen, Task completion)
    {
        if (_pending is not { } pending ||
            !ReferenceEquals(Snapshot.Read(pending.Reward, "_currentlyShownScreen"), screen)) return;
        // Human input may have completed the first round before our timer runs.
        // A multi-pick reward must never apply that old choice to its next round.
        if (pending.RoundTask != null && !ReferenceEquals(pending.RoundTask, completion))
        { Reset(); return; }
        pending.RoundTask = completion;
    }

    internal static bool Handles(string command) => command is "take_card_reward" or "take_reward_alternative";

    internal static void Expand(JsonNode state, Snapshot snapshot)
    {
        if (state["legal"] is not JsonObject legal || legal["scope"]?.GetValue<string>() != "reward_choice" ||
            legal["context"]?["rewards"] is not JsonArray rewards) return;
        // LiveBridge has already removed every action whose native button cannot
        // be clicked. Never populate a reward or open a screen while observing.
        foreach (var action in legal["actions"]!.AsArray().ToArray())
        {
            if (action?["command"]?.GetValue<string>() != "take_reward") continue;
            long id = action["args"]!["reward_instance_id"]!.GetValue<long>();
            if (snapshot.Resolve(id) is not CardReward { IsPopulated: true } reward) continue;
            var raw = rewards.OfType<JsonObject>().FirstOrDefault(r => r["instance_id"]?.GetValue<long>() == id);
            if (raw != null) ExpandReward(legal, raw, CardRewardAlternative.Generate(reward));
        }
    }

    // Kept separate from Godot so the transformation and alternative semantics
    // can be checked against real game types without opening a game window.
    internal static void ExpandReward(JsonObject legal, JsonObject reward, IReadOnlyList<CardRewardAlternative> alternatives)
    {
        if (reward["cards"] is not JsonArray { Count: > 0 } cards) return;
        long id = reward["instance_id"]!.GetValue<long>();
        var actions = legal["actions"]!.AsArray();
        int index = actions.ToList().FindIndex(a => a?["command"]?.GetValue<string>() == "take_reward" &&
            a["args"]?["reward_instance_id"]?.GetValue<long>() == id);
        if (index < 0) return;
        actions.RemoveAt(index);
        foreach (var card in cards)
            actions.Insert(index++, JsonSerializer.SerializeToNode(new { command = "take_card_reward",
                args = new { reward_instance_id = id, card_instance_id = card!["instance_id"]!.GetValue<long>() } }));
        var effectful = alternatives.Where(a =>
            a.AfterSelected != PostAlternateCardRewardAction.EndSelectionAndDoNotCompleteReward).ToArray();
        // Generated alternative objects have a new identity on each observation.
        // Only stable, public option IDs belong in the decision hash.
        reward["alternatives"] = JsonSerializer.SerializeToNode(effectful.Select(a => new
            { a.OptionId, AfterSelected = a.AfterSelected.ToString() }));
        reward["expanded_card_reward"] = true;
        legal["context"]!["expanded_card_rewards"] = true;
        foreach (var alternative in effectful)
            actions.Insert(index++, JsonSerializer.SerializeToNode(new { command = "take_reward_alternative",
                args = new { reward_instance_id = id, option_id = alternative.OptionId } }));
    }

    internal static void Begin(string command, JsonElement args, Snapshot snapshot, RunState run)
    {
        if (_pending != null) throw new InvalidOperationException("Card reward selection is already pending");
        var reward = snapshot.Resolve(args.GetProperty("reward_instance_id").GetInt64()) as CardReward
            ?? throw new InvalidOperationException("Card reward is no longer available");
        CardModel? card = command == "take_card_reward"
            ? snapshot.Resolve(args.GetProperty("card_instance_id").GetInt64()) as CardModel : null;
        string? alternative = command == "take_reward_alternative" ? args.GetProperty("option_id").GetString() : null;
        FindSelectionIndex(reward.Cards.ToArray(), CardRewardAlternative.Generate(reward), card, alternative);
        var button = LiveUi.Resolve("take_reward", args, snapshot)
            ?? throw new InvalidOperationException("Card reward button is no longer ready");
        _pending = new PendingChoice(run, reward, card, alternative, DateTime.UtcNow.AddSeconds(30));
        try
        {
            LiveUi.Click(button);
            if (_pending != null)
                _pending.Screen = Snapshot.Read(reward, "_currentlyShownScreen") as NCardRewardSelectionScreen;
        }
        catch { Reset(); throw; }
        finally { if (_pending != null) _pending.Opening = false; }
    }

    internal static bool BlocksCapture()
    {
        if (_pending == null) return false;
        // Hooks can open another selection while entering the reward. Let the
        // model handle that selection; only hide the preselected reward itself.
        var screen = Snapshot.Read(_pending.Reward, "_currentlyShownScreen");
        return SelectionCapture.Active?.Offer == null ||
            (screen != null && ReferenceEquals(NOverlayStack.Instance?.Peek(), screen));
    }

    internal static void Tick()
    {
        if (_pending is not { } pending) return;
        if (pending.NativeTask?.IsCompleted == true || pending.RoundTask?.IsCompleted == true)
        { Reset(); return; }
        if (RunManager.Instance?.DebugOnlyGetState() != pending.Run || pending.Reward.SuccessfullySelected)
        { Reset(); return; }
        if (DateTime.UtcNow >= pending.Deadline)
        { Reset(); throw new InvalidOperationException("Timed out opening card reward; the visible selection can be retried"); }
        if (Snapshot.Read(pending.Reward, "_currentlyShownScreen") is not NCardRewardSelectionScreen screen)
        {
            if (pending.Screen != null) Reset(); // A human closed the native offer.
            return;
        }
        pending.Screen = screen;
        if (!GodotObject.IsInstanceValid(screen)) { Reset(); return; }
        if (!ReferenceEquals(NOverlayStack.Instance?.Peek(), screen) ||
            SelectionCapture.Active is not { Offer: { } offer } scope || !LiveUi.SelectionReady(scope)) return;
        if (pending.RoundTask == null || !ReferenceEquals(pending.RoundTask,
                Snapshot.Read(Snapshot.Read(screen, "_completionSource"), "Task")))
        { Reset(); return; }
        if (Snapshot.Read(screen, "_options") is not IReadOnlyList<CardCreationResult> options ||
            Snapshot.Read(screen, "_extraOptions") is not IReadOnlyList<CardRewardAlternative> alternatives)
        { Reset(); throw new InvalidOperationException("Card reward screen has no current offer"); }
        var cards = options.Select(result => result.Card).ToArray();
        if (!offer.Cards.SequenceEqual(cards) || offer.Alternatives == null ||
            !offer.Alternatives.Select(a => a.OptionId).SequenceEqual(alternatives.Select(a => a.OptionId))) return;
        // Re-resolve by object identity / option ID after entering the screen.
        // Relic hooks or rerolls may have changed the offer or its ordering.
        int index;
        try { index = FindSelectionIndex(cards, alternatives, pending.Card, pending.Alternative); }
        catch { Reset(); throw; }
        Reset(); // Never submit twice if continuations run synchronously.
        if (!LiveUi.CompleteSelectionTask<int?>(screen, index))
            throw new InvalidOperationException("Card reward selection closed before submission");
    }

    internal static int FindSelectionIndex(IReadOnlyList<CardModel> cards,
        IReadOnlyList<CardRewardAlternative> alternatives, CardModel? selectedCard, string? selectedAlternative)
    {
        if ((selectedCard == null) == (selectedAlternative == null))
            throw new InvalidOperationException("A card reward must select exactly one card or alternative");
        if (selectedCard != null)
        {
            for (int i = 0; i < cards.Count; i++) if (ReferenceEquals(cards[i], selectedCard)) return i;
        }
        else
            for (int i = 0; i < alternatives.Count; i++)
                if (alternatives[i].OptionId == selectedAlternative &&
                    alternatives[i].AfterSelected != PostAlternateCardRewardAction.EndSelectionAndDoNotCompleteReward)
                    return cards.Count + i;
        throw new InvalidOperationException("The selected card reward changed before submission");
    }
}
