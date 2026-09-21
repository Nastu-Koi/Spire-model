using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace RunRecorder;

// File transport works on native Linux and on a shared directory under Proton/WSL.
// Tick is called by the Godot timer: no game object is touched by a worker thread.
internal static class LiveBridge
{
    private static readonly AsyncLocal<bool> Origin = new();
    internal static bool Executing => Origin.Value;
    private static Snapshot _snapshot = new() { TrackObjects = true };
    private static RunState? _run;
    private static string _episode = "", _token = "", _hash = "";
    private static JsonElement _state;
    private static readonly List<Task> Pending = new();
    private static DateTime _lastAdvance;
    private static object? _skippedTreasure;
    private static string DirectoryPath => Path.Combine(Recorder.OutputDirectory, "bridge");

    internal static void Tick()
    {
        var requestPath = Path.Combine(DirectoryPath, "request.json");
        if (!File.Exists(requestPath)) return;
        string id = "";
        try
        {
            using var document = JsonDocument.Parse(File.ReadAllText(requestPath));
            var request = document.RootElement;
            id = request.GetProperty("id").GetString()!;
            File.Delete(requestPath); // Consume before any side effect; never replay a request.
            foreach (var task in Pending.Where(t => t.IsCompleted).ToArray())
            {
                Pending.Remove(task);
                task.GetAwaiter().GetResult();
            }
            object result;
            Origin.Value = true;
            try
            {
                result = request.GetProperty("command").GetString() switch
                {
                    "observe" => Observe(),
                    "execute" => Execute(request),
                    "advance" => Advance(),
                    _ => throw new InvalidOperationException("Unknown bridge command")
                };
            }
            finally { Origin.Value = false; }
            Reply(id, result);
        }
        catch (Exception ex)
        {
            Reply(id, new { status = "error", error = ex.GetBaseException().Message });
            GD.PushError("[RunRecorder bridge] " + ex);
        }
    }

    private static void Reply(string id, object payload)
    {
        Directory.CreateDirectory(DirectoryPath);
        var node = JsonSerializer.SerializeToNode(payload)!.AsObject();
        node["id"] = id;
        string destination = Path.Combine(DirectoryPath, "response.json");
        File.WriteAllText(destination + ".tmp", node.ToJsonString());
        File.Move(destination + ".tmp", destination, true);
    }

    private static IEnumerable<Node> Nodes(Node node)
    {
        yield return node;
        foreach (var child in node.GetChildren())
            foreach (var item in Nodes(child)) yield return item;
    }

    private static bool Ready()
    {
        var manager = RunManager.Instance;
        var run = manager?.DebugOnlyGetState();
        if (run == null || manager!.NetService == null || manager.IsCleaningUp ||
            !manager.IsSingleplayerOrFakeMultiplayer || run.Players.Count != 1) return false;
        if (run != _run)
        {
            _run = run;
            _episode = Guid.NewGuid().ToString("N");
            _snapshot = new Snapshot { TrackObjects = true };
            _token = _hash = "";
            Pending.Clear();
        }
        if (run.AscensionLevel != 10) throw new InvalidOperationException("Model expects a single-player A10 game");
        return true;
    }

    private static CrystalSphereMinigame? Crystal()
    {
        var screen = NOverlayStack.Instance?.Peek();
        return Snapshot.Read(screen, "_entity") as CrystalSphereMinigame
            ?? Snapshot.Read(screen, "Minigame") as CrystalSphereMinigame;
    }

    private static JsonElement? Capture()
    {
        var selection = SelectionCapture.Active;
        if (selection?.Offer != null)
            return JsonSerializer.SerializeToElement(_snapshot.Capture(_run!, decision: "card_selection", source: selection.Offer));
        if (_skippedTreasure == _run!.CurrentRoom && _skippedTreasure != null
            && NMapScreen.Instance?.IsOpen != true) return null;
        if (Pending.Any(t => !t.IsCompleted) || RunManager.Instance.ActionExecutor?.IsRunning == true) return null;
        if (CombatManager.Instance.IsInProgress && CombatManager.Instance.PlayerActionsDisabled) return null;
        var overlay = NOverlayStack.Instance?.Peek();
        if (Crystal() is { IsFinished: false } crystal)
            return JsonSerializer.SerializeToElement(_snapshot.Capture(_run!, decision: "crystal_sphere_cell", source: crystal));
        if (overlay != null && overlay.GetType().Name != "NRewardsScreen") return null;
        return JsonSerializer.SerializeToElement(_snapshot.Capture(_run!, decision: "UsePotionAction"));
    }

    private static string Hash(JsonElement state)
    {
        var node = JsonNode.Parse(state.GetRawText())!.AsObject();
        node.Remove("ui"); // Animation positions are not a decision change.
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(node.ToJsonString())));
    }

    private static object Observe()
    {
        if (!Ready()) return new { status = "waiting", reason = "Start or load a single-player A10 run" };
        if (Recorder.LiveOutcome is bool outcome) return new { status = "terminal", victory = outcome };
        var captured = Capture();
        if (captured == null) return new { status = "waiting", reason = "Game is animating or waiting for its UI" };
        var state = captured.Value;
        var legal = state.GetProperty("legal");
        if (legal.GetProperty("status").GetString() != "complete" || legal.GetProperty("actions").GetArrayLength() == 0)
            return new { status = "waiting", reason = legal.GetProperty("reason").ToString() };
        _state = state;
        _hash = Hash(state);
        _token = Guid.NewGuid().ToString("N");
        return new { status = "decision", token = _token, episode = _episode, state };
    }

    private static object Execute(JsonElement request)
    {
        if (!Ready() || request.GetProperty("token").GetString() != _token || _token == "")
            return new { status = "stale" };
        var current = Capture();
        if (current == null || Hash(current.Value) != _hash) { _token = ""; return new { status = "stale" }; }
        var action = request.GetProperty("action");
        var command = action.GetProperty("command").GetString()!;
        var args = action.GetProperty("args");
        var legal = _state.GetProperty("legal");
        if (command == "select_cards") ValidateSelection(args, legal);
        else if (!legal.GetProperty("actions").EnumerateArray().Any(a => JsonNode.DeepEquals(
                     JsonNode.Parse(a.GetRawText()), JsonNode.Parse(action.GetRawText()))))
            throw new InvalidOperationException("Action is absent from the current legal set");
        _token = ""; // Invalidate before execution, including on exception.
        Dispatch(command, args);
        return new { status = "executed" };
    }

    internal static void ValidateSelection(JsonElement args, JsonElement legal)
    {
        var bounds = legal.GetProperty("selection");
        var ids = args.GetProperty("card_instance_ids").EnumerateArray().Select(x => x.GetInt64()).ToArray();
        var offered = legal.GetProperty("actions").EnumerateArray()
            .Select(a => a.GetProperty("args").GetProperty("card_instance_id").GetInt64()).ToArray();
        if (ids.Distinct().Count() != ids.Length || ids.Any(x => !offered.Contains(x)) ||
            ids.Length > bounds.GetProperty("max").GetInt32() ||
            (ids.Length < bounds.GetProperty("min").GetInt32() && !(ids.Length == 0 && bounds.GetProperty("cancelable").GetBoolean())))
            throw new InvalidOperationException("Invalid buffered card selection");
    }

    private static void Track(Task task) => Pending.Add(task);
    private static int? Index(JsonElement args) => args.GetProperty("index").ValueKind == JsonValueKind.Null ? null : args.GetProperty("index").GetInt32();

    private static void Dispatch(string command, JsonElement args)
    {
        var manager = RunManager.Instance;
        var player = _run!.Players[0];
        object Object(string key) => _snapshot.Resolve(args.GetProperty(key).GetInt64());
        Creature? Target() => args.TryGetProperty("target_instance_id", out var id) && id.ValueKind != JsonValueKind.Null
            ? (Creature)_snapshot.Resolve(id.GetInt64()) : null;
        switch (command)
        {
            case "play_card": manager.ActionQueueSynchronizer.RequestEnqueue(new PlayCardAction((CardModel)Object("card_instance_id"), Target())); break;
            case "end_turn": PlayerCmd.EndTurn(player, canBackOut: false); break;
            case "use_potion": ((PotionModel)Object("potion_instance_id")).EnqueueManualUse(Target()); break;
            case "discard_potion": Track(PotionCmd.Discard((PotionModel)Object("potion_instance_id"))); break;
            case "select_map_node":
                manager.ActionQueueSynchronizer.RequestEnqueue(new MoveToMapCoordAction(player,
                    new MapCoord(args.GetProperty("col").GetInt32(), args.GetProperty("row").GetInt32()))); break;
            case "choose_event_option": Track(((EventOption)Object("option_instance_id")).Chosen()); break;
            case "choose_rest_option": Track(manager.RestSiteSynchronizer.ChooseLocalOption(args.GetProperty("index").GetInt32())); break;
            case "purchase": Track(((MerchantEntry)Object("entry_instance_id")).OnTryPurchaseWrapper(((MerchantRoom)_run.CurrentRoom!).GetLocalInventory())); break;
            case "leave_shop": NMapScreen.Instance.Open(); break;
            case "pick_relic":
                if (Index(args) == null) _skippedTreasure = _run.CurrentRoom;
                manager.TreasureRoomRelicSynchronizer.PickRelicLocally(Index(args)); break;
            case "take_reward": Track(manager.RewardsSetSynchronizer.SelectLocalReward((Reward)Object("reward_instance_id"))); break;
            case "skip_rewards": manager.RewardsSetSynchronizer.SkipLocalRewardsSet(); break;
            case "select_reward_option": CompleteSelection<int?>(Index(args)); break;
            case "select_bundle":
                CompleteSelection<IEnumerable<IReadOnlyList<CardModel>>>(new[] { SelectionCapture.Active!.Offer!.Bundles![Index(args)!.Value] }); break;
            case "select_cards":
                CompleteSelection<IEnumerable<CardModel>>(args.GetProperty("card_instance_ids").EnumerateArray()
                    .Select(id => (CardModel)_snapshot.Resolve(id.GetInt64())).ToArray()); break;
            case "crystal_sphere_cell":
                var crystal = Crystal() ?? throw new InvalidOperationException("Crystal sphere closed");
                crystal.SetTool(Enum.Parse<CrystalSphereMinigame.CrystalSphereToolType>(args.GetProperty("tool").GetString()!));
                Track(crystal.CellClicked(crystal.cells[args.GetProperty("x").GetInt32(), args.GetProperty("y").GetInt32()])); break;
            default: throw new NotSupportedException("Steam command: " + command);
        }
    }

    private static void CompleteSelection<T>(T value)
    {
        var overlay = NOverlayStack.Instance?.Peek();
        var nodes = overlay is Node top ? new[] { top } : Nodes(NGame.Instance).Where(n => n.GetType().Name == "NPlayerHand");
        foreach (var node in nodes)
        {
            var source = Snapshot.Read(node, "_completionSource") ?? Snapshot.Read(node, "_selectionCompletionSource");
            if (source is TaskCompletionSource<T> completion && !completion.Task.IsCompleted)
            {
                if (!completion.TrySetResult(value)) throw new InvalidOperationException("Selection already completed");
                // Grid screens normally close in their click handlers; other selectors close in their awaiting task.
                if (node is NCardGridSelectionScreen grid && GodotObject.IsInstanceValid(grid) && grid.IsInsideTree())
                    NOverlayStack.Instance.Remove(grid);
                return;
            }
        }
        throw new InvalidOperationException("Selection UI is not ready");
    }

    private static object Advance()
    {
        if (!Ready() || Recorder.LiveOutcome != null || SelectionCapture.Active != null ||
            Pending.Any(t => !t.IsCompleted) || RunManager.Instance.ActionExecutor?.IsRunning == true ||
            CombatManager.Instance.IsInProgress || DateTime.UtcNow - _lastAdvance < TimeSpan.FromSeconds(.5))
            return new { status = "waiting" };
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay != null)
        {
            if (overlay.GetType().Name != "NRewardsScreen") return new { status = "waiting" };
            var rewards = Snapshot.ActiveRewardsSet(_run!, null);
            if (rewards?.Rewards.Any(r => !r.SuccessfullySelected) == true) return new { status = "waiting" };
        }
        // Advance only presentation controls. Never pick a card, event option or map node here.
        foreach (var node in Nodes(NGame.Instance).OfType<NButton>())
        {
            if (!node.IsVisibleInTree() || Snapshot.Read(node, "IsEnabled") is not true) continue;
            var name = node.GetType().Name;
            if (name == "NProceedButton" && Snapshot.Read(node, "IsSkip") is true && _skippedTreasure != _run!.CurrentRoom) continue;
            if (name is "NProceedButton" or "NAncientDialogueHitbox" or "NTreasureButton")
            {
                _lastAdvance = DateTime.UtcNow;
                node.EmitSignal("Released", node);
                return new { status = "advanced" };
            }
        }
        return new { status = "waiting" };
    }
}
