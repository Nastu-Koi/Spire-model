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
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace RunRecorder;

// File transport works on native Linux and on a shared directory under Proton/WSL.
// Tick is called by the Godot timer: no game object is touched by a worker thread.
internal static class LiveBridge
{
    // Card rewards are published with their cards; a chest is opened by an action.
    private const string Protocol = "steam-live-v2";
    private static readonly AsyncLocal<bool> Origin = new();
    internal static bool Executing => Origin.Value;
    private static Snapshot _snapshot = new() { TrackObjects = true };
    private static RunState? _run;
    private static string _episode = "", _token = "", _hash = "";
    private static JsonElement _state;
    private static readonly List<Task> Pending = new();
    private static DateTime _lastAdvance;
    private static DateTime _lastRequest;
    private static string? _rewardError;
    private static bool? _outcome;
    private static RunState? _outcomeRun;
    internal static bool Enabled { get; private set; } = true;
    internal static string Status => !Enabled ? "接管：已暂停" :
        DateTime.UtcNow - _lastRequest < TimeSpan.FromSeconds(30) ? "接管：已连接" : "接管：等待模型连接";
    private static string DirectoryPath => Path.Combine(Recorder.OutputDirectory, "bridge");

    internal static void SetEnabled(bool enabled)
    {
        if (Enabled != enabled) _token = _hash = "";
        if (!enabled) { LiveRewards.Reset(); _rewardError = null; }
        Enabled = enabled;
    }

    internal static void RunEnded(bool victory)
    {
        _outcomeRun = RunManager.Instance?.DebugOnlyGetState();
        _outcome = victory;
    }

    internal static void Reset()
    {
        _run = null;
        _outcome = null;
        _outcomeRun = null;
        _token = _hash = "";
        Pending.Clear();
        _lastAdvance = default;
        LiveRewards.Reset();
        _rewardError = null;
    }

    internal static void Tick()
    {
        // Complete only a previously authorized reward choice. This runs on the
        // Godot thread even if the controller is waiting between requests.
        if (Enabled)
        {
            Origin.Value = true;
            try { LiveRewards.Tick(); }
            catch (Exception ex)
            {
                LiveRewards.Reset();
                _rewardError = ex.GetBaseException().Message;
                GD.PushError("[RunRecorder card reward] " + ex);
            }
            finally { Origin.Value = false; }
        }
        var requestPath = Path.Combine(DirectoryPath, "request.json");
        if (!File.Exists(requestPath)) return;
        string id = "";
        bool uniqueResponse = false;
        try
        {
            using var document = TakeRequest(requestPath);
            if (document == null) return;
            var request = document.RootElement;
            id = request.GetProperty("id").GetString()!;
            uniqueResponse = request.TryGetProperty("unique_response", out var unique) && unique.GetBoolean();
            if (uniqueResponse && !Guid.TryParseExact(id, "N", out _))
                throw new InvalidOperationException("Invalid bridge request ID");
            _lastRequest = DateTime.UtcNow;
            if (!Enabled)
            {
                Reply(id, new { status = "paused", reason = "Model control is disabled in the in-game RunRecorder switches" }, uniqueResponse);
                return;
            }
            foreach (var task in Pending.Where(t => t.IsCompleted).ToArray())
            {
                Pending.Remove(task);
                task.GetAwaiter().GetResult();
            }
            if (_rewardError is { } rewardError)
            {
                _rewardError = null;
                Reply(id, new { status = "error", error = rewardError }, uniqueResponse);
                return;
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
            Reply(id, result, uniqueResponse);
        }
        catch (Exception ex)
        {
            Reply(id, new { status = "error", error = ex.GetBaseException().Message }, uniqueResponse);
            GD.PushError("[RunRecorder bridge] " + ex);
        }
    }

    private static JsonDocument? TakeRequest(string path)
    {
        try
        {
            JsonDocument document;
            // WSL can briefly retain a handle after publishing request.json.
            // Permit that handle, and retry transient conflicts on the next tick.
            using (var stream = new FileStream(path, FileMode.Open, System.IO.FileAccess.Read,
                       FileShare.ReadWrite | FileShare.Delete))
                document = JsonDocument.Parse(stream);
            try
            {
                File.Delete(path); // Consume before any side effect; never replay a request.
                return document;
            }
            catch { document.Dispose(); throw; }
        }
        catch (IOException) { return null; }
        catch (UnauthorizedAccessException) { return null; }
        catch (JsonException) { return null; }
    }

    private static void Reply(string id, object payload, bool uniqueResponse = false)
    {
        Directory.CreateDirectory(DirectoryPath);
        var node = JsonSerializer.SerializeToNode(payload)!.AsObject();
        node["id"] = id;
        // The controller refuses a bridge that publishes or executes differently.
        node["protocol"] = Protocol;
        node["recorder"] = typeof(LiveBridge).Assembly.GetName().Version?.ToString(3);
        // Python may be reading the previous response on a WSL mount. A fresh
        // filename avoids Windows denying replacement of an open response.json.
        string destination = Path.Combine(DirectoryPath, uniqueResponse && Guid.TryParseExact(id, "N", out _)
            ? "response-" + id + ".json" : "response.json");
        File.WriteAllText(destination + ".tmp", node.ToJsonString());
        File.Move(destination + ".tmp", destination, true);
    }

    private static bool Ready()
    {
        var manager = RunManager.Instance;
        var run = manager?.DebugOnlyGetState();
        if (run == null || manager!.NetService == null || manager.IsCleaningUp ||
            !manager.IsSingleplayerOrFakeMultiplayer || run.Players.Count != 1) return false;
        if (run != _run)
        {
            if (_outcomeRun != run) _outcome = null;
            _run = run;
            _episode = Guid.NewGuid().ToString("N");
            _snapshot = new Snapshot { TrackObjects = true };
            _token = _hash = "";
            Pending.Clear();
            LiveRewards.Reset();
            _rewardError = null;
        }
        if (run.AscensionLevel is < 0 or > 10)
            throw new InvalidOperationException("Model expects a single-player A0-A10 game");
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
        if (LiveRewards.BlocksCapture()) return null;
        var selection = SelectionCapture.Active;
        if (selection?.Offer != null)
            return LiveUi.SelectionReady(selection)
                ? JsonSerializer.SerializeToElement(_snapshot.Capture(_run!, decision: "card_selection", source: selection.Offer)) : null;
        if (RunManager.Instance.ActionExecutor?.IsRunning == true) return null;
        if (CombatManager.Instance.IsInProgress && CombatManager.Instance.PlayerActionsDisabled) return null;
        // Terminal rewards remain in the overlay stack underneath the map.
        // The visible map takes precedence over that retained rewards screen.
        if (NMapScreen.Instance is { IsOpen: true } map)
            return map.IsTravelEnabled
                ? JsonSerializer.SerializeToElement(_snapshot.Capture(_run!, decision: "MoveToMapCoordAction")) : null;
        var overlay = NOverlayStack.Instance?.Peek();
        if (Crystal() is { IsFinished: false } crystal)
            return JsonSerializer.SerializeToElement(_snapshot.Capture(_run!, decision: "crystal_sphere_cell", source: crystal));
        if (overlay != null && overlay.GetType().Name != "NRewardsScreen") return null;
        if (overlay == null && Pending.Any(t => !t.IsCompleted)) return null;
        string decision = overlay is NRewardsScreen ? "take_reward" : "UsePotionAction";
        if (overlay == null && NMapScreen.Instance?.IsOpen != true && _run!.CurrentRoom is TreasureRoom)
        {
            var room = LiveUi.Nodes(NGame.Instance).OfType<NTreasureRoom>().FirstOrDefault(r => r.IsVisibleInTree());
            if (room == null) return null;
            // Opening a chest has effects and reveals its relics: it is the policy's action.
            if (LiveUi.ChestButton(room) != null) decision = "open_chest";
            else if (!LiveUi.TreasureReady(room)) return null;
        }
        var captured = JsonSerializer.SerializeToNode(_snapshot.Capture(_run!, decision: decision))!;
        var actions = captured["legal"]!["actions"]!.AsArray();
        bool hadUiAction = false, hasUiAction = false;
        for (int i = actions.Count - 1; i >= 0; i--)
        {
            var action = JsonSerializer.SerializeToElement(actions[i]);
            var command = action.GetProperty("command").GetString()!;
            if (LiveUi.Handles(command))
            {
                hadUiAction = true;
                if (LiveUi.Resolve(command, action.GetProperty("args"), _snapshot) == null) actions.RemoveAt(i);
                else hasUiAction = true;
            }
        }
        if (hadUiAction && !hasUiAction) return null;
        LiveRewards.Expand(captured, _snapshot);
        return JsonSerializer.SerializeToElement(captured);
    }

    private static string Hash(JsonElement state)
    {
        var node = JsonNode.Parse(state.GetRawText())!.AsObject();
        node.Remove("ui"); // Animation positions are not a decision change.
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(node.ToJsonString())));
    }

    private static object Observe()
    {
        if (!Ready()) return new { status = "waiting", reason = "Start or load a single-player run matching the model's ascension" };
        if (_outcome is bool outcome) return new { status = "terminal", victory = outcome };
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

    internal static void Track(Task task)
    {
        LiveRewards.Track(task);
        if (!Pending.Contains(task)) Pending.Add(task);
    }
    private static int? Index(JsonElement args) => args.GetProperty("index").ValueKind == JsonValueKind.Null ? null : args.GetProperty("index").GetInt32();

    private static void Dispatch(string command, JsonElement args)
    {
        if (LiveRewards.Handles(command))
        {
            LiveRewards.Begin(command, args, _snapshot, _run!);
            return;
        }
        if (LiveUi.Handles(command))
        {
            var button = LiveUi.Resolve(command, args, _snapshot)
                ?? throw new InvalidOperationException("Game UI is not ready for " + command);
            LiveUi.Click(button);
            return;
        }
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
            case "purchase": Track(((MerchantEntry)Object("entry_instance_id")).OnTryPurchaseWrapper(((MerchantRoom)_run.CurrentRoom!).GetLocalInventory())); break;
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
        var nodes = overlay is Node top ? new[] { top } : LiveUi.Nodes(NGame.Instance).Where(n => n.GetType().Name == "NPlayerHand");
        foreach (var node in nodes)
        {
            if (LiveUi.CompleteSelectionTask(node, value))
            {
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
        if (!Ready() || _outcome != null || LiveRewards.BlocksCapture() || SelectionCapture.Active != null ||
            RunManager.Instance.ActionExecutor?.IsRunning == true ||
            CombatManager.Instance.IsInProgress || DateTime.UtcNow - _lastAdvance < TimeSpan.FromSeconds(.5))
            return new { status = "waiting" };
        // Pending event/reward tasks may themselves await a visible Continue.
        // Their lifetime must not block presentation or nested reward screens.
        var button = LiveUi.PresentationButton(_run!);
        if (button != null)
        {
            _lastAdvance = DateTime.UtcNow;
            LiveUi.Click(button);
            return new { status = "advanced", control = button.GetType().Name };
        }
        return new { status = "waiting" };
    }
}
