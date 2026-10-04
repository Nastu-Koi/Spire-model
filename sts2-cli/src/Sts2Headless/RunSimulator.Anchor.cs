using System.Reflection;
using System.Text.Json;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace Sts2Headless;

public partial class RunSimulator
{
    // A run entered at a reconstructed node boundary instead of through start_run.
    // The contract names the initialization so these frames never pass for a
    // continuous native run: they are supervised-only material.
    public const string AnchorInitialization = "summary-anchor-v1";
    private string? _protocolInitialization;
    private bool _anchorRoomEntered;

    /// Install a serialized run at one node. A summary records the type of every
    /// visited node, not its coordinates: `route` (the act's recorded types) places
    /// the run on the one path through the native map that spells them. Without a
    /// single such path the position falls back to a native point of the recorded
    /// type on the recorded floor, and the reply says so. anchor_state then shows
    /// what was installed, and anchor_room enters the node's room.
    public Dictionary<string, object?> EnterAnchor(JsonElement cmd)
    {
        if (!cmd.TryGetProperty("save", out var saveElement) || saveElement.ValueKind != JsonValueKind.Object
            || !cmd.TryGetProperty("act_floor", out var floorElement) || !floorElement.TryGetInt32(out var actFloor)
            || !cmd.TryGetProperty("map_point_type", out var typeElement) || typeElement.ValueKind != JsonValueKind.String)
            return Error("enter_anchor requires save, act_floor and map_point_type");
        var saveJson = saveElement.GetRawText();
        var loaded = LoadSave(saveJson, "en", diagnosticProtocol: true);
        if (loaded.GetValueOrDefault("type") as string == "error") return loaded;
        try
        {
            var save = SaveManager.FromJson<SerializableRun>(saveJson).SaveData!;
            if (save.Ascension is < 0 or > 10) return Error("Anchor ascension must be 0-10");
            var runState = _runState!;
            var player = runState.Players[0];
            // Loading enters the act and runs its hooks. Reinstall the supplied
            // state silently so owned relics do not grant their pickup effects again.
            player.SyncWithSerializedPlayer(save.Players[0]);

            var route = cmd.TryGetProperty("route", out var recorded) && recorded.ValueKind == JsonValueKind.Array
                ? RecordedRoute(runState.Map, recorded.EnumerateArray().Select(t => t.GetString() ?? "").ToList()) : null;
            if (route != null && actFloor >= 1 && actFloor <= route.Count)
                foreach (var visited in route.Take(actFloor)) runState.AddVisitedMapCoord(visited.coord);
            else
            {
                route = null;
                var wanted = typeElement.GetString()!.Replace("_", "");
                MapPoint? point;
                if (wanted.Equals("boss", StringComparison.OrdinalIgnoreCase))
                    point = cmd.TryGetProperty("second_boss", out var second) && second.ValueKind == JsonValueKind.True
                        ? runState.Map.SecondBossMapPoint : runState.Map.BossMapPoint;
                else
                    point = runState.Map.GetAllMapPoints()
                        .Where(p => p.PointType.ToString().Equals(wanted, StringComparison.OrdinalIgnoreCase))
                        .OrderBy(p => Math.Abs(p.coord.row - (actFloor - 1))).ThenBy(p => p.coord.col).FirstOrDefault();
                if (point == null) return Error("No native map point of the anchor's type");
                runState.AddVisitedMapCoord(point.coord);
            }
            runState.ActFloor = actFloor;

            _protocolInitialization = AnchorInitialization;
            _protocolTrainingRun = true;
            _anchorRoomEntered = false;
            EnsureDecisionProtocol();
            return new()
            {
                ["type"] = "anchor_installed", ["initialization"] = AnchorInitialization,
                ["map_position"] = route != null ? "recorded_route" : "representative_point",
                ["route"] = route?.Select(p => new[] { (int)p.coord.col, (int)p.coord.row }).ToArray(),
                // Drawn exits of every route node: one exit is no travel decision.
                ["route_exits"] = route?.Select(p => RouteExits(runState.Map, p).Count()).ToArray(),
            };
        }
        catch (Exception ex)
        {
            InvalidateDecisionProtocol();
            return ErrorWithTrace("EnterAnchor failed", ex);
        }
    }

    /// The act's route when exactly one path from the start spells the recorded
    /// node types; otherwise null.
    // The drawn rows end at the boss; a second boss follows the first.
    private static IEnumerable<MapPoint> RouteExits(ActMap map, MapPoint p)
    {
        if (p.coord == map.BossMapPoint.coord)
            return map.SecondBossMapPoint is { } second ? [second] : [];
        if (map.SecondBossMapPoint is { } last && p.coord == last.coord) return [];
        return p.coord.row == map.GetRowCount() - 1
            ? p.Children.Append(map.BossMapPoint).DistinctBy(c => c.coord) : p.Children;
    }

    private static List<MapPoint>? RecordedRoute(ActMap map, IReadOnlyList<string> types)
    {
        if (types.Count == 0) return null;
        string Kind(MapPoint p) => p.PointType.ToString().ToLowerInvariant();
        IEnumerable<MapPoint> Next(MapPoint p) => RouteExits(map, p);
        var wanted = types.Select(t => t.Replace("_", "").ToLowerInvariant()).ToList();
        var forward = new List<List<MapPoint>> { Kind(map.StartingMapPoint) == wanted[0] ? [map.StartingMapPoint] : [] };
        for (int i = 1; i < wanted.Count; i++)
            forward.Add(forward[i - 1].SelectMany(Next).Where(p => Kind(p) == wanted[i]).DistinctBy(p => p.coord).ToList());
        // Keep only points that still reach the end of the recorded sequence.
        var alive = forward[^1];
        var route = new List<MapPoint>();
        for (int i = wanted.Count - 1; i >= 0; i--)
        {
            if (i < wanted.Count - 1)
            {
                var reachable = alive.Select(p => p.coord).ToHashSet();
                alive = forward[i].Where(p => Next(p).Any(c => reachable.Contains(c.coord))).ToList();
            }
            if (alive.Count != 1) return null;
            route.Add(alive[0]);
        }
        route.Reverse();
        return route;
    }

    /// Enter the anchor's room and stop at its first decision. room.type: combat
    /// (encounter), rest_site, card_reward (cards, as serialized deck cards), map
    /// (the travel decision after the node), or ancient (event, options: the recorded
    /// offer in its recorded order).
    public Dictionary<string, object?> AnchorRoom(JsonElement cmd)
    {
        if (_protocolInitialization == null || !_protocolTrainingRun || _anchorRoomEntered || _runState == null)
            return Error("anchor_room needs a freshly installed anchor");
        if (!cmd.TryGetProperty("room", out var room) || room.ValueKind != JsonValueKind.Object)
            return Error("anchor_room requires room");
        _anchorRoomEntered = true;
        try
        {
            var runState = _runState;
            var player = runState.Players[0];
            switch (room.TryGetProperty("type", out var kind) ? kind.GetString() : null)
            {
                case "combat":
                    return EnterRoom("combat", room.GetProperty("encounter").GetString(), null, decisionProtocol: true);
                case "rest_site":
                    return EnterRoom("rest_site", null, null, decisionProtocol: true);
                case "map":
                    _protocolMapVisible = true;
                    return AdvanceToBoundary();
                case "ancient":
                {
                    var model = ModelDb.GetById<EventModel>(new ModelId("EVENT", room.GetProperty("event").GetString()!));
                    RunManager.Instance.EnterRoom(new EventRoom(model)).GetAwaiter().GetResult();
                    _syncCtx.Pump();
                    WaitForActionExecutor();
                    if (RunManager.Instance.EventSynchronizer.GetLocalEvent() is not AncientEventModel ancient)
                        return Error("Anchor event is not an ancient");
                    var all = ancient.AllPossibleOptions.ToList();
                    var options = new List<EventOption>();
                    foreach (var key in room.GetProperty("options").EnumerateArray().Select(o => o.GetString() ?? ""))
                    {
                        var option = all.FirstOrDefault(o => o.Relic?.Id.Entry == key || o.TextKey == key
                            || o.TextKey.EndsWith("." + key, StringComparison.Ordinal));
                        if (option == null) return Error("Recorded option is not one of this ancient's: " + key);
                        options.Add(option);
                    }
                    // The ancient drew its own offer on entry; show the recorded one instead.
                    const BindingFlags hidden = BindingFlags.Instance | BindingFlags.NonPublic;
                    typeof(AncientEventModel).GetField("_generatedOptions", hidden)!.SetValue(ancient, options);
                    typeof(EventModel).GetMethod("SetEventState", hidden)!
                        .Invoke(ancient, [ancient.InitialDescription, options]);
                    return AdvanceToBoundary();
                }
                case "card_reward":
                {
                    var cards = room.GetProperty("cards").EnumerateArray().Select(card => runState.LoadCard(
                        JsonSerializer.Deserialize<SerializableCard>(card.GetRawText(), JsonSerializationUtility.Options)!,
                        player)).ToList();
                    if (cards.Count == 0) return Error("card_reward needs at least one card");
                    var reward = new CardReward(cards, CardCreationSource.Encounter, player,
                        CardCreationOptions.ForRoom(player, RoomType.Monster));
                    var set = new RewardsSet(player).WithCustomRewards([reward]);
                    // The recorded offer already shows every change the player saw
                    // (relic upgrades, enchantments). Generating the set again would
                    // apply those relics a second time, from counters no summary keeps.
                    typeof(RewardsSet).GetField("_isGenerated", BindingFlags.Instance | BindingFlags.NonPublic)!
                        .SetValue(set, true);
                    _pendingOperation.Start("anchor card reward", set.Offer);
                    WaitForPendingOperation();
                    return AdvanceToBoundary();
                }
                default:
                    return Error("Unknown anchor room type");
            }
        }
        catch (Exception ex)
        {
            InvalidateDecisionProtocol();
            return ErrorWithTrace("AnchorRoom failed", ex);
        }
    }

    /// Read-only: the installed run as the game serializes it, to check an anchor.
    public Dictionary<string, object?> AnchorState()
    {
        if (_runState == null) return Error("No run in progress");
        var json = SaveManager.ToJson(RunManager.Instance.ToSave(null));
        return new() { ["type"] = "anchor_state", ["save"] = JsonDocument.Parse(json).RootElement.Clone() };
    }
}
