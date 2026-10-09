using System.Reflection;
using System.Text.Json;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.CardPools;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Runs.History;
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
    // Keep game types out of value-type fields: their layout is resolved before
    // Program.Main installs the assembly resolver in a standalone headless run.
    private object? _anchorChestRoom;
    private int? _anchorChestGoldRoll;

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
    /// (the travel decision after the node), ancient (event, options: the recorded
    /// offer in its recorded order), shop (cards: the recorded character card of
    /// each slot, relics: the recorded relic of each slot), event (event: the
    /// event entered as the game generates it for this run and state), or rewards
    /// (potions, relics, cards: a reward screen holding exactly these), treasure
    /// (relic: the recorded single-player offer, gold_roll: one possible raw chest
    /// roll, 42..52, to be constrained by the caller against the node outcome).
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
                case "treasure":
                {
                    var roll = room.GetProperty("gold_roll").GetInt32();
                    if (roll is < 42 or > 52) return Error("Chest gold_roll must be 42-52");
                    var relic = ModelDb.GetById<RelicModel>(AnchorId(room.GetProperty("relic").GetString()));
                    AnchorTreasure.Install();
                    var frame = EnterRoom("treasure", null, null, decisionProtocol: true);
                    // The grab bag is not in .run. Replace only the hidden offer,
                    // before opening; room-entry hooks and treasure suppression ran.
                    var synchronizer = RunManager.Instance.TreasureRoomRelicSynchronizer;
                    if (synchronizer.CurrentRelics is not { Count: 1 })
                        return Error("Chest does not have one relic offer");
                    _anchorChestRoom = runState.CurrentRoom;
                    _anchorChestGoldRoll = roll;
                    typeof(MegaCrit.Sts2.Core.Multiplayer.Game.TreasureRoomRelicSynchronizer)
                        .GetField("_currentRelics", BindingFlags.Instance | BindingFlags.NonPublic)!
                        .SetValue(synchronizer, new List<RelicModel> { relic });
                    return frame;
                }
                case "event":
                    // An event draws from a stream of its own, seeded by the run and the event.
                    return EnterRoom("event", null, room.GetProperty("event").GetString(), decisionProtocol: true);
                case "shop":
                {
                    // The merchant draws its stock on entry. The record gives the rarity of
                    // each character card and the relics; the shop stream installed with the
                    // anchor draws the rest. Whether the stock then equals the record is for
                    // the caller to judge from the frame: nothing is corrected here.
                    var rarities = room.GetProperty("cards").EnumerateArray()
                        .Select(id => ModelDb.GetById<CardModel>(AnchorId(id.GetString())).Rarity).ToList();
                    var relics = room.GetProperty("relics").EnumerateArray()
                        .Select(id => ModelDb.GetById<RelicModel>(AnchorId(id.GetString()))).ToList();
                    AnchorMerchant.Begin(rarities, relics);
                    Dictionary<string, object?> frame;
                    try { frame = EnterRoom("shop", null, null, decisionProtocol: true); }
                    finally
                    {
                        if (!AnchorMerchant.End()) InvalidateDecisionProtocol();
                    }
                    return _protocolTrainingRun ? frame : Error("The merchant did not take the recorded stock");
                }
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
                case "rewards":
                {
                    // A reward screen as the record leaves it: the rewards named, in the
                    // order the game lists them. Nothing is generated, so no relic or
                    // stream changes what the record shows.
                    var rewards = new List<Reward>();
                    if (room.TryGetProperty("potions", out var potions))
                        rewards.AddRange(potions.EnumerateArray().Select(id =>
                            new PotionReward(ModelDb.GetById<PotionModel>(AnchorId(id.GetString())).ToMutable(), player)));
                    if (room.TryGetProperty("relics", out var relics))
                        rewards.AddRange(relics.EnumerateArray().Select(id =>
                            new RelicReward(ModelDb.GetById<RelicModel>(AnchorId(id.GetString())).ToMutable(), player)));
                    if (room.TryGetProperty("cards", out var offer) && offer.GetArrayLength() > 0)
                        rewards.Add(new CardReward(offer.EnumerateArray().Select(card => runState.LoadCard(
                            JsonSerializer.Deserialize<SerializableCard>(card.GetRawText(), JsonSerializationUtility.Options)!,
                            player)).ToList(), CardCreationSource.Encounter, player,
                            CardCreationOptions.ForRoom(player, RoomType.Monster)));
                    if (rewards.Count == 0) return Error("rewards needs at least one reward");
                    var screen = new RewardsSet(player).WithCustomRewards(
                        rewards.OrderBy(reward => reward.RewardsSetIndex).ToList());
                    typeof(RewardsSet).GetField("_isGenerated", BindingFlags.Instance | BindingFlags.NonPublic)!
                        .SetValue(screen, true);
                    _pendingOperation.Start("anchor rewards", screen.Offer);
                    WaitForPendingOperation();
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

    // Not a generic helper constrained to a game type: the runtime would need the game
    // assembly to load this class, before Program.Main has said where that assembly is.
    private static ModelId AnchorId(string? id)
    {
        var parts = (id ?? "").Split('.', 2);
        if (parts.Length != 2) throw new ArgumentException("Content IDs are CATEGORY.ENTRY: " + id);
        return new ModelId(parts[0], parts[1]);
    }

    /// Read-only: what the game's shop rules read from a card, relic or potion.
    /// These are static properties of the content, the same for every run.
    public Dictionary<string, object?> ContentInfo(JsonElement cmd)
    {
        if (!cmd.TryGetProperty("ids", out var ids) || ids.ValueKind != JsonValueKind.Array)
            return Error("content_info requires ids");
        try
        {
            EnsureModelDbInitialized();
            var items = new Dictionary<string, object?>();
            foreach (var id in ids.EnumerateArray().Select(i => i.GetString() ?? ""))
            {
                items[id] = id.Split('.', 2)[0] switch
                {
                    "CARD" => ModelDb.GetById<CardModel>(AnchorId(id)) is var card ? new Dictionary<string, object?>
                    {
                        ["card_type"] = card.Type.ToString(), ["rarity"] = card.Rarity.ToString(),
                        ["colorless"] = card.Pool is ColorlessCardPool,
                    } : null,
                    "RELIC" => ModelDb.GetById<RelicModel>(AnchorId(id)) is var relic ? new Dictionary<string, object?>
                    {
                        ["rarity"] = relic.Rarity.ToString(), ["allowed_in_shops"] = relic.IsAllowedInShops,
                    } : null,
                    "POTION" => new Dictionary<string, object?> { ["rarity"] = ModelDb.GetById<PotionModel>(AnchorId(id)).Rarity.ToString() },
                    _ => throw new ArgumentException("Unknown content category: " + id),
                };
            }
            return new() { ["type"] = "content_info", ["items"] = items };
        }
        catch (Exception ex) { return Error(ex.Message); }
    }

    /// Read-only: for each option of the event page on screen, the entry the game
    /// would write to the run history if it were chosen (title and variables, in
    /// the game's own serialization), or null where the game writes none; and the
    /// event's variables as they stand. A history entry holds the variables
    /// themselves, so what a summary shows for a choice is their value when the
    /// run was saved, after the event, not when the choice was made.
    public Dictionary<string, object?> AnchorEvent()
    {
        var localEvent = _runState == null ? null : RunManager.Instance.EventSynchronizer.GetLocalEvent();
        if (localEvent == null) return Error("No event in progress");
        try
        {
            static JsonElement Written(EventOptionHistoryEntry entry) => JsonDocument.Parse(
                JsonSerializer.Serialize(entry, JsonSerializationUtility.Options)).RootElement.Clone();
            var all = new EventOptionHistoryEntry { Title = new LocString("events", localEvent.Id.Entry), Variables = new() };
            foreach (var variable in localEvent.DynamicVars.Values) all.Variables[variable.Name] = variable;
            var options = localEvent.CurrentOptions.Select(option =>
            {
                object? history = null;
                if (option.ShouldSaveChoiceToHistory)
                {
                    var name = option.HistoryName;
                    if (ReferenceEquals(name, option.Title))
                    {
                        // The game's option button adds the event's variables to the title it
                        // shows, and the history takes them from there. No button is shown
                        // here: add them to a copy, so that the option on screen stays as it is.
                        name = new LocString(option.Title.LocTable, option.Title.LocEntryKey);
                        name.AddVariablesFrom(option.Title);
                        localEvent.DynamicVars.AddTo(name);
                    }
                    var entry = new EventOptionHistoryEntry { Title = name, Variables = new() };
                    if (option.ShouldSaveVariablesToHistory)
                        foreach (var (key, value) in name.Variables) entry.Variables[key] = value;
                    history = Written(entry);
                }
                return new Dictionary<string, object?>
                    { ["text_key"] = option.TextKey, ["locked"] = option.IsLocked, ["history"] = history };
            }).ToList();
            return new() { ["type"] = "anchor_event", ["event"] = localEvent.Id.ToString(),
                ["finished"] = localEvent.IsFinished, ["options"] = options,
                ["variables"] = Written(all).TryGetProperty("variables", out var variables) ? variables : null };
        }
        catch (Exception ex) { return Error(ex.Message); }
    }

    /// Read-only: the installed run as the game serializes it, to check an anchor.
    public Dictionary<string, object?> AnchorState()
    {
        if (_runState == null) return Error("No run in progress");
        var json = SaveManager.ToJson(RunManager.Instance.ToSave(null));
        return new() { ["type"] = "anchor_state", ["save"] = JsonDocument.Parse(json).RootElement.Clone() };
    }
}
