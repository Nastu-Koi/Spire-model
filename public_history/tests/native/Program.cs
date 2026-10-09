using System.Collections;
using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;
using System.Text.Json.Serialization;

// Load both independently compiled adapters into a native diagnostic process.
// Only this test reads private odds / actual pile order, as assertion oracles.
var root = Path.GetFullPath(args[0]);
var config = JsonDocument.Parse(File.ReadAllText(args[1])).RootElement;
var steamPath = Path.GetFullPath(args[2]);
var dirs = new[] { Path.GetDirectoryName(config.GetProperty("worker_dll").GetString())!,
    Path.Combine(root, "sts2-cli/lib"), Path.GetDirectoryName(config.GetProperty("solver_dll").GetString())! }
    .Concat(config.GetProperty("dependency_dirs").EnumerateArray().Select(x => x.GetString()!));
AssemblyLoadContext.Default.Resolving += (ctx, name) => {
    foreach (var dir in dirs) {
        var path = Path.Combine(dir, name.Name + ".dll");
        if (File.Exists(path)) return ctx.LoadFromAssemblyPath(path);
    }
    return null;
};
Directory.SetCurrentDirectory(Path.Combine(root, "sts2-cli"));
const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance;
object? Get(object obj, string name) => obj.GetType().GetProperty(name, flags)?.GetValue(obj)
    ?? obj.GetType().GetField(name, flags)?.GetValue(obj);
IEnumerable<object> Items(object? obj) => obj is IEnumerable e ? e.Cast<object>() : [];
void Check(bool ok, string message) { if (!ok) throw new Exception(message); }
var options = new JsonSerializerOptions { DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
JsonElement Element(object? obj) => JsonSerializer.SerializeToElement(obj, options);
var engine = Assembly.Load("Sts2Headless");
var sim = Activator.CreateInstance(engine.GetType("Sts2Headless.RunSimulator")!)!;
var handle = engine.GetType("Sts2Headless.Program")!.GetMethod("HandleCommand", flags)!;
JsonElement Send(object obj) {
    object[] arguments = [sim, Element(obj)];
    var result = handle.Invoke(null, arguments);
    sim = arguments[0]; // HandleCommand replaces its ref RunSimulator on restart.
    return Element(result);
}
JsonElement Settle(JsonElement f) {
    var deadline = DateTime.UtcNow.AddSeconds(30);
    while (!f.TryGetProperty("boundary", out var b) || b.GetString() == "waiting") {
        Check(DateTime.UtcNow < deadline, "native boundary timeout: " + f);
        if (f.TryGetProperty("type", out var t)) Check(t.GetString() != "error", "native error: " + f);
        f = Send(new { cmd = "advance_to_boundary" });
    }
    return f;
}
var worker = Assembly.Load("CombatSolverCli");
var solver = Assembly.LoadFrom(config.GetProperty("solver_dll").GetString()!);
var adapterType = worker.GetType("CombatSolverCli.SolverAdapter")!;
var adapter = Activator.CreateInstance(adapterType, flags, null, [solver], null)!;
JsonElement Start(string character, string seed, bool training = true) {
    if (Get(sim, "HasBegun") is true) sim.GetType().GetMethod("CleanUp")!.Invoke(sim, null);
    var f = Settle(Send(new { cmd = "start_run", character, seed, ascension = 0, lang = "en", decision_protocol = training }));
    adapterType.GetMethod("PrepareEngine", flags)!.Invoke(adapter, null);
    return f;
}
Start("Ironclad", "history-install"); // Initialize native model/localization services.
var steam = Assembly.LoadFrom(steamPath);
var historyType = steam.GetType("Spire.PublicHistory.NativeHistory")!;
var harmonyType = Assembly.Load("0Harmony").GetType("HarmonyLib.Harmony")!;
historyType.GetMethod("Install")!.Invoke(null, [Activator.CreateInstance(harmonyType, ["history-test.steam"])]);
var game = Assembly.Load("sts2");
var manager = game.GetType("MegaCrit.Sts2.Core.Runs.RunManager")!.GetProperty("Instance", flags)!.GetValue(null)!;
object Run() => manager.GetType().GetMethod("DebugOnlyGetState", flags)!.Invoke(manager, null)!;
int frames = 0, positions = 0, previousIntents = 0, acts = 0;
var memorySnapshots = new List<string>();
void Verify(JsonElement frame, bool checkOdds = true) {
    if (frame.GetProperty("boundary").GetString() != "decision") return;
    var snapshot = sim.GetType().GetMethod("BuildPublicSnapshot", flags)!.Invoke(sim, null)!;
    var cards = (IDictionary)Get(snapshot, "Cards")!;
    var creatures = (IDictionary)Get(snapshot, "Creatures")!;
    string? Reference(object obj) => (cards.Contains(obj) ? cards[obj] : creatures.Contains(obj) ? creatures[obj] : null) as string;
    var fromSteam = Element(historyType.GetMethod("Capture")!.Invoke(null, [Run(), (Func<object, string?>)Reference]));
    var fromEngine = frame.GetProperty("public").GetProperty("memory");
    Check(fromSteam.GetRawText() == fromEngine.GetRawText(), "independent Steam/headless histories diverged");
    var player = Items(Get(Run(), "Players")).First();
    foreach (var item in fromEngine.EnumerateArray()) {
        string kind = item.GetProperty("entity_type").GetString()!;
        if (kind == "known_draw_position") {
            string owner = item.GetProperty("owner_ref").GetString()!;
            var card = cards.Keys.Cast<object>().Single(c => Equals(cards[c], owner));
            var draw = Items(Get(Get(Get(player, "PlayerCombatState")!, "DrawPile")!, "Cards")).ToArray();
            Check(ReferenceEquals(draw[item.GetProperty("position").GetInt32()], card), "published draw position disagrees with native pile");
            positions++;
        }
        if (kind == "previous_intent" && item.TryGetProperty("intent", out _)) previousIntents++;
        if (checkOdds && kind == "history_probability" && item.GetProperty("content_id").GetString() == "potion_drop") {
            string scope = item.GetProperty("scope").GetString()!;
            var native = Convert.ToSingle(Get(Get(Get(player, "PlayerOdds")!, "PotionReward")!, "CurrentValue"));
            bool forced = scope != "Event" && Items(Get(player, "Relics")).Any(r => Get(Get(r, "Id")!, "Entry")?.ToString() == "WHITE_BEAST_STATUE");
            float expected = forced ? 1 : Math.Clamp(native + (scope == "Elite" ? .125f : 0), 0, 1);
            var probability = item.GetProperty("probability");
            Check(probability.GetProperty("known").GetBoolean(), "fresh prefix became unknown");
            Check(Math.Abs(probability.GetProperty("value").GetSingle() - expected) < .00001f, "public potion accumulation differs from native rule");
        }
        if (checkOdds && kind == "history_probability" && item.GetProperty("content_id").GetString() == "question_room") {
            var scope = item.GetProperty("scope").GetString()!.Split(':');
            var roomType = game.GetType("MegaCrit.Sts2.Core.Rooms.RoomType")!;
            var setType = typeof(HashSet<>).MakeGenericType(roomType);
            var eligibleSet = Activator.CreateInstance(setType)!;
            string[] types = ["Monster", "Elite", "Treasure", "Shop", "Event"];
            foreach (var type in types.Where(t => !(scope[0] == "shop_blocked" && t == "Shop")))
                setType.GetMethod("Add")!.Invoke(eligibleSet, [Enum.Parse(roomType, type)]);
            var eligible = Items(game.GetType("MegaCrit.Sts2.Core.Hooks.Hook")!.GetMethod("ModifyUnknownMapPointRoomTypes", flags)!
                .Invoke(null, [Run(), eligibleSet])).Select(t => t.ToString()).ToHashSet();
            var nativeOdds = Get(Get(Run(), "Odds")!, "UnknownMapPoint")!;
            float remaining = 1, expected = 0;
            foreach (string type in types[..^1]) {
                if (!eligible.Contains(type)) continue;
                float mass = Math.Min(remaining, Math.Max(0, Convert.ToSingle(Get(nativeOdds, type + "Odds"))));
                if (scope[1] == type) expected = mass;
                remaining -= mass;
            }
            if (scope[1] == "Event") expected = remaining;
            Check(Math.Abs(item.GetProperty("probability").GetProperty("value").GetSingle() - expected) < .00001f,
                "public question odds differ from native roll rules: " + scope[0] + ":" + scope[1]);
        }
    }
    memorySnapshots.Add(fromEngine.GetRawText());
    frames++;
}
foreach (string filename in new[] { "final-victory-prefix.json", "battleworn-rewards-prefix.json" }) {
    var prefix = JsonDocument.Parse(File.ReadAllText(Path.Combine(root, "combat_solver_cli/tests/native/fixtures", filename))).RootElement;
    var frame = Start(prefix.GetProperty("character").GetString()!, prefix.GetProperty("seed").GetString()!);
    foreach (var record in prefix.GetProperty("records").EnumerateArray()) {
        frame = Settle(frame); Verify(frame);
        var action = record.GetProperty("action");
        var candidate = frame.GetProperty("legal").GetProperty("candidates").EnumerateArray().Single(c =>
            c.GetProperty("verb").GetString() == action.GetProperty("verb").GetString() &&
            c.GetProperty("decoder_slot_ref").GetString() == action.GetProperty("decoder_slot_ref").GetString());
        var command = new Dictionary<string, object> { ["cmd"] = "execute_candidate", ["candidate_ref"] = candidate.GetProperty("candidate_ref").GetString()! };
        foreach (string k in new[] { "decision_id", "state_version", "selection_revision" })
            if (frame.GetProperty("routing").TryGetProperty(k, out var value)) command[k] = value;
        frame = Settle(Send(command));
        if (frame.GetProperty("events").EnumerateArray().Any(e => e.TryGetProperty("final_in_act", out var v) && v.ValueKind == JsonValueKind.True)) acts++;
    }
    Verify(frame);
    Console.WriteLine("PASS native public history replay: " + filename);
}
JsonElement Execute(JsonElement frame, JsonElement candidate) {
    var command = new Dictionary<string, object> { ["cmd"] = "execute_candidate", ["candidate_ref"] = candidate.GetProperty("candidate_ref").GetString()! };
    foreach (string k in new[] { "decision_id", "state_version", "selection_revision" })
        if (frame.GetProperty("routing").TryGetProperty(k, out var value)) command[k] = value;
    return Settle(Send(command));
}
var controlled = Start("Ironclad", "history-headbutt", training: false);
for (int i = 0; controlled.GetProperty("public").GetProperty("phase").GetString() != "map"; i++) {
    Check(i < 30, "could not finish initial event");
    controlled = Execute(controlled, controlled.GetProperty("legal").GetProperty("candidates")[0]);
}
string[] deck = ["HEADBUTT", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD",
    "STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"];
Send(new { cmd = "set_player", hp = 500, max_hp = 500, relics = Array.Empty<string>(), potions = Array.Empty<string>(), deck });
Send(new { cmd = "set_draw_order", cards = deck });
Send(new { cmd = "enter_room", type = "combat", encounter = "SHRINKER_BEETLE_WEAK" });
controlled = Settle(Send(new { cmd = "advance_to_boundary" }));
foreach (var type in new[] { historyType, engine.GetType("Spire.PublicHistory.NativeHistory")! }) {
    // Simulate attaching observers after combat has started: discard observation
    // caches only. The native combat, intents and pile order remain untouched.
    foreach (string name in new[] { "Combats", "Draws" }) {
        var cache = type.GetField(name, flags)!.GetValue(null)!;
        cache.GetType().GetMethod("Clear")!.Invoke(cache, null);
    }
}
// Re-publish after the deliberately missing observation prefix.
sim.GetType().GetMethod("InvalidateDecisionProtocol", flags)!.Invoke(sim, null);
controlled = Settle(Send(new { cmd = "advance_to_boundary" }));
Check(controlled.GetProperty("public").GetProperty("memory").EnumerateArray().Where(m => m.GetProperty("entity_type").GetString() == "previous_intent")
    .All(m => !m.GetProperty("known").GetBoolean()), "midcombat attach invented a previous intent");
JsonElement CardCandidate(JsonElement f, string verb, string content) {
    var refs = f.GetProperty("public").GetProperty("entities").EnumerateArray()
        .Where(e => e.TryGetProperty("content_id", out var id) && id.GetString() == content && e.TryGetProperty("ref", out _))
        .Select(e => e.GetProperty("ref").GetString()).ToHashSet();
    return f.GetProperty("legal").GetProperty("candidates").EnumerateArray().First(c => c.GetProperty("verb").GetString() == verb &&
        c.GetProperty("source_refs").EnumerateArray().Any(r => refs.Contains(r.GetString())));
}
Verify(controlled);
controlled = Execute(controlled, CardCandidate(controlled, "PLAY_CARD", "CARD.DEFEND_IRONCLAD"));
controlled = Execute(controlled, CardCandidate(controlled, "PLAY_CARD", "CARD.STRIKE_IRONCLAD"));
controlled = Execute(controlled, CardCandidate(controlled, "PLAY_CARD", "CARD.HEADBUTT"));
controlled = Execute(controlled, CardCandidate(controlled, "SELECT_ONE", "CARD.DEFEND_IRONCLAD"));
controlled = Execute(controlled, controlled.GetProperty("legal").GetProperty("candidates").EnumerateArray().Single(c => c.GetProperty("verb").GetString() == "FINISH_SELECTION"));
Verify(controlled);
Check(controlled.GetProperty("public").GetProperty("memory").EnumerateArray().Any(m =>
    m.GetProperty("entity_type").GetString() == "known_draw_position" && m.GetProperty("position").GetInt32() == 0), "Headbutt did not expose its public top placement");
controlled = Execute(controlled, controlled.GetProperty("legal").GetProperty("candidates").EnumerateArray().Single(c => c.GetProperty("verb").GetString() == "END_TURN"));
Verify(controlled);
Check(!controlled.GetProperty("public").GetProperty("memory").EnumerateArray().Any(m => m.GetProperty("entity_type").GetString() == "known_draw_position"), "draw/shuffle retained a stale known position");
Check(positions > 0 && previousIntents > 0, $"native coverage missing: positions={positions}, previous={previousIntents}");
var fresh = Start("Ironclad", "history-load");
var path = Path.Combine(Path.GetTempPath(), "public-history-test-" + Guid.NewGuid().ToString("N") + ".save");
try {
    var save = Send(new { cmd = "write_continue_save", path });
    Check(save.GetProperty("success").GetBoolean(), "could not create native load fixture");
    var loaded = Settle(Send(new { cmd = "load_save", path, diagnostic_protocol = true }));
    Verify(loaded, checkOdds: false);
    Check(loaded.GetProperty("public").GetProperty("memory").EnumerateArray().Where(m =>
        m.GetProperty("entity_type").GetString() == "history_probability" && m.GetProperty("scope").GetString() != "shop_blocked:Shop")
        .All(m => !m.GetProperty("probability").GetProperty("known").GetBoolean()), "native load invented historical probabilities");
} finally { File.Delete(path); }
sim.GetType().GetMethod("CleanUp")!.Invoke(sim, null);
Console.WriteLine($"PASS adapters: {frames} frames, {positions} known positions, {previousIntents} previous intents, {acts} act completion notifications, save/load unknown.");
