using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;
using System.Text.Json.Serialization;

var root = Path.GetFullPath(args[0]);
var config = JsonDocument.Parse(File.ReadAllText(args[1])).RootElement;
var dirs = new[] { Path.Combine(root, "combat_solver_cli/bin/Debug/net9.0"),
    Path.GetDirectoryName(config.GetProperty("game_dll").GetString())! };
AssemblyLoadContext.Default.Resolving += (ctx, name) => {
    foreach (var dir in dirs) {
        var path = Path.Combine(dir, name.Name + ".dll");
        if (File.Exists(path)) return ctx.LoadFromAssemblyPath(path);
    }
    return null;
};
Directory.SetCurrentDirectory(Path.Combine(root, "sts2-cli"));
const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance;
var engine = Assembly.Load("Sts2Headless");
var sim = Activator.CreateInstance(engine.GetType("Sts2Headless.RunSimulator")!)!;
var handle = engine.GetType("Sts2Headless.Program")!.GetMethod("HandleCommand", flags)!;
var options = new JsonSerializerOptions { DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    Converters = { new JsonStringEnumConverter(JsonNamingPolicy.SnakeCaseLower) } };
JsonElement Element(object obj) => JsonSerializer.SerializeToElement(obj, options);
var observed = new List<JsonElement>();
JsonElement Send(object obj) {
    var result = Element(handle.Invoke(null, new[] { sim, (object)Element(obj) })!);
    if (result.TryGetProperty("events", out var events)) observed.AddRange(events.EnumerateArray().Select(e => e.Clone()));
    return result;
}
JsonElement Settle(JsonElement frame) {
    var until = DateTime.UtcNow.AddSeconds(5);
    while (frame.GetProperty("boundary").GetString() == "waiting" && DateTime.UtcNow < until) {
        Thread.Sleep(10);
        frame = Send(new { cmd = "advance_to_boundary" });
    }
    return frame;
}
var game = Assembly.Load("sts2");
var harmonyAssembly = Assembly.Load("0Harmony");
var harmonyType = harmonyAssembly.GetType("HarmonyLib.Harmony")!;
var methodType = harmonyAssembly.GetType("HarmonyLib.HarmonyMethod")!;
var harmony = Activator.CreateInstance(harmonyType, new object[] { "test.combat-save-barrier" });
var save = game.GetType("MegaCrit.Sts2.Core.Saves.SaveManager")!.GetMethods(flags)
    .Single(m => m.Name == "SaveRun" && m.GetParameters().Length == 2);
var postfix = Activator.CreateInstance(methodType, new object[] { typeof(SaveBarrier).GetMethod(nameof(SaveBarrier.AfterSave))! });
var patch = harmonyType.GetMethods().Single(m => m.Name == "Patch");
patch.Invoke(harmony, patch.GetParameters().Select(p => p.Name == "original" ? save : p.Name == "postfix" ? postfix : null).ToArray());
var data = JsonDocument.Parse(File.ReadAllText(Path.Combine(root, "combat_solver_cli/tests/native/fixtures/final-victory-prefix.json"))).RootElement;
Send(new { cmd = "start_run", character = data.GetProperty("character").GetString(),
    seed = data.GetProperty("seed").GetString(), ascension = 0, lang = "en", decision_protocol = true });
var frame = Send(new { cmd = "advance_to_boundary" });
var records = data.GetProperty("records").EnumerateArray().ToArray();
bool normal = args.Length > 2 && args[2] == "normal";
if (normal) {
    int end = Array.FindIndex(records, r => r.GetProperty("phase").GetString() == "rewards");
    if (end <= 0 || records[end - 1].GetProperty("phase").GetString() != "combat") throw new Exception("Missing combat/reward fixture boundary");
    records = records[..end];
}
try {
    for (int i = 0; i < records.Length; i++) {
        frame = Settle(frame);
        var action = records[i].GetProperty("action");
        var candidate = frame.GetProperty("legal").GetProperty("candidates").EnumerateArray().Single(c =>
            c.GetProperty("verb").GetString() == action.GetProperty("verb").GetString() &&
            c.GetProperty("decoder_slot_ref").GetString() == action.GetProperty("decoder_slot_ref").GetString());
        var command = new Dictionary<string, object> { ["cmd"] = "execute_candidate", ["candidate_ref"] = candidate.GetProperty("candidate_ref").GetString()! };
        foreach (var k in new[] { "decision_id", "state_version", "selection_revision" })
            if (frame.GetProperty("routing").TryGetProperty(k, out var value)) command[k] = value.Clone();
        if (i == records.Length - 1) { observed.Clear(); SaveBarrier.Armed = true; }
        frame = Send(command);
    }
    if (!SaveBarrier.Entered.Wait(3000)) throw new Exception("Save barrier was not reached");
    // The real combat is pre-finished, but its real CombatWon callback is still
    // awaiting SaveRun. No game state or victory flag has been manufactured.
    for (int i = 0; i < 3; i++) {
        if (frame.GetProperty("boundary").GetString() != "waiting")
            throw new Exception("Premature post-combat boundary: " + frame);
        if (frame.GetProperty("events").EnumerateArray().Any(e => e.GetProperty("type").GetString() == "run_completed"))
            throw new Exception("Victory published before native confirmation");
        frame = Send(new { cmd = "advance_to_boundary" });
    }
    SaveBarrier.Release.TrySetResult();
    frame = Settle(Send(new { cmd = "advance_to_boundary" }));
    var encounters = observed.Where(e => e.GetProperty("type").GetString() == "encounter_completed").ToArray();
    if (encounters.Length != 1 || encounters[0].GetProperty("result").GetString() != "victory") throw new Exception("Missing/duplicate encounter evidence");
    if (normal) {
        if (frame.GetProperty("public").GetProperty("phase").GetString() != "rewards" || observed.Any(e => e.GetProperty("type").GetString() == "run_completed"))
            throw new Exception("Normal combat must offer rewards, not final victory: " + frame);
        Console.WriteLine("PASS: pending native save waits; normal rewards follow CombatWon");
        return;
    }
    if (frame.GetProperty("boundary").GetString() != "terminal" || !frame.GetProperty("public").GetProperty("outcome").GetProperty("victory").GetBoolean())
        throw new Exception("Missing confirmed victory: " + frame);
    var completed = observed.Where(e => e.GetProperty("type").GetString() == "run_completed").ToArray();
    if (completed.Length != 1 || !completed[0].GetProperty("final_boss_defeated").GetBoolean())
        throw new Exception("Missing/duplicate final boss evidence");
    Console.WriteLine("PASS: pending native save waits; real CombatWon confirms victory once");
} finally { SaveBarrier.Release.TrySetResult(); }

public static class SaveBarrier {
    public static volatile bool Armed;
    public static readonly ManualResetEventSlim Entered = new(false);
    public static readonly TaskCompletionSource Release = new(TaskCreationOptions.RunContinuationsAsynchronously);
    public static void AfterSave(object __0, ref Task __result) {
        if (!Armed || __0?.GetType().GetProperty("IsPreFinished")?.GetValue(__0) is not true) return;
        Entered.Set();
        __result = Hold(__result);
    }
    private static async Task Hold(Task original) { await original; await Release.Task; }
}
