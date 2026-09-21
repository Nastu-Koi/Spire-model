using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
var game = Path.GetFullPath(args[0]);
AssemblyLoadContext.Default.Resolving += (_, name) => File.Exists(Path.Combine(game, name.Name + ".dll"))
    ? AssemblyLoadContext.Default.LoadFromAssemblyPath(Path.Combine(game, name.Name + ".dll")) : null;
var assembly = AssemblyLoadContext.Default.LoadFromAssemblyPath(Path.Combine(game, "sts2.dll"));
var managerType = assembly.GetType("MegaCrit.Sts2.Core.Runs.RunManager")!;
var manager = managerType.GetProperty("Instance")!.GetValue(null)!;
var stateProperty = managerType.GetProperty("State", BindingFlags.Instance | BindingFlags.NonPublic)!;
var run = RuntimeHelpers.GetUninitializedObject(assembly.GetType("MegaCrit.Sts2.Core.Runs.RunState")!);
stateProperty.SetValue(manager, run);
try {
    _ = managerType.GetProperty("IsSingleplayerOrFakeMultiplayer")!.GetValue(manager);
    throw new Exception("Fixture failed to reproduce the Steam load race");
} catch (TargetInvocationException e) when (e.InnerException is NullReferenceException) { }
var recorder = typeof(RunRecorder.ModEntry).Assembly.GetType("RunRecorder.Recorder")!;
var ensure = recorder.GetMethod("EnsureRun", BindingFlags.Static | BindingFlags.NonPublic)!;
for (int i = 0; i < 100; i++)
    if ((bool)ensure.Invoke(null, null)!) throw new Exception("Recorded a partially initialized run");
stateProperty.SetValue(manager, null);
if ((bool)ensure.Invoke(null, null)!) throw new Exception("Recorded an absent run");
Console.WriteLine("PASS: real game load race reproduced; 100 recorder polls wait without throwing; unloaded run waits.");
var traceType = typeof(RunRecorder.ModEntry).Assembly.GetType("RunRecorder.SelectionTrace")!;
var traceMethod = traceType.GetMethod("Build", BindingFlags.Static | BindingFlags.NonPublic)!;
System.Text.Json.JsonElement Freeze(object obj) => System.Text.Json.JsonSerializer.SerializeToElement(obj);
var offer = new { legal = new { status = "complete", selection = new { min = 2, max = 2, cancelable = false },
    actions = Enumerable.Range(1, 3).Select(id => new { command = "select_card", args = new { card_instance_id = id } }).ToArray() } };
var selected = new { command = "select_cards", args = new { card_instance_ids = new[] { 3, 1 } } };
var path = Freeze(traceMethod.Invoke(null, new object[] { Freeze(offer), Freeze(selected) })!);
var steps = path.GetProperty("steps").EnumerateArray().ToArray();
if (steps.Length != 3 || steps[0].GetProperty("actions").GetArrayLength() != 3
    || steps[1].GetProperty("actions").GetArrayLength() != 2
    || steps[2].GetProperty("actions").GetArrayLength() != 1
    || steps[2].GetProperty("choice_key").GetProperty("command").GetString() != "finish_selection")
    throw new Exception("Selection masks or forced finish are incorrect");
var repeated = new { command = "select_cards", args = new { card_instance_ids = new[] { 1, 1 } } };
if (traceMethod.Invoke(null, new object[] { Freeze(offer), Freeze(repeated) }) != null)
    throw new Exception("Duplicate selection accepted");
if (args.Length > 1) File.WriteAllText(args[1], path.GetRawText());
Console.WriteLine("PASS: recorded 3-choose-2 prefix masks, submitted order, forced finish and duplicate rejection.");

// Resolve every Harmony target against the installed game's real API surface.
var hookDirectory = Path.Combine(Path.GetTempPath(), "spire-recorder-hooks-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(hookDirectory);
try {
    recorder.GetProperty("OutputDirectory", BindingFlags.Static | BindingFlags.NonPublic)!.SetValue(null, hookDirectory);
    typeof(RunRecorder.ModEntry).GetMethod("InstallHooks", BindingFlags.Static | BindingFlags.NonPublic)!.Invoke(null, null);
    if (!File.Exists(Path.Combine(hookDirectory, "hooks.json"))) throw new Exception("Missing hook report");
    Console.WriteLine("PASS: all recorder Harmony hooks installed against original game assemblies.");
} finally { Directory.Delete(hookDirectory, recursive: true); }

// Exercise the actual Steam model getters through the recorder serializer.
// Fixture construction uses reflection; no Godot scene or running game is needed.
var snapshotType = typeof(RunRecorder.ModEntry).Assembly.GetType("RunRecorder.Snapshot")!;
var snapshot = Activator.CreateInstance(snapshotType)!;
var valueMethod = snapshotType.GetMethod("Value")!;
var darkType = assembly.GetType("MegaCrit.Sts2.Core.Models.Orbs.DarkOrb")!;
var playerType = assembly.GetType("MegaCrit.Sts2.Core.Entities.Players.Player")!;
var creatureType = assembly.GetType("MegaCrit.Sts2.Core.Entities.Creatures.Creature")!;
var owner = RuntimeHelpers.GetUninitializedObject(playerType);
playerType.GetField("<Creature>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!
    .SetValue(owner, RuntimeHelpers.GetUninitializedObject(creatureType));
object Dark(decimal accumulated)
{
    var canonical = Activator.CreateInstance(darkType)!;
    var orb = darkType.GetMethod("ToMutable", new[] { typeof(int) })!.Invoke(canonical, new object[] { 0 })!;
    darkType.GetProperty("Owner")!.SetValue(orb, owner);
    darkType.GetField("_evokeVal", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(orb, accumulated);
    return orb;
}
var darks = new[] { Dark(36m), Dark(18m) };
var beforeOrbs = Freeze(valueMethod.Invoke(snapshot, new object[] { darks, 0 })!);
if (beforeOrbs[0].GetProperty("evoke_value").GetDecimal() != 36m
    || beforeOrbs[1].GetProperty("evoke_value").GetDecimal() != 18m
    || beforeOrbs[0].GetProperty("passive_value").GetDecimal() != 6m)
    throw new Exception("Dark orb current values or ordering lost");
((Task)darkType.GetMethod("Passive")!.Invoke(darks[0], new object?[] { null, null })!).GetAwaiter().GetResult();
var afterOrb = Freeze(valueMethod.Invoke(snapshot, new object[] { darks[0], 0 })!);
if (afterOrb.GetProperty("evoke_value").GetDecimal() != 42m
    || beforeOrbs[0].GetProperty("evoke_value").GetDecimal() != 36m)
    throw new Exception("Dark orb growth was cached or changed a frozen snapshot");

var madType = assembly.GetType("MegaCrit.Sts2.Core.Models.Cards.MadScience")!;
var typeField = madType.GetField("_tinkerTimeType", BindingFlags.Instance | BindingFlags.NonPublic)!;
var riderField = madType.GetField("_tinkerTimeRider", BindingFlags.Instance | BindingFlags.NonPublic)!;
var variants = new List<System.Text.Json.JsonElement>();
var headlessCards = new List<System.Text.Json.JsonElement>();
var publicCard = args.Length > 3
    ? AssemblyLoadContext.Default.LoadFromAssemblyPath(Path.GetFullPath(args[3]))
        .GetType("Sts2Headless.RunSimulator")!.GetMethod("PublicSelectionCard", BindingFlags.Static | BindingFlags.NonPublic)
    : null;
foreach (var cardType in new[] { "Attack", "Skill", "Power" })
foreach (var rider in Enum.GetValues(riderField.FieldType))
{
    // Canonical fixture bypasses save-cache initialization; public getters and
    // serializer are real. Enumerating all combinations tests representation,
    // not which combinations the event offers.
    var card = Activator.CreateInstance(madType)!;
    typeField.SetValue(card, Enum.Parse(typeField.FieldType, cardType));
    riderField.SetValue(card, rider);
    var raw = Freeze(valueMethod.Invoke(snapshot, new object[] { card, 0 })!);
    if (raw.GetProperty("mad_science").GetProperty("rider_effect").GetString() != rider.ToString()
        || raw.GetProperty("card_type").GetString() != cardType
        || raw.GetProperty("vars").EnumerateObject().Count() != 11)
        throw new Exception("Mad Science variant or dynamic variables lost");
    variants.Add(raw);
    if (publicCard != null) headlessCards.Add(Freeze(publicCard.Invoke(null, new[] { card })!));
}
if (args.Length > 2) File.WriteAllText(args[2], Freeze(new { orbs = beforeOrbs, after_orb = afterOrb, cards = variants, headless_cards = headlessCards }).GetRawText());
Console.WriteLine("PASS: real DarkOrb ordered values 36/18, passive growth 36->42, frozen snapshot; all Mad Science type/rider fields and 11 variables.");

// The live bridge checks final selections against the game's frozen offer.
var liveType = typeof(RunRecorder.ModEntry).Assembly.GetType("RunRecorder.LiveBridge")!;
var validateSelection = liveType.GetMethod("ValidateSelection", BindingFlags.Static | BindingFlags.NonPublic)!;
var liveLegal = Freeze(new { selection = new { min = 2, max = 2, cancelable = false },
    actions = new[] { 10, 20, 30 }.Select(id => new { command = "select_card", args = new { card_instance_id = id } }) });
validateSelection.Invoke(null, new object[] { Freeze(new { card_instance_ids = new[] { 10, 20 } }), liveLegal });
foreach (var invalid in new[] { new[] { 10, 10 }, new[] { 99, 20 }, new[] { 10 }, new[] { 10, 20, 30 } }) {
    try {
        validateSelection.Invoke(null, new object[] { Freeze(new { card_instance_ids = invalid }), liveLegal });
        throw new Exception("Live bridge accepted an invalid selection");
    } catch (TargetInvocationException e) when (e.InnerException is InvalidOperationException) { }
}
Console.WriteLine("PASS: live bridge selection bounds, membership and duplicate rejection.");
