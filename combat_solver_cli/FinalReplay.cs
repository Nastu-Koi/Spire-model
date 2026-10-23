using System.Reflection;
using System.Reflection.Emit;
using HarmonyLib;

namespace CombatSolverCli;

// CombatSolver 0.44 replays the route it selected from the root before returning it, to annotate
// relic effects, and fails the whole search when the replayed final state is not the searched one
// (MaterializeSelectedRoute, "最终路线的遗物标注回放与选中状态不一致"). In a route that wins the fight
// the replay can count the last enemy's death where the search did not: Melancholy, which loses
// one energy of cost for every death, then ends the fight one cheaper in the exhaust pile. A won
// route whose replay differs only in its state key is kept, with the same HP, enemy HP and
// boundary still required; any other difference still fails the search.
internal static class FinalReplay
{
    private const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
    [ThreadStatic] private static bool _won;

    internal static void Install(Assembly solver)
    {
        var method = solver.GetType("CombatSolver.CombatBeamSolver", true)!.GetNestedTypes(Flags)
            .SelectMany(t => t.GetMethods(Flags)).Single(m => m.Name.StartsWith("<SolveCore>g__MaterializeSelectedRoute|", StringComparison.Ordinal));
        var prefix = typeof(FinalReplay).GetMethod(nameof(Selected), Flags)!;
        if (Harmony.GetPatchInfo(method)?.Prefixes.Any(p => p.PatchMethod == prefix) == true) return;
        new Harmony("combat-solver-cli.final-replay").Patch(method, prefix: new HarmonyMethod(prefix),
            transpiler: new HarmonyMethod(typeof(FinalReplay).GetMethod(nameof(CompareKeys), Flags)!));
    }

    private static object Get(object obj, string name) =>
        obj.GetType().GetProperty(name, Flags)?.GetValue(obj) ?? obj.GetType().GetField(name, Flags)!.GetValue(obj)!;

    private static void Selected(object[] __args)
    {
        object snapshot = Get(Get(Get(__args[0], "Candidate"), "Node"), "Snapshot");
        _won = (bool)Get(snapshot, "AllEnemiesDead") && !(bool)Get(snapshot, "PlayerDead");
    }

    // The first state-key comparison after the route's replay: `replayed.StateKey != searched.StateKey`.
    private static IEnumerable<CodeInstruction> CompareKeys(IEnumerable<CodeInstruction> instructions, ILGenerator generator)
    {
        bool replayed = false;
        int replaced = 0;
        foreach (var instruction in instructions)
        {
            if (instruction.operand is MethodInfo call)
            {
                if (call.Name == "Replay" && call.DeclaringType?.Name == "CombatBeamSolver") replayed = true;
                else if (replayed && replaced == 0 && call.Name == "op_Inequality" && call.DeclaringType?.Name == "StateFingerprint")
                {
                    var key = call.DeclaringType;
                    var searched = generator.DeclareLocal(key);
                    yield return new CodeInstruction(OpCodes.Stloc, searched).WithLabels(instruction.ExtractLabels());
                    yield return new CodeInstruction(OpCodes.Box, key);
                    yield return new CodeInstruction(OpCodes.Ldloc, searched);
                    yield return new CodeInstruction(OpCodes.Box, key);
                    yield return new CodeInstruction(OpCodes.Call, typeof(FinalReplay).GetMethod(nameof(KeysDiffer), Flags)!);
                    replaced++;
                    continue;
                }
            }
            yield return instruction;
        }
        if (replaced != 1) throw new InvalidOperationException("MaterializeSelectedRoute's final replay check was not found");
    }

    private static bool KeysDiffer(object replayed, object searched)
    {
        if (Equals(replayed, searched)) return false;
        if (!_won) return true;
        Console.Error.WriteLine("[CombatSolverCli] final_replay_key_differs_in_won_route");
        return false;
    }
}
