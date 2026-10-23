using System.Reflection;
using HarmonyLib;

namespace CombatSolverCli;

// CombatSolver 0.44 adds a block potion to a won potion-free route before the turn that lost the
// most HP (TryInsertBlockPotion) and replays the adjusted route. When that replay no longer finds
// a card the route chose (a later turn's draw differs), it throws the exception a search prunes
// such a branch with, but this caller does not catch it, and the whole search fails. The
// insertion is optional: the route is kept without it, as the solver keeps a route whose
// adjusted replay for frontloaded Afterimages fails the same way.
internal static class BlockPotionInsertion
{
    private const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

    internal static void Install(Assembly solver)
    {
        var method = solver.GetType("CombatSolver.CombatBeamSolver", true)!.GetMethod("TryInsertBlockPotion", Flags)!;
        var finalizer = typeof(BlockPotionInsertion).GetMethod(nameof(Finalizer), Flags)!;
        if (Harmony.GetPatchInfo(method)?.Finalizers.Any(p => p.PatchMethod == finalizer) == true) return;
        new Harmony("combat-solver-cli.block-potion-insertion").Patch(method, finalizer: new HarmonyMethod(finalizer));
    }

    private static Exception? Finalizer(Exception? __exception, ref object? __result)
    {
        if (__exception?.GetType().FullName != "CombatSolver.InvalidPlannedChoiceBranchException") return __exception;
        __result = null;
        Console.Error.WriteLine("[CombatSolverCli] block_potion_insertion_skipped reason=" + __exception.Message);
        return null;
    }
}
