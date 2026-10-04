using System.Reflection;
using HarmonyLib;

namespace CombatSolverCli;

// CombatSolver 0.44 starts the optional opening-power portfolio even after
// ApplyCurrentTurn was requested. Return its already-computed native baseline.
internal static class PortfolioDeadline
{
    private const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

    internal static void Install(Assembly solver)
    {
        var method = solver.GetType("CombatSolver.CombatSearchCoordinator", true)!
            .GetMethod("RunOpeningPowerRoutePortfolio", BindingFlags.Static | BindingFlags.NonPublic)!;
        var prefix = typeof(PortfolioDeadline).GetMethod(nameof(Prefix), BindingFlags.Static | BindingFlags.NonPublic)!;
        if (Harmony.GetPatchInfo(method)?.Prefixes.Any(p => p.PatchMethod == prefix) == true) return;
        var finalizer = typeof(PortfolioDeadline).GetMethod(nameof(Finalizer), BindingFlags.Static | BindingFlags.NonPublic)!;
        new Harmony("combat-solver-cli.portfolio-deadline").Patch(method, prefix: new HarmonyMethod(prefix), finalizer: new HarmonyMethod(finalizer));
    }

    private static bool StopRequested(object policy)
    {
        var interaction = policy.GetType().GetProperty("Interaction", Flags)!.GetValue(policy);
        var request = interaction?.GetType().GetProperty("CurrentTakeoverRequest", Flags)!.GetValue(interaction);
        return request != null && request.GetType().GetProperty("Kind", Flags)!.GetValue(request)!.ToString() == "ApplyCurrentTurn";
    }

    private static bool Prefix(object[] __args, ref object __result)
    {
        if (!StopRequested(__args[3])) return true;
        __result = __args[8];
        Console.Error.WriteLine("[CombatSolverCli] power_portfolio_skipped_after_deadline");
        return false;
    }
    private static Exception? Finalizer(Exception? __exception, object[] __args, ref object __result)
    {
        var token = (CancellationToken)__args[4];
        if (__exception is not OperationCanceledException canceled
            || !token.IsCancellationRequested || canceled.CancellationToken != token
            || !StopRequested(__args[3]) || __args[8] == null)
            return __exception;
        // Cancellation interrupted optional refinement, not the completed baseline.
        // The adapter still audits live-state isolation before using this result.
        __result = __args[8];
        Console.Error.WriteLine("[CombatSolverCli] power_portfolio_canceled_using_native_baseline");
        return null;
    }
}
