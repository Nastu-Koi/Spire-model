using System.Diagnostics;
using System.Reflection;
using HarmonyLib;

namespace CombatSolverCli;

// CombatSolver 0.44 searches Smart potions after the potion-free search, in what is left of
// the budget (RunSupplementalAudits). The potion-free search does not stop by itself before
// the budget when the root holds power cards: its opening power route portfolio runs member
// after member (up to 10 s each) until the deadline, and the potion layers get nothing.
// A search that reserves time stops the potion-free part at `_primary` ms: every beam search
// of that part stops there on its own soft budget, keeping the routes it found, and no new
// portfolio member starts after it. The audits then have the rest of the budget, and their
// beam searches stop on their soft budget `AuditMargin` ms before the worker's takeover.
internal static class PotionReserve
{
    private const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
    private static readonly Stopwatch Clock = new();
    private static bool _installed, _active, _audits, _stopLogged;
    private static long _primary, _auditDeadline;
    private static FieldInfo? _beamProfile, _softBudget;
    private static MethodInfo? _clone;

    // Margin before the worker's takeover at the full budget: a beam search checks its soft
    // budget between play depths only.
    private const int AuditMargin = 300;

    internal static void Install(Assembly solver)
    {
        if (_installed) return;
        var coordinator = solver.GetType("CombatSolver.CombatSearchCoordinator", true)!;
        var beam = solver.GetType("CombatSolver.CombatBeamSolver", true)!;
        var profile = solver.GetType("CombatSolver.SolverSearchProfile", true)!;
        _beamProfile = beam.GetField("_profile", Flags) ?? throw new InvalidOperationException("CombatBeamSolver._profile was not found");
        _softBudget = profile.GetField("<SoftTimeBudgetMilliseconds>k__BackingField", Flags)
            ?? throw new InvalidOperationException("SolverSearchProfile.SoftTimeBudgetMilliseconds was not found");
        _clone = profile.GetMethod("<Clone>$", Flags) ?? throw new InvalidOperationException("SolverSearchProfile cannot be copied");
        var harmony = new Harmony("combat-solver-cli.potion-reserve");
        // The first audit, which a Smart search without forced potions always calls; Harmony cannot
        // patch RunSupplementalAudits itself, whose exception filters it fails to re-emit.
        harmony.Patch(coordinator.GetMethod("AuditRequiredPotionUse", Flags)!,
            prefix: new HarmonyMethod(typeof(PotionReserve).GetMethod(nameof(BeforeAudits), Flags)));
        harmony.Patch(coordinator.GetMethod("CanFinishTargetPortfolio", Flags)!,
            postfix: new HarmonyMethod(typeof(PotionReserve).GetMethod(nameof(StopPortfolio), Flags)));
        harmony.Patch(beam.GetMethod("Solve", Flags, Type.EmptyTypes)!,
            prefix: new HarmonyMethod(typeof(PotionReserve).GetMethod(nameof(BeforeBeam), Flags)));
        _installed = true;
    }

    // A search of `budget` ms that keeps `reserve` ms of it for the potion audits.
    internal static void Begin(int budget, int reserve)
    {
        _active = reserve > 0;
        _audits = _stopLogged = false;
        _primary = budget - reserve;
        _auditDeadline = Math.Max(_primary, budget - AuditMargin);
        Clock.Restart();
    }

    internal static void End() => _active = false;

    private static int Soft(object profile) => (int)_softBudget!.GetValue(profile)!;

    private static void BeforeAudits()
    {
        if (!_active || _audits) return;
        _audits = true;
        Console.Error.WriteLine($"[CombatSolverCli] potion_reserve_audits elapsed_ms={Clock.ElapsedMilliseconds} until_ms={_auditDeadline}");
    }

    // No new member of a portfolio once the potion-free part is over.
    private static void StopPortfolio(ref bool __result)
    {
        if (!_active || _audits || __result || Clock.ElapsedMilliseconds < _primary) return;
        __result = true;
        if (_stopLogged) return;
        _stopLogged = true;
        Console.Error.WriteLine($"[CombatSolverCli] potion_reserve_portfolio_stopped elapsed_ms={Clock.ElapsedMilliseconds}");
    }

    // A beam search stops on its own soft budget at the deadline of its part.
    private static void BeforeBeam(object __instance)
    {
        if (!_active) return;
        long left = (_audits ? _auditDeadline : _primary) - Clock.ElapsedMilliseconds;
        object profile = _beamProfile!.GetValue(__instance)!;
        if (Soft(profile) <= left) return;
        object copy = _clone!.Invoke(profile, null)!;
        _softBudget!.SetValue(copy, (int)Math.Max(1, left));
        _beamProfile.SetValue(__instance, copy);
    }
}
