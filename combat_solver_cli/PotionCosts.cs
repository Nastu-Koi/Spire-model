using System.Reflection;
using System.Reflection.Emit;
using System.Text.Json;
using HarmonyLib;

namespace CombatSolverCli;

// CombatSolver 0.44 prices a potion in HP: a route may spend one only where it saves
// that much. The price is 9, or 14 and 18 for the two lists of potions it holds dearer
// (SolverWeights.PotionMinimumHpSaved, PotionElevatedValueHpSaved, PotionHighValueHpSaved).
// They are constants, so the compiler copied them into every method that uses one; those
// methods read the price of the current request here instead. A request without
// `potion_costs` searches at the solver's own prices.
internal static class PotionCosts
{
    private const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
    private static readonly int[] Native = { 9, 14, 18 };
    private static readonly FieldInfo[] Fields = new[] { nameof(Base), nameof(Elevated), nameof(High) }
        .Select(name => typeof(PotionCosts).GetField(name, Flags)!).ToArray();
    private static bool _installed;

    internal static int Base = 9, Elevated = 14, High = 18;

    internal static int[] Current => new[] { Base, Elevated, High };

    internal static void Install(Assembly solver)
    {
        if (_installed) return;
        var policy = solver.GetType("CombatSolver.PotionUsePolicy", true)!;
        var beam = solver.GetType("CombatSolver.CombatBeamSolver", true)!;
        // The turn a block potion is inserted into is the first that loses a base price or more.
        var insertion = beam.GetNestedTypes(Flags).SelectMany(type => type.GetMethods(Flags | BindingFlags.DeclaredOnly))
            .Where(method => method.Name.Contains("<TryInsertBlockPotion>") && method.ReturnType == typeof(bool)
                && Uses(method).SequenceEqual(new[] { 1, 0, 0 })).ToArray();
        if (insertion.Length != 1) throw new InvalidOperationException("CombatSolver's block potion threshold was not found");
        var sites = new (MethodBase Method, int[] Uses)[]
        {
            (policy.GetMethod("RequiredHpSaved", Flags)!, new[] { 1, 0, 0 }),
            (policy.GetMethod("AdditionalRequiredUseStrategicHpCost", Flags)!, new[] { 1, 0, 0 }),
            (policy.GetMethods(Flags).Single(m => m.Name == "StrategicHpCost" && m.GetParameters()[0].ParameterType != typeof(string)), new[] { 1, 1, 1 }),
            (policy.GetMethod("EffectiveStrategicHpCost", Flags)!, new[] { 1, 0, 0 }),
            (solver.GetType("CombatSolver.CombatSearchCoordinator", true)!.GetMethod("MaximumSmartPotionUses", Flags)!, new[] { 2, 0, 0 }),
            (insertion[0], new[] { 1, 0, 0 }),
        };
        foreach (var (method, uses) in sites)
        {
            if (!Uses(method).SequenceEqual(uses))
                throw new InvalidOperationException($"CombatSolver's potion prices in {method.Name} are not where version 0.44 has them");
        }
        var transpiler = new HarmonyMethod(typeof(PotionCosts).GetMethod(nameof(Transpiler), Flags)!);
        var harmony = new Harmony("combat-solver-cli.potion-costs");
        foreach (var (method, _) in sites) harmony.Patch(method, transpiler: transpiler);
        _installed = true;
    }

    // How often a method loads each of the three native prices.
    private static int[] Uses(MethodBase method)
    {
        var code = PatchProcessor.GetOriginalInstructions(method);
        return Native.Select(price => code.Count(instruction => instruction.LoadsConstant(price))).ToArray();
    }

    private static IEnumerable<CodeInstruction> Transpiler(IEnumerable<CodeInstruction> instructions)
    {
        foreach (var instruction in instructions)
        {
            int tier = Array.FindIndex(Native, price => instruction.LoadsConstant(price));
            if (tier >= 0)
            {
                instruction.opcode = OpCodes.Ldsfld;
                instruction.operand = Fields[tier];
            }
            yield return instruction;
        }
    }

    // The prices a request names in `field`: [base, elevated, high], in HP; the solver's own without it.
    // A price of 0 is refused: MaximumSmartPotionUses divides by the base price.
    internal static int[] Read(JsonElement request, string field)
    {
        int[] costs = request.TryGetProperty(field, out var value)
            ? value.EnumerateArray().Select(cost => cost.GetInt32()).ToArray() : Native;
        if (costs.Length != 3 || costs[0] < 1 || costs[0] > costs[1] || costs[1] > costs[2] || costs[2] > 100)
            throw new ArgumentException(field + " must be three prices in HP, from 1 to 100 and not decreasing");
        return costs;
    }

    internal static void Set(int[] costs)
    {
        (Base, Elevated, High) = (costs[0], costs[1], costs[2]);
    }
}
