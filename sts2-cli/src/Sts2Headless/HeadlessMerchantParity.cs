using System.Reflection.Emit;
using HarmonyLib;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.TestSupport;

namespace Sts2Headless;

/// <summary>Keep production merchant RNG semantics while TestMode disables UI.</summary>
internal static class HeadlessMerchantParity
{
    public const string Version = "native-potion-price-v1";
    private static bool _installed;

    public static void Install()
    {
        if (_installed) return;
        new Harmony("sts2headless.merchant-parity").Patch(
            AccessTools.Method(typeof(MerchantPotionEntry), nameof(MerchantPotionEntry.CalcCost)),
            transpiler: new HarmonyMethod(typeof(HeadlessMerchantParity), nameof(KeepNativePriceRoll)));
        _installed = true;
    }

    private static IEnumerable<CodeInstruction> KeepNativePriceRoll(IEnumerable<CodeInstruction> instructions)
    {
        var testModeOff = AccessTools.PropertyGetter(typeof(TestMode), nameof(TestMode.IsOff));
        int replaced = 0;
        foreach (var instruction in instructions)
        {
            if (instruction.Calls(testModeOff))
            {
                // Run the game's original float roll, rounding and assignment.
                // Retain labels/exception regions and the rest of the method.
                instruction.opcode = OpCodes.Ldc_I4_1;
                instruction.operand = null;
                replaced++;
            }
            yield return instruction;
        }
        if (replaced != 1)
            throw new InvalidOperationException("Merchant potion pricing changed: expected one TestMode.IsOff guard.");
    }
}
