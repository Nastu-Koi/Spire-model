using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Rooms;

namespace Sts2Headless;

public partial class RunSimulator
{
    private readonly HashSet<AbstractRoom> _protocolClosedMerchants = new(ReferenceEqualityComparer.Instance);
    private static bool _potionPresentationPatched;

    private static void InstallPotionPresentationBridge()
    {
        if (_potionPresentationPatched) return;
        var harmony = new Harmony("sts2headless.potion-presentation");
        harmony.Patch(AccessTools.PropertyGetter(typeof(FoulPotion), nameof(FoulPotion.PassesCustomUsabilityCheck)),
            prefix: new HarmonyMethod(typeof(RunSimulator), nameof(CanThrowFoulPotion)));
        harmony.Patch(AccessTools.Method(typeof(FoulPotion), "OnUse"),
            prefix: new HarmonyMethod(typeof(RunSimulator), nameof(ThrowFoulPotion)));
        _potionPresentationPatched = true;
    }

    private static bool CanThrowFoulPotion(FoulPotion __instance, ref bool __result)
    {
        var sim = _protocolUiOwner;
        if (sim == null || !sim._protocolEnabled || CombatManager.Instance.IsInProgress) return true;
        var room = __instance.Owner.RunState.CurrentRoom;
        __result = !sim._protocolClosedMerchants.Contains(room!) && (room is MerchantRoom
            || room is EventRoom { LocalMutableEvent: FakeMerchant { StartedFight: false } });
        return false;
    }

    private static bool ThrowFoulPotion(FoulPotion __instance, ref Task __result)
    {
        var sim = _protocolUiOwner;
        if (sim == null || !sim._protocolEnabled || CombatManager.Instance.IsInProgress) return true;
        var room = __instance.Owner.RunState.CurrentRoom;
        if (room is EventRoom { LocalMutableEvent: FakeMerchant fake })
        {
            __result = fake.FoulPotionThrown(__instance);
            return false; // Original branch requires a Godot merchant node; effects are native.
        }
        if (room is MerchantRoom) sim._protocolClosedMerchants.Add(room);
        return true; // Gold, consumption and potion hooks stay in the original methods.
    }
}
