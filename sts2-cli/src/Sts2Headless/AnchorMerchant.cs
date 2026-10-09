using HarmonyLib;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Models;

namespace Sts2Headless;

/// <summary>
/// The parts of a merchant's stock that a run summary names and the game drew
/// from streams no summary keeps: the rarity of each character card (reward
/// stream and pity counter) and the three relics (grab bag). While an anchor
/// supplies them, the game's own shop code runs otherwise untouched: its shop
/// stream still picks the cards within each rarity, the potions and every
/// price, so what it shows can be held against the record.
/// </summary>
internal static class AnchorMerchant
{
    private static bool _installed;
    private static Queue<CardRarity>? _rarities;
    private static List<RelicModel>? _relics;

    public static void Install()
    {
        if (_installed) return;
        var harmony = new Harmony("sts2headless.anchor-merchant");
        harmony.Patch(AccessTools.Method(typeof(Hook), nameof(Hook.ModifyMerchantCardRarity)),
            postfix: new HarmonyMethod(typeof(AnchorMerchant), nameof(RecordedRarity)));
        harmony.Patch(AccessTools.Method(typeof(MerchantInventory), "PopulateRelicEntries"),
            prefix: new HarmonyMethod(typeof(AnchorMerchant), nameof(RecordedRelics)));
        _installed = true;
    }

    /// Supply the next merchant with `rarities` for its character cards, in slot
    /// order, and with `relics` for its relic slots.
    public static void Begin(IEnumerable<CardRarity> rarities, IEnumerable<RelicModel> relics)
    {
        _rarities = new Queue<CardRarity>(rarities);
        _relics = relics.ToList();
    }

    /// True when the merchant consumed exactly what was supplied.
    public static bool End()
    {
        bool consumed = _rarities is { Count: 0 } && _relics == null;
        _rarities = null;
        _relics = null;
        return consumed;
    }

    // The game asks this hook once for each card it creates, character cards first.
    private static void RecordedRarity(ref CardRarity __result)
    {
        if (_rarities is { Count: > 0 }) __result = _rarities.Dequeue();
    }

    // Each entry prices itself from the shop stream as it is created, as the game's own would.
    private static bool RecordedRelics(MerchantInventory __instance)
    {
        if (_relics == null) return true;
        foreach (var relic in _relics)
            __instance.AddRelicEntry(new MerchantRelicEntry(relic.ToMutable(), __instance.Player));
        _relics = null;
        return false;
    }
}
