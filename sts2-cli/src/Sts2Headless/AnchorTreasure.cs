using HarmonyLib;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;

namespace Sts2Headless;

/// <summary>
/// Condition only the chest's automatic gold roll on summary evidence. The
/// original draw still advances the stream; native ascension, gold modifiers
/// and pickup effects still run. The caller must check the resulting state and
/// reject different public observations consistent with the same summary.
/// This is not a restoration of the historical reward RNG.
/// </summary>
internal static class AnchorTreasure
{
    private sealed class Draw(Rng rng, int value)
    {
        public readonly Rng Rng = rng;
        public readonly int Value = value;
        public int Calls;
    }

    private static readonly AsyncLocal<Draw?> _draw = new();
    private static bool _installed;

    public static void Install()
    {
        if (_installed) return;
        new Harmony("sts2headless.anchor-treasure").Patch(
            AccessTools.Method(typeof(Rng), nameof(Rng.NextInt), [typeof(int), typeof(int)]),
            postfix: new HarmonyMethod(typeof(AnchorTreasure), nameof(RecordedDraw)));
        _installed = true;
    }

    public static async Task NormalRewards(TreasureRoom room, Rng rng, int? goldRoll)
    {
        if (goldRoll == null) { await room.DoNormalRewards(); return; }
        if (goldRoll is < 42 or > 52) throw new ArgumentOutOfRangeException(nameof(goldRoll));
        var previous = _draw.Value;
        var draw = new Draw(rng, goldRoll.Value);
        _draw.Value = draw;
        try
        {
            await room.DoNormalRewards();
            if (draw.Calls != 1) throw new InvalidOperationException("Chest did not draw exactly one gold amount");
        }
        finally { _draw.Value = previous; }
    }

    private static void RecordedDraw(Rng __instance, int minInclusive, int maxExclusive, ref int __result)
    {
        if (_draw.Value is not { } draw || !ReferenceEquals(__instance, draw.Rng)
            || minInclusive != 42 || maxExclusive != 53) return;
        draw.Calls++;
        __result = draw.Value;
    }
}
