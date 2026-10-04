using HarmonyLib;
using MegaCrit.Sts2.Core.Models.Relics;

namespace Sts2Headless;

// Game code can start state-changing work with TaskHelper.RunSafely that neither
// the action executor nor a headless operation owns. Lord's Parasol buys the whole
// shop after the room is entered and opens deck selections (e.g. Kifuda, then the
// card removal) between purchases. A boundary published in such a gap exposes stale
// shop candidates, so the protocol waits until the work completes or asks for input.
internal static class NativeContinuations
{
    private static readonly object Gate = new();
    private static readonly List<Task> Tasks = new();
    private static bool _installed;

    public static void Install()
    {
        if (_installed) return;
        new Harmony("sts2headless.native-continuations").Patch(
            AccessTools.DeclaredMethod(typeof(LordsParasol), "PurchaseEverything")
                ?? throw new MissingMethodException(typeof(LordsParasol).FullName, "PurchaseEverything"),
            postfix: new HarmonyMethod(typeof(NativeContinuations), nameof(Track)));
        _installed = true;
    }

    private static void Track(Task __result)
    {
        lock (Gate) Tasks.Add(__result);
    }

    public static void Clear()
    {
        lock (Gate) Tasks.Clear();
    }

    /// <summary>True while tracked work is still running. A faulted task is
    /// rethrown once so the caller can fail the protocol instead of continuing
    /// from a partially applied native effect.</summary>
    public static bool Pending
    {
        get
        {
            lock (Gate)
            {
                var faulted = Tasks.FirstOrDefault(t => t.IsFaulted || t.IsCanceled);
                Tasks.RemoveAll(t => t.IsCompleted);
                faulted?.GetAwaiter().GetResult();
                return Tasks.Count > 0;
            }
        }
    }
}
