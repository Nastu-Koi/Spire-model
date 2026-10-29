using System.Threading.Tasks;

namespace SpireTakeover;

internal static class Patches
{
	internal static void End(bool __0)
	{
		LiveBridge.RunEnded(__0);
	}

	internal static void Cleanup()
	{
		LiveBridge.Reset();
		SelectionCapture.Reset();
	}

    // Track the full native task an executed action started, including removal and
    // animation after the synchronizer returns.
    internal static void ControlTask(object? __result)
    {
        if (LiveBridge.Executing && __result is Task task) LiveBridge.Track(task);
    }
}
