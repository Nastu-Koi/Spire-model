using System;
using System.Reflection;
using MegaCrit.Sts2.Core.GameActions;

namespace RunRecorder;

internal static class Patches
{
	internal static void InputBegin(MethodBase __originalMethod, out InputSnapshot? __state)
	{
		__state = Recorder.BeginInput((__originalMethod.Name == "EndTurn") ? "EndPlayerTurnAction" : "UsePotionAction");
	}

	internal static Exception? InputEnd(InputSnapshot? __state, Exception? __exception)
	{
		Recorder.EndInput(__state);
		return __exception;
	}

	internal static void Enqueue(object __instance, GameAction __0)
	{
		Recorder.Enqueue(__instance, __0);
	}

	internal static void End(bool __0)
	{
		Recorder.RunEnded(__0);
	}

	internal static void Cleanup()
	{
		Recorder.Safe(delegate
		{
			Recorder.Close("run_cleanup");
		});
	}

	internal static void Prefix(MethodBase __originalMethod, object? __instance, object[] __args, out Pending? __state)
	{
		__state = Recorder.Begin(__originalMethod, __instance, __args);
	}

	internal static void Postfix(Pending? __state, object? __result)
	{
		Recorder.Returned(__state, __result);
	}

	internal static void VoidPostfix(Pending? __state)
	{
		Recorder.Returned(__state, null);
	}

	internal static Exception? Finalizer(Pending? __state, Exception? __exception)
	{
		Recorder.Failed(__state, __exception);
		return __exception;
	}
}
