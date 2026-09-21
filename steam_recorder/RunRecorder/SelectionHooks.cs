using System;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Rewards;

namespace RunRecorder;

internal static class SelectionHooks
{
	internal static void Install(Harmony harmony, Assembly assembly, bool includeRewardRefresh = true)
	{
		MethodInfo[] array = (from m in typeof(CardSelectCmd).GetMethods(BindingFlags.Static | BindingFlags.Public)
			where m.Name.StartsWith("From", StringComparison.Ordinal) && typeof(Task).IsAssignableFrom(m.ReturnType)
			select m).Append(AccessTools.DeclaredMethod(typeof(CardReward), "OnSelect")).ToArray();
		foreach (MethodInfo methodInfo in array)
		{
			harmony.Patch(methodInfo, new HarmonyMethod(typeof(SelectionPatches), "SelectionEnter")
			{
				priority = 800
			}, new HarmonyMethod(typeof(SelectionPatches), "SelectionExit"), null, new HarmonyMethod(typeof(SelectionPatches), "SelectionFailed"));
			Type type = methodInfo.GetCustomAttribute<AsyncStateMachineAttribute>()?.StateMachineType;
			if (type != null)
			{
				harmony.Patch(AccessTools.Method(type, "MoveNext"), null, null, new HarmonyMethod(typeof(SelectionPatches), "ObserveSelectorCalls"));
			}
			Recorder.HookReport.Add(new
			{
				type = methodInfo.DeclaringType.FullName,
				method = methodInfo.Name,
				purpose = "selection_scope"
			});
		}
		harmony.Patch(AccessTools.Method(typeof(PlayerChoiceSynchronizer), "ReserveChoiceId"), null, new HarmonyMethod(typeof(SelectionPatches), "ChoiceReserved"));
		harmony.Patch(AccessTools.Method(typeof(NPlayerHand), "UpdateSelectModeCardVisibility"), null, new HarmonyMethod(typeof(SelectionPatches), "SelectionHand"));
        harmony.Patch(AccessTools.Method(typeof(NPlayerHand), "SelectCards"),
            new HarmonyMethod(typeof(SelectionPatches), "SelectionHandEntry") { priority = Priority.First });
        Recorder.HookReport.Add(new { type = typeof(NPlayerHand).FullName,
            method = "SelectCards", purpose = "selection_before_submission" });
		Recorder.HookReport.Add(new
		{
			type = "NPlayerHand",
			purpose = "selection_hand",
			method = "UpdateSelectModeCardVisibility"
		});
		Type typeFromHandle = typeof(NCardRewardSelectionScreen);
		harmony.Patch(AccessTools.Method(typeFromHandle, "ShowScreen"), null, new HarmonyMethod(typeof(SelectionPatches), "RewardScreenCreated"));
		if (includeRewardRefresh)
		{
			harmony.Patch(AccessTools.Method(typeFromHandle, "RefreshOptions"), null, new HarmonyMethod(typeof(SelectionPatches), "RewardScreenRefreshed"));
			Recorder.HookReport.Add(new
			{
				type = typeFromHandle.FullName,
				purpose = "selection_refresh",
				method = "RefreshOptions"
			});
		}
		string[] array2 = new string[8] { "NChooseACardSelectionScreen", "NChooseABundleSelectionScreen", "NSimpleCardSelectScreen", "NDeckCardSelectScreen", "NDeckEnchantSelectScreen", "NDeckTransformSelectScreen", "NDeckUpgradeSelectScreen", "NCardRewardSelectionScreen" };
		foreach (string text in array2)
		{
			Type type2 = assembly.GetType("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection." + text, throwOnError: true);
			MethodInfo[] array3 = type2.GetMethods(BindingFlags.DeclaredOnly | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).Where(delegate(MethodInfo m)
			{
				string name = m.Name;
				return (name == "Create" || name == "ShowScreen") ? true : false;
			}).ToArray();
			if (array3.Length == 0)
			{
				throw new MissingMethodException(type2.FullName, "Create/ShowScreen");
			}
			array = array3;
			foreach (MethodInfo original in array)
			{
				harmony.Patch(original, new HarmonyMethod(typeof(SelectionPatches), "SelectionScreen")
				{
					priority = 800
				});
			}
			Recorder.HookReport.Add(new
			{
				type = type2.FullName,
				purpose = "selection_offer",
				methods = array3.Select((MethodInfo m) => m.Name).ToArray()
			});
		}
	}
}
