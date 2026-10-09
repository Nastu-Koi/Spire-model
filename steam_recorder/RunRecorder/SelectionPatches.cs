using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.TestSupport;

namespace RunRecorder;

internal static class SelectionPatches
{
    internal static void CombatPileScreenCreated(object __result) =>
        SelectionCapture.Safe(() => SelectionCapture.BindCombatPileScreen(__result));

    internal static void CombatPileScreenRefreshed(object __instance) =>
        SelectionCapture.Safe(() => SelectionCapture.RefreshCombatPileScreen(__instance));

    internal static void SelectionHandEntry(CardSelectorPrefs __0, Func<CardModel, bool>? __1)
    {
        SelectionCapture.Safe(() => SelectionCapture.ObserveHandEntry(__0, __1));
    }
	internal static void RewardScreenCreated(object? __result)
	{
		SelectionCapture.Safe(delegate
		{
			SelectionCapture.BindRewardScreen(__result);
		});
	}

	internal static void RewardScreenRefreshed(object __instance, IReadOnlyList<CardCreationResult> __0, IReadOnlyList<CardRewardAlternative> __1)
	{
		SelectionCapture.Safe(delegate
		{
			SelectionCapture.RefreshRewardScreen(__instance, __0.ToArray(), __1.ToArray());
		});
	}

    internal static void RewardSelectionStarted(object __instance)
    {
        SelectionCapture.Safe(() =>
        {
            // CardReward removes a taken card from its mutable list without
            // calling RefreshOptions when a relic allows another pick.
            if (Snapshot.Read(__instance, "_options") is IReadOnlyList<CardCreationResult> cards &&
                Snapshot.Read(__instance, "_extraOptions") is IReadOnlyList<CardRewardAlternative> alternatives)
                SelectionCapture.RefreshRewardScreen(__instance, cards.ToArray(), alternatives.ToArray());
            if (Snapshot.Read(Snapshot.Read(__instance, "_completionSource"), "Task") is Task completion)
                LiveRewards.RewardRoundStarted(__instance, completion);
        });
    }

	internal static void SelectionEnter(MethodBase __originalMethod, object? __instance, object[] __args, out SelectionScope? __state)
	{
		__state = SelectionCapture.Enter(__originalMethod, __args, __instance);
	}

	internal static void SelectionExit(SelectionScope? __state, object? __result)
	{
		SelectionCapture.Exit(__state, __result);
	}

	internal static Exception? SelectionFailed(SelectionScope? __state, Exception? __exception)
	{
		if (__exception != null)
		{
			SelectionCapture.Exit(__state, null);
		}
		return __exception;
	}

	internal static void ChoiceReserved(Player __0, uint __result)
	{
		SelectionCapture.Safe(delegate
		{
			SelectionCapture.Reserved(__0, __result);
		});
	}

	internal static void SelectionScreen(MethodBase __originalMethod, object[] __args)
	{
		SelectionCapture.Safe(delegate
		{
			SelectionCapture.ObserveScreen(__originalMethod, __args);
		});
	}

	internal static void SelectionHand(NPlayerHand __instance)
	{
		SelectionCapture.Safe(delegate
		{
			SelectionCapture.ObserveHand(__instance);
		});
	}

	internal static IEnumerable<CodeInstruction> ObserveSelectorCalls(IEnumerable<CodeInstruction> instructions, MethodBase __originalMethod)
	{
		FieldInfo[] fields = __originalMethod.DeclaringType.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
		FieldInfo prefs = fields.SingleOrDefault((FieldInfo f) => f.FieldType == typeof(CardSelectorPrefs));
		FieldInfo skip = fields.SingleOrDefault((FieldInfo f) => f.Name == "canSkip" && f.FieldType == typeof(bool));
		foreach (CodeInstruction instruction in instructions)
		{
			if (instruction.opcode == OpCodes.Callvirt && instruction.operand is MethodInfo methodInfo && methodInfo.DeclaringType == typeof(ICardSelector) && methodInfo.Name == "GetSelectedCards")
			{
				if (prefs != null || skip != null)
				{
					CodeInstruction codeInstruction = new CodeInstruction(OpCodes.Ldarg_0);
					codeInstruction.labels.AddRange(instruction.labels);
					instruction.labels.Clear();
					yield return codeInstruction;
					yield return new CodeInstruction(OpCodes.Ldfld, prefs ?? skip);
				}
				instruction.opcode = OpCodes.Call;
				instruction.operand = AccessTools.Method(typeof(SelectionCapture), (prefs != null) ? "GetSelectedCardsWithPrefs" : ((skip != null) ? "GetSelectedCardsWithSkip" : "GetSelectedCards"));
			}
			yield return instruction;
		}
	}
}
