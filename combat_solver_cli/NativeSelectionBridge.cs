using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using HarmonyLib;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.TestSupport;
using Sts2Headless;

namespace CombatSolverCli;

// Version-pinned instrumentation. Count is the first use of the actual, already
// filtered materialized candidates. No filter, RNG, or selection is replayed.
internal static class NativeSelectionBridge
{
    private sealed record Scope(ChoiceTransaction Transaction, ChoiceTransaction.Request Request, Scope? Parent, bool SingleCardResult = false);
    private static readonly AsyncLocal<Scope?> current = new();
    private static ChoiceTransaction? active;
    private static bool installed;
    internal static void Activate(ChoiceTransaction? transaction) => Volatile.Write(ref active, transaction);

    internal static void Install()
    {
        if (installed) return;
        var harmony = new Harmony("combat-solver-cli.native-selection-transactions");
        var methods = typeof(CardSelectCmd).GetMethods(BindingFlags.Public | BindingFlags.Static)
            .Where(m => m.Name.StartsWith("From", StringComparison.Ordinal)).ToArray();
        int leaves = 0;
        var wrappers = new List<MethodInfo>();
        foreach (var method in methods)
        {
            bool wrapper = method.Name is "FromHandForDiscard" or "FromDeckForRemoval"
                || method.Name == "FromCombatPile" && method.GetParameters().Length == 4
                || method.Name == "FromDeckForEnchantment" && method.GetParameters()[0].ParameterType == typeof(Player);
            if (wrapper) { wrappers.Add(method); continue; }
            var stateMachine = method.GetCustomAttribute<AsyncStateMachineAttribute>()?.StateMachineType
                ?? throw new InvalidOperationException("Unsupported native selection entry: " + method);
            harmony.Patch(method, prefix: new HarmonyMethod(typeof(NativeSelectionBridge), nameof(Before)) { priority = Priority.First },
                postfix: new HarmonyMethod(typeof(NativeSelectionBridge), method.ReturnType == typeof(Task<CardModel>) ? nameof(AfterCard) : nameof(AfterCards)),
                finalizer: new HarmonyMethod(typeof(NativeSelectionBridge), nameof(NativeFailed)));
            harmony.Patch(stateMachine.GetMethod("MoveNext", BindingFlags.NonPublic | BindingFlags.Instance)!,
                transpiler: new HarmonyMethod(typeof(NativeSelectionBridge), nameof(CaptureCandidates)));
            leaves++;
        }
        // Existing headless metadata patches may have compiled a forwarding
        // method with an inlined pre-patch leaf. Rebuild forwarding bodies after
        // installing leaf hooks; they still do not create requests.
        foreach (var wrapper in wrappers)
        {
            harmony.Patch(wrapper, prefix: new HarmonyMethod(typeof(NativeSelectionBridge), nameof(Forwarding)));
            var machine = wrapper.GetCustomAttribute<AsyncStateMachineAttribute>()?.StateMachineType;
            if (machine != null) harmony.Patch(machine.GetMethod("MoveNext", BindingFlags.NonPublic | BindingFlags.Instance)!,
                prefix: new HarmonyMethod(typeof(NativeSelectionBridge), nameof(Forwarding)));
        }
        if (leaves != 11) throw new InvalidOperationException($"Native selection manifest changed: {leaves} leaves");
        harmony.Patch(typeof(CardSelectCmd).GetMethod("LogChoice", BindingFlags.NonPublic | BindingFlags.Static)!,
            prefix: new HarmonyMethod(typeof(NativeSelectionBridge), nameof(Result)));
        harmony.Patch(typeof(CardPile).GetMethod("AddInternal", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)!,
            prefix: new HarmonyMethod(typeof(NativeSelectionBridge), nameof(Moving)));
        var reward = typeof(RunSimulator).GetNestedType("HeadlessCardSelector", BindingFlags.NonPublic)!.GetMethod("GetSelectedCardReward")!;
        harmony.Patch(reward, prefix: new HarmonyMethod(typeof(NativeSelectionBridge), nameof(BeforeReward)),
            postfix: new HarmonyMethod(typeof(NativeSelectionBridge), nameof(AfterReward)),
            finalizer: new HarmonyMethod(typeof(NativeSelectionBridge), nameof(RewardFailed)));
        installed = true;
    }

    private static void Forwarding() { }

    private static void BeforeReward(IReadOnlyList<CardCreationResult> options, out object? __state)
    {
        __state = null;
        var transaction = Volatile.Read(ref active);
        if (transaction == null) return;
        var request = transaction.BeginRequest("GetSelectedCardReward", "None", "unknown", 0, 1, true);
        var scope = new Scope(transaction, request, current.Value);
        __state = scope;
        transaction.OpenRequest(request, options.Select(c => c.Card));
    }
    private static void AfterReward(object? __state, CardRewardSelection __result)
    {
        if (__state is Scope scope)
        {
            if (__result.alternative != null && !__result.alternative.OptionId.Equals("Skip", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("Unadapted planned reward alternative: " + __result.alternative.OptionId);
            scope.Transaction.CompleteRequest(scope.Request, __result.card == null ? [] : [__result.card]);
        }
    }
    private static void RewardFailed(object? __state, Exception? __exception)
    {
        if (__state is Scope scope && __exception != null) scope.Transaction.Fault(__exception);
    }

    private static void Moving(CardPile __instance, CardModel __0, int __1) =>
        Volatile.Read(ref active)?.ObserveMove(__0, __instance.Type.ToString(), __1);

    private static void Before(MethodBase __originalMethod, object[] __args, out object? __state)
    {
        __state = null;
        var transaction = Volatile.Read(ref active);
        if (transaction == null) return;
        var player = __args.OfType<Player>().FirstOrDefault();
        if (player != null && !ReferenceEquals(player, transaction.Player)) return;
        var prefs = __args.OfType<CardSelectorPrefs>().Cast<CardSelectorPrefs?>().FirstOrDefault();
        var name = __originalMethod.Name;
        var source = __args.OfType<CardPile>().FirstOrDefault()?.Type.ToString()
            ?? (name.StartsWith("FromHand") ? "Hand" : name.StartsWith("FromDeck") ? "Deck" : "None");
        var effect = prefs?.Prompt.LocEntryKey switch
        {
            "TO_DISCARD" => "Discard", "TO_EXHAUST" => "Exhaust", "TO_UPGRADE" => "Upgrade",
            "TO_TRANSFORM" => "Transform", "TO_REMOVE" => "Remove", _ => "unknown"
        };
        if (name == "FromHandForUpgrade" || name == "FromDeckForUpgrade") effect = "Upgrade";
        int minimum = prefs?.MinSelect ?? (__args.OfType<bool>().FirstOrDefault() ? 0 : 1);
        int maximum = prefs?.MaxSelect ?? 1;
        if (name == "FromChooseABundleScreen") { minimum = 0; maximum = int.MaxValue; }
        var scope = new Scope(transaction, transaction.BeginRequest(name, source, effect, minimum, maximum, prefs?.Cancelable ?? false, current.Value?.Request.Id), current.Value, ((MethodInfo)__originalMethod).ReturnType == typeof(Task<CardModel>));
        __state = scope;
        current.Value = scope;
        // The CLI's existing bundle prefix replaces the entire native method.
        if (name == "FromChooseABundleScreen")
        {
            var bundles = __args.OfType<IReadOnlyList<IReadOnlyList<CardModel>>>().Single();
            transaction.OpenRequest(scope.Request, bundles.SelectMany(b => b));
        }
    }

    private static IEnumerable<CodeInstruction> CaptureCandidates(IEnumerable<CodeInstruction> instructions, MethodBase __originalMethod)
    {
        bool found = false;
        foreach (var instruction in instructions)
        {
            if (!found && instruction.operand is MethodInfo call &&
                (call.Name == "get_Count" || call.Name == "Count") &&
                (call.DeclaringType?.IsGenericType == true || call.IsGenericMethod))
            {
                var element = call.IsGenericMethod ? call.GetGenericArguments()[0] : call.DeclaringType!.GetGenericArguments()[0];
                string? observer = element == typeof(CardModel) ? nameof(Options)
                    : element == typeof(CardCreationResult) ? nameof(RewardOptions)
                    : element == typeof(IReadOnlyList<CardModel>) ? nameof(BundleOptions) : null;
                if (observer != null)
                {
                    var duplicate = new CodeInstruction(OpCodes.Dup);
                    duplicate.labels.AddRange(instruction.labels); instruction.labels.Clear();
                    duplicate.blocks.AddRange(instruction.blocks); instruction.blocks.Clear();
                    yield return duplicate;
                    yield return new CodeInstruction(OpCodes.Call, typeof(NativeSelectionBridge).GetMethod(observer, BindingFlags.NonPublic | BindingFlags.Static)!);
                    found = true;
                }
            }
            yield return instruction;
        }
        if (!found) throw new InvalidOperationException("Native candidate checkpoint missing: " + __originalMethod.DeclaringType);
    }

    private static void Options(IEnumerable<CardModel> cards)
    {
        var scope = current.Value;
        if (scope != null) scope.Transaction.OpenRequest(scope.Request, cards);
    }
    private static void RewardOptions(IEnumerable<CardCreationResult> cards) => Options(cards.Select(c => c.Card));
    private static void BundleOptions(IEnumerable<IReadOnlyList<CardModel>> cards) => Options(cards.SelectMany(c => c));
    private static void Result(Player player, IEnumerable<CardModel> cards)
    {
        var scope = current.Value;
        if (scope != null && ReferenceEquals(scope.Transaction.Player, player))
            scope.Transaction.CompleteRequest(scope.Request, cards, scope.SingleCardResult);
    }
    private static void NativeFailed(object? __state, Exception? __exception)
    {
        if (__state is Scope scope && __exception != null)
        {
            if (ReferenceEquals(current.Value, scope)) current.Value = scope.Parent;
            scope.Transaction.Fault(__exception);
        }
    }
    private static void AfterCards(object? __state, ref Task<IEnumerable<CardModel>> __result)
    {
        if (__state is not Scope scope) return;
        current.Value = scope.Parent;
        __result = FinishCards(__result, scope);
    }
    private static async Task<IEnumerable<CardModel>> FinishCards(Task<IEnumerable<CardModel>> task, Scope scope)
    {
        try
        {
            var result = await task.ConfigureAwait(false);
            if (!scope.Request.Completed) scope.Transaction.CompleteRequest(scope.Request, result);
            return result;
        }
        catch (Exception ex) { scope.Transaction.Fault(ex); throw; }
    }
    private static void AfterCard(object? __state, ref Task<CardModel> __result)
    {
        if (__state is not Scope scope) return;
        current.Value = scope.Parent;
        __result = FinishCard(__result, scope);
    }
    private static async Task<CardModel> FinishCard(Task<CardModel> task, Scope scope)
    {
        try
        {
            var result = await task.ConfigureAwait(false);
            if (!scope.Request.Completed) scope.Transaction.CompleteRequest(scope.Request, [result], singleCardResult: true);
            return result;
        }
        catch (Exception ex) { scope.Transaction.Fault(ex); throw; }
    }
}
