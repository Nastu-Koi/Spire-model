using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Commands;

namespace Sts2Headless;

internal sealed record SelectionMetadata(string Operation = "unknown", string Source = "unknown",
    string Destination = "unknown", bool Cancelable = false, int? Minimum = null)
{
    public static readonly AsyncLocal<SelectionMetadata?> Current = new();
    private static bool _installed;

    public static void Install()
    {
        if (_installed) return;
        var harmony = new Harmony("sts2headless.selection-context");
        foreach (var method in typeof(CardSelectCmd).GetMethods(BindingFlags.Public | BindingFlags.Static)
            .Where(m => m.Name.StartsWith("From", StringComparison.Ordinal) && m.Name != "FromChooseABundleScreen"))
            harmony.Patch(method, prefix: new HarmonyMethod(typeof(SelectionMetadata), nameof(Before)),
                postfix: new HarmonyMethod(typeof(SelectionMetadata), nameof(After)));
        _installed = true;
    }

    private static void Before(MethodBase __originalMethod, object[] __args, out SelectionMetadata? __state)
    {
        __state = Current.Value;
        var previous = __state ?? new SelectionMetadata();
        var parameters = __originalMethod.GetParameters();
        var prefsIndex = Array.FindIndex(parameters, p => p.ParameterType == typeof(CardSelectorPrefs));
        CardSelectorPrefs? prefs = prefsIndex >= 0 ? (CardSelectorPrefs)__args[prefsIndex] : null;
        string operation = prefs?.Prompt.LocEntryKey switch
        {
            "TO_TRANSFORM" => "transform", "TO_EXHAUST" => "exhaust", "TO_REMOVE" => "remove",
            "TO_ENCHANT" => "enchant", "TO_DISCARD" => "discard", "TO_UPGRADE" => "upgrade",
            _ => previous.Operation,
        };
        var name = __originalMethod.Name;
        string source = name.Contains("FromHand") ? "hand" : name.Contains("FromDeck") ? "deck" : previous.Source;
        string destination = operation == "discard" ? "discard_pile" : operation == "exhaust" ? "exhaust_pile"
            : operation is "upgrade" or "enchant" ? source : operation == "remove" ? "removed" : "unknown";
        int? minimum = previous.Minimum;
        if (name == "FromChooseACardScreen")
        {
            int index = Array.FindIndex(parameters, p => p.Name == "canSkip");
            minimum = index >= 0 && (bool)__args[index] ? 0 : 1;
        }
        Current.Value = new(operation, source, destination, prefs?.Cancelable ?? previous.Cancelable, minimum);
    }

    private static void After(SelectionMetadata? __state) => Current.Value = __state;
}
