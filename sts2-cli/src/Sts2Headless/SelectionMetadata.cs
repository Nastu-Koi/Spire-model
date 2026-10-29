using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;

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

    private sealed record State(SelectionMetadata? Previous, IDisposable? Restore);

    private static void Before(MethodBase __originalMethod, object[] __args, out State __state)
    {
        var previous = Current.Value ?? new SelectionMetadata();
        // With a test selector installed the game skips a selection's pause signal, so a
        // turn's setup ran to its end before the side's turn-start effects (D6C3). Where
        // the selection has a choice context, the headless selector steps aside until the
        // method first yields: the game then signals as it does for the local player,
        // and answers through the local selector, which is the same headless selector.
        __state = new(Current.Value,
            CardSelectCmd.Selector is RunSimulator.HeadlessCardSelector && __args.OfType<PlayerChoiceContext>().Any()
                ? CardSelectCmd.SuspendSelectorForTest() : null);
        var parameters = __originalMethod.GetParameters();
        var prefsIndex = Array.FindIndex(parameters, p => p.ParameterType == typeof(CardSelectorPrefs));
        CardSelectorPrefs? prefs = prefsIndex >= 0 ? (CardSelectorPrefs)__args[prefsIndex] : null;
        var name = __originalMethod.Name;
        // The same derivation as the Steam recorder's (steam_takeover/SpireTakeover/SelectionMetadata.cs).
        string operation = prefs?.Prompt.LocEntryKey switch
        {
            "TO_TRANSFORM" => "transform", "TO_EXHAUST" => "exhaust", "TO_REMOVE" => "remove",
            "TO_ENCHANT" => "enchant", "TO_DISCARD" => "discard",
            "TO_UPGRADE" or "CHOOSE_CARD_UPGRADE_HEADER" => "upgrade",
            _ => name switch
            {
                "FromHandForDiscard" => "discard",
                "FromHandForUpgrade" or "FromDeckForUpgrade" => "upgrade",
                "FromDeckForRemoval" => "remove",
                "FromDeckForTransformation" => "transform",
                "FromDeckForEnchantment" => "enchant",
                _ => previous.Operation,
            },
        };
        var pile = __args.OfType<CardPile>().FirstOrDefault();
        string source = pile != null ? PileSource(pile.Type)
            : name.Contains("FromHand") ? "hand" : name.Contains("FromDeck") ? "deck" : previous.Source;
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

    private static void After(State __state)
    {
        __state.Restore?.Dispose();
        Current.Value = __state.Previous;
    }

    private static string PileSource(PileType pile) => pile switch
    {
        PileType.Hand => "hand",
        PileType.Deck => "deck",
        PileType.Draw => "draw_pile",
        PileType.Discard => "discard_pile",
        PileType.Exhaust => "exhaust_pile",
        _ => "unknown",
    };
}
