using System;
using System.Linq;
using System.Reflection;
using System.Text.Json.Serialization;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Entities.Cards;

namespace RunRecorder;

// Use the training selector's public vocabulary. Unrecognized prompts stay
// unknown; neither candidate order nor an internal card identity implies an effect.
internal sealed record SelectionMetadata(
    [property: JsonPropertyName("operation")] string Operation = "unknown",
    [property: JsonPropertyName("source")] string Source = "unknown",
    [property: JsonPropertyName("destination")] string Destination = "unknown")
{
    [JsonPropertyName("known_masks")]
    public object KnownMasks => new
    {
        operation = Operation != "unknown",
        source = Source != "unknown",
        destination = Destination != "unknown",
        order_matters = false
    };

    internal static SelectionMetadata From(MethodBase method, object[] args, SelectionMetadata? previous = null)
    {
        var prefs = args.OfType<CardSelectorPrefs>().Select(p => (CardSelectorPrefs?)p).FirstOrDefault();
        var pile = args.OfType<CardPile>().FirstOrDefault();
        return Derive(method.Name, prefs, pile?.Type, previous);
    }

    internal static SelectionMetadata Derive(string method, CardSelectorPrefs? prefs = null,
        PileType? pile = null, SelectionMetadata? previous = null)
    {
        previous ??= new();
        var operation = prefs?.Prompt?.LocEntryKey switch
        {
            "TO_TRANSFORM" => "transform",
            "TO_EXHAUST" => "exhaust",
            "TO_REMOVE" => "remove",
            "TO_ENCHANT" => "enchant",
            "TO_DISCARD" => "discard",
            "TO_UPGRADE" or "CHOOSE_CARD_UPGRADE_HEADER" => "upgrade",
            _ => method switch
            {
                "FromHandForDiscard" => "discard",
                "FromHandForUpgrade" or "FromDeckForUpgrade" => "upgrade",
                "FromDeckForRemoval" => "remove",
                "FromDeckForTransformation" => "transform",
                "FromDeckForEnchantment" => "enchant",
                _ => previous.Operation
            }
        };
        var source = pile.HasValue ? PileSource(pile.Value)
            : method.Contains("FromHand", StringComparison.Ordinal) ? "hand"
            : method.Contains("FromDeck", StringComparison.Ordinal) ? "deck" : previous.Source;
        var destination = operation switch
        {
            "discard" => "discard_pile",
            "exhaust" => "exhaust_pile",
            "upgrade" or "enchant" => source,
            "remove" => "removed",
            _ => "unknown"
        };
        return new(operation, source, destination);
    }

    private static string PileSource(PileType pile) => pile switch
    {
        PileType.Hand => "hand",
        PileType.Deck => "deck",
        PileType.Draw => "draw_pile",
        PileType.Discard => "discard_pile",
        PileType.Exhaust => "exhaust_pile",
        _ => "unknown"
    };
}
