// Small domain fakes: transaction tests exercise the production cursor and identity rules.
namespace MegaCrit.Sts2.Core.Models {
    public record ModelId(string Entry);
    public sealed class CardModel(string id, int upgrade = 0, string state = "base") {
        public ModelId Id { get; } = new(id);
        public int CurrentUpgradeLevel { get; set; } = upgrade;
        public string State { get; set; } = state;
    }
}
namespace MegaCrit.Sts2.Core.Entities.Players {
    public sealed class Player { public Dictionary<Cards.PileType, Cards.CardPile> Piles { get; } = new(); }
}
namespace MegaCrit.Sts2.Core.Entities.Cards {
    public enum PileType { None, Hand, Discard, Draw, Exhaust, Deck }
    public sealed class CardPile { public List<Models.CardModel> Cards { get; } = new(); }
    public static class PileTypeExtensions { public static CardPile GetPile(this PileType type, Players.Player player) => player.Piles[type]; }
}
