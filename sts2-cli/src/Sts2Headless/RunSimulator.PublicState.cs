using System.Text.Json;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Combat.History.Entries;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using System.Text.RegularExpressions;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;

namespace Sts2Headless;

public partial class RunSimulator
{
    private sealed class PublicSnapshot
    {
        public List<Dictionary<string, object?>> Entities { get; } = new();
        public List<object> Relations { get; } = new();
        public List<Dictionary<string, object?>> Memory { get; set; } = new();
        public Dictionary<CardModel, string> Cards { get; } = new(ReferenceEqualityComparer.Instance);
        public Dictionary<Creature, string> Creatures { get; } = new(ReferenceEqualityComparer.Instance);
        public Dictionary<PotionModel, string> Potions { get; } = new(ReferenceEqualityComparer.Instance);
        public Dictionary<MapCoord, string> Map { get; } = new();

        // CombatSolverCli invokes AddCard by reflection with exactly these three
        // arguments; keep the name unique and the signature unchanged.
        public string AddCard(CardModel card, string zone, int? position = null) =>
            Cards.TryGetValue(card, out var existing) ? existing
                : AddDescribedCard(card, PublicSelectionCard(card), zone, position);

        private string AddDescribedCard(CardModel card, Dictionary<string, object?> entity, string zone, int? position)
        {
            var reference = $"card:{Cards.Count}";
            entity["ref"] = reference;
            entity["zone"] = zone;
            entity["owner_ref"] = "player";
            if (position != null) entity["position"] = position;
            Entities.Add(entity);
            Cards[card] = reference;
            return reference;
        }

        public void AddPile(IEnumerable<CardModel> cards, string zone, bool ordered = false)
        {
            // Describing a card runs the game's cost and keyword hooks, and every pile is
            // exported on every frame: describe each card once and reuse it as the sort key.
            var list = cards.Select(card => (card, entity: PublicSelectionCard(card))).ToList();
            if (!ordered)
                list = list.OrderBy(pair => JsonSerializer.Serialize(pair.entity), StringComparer.Ordinal).ToList();
            for (int i = 0; i < list.Count; i++)
                if (!Cards.ContainsKey(list[i].card))
                    AddDescribedCard(list[i].card, list[i].entity, zone, ordered ? i : null);
            Entities.Add(new() { ["entity_type"] = "pile_summary", ["zone"] = zone,
                ["count"] = list.Count, ["complete"] = true, ["order_known"] = ordered });
        }
    }

    private PublicSnapshot BuildPublicSnapshot()
    {
        var snapshot = new PublicSnapshot();
        var player = _runState!.Players[0];
        var pcs = player.PlayerCombatState;
        var combat = player.Creature.CombatState;
        snapshot.Entities.Add(new()
        {
            ["ref"] = "player", ["entity_type"] = "player", ["character"] = player.Character.Id.Entry,
            ["ascension"] = _protocolAscension,
            ["act"] = _runState.CurrentActIndex + 1, ["floor"] = _runState.ActFloor,
            ["hp"] = player.Creature.CurrentHp, ["max_hp"] = player.Creature.MaxHp,
            ["block"] = player.Creature.Block, ["gold"] = player.Gold,
            ["energy"] = pcs?.Energy, ["max_energy"] = pcs?.MaxEnergy, ["stars"] = pcs?.Stars,
            ["round"] = combat?.RoundNumber, ["capacity"] = player.MaxPotionCount,
            // The same rule the map screen applies: while it holds, every room of
            // the next floor can be entered, whatever the drawn paths.
            ["free_travel"] = Hook.ShouldAllowFreeTravel(_runState),
            // Same visible, turn-scoped damage fact used by Spite's glow. Healing
            // afterwards does not undo the event; no hidden history is exported.
            ["lost_hp_this_turn"] = combat != null && CombatManager.Instance.IsInProgress
                ? CombatManager.Instance.History.Entries.OfType<DamageReceivedEntry>().Any(e =>
                    e.HappenedThisTurn(combat) && e.Receiver == player.Creature && e.Result.UnblockedDamage > 0)
                : (bool?)null,
        });
        snapshot.Creatures[player.Creature] = "player";
        snapshot.AddPile(player.Deck.Cards, "deck");
        if (CombatManager.Instance.IsInProgress && pcs != null)
        {
            snapshot.AddPile(pcs.Hand.Cards, "hand", ordered: true);
            snapshot.AddPile(pcs.DrawPile.Cards, "draw_pile");
            snapshot.AddPile(pcs.DiscardPile.Cards, "discard_pile");
            snapshot.AddPile(pcs.ExhaustPile.Cards, "exhaust_pile");
            if (combat != null)
            {
                foreach (var creature in combat.Creatures.Where(c => !ReferenceEquals(c, player.Creature)))
                {
                    var reference = $"creature:{snapshot.Creatures.Count}";
                    snapshot.Creatures[creature] = reference;
                    snapshot.Entities.Add(new()
                    {
                        ["ref"] = reference,
                        ["entity_type"] = creature.Side == player.Creature.Side ? "summon" : "enemy",
                        ["content_id"] = creature.Monster?.Id.ToString(),
                        ["hp"] = creature.CurrentHp, ["max_hp"] = creature.MaxHp, ["block"] = creature.Block,
                        ["known"] = true,
                    });
                    // NextMove's identity and internal AI state are not serialized. Only
                    // the public intent icons and displayed attack amounts are exported.
                    foreach (var intent in creature.Monster?.NextMove?.Intents ?? [])
                    {
                        var visible = new Dictionary<string, object?>
                        {
                            ["entity_type"] = "intent", ["owner_ref"] = reference,
                            ["intent"] = intent.IntentType.ToString(),
                        };
                        if (intent is AttackIntent attack)
                        {
                            var targets = combat.PlayerCreatures.ToList();
                            visible["damage"] = attack.GetSingleDamage(targets, creature);
                            visible["hits"] = attack.Repeats;
                            visible["total_damage"] = attack.GetTotalDamage(targets, creature);
                        }
                        snapshot.Entities.Add(visible);
                    }
                }
            }
            if (pcs.OrbQueue != null)
            {
                snapshot.Entities.Add(new() { ["entity_type"] = "orb_slots", ["capacity"] = pcs.OrbQueue.Capacity });
                foreach (var (orb, i) in pcs.OrbQueue.Orbs.Select((o, i) => (o, i)))
                    snapshot.Entities.Add(new() { ["entity_type"] = "orb", ["content_id"] = orb.Id.ToString(),
                        ["owner_ref"] = "player", ["position"] = i,
                        ["stats"] = new { passive = orb.PassiveVal, evoke = orb.EvokeVal } });
            }
        }
        // Powers and relics export the values their hover text displays, and a
        // relic its counter when the game draws one on its icon.
        foreach (var (creature, reference) in snapshot.Creatures)
            foreach (var power in creature.Powers)
                snapshot.Entities.Add(new() { ["entity_type"] = "power", ["content_id"] = power.Id.ToString(),
                    ["owner_ref"] = reference, ["stacks"] = power.Amount,
                    ["stats"] = power.DynamicVars.Values.ToDictionary(v => v.Name, v => (object?)v.BaseValue) });
        foreach (var relic in player.Relics)
        {
            var entity = new Dictionary<string, object?> { ["entity_type"] = "relic",
                ["content_id"] = relic.Id.ToString(), ["owner_ref"] = "player",
                ["stats"] = relic.DynamicVars.Values.ToDictionary(v => v.Name, v => (object?)v.BaseValue),
                ["effect_coverage"] = "opaque", ["semantic_program"] = OpaqueProgram(relic.Id.ToString()) };
            if (relic.ShowCounter) entity["counter"] = relic.DisplayAmount;
            snapshot.Entities.Add(entity);
        }
        for (int i = 0; i < player.MaxPotionCount; i++)
        {
            var potion = player.GetPotionAtSlotIndex(i);
            if (potion == null) continue;
            var reference = $"potion:{i}";
            snapshot.Potions[potion] = reference;
            snapshot.Entities.Add(new() { ["ref"] = reference, ["entity_type"] = "potion", ["slot"] = i,
                ["content_id"] = potion.Id.ToString(), ["owner_ref"] = "player", ["target_type"] = potion.TargetType.ToString(),
                ["stats"] = potion.DynamicVars.Values.ToDictionary(v => v.Name, v => (object?)v.BaseValue),
                ["effect_coverage"] = "opaque", ["semantic_program"] = OpaqueProgram(potion.Id.ToString()) });
        }
        if (_runState.Map is { } map)
        {
            var points = map.GetAllMapPoints().Concat(new[] { map.StartingMapPoint, map.BossMapPoint })
                .Concat(map.SecondBossMapPoint is { } second ? new[] { second } : []).Distinct().ToList();
            foreach (var point in points)
            {
                var reference = $"map:{point.coord.col}:{point.coord.row}";
                snapshot.Map[point.coord] = reference;
                var node = new Dictionary<string, object?> { ["ref"] = reference, ["entity_type"] = "map_node",
                    ["content_id"] = point.PointType.ToString(), ["floor"] = (int)point.coord.row,
                    ["col"] = (int)point.coord.col, ["current"] = _runState.CurrentMapCoord == point.coord,
                    ["visited"] = _runState.VisitedMapCoords.Contains(point.coord) };
                // The map shows who ends the act from the moment the act begins.
                var boss = point.coord == map.BossMapPoint.coord ? _runState.Act?.BossEncounter
                    : point.coord == map.SecondBossMapPoint?.coord ? _runState.Act?.SecondBossEncounter : null;
                if (boss != null) node["encounter"] = boss.Id.ToString();
                snapshot.Entities.Add(node);
            }
            foreach (var point in points)
                foreach (var child in point.Children)
                    if (snapshot.Map.TryGetValue(child.coord, out var target))
                        snapshot.Relations.Add(new { source = snapshot.Map[point.coord], target, role = "map_edge" });
        }
        snapshot.Memory = Spire.PublicHistory.NativeHistory.Capture(_runState, obj => obj switch
        {
            CardModel card when snapshot.Cards.TryGetValue(card, out var reference) => reference,
            Creature creature when snapshot.Creatures.TryGetValue(creature, out var reference) => reference,
            _ => null,
        });
        var known = snapshot.Memory.Where(m => Equals(m["entity_type"], "known_draw_position"))
            .OrderBy(m => (int)m["position"]!).ToArray();
        for (int i = 1; i < known.Length; i++)
            snapshot.Relations.Add(new { source = known[i - 1]["owner_ref"], target = known[i]["owner_ref"], role = "known_draw_before" });
        return snapshot;
    }

    private static object OpaqueProgram(string content) => new
    {
        kind = "effect", op = "OPAQUE_RULE", content_id = content, coverage = "opaque",
    };

    private static Dictionary<string, string?>? _contentTitles;

    // The card, relic, potion or enchantment a displayed name stands for: the one
    // whose title it is. A title two contents share names neither of them.
    private static string? ContentNamed(string? title)
    {
        if (string.IsNullOrEmpty(title)) return null;
        if (_contentTitles == null)
        {
            var titles = new Dictionary<string, string?>(StringComparer.Ordinal);
            foreach (var model in ModelDb.All)
            {
                string? shown;
                try
                {
                    shown = model switch
                    {
                        CardModel card => card.Title, RelicModel relic => relic.Title.GetFormattedText(),
                        PotionModel potion => potion.Title.GetFormattedText(),
                        EnchantmentModel enchantment => enchantment.Title.GetFormattedText(), _ => null,
                    };
                }
                catch (Exception) { continue; }
                if (string.IsNullOrEmpty(shown)) continue;
                titles[shown] = titles.ContainsKey(shown) ? null : model.Id.ToString();
            }
            _contentTitles = titles;
        }
        return _contentTitles.GetValueOrDefault(title);
    }

    private static void AddDisplayedVariables(PublicSnapshot snapshot, string owner, params LocString?[] texts)
    {
        var emitted = new HashSet<string>();
        foreach (var text in texts.Where(t => t != null))
        {
            // LocString can carry variables not used on the current page. Export
            // only the variables actually referenced by its visible template: a
            // number as its amount, a name as the content it names. The game's own
            // table lookup is replaced here by one that returns the key, so the
            // template is read from the localization data loaded beside it.
            var template = _loc.En(text!.LocTable, text.LocEntryKey) ?? text.GetRawText();
            var names = Regex.Matches(template, @"\{([A-Za-z_][A-Za-z0-9_]*)")
                .Select(m => m.Groups[1].Value).ToHashSet(StringComparer.Ordinal);
            foreach (var (name, value) in text.Variables)
            {
                if (!names.Contains(name) || !emitted.Add(name)) continue;
                if (value is StringVar shown)
                {
                    // Other text (a description quoted in the option) names no content.
                    if (ContentNamed(shown.StringValue) is { } content)
                        snapshot.Entities.Add(new() { ["entity_type"] = "displayed_variable", ["owner_ref"] = owner,
                            ["content_id"] = name, ["names"] = content, ["known"] = true });
                    continue;
                }
                object? number = value is DynamicVar variable ? variable.BaseValue
                    : value is int or long or decimal or float or double ? value : null;
                if (number == null) continue;
                snapshot.Entities.Add(new() { ["entity_type"] = "displayed_variable", ["owner_ref"] = owner,
                    ["content_id"] = name, ["amount"] = number, ["known"] = true });
            }
        }
    }
}
