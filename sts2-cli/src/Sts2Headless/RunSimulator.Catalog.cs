using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;

namespace Sts2Headless;

public partial class RunSimulator
{
    // The game assembly also holds Godot node types which the headless stubs cannot load.
    private static IEnumerable<Type> LoadableTypes(System.Reflection.Assembly assembly)
    {
        try { return assembly.GetTypes(); }
        catch (System.Reflection.ReflectionTypeLoadException e) { return e.Types.Where(t => t != null)!; }
    }

    public Dictionary<string, object?> PublicCatalog()
    {
        EnsureModelDbInitialized();
        // Static content names only. This does not visit rooms, generate offers,
        // inspect a run RNG, or expose which content will occur in a particular run.
        var entities = ModelDb.All.OrderBy(m => m.Id.ToString(), StringComparer.Ordinal)
            .Select(m => new Dictionary<string, object?> { ["entity_type"] = m.Id.Category.ToLowerInvariant(),
                ["content_id"] = m.Id.ToString(), ["semantic_program"] = OpaqueProgram(m.Id.ToString()) }).ToList();
        foreach (var table in new[] { "events", "ancients" })
            foreach (var text in LocManager.Instance.GetTable(table).GetLocStringsWithPrefix(""))
                if (text.LocEntryKey.Contains(".options.") && text.LocEntryKey.EndsWith(".title"))
                    entities.Add(new() { ["entity_type"] = "event_option", ["content_id"] = text.LocEntryKey[..^6],
                        ["semantic_program"] = OpaqueProgram(text.LocEntryKey[..^6]) });
        // Ancient relic choices synthesize their titles from relics rather than
        // storing option-title localization entries. Enumerate the static pool,
        // never GenerateInitialOptions (which samples the current run's offers).
        foreach (var ancient in ModelDb.All.OfType<AncientEventModel>())
            foreach (var option in ancient.AllPossibleOptions)
                entities.Add(new() { ["entity_type"] = "event_option", ["content_id"] = option.TextKey,
                    ["semantic_program"] = OpaqueProgram(option.TextKey) });
        foreach (var text in LocManager.Instance.GetTable("rest_site_ui").GetLocStringsWithPrefix("OPTION_"))
            if (text.LocEntryKey.EndsWith(".name", StringComparison.Ordinal))
            {
                var id = text.LocEntryKey[7..^5];
                entities.Add(new() { ["entity_type"] = "rest_option", ["content_id"] = id,
                    ["semantic_program"] = OpaqueProgram(id) });
            }
        // Rewards and shop entries without a model of their own are named by their
        // type; card keywords are a closed set. Both are static content names.
        foreach (var (kind, root) in new[] { ("reward", typeof(Reward)), ("shop_item", typeof(MerchantEntry)) })
            foreach (var type in LoadableTypes(root.Assembly).Where(t => !t.IsAbstract && root.IsAssignableFrom(t))
                         .OrderBy(t => t.Name, StringComparer.Ordinal))
                entities.Add(new() { ["entity_type"] = kind, ["content_id"] = type.Name,
                    ["semantic_program"] = OpaqueProgram(type.Name) });
        entities.Add(new() { ["entity_type"] = "card",
            ["keywords"] = Enum.GetNames<CardKeyword>().Order().ToArray() });
        string[] verbs = ["PLAY_CARD", "END_TURN", "DISCARD_POTION", "USE_POTION", "MOVE_TO_NODE",
            "CHOOSE_EVENT_OPTION", "CHOOSE_REST_OPTION", "LEAVE_ROOM", "BUY_ITEM", "OPEN_CHEST",
            "TAKE_TREASURE_RELIC", "TAKE_CARD_REWARD", "TAKE_REWARD", "CHOOSE_REWARD_ALTERNATIVE",
            "LEAVE_REWARDS", "SELECT_ONE", "FINISH_SELECTION", "CANCEL", "SELECT_BUNDLE", "DIVINE_CELL", "ABANDON_RUN"];
        return new()
        {
            ["type"] = "public_catalog", ["contract"] = CurrentProtocolContract,
            ["vocabulary_catalog_version"] = 1,
            ["public"] = new { phase = "catalog", entities },
            ["legal"] = new { candidates = verbs.Select(v => Candidate(v, v)).ToArray() },
        };
    }
}
