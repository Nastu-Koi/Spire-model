using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;

namespace Sts2Headless;

public partial class RunSimulator
{
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
        string[] verbs = ["PLAY_CARD", "END_TURN", "DISCARD_POTION", "USE_POTION", "MOVE_TO_NODE",
            "CHOOSE_EVENT_OPTION", "CHOOSE_REST_OPTION", "LEAVE_ROOM", "BUY_ITEM", "OPEN_CHEST",
            "TAKE_TREASURE_RELIC", "TAKE_CARD_REWARD", "TAKE_REWARD", "CHOOSE_REWARD_ALTERNATIVE",
            "LEAVE_REWARDS", "SELECT_ONE", "FINISH_SELECTION", "CANCEL", "SELECT_BUNDLE", "DIVINE_CELL", "ABANDON_RUN"];
        return new()
        {
            ["type"] = "public_catalog", ["contract"] = CurrentProtocolContract,
            ["public"] = new { phase = "catalog", entities },
            ["legal"] = new { candidates = verbs.Select(v => Candidate(v, v)).ToArray() },
        };
    }
}
