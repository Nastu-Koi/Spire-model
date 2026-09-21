using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace RunRecorder;

// A Bootstrap label path reconstructed from the submitted order of a buffered native
// selection. This is not a claim to record mouse click order. The offer and bounds
// were frozen before selection; no intermediate game effect/reveal runs here.
internal static class SelectionTrace
{
    internal static object? Build(JsonElement state, JsonElement choice)
    {
        if (choice.ValueKind != JsonValueKind.Object
            || choice.GetProperty("command").GetString() != "select_cards"
            || !state.TryGetProperty("legal", out var legal)
            || legal.GetProperty("status").GetString() != "complete"
            || !legal.TryGetProperty("selection", out var selection)
            || selection.ValueKind != JsonValueKind.Object) return null;
        var min = selection.GetProperty("min").GetInt32();
        var max = selection.GetProperty("max").GetInt32();
        var cancelable = selection.GetProperty("cancelable").GetBoolean();
        var offered = legal.GetProperty("actions").EnumerateArray()
            .Where(a => a.GetProperty("command").GetString() == "select_card").ToArray();
        var ids = choice.GetProperty("args").GetProperty("card_instance_ids").EnumerateArray()
            .Select(v => v.GetInt64()).ToArray();
        var offeredIds = offered.Select(a => a.GetProperty("args").GetProperty("card_instance_id").GetInt64()).ToArray();
        if (min < 0 || max < min || offeredIds.Distinct().Count() != offeredIds.Length
            || ids.Distinct().Count() != ids.Length || ids.Any(id => !offeredIds.Contains(id))
            || ids.Length > max || (ids.Length < min && !(ids.Length == 0 && cancelable))) return null;
        object Control(string command) => new { command, args = new { } };
        var steps = new List<object>();
        var selected = new List<long>();
        foreach (var next in ids.Select(id => (long?)id).Append(null))
        {
            var actions = new List<object>();
            if (selected.Count < max)
                actions.AddRange(offered.Where(a => !selected.Contains(a.GetProperty("args").GetProperty("card_instance_id").GetInt64())).Cast<object>());
            if (selected.Count >= min) actions.Add(Control("finish_selection"));
            if (cancelable) actions.Add(Control("cancel"));
            object label = next.HasValue
                ? offered[Array.IndexOf(offeredIds, next.Value)]
                : Control(ids.Length < min ? "cancel" : "finish_selection");
            steps.Add(new { selected_ids = selected.ToArray(), actions, choice_key = label });
            if (next.HasValue) selected.Add(next.Value);
        }
        return new { mode = "buffered", reconstructed = true, order_source = "submitted_result", steps };
    }
}
