using System;
using System.Collections.Generic;
using System.Linq;

namespace Spire.PublicHistory;

// This module accepts public events, never a save's odds or a shuffled card list.
// The two native adapters compile the same source; null means an unobserved prefix.
public sealed class OddsHistory
{
    public const string Version = "public-history-v1";
    public static readonly string[] RoomTypes = { "Monster", "Elite", "Treasure", "Shop", "Event" };
    private readonly Dictionary<string, float> _bases;
    private readonly Dictionary<string, float?> _rooms;
    private readonly bool _deadlyEvents;
    private float? _potion;
    private int? _tutorialRolls;
    private int _observedTutorialRolls;
    private readonly bool _tutorial;
    public int Act { get; private set; }

    public OddsHistory(bool fresh, int act, bool tutorial = false, bool deadlyEvents = false)
    {
        Act = act;
        _tutorial = tutorial;
        _tutorialRolls = fresh ? 0 : null;
        _deadlyEvents = deadlyEvents;
        _bases = new() { ["Monster"] = .1f, ["Elite"] = deadlyEvents ? .1f : -1f,
            ["Treasure"] = .02f, ["Shop"] = .03f };
        _rooms = _bases.ToDictionary(p => p.Key, p => fresh ? (float?)p.Value : null);
        _potion = fresh ? .4f : null;
    }

    public void EnterAct(int act)
    {
        // Loading a save re-enters the SAME act. That is not a public reset.
        if (act == Act) return;
        Act = act;
        foreach (var pair in _bases) _rooms[pair.Key] = pair.Value;
    }

    public float? PotionProbability(bool elite, bool forced) => forced ? 1f
        : _potion is float value ? Math.Clamp(value + (elite ? .125f : 0f), 0f, 1f) : null;

    public void PotionGenerated(bool dropped, bool forced)
    {
        // Roll changes its accumulator at generation, including a full potion belt.
        // Do not clamp the accumulator: only the displayed probability is clamped.
        if (!forced && _potion.HasValue) _potion += dropped ? -.1f : .1f;
    }

    public Dictionary<string, float?> RoomProbabilities(IReadOnlySet<string> eligible)
    {
        var result = RoomTypes.ToDictionary(t => t, _ => (float?)0f);
        if (_tutorial && _tutorialRolls is int n && n < 3)
        {
            result[n < 2 ? "Event" : "Monster"] = 1f;
            return result;
        }
        if (_tutorial && !_tutorialRolls.HasValue)
            return RoomTypes.ToDictionary(t => t, _ => (float?)null);
        if (eligible.Count == 1)
        {
            result[eligible.Single()] = 1f;
            return result;
        }
        // Native accumulation order matters when the sum exceeds one. Disabled
        // negative odds contribute zero, and excluded rooms retain their accumulator.
        float? remaining = 1f;
        foreach (var type in RoomTypes.Where(t => t != "Event"))
        {
            if (!eligible.Contains(type)) continue;
            if (remaining == 0 || _rooms[type] <= 0) continue;
            if (!_rooms[type].HasValue || !remaining.HasValue)
            {
                result[type] = null;
                remaining = null;
                continue;
            }
            float mass = Math.Min(remaining.Value, _rooms[type]!.Value);
            result[type] = mass;
            remaining -= mass;
        }
        // All supported public rules retain Event as the fallback.
        if (eligible.Contains("Event")) result["Event"] = remaining;
        else throw new InvalidOperationException("Unsupported public room fallback");
        return result;
    }

    public void QuestionRoomGenerated(string result, IReadOnlySet<string> eligible)
    {
        if (_tutorial && (!_tutorialRolls.HasValue || _tutorialRolls < 3))
        {
            if (_tutorialRolls.HasValue) _tutorialRolls++;
            else if (++_observedTutorialRolls >= 3) _tutorialRolls = 3;
            return;
        }
        foreach (var pair in _bases)
        {
            if (result == pair.Key) _rooms[pair.Key] = pair.Value;
            else if (eligible.Contains(pair.Key) && _rooms[pair.Key].HasValue)
                _rooms[pair.Key] += pair.Value * (_deadlyEvents && pair.Key == "Treasure" ? 2 : 1);
        }
    }
}

public enum Placement { Unknown, Top, Bottom }

public sealed class DrawKnowledge<T> where T : class
{
    private List<T?> _slots = new();
    public void Invalidate(int count) => _slots = Enumerable.Repeat<T?>(null, count).ToList();
    public void SynchronizeCount(int count) { if (_slots.Count != count) Invalidate(count); }

    public void Added(T card, Placement placement, int countBefore)
    {
        SynchronizeCount(countBefore);
        switch (placement)
        {
            case Placement.Top: _slots.Insert(0, card); break;
            case Placement.Bottom: _slots.Add(card); break;
            default: Invalidate(countBefore + 1); break;
        }
    }

    public void Removed(T card, int countBefore)
    {
        SynchronizeCount(countBefore);
        int known = _slots.FindIndex(c => ReferenceEquals(c, card));
        if (known >= 0) { _slots.RemoveAt(known); return; }
        int first = _slots.FindIndex(c => c == null), last = _slots.FindLastIndex(c => c == null);
        if (first < 0) { Invalidate(Math.Max(0, countBefore - 1)); return; }
        // The revealed removed card could occupy ANY unknown slot. Keep only
        // positions shared by every possibility; never consult its actual index.
        _slots = Enumerable.Range(0, countBefore - 1)
            .Select(i => i < first ? _slots[i] : i >= last ? _slots[i + 1] : null).ToList();
    }

    public IEnumerable<(T Card, int Position)> KnownPositions(int count)
    {
        SynchronizeCount(count);
        return _slots.Select((card, i) => (card, i)).Where(p => p.card != null)
            .Select(p => (p.card!, p.i)).ToArray();
    }
}
