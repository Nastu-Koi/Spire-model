namespace Sts2Headless;

internal sealed record CandidateBinding(
    Dictionary<string, object?> Public,
    Func<bool> IsLegal,
    Action Execute);

/// <summary>
/// Serial command-thread gate. Validation does not pump the engine or mutate a
/// prefix; an accepted handle is consumed before invoking its continuation.
/// </summary>
internal sealed class DecisionGate
{
    private Dictionary<string, CandidateBinding> _bindings = new();
    public long StateVersion { get; private set; }
    public string? DecisionId { get; private set; }
    public long? SelectionRevision { get; private set; }
    public Dictionary<string, object?>? Frame { get; private set; }

    public void Invalidate()
    {
        StateVersion++;
        DecisionId = null;
        SelectionRevision = null;
        Frame = null;
        _bindings.Clear();
    }

    public void Publish(Dictionary<string, object?> frame, string decisionId,
        long? selectionRevision, IReadOnlyList<CandidateBinding> bindings)
    {
        if (Frame != null) throw new InvalidOperationException("Invalidate before publishing a new frame.");
        DecisionId = decisionId;
        SelectionRevision = selectionRevision;
        _bindings = bindings.Select((b, i) => (b, key: $"c{i}"))
            .ToDictionary(pair => pair.key, pair => pair.b);
        Frame = frame;
    }

    public string? Validate(string decisionId, long stateVersion, string candidateRef, long? revision)
    {
        if (Frame == null || DecisionId != decisionId || StateVersion != stateVersion)
            return "stale_decision";
        if (SelectionRevision != revision) return "stale_selection";
        if (!_bindings.TryGetValue(candidateRef, out var binding)) return "unknown_candidate";
        if (!binding.IsLegal()) return "candidate_no_longer_legal";
        return null;
    }

    public Dictionary<string, object?> Consume(string candidateRef)
    {
        var binding = _bindings[candidateRef];
        Invalidate();
        binding.Execute();
        return binding.Public;
    }
}
