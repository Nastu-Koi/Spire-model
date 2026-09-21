namespace Sts2Headless;

/// <summary>
/// A buffered, non-repeating selection over an engine-filtered, fixed option bank.
/// Only the command thread edits the prefix. No game effects run until Commit.
/// More elaborate constraints require a different provider, not guessed min/max rules.
/// </summary>
internal sealed class SelectionSession
{
    private readonly List<int> _selected = new();
    private readonly HashSet<int> _selectedSet = new();
    public string Id { get; } = Guid.NewGuid().ToString("N");
    public int OptionCount { get; }
    public int Min { get; }
    public int Max { get; }
    public bool OrderMatters { get; }
    public long Revision { get; private set; }
    public bool IsCommitted { get; private set; }
    public IReadOnlyList<int> Selected => _selected.AsReadOnly();
    public bool CanFinish => !IsCommitted && _selected.Count >= Min;

    // Collapse permutations only when the provider has established set semantics.
    public bool HasUniqueCompletion => !IsCommitted && !OrderMatters
        && Min - _selected.Count == OptionCount - _selected.Count;

    public SelectionSession(int optionCount, int min, int max, bool orderMatters = true)
    {
        if (optionCount < 0 || min < 0 || max < min || min > optionCount)
            throw new ArgumentException("Unsatisfiable selection bounds.");
        OptionCount = optionCount;
        Min = min;
        Max = max;
        OrderMatters = orderMatters;
    }

    public IEnumerable<int> LegalItems() => IsCommitted || _selected.Count >= Max || HasUniqueCompletion
        ? Enumerable.Empty<int>()
        : Enumerable.Range(0, OptionCount).Where(i => !_selectedSet.Contains(i));

    public void Select(int index)
    {
        if (!LegalItems().Contains(index))
            throw new ArgumentException("Item is not legal at this selection prefix.");
        _selected.Add(index);
        _selectedSet.Add(index);
        Revision++;
    }

    public void ValidateSubmission(IReadOnlyList<int> indices)
    {
        if (IsCommitted) throw new InvalidOperationException("Selection already committed.");
        if (indices.Count < Min || indices.Count > Max)
            throw new ArgumentException($"Selection requires {Min}..{Max} items.");
        if (indices.Any(i => i < 0 || i >= OptionCount) || indices.Distinct().Count() != indices.Count)
            throw new ArgumentException("Selection contains an invalid or repeated index.");
    }

    public int[] Commit()
    {
        var result = HasUniqueCompletion
            ? _selected.Concat(Enumerable.Range(0, OptionCount).Except(_selected)).ToArray()
            : _selected.ToArray();
        ValidateSubmission(result);
        IsCommitted = true;
        Revision++;
        return result;
    }

    public int[] CommitLegacy(int[] indices)
    {
        // Mixing an array submission with an already edited prefix would silently
        // replace the model's choices. Require the versioned FINISH instead.
        if (_selected.Count != 0)
            throw new InvalidOperationException("Finish the active selection prefix through execute_candidate.");
        ValidateSubmission(indices);
        IsCommitted = true;
        Revision++;
        return indices.ToArray();
    }
}
