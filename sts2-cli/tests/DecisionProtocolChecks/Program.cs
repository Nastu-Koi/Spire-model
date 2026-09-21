using Sts2Headless;

static void Check(bool condition, string message)
{
    if (!condition) throw new Exception(message);
}

static void Reject(Action action)
{
    try { action(); }
    catch (ArgumentException) { return; }
    catch (InvalidOperationException) { return; }
    throw new Exception("Invalid mutation was accepted");
}

// Exhaust every ordered 10-choose-5 trajectory. No subset or permutation may
// disappear from the prefix tree, and STOP must be absent before the fifth item.
var subsets = new HashSet<string>();
var trajectories = 0;
void Walk(List<int> path)
{
    var session = new SelectionSession(10, 5, 5);
    foreach (var index in path) session.Select(index);
    Check(session.CanFinish == (path.Count == 5), "Premature or missing STOP");
    Check(session.LegalItems().Count() == (path.Count == 5 ? 0 : 10 - path.Count), "Candidate lost");
    if (path.Count == 5)
    {
        var committed = session.Commit();
        Check(committed.SequenceEqual(path), "Selection order changed");
        Reject(() => session.Commit());
        subsets.Add(string.Join(",", committed.Order()));
        trajectories++;
        return;
    }
    foreach (var item in session.LegalItems()) Walk(path.Append(item).ToList());
}
Walk(new());
Check(trajectories == 30240 && subsets.Count == 252, "10 choose 5 coverage mismatch");

foreach (var min in new[] { 0, 1 })
foreach (var count in Enumerable.Range(min, 4 - min))
{
    var session = new SelectionSession(7, min, 3);
    for (var i = 0; i < count; i++) session.Select(i);
    Check(session.Commit().SequenceEqual(Enumerable.Range(0, count)), "STOP lost or added cards");
}

var optionalOne = new SelectionSession(1, 0, 1);
Check(optionalOne.CanFinish && optionalOne.LegalItems().Count() == 1, "Optional card must compete with STOP");
var all = new SelectionSession(5, 5, 5, orderMatters: false);
Check(all.HasUniqueCompletion && !all.LegalItems().Any() && all.Commit().Length == 5, "Forced set was not collapsed");
var ordered = new SelectionSession(5, 5, 5, orderMatters: true);
Check(!ordered.HasUniqueCompletion && ordered.LegalItems().Count() == 5, "Ordered choice was collapsed");
Check(new SelectionSession(0, 0, 3).Commit().Length == 0, "Legal empty result rejected");
Reject(() => new SelectionSession(2, 3, 3));
Reject(() => new SelectionSession(3, 2, 1));

var guarded = new SelectionSession(4, 1, 3);
foreach (var invalid in new[] { Array.Empty<int>(), new[] { -1 }, new[] { 4 }, new[] { 0, 0 }, new[] { 0, 1, 2, 3 } })
{
    Reject(() => guarded.CommitLegacy(invalid));
    Check(!guarded.IsCommitted && guarded.Revision == 0, "Rejected input changed selection");
}
guarded.Select(1);
Reject(() => guarded.Select(1));
Reject(() => guarded.CommitLegacy(new[] { 2 }));
Check(guarded.Revision == 1 && guarded.Selected.SequenceEqual(new[] { 1 }), "Invalid prefix mutation");
Check(guarded.Commit().SequenceEqual(new[] { 1 }), "Prefix was not preserved");

var gate = new DecisionGate();
var executed = 0;
var stillLegal = true;
gate.Publish(new(), "decision-1", 0, new[] {
    new CandidateBinding(new() { ["verb"] = "SELECT_ONE" }, () => stillLegal, () => executed++)
});
Check(gate.Validate("old", 0, "c0", 0) == "stale_decision", "Old ID accepted");
Check(gate.Validate("decision-1", 1, "c0", 0) == "stale_decision", "Old state accepted");
Check(gate.Validate("decision-1", 0, "c0", null) == "stale_selection", "Missing prefix accepted");
Check(gate.Validate("decision-1", 0, "other", 0) == "unknown_candidate", "Unknown handle accepted");
stillLegal = false;
Check(gate.Validate("decision-1", 0, "c0", 0) == "candidate_no_longer_legal", "Live legality not checked");
Check(executed == 0 && gate.StateVersion == 0, "Validation changed state");
stillLegal = true;
Check(gate.Validate("decision-1", 0, "c0", 0) == null, "Valid command rejected");
gate.Consume("c0");
Check(executed == 1 && gate.Validate("decision-1", 0, "c0", 0) == "stale_decision", "Replay accepted");
gate.Publish(new(), "decision-2", null, new[] {
    new CandidateBinding(new(), () => true, () => throw new InvalidOperationException("engine failure"))
});
Reject(() => gate.Consume("c0"));
Check(gate.Frame == null && gate.StateVersion == 2, "Faulting handle may be replayed");
Console.WriteLine($"PASS: {trajectories} trajectories, {subsets.Count} subsets; STOP, atomic validation, replay and failure guards.");
