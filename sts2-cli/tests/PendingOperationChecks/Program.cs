using Sts2Headless;

static void Check(bool condition, string message)
{
    if (!condition) throw new Exception(message);
}
static void Throws<T>(Action action) where T : Exception
{
    try { action(); }
    catch (T) { return; }
    throw new Exception($"Expected {typeof(T).Name}");
}
static TaskCompletionSource Gate() => new(TaskCreationOptions.RunContinuationsAsynchronously);
var timeout = TimeSpan.FromSeconds(5);

// A prompt is a boundary, not task completion. A second prompt must survive,
// and the completion callback must run exactly once after both are resolved.
var operation = new PendingOperation();
var first = Gate();
var second = Gate();
int prompt = 0, completions = 0;
operation.Start("two selections", async () =>
{
    Volatile.Write(ref prompt, 1);
    operation.NotifyProgress();
    await first.Task;
    Volatile.Write(ref prompt, 2);
    operation.NotifyProgress();
    await second.Task;
}, () => completions++);
operation.WaitForBoundary(() => { }, () => Volatile.Read(ref prompt) != 0, timeout);
Check(prompt == 1 && operation.IsActive && completions == 0, "First selection was skipped");
Throws<InvalidOperationException>(() => operation.Start("duplicate", () => Task.CompletedTask));
Volatile.Write(ref prompt, 0);
first.SetResult();
operation.WaitForBoundary(() => { }, () => Volatile.Read(ref prompt) != 0, timeout);
Check(prompt == 2 && completions == 0, "Second selection was skipped");
Volatile.Write(ref prompt, 0);
second.SetResult();
operation.WaitForBoundary(() => { }, () => Volatile.Read(ref prompt) != 0, timeout);
operation.WaitForBoundary(() => { }, () => false, timeout);
Check(!operation.IsActive && completions == 1, "Completion callback was lost or repeated");

// Notifications published before the command thread waits must not be lost.
for (int i = 0; i < 100; i++)
{
    operation.Start("immediate", () => Task.CompletedTask);
    operation.WaitForBoundary(() => { }, () => false, timeout);
    Check(!operation.IsActive, "Immediate completion lost");
}

// A timeout must retain ownership: no forced success, cancellation or new task.
var blocked = new PendingOperation();
var release = Gate();
blocked.Start("blocked", () => release.Task);
Throws<TimeoutException>(() => blocked.WaitForBoundary(() => { }, () => false, TimeSpan.FromMilliseconds(20)));
Check(blocked.IsActive, "Timeout discarded the running operation");
Throws<InvalidOperationException>(() => blocked.Start("replacement", () => Task.CompletedTask));
release.SetResult();
blocked.WaitForBoundary(() => { }, () => false, timeout);
Check(!blocked.IsActive, "Timed out operation could not finish later");

// Faults must stay visible and must not invoke a successful completion callback.
var failed = new PendingOperation();
failed.Start("failure", () => Task.FromException(new InvalidOperationException("injected")), () => completions++);
Throws<InvalidOperationException>(() => failed.WaitForBoundary(() => { }, () => false, timeout));
Throws<InvalidOperationException>(() => failed.WaitForBoundary(() => { }, () => false, timeout));
Check(failed.IsActive && completions == 1, "Fault was hidden or finalized as success");

// Concurrent Pump/Post callers must not race Queue<T> or execute two callbacks
// at once. Nested posts must still drain without recursion or a held callback lock.
var context = new InlineSynchronizationContext();
using var drained = new CountdownEvent(1600);
int executing = 0, overlap = 0;
var producers = Enumerable.Range(0, 8).Select(_ => Task.Run(() =>
{
    for (int i = 0; i < 100; i++)
    {
        context.Post(_ =>
        {
            if (Interlocked.Increment(ref executing) != 1) Interlocked.Increment(ref overlap);
            context.Post(_ => drained.Signal(), null);
            Interlocked.Decrement(ref executing);
            drained.Signal();
        }, null);
        context.Pump();
    }
})).ToArray();
Task.WaitAll(producers);
Check(drained.Wait(timeout), "Posted callbacks were lost");
Check(overlap == 0, "Posted callbacks executed concurrently");
Console.WriteLine("Checks passed: nested boundaries, duplicate protection, 100 immediate completions, timeout recovery, persistent faults, 1600 concurrent/nested callbacks.");
