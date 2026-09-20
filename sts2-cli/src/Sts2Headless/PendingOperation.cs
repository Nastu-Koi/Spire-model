using System.Diagnostics;

namespace Sts2Headless;

/// <summary>
/// Owns an operation across external selections. A prompt or task completion wakes
/// the command thread; elapsed time is never used as evidence of completion.
/// Start and WaitForBoundary are called only by the command thread.
/// </summary>
internal sealed class PendingOperation
{
    private readonly AutoResetEvent _changed = new(false);
    private Task? _task;
    private Action? _onCompleted;
    private string _description = "";

    public bool IsActive => _task != null;

    public void NotifyProgress() => _changed.Set();

    public void Start(string description, Func<Task> operation, Action? onCompleted = null)
    {
        if (_task != null)
            throw new InvalidOperationException($"Operation '{_description}' is still pending.");

        _description = description;
        _onCompleted = onCompleted;
        _task = Task.Run(operation);
        _ = _task.ContinueWith(_ => NotifyProgress(), CancellationToken.None,
            TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
    }

    public void WaitForBoundary(Action pump, Func<bool> hasSelection, TimeSpan timeout)
    {
        if (_task == null) return;
        var timer = Stopwatch.StartNew();
        while (true)
        {
            pump();
            // Faults remain attached to this operation, preventing further mutations.
            if (_task.IsFaulted || _task.IsCanceled)
                _task.GetAwaiter().GetResult();
            if (hasSelection()) return;
            if (_task.IsCompleted)
            {
                _task.GetAwaiter().GetResult();
                var onCompleted = _onCompleted;
                _task = null;
                _onCompleted = null;
                onCompleted?.Invoke();
                return;
            }

            var remaining = timeout - timer.Elapsed;
            if (remaining <= TimeSpan.Zero)
                throw new TimeoutException($"Operation '{_description}' did not reach a decision boundary within {timeout.TotalSeconds:g} seconds.");
            // AutoResetEvent retains a notification arriving between the checks and wait.
            _changed.WaitOne(remaining);
        }
    }
}
