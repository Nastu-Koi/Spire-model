namespace Sts2Headless;

/// <summary>
/// Executes posted continuations immediately when idle, queuing nested posts.
/// Background selection continuations and the command thread may both call Pump;
/// only one drainer can execute callbacks at a time, without holding a lock across
/// a callback that may itself wait for external player input.
/// </summary>
internal sealed class InlineSynchronizationContext : SynchronizationContext
{
    private readonly object _gate = new();
    private readonly Queue<(SendOrPostCallback Callback, object? State)> _queue = new();
    private bool _executing;

    public override void Post(SendOrPostCallback d, object? state)
    {
        lock (_gate) _queue.Enqueue((d, state));
        Pump();
    }

    public override void Send(SendOrPostCallback d, object? state) => d(state);

    public void Pump()
    {
        lock (_gate)
        {
            if (_executing) return;
            _executing = true;
        }

        try
        {
            while (true)
            {
                (SendOrPostCallback Callback, object? State) work;
                lock (_gate)
                {
                    if (_queue.Count == 0)
                    {
                        _executing = false;
                        return;
                    }
                    work = _queue.Dequeue();
                }
                work.Callback(work.State);
            }
        }
        catch
        {
            lock (_gate) _executing = false;
            throw;
        }
    }
}
