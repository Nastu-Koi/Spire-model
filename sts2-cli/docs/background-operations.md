# Background operation completion

Rest-site options, shop card removal, shop relic purchases and event options run
through `PendingOperation`. The command thread waits for a selection notification
or task completion, instead of sleeping for a fixed 50/200 ms or polling these
operations every 10 ms.

When a card, reward or bundle prompt appears, the original operation remains
owned by the simulator. Resolving it resumes the operation until the next prompt
or completion. A rest site only transitions to the map after its option task
finishes, including all nested selections. Timeouts and faults are reported as
errors; they do not imply success or authorize a replacement operation.

Selectors clear the old request before resuming its continuation. Reward choices
belong to individual requests, and asynchronous continuations prevent the command
thread from blocking in the next reward prompt. Posted synchronization callbacks
use a single drainer with a protected queue; callbacks run outside the queue lock.

The three-second operation timeout applies separately to each advancement, not to
time the player spends choosing. The existing five-minute reward-selection limit
reports a timeout instead of silently skipping. Combat execution and end-turn
recovery retain their existing conditional waits; this change does not disable
all engine synchronization or implement a new run-reset lifecycle.

## Validation

The operation and callback checks run without game assemblies or test packages:

```bash
dotnet run --project tests/PendingOperationChecks/PendingOperationChecks.csproj -c Release
```

With the normal local game dependencies and a Debug build available:

```bash
dotnet build src/Sts2Headless/Sts2Headless.csproj
python3 -m pytest tests/test_pending_operations.py tests/test_rest_site.py tests/test_shop.py tests/test_event.py tests/test_card_reward.py
```

The checks cover nested boundaries, repeated completion, timeout recovery, task
faults, concurrent callbacks, completed upgrades, exactly-once removal costs and
three consecutive event rewards (including skipping the middle reward). Run the
five-character full-run regression described in `CLAUDE.md` after core changes.
