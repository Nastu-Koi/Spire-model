using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Runs;

namespace RunRecorder;

internal sealed class Pending
{
	public string Id { get; } = Guid.NewGuid().ToString("N");

	public required RunState Run { get; init; }

	public required string Command { get; init; }

	public required Origin Origin { get; init; }

	public required string Domain { get; init; }

	public required int Floor { get; init; }

	public required long Order { get; init; }

	public DateTimeOffset StartedUtc { get; set; } = DateTimeOffset.UtcNow;

	public JsonElement State { get; set; }

	public JsonElement Action { get; set; }

	public JsonElement ChoiceKey { get; set; }

	public InputSnapshot? Input { get; set; }

    public DateTimeOffset? SelectionCapturedUtc { get; set; }

	public JsonElement ExecutionState { get; set; }

	public bool NativeNoOp { get; set; }

	public bool ProvisionalChoice { get; set; }

	public Task? Task { get; set; }

	public Pending? Parent { get; set; }

	public object? OnceKey { get; set; }

	public List<JsonElement> Choices { get; } = new List<JsonElement>();

	public bool Finished { get; set; }

	public bool Started { get; set; }
}
