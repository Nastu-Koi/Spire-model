using System;
using System.Text.Json;

namespace RunRecorder;

internal sealed class InputSnapshot
{
	public required string ActionType { get; init; }

	public required JsonElement State { get; init; }

	public required Origin Origin { get; init; }

	public DateTimeOffset CapturedUtc { get; } = DateTimeOffset.UtcNow;

	public InputSnapshot? Parent { get; init; }

	public bool Used { get; set; }
}
