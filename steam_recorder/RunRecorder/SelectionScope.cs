using System;
using System.Text.Json;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.Players;

namespace RunRecorder;

internal sealed class SelectionScope
{
	public required Player Player { get; init; }

	public required string Source { get; init; }

	public SelectionScope? Parent { get; init; }

	public SelectionOffer? Offer { get; set; }

	public JsonElement State { get; set; }

    public DateTimeOffset CapturedUtc { get; set; }

	public Task? Task { get; set; }
}
