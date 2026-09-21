using System;
using System.IO;
using System.Text;
using System.Text.Json;

namespace RunRecorder;

internal sealed class Journal : IDisposable
{
	internal static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions
	{
		PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
		IncludeFields = true
	};

	private readonly StreamWriter _writer;

	private long _sequence;

	public string RunId { get; }

	public string SegmentId { get; } = Guid.NewGuid().ToString("N");

	public string PathOnDisk { get; }

	public long Count => _sequence;

	public Journal(string directory, string runId)
	{
		Directory.CreateDirectory(directory);
		RunId = runId;
		PathOnDisk = Path.Combine(directory, $"{DateTime.UtcNow:yyyyMMddTHHmmssfffZ}_{runId}_{SegmentId}.jsonl");
		_writer = new StreamWriter(new FileStream(PathOnDisk, FileMode.CreateNew, FileAccess.Write, FileShare.Read, 65536), new UTF8Encoding(encoderShouldEmitUTF8Identifier: false))
		{
			AutoFlush = true
		};
	}

	public void Write(string kind, object? data)
	{
		string value = JsonSerializer.Serialize(new
		{
			schema = "sts2-run-recorder/v2",
			run_id = RunId,
			segment_id = SegmentId,
			seq = _sequence + 1,
			utc = DateTimeOffset.UtcNow,
			kind = kind,
			data = data
		}, JsonOptions);
		_writer.WriteLine(value);
		_sequence++;
	}

	public static JsonElement Freeze(object? value)
	{
		return JsonSerializer.SerializeToElement(value, JsonOptions);
	}

	public void Dispose()
	{
		_writer.Dispose();
	}
}
