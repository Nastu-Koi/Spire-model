using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace RunRecorder;

internal sealed class FeatureSettings
{
    [JsonPropertyName("recording_enabled")]
    public bool RecordingEnabled { get; set; } = true;

    [JsonPropertyName("control_enabled")]
    public bool ControlEnabled { get; set; } = true;

    internal static FeatureSettings Load(string path) => File.Exists(path)
        ? JsonSerializer.Deserialize<FeatureSettings>(File.ReadAllText(path))
            ?? throw new JsonException("RunRecorder settings must be a JSON object")
        : new FeatureSettings();

    internal void Save(string path)
    {
        File.WriteAllText(path + ".tmp", JsonSerializer.Serialize(this,
            new JsonSerializerOptions { WriteIndented = true }));
        File.Move(path + ".tmp", path, true);
    }
}
