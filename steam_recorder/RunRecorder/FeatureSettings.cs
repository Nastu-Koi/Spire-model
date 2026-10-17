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

    // Which phases the model plays while control is on; the other is the player's.
    [JsonPropertyName("control_combat")]
    public bool ControlCombat { get; set; } = true;

    [JsonPropertyName("control_outside")]
    public bool ControlOutside { get; set; } = true;

    // Seconds the controller waits between two of its actions.
    [JsonPropertyName("step_interval_seconds")]
    public double StepIntervalSeconds { get; set; } = 0;

    [JsonPropertyName("panel_open")]
    public bool PanelOpen { get; set; } = false;

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
