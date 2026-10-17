using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Nodes;

namespace RunRecorder;

// The one place the player meets the mod: the switches (recording, control, which
// phases the model plays), the step interval, what the bridge is doing, the error it
// stopped on with the reset, the current decision and the recent ones. A small
// button in the corner folds it away.
internal sealed class LivePanel
{
    private const int HistoryShown = 20;
    private readonly FeatureSettings _settings;
    private readonly Action _apply;
    private readonly Button _toggle;
    private readonly PanelContainer _panel;
    private readonly CheckButton _recording, _control, _combat, _outside;
    private readonly SpinBox _interval;
    private readonly Label _status, _notice, _error, _current;
    private readonly Button _reset;
    private readonly ItemList _history;
    private string _shownHistory = "";
    private bool _binding;

    private LivePanel(FeatureSettings settings, Action apply)
    {
        _settings = settings;
        _apply = apply;
        _toggle = new Button { Text = Caption(settings.PanelOpen), FocusMode = Control.FocusModeEnum.None };
        _toggle.AddThemeFontSizeOverride("font_size", 16);
        _toggle.Pressed += () => { _settings.PanelOpen = !_settings.PanelOpen; Show(_settings.PanelOpen); _apply(); };

        _panel = new PanelContainer { CustomMinimumSize = new Vector2(460, 0), MouseFilter = Control.MouseFilterEnum.Stop };
        var margin = new MarginContainer();
        foreach (var side in new[] { "margin_left", "margin_right", "margin_top", "margin_bottom" })
            margin.AddThemeConstantOverride(side, 10);
        var column = new VBoxContainer();
        column.AddThemeConstantOverride("separation", 6);
        margin.AddChild(column);
        _panel.AddChild(margin);

        column.AddChild(Heading("模型接管"));
        var switches = new GridContainer { Columns = 2 };
        _recording = Switch(switches, "录制数据", "记录状态和动作供训练使用；关闭后模型仍可接管。",
            settings.RecordingEnabled, value => _settings.RecordingEnabled = value);
        _control = Switch(switches, "允许模型接管", "关闭后桥接只回答“已暂停”，可随时手动操作。",
            settings.ControlEnabled, value => _settings.ControlEnabled = value);
        _combat = Switch(switches, "接管战斗", "关闭后战斗中的出牌、用药由你操作，模型在战斗结束后继续。",
            settings.ControlCombat, value => _settings.ControlCombat = value);
        _outside = Switch(switches, "接管局外", "关闭后地图、事件、商店、营火、奖励由你操作，模型只打战斗。",
            settings.ControlOutside, value => _settings.ControlOutside = value);
        column.AddChild(switches);

        var pace = new HBoxContainer();
        pace.AddChild(new Label { Text = "每步间隔（秒）" });
        _interval = new SpinBox { MinValue = 0, MaxValue = 30, Step = 0.5, Value = settings.StepIntervalSeconds,
                                  CustomMinimumSize = new Vector2(110, 0) };
        _interval.ValueChanged += value => { if (_binding) return; _settings.StepIntervalSeconds = value; _apply(); };
        pace.AddChild(_interval);
        column.AddChild(pace);

        _status = Text(column, 15);
        _notice = Text(column, 14);
        _notice.Modulate = new Color(1f, .85f, .4f);
        _error = Text(column, 14);
        _error.Modulate = new Color(1f, .55f, .5f);
        _reset = new Button { Text = "重置并重新读取状态", FocusMode = Control.FocusModeEnum.None,
                              TooltipText = "清掉模型停下时的错误，让它从当前画面重新观察并继续。" };
        _reset.Pressed += () => { LiveBridge.ResetByUser(); Refresh(); };
        column.AddChild(_reset);

        column.AddChild(Heading("当前决策"));
        _current = Text(column, 15);
        column.AddChild(Heading("最近决策"));
        _history = new ItemList { CustomMinimumSize = new Vector2(0, 230), SelectMode = ItemList.SelectModeEnum.Single,
                                  FocusMode = Control.FocusModeEnum.None, AutoHeight = false };
        _history.AddThemeFontSizeOverride("font_size", 13);
        column.AddChild(_history);
        Show(settings.PanelOpen);
    }

    internal static LivePanel Create(Node parent, FeatureSettings settings, Action apply)
    {
        var panel = new LivePanel(settings, apply);
        var layer = new CanvasLayer { Name = "RunRecorderPanel", Layer = 999 };
        var root = new Control { MouseFilter = Control.MouseFilterEnum.Ignore };
        root.SetAnchorsPreset(Control.LayoutPreset.FullRect);
        panel._toggle.SetAnchorsPreset(Control.LayoutPreset.BottomLeft);
        panel._toggle.OffsetLeft = 18f; panel._toggle.OffsetRight = 118f;
        panel._toggle.OffsetTop = -44f; panel._toggle.OffsetBottom = -12f;
        panel._panel.SetAnchorsPreset(Control.LayoutPreset.BottomLeft);
        panel._panel.GrowVertical = Control.GrowDirection.Begin;
        panel._panel.OffsetLeft = 18f; panel._panel.OffsetRight = 478f;
        panel._panel.OffsetBottom = -52f; panel._panel.OffsetTop = -52f;
        root.AddChild(panel._panel);
        root.AddChild(panel._toggle);
        layer.AddChild(root);
        parent.AddChild(layer, forceReadableName: false, Node.InternalMode.Disabled);
        return panel;
    }

    private static string Caption(bool open) => open ? "模型 ▾" : "模型 ▸";

    private void Show(bool open)
    {
        _panel.Visible = open;
        _toggle.Text = Caption(open);
    }

    private static Label Heading(string text)
    {
        var label = new Label { Text = text };
        label.AddThemeFontSizeOverride("font_size", 17);
        return label;
    }

    private static Label Text(VBoxContainer column, int size)
    {
        var label = new Label { AutowrapMode = TextServer.AutowrapMode.WordSmart, MouseFilter = Control.MouseFilterEnum.Ignore };
        label.AddThemeFontSizeOverride("font_size", size);
        column.AddChild(label);
        return label;
    }

    private CheckButton Switch(Control parent, string text, string tooltip, bool value, Action<bool> set)
    {
        var button = new CheckButton { Text = text, ButtonPressed = value, TooltipText = tooltip, FocusMode = Control.FocusModeEnum.None };
        button.AddThemeFontSizeOverride("font_size", 15);
        button.Toggled += pressed => { if (_binding) return; set(pressed); _apply(); };
        parent.AddChild(button);
        return button;
    }

    // Called by the mod's timer: what the bridge knows, onto the panel. Never throws:
    // a panel that cannot draw must not stop the recorder or the bridge.
    internal void Refresh()
    {
        try
        {
            _toggle.Text = LiveBridge.HaltError != null && !_panel.Visible ? "模型 ▸ 已停止" : Caption(_panel.Visible);
            if (!_panel.Visible) return;
            _binding = true;
            try
            {
                _recording.ButtonPressed = _settings.RecordingEnabled;
                _control.ButtonPressed = _settings.ControlEnabled;
                _combat.ButtonPressed = _settings.ControlCombat;
                _outside.ButtonPressed = _settings.ControlOutside;
                if (Math.Abs(_interval.Value - _settings.StepIntervalSeconds) > 1e-9) _interval.Value = _settings.StepIntervalSeconds;
            }
            finally { _binding = false; }
            _status.Text = Recorder.Status + "\n" + LiveBridge.Status;
            _notice.Text = LiveBridge.Notice ?? "";
            _notice.Visible = LiveBridge.Notice != null;
            string? error = LiveBridge.HaltError;
            _error.Text = error == null ? "" : $"模型已停止（{LiveBridge.HaltedAt.ToLocalTime():HH:mm:ss}）：{error}";
            _error.Visible = error != null;
            _reset.Visible = error != null;
            _current.Text = LiveBridge.Current ?? "—";
            var recent = LiveBridge.Decisions.AsEnumerable().Reverse().Take(HistoryShown).ToArray();
            string key = string.Join("\n", recent);
            if (key != _shownHistory)
            {
                _shownHistory = key;
                _history.Clear();
                foreach (var line in recent) _history.AddItem(line, null, false);
            }
        }
        catch (Exception ex)
        {
            GD.PushWarning("[RunRecorder panel] " + ex.Message);
        }
    }
}
