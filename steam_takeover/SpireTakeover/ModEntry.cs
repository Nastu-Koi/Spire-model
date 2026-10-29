using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using System.Text.RegularExpressions;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Runs;

namespace SpireTakeover;

[ModInitializer("Initialize")]
public static class ModEntry
{
	private static Timer? _timer;
    private static FeatureSettings _settings = new();
    internal static string DataDirectory { get; private set; } = "";
    private static string SettingsPath => Path.Combine(DataDirectory, "settings.json");

	public static void Initialize()
	{
		if (DisplayServer.GetName().Equals("headless", StringComparison.OrdinalIgnoreCase) || OS.GetCmdlineArgs().Contains("--headless"))
		{
			return;
		}
		try
		{
			string text = NGame.GetGameVersion() ?? "";
			if (!Regex.IsMatch(text, "(?<!\\d)0\\.111\\.0(?!\\d)"))
			{
				throw new NotSupportedException("Game " + text + " has not been adapted. Expected 0.111.0.");
			}
			DataDirectory = ProjectSettings.GlobalizePath("user://spire_takeover");
			Directory.CreateDirectory(DataDirectory);
            try { _settings = FeatureSettings.Load(SettingsPath); }
            catch (Exception ex) { GD.PushWarning("[SpireTakeover] Cannot read settings; using defaults: " + ex.Message); }
            ApplySettings();
			InstallHooks();
			Callable.From(Start).CallDeferred();
			GD.Print("[SpireTakeover] initialized; data=" + DataDirectory);
		}
		catch (Exception ex)
		{
			GD.PushError("[SpireTakeover] disabled: " + ex);
		}
	}

	private static void Start()
	{
		if (_timer != null || NGame.Instance == null)
		{
			return;
		}
		_timer = new Timer
		{
			Name = "SpireTakeoverTimer",
			WaitTime = 0.1,
			Autostart = true
		};
        var panel = LivePanel.Create(NGame.Instance, _settings, ApplySettings);
		_timer.Timeout += delegate
		{
            SelectionCapture.Safe(SelectionCapture.Tick);
            LiveBridge.Tick();
            panel.Refresh();
		};
		NGame.Instance.AddChild(_timer, forceReadableName: false, Node.InternalMode.Disabled);
	}

    private static void ApplySettings()
    {
        LiveBridge.Configure(_settings);
        try { _settings.Save(SettingsPath); }
        catch (Exception ex) { GD.PushWarning("[SpireTakeover] Settings changed for this session but could not be saved: " + ex.Message); }
    }

	private static void InstallHooks()
	{
		Harmony harmony = new Harmony("local.sts2.spire-takeover");
        Spire.PublicHistory.NativeHistory.Install(harmony);
		Assembly assembly = typeof(RunManager).Assembly;
        // The native tasks an executed action starts: the bridge waits for them (and for
        // the removal and animation after the synchronizer returns) before observing again.
        var tasks = new[] {
            ("MegaCrit.Sts2.Core.Events.EventOption", "Chosen"),
            ("MegaCrit.Sts2.Core.Multiplayer.Game.RestSiteSynchronizer", "ChooseLocalOption"),
            ("MegaCrit.Sts2.Core.Multiplayer.Game.RewardsSetSynchronizer", "SelectLocalReward"),
            ("MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry", "OnTryPurchaseWrapper"),
            ("MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame", "CellClicked"),
            ("MegaCrit.Sts2.Core.Nodes.Rewards.NRewardButton", "GetReward"),
            ("MegaCrit.Sts2.Core.Nodes.RestSite.NRestSiteButton", "SelectOption") };
		try
		{
			harmony.Patch(AccessTools.Method(typeof(RunManager), "OnEnded"), null, new HarmonyMethod(typeof(Patches), "End"));
			harmony.Patch(AccessTools.Method(typeof(RunManager), "CleanUp"), new HarmonyMethod(typeof(Patches), "Cleanup"));
			harmony.Patch(AccessTools.Method(typeof(NGame), "_Ready"), null, new HarmonyMethod(typeof(ModEntry), "Start"));
            foreach (var target in tasks)
            {
                var method = AccessTools.DeclaredMethod(assembly.GetType(target.Item1), target.Item2)
                    ?? throw new MissingMethodException(target.Item1, target.Item2);
                if (!typeof(Task).IsAssignableFrom(method.ReturnType))
                    throw new MissingMethodException(target.Item1, target.Item2 + " no longer returns a Task");
                harmony.Patch(method, postfix: new HarmonyMethod(typeof(Patches), "ControlTask"));
            }
			SelectionHooks.Install(harmony, assembly);
		}
		catch
		{
			harmony.UnpatchAll(harmony.Id);
			throw;
		}
	}
}
