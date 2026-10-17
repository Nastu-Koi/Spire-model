using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Text.RegularExpressions;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Runs;

namespace RunRecorder;

[ModInitializer("Initialize")]
public static class ModEntry
{
	private static Timer? _timer;
    private static FeatureSettings _settings = new();
    private static string SettingsPath => Path.Combine(Recorder.OutputDirectory, "settings.json");

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
			Recorder.OutputDirectory = ProjectSettings.GlobalizePath("user://run_recorder");
			Directory.CreateDirectory(Recorder.OutputDirectory);
            try { _settings = FeatureSettings.Load(SettingsPath); }
            catch (Exception ex) { GD.PushWarning("[RunRecorder] Cannot read settings; using defaults: " + ex.Message); }
            ApplySettings();
			InstallHooks();
			Callable.From(Start).CallDeferred();
			GD.Print("[RunRecorder] initialized; output=" + Recorder.OutputDirectory);
		}
		catch (Exception ex)
		{
			GD.PushError("[RunRecorder] disabled: " + ex);
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
			Name = "RunRecorderTimer",
			WaitTime = 0.1,
			Autostart = true
		};
        var panel = LivePanel.Create(NGame.Instance, _settings, ApplySettings);
		_timer.Timeout += delegate
		{
            SelectionCapture.Safe(SelectionCapture.Tick);
			Recorder.Tick();
            LiveBridge.Tick();
            panel.Refresh();
		};
		_timer.TreeExiting += delegate
		{
			Recorder.Safe(delegate
			{
				Recorder.Close("game_exit");
			});
		};
		NGame.Instance.AddChild(_timer, forceReadableName: false, Node.InternalMode.Disabled);
	}

    private static void ApplySettings()
    {
        Recorder.SetEnabled(_settings.RecordingEnabled);
        LiveBridge.Configure(_settings);
        try { _settings.Save(SettingsPath); }
        catch (Exception ex) { GD.PushWarning("[RunRecorder] Settings changed for this session but could not be saved: " + ex.Message); }
    }

	private static void InstallHooks()
	{
		Harmony harmony = new Harmony("local.sts2.run-recorder");
        Spire.PublicHistory.NativeHistory.Install(harmony);
		Assembly assembly = typeof(RunManager).Assembly;
		(string, string)[] obj = new(string, string)[9]
		{
			("MegaCrit.Sts2.Core.Events.EventOption", "Chosen"),
			("MegaCrit.Sts2.Core.Multiplayer.Game.RestSiteSynchronizer", "ChooseLocalOption"),
			("MegaCrit.Sts2.Core.Multiplayer.Game.RewardsSetSynchronizer", "SelectLocalReward"),
			("MegaCrit.Sts2.Core.Multiplayer.Game.RewardsSetSynchronizer", "SkipRewardsSet"),
			("MegaCrit.Sts2.Core.GameActions.Multiplayer.PlayerChoiceSynchronizer", "SyncLocalChoice"),
			("MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry", "OnTryPurchaseWrapper"),
			("MegaCrit.Sts2.Core.Rooms.MerchantRoom", "Exit"),
			("MegaCrit.Sts2.Core.Multiplayer.Game.TreasureRoomRelicSynchronizer", "OnRoomExited"),
			("MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame", "CellClicked")
		};
		List<MethodInfo> list = new List<MethodInfo>();
		(string, string)[] array = obj;
		for (int i = 0; i < array.Length; i++)
		{
			(string, string) tuple = array[i];
			MethodInfo item = AccessTools.DeclaredMethod(assembly.GetType(tuple.Item1), tuple.Item2) ?? throw new MissingMethodException(tuple.Item1, tuple.Item2);
			list.Add(item);
		}
		try
		{
			harmony.Patch(AccessTools.Method(typeof(ActionQueueSynchronizer), "RequestEnqueue"), new HarmonyMethod(typeof(Patches), "Enqueue"));
			harmony.Patch(AccessTools.Method(typeof(RunManager), "OnEnded"), null, new HarmonyMethod(typeof(Patches), "End"));
			harmony.Patch(AccessTools.Method(typeof(RunManager), "CleanUp"), new HarmonyMethod(typeof(Patches), "Cleanup"));
			harmony.Patch(AccessTools.Method(typeof(NGame), "_Ready"), null, new HarmonyMethod(typeof(ModEntry), "Start"));
			foreach (MethodInfo item2 in list)
			{
				harmony.Patch(item2, new HarmonyMethod(typeof(Patches), "Prefix"), new HarmonyMethod(typeof(Patches), (item2.ReturnType == typeof(void)) ? "VoidPostfix" : "Postfix"), null, new HarmonyMethod(typeof(Patches), "Finalizer"));
				Recorder.HookReport.Add(new
				{
					type = item2.DeclaringType.FullName,
					method = item2.Name,
					return_type = item2.ReturnType.FullName
				});
			}
			SelectionHooks.Install(harmony, assembly);
            // Track the full UI task, including removal/animation after the
            // synchronizer returns. These hooks also work with recording off.
            foreach (var target in new[] {
                ("MegaCrit.Sts2.Core.Nodes.Rewards.NRewardButton", "GetReward"),
                ("MegaCrit.Sts2.Core.Nodes.RestSite.NRestSiteButton", "SelectOption") })
            {
                var method = AccessTools.DeclaredMethod(assembly.GetType(target.Item1), target.Item2)
                    ?? throw new MissingMethodException(target.Item1, target.Item2);
                harmony.Patch(method, postfix: new HarmonyMethod(typeof(Patches), "ControlTask"));
            }
			MethodInfo[] array2 = new MethodInfo[1] { AccessTools.Method(typeof(PotionModel), "EnqueueManualUse") };
			foreach (MethodInfo methodInfo in array2)
			{
				harmony.Patch(methodInfo, new HarmonyMethod(typeof(Patches), "InputBegin")
				{
					priority = 800
				}, null, null, new HarmonyMethod(typeof(Patches), "InputEnd"));
				Recorder.HookReport.Add(new
				{
					type = methodInfo.DeclaringType.FullName,
					method = methodInfo.Name,
					purpose = "input_boundary"
				});
			}
			File.WriteAllText(Path.Combine(Recorder.OutputDirectory, "hooks.json"), JsonSerializer.Serialize(Recorder.HookReport, new JsonSerializerOptions
			{
				WriteIndented = true
			}));
		}
		catch
		{
			harmony.UnpatchAll(harmony.Id);
			throw;
		}
	}
}
