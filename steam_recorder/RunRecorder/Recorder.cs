using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace RunRecorder;

internal static class Recorder
{
	private static readonly int MainThread = System.Environment.CurrentManagedThreadId;

	private static Journal? _journal;

	private static RunState? _run;

	private static Snapshot _snapshot = new Snapshot();

	private static readonly Dictionary<GameAction, Pending> Actions = new Dictionary<GameAction, Pending>(ReferenceEqualityComparer.Instance);

	private static readonly List<Pending> Semantics = new List<Pending>();

	private static readonly HashSet<object> SkippedSets = new HashSet<object>(ReferenceEqualityComparer.Instance);

	private static ActionExecutor? _executor;

	private static bool _failed;

	private static bool _draining;

	private static bool _ended;

	private static bool? _victory;
    internal static bool? LiveOutcome => _ended ? _victory : null;

	private static int _captureErrors;

	private static int _discardedIncomplete;

	private static int _ignoredNoOps;

	private static long _order;

	private static readonly AsyncLocal<InputSnapshot?> CurrentInput = new AsyncLocal<InputSnapshot>();

	private static readonly HashSet<string> PlayerActions = new HashSet<string>(StringComparer.Ordinal) { "PlayCardAction", "UsePotionAction", "EndPlayerTurnAction", "MoveToMapCoordAction", "PickRelicAction", "DiscardPotionGameAction" };

	internal static string Status
	{
		get
		{
			if (!_failed)
			{
				if (_journal != null)
				{
					return $"记录器：{(_ended ? "本局结束" : "记录中")} · {_journal.Count} 条" + ((_captureErrors > 0) ? $" · {_captureErrors} 次状态读取异常" : "");
				}
				return "记录器：等待单人对局";
			}
			return "本局记录已停止：请检查游戏日志；新对局会自动重试";
		}
	}

	internal static bool Available
	{
		get
		{
			if (!_failed)
			{
				return System.Environment.CurrentManagedThreadId == MainThread;
			}
			return false;
		}
	}

	internal static string OutputDirectory { get; set; } = "";

	internal static List<object> HookReport { get; } = new List<object>();

	internal static void Safe(Action action)
	{
		if (!Available)
		{
			return;
		}
		try
		{
			action();
		}
		catch (Exception ex)
		{
			_failed = true;
			GD.PushError($"[RunRecorder] RECORDING STOPPED: {ex}");
			try
			{
				_journal?.Write("recorder_error", new
				{
					error = ex.ToString()
				});
			}
			catch
			{
			}
			try
			{
				_journal?.Dispose();
			}
			catch
			{
			}
			_journal = null;
			Detach();
		}
	}

	private static bool EnsureRun()
	{
		RunManager instance = RunManager.Instance;
		RunState runState = instance?.DebugOnlyGetState();
		// Loading a save publishes State before InitializeShared installs NetService.
		// IsInProgress only tests State, whereas the convenience single-player getter
		// dereferences NetService. Treat this interval as waiting, not a fatal error.
		if (runState == null || instance?.NetService == null || instance.IsCleaningUp || instance.ActionExecutor == null
			|| !instance.IsInProgress || !instance.IsSingleplayerOrFakeMultiplayer || runState.Players.Count != 1)
		{
			return false;
		}
		if (runState == _run)
		{
			if (!_ended)
			{
				BindExecutor(instance.ActionExecutor);
			}
			return !_ended;
		}
		Close("run_replaced");
		_run = runState;
		_ended = false;
		_victory = null;
		_captureErrors = 0;
		_discardedIncomplete = 0;
		_ignoredNoOps = 0;
		_order = 0L;
		_snapshot = new Snapshot();
		object obj = Snapshot.Read(instance, "_startTime");
		string s = $"{runState.Rng.StringSeed}|{obj}|{runState.Players[0].Character.Id.Entry}|{runState.AscensionLevel}";
		string runId = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(s))).Substring(0, 24).ToLowerInvariant();
		_journal = new Journal(OutputDirectory, runId);
		_journal.Write("segment_start", new
		{
			game_version = NGame.GetGameVersion(),
			recorder_version = "0.5.0",
			recording_mode = "final_choices",
			seed = runState.Rng.StringSeed,
			start_time = obj,
			start_floor = runState.TotalFloor,
			begins_at_run_start = (runState.TotalFloor == 0),
			loaded_assemblies = (from a in AppDomain.CurrentDomain.GetAssemblies().Where(delegate(Assembly a)
				{
					switch (a.GetName().Name)
					{
					case "CombatSolver":
					case "SteamModelBridge":
					case "sts2":
						return true;
					default:
						return false;
					}
				})
				select new
				{
					name = a.GetName().Name,
					version = a.GetName().Version?.ToString(),
					mvid = a.ManifestModule.ModuleVersionId
				}).ToArray(),
			hooks = HookReport,
			state = Capture()
		});
		BindExecutor(instance.ActionExecutor);
		GD.Print("[RunRecorder] recording " + _journal.PathOnDisk);
		return true;
	}

	private static void BindExecutor(ActionExecutor? executor)
	{
		if (executor != _executor)
		{
			if (_executor != null)
			{
				_executor.BeforeActionExecuted -= OnBeforeExecution;
				_executor.AfterActionExecuted -= OnAfterExecution;
			}
			_executor = executor;
			if (_executor != null)
			{
				_executor.BeforeActionExecuted += OnBeforeExecution;
				_executor.AfterActionExecuted += OnAfterExecution;
			}
		}
	}

	private static object Capture(string? decision = null, object? source = null, object[]? arguments = null)
	{
		if (_run == null)
		{
			return new
			{
				unavailable = true
			};
		}
		if (RunManager.Instance?.DebugOnlyGetState() != _run)
		{
			return new
			{
				capture_error = "Run already unloaded; no state from the next run was substituted."
			};
		}
		try
		{
			return _snapshot.Capture(_run, _ended, _victory, decision, source, arguments);
		}
		catch (Exception ex)
		{
			_captureErrors++;
			if (_captureErrors == 1)
			{
				GD.PushError("[RunRecorder] snapshot read failed: " + ex);
			}
			return new
			{
				capture_error = ex.ToString()
			};
		}
	}

	internal static Origin DetectOrigin(bool systemDefault = false)
	{
		if (LiveBridge.Executing) return new Origin("model", "steam_live_bridge");
		StackFrame[] array = new StackTrace(fNeedFileInfo: false).GetFrames() ?? Array.Empty<StackFrame>();
		StackFrame[] array2 = array;
		for (int num = 0; num < array2.Length; num++)
		{
			Type type = array2[num].GetMethod()?.DeclaringType;
			if (type?.Assembly.GetName().Name == "CombatSolver")
			{
				string? fullName = type.FullName;
				if (fullName == null || !fullName.StartsWith("CombatSolver.SolverController", StringComparison.Ordinal))
				{
					string? fullName2 = type.FullName;
					if (fullName2 == null || !fullName2.StartsWith("CombatSolver.NativeChoice", StringComparison.Ordinal))
					{
						string? fullName3 = type.FullName;
						if (fullName3 == null || !fullName3.StartsWith("CombatSolver.PlayerTurnSetupCoordinator", StringComparison.Ordinal))
						{
							goto IL_00be;
						}
					}
				}
				return new Origin("combat_solver", "call_stack:" + type.FullName);
			}
			goto IL_00be;
			IL_00be:
			if (type?.Assembly.GetName().Name == "SteamModelBridge")
			{
				return new Origin("other_mod", "call_stack:SteamModelBridge");
			}
		}
		if (array.Any(delegate(StackFrame f)
		{
			MethodBase? method = f.GetMethod();
			return (object)method != null && method.DeclaringType?.Namespace?.StartsWith("MegaCrit.Sts2.Core.Nodes", StringComparison.Ordinal) == true;
		}))
		{
			return new Origin("human", "game_ui_call_stack");
		}
		return new Origin(systemDefault ? "system" : "unknown", "no_direct_caller_evidence");
	}

	internal static void Tick()
	{
		// A corrupt segment stays failed, but must not disable later runs for the
		// entire game process. Never retry an error in the same run as clean data.
		if (System.Environment.CurrentManagedThreadId == MainThread && _failed
			&& RunManager.Instance?.NetService != null
			&& RunManager.Instance.DebugOnlyGetState() is { } next && next != _run)
			_failed = false;
		Safe(delegate
		{
			SelectionCapture.Prune();
			if (_run != null && RunManager.Instance?.DebugOnlyGetState() != _run)
			{
				Close("run_unloaded");
			}
			if (_journal != null)
			{
				Drain();
			}
			if (EnsureRun())
			{
				Drain();
			}
		});
	}

	internal static void CaptureSelection(SelectionScope scope, SelectionOffer offer)
	{
		Safe(delegate
		{
			if (EnsureRun() && scope.Player == _run.Players.Single())
			{
				scope.Offer = offer;
				scope.State = Journal.Freeze(Capture("card_selection", offer));
                scope.CapturedUtc = DateTimeOffset.UtcNow;
			}
		});
	}

	internal static bool PrepareSelection(Player? player)
	{
		bool ready = false;
		Safe(delegate
		{
			ready = player != null && EnsureRun() && player == _run.Players.Single();
		});
		return ready;
	}

	private static Pending New(string command, Origin origin)
	{
		return new Pending
		{
			Run = _run,
			Command = command,
			Origin = origin,
			Domain = Domain(),
			Floor = _run.TotalFloor,
			Order = ++_order
		};
	}

	private static string Domain()
	{
		CombatManager instance = CombatManager.Instance;
		if (instance == null || !instance.IsInProgress)
		{
			return "strategic";
		}
		return "combat";
	}

	internal static InputSnapshot? BeginInput(string actionType)
	{
		InputSnapshot captured = null;
		Safe(delegate
		{
			if (EnsureRun())
			{
				captured = new InputSnapshot
				{
					ActionType = actionType,
					State = Journal.Freeze(Capture(actionType)),
					Origin = DetectOrigin(),
					Parent = CurrentInput.Value
				};
				CurrentInput.Value = captured;
			}
		});
		return captured;
	}

	internal static void EndInput(InputSnapshot? input)
	{
		if (input != null && CurrentInput.Value == input)
		{
			CurrentInput.Value = input.Parent;
		}
	}

	private static InputSnapshot? TakeInput(string actionType)
	{
		InputSnapshot value = CurrentInput.Value;
		if (value == null || value.Used || value.ActionType != actionType)
		{
			return null;
		}
		value.Used = true;
		return value;
	}

	internal static void Enqueue(object synchronizer, GameAction action)
	{
		Safe(delegate
		{
			if (PlayerActions.Contains(action.GetType().Name) && synchronizer == RunManager.Instance?.ActionQueueSynchronizer && EnsureRun())
			{
				Drain();
				if (!Actions.ContainsKey(action))
				{
					InputSnapshot inputSnapshot = TakeInput(action.GetType().Name);
					Pending pending = New(action.GetType().Name, inputSnapshot?.Origin ?? DetectOrigin());
					pending.Input = inputSnapshot;
					Actions.Add(action, pending);
					action.BeforeCancelled += OnCancelled;
				}
			}
		});
	}

	private static void OnBeforeExecution(GameAction action)
	{
		Safe(delegate
		{
			if (_journal == null || !PlayerActions.Contains(action.GetType().Name))
			{
				return;
			}
			Drain();
			if (!Actions.TryGetValue(action, out Pending value))
			{
				InputSnapshot inputSnapshot = TakeInput(action.GetType().Name);
				value = New(action.GetType().Name, inputSnapshot?.Origin ?? new Origin("system", "executor_without_local_submission"));
				value.Input = inputSnapshot;
				Actions.Add(action, value);
				action.BeforeCancelled += OnCancelled;
			}
			if (value.Started)
			{
				return;
			}
			value.Started = true;
			value.StartedUtc = DateTimeOffset.UtcNow;
			Pending pending = value;
			int nativeNoOp;
			if (action is EndPlayerTurnAction && Snapshot.Read(action, "_turnNumber") is int num && Snapshot.Read(action, "_player") is Player player)
			{
				PlayerCombatState playerCombatState = player.PlayerCombatState;
				if (playerCombatState != null)
				{
					nativeNoOp = ((num != playerCombatState.TurnNumber) ? 1 : 0);
					goto IL_014b;
				}
			}
			nativeNoOp = 0;
			goto IL_014b;
			IL_014b:
			pending.NativeNoOp = (byte)nativeNoOp != 0;
			if (action is PickRelicAction)
			{
				value.NativeNoOp = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics == null;
				value.ProvisionalChoice = Snapshot.Read(action, "_relicIndex") == null;
			}
			value.ExecutionState = Journal.Freeze(Capture(value.Command));
			value.State = value.Input?.State ?? value.ExecutionState;
			value.Action = Journal.Freeze(_snapshot.Action(action));
			value.ChoiceKey = Journal.Freeze(_snapshot.ChoiceKey(action, value.Command));
		});
	}

	private static void OnAfterExecution(GameAction action)
	{
		Safe(delegate
		{
			if (Actions.Remove(action, out Pending value))
			{
				action.BeforeCancelled -= OnCancelled;
				if (!value.Started)
				{
					_discardedIncomplete++;
					_journal?.Write("recording_gap", new
					{
						action_id = value.Id,
						command = value.Command,
						reason = "completion_without_pre_action_snapshot"
					});
					Discard(value);
				}
				else
				{
					Drain();
					if (value.NativeNoOp)
					{
						_ignoredNoOps++;
						Discard(value);
					}
					else if (value.ProvisionalChoice)
					{
						Discard(value);
					}
					else if (action.Exception != null || action.State.ToString().Contains("Cancel", StringComparison.OrdinalIgnoreCase))
					{
						Discard(value);
					}
					else
					{
						Finish(value, null, "executor_after_action");
					}
				}
			}
		});
	}

	private static void OnCancelled(GameAction action)
	{
		Safe(delegate
		{
			if (Actions.Remove(action, out Pending value))
			{
				action.BeforeCancelled -= OnCancelled;
				Discard(value);
			}
		});
	}

	private static Pending? Parent()
	{
		Pending pending = Semantics.LastOrDefault((Pending p) => !p.Finished && p.Floor == _run.TotalFloor && p.Domain == Domain());
		if (pending != null)
		{
			return pending;
		}
		GameAction gameAction = _executor?.CurrentlyRunningAction;
		if (gameAction == null || !Actions.TryGetValue(gameAction, out Pending value))
		{
			return null;
		}
		return value;
	}

	internal static Pending? Begin(MethodBase method, object? instance, object[] args)
	{
		Pending result = null;
		Safe(delegate
		{
			if (EnsureRun() && IsLive(instance, args))
			{
				Drain();
				string source = method.DeclaringType.Name + "." + method.Name;
				string text = method.Name switch
				{
					"Chosen" => "choose_event_option", 
					"ChooseLocalOption" => "choose_rest_option", 
					"SelectLocalReward" => "take_reward", 
					"SkipRewardsSet" => "skip_rewards", 
					"OnTryPurchaseWrapper" => "purchase", 
					"SyncLocalChoice" => "select_cards_or_option", 
					"CellClicked" => "crystal_sphere_cell", 
					_ => null, 
				};
				Pending value = null;
				bool flag = instance is MerchantRoom && method.Name == "Exit";
				bool flag2 = instance is TreasureRoomRelicSynchronizer && method.Name == "OnRoomExited";
				if (flag || flag2)
				{
					if (!(_executor?.CurrentlyRunningAction is MoveToMapCoordAction key) || !Actions.TryGetValue(key, out value) || !value.Started || value.Finished || (flag && !args.Any((object a) => a == _run)))
					{
						return;
					}
					if (flag2)
					{
						object obj = Snapshot.Read(instance, "_singleplayerSkipped");
						if (!(obj is bool) || !(bool)obj || RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics == null)
						{
							return;
						}
					}
					text = (flag ? "leave_shop" : "skip_treasure");
				}
				if (text != null)
				{
					object onceKey = null;
					Dictionary<string, object> dictionary = method.GetParameters().Select((ParameterInfo p, int i) => new
					{
						Name = p.Name,
						Value = _snapshot.Value(args[i])
					}).ToDictionary(p => p.Name ?? "arg", p => p.Value);
					if (instance is EventOption eventOption)
					{
						EventModel eventModel = Snapshot.CurrentEvent(_run);
						if (eventModel == null || !eventModel.CurrentOptions.Contains(eventOption) || !Snapshot.IsFinalEventOption(eventOption))
						{
							return;
						}
						dictionary["event_id"] = eventModel.Id.Entry;
						dictionary["option"] = _snapshot.Option(eventOption);
						dictionary["option_index"] = eventModel.CurrentOptions.ToList().IndexOf(eventOption);
					}
					if (method.Name == "SyncLocalChoice")
					{
						dictionary["choice"] = _snapshot.Choice(args[^1]);
					}
					if (method.Name == "SkipRewardsSet")
					{
						if (!(Snapshot.Read(args[0], "set") is RewardsSet rewardsSet) || rewardsSet.Player != _run.Players[0] || SkippedSets.Contains(rewardsSet))
						{
							return;
						}
						Reward[] array = rewardsSet.Rewards.Where((Reward r) => !r.SuccessfullySelected).ToArray();
						if (array.Length == 0)
						{
							return;
						}
						onceKey = rewardsSet;
						dictionary.Clear();
						dictionary["reward_set_id"] = rewardsSet.Id;
						dictionary["rewards"] = array.Select(_snapshot.Reward).ToArray();
					}
					Pending pending = value ?? Parent();
					Origin origin = DetectOrigin();
					if (origin.Actor == "unknown" && pending != null)
					{
						origin = new Origin(pending.Origin.Actor, "parent_action:" + pending.Id);
					}
					result = New(text, origin);
					result.Parent = pending;
					result.OnceKey = onceKey;
					result.State = Journal.Freeze(Capture(text, instance, args));
					result.Action = Journal.Freeze(new
					{
						command = text,
						source = source,
						instance = _snapshot.Value(instance),
						args = dictionary
					});
					result.ChoiceKey = Journal.Freeze(_snapshot.ChoiceKey(instance, text, args));
					if (text == "select_cards_or_option" && args[0] is Player player)
					{
						SelectionScope selectionScope = SelectionCapture.Consume(player, Convert.ToUInt32(args[1]));
						if (selectionScope != null && (object)selectionScope.Offer != null && selectionScope.State.ValueKind == JsonValueKind.Object)
						{
							result.State = selectionScope.State;
                            result.SelectionCapturedUtc = selectionScope.CapturedUtc;
							result.ChoiceKey = Journal.Freeze(_snapshot.SelectionKey(selectionScope.Offer, (PlayerChoiceResult)args[^1]));
						}
					}
					result.Started = true;
					Semantics.Add(result);
				}
			}
		});
		return result;
	}

	private static bool IsLive(object? instance, object[] args)
	{
		RunManager instance2 = RunManager.Instance;
		if (instance != null && instance.GetType().Name.EndsWith("Synchronizer", StringComparison.Ordinal) && Snapshot.Read(instance2, instance.GetType().Name) != instance)
		{
			return false;
		}
		foreach (Player player in args.OfType<Player>())
		{
			if (_run == null || !_run.Players.Any((Player p) => p == player))
			{
				return false;
			}
		}
		object owner = Snapshot.Read(instance, "_player");
		if (owner != null && !_run.Players.Any((Player p) => p == owner))
		{
			return false;
		}
		if (instance is Node node)
		{
			if (GodotObject.IsInstanceValid(node))
			{
				return node.IsInsideTree();
			}
			return false;
		}
		return true;
	}

	internal static void Returned(Pending? pending, object? result)
	{
		Safe(delegate
		{
			if (pending != null && !pending.Finished && pending.Run == _run && _journal != null)
			{
				if (result is Task task)
				{
					pending.Task = task;
					Drain();
				}
				else if (result is bool && !(bool)result)
				{
					Discard(pending);
				}
				else
				{
					Finish(pending, result, "selection_submitted");
				}
			}
		});
	}

	internal static void Failed(Pending? pending, Exception? exception)
	{
		Safe(delegate
		{
			if (pending != null && exception != null && !pending.Finished)
			{
				Discard(pending);
			}
		});
	}

	private static void Drain()
	{
		if (_draining)
		{
			return;
		}
		_draining = true;
		try
		{
			Pending[] array = Semantics.Where((Pending p) => p.Task?.IsCompleted ?? false).Reverse().ToArray();
			foreach (Pending pending in array)
			{
				if (pending.Finished)
				{
					continue;
				}
				Task task = pending.Task;
				if (!task.IsCompletedSuccessfully)
				{
					Discard(pending);
					continue;
				}
				object obj = Snapshot.Read(task, "Result");
				if (obj is bool && !(bool)obj)
				{
					Discard(pending);
				}
				else
				{
					Finish(pending, obj, "task_observed");
				}
			}
		}
		finally
		{
			_draining = false;
		}
	}

	private static void Discard(Pending pending)
	{
		pending.Finished = true;
		Semantics.Remove(pending);
		Pending[] array = Semantics.Where((Pending p) => p.Parent == pending).ToArray();
		for (int num = 0; num < array.Length; num++)
		{
			Discard(array[num]);
		}
	}

	private static void Finish(Pending pending, object? result, string boundary)
	{
		if (pending.Finished || _journal == null)
		{
			return;
		}
		pending.Finished = true;
		Semantics.Remove(pending);
		if (pending.OnceKey != null)
		{
			SkippedSets.Add(pending.OnceKey);
		}
		JsonElement jsonElement = Journal.Freeze(new
		{
			action_id = pending.Id,
			order = pending.Order,
			started_utc = pending.StartedUtc,
			committed_utc = DateTimeOffset.UtcNow,
			actor = pending.Origin.Actor,
			actor_evidence = pending.Origin.Evidence,
			domain = pending.Domain,
			action = pending.Action,
			state = pending.State,
			state_boundary = pending.SelectionCapturedUtc.HasValue ? "selection_offer" : ((pending.Input == null) ? "execution_or_semantic_entry" : "input_submission"),
			state_captured_utc = (pending.SelectionCapturedUtc ?? pending.Input?.CapturedUtc ?? pending.StartedUtc),
			execution_state = ((pending.Input == null) ? ((JsonElement?)null) : new JsonElement?(pending.ExecutionState)),
			choice_key = ((pending.ChoiceKey.ValueKind == JsonValueKind.Undefined) ? ((JsonElement?)null) : new JsonElement?(pending.ChoiceKey)),
			legal_match = Snapshot.MatchLegal(pending.State, pending.ChoiceKey),
            selection_path = SelectionTrace.Build(pending.State, pending.ChoiceKey),
			next_state = Capture(),
			status = "completed",
			boundary = boundary,
			choices = pending.Choices.ToArray(),
			result = _snapshot.Value(result)
		});
		if (pending.Parent != null)
		{
			if (!pending.Parent.Finished)
			{
				pending.Parent.Choices.Add(jsonElement);
			}
		}
		else
		{
			_journal.Write("decision_committed", jsonElement);
		}
	}

	internal static void RunEnded(bool victory)
	{
		Safe(delegate
		{
			if (EnsureRun())
			{
				Drain();
				_ended = true;
				_victory = victory;
				_journal.Write("run_ended", new
				{
					victory = victory,
					abandoned = RunManager.Instance.IsAbandoned,
					state = Capture()
				});
			}
		});
	}

	internal static void Close(string reason)
	{
		if (_journal != null)
		{
			Drain();
			_discardedIncomplete += Actions.Values.Concat(Semantics).Count((Pending p) => !p.Finished);
			_journal.Write("segment_end", new
			{
				reason = reason,
				run_ended = _ended,
				uncommitted_operations = _discardedIncomplete,
				state_capture_errors = _captureErrors,
				ignored_noop_actions = _ignoredNoOps
			});
			_journal.Dispose();
			_journal = null;
		}
		Detach();
		_run = null;
		_ended = false;
		_victory = null;
	}

	private static void Detach()
	{
		if (_executor != null)
		{
			_executor.BeforeActionExecuted -= OnBeforeExecution;
			_executor.AfterActionExecuted -= OnAfterExecution;
		}
		foreach (GameAction key in Actions.Keys)
		{
			key.BeforeCancelled -= OnCancelled;
		}
		_executor = null;
		Actions.Clear();
		Semantics.Clear();
		SkippedSets.Clear();
		SelectionCapture.Reset();
		CurrentInput.Value = null;
	}
}
