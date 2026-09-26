using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Cards;
using MegaCrit.Sts2.Core.Rooms;
using Sts2Headless;

namespace CombatSolverCli;

internal sealed class SolverAdapter(Assembly assembly)
{
	private const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

	private object? _selector;

	private List<CardModel>? _selectedCards;

	private string? _selectionId;

	private bool _poisoned;
	private string? _refinementFailure;
	private int _effectiveBudgetMs;
	private readonly Queue<object> _turnPlan = new();
	private int _planTurn = -1;
	private string? _planPolicy;

	public bool Poisoned => _poisoned;

	private Type Type(string name)
	{
		return assembly.GetType("CombatSolver." + name, throwOnError: true);
	}

	private object New(string name, params object?[] args)
	{
		return Activator.CreateInstance(Type(name), BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic, null, args, null);
	}

	private object? Call(string type, string name, params object?[] args)
	{
		return Type(type).GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).Single((MethodInfo m) => m.Name == name && m.GetParameters().Length == args.Length && m.GetParameters().Zip(args).All<(ParameterInfo, object)>(((ParameterInfo First, object Second) p) => p.Second == null || p.First.ParameterType.IsInstanceOfType(p.Second))).Invoke(null, args);
	}

	private static object? Get(object obj, string name)
	{
		return obj.GetType().GetProperty(name, BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).GetValue(obj);
	}

	private object Enum(string type, string name)
	{
		return System.Enum.Parse(Type(type), name);
	}

	public object Info()
	{
		return new
		{
			type = "solver_info",
			adapter_version = "combat-solver-cli-v1",
			solver_assembly = assembly.FullName,
			solver_sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(assembly.Location))).ToLowerInvariant(),
			teacher_visibility = "unverified",
			execution = "engine-candidates-v1",
			strategy = "optional_reuse_within_turn",
			game = typeof(CardModel).Assembly.FullName,
			combat_in_progress = CombatManager.Instance.IsInProgress,
			boss_combat = IsBossCombat()
		};
	}

	private static bool IsBossCombat()
	{
		return CombatManager.Instance.IsInProgress
			&& CombatManager.Instance.DebugOnlyGetState()?.RunState.CurrentRoom?.RoomType == RoomType.Boss;
	}

	private static int EffectiveBudget(JsonElement request)
	{
		int normal = request.TryGetProperty("budget_ms", out var normalValue) ? normalValue.GetInt32() : 1000;
		int boss = request.TryGetProperty("boss_budget_ms", out var bossValue) ? bossValue.GetInt32() : normal;
		if (normal < 1 || normal > 120000 || boss < 1 || boss > 120000)
			throw new ArgumentOutOfRangeException("budget_ms / boss_budget_ms");
		return IsBossCombat() ? boss : normal;
	}

	public void PrepareEngine()
	{
		// Restore the native Neutralize action; the CLI legacy prefix bypasses native attack commands.
		// Keep the solver patch audit enabled and remove only this known compatibility prefix.
		MethodInfo method = typeof(Neutralize).GetMethod("OnPlay", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
		MethodInfo prefix = typeof(RunSimulator).GetNestedType("LocPatches", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).GetMethod("NeutralizePrefix", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
		Patches patchInfo = Harmony.GetPatchInfo((MethodBase)method);
		if (patchInfo != null && patchInfo.Prefixes.Any((Patch p) => p.PatchMethod == prefix))
		{
			new Harmony("combat-solver-cli").Unpatch((MethodBase)method, prefix);
		}
	}

	public void Reset()
	{
		_turnPlan.Clear();
		_planTurn = -1;
		_planPolicy = null;
		ClearSelection();
	}

	private void ClearSelection()
	{
		_selector = null;
		_selectedCards = null;
		_selectionId = null;
	}

	private static void CheckRequest(JsonElement request, JsonElement frame)
	{
		JsonElement property = frame.GetProperty("routing");
		if (frame.GetProperty("boundary").GetString() != "decision" || !request.TryGetProperty("decision_id", out var value) || value.GetString() != property.GetProperty("decision_id").GetString() || !request.TryGetProperty("state_version", out var value2) || value2.GetInt64() != property.GetProperty("state_version").GetInt64())
		{
			throw new InvalidOperationException("stale_solver_request: send the current decision_id and state_version");
		}
		if (property.TryGetProperty("selection_revision", out var value3) && (!request.TryGetProperty("selection_revision", out var value4) || value4.GetInt64() != value3.GetInt64()))
		{
			throw new InvalidOperationException("stale_solver_selection");
		}
	}

	public object Solve(JsonElement request, JsonElement frame)
	{
		CheckRequest(request, frame);
		object result = Search(request, frame);
		return Describe(result);
	}

	private object Describe(object result)
	{
		return new
		{
			type = "solver_result",
			teacher_visibility = "unverified",
			actions = Get(Get(result, "BestNode"), "Actions"),
			snapshot = Get(result, "Snapshot"),
			boundary_reason = Get(result, "BoundaryReason"),
			result_scope = Get(result, "ResultScope"),
			expanded_nodes = Get(result, "ExpandedNodes"),
			only_death_routes_found = Get(result, "OnlyDeathRoutesFound"),
			refinement_failure = _refinementFailure,
			budget_ms = _effectiveBudgetMs,
			boss_combat = IsBossCombat()
		};
	}

	public object Step(JsonElement request, JsonElement frame, object simulator)
	{
		CheckRequest(request, frame);
		if (_poisoned)
		{
			throw new InvalidOperationException("Solver isolation failed; restart this worker");
		}
		RunSimulator runSimulator = (RunSimulator)simulator;
		CombatState val = CombatManager.Instance.DebugOnlyGetState();
		if (!CombatManager.Instance.IsInProgress || val == null)
		{
			throw new InvalidOperationException("solver_step requires an active combat");
		}
		// Entry effects and replayed actions can expose choices without a solver plan.
		// Return the unchanged boundary so the run search can retain every legal branch.
		if (_selector == null &&
			((frame.GetProperty("public").TryGetProperty("selection_context", out var selection) && selection.ValueKind == JsonValueKind.Object)
			 || frame.GetProperty("public").GetProperty("phase").GetString() == "card_reward"))
		{
			return new { type = "solver_selection_required", frame = frame };
		}
		Player player = val.Players.Single();
		object search = null;
		JsonElement jsonElement;
		if (frame.GetProperty("public").TryGetProperty("selection_context", out var value) && value.ValueKind == JsonValueKind.Object)
		{
			jsonElement = SelectCandidate(runSimulator, frame, value);
		}
		else if (frame.GetProperty("public").GetProperty("phase").GetString() == "card_reward")
		{
			jsonElement = SelectReward(runSimulator, frame);
		}
		else
		{
			if (_selector != null)
			{
				_selector.GetType().GetMethod("ReconcileImplicitChoices", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).Invoke(_selector, new object[1] { player });
				_selector.GetType().GetMethod("AssertConsumed", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).Invoke(_selector, null);
				ClearSelection();
			}
			bool reuse = request.TryGetProperty("reuse_turn_plan", out var reuseValue) && reuseValue.GetBoolean();
			int turn = player.PlayerCombatState.TurnNumber;
			string policy = string.Join("|", new[] { "budget_ms", "boss_budget_ms", "potions", "potion_policy", "beam_width", "beam_portfolio" }
				.Select(k => request.TryGetProperty(k, out var v) ? v.GetRawText() : ""));
			if (!reuse || _planTurn != turn || _planPolicy != policy) _turnPlan.Clear();
			object obj2;
			if (_turnPlan.Count > 0)
			{
				try { jsonElement = MapAction(runSimulator, frame, player, _turnPlan.Peek()); }
				catch (InvalidOperationException) { _turnPlan.Clear(); }
			}
			if (_turnPlan.Count == 0)
			{
				object result = Search(request, frame);
				search = Describe(result);
				foreach (object action in ((IEnumerable)Get(Get(result, "BestNode"), "Actions")).Cast<object>())
				{
					if ((int)Get(action, "Turn") != turn) continue;
					if (Get(action, "Kind").ToString() is "PlayCard" or "UsePotion" or "EndTurn") _turnPlan.Enqueue(action);
				}
				_planTurn = turn;
				_planPolicy = policy;
			}
			if (_turnPlan.Count == 0) throw new InvalidOperationException("Solver returned no executable action; increase the search budget");
			obj2 = _turnPlan.Dequeue();
			jsonElement = MapAction(runSimulator, frame, player, obj2);
			List<object> list = ((IEnumerable)obj2.GetType().GetMethod("GetActionChoicesInExecutionOrder", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).Invoke(obj2, null)).Cast<object>().ToList();
			if (Get(obj2, "Kind").ToString() == "EndTurn" || (bool)Get(obj2, "EndsPlayerTurn"))
			{
				list.AddRange((((IEnumerable)Get(obj2, "TurnStartChoices")) ?? Array.Empty<object>()).Cast<object>());
			}
			Array array = Array.CreateInstance(Type("PlanCardChoice"), list.Count);
			for (int num = 0; num < list.Count; num++)
			{
				array.SetValue(list[num], num);
			}
			_selector = New("PlannedCardSelector", array);
			_selector.GetType().GetMethod("CaptureBefore", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).Invoke(_selector, new object[1] { player });
		}
		JsonElement property = frame.GetProperty("routing");
		JsonElement value2;
		Dictionary<string, object> frame2 = runSimulator.ExecuteCandidate(property.GetProperty("decision_id").GetString(), property.GetProperty("state_version").GetInt64(), jsonElement.GetProperty("candidate_ref").GetString(), property.TryGetProperty("selection_revision", out value2) ? new long?(value2.GetInt64()) : ((long?)null));
		return new
		{
			type = "solver_step",
			teacher = "CombatSolver",
			teacher_visibility = "unverified",
			before = frame,
			candidate_ref = jsonElement.GetProperty("candidate_ref").GetString(),
			frame = frame2,
			search = search
		};
	}

	private static object Snapshot(RunSimulator sim)
	{
		return typeof(RunSimulator).GetMethod("BuildPublicSnapshot", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).Invoke(sim, null);
	}

	private static JsonElement Find(JsonElement frame, string verb, string? source = null, string? target = null)
	{
		JsonElement[] array = (from c in frame.GetProperty("legal").GetProperty("candidates").EnumerateArray()
			where c.GetProperty("verb").GetString() == verb && (from x in c.GetProperty("source_refs").EnumerateArray()
				select x.GetString()).SequenceEqual<string>((source == null) ? ((IEnumerable<string>)Array.Empty<string>()) : ((IEnumerable<string>)new string[1] { source })) && (from x in c.GetProperty("target_refs").EnumerateArray()
				select x.GetString()).SequenceEqual<string>((target == null) ? ((IEnumerable<string>)Array.Empty<string>()) : ((IEnumerable<string>)new string[1] { target }))
			select c).ToArray();
		if (array.Length != 1)
		{
			throw new InvalidOperationException($"No unique legal candidate for {verb}/{source}/{target}");
		}
		return array[0];
	}

	private JsonElement MapAction(RunSimulator sim, JsonElement frame, Player player, object action)
	{
		string text = Get(action, "Kind").ToString();
		if (text == "EndTurn")
		{
			return Find(frame, "END_TURN");
		}
		object obj = Snapshot(sim);
		Dictionary<Creature, string> dictionary = (Dictionary<Creature, string>)Get(obj, "Creatures");
		uint? targetId = (uint?)Get(action, "TargetCombatId");
		string text2 = ((!targetId.HasValue) ? null : dictionary.Single((KeyValuePair<Creature, string> p) => p.Key.CombatId == targetId).Value);
		if (text == "UsePotion")
		{
			PotionModel potionAtSlotIndex = player.GetPotionAtSlotIndex((int)Get(action, "PotionSlot"));
			if (potionAtSlotIndex == null || ((AbstractModel)potionAtSlotIndex).Id.Entry != (string)Get(action, "PotionId"))
			{
				throw new InvalidOperationException("Planned potion is missing or changed");
			}
			if (text2 == null && potionAtSlotIndex.IsValidTarget(player.Creature))
			{
				text2 = dictionary[player.Creature];
			}
			return Find(frame, "USE_POTION", ((IDictionary)Get(obj, "Potions"))[potionAtSlotIndex] as string, text2);
		}
		IReadOnlyList<CardModel> cards = player.PlayerCombatState.Hand.Cards;
		string key = (string)Get(action, "CardStateKey");
		IEnumerable<CardModel> source = cards.Where((CardModel c) => (!string.IsNullOrEmpty(key)) ? ((string)Call("CardChoiceSupport", "ChoiceCardKey", c) == key) : (((AbstractModel)c).Id.Entry == (string)Get(action, "CardId")));
		CardModel key2 = source.Skip((int)Get(action, string.IsNullOrEmpty(key) ? "CardOccurrence" : "CardStateOccurrence")).FirstOrDefault() ?? throw new InvalidOperationException("Planned hand card is missing or changed");
		return Find(frame, "PLAY_CARD", ((Dictionary<CardModel, string>)Get(obj, "Cards"))[key2], text2);
	}

	private JsonElement SelectCandidate(RunSimulator sim, JsonElement frame, JsonElement context)
	{
		if (_selector == null)
		{
			throw new InvalidOperationException("Combat selection has no solver plan; complete it externally before solving");
		}
		object value = typeof(RunSimulator).GetField("_cardSelector", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).GetValue(sim);
		List<CardModel> list = (List<CardModel>)Get(value, "PendingOptions");
		if (list == null)
		{
			throw new InvalidOperationException("Unsupported combat selection surface");
		}
		string text = frame.GetProperty("routing").GetProperty("selection_id").GetString();
		if (_selectionId != text)
		{
			Task<IEnumerable<CardModel>> task = (Task<IEnumerable<CardModel>>)_selector.GetType().GetMethod("GetSelectedCards", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).Invoke(_selector, new object[3]
			{
				list,
				context.GetProperty("min_total").GetInt32(),
				context.GetProperty("max_total").GetInt32()
			});
			_selectedCards = task.GetAwaiter().GetResult().ToList();
			_selectionId = text;
		}
		int @int = context.GetProperty("selected_count").GetInt32();
		if (@int == _selectedCards.Count)
		{
			return Find(frame, "FINISH_SELECTION");
		}
		JsonElement[] array = frame.GetProperty("legal").GetProperty("candidates").EnumerateArray()
			.ToArray();
		if (array.Length == 1 && array[0].GetProperty("verb").GetString() == "FINISH_SELECTION")
		{
			return array[0];
		}
		object obj = Snapshot(sim);
		MethodInfo describe = typeof(RunSimulator).GetMethod("PublicSelectionCard", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
		foreach (CardModel item in list.OrderBy<CardModel, string>((CardModel c) => JsonSerializer.Serialize(describe.Invoke(null, new object[1] { c })), StringComparer.Ordinal))
		{
			obj.GetType().GetMethod("AddCard", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).Invoke(obj, new object[3] { item, "selection", null });
		}
		return Find(frame, "SELECT_ONE", ((Dictionary<CardModel, string>)Get(obj, "Cards"))[_selectedCards[@int]]);
	}

	private JsonElement SelectReward(RunSimulator sim, JsonElement frame)
	{
		if (_selector == null)
		{
			throw new InvalidOperationException("Combat reward has no solver plan");
		}
		object value = typeof(RunSimulator).GetField("_cardSelector", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).GetValue(sim);
		object obj = Get(value, "PendingRewardCards");
		object obj2 = Get(value, "PendingRewardAlternatives");
		object obj3 = _selector.GetType().GetMethod("GetSelectedCardReward", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).Invoke(_selector, new object[2] { obj, obj2 });
		object? value2 = obj3.GetType().GetField("card", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).GetValue(obj3);
		CardModel key = (CardModel)(((value2 is CardModel) ? value2 : null) ?? throw new InvalidOperationException("Solver requested an unadapted combat reward alternative"));
		object obj4 = Snapshot(sim);
		foreach (object item in (IEnumerable)obj)
		{
			obj4.GetType().GetMethod("AddCard", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).Invoke(obj4, new object[3]
			{
				Get(item, "Card"),
				"reward",
				null
			});
		}
		return Find(frame, "TAKE_CARD_REWARD", ((Dictionary<CardModel, string>)Get(obj4, "Cards"))[key]);
	}

	private object Search(JsonElement request, JsonElement frame)
	{
		if (_poisoned)
		{
			throw new InvalidOperationException("Solver isolation failed; restart this worker");
		}
		if (frame.GetProperty("public").GetProperty("phase").GetString() != "combat" || (frame.GetProperty("public").TryGetProperty("selection_context", out var value) && value.ValueKind == JsonValueKind.Object))
		{
			throw new InvalidOperationException("solver_solve requires a combat play decision");
		}
		if (Type("Entry").GetProperty("Logger", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).GetValue(null) == null)
		{
			Type("Entry").GetProperty("Logger", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).SetValue(null, New("CombatSolverLog", Path.Combine(Path.GetTempPath(), "combat-solver-cli", Environment.ProcessId.ToString())));
		}
		int num = EffectiveBudget(request);
		_effectiveBudgetMs = num;
		Assembly assembly = Assembly.Load("sts2");
		Type type = assembly.GetType("MegaCrit.Sts2.Core.Combat.CombatManager", throwOnError: true);
		object value3 = type.GetProperty("Instance", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).GetValue(null);
		object obj = type.GetMethod("DebugOnlyGetState", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).Invoke(value3, null) ?? throw new InvalidOperationException("No active combat");
		object objA = Call("ContinuationStamp", "CaptureLive", obj);
		try
		{
			object obj2 = Call("CombatRootSnapshot", "Capture", obj, false);
			object obj3 = Call("SolverDisplayNames", "Capture", obj);
			object obj4 = New("BattleDamageSnapshot", 0, 0, 0, Array.Empty<string>(), 0);
			int beam = request.TryGetProperty("beam_width", out var beamValue) ? beamValue.GetInt32() : 60;
			if (beam < 1 || beam > 2048) throw new ArgumentOutOfRangeException("beam_width");
			object obj5 = New("SolverSearchProfile", beam, 120000, 32, 18, 24, num);
			JsonElement value4;
			JsonElement value5;
			string text = (request.TryGetProperty("potion_policy", out value4) ? value4.GetString() : ((request.TryGetProperty("potions", out value5) && value5.GetBoolean()) ? "Smart" : "Disabled"));
			bool flag;
			switch (text)
			{
			case "Smart":
			case "Disabled":
			case "RequireAtLeastOne":
				flag = true;
				break;
			default:
				flag = false;
				break;
			}
			if (!flag)
			{
				throw new ArgumentException("potion_policy must be Smart, Disabled or RequireAtLeastOne");
			}
			object obj6 = Enum("SolverPotionPolicy", text);
			object obj7 = New("PotionStrategySnapshot", obj6, Array.CreateInstance(Type("PotionSlotDirective"), 0));
			Action<string> action = delegate(string s)
			{
				Console.Error.WriteLine(s);
			};
			object obj8 = New("SearchDiagnosticsSink", action, action, null);
			object obj9 = New("SearchPolicySnapshot", obj5, obj6, obj7, false, false, false, false, 1, num, false, null, System.Enum.ToObject(Type("BossHpStrategy"), 0), System.Enum.ToObject(Type("BossHpStrategy"), 0), 0, obj8, New("SearchFramePressureSignal"), New("SearchMemoryPressureSignal"));
			bool portfolio = !request.TryGetProperty("beam_portfolio", out var portfolioValue) || portfolioValue.GetBoolean();
			Type("SearchPolicySnapshot").GetProperty("UseBeamWidthPortfolio", Flags)!.SetValue(obj9, portfolio);
			var combat = (CombatState)obj;
			bool act3Boss = (bool)Call("SearchPolicySnapshot", "IsAct3BossEncounter", combat.RunState.CurrentActIndex, combat.Encounter?.Id.Entry)!;
			Type("SearchPolicySnapshot").GetProperty("Act3BossStrategy", Flags)!.SetValue(obj9, act3Boss);
			// The native coordinator publishes adoptable routes through its interaction state.
			// Ask it to adopt the current turn at the soft deadline, before the hard cancellation.
			object interaction = New("SearchInteractionState");
			Type("SearchPolicySnapshot").GetProperty("Interaction", Flags)!.SetValue(obj9, interaction);
			MethodInfo requestTakeover = Type("SearchInteractionState").GetMethod("RequestApplyCurrentTurn", Flags)!;
			using CancellationTokenSource cancellationTokenSource = new(TimeSpan.FromMilliseconds(num + 10000));
			using Timer softDeadline = new(_ => requestTakeover.Invoke(interaction, null), null, num, Timeout.Infinite);
			_refinementFailure = null;
			object? routeSeed = null;
			Action<object> progress = update =>
			{
				object? seed = Get(update, "RouteAdoptionSeed");
				if (seed != null) routeSeed = seed;
			};
			try
			{
				return Call("CombatSearchCoordinator", "Solve", obj2, obj3, obj4, obj9, cancellationTokenSource.Token, progress);
			}
			catch (TargetInvocationException exception) when (routeSeed != null
				&& exception.InnerException?.GetType().FullName == "CombatSolver.SearchTransitionException")
			{
				// Adopt only a complete native route published before refinement failed.
				// The finally block still audits the live state before this result is returned.
				object completeResult = routeSeed.GetType().GetMethod("Materialize", Flags)!.Invoke(routeSeed, null)!;
				object snapshot = Get(completeResult, "Snapshot")!;
				if ((bool)Get(completeResult, "OnlyDeathRoutesFound")!
					|| !(bool)Get(snapshot, "AllEnemiesDead")!
					|| (bool)Get(snapshot, "PlayerDead")!
					|| Convert.ToInt32(Get(snapshot, "ProjectedPlayerHp")) <= 0) throw;
				_refinementFailure = exception.InnerException.Message;
				Console.Error.WriteLine("[CombatSolverCli] refinement_failed_using_complete_native_route: " + _refinementFailure);
				return completeResult;
			}
		}
		finally
		{
			if (!object.Equals(objA, Call("ContinuationStamp", "CaptureLive", obj)))
			{
				_poisoned = true;
				throw new InvalidOperationException("CombatSolver changed live state while searching; worker quarantined");
			}
		}
	}
}
