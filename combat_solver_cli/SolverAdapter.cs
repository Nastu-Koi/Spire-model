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
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Rooms;
using Sts2Headless;

namespace CombatSolverCli;

internal sealed class SolverAdapter(Assembly assembly)
{
	private const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

	private ChoiceTransaction? _selector;

	private List<CardModel>? _selectedCards;

	private string? _selectionId;

	private bool _poisoned;
	private string? _refinementFailure;
	private int _effectiveBudgetMs;
	private bool _finalBossPriced;
	private bool _withinBudget;
	private int _potionReserveMs;
	// The last search a later turn may continue (`reuse_route`), its fight and its request policy.
	private object? _routeSource;
	private CombatState? _routeCombat;
	private string? _routePolicy;
	private readonly Queue<object> _turnPlan = new();
	private int _planTurn = -1;
	private string? _planPolicy;
	// The active transaction holds the choices the route planned for the next turn's setup.
	private bool _selectorPlansSetup;
	// The solver's root of a turn's setup, taken before the setup ran (see BeforeTurnSetup).
	private object? _setupRoot;
	private static SolverAdapter? _current;
	private static bool _setupInstalled;

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

	// Time a Smart search keeps from its potion-free part for the potion audits: the share
	// `potion_reserve` of the budget, when the belt holds a potion and no use is forced.
	private static int PotionReserveMs(JsonElement request, JsonElement frame, int budget)
	{
		double share = request.TryGetProperty("potion_reserve", out var value) ? value.GetDouble() : 0;
		if (share < 0 || share > 0.9) throw new ArgumentOutOfRangeException("potion_reserve");
		bool smart = request.TryGetProperty("potion_policy", out var policy) ? policy.GetString() == "Smart"
			: request.TryGetProperty("potions", out var potions) && potions.GetBoolean();
		bool forced = request.TryGetProperty("potion_directives", out var directives) && directives.GetArrayLength() > 0;
		bool belt = frame.GetProperty("public").GetProperty("entities").EnumerateArray()
			.Any(e => e.TryGetProperty("entity_type", out var type) && type.GetString() == "potion");
		return smart && !forced && belt ? (int)(budget * share) : 0;
	}

	public void PrepareEngine()
	{
		PortfolioDeadline.Install(assembly);
		PotionCosts.Install(assembly);
		PotionReserve.Install(assembly);
		BlockPotionInsertion.Install(assembly);
		FinalReplay.Install(assembly);
		NativeSelectionBridge.Install();
		// The game takes the first thread that asks for its main thread, and the solver captures
		// roots only there: make it this one, before a setup running elsewhere can ask.
		NGame.IsMainThread();
		_current = this;
		if (!_setupInstalled)
		{
			MethodInfo setup = typeof(CombatManager).GetMethods(Flags).Single(m => m.Name == "SetupPlayerTurn" && m.GetParameters().Length == 3);
			new Harmony("combat-solver-cli.turn-setup").Patch(setup, prefix: new HarmonyMethod(typeof(SolverAdapter).GetMethod(nameof(BeforeTurnSetup), Flags)));
			_setupInstalled = true;
		}
		RestoreNativeNeutralize();
	}

	// Restore the native Neutralize action; the CLI legacy prefix bypasses native attack commands.
	// Keep the solver patch audit enabled and remove only this known compatibility prefix.
	// A run begun by a command gets the prefix again; the solver refuses roots while it is there.
	private static void RestoreNativeNeutralize()
	{
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
		(_routeSource, _routeCombat, _routePolicy) = (null, null, null);
		ClearSelection();
	}

	// The rest of the last searched route, when this turn starts in exactly the state the route
	// predicted for it: the solver's own continuation check (ContinuationStamp), as its full-auto
	// mode uses between turns. Null when the route predicted another state or no later turn.
	private object? Continue(CombatState combat, Player player)
	{
		if (_routeSource == null) return null;
		object?[] args = { Call("ContinuationStamp", "CaptureLive", combat), player.Creature.CurrentHp, New("BattleDamageSnapshot", 0, 0, 0, Array.Empty<string>(), 0), null };
		return (bool)_routeSource.GetType().GetMethod("TryCreateContinuation", Flags)!.Invoke(_routeSource, args)! ? args[3] : null;
	}

	private void ClearSelection()
	{
		NativeSelectionBridge.Activate(null);
		// A setup no search was asked to plan (a replaying process, the selection rule outside the
		// worker), or one that left its plan, may still finish its native requests; they no longer
		// concern the worker.
		if (_selector?.AwaitingPlan == true || _selector?.Diverged != null) _selector.Abandon();
		_selector?.Dispose();
		_selector = null;
		_selectedCards = null;
		_selectionId = null;
		_selectorPlansSetup = false;
		_setupRoot = null;
	}

	private string ChoiceKey(CardModel card) => (string)Call("CardChoiceSupport", "ChoiceCardKey", card)!;

	// The solver mod's turn-setup takeover (PlayerTurnSetupCoordinator): a setup that asks for a
	// choice the solver has to make (its own RequiresSolverChoice: Gambling Chip, Toasty Mittens,
	// Toolbox, ...) is searched from the state before the setup ran, with the setup included.
	// The native choice starts before the worker is asked to solve, so here the root is taken and
	// a transaction opened that records the native requests; Step searches when the choice shows.
	private static void BeforeTurnSetup(CombatManager __instance, object[] __args)
	{
		try
		{
			_current?.CaptureTurnSetup(__instance, (Player)__args[1]);
		}
		catch (Exception exception)
		{
			// Without the root the choice is answered outside the solver, as before.
			Console.Error.WriteLine("[CombatSolverCli] turn_setup_capture_failed: " + exception);
		}
	}

	private void CaptureTurnSetup(CombatManager manager, Player player)
	{
		CombatState? combat = manager.DebugOnlyGetState();
		if (combat == null || player.PlayerCombatState == null || Convert.ToInt32(player.PlayerCombatState.Phase) != 1) return;
		if (_selector != null && _selectorPlansSetup) return;
		if (!(bool)Call("PlayerTurnSetupCoordinator", "RequiresSolverChoice", player)!) return;
		// Otherwise the choice is answered outside the solver, as before.
		// Vakuu plays the first turn with choices of its own, and the solver cannot search a setup
		// that ends in that turn ("Cannot fork before a requested player turn end is resolved").
		if (player.PlayerCombatState.TurnNumber <= 1 && player.Relics.Any(r => r is WhisperingEarring && !r.IsMelted))
		{
			Console.Error.WriteLine("[CombatSolverCli] turn_setup_not_taken reason=whispering_earring");
			return;
		}
		if (!NGame.IsMainThread())
		{
			Console.Error.WriteLine("[CombatSolverCli] turn_setup_not_taken reason=off_main_thread");
			return;
		}
		if (_selector != null && !_selector.Settled)
		{
			Console.Error.WriteLine("[CombatSolverCli] turn_setup_not_taken reason=unsettled_choices");
			return;
		}
		// A resumed fight begins inside the command that began its run, before PrepareEngine ran after it.
		RestoreNativeNeutralize();
		object root = Call("CombatRootSnapshot", "Capture", combat, false)!;
		_selector?.FinishAction();
		ClearSelection();
		_setupRoot = root;
		_selector = new ChoiceTransaction(player, ChoiceKey);
		NativeSelectionBridge.Activate(_selector);
	}

	public void ObserveBoundary(JsonElement frame)
	{
		if (_selector == null) return;
		_selector.Check();
		if (frame.GetProperty("boundary").GetString() == "waiting") return;
		var pub = frame.GetProperty("public");
		if (pub.TryGetProperty("selection_context", out var selection) && selection.ValueKind == JsonValueKind.Object) return;
		bool combatEnded = !CombatManager.Instance.IsInProgress
			|| frame.GetProperty("boundary").GetString() == "terminal" && CombatManager.Instance.IsOverOrEnding;
		if (!combatEnded && (pub.GetProperty("phase").GetString() is "card_reward" or "bundle_select")) return;
		_selector.FinishAction(combatEnded);
		ClearSelection();
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

	private object Describe(object result, bool turnSetup = false)
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
			potion_costs = PotionCosts.Current,
			// What the search kept from its potion-free part for the potion audits (PotionReserve).
			potion_reserve_ms = _potionReserveMs,
			// What the route's potions cost at those prices, as the solver counted it.
			potion_strategic_cost = ((IEnumerable)Get(Get(result, "PotionStrategicCostByTurn"), "Values")).Cast<int>().Sum(),
			final_boss_priced = _finalBossPriced,
			// A continued route: the turn the search that found it started on.
			reused_from_turn = (bool)Get(result, "WasReused")! ? Get(result, "ReusedFromTurn") : null,
			// The last search ended before its budget (for a continued turn: the last search of the fight).
			within_budget = _withinBudget,
			// A search of the turn's setup, made for its choices (BeforeTurnSetup).
			turn_setup = turnSetup ? true : (bool?)null,
			boss_combat = IsBossCombat()
		};
	}

	public object Step(JsonElement request, JsonElement frame, object simulator)
	{
		ObserveBoundary(frame);
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
		object search = null;
		bool asksChoice = (frame.GetProperty("public").TryGetProperty("selection_context", out var shown) && shown.ValueKind == JsonValueKind.Object)
			|| frame.GetProperty("public").GetProperty("phase").GetString() is "card_reward" or "bundle_select";
		if (asksChoice && _selector != null && _selector.AwaitingPlan)
		{
			object setup;
			try
			{
				setup = Search(request, frame, _setupRoot);
			}
			catch (Exception exception) when (!_poisoned)
			{
				// The solver could not play this setup: the choice is answered outside it, as before.
				Console.Error.WriteLine("[CombatSolverCli] turn_setup_not_taken reason=search_failed: " + (exception.InnerException ?? exception).Message);
				ClearSelection();
				return new { type = "solver_selection_required", frame = frame };
			}
			search = Describe(setup, turnSetup: true);
			_selector.Supply(((IEnumerable)Get(setup, "TurnSetupChoices")!).Cast<object>());
		}
		if (asksChoice && _selector?.Diverged != null)
		{
			// The native setup offered a choice its plan does not have (ChoiceTransaction.Match):
			// the rule answers it, and the turn is searched again once it is played.
			ClearSelection();
			return new { type = "solver_selection_required", frame = frame };
		}
		// Entry effects and replayed actions can expose choices without a solver plan.
		// Return the unchanged boundary so the run search can retain every legal branch.
		if (_selector == null &&
			((frame.GetProperty("public").TryGetProperty("selection_context", out var selection) && selection.ValueKind == JsonValueKind.Object)
			 || frame.GetProperty("public").GetProperty("phase").GetString() is "card_reward" or "bundle_select"))
		{
			return new { type = "solver_selection_required", frame = frame };
		}
		Player player = val.Players.Single();
		JsonElement jsonElement;
		if (frame.GetProperty("public").TryGetProperty("selection_context", out var value) && value.ValueKind == JsonValueKind.Object)
		{
			jsonElement = SelectCandidate(runSimulator, frame, value);
		}
		else if (frame.GetProperty("public").GetProperty("phase").GetString() == "bundle_select")
		{
			jsonElement = SelectBundle(runSimulator, frame);
		}
		else if (frame.GetProperty("public").GetProperty("phase").GetString() == "card_reward")
		{
			jsonElement = SelectReward(runSimulator, frame);
		}
		else
		{
			if (_selector != null)
			{
				_selector.FinishAction();
				ClearSelection();
			}
			bool reuse = request.TryGetProperty("reuse_turn_plan", out var reuseValue) && reuseValue.GetBoolean();
			bool reuseRoute = request.TryGetProperty("reuse_route", out var reuseRouteValue) && reuseRouteValue.GetBoolean();
			// `reuse_within_budget`: continue only routes whose search ended before its budget.
			bool withinBudgetOnly = request.TryGetProperty("reuse_within_budget", out var withinValue) && withinValue.GetBoolean();
			int turn = player.PlayerCombatState.TurnNumber;
			string policy = string.Join("|", new[] { "budget_ms", "boss_budget_ms", "potions", "potion_policy", "potion_directives", "potion_costs", "final_boss_potion_costs", "potion_reserve", "beam_width", "beam_portfolio", "reuse_route", "reuse_within_budget" }
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
				object? result = reuseRoute && _routePolicy == policy && ReferenceEquals(_routeCombat, val) ? Continue(val, player) : null;
				if (result == null)
				{
					result = Search(request, frame);
					// A route cut at the deadline covers the current turn only; it predicts no later turn.
					_routeSource = Get(result, "ResultScope")!.ToString() == "CurrentTurnAdoption" || withinBudgetOnly && !_withinBudget
						? null : result;
					(_routeCombat, _routePolicy) = (val, policy);
				}
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
			int planned = list.Count;
			if (Get(obj2, "Kind").ToString() == "EndTurn" || (bool)Get(obj2, "EndsPlayerTurn"))
			{
				list.AddRange((((IEnumerable)Get(obj2, "TurnStartChoices")) ?? Array.Empty<object>()).Cast<object>());
			}
			_selector = new ChoiceTransaction(list, player, ChoiceKey, setupFrom: planned);
			_selectorPlansSetup = list.Count > planned;
			NativeSelectionBridge.Activate(_selector);
		}
		JsonElement property = frame.GetProperty("routing");
		JsonElement value2;
		Dictionary<string, object> frame2 = runSimulator.ExecuteCandidate(property.GetProperty("decision_id").GetString(), property.GetProperty("state_version").GetInt64(), jsonElement.GetProperty("candidate_ref").GetString(), property.TryGetProperty("selection_revision", out value2) ? new long?(value2.GetInt64()) : ((long?)null));
		ObserveBoundary(JsonSerializer.SerializeToElement(frame2));
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
			_selectedCards = _selector.Select(list, context.GetProperty("min_total").GetInt32(),
				context.GetProperty("max_total").GetInt32()).ToList();
			_selectionId = text;
		}
		int @int = context.GetProperty("selected_count").GetInt32();
		if (@int == _selectedCards.Count)
		{
			if (_selectedCards.Count == 0 && context.GetProperty("min_total").GetInt32() > 0) return Find(frame, "CANCEL");
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

	private JsonElement SelectBundle(RunSimulator sim, JsonElement frame)
	{
		var bundles = (IReadOnlyList<IReadOnlyList<CardModel>>)typeof(RunSimulator).GetField("_pendingBundles", Flags)!.GetValue(sim)!;
		var selected = _selector!.Select(bundles.SelectMany(b => b), 0, int.MaxValue);
		var indices = Enumerable.Range(0, bundles.Count).Where(i => bundles[i].SequenceEqual(selected)).ToArray();
		if (indices.Length != 1) throw new InvalidOperationException("Planned bundle has no unique native match");
		return Find(frame, "SELECT_BUNDLE", $"bundle:{indices[0]}");
	}

	private JsonElement SelectReward(RunSimulator sim, JsonElement frame)
	{
		if (_selector == null)
		{
			throw new InvalidOperationException("Combat reward has no solver plan");
		}
		object value = typeof(RunSimulator).GetField("_cardSelector", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).GetValue(sim);
		object obj = Get(value, "PendingRewardCards");
		var chosen = _selector.Select(((IEnumerable)obj).Cast<object>().Select(item => (CardModel)Get(item, "Card")!), 0, 1);
		if (chosen.Length == 0)
		{
			var alternatives = ((IEnumerable)Get(value, "PendingRewardAlternatives")!).Cast<object>().ToArray();
			int skip = Array.FindIndex(alternatives, a => Get(a, "OptionId")?.ToString()?.Equals("Skip", StringComparison.OrdinalIgnoreCase) == true);
			if (skip < 0) throw new InvalidOperationException("Planned empty reward has no legal skip");
			return Find(frame, "CHOOSE_REWARD_ALTERNATIVE", $"reward_alternative:{skip}");
		}
		CardModel key = chosen.Single();
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

	// `setupRoot`: a root taken before a turn's setup ran; the search plays the setup as well.
	private object Search(JsonElement request, JsonElement frame, object? setupRoot = null)
	{
		ObserveBoundary(frame);
		if (_selector != null && setupRoot == null) throw new InvalidOperationException("Cannot search during a pending execution selection");
		if (_poisoned)
		{
			throw new InvalidOperationException("Solver isolation failed; restart this worker");
		}
		if (setupRoot == null && (frame.GetProperty("public").GetProperty("phase").GetString() != "combat" || (frame.GetProperty("public").TryGetProperty("selection_context", out var value) && value.ValueKind == JsonValueKind.Object)))
		{
			throw new InvalidOperationException("solver_solve requires a combat play decision");
		}
		if (Type("Entry").GetProperty("Logger", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).GetValue(null) == null)
		{
			Type("Entry").GetProperty("Logger", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).SetValue(null, New("CombatSolverLog", Path.Combine(Path.GetTempPath(), "combat-solver-cli", Environment.ProcessId.ToString())));
		}
		int num = EffectiveBudget(request);
		_effectiveBudgetMs = num;
		int reserve = PotionReserveMs(request, frame, num);
		_potionReserveMs = reserve;
		int[] costs = PotionCosts.Read(request, "potion_costs");
		int[]? finalCosts = request.TryGetProperty("final_boss_potion_costs", out _) ? PotionCosts.Read(request, "final_boss_potion_costs") : null;
		Assembly assembly = Assembly.Load("sts2");
		Type type = assembly.GetType("MegaCrit.Sts2.Core.Combat.CombatManager", throwOnError: true);
		object value3 = type.GetProperty("Instance", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).GetValue(null);
		object obj = type.GetMethod("DebugOnlyGetState", BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic).Invoke(value3, null) ?? throw new InvalidOperationException("No active combat");
		// The run's last fight: the solver values neither HP nor potions past it, so its own policy
		// spends a potion there only to win. `final_boss_potion_costs` prices potions there as in any
		// other fight and counts the HP they save in full (BossHpStrategy.MinimizeHpLoss).
		_finalBossPriced = finalCosts != null && Call("ActEndingBossPolicy", "ResolveHpRelief", obj)!.ToString() == "RunEnding";
		PotionCosts.Set(_finalBossPriced ? finalCosts! : costs);
		object objA = Call("ContinuationStamp", "CaptureLive", obj);
		try
		{
			object obj2 = setupRoot ?? Call("CombatRootSnapshot", "Capture", obj, false);
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
			// A directive names a belt slot and the potion in it: Force (the route must use it), Disabled or Smart.
			List<JsonElement> given = request.TryGetProperty("potion_directives", out var directivesValue) ? directivesValue.EnumerateArray().ToList() : new List<JsonElement>();
			Array directives = Array.CreateInstance(Type("PotionSlotDirective"), given.Count);
			for (int i = 0; i < given.Count; i++)
			{
				directives.SetValue(New("PotionSlotDirective", given[i].GetProperty("slot").GetInt32(), given[i].GetProperty("potion").GetString(), Enum("SolverPotionDirective", given[i].GetProperty("directive").GetString()!)), i);
			}
			object obj7 = New("PotionStrategySnapshot", obj6, directives);
			Action<string> action = delegate(string s)
			{
				Console.Error.WriteLine(s);
			};
			object obj8 = New("SearchDiagnosticsSink", action, action, null);
			object obj9 = New("SearchPolicySnapshot", obj5, obj6, obj7, false, false, true, false, 1, num, setupRoot != null, null, System.Enum.ToObject(Type("BossHpStrategy"), 0), Enum("BossHpStrategy", _finalBossPriced ? "MinimizeHpLoss" : "ProgressionFirst"), 0, obj8, New("SearchFramePressureSignal"), New("SearchMemoryPressureSignal"));
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
			_withinBudget = false;
			object? routeSeed = null;
			Action<object> progress = update =>
			{
				object? seed = Get(update, "RouteAdoptionSeed");
				if (seed != null) routeSeed = seed;
			};
			PotionReserve.Begin(num, reserve);
			try
			{
				object found = Call("CombatSearchCoordinator", "Solve", obj2, obj3, obj4, obj9, cancellationTokenSource.Token, progress)!;
				// No deadline reached the search: the takeover the timer asks for at the budget is still
				// unasked, so the potion audits, cancelled no earlier than that, ran to their end too.
				_withinBudget = Get(interaction, "CurrentTakeoverRequest") == null;
				return found;
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
			PotionReserve.End();
			if (!object.Equals(objA, Call("ContinuationStamp", "CaptureLive", obj)))
			{
				_poisoned = true;
				throw new InvalidOperationException("CombatSolver changed live state while searching; worker quarantined");
			}
		}
	}
}
