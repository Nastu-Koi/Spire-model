using System.Security.Cryptography;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Rooms;

namespace Sts2Headless;

public partial class RunSimulator
{
    private readonly DecisionGate _decisionGate = new();
    private string _episodeId = Guid.NewGuid().ToString("N");
    private int? _protocolAscension;
    private string? _protocolFailure;
    private long _decisionSequence;
    private long _selectionBaseVersion;
    private string? _protocolSelectionId;
    private bool _protocolTrainingRun;
    private static readonly Lazy<Dictionary<string, object?>> ProtocolContract = new(() => new()
    {
        ["adapter_version"] = "engine-candidates-v1",
        ["observation_schema"] = "public-state-v1",
        ["action_schema"] = "candidate-v0",
        ["auto_advance_version"] = "pending-continuation-v0",
        ["fixed_ascension"] = 10,
        ["training_ready"] = false,
        ["game_assembly_sha256"] = AssemblyHash(typeof(RunState)),
        ["adapter_assembly_sha256"] = AssemblyHash(typeof(RunSimulator)),
        ["effect_coverage"] = "typed-public-fields-with-explicit-opaque-rules",
        ["supported_interactions"] = new[] { "combat", "map", "event", "rest_site", "shop", "treasure",
            "rewards", "card_reward", "card_select", "bundle_select", "crystal_sphere" },
    });

    private Dictionary<string, object?> CurrentProtocolContract => new(ProtocolContract.Value)
    {
        ["training_ready"] = _protocolTrainingRun,
        ["debug_mutations_allowed"] = false,
    };

    private static string AssemblyHash(Type type) =>
        Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(type.Assembly.Location))).ToLowerInvariant();

    private void ResetDecisionProtocol(int? ascension)
    {
        _episodeId = Guid.NewGuid().ToString("N");
        _protocolAscension = ascension;
        _protocolFailure = null;
        _protocolSelectionId = null;
        _decisionSequence = 0;
        _protocolMapVisible = false;
        _protocolEnabled = false;
        _protocolTrainingRun = false;
        _protocolRewardsOffered.Clear();
        _protocolOpenedChests.Clear();
        _protocolFinishedChests.Clear();
        _protocolClosedMerchants.Clear();
        _protocolCompletedRestSites.Clear();
        _protocolWon.Clear();
        _protocolEvents.Clear();
        _protocolVictory = false;
        _protocolLoss = false;
        _protocolEncounterSequence = 0;
        _decisionGate.Invalidate();
    }

    // Debug/legacy mutations cannot leave a live candidate handle behind.
    public void InvalidateDecisionProtocol()
    {
        _protocolTrainingRun = false;
        _decisionGate.Invalidate();
        _protocolSelectionId = null;
        _protocolMapVisible = false;
    }

    private static Dictionary<string, object?> PublicSelectionCard(CardModel card) => new()
    {
        ["entity_type"] = "card",
        ["content_id"] = card.Id.ToString(),
        ["cost"] = card.EnergyCost?.GetResolved() ?? 0,
        ["card_type"] = card.Type.ToString(),
        ["rarity"] = card.Rarity.ToString(),
        ["upgraded"] = card.IsUpgraded,
        ["upgrade_level"] = card.CurrentUpgradeLevel,
        ["x_cost"] = card.EnergyCost?.CostsX ?? false,
        ["star_cost"] = card.CurrentStarCost,
        ["base_star_cost"] = card.BaseStarCost,
        ["stars_x"] = card.HasStarCostX,
        ["retain"] = !card.IsCanonical && card.ShouldRetainThisTurn,
        ["exhaust_on_next_play"] = card.ExhaustOnNextPlay,
        ["rider_effect"] = card is MegaCrit.Sts2.Core.Models.Cards.MadScience mad
            ? mad.TinkerTimeRider.ToString() : null,
        ["keywords"] = card.Keywords.Select(k => k.ToString()).Order().ToArray(),
        ["stats"] = card.DynamicVars.Values.ToDictionary(v => v.Name, v => (object?)v.BaseValue),
        ["enchantment"] = card.Enchantment?.Id.ToString(),
        ["affliction"] = card.Affliction?.Id.ToString(),
        ["target_type"] = card.TargetType.ToString(),
        ["effect_coverage"] = "opaque",
        ["semantic_program"] = OpaqueProgram(card.Id.ToString()),
    };

    private Dictionary<string, object?> ProtocolError(string code) => new()
    {
        ["type"] = "decision_frame",
        ["contract"] = CurrentProtocolContract,
        ["boundary"] = "error",
        ["error"] = new { code },
        ["routing"] = new { episode_id = _episodeId, state_version = _decisionGate.StateVersion },
        ["legal"] = new { candidates = Array.Empty<object>() },
        ["events"] = Array.Empty<object>(),
    };

    public Dictionary<string, object?> AdvanceToBoundary()
    {
        if (_protocolFailure != null) return ProtocolError(_protocolFailure);
        if (_runState == null) return ProtocolError("no_active_run");
        if (_protocolAscension != 10) return ProtocolError("unverified_run_contract");
        if (!_protocolEnabled)
        {
            _protocolEnabled = true;
            MegaCrit.Sts2.Core.Rewards.RewardsSet.testSelector = SelectProtocolRewards;
            RegisterProtocolMilestones();
            InstallProtocolUiBridge();
        }
        if (_decisionGate.Frame != null)
            return _decisionGate.Frame; // Queries never resume an already published decision.

        try
        {
            // Resume only the already committed parent operation. In particular,
            // do NOT call legacy DetectDecisionPoint: it auto-claims some rewards.
            _syncCtx.Pump();
            _protocolUiOperation.WaitForBoundary(_syncCtx.Pump,
                () => _cardSelector.HasPending || _cardSelector.HasPendingReward || HasProtocolRewardMenu,
                TimeSpan.FromSeconds(3));
            if (!HasPendingInteraction()) WaitForActionExecutor();
            WaitForPendingOperation();
            if (_protocolFailure != null) return ProtocolError(_protocolFailure);
            return WithProtocolEvents(PublishProtocolBoundary());
        }
        catch (TimeoutException)
        {
            return WithProtocolEvents(NonDecisionBoundary("waiting"));
        }
        catch (Exception ex)
        {
            Log("Protocol advance failed: " + ex);
            _protocolFailure = "engine_operation_failed";
            return ProtocolError(_protocolFailure);
        }
    }

    private Dictionary<string, object?> NonDecisionBoundary(string boundary, object? outcome = null) => new()
    {
        ["type"] = "decision_frame",
        ["contract"] = CurrentProtocolContract,
        ["boundary"] = boundary,
        ["routing"] = new { episode_id = _episodeId, state_version = _decisionGate.StateVersion },
        ["public"] = new { phase = boundary, outcome },
        ["legal"] = new { candidates = Array.Empty<object>() },
        ["events"] = Array.Empty<object>(),
    };

    private Dictionary<string, object?> PublishProtocolBoundary()
    {
        var player = _runState!.Players[0];
        if (_protocolVictory || _protocolLoss || player.Creature.IsDead || RunManager.Instance.IsAbandoned)
            return NonDecisionBoundary("terminal", new { victory = _protocolVictory });
        if (RunManager.Instance.IsGameOver) return ProtocolError("unconfirmed_terminal_outcome");
        if (_cardSelector.HasPendingReward)
            return PublishCardRewardBoundary();
        if (_pendingBundles != null && _pendingBundleTcs is { Task.IsCompleted: false })
            return PublishBundleBoundary();
        if (_cardSelector.HasPending && _cardSelector.Session is { } session)
            return PublishSelectionBoundary(session);
        if (_protocolRewards is { Busy: false } menu) return PublishRewardsBoundary(menu);
        if (HasProtocolCrystal) return PublishCrystalBoundary(_protocolCrystal!);
        if (_pendingOperation.IsActive || RunManager.Instance.ActionExecutor.IsRunning
            || (CombatManager.Instance.IsInProgress && !IsPlayPhase()))
            return NonDecisionBoundary("waiting");
        if (_protocolMapVisible || _runState.CurrentRoom is null or MapRoom) return PublishMapBoundary();
        return _runState.CurrentRoom switch
        {
            CombatRoom room when CombatManager.Instance.IsInProgress => PublishCombatBoundary(),
            CombatRoom room => PublishPostCombatBoundary(room),
            EventRoom => PublishEventBoundary(),
            RestSiteRoom room => PublishRestBoundary(room),
            MerchantRoom room => PublishMerchantBoundary(room),
            TreasureRoom room => PublishTreasureBoundary(room),
            _ => ProtocolError("unsupported_interaction:" + _runState.CurrentRoom.GetType().Name),
        };
    }

    private Dictionary<string, object?> PublishSelectionBoundary(SelectionSession session)
    {
        if (_protocolSelectionId != session.Id)
        {
            _protocolSelectionId = session.Id;
            _selectionBaseVersion = _decisionGate.StateVersion;
        }
        // Sort by public content, never by engine instance IDs or hidden pile order.
        var options = _cardSelector.PendingOptions!;
        var bank = options.Select((card, index) => (index, entity: PublicSelectionCard(card)))
            .OrderBy(pair => System.Text.Json.JsonSerializer.Serialize(pair.entity), StringComparer.Ordinal).ToList();
        var snapshot = BuildPublicSnapshot();
        var refs = bank.ToDictionary(pair => pair.index,
            pair => snapshot.AddCard(options[pair.index], "selection"));
        var bindings = new List<CandidateBinding>();
        var metadata = _cardSelector.Metadata;
        var legalItems = session.LegalItems().ToHashSet();
        foreach (var pair in bank.Where(pair => legalItems.Contains(pair.index)))
        {
            var index = pair.index;
            bindings.Add(new CandidateBinding(Candidate("SELECT_ONE", refs[index], refs[index]),
                () => ReferenceEquals(_cardSelector.Session, session) && session.LegalItems().Contains(index),
                () => session.Select(index)));
        }
        if (session.CanFinish || session.HasUniqueCompletion)
            bindings.Add(new CandidateBinding(Candidate("FINISH_SELECTION", "stop"),
                () => ReferenceEquals(_cardSelector.Session, session) && (session.CanFinish || session.HasUniqueCompletion),
                _cardSelector.FinishSession));
        bool canCancel = metadata.Cancelable;
        if (canCancel)
            bindings.Add(new(Candidate("CANCEL", "cancel"),
                () => ReferenceEquals(_cardSelector.Session, session) && _cardSelector.Metadata.Cancelable,
                _cardSelector.CancelSession));

        var context = new Dictionary<string, object?>
        {
            ["mode"] = "buffered",
            ["operation"] = metadata.Operation,
            ["source"] = metadata.Source,
            ["destination"] = metadata.Destination,
            ["selected_refs"] = session.Selected.Select(i => refs[i]).ToArray(),
            ["selected_count"] = session.Selected.Count,
            ["min_total"] = session.Min,
            ["max_total"] = session.Max,
            ["remaining_required"] = Math.Max(0, session.Min - session.Selected.Count),
            ["can_finish"] = session.CanFinish || session.HasUniqueCompletion,
            ["can_skip"] = false, // Empty FINISH is the only empty result of this selector.
            ["can_cancel"] = canCancel,
            ["repetition_allowed"] = false,
            ["order_matters"] = session.OrderMatters,
            ["known_masks"] = new { operation = metadata.Operation != "unknown", source = metadata.Source != "unknown",
                destination = metadata.Destination != "unknown", order_matters = false },
        };
        var slots = refs.Values.Select(reference => Candidate("SELECT_ONE", reference, reference))
            .Append(Candidate("FINISH_SELECTION", "stop"))
            .Concat(metadata.Cancelable ? new[] { Candidate("CANCEL", "cancel") } : []).ToArray();
        return PublishDecision("card_select", snapshot.Entities, bindings, context, slots, session.Id, session.Revision,
            snapshot.Relations);
    }

    private Dictionary<string, object?> PublishBundleBoundary()
    {
        var request = _pendingBundleTcs!;
        var bundles = _pendingBundles!;
        var snapshot = BuildPublicSnapshot();
        var entities = bundles.Select((cards, i) => new Dictionary<string, object?>
        {
            ["ref"] = $"bundle:{i}", ["entity_type"] = "bundle",
            ["cards"] = cards.Select(PublicSelectionCard).ToArray(),
        }).ToList();
        var bindings = Enumerable.Range(0, bundles.Count).Select(i => new CandidateBinding(
            Candidate("SELECT_BUNDLE", $"bundle:{i}", $"bundle:{i}"),
            () => ReferenceEquals(_pendingBundleTcs, request) && !request.Task.IsCompleted,
            () =>
            {
                _pendingBundleTcs = null;
                _pendingBundles = null;
                if (!request.TrySetResult(bundles[i])) throw new InvalidOperationException("Bundle already resolved.");
            })).ToList();
        snapshot.Entities.AddRange(entities);
        return PublishSnapshot("bundle_select", snapshot, bindings);
    }

    private static Dictionary<string, object?> Candidate(string verb, string slot, string? source = null, string? target = null) => new()
    {
        ["verb"] = verb, ["decoder_slot_ref"] = slot,
        ["source_refs"] = source == null ? Array.Empty<string>() : new[] { source },
        ["target_refs"] = target == null ? Array.Empty<string>() : new[] { target },
        ["effect_coverage"] = "opaque",
    };

    private Dictionary<string, object?> PublishDecision(string phase,
        List<Dictionary<string, object?>> entities, List<CandidateBinding> bindings,
        object? context = null, object? bank = null, string? selectionId = null, long? revision = null,
        object? relations = null)
    {
        if (bindings.Count == 0) return ProtocolError("empty_legal_set");
        var decisionId = $"{_episodeId}:{++_decisionSequence}";
        var frame = new Dictionary<string, object?>
        {
            ["type"] = "decision_frame", ["contract"] = CurrentProtocolContract, ["boundary"] = "decision",
            ["routing"] = new
            {
                episode_id = _episodeId, decision_id = decisionId, state_version = _decisionGate.StateVersion,
                selection_id = selectionId, selection_revision = revision,
                base_public_version = selectionId == null ? _decisionGate.StateVersion : _selectionBaseVersion,
                action_bank_version = selectionId ?? decisionId,
            },
            ["public"] = new
            {
                phase, entities, relations = relations ?? Array.Empty<object>(), memory = Array.Empty<object>(),
                selection_context = context, decoder_bank = bank,
            },
            ["legal"] = new { candidates = bindings.Select((binding, i) => new Dictionary<string, object?>(binding.Public)
                { ["candidate_ref"] = $"c{i}" }).ToArray() },
            ["events"] = Array.Empty<object>(),
        };
        _decisionGate.Publish(frame, decisionId, revision, bindings);
        return frame;
    }

    public Dictionary<string, object?> ExecuteCandidate(string decisionId, long stateVersion,
        string candidateRef, long? selectionRevision)
    {
        var error = _decisionGate.Validate(decisionId, stateVersion, candidateRef, selectionRevision);
        if (error != null) return ProtocolError(error);
        try
        {
            var action = _decisionGate.Consume(candidateRef);
            var next = AdvanceToBoundary();
            // Events belong to this response only. Reading the next cached frame
            // again must not deliver the same transition twice.
            return new Dictionary<string, object?>(next)
            {
                ["events"] = ((IEnumerable<object>)next["events"]!).Prepend(new { type = "candidate_executed", decision_id = decisionId,
                    state_version = stateVersion, candidate_ref = candidateRef, action }).ToArray(),
            };
        }
        catch (Exception ex)
        {
            Log("Protocol execution failed: " + ex);
            _protocolFailure = "candidate_execution_failed";
            return ProtocolError(_protocolFailure);
        }
    }
}
