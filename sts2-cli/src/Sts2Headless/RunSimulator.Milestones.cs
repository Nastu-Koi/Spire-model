using System.Collections.Concurrent;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

public partial class RunSimulator
{
    private readonly ConcurrentQueue<object> _protocolEvents = new();
    private readonly HashSet<CombatRoom> _protocolWon = new(ReferenceEqualityComparer.Instance);
    private volatile bool _protocolVictory;
    private volatile bool _protocolLoss;
    private volatile bool _protocolPlayerDied;
    private Creature? _protocolPlayerCreature;
    private int _protocolEncounterSequence;

    private void RegisterProtocolMilestones()
    {
        var manager = CombatManager.Instance;
        manager.CombatWon -= ProtocolCombatWon;
        manager.CombatEnded -= ProtocolCombatEnded;
        manager.CombatWon += ProtocolCombatWon;
        manager.CombatEnded += ProtocolCombatEnded;
        RunManager.Instance.ActionExecutor.AfterActionExecuted -= ProtocolActionEnded;
        RunManager.Instance.ActionExecutor.AfterActionExecuted += ProtocolActionEnded;
        // HP can reach zero before death prevention (Lizard Tail) runs; only the
        // native Died event, raised after every preventer declined, is a death.
        if (_protocolPlayerCreature != null) _protocolPlayerCreature.Died -= ProtocolPlayerDied;
        _protocolPlayerCreature = _runState!.Players[0].Creature;
        _protocolPlayerCreature.Died += ProtocolPlayerDied;
    }

    private void ProtocolPlayerDied(Creature creature) => _protocolPlayerDied = true;

    private void ProtocolActionEnded(GameAction action)
    {
        if (!_protocolEnabled || action.Exception == null) return;
        Log("Protocol game action failed: " + action.Exception);
        _protocolFailure = "native_action_failed";
    }

    private void ProtocolCombatWon(CombatRoom room)
    {
        if (!_protocolEnabled || !ReferenceEquals(room.CombatState.RunState, _runState)) return;
        lock (_protocolWon)
        {
            if (!_protocolWon.Add(room)) return;
            var state = _runState!;
            int act = state.CurrentActIndex + 1;
            bool final = room.RoomType == RoomType.Boss && (state.Map.SecondBossMapPoint is { } second
                ? state.CurrentMapCoord == second.coord : state.CurrentMapCoord == state.Map.BossMapPoint.coord);
            string kind = room.RoomType == RoomType.Boss ? "boss" : room.RoomType == RoomType.Elite ? "elite" : "normal";
            _protocolEvents.Enqueue(new { type = "encounter_completed", encounter_id = $"{_episodeId}:encounter:{++_protocolEncounterSequence}",
                act, kind, result = "victory", final_in_act = final });
            if (act == 3 && final)
            {
                _protocolEvents.Enqueue(new { type = "run_completed", victory = true, act, final_boss_defeated = true });
                _protocolVictory = true;
            }
        }
    }

    private void ProtocolCombatEnded(CombatRoom room)
    {
        if (!_protocolEnabled || !ReferenceEquals(room.CombatState.RunState, _runState)) return;
        lock (_protocolWon)
        {
            if (_protocolWon.Contains(room)) return;
            _protocolEvents.Enqueue(new { type = "run_completed", victory = false,
                act = _runState!.CurrentActIndex + 1, final_boss_defeated = false });
            _protocolLoss = true;
        }
    }

    private Dictionary<string, object?> WithProtocolEvents(Dictionary<string, object?> frame)
    {
        var events = new List<object>();
        while (_protocolEvents.TryDequeue(out var item)) events.Add(item);
        if (events.Count == 0) return frame;
        return new Dictionary<string, object?>(frame) { ["events"] = events.ToArray() };
    }
}
