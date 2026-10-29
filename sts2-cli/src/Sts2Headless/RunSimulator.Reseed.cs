using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Saves;

namespace Sts2Headless;

public partial class RunSimulator
{
    // Another future from the same state: every stream of the run's and each player's
    // random sets takes the state of a fresh generator seeded by `seed` and its name,
    // loaded in place (the game's own references keep their generators). Counters and
    // the run's seed string stay; nothing public changes, so the decision in progress
    // stays valid. What the run rolled up front (the acts' encounters and events, the
    // bosses, the map, the shared relic bag) and generators kept by relics are left as
    // they are.
    public Dictionary<string, object?> Reseed(ulong seed)
    {
        try
        {
            if (_runState == null) return Error("No run in progress");
            var run = _runState.Rng.ToSerializable();
            foreach (var (type, state) in run.Rngs)
                Refill(state, seed, $"run:{type}");
            _runState.Rng.LoadFromSerializable(run);
            var players = 0;
            foreach (var player in _runState.Players)
            {
                var own = player.PlayerRng.ToSerializable();
                foreach (var (type, state) in own.Rngs)
                    Refill(state, seed, $"player{players}:{type}");
                player.PlayerRng.LoadFromSerializable(own);
                players++;
            }
            Log($"Reseed: {run.Rngs.Count} run streams, {players} players, seed={seed}");
            return new Dictionary<string, object?>
            {
                ["type"] = "ok",
                ["run_streams"] = run.Rngs.Count,
                ["players"] = players,
                ["shuffle_state0"] = _runState.Rng.Shuffle.ToSerializable().state0.ToString(),
            };
        }
        catch (Exception ex) { return ErrorWithTrace("Reseed failed", ex); }
    }

    private static void Refill(SerializableRng state, ulong seed, string name)
    {
        var fresh = new Rng(seed, name).ToSerializable();
        state.state0 = fresh.state0;
        state.state1 = fresh.state1;
        state.state2 = fresh.state2;
        state.state3 = fresh.state3;
    }
}
