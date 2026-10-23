using System.Runtime.CompilerServices;
using HarmonyLib;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace Sts2Headless;

// The run as it stood when a fight's room was entered, before the room's own effects:
// relics at the start of combat (Blood Vial's heal, Maw Bank's gold, Petrified Toad's
// potion, Ember Tea's count), the enemies' HP and the opening shuffle. A continue save
// written during the fight is this run (SaveCheckpoint), so resume_fight enters the
// fight as it began and its start happens once. The game too saves a map travel's run
// before the room is entered; the headless run does not save on its own.
internal static class RoomEntrySave
{
    private static readonly ConditionalWeakTable<AbstractRoom, string> Saves = new();
    private static bool _installed;

    public static void Install()
    {
        if (_installed) return;
        new Harmony("sts2headless.room-entry-save").Patch(
            AccessTools.DeclaredMethod(typeof(RunManager), nameof(RunManager.EnterRoom))
                ?? throw new MissingMethodException(typeof(RunManager).FullName, nameof(RunManager.EnterRoom)),
            prefix: new HarmonyMethod(typeof(RoomEntrySave), nameof(Entering)));
        _installed = true;
    }

    // Serialized at once: the run's serializable objects are not guaranteed to be
    // detached from the live state. A run that cannot be serialized still enters the
    // room; a save written in it then fails the same way (SaveCheckpoint).
    private static void Entering(RunManager __instance, AbstractRoom room)
    {
        if (room is not CombatRoom || __instance.DebugOnlyGetState() == null) return;
        try
        {
            Saves.AddOrUpdate(room, SaveManager.ToJson(__instance.ToSave(null)));
        }
        catch (Exception ex)
        {
            Saves.Remove(room);
            Console.Error.WriteLine($"[WARN] RoomEntrySave: {ex.Message}");
        }
    }

    /// The run as it stood when the room was entered; null for a room not entered
    /// with RunManager.EnterRoom (an event's fight) and for rooms other than fights.
    public static string? For(AbstractRoom room) => Saves.TryGetValue(room, out var save) ? save : null;
}
