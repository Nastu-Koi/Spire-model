using System.Reflection;
using System.Reflection.Emit;
using HarmonyLib;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Monsters;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;

namespace Sts2Headless;

// The game DLL already bypasses most Godot presentation in TestMode. These
// remaining event calls are purely audio/portrait/shake operations. Remove only
// their calls, retaining argument evaluation, RNG, choices and game commands.
internal static class HeadlessEventPresentation
{
    private static bool _installed;
    public static void Install()
    {
        if (_installed) return;
        var harmony = new Harmony("sts2headless.event-presentation");
        var transpiler = new HarmonyMethod(typeof(HeadlessEventPresentation), nameof(RemovePresentationCalls));
        foreach (var eventType in AbstractModelSubtypes.All.Where(t => t.IsSubclassOf(typeof(EventModel))))
        {
            foreach (var type in new[] { eventType }.Concat(eventType.GetNestedTypes(BindingFlags.Public | BindingFlags.NonPublic)))
                foreach (var method in type.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance
                    | BindingFlags.Static | BindingFlags.DeclaredOnly).Where(m => m.GetMethodBody() != null && !m.ContainsGenericParameters))
                {
                    if (eventType.Name == "Trial" && method.Name == "AddVfxAnchoredToPortrait")
                        harmony.Patch(method, prefix: new HarmonyMethod(typeof(HeadlessEventPresentation), nameof(SkipPortraitVfx)));
                    else harmony.Patch(method, transpiler: transpiler);
                }
        }
        // The Kaiser Crab monsters play audio through NAudioManager.Instance, which is
        // deliberately absent in the headless node tree. Remove only those presentation
        // calls; death handling, music progress and combat hooks still run.
        harmony.Patch(typeof(Crusher).GetMethod(nameof(Crusher.BeforeDeath))!, transpiler: transpiler);
        harmony.Patch(typeof(Rocket).GetMethod(nameof(Rocket.BeforeDeath))!, transpiler: transpiler);
        // SoulNexus first unsubscribes its death callback, then clears a Spine
        // animation. Preserve the callback cleanup and its existing null-node
        // branch; only make the missing headless room lookup null-safe.
        harmony.Patch(AccessTools.DeclaredMethod(typeof(SoulNexus), "AfterDeath", [typeof(Creature)])
            ?? throw new MissingMethodException(typeof(SoulNexus).FullName, "AfterDeath"),
            transpiler: new HarmonyMethod(typeof(HeadlessEventPresentation), nameof(GuardSoulNexusRoom)));
        _installed = true;
    }

    private static NCreature? GetCreatureNodeIfPresent(NCombatRoom? room, Creature creature) =>
        room?.GetCreatureNode(creature);

    private static IEnumerable<CodeInstruction> GuardSoulNexusRoom(IEnumerable<CodeInstruction> instructions)
    {
        var lookup = AccessTools.Method(typeof(NCombatRoom), nameof(NCombatRoom.GetCreatureNode), [typeof(Creature)]);
        var guarded = AccessTools.Method(typeof(HeadlessEventPresentation), nameof(GetCreatureNodeIfPresent));
        int replaced = 0;
        foreach (var instruction in instructions)
        {
            if (instruction.opcode == OpCodes.Callvirt && Equals(instruction.operand, lookup))
            {
                // Same arguments/return value; retain all labels and exception blocks.
                instruction.opcode = OpCodes.Call;
                instruction.operand = guarded;
                replaced++;
            }
            yield return instruction;
        }
        if (replaced != 1)
            throw new InvalidOperationException("SoulNexus death presentation changed: expected one room lookup.");
    }

    private static bool SkipPortraitVfx() => false;

    private static bool IsPresentation(MethodInfo method)
    {
        var type = method.DeclaringType?.Name;
        return type == "NDebugAudioManager" && method.Name is "Play" or "Stop"
            || type == "NAudioManager" && method.Name == "PlayOneShot"
            || type == "NGame" && method.Name is "ScreenShake" or "ScreenShakeTrauma" or "ScreenRumble"
            || type == "NEventRoom" && method.Name is "get_Layout" or "SetPortrait"
            || type == "NEventLayout" && method.Name is "RemoveNodesOnPortrait" or "AddVfxAnchoredToPortrait";
    }

    private static IEnumerable<CodeInstruction> RemovePresentationCalls(IEnumerable<CodeInstruction> instructions)
    {
        foreach (var instruction in instructions)
        {
            if (instruction.operand is not MethodInfo method || !IsPresentation(method)
                || (instruction.opcode != OpCodes.Call && instruction.opcode != OpCodes.Callvirt))
            {
                yield return instruction;
                continue;
            }
            int count = method.GetParameters().Length + (method.IsStatic ? 0 : 1);
            var replacement = new List<CodeInstruction>();
            for (int i = 0; i < count; i++) replacement.Add(new(OpCodes.Pop));
            if (method.ReturnType != typeof(void))
            {
                if (!method.ReturnType.IsValueType) replacement.Add(new(OpCodes.Ldnull));
                else if (method.ReturnType == typeof(int)) replacement.Add(new(OpCodes.Ldc_I4_0));
                else throw new InvalidOperationException("Unexpected presentation return type: " + method);
            }
            if (replacement.Count == 0) replacement.Add(new(OpCodes.Nop));
            replacement[0].labels.AddRange(instruction.labels);
            replacement[0].blocks.AddRange(instruction.blocks);
            foreach (var item in replacement) yield return item;
        }
    }
}
