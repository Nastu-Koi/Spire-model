using HarmonyLib;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;

namespace Sts2Headless;

public partial class RunSimulator
{
    private volatile CrystalSphereMinigame? _protocolCrystal;
    private readonly PendingOperation _protocolUiOperation = new();
    private static RunSimulator? _protocolUiOwner;
    private static bool _protocolUiPatched;
    private bool HasProtocolCrystal => _protocolCrystal is { IsFinished: false } && !_protocolUiOperation.IsActive;

    private void InstallProtocolUiBridge()
    {
        _protocolUiOwner = this;
        InstallPotionPresentationBridge();
        if (_protocolUiPatched) return;
        new Harmony("sts2headless.protocol-ui").Patch(
            AccessTools.Method(typeof(NCrystalSphereScreen), nameof(NCrystalSphereScreen.ShowScreen)),
            prefix: new HarmonyMethod(typeof(RunSimulator), nameof(ShowProtocolCrystal)));
        _protocolUiPatched = true;
    }

    private static bool ShowProtocolCrystal(CrystalSphereMinigame grid, ref NCrystalSphereScreen? __result)
    {
        var owner = _protocolUiOwner;
        if (owner == null || !owner._protocolEnabled) return true;
        owner._protocolCrystal = grid;
        grid.Finished += () =>
        {
            owner._protocolCrystal = null;
            owner._pendingOperation.NotifyProgress();
        };
        owner._pendingOperation.NotifyProgress();
        __result = null;
        return false; // Replace only the unavailable view; PlayMinigame and all rewards remain native.
    }

    private Dictionary<string, object?> PublishCrystalBoundary(CrystalSphereMinigame grid)
    {
        var snapshot = BuildPublicSnapshot();
        var bindings = new List<CandidateBinding>();
        snapshot.Entities.Add(new() { ["entity_type"] = "event_state", ["content_id"] = "CRYSTAL_SPHERE",
            ["remaining_steps"] = grid.DivinationCount });
        var tools = new[] { CrystalSphereMinigame.CrystalSphereToolType.Small, CrystalSphereMinigame.CrystalSphereToolType.Big };
        foreach (var tool in tools)
            snapshot.Entities.Add(new() { ["entity_type"] = "tool", ["ref"] = "tool:" + tool,
                ["content_id"] = tool.ToString(), ["cost"] = 1,
                ["semantic_program"] = new { kind = "effect", op = "REVEAL", shape = tool == tools[0] ? "cell" : "square",
                    radius = tool == tools[0] ? 0 : 1, clip_to_board = true, count = 1 } });
        foreach (var cell in grid.cells)
        {
            string reference = $"cell:{cell.X}:{cell.Y}";
            var entity = new Dictionary<string, object?> { ["entity_type"] = "board_cell", ["ref"] = reference,
                ["x"] = cell.X, ["y"] = cell.Y, ["revealed"] = !cell.IsHidden, ["known"] = !cell.IsHidden };
            // Never traverse Items or serialize item extents behind fog. At an
            // uncovered cell the displayed fragment identifies only that item kind.
            if (!cell.IsHidden) entity["content_id"] = cell.Item?.GetType().Name ?? "empty";
            snapshot.Entities.Add(entity);
            if (!cell.IsHidden) continue; // The native cell view disables revealed cells.
            foreach (var tool in tools)
            {
                var candidate = Candidate("DIVINE_CELL", $"{tool}:{reference}", "tool:" + tool, reference);
                bindings.Add(new(candidate,
                    () => ReferenceEquals(_protocolCrystal, grid) && !grid.IsFinished && cell.IsHidden,
                    () => _protocolUiOperation.Start("crystal divination", async () =>
                    {
                        grid.SetTool(tool);
                        await grid.CellClicked(cell);
                    })));
            }
        }
        return PublishSnapshot("crystal_sphere", snapshot, bindings);
    }
}
