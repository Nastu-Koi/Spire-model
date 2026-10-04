"""Exercise the real Harmony hook, including errors it must not suppress."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_portfolio_deadline_preserves_native_baseline_and_real_errors(tmp_path):
    if not shutil.which('dotnet'):
        pytest.skip('.NET SDK required')
    root = Path(__file__).resolve().parents[2]
    (tmp_path / 'check.csproj').write_text(f'''<Project Sdk="Microsoft.NET.Sdk">
<PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net9.0</TargetFramework>
<ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup>
<ItemGroup><Compile Include="{root / 'combat_solver_cli/PortfolioDeadline.cs'}" />
<Reference Include="0Harmony"><HintPath>{root / 'sts2-cli/lib/0Harmony.dll'}</HintPath></Reference>
</ItemGroup></Project>''')
    (tmp_path / 'Program.cs').write_text('''using System.Reflection;
using System.Runtime.CompilerServices;
using CombatSolverCli;
PortfolioDeadline.Install(Assembly.GetExecutingAssembly());
PortfolioDeadline.Install(Assembly.GetExecutingAssembly());
foreach (var mode in new[] { "normal", "skip", "timeout", "bug", "foreign", "external", "empty" }) {
    using var cts = new CancellationTokenSource();
    var p = new CombatSolver.Policy { Mode = mode, Cancellation = cts };
    var baseline = mode == "empty" ? null : new object();
    if (mode == "skip") p.Interaction.CurrentTakeoverRequest = new();
    object? result = null; Exception? error = null;
    try { result = CombatSolver.CombatSearchCoordinator.Run(p, cts.Token, baseline); }
    catch (Exception e) { error = e; }
    bool ok = mode switch {
        "normal" => error == null && result != baseline && p.Calls == 1,
        "skip" => error == null && ReferenceEquals(result, baseline) && p.Calls == 0,
        "timeout" => error == null && ReferenceEquals(result, baseline) && p.Calls == 1,
        "bug" => error is InvalidOperationException,
        _ => error is OperationCanceledException
    };
    if (!ok) throw new Exception($"Failed {mode}: {error}");
}
namespace CombatSolver {
    public class Request { public string Kind => "ApplyCurrentTurn"; }
    public class Interaction { public Request? CurrentTakeoverRequest { get; set; } }
    public class Policy {
        public Interaction Interaction { get; } = new();
        public string Mode = ""; public int Calls;
        public CancellationTokenSource Cancellation = null!;
    }
    public static class CombatSearchCoordinator {
        public static object? Run(Policy p, CancellationToken token, object? baseline) =>
            RunOpeningPowerRoutePortfolio(null, null, null, p, token, null, null, null, baseline);
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static object? RunOpeningPowerRoutePortfolio(object? a, object? b, object? c,
            Policy p, CancellationToken token, object? d, object? e, object? f, object? baseline) {
            p.Calls++;
            if (p.Mode == "normal") return new object();
            if (p.Mode != "external") p.Interaction.CurrentTakeoverRequest = new();
            p.Cancellation.Cancel();
            if (p.Mode == "bug") throw new InvalidOperationException("real bug");
            if (p.Mode == "foreign") throw new OperationCanceledException(new CancellationToken(true));
            token.ThrowIfCancellationRequested();
            return baseline;
        }
    }
}
''')
    result = subprocess.run(['dotnet', 'run', '--project', str(tmp_path / 'check.csproj'),
                             '--verbosity', 'quiet'], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
