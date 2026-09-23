using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace CombatSolverCli;

internal static class Program
{
    internal static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.SnakeCaseLower) }
    };

    private static void Main()
    {
        string lib = Path.GetFullPath(Environment.GetEnvironmentVariable("STS2_LIB") ?? "../sts2-cli/lib");
        string solver = Path.GetFullPath(Environment.GetEnvironmentVariable("COMBAT_SOLVER_DLL")
            ?? throw new InvalidOperationException("Set COMBAT_SOLVER_DLL to CombatSolver 0.44.0 DLL"));
        string[] paths = new[] { lib, Path.GetDirectoryName(solver)! }
            .Concat((Environment.GetEnvironmentVariable("COMBAT_SOLVER_DEPENDENCIES") ?? "")
                .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)).ToArray();
        AssemblyLoadContext.Default.Resolving += (context, name) =>
        {
            foreach (string directory in paths)
            {
                string path = Path.Combine(directory, name.Name + ".dll");
                if (File.Exists(path)) return context.LoadFromAssemblyPath(Path.GetFullPath(path));
            }
            return null;
        };
        Run(solver);
    }

    // Defer game type loading until the external dependency resolver is installed.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Run(string solverPath)
    {
        Assembly engine = Assembly.Load("Sts2Headless");
        object simulator = Activator.CreateInstance(engine.GetType("Sts2Headless.RunSimulator", true)!)!;
        MethodInfo handle = engine.GetType("Sts2Headless.Program", true)!
            .GetMethod("HandleCommand", BindingFlags.Static | BindingFlags.NonPublic)!;
        var solver = new SolverAdapter(AssemblyLoadContext.Default.LoadFromAssemblyPath(solverPath));
        JsonElement? frame = null;
        Console.WriteLine("{\"type\":\"ready\",\"version\":\"combat-solver-cli-v1\"}");
        string? line;
        while ((line = Console.ReadLine()) != null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            object response;
            try
            {
                using JsonDocument document = JsonDocument.Parse(line);
                JsonElement request = document.RootElement;
                string command = request.GetProperty("cmd").GetString()!;
                if (command == "solver_info") response = solver.Info();
                else if (command is "solver_solve" or "solver_step")
                {
                    if (frame is null) throw new InvalidOperationException("Obtain a decision frame first");
                    response = command == "solver_solve"
                        ? solver.Solve(request, frame.Value) : solver.Step(request, frame.Value, simulator);
                    if (command == "solver_step")
                        frame = JsonSerializer.SerializeToElement(response, Json).GetProperty("frame").Clone();
                }
                else
                {
                    if (command is "start_run" or "load_save" or "enter_room" or "action" or "set_player"
                        or "set_draw_order" or "execute_candidate" or "quit") solver.Reset();
                    response = handle.Invoke(null, new[] { simulator, (object)request })!;
                    if (command is "start_run" or "load_save") solver.PrepareEngine();
                    JsonElement result = JsonSerializer.SerializeToElement(response, Json);
                    if (result.TryGetProperty("type", out var type) && type.GetString() == "decision_frame") frame = result;
                    else if (command is "start_run" or "load_save" or "enter_room" or "action" or "set_player" or "quit") frame = null;
                }
            }
            catch (Exception exception)
            {
                while (exception is TargetInvocationException && exception.InnerException is not null)
                    exception = exception.InnerException;
                response = new { type = "error", message = exception.Message, stack_trace = exception.ToString() };
            }
            JsonElement packet = JsonSerializer.SerializeToElement(response, Json);
            Console.WriteLine(packet.GetRawText());
            if (solver.Poisoned || (packet.TryGetProperty("type", out var kind) && kind.GetString() == "quit_result")) break;
        }
    }
}
