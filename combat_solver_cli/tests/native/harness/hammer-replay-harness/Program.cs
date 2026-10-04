using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;
using System.Text.Json.Serialization;
var root=Path.GetFullPath(args[0]);
var configPath=args.Length>1 ? Path.GetFullPath(args[1]) : Path.Combine(root,"combat_solver_cli/lib/config.json");
var config=JsonDocument.Parse(File.ReadAllText(configPath)).RootElement;
var dirs=new[]{Path.Combine(root,"combat_solver_cli/bin/Debug/net9.0"),Path.Combine(root,"sts2-cli/lib"),Path.GetDirectoryName(config.GetProperty("solver_dll").GetString())!}.Concat(config.GetProperty("dependency_dirs").EnumerateArray().Select(x=>x.GetString()!));
AssemblyLoadContext.Default.Resolving+=(ctx,name)=>{foreach(var dir in dirs){var path=Path.Combine(dir,name.Name+".dll");if(File.Exists(path))return ctx.LoadFromAssemblyPath(path);}return null;};
Directory.SetCurrentDirectory(Path.Combine(root,"sts2-cli"));
const BindingFlags flags=BindingFlags.Public|BindingFlags.NonPublic|BindingFlags.Static|BindingFlags.Instance;
var engine=Assembly.Load("Sts2Headless");var sim=Activator.CreateInstance(engine.GetType("Sts2Headless.RunSimulator")!)!;
var handle=engine.GetType("Sts2Headless.Program")!.GetMethod("HandleCommand",flags)!;
var worker=Assembly.Load("CombatSolverCli");var solver=Assembly.LoadFrom(config.GetProperty("solver_dll").GetString()!);
var adapterType=worker.GetType("CombatSolverCli.SolverAdapter")!;var adapter=Activator.CreateInstance(adapterType,flags,null,new object[]{solver},null)!;
var jsonOptions=new JsonSerializerOptions { DefaultIgnoreCondition=JsonIgnoreCondition.WhenWritingNull,PropertyNamingPolicy=JsonNamingPolicy.SnakeCaseLower,Converters={new JsonStringEnumConverter(JsonNamingPolicy.SnakeCaseLower)}};
JsonElement Element(object obj)=>JsonSerializer.SerializeToElement(obj,jsonOptions);
JsonElement Send(object obj)=>Element(handle.Invoke(null,new object[]{sim,Element(obj)})!);
JsonElement Settle(JsonElement f){var until=DateTime.UtcNow.AddSeconds(5);while(f.GetProperty("boundary").GetString()=="waiting"&&DateTime.UtcNow<until){Thread.Sleep(10);f=Send(new{cmd="advance_to_boundary"});}if(f.GetProperty("boundary").GetString()=="waiting")throw new Exception("stuck");return f;}
var prefix=JsonDocument.Parse(File.ReadAllText(Path.Combine(root,"combat_solver_cli/tests/native/fixtures/hammer-replay-prefix.json"))).RootElement;
var frame=Send(new{cmd="start_run",character=prefix.GetProperty("character").GetString(),seed=prefix.GetProperty("seed").GetString(),ascension=0,lang="en",decision_protocol=true});
// start_run returns a legacy response; acquire the public boundary explicitly.
frame=Send(new{cmd="advance_to_boundary"});
adapterType.GetMethod("PrepareEngine",flags)!.Invoke(adapter,null);
var records=prefix.GetProperty("records").EnumerateArray().ToArray();
foreach(var r in records[..^3]){
 frame=Settle(frame);var action=r.GetProperty("action");
 var c=frame.GetProperty("legal").GetProperty("candidates").EnumerateArray().Single(c=>c.GetProperty("verb").GetString()==action.GetProperty("verb").GetString()&&c.GetProperty("decoder_slot_ref").GetString()==action.GetProperty("decoder_slot_ref").GetString());
 var command=new Dictionary<string,object>{{"cmd","execute_candidate"},{"candidate_ref",c.GetProperty("candidate_ref").GetString()!}};
 foreach(var k in new[]{"decision_id","state_version","selection_revision"})if(frame.GetProperty("routing").TryGetProperty(k,out var v))command[k]=v.Clone();
 frame=Send(command);
}
frame=Settle(frame);
var planOptions=new JsonSerializerOptions{PropertyNameCaseInsensitive=true,Converters={new JsonStringEnumConverter()}};
var actionJson=File.ReadAllText(Path.Combine(root,"combat_solver_cli/tests/native/fixtures/hammer-replay-action.json"));
if(args.Length>2 && args[2]=="mismatch") actionJson=actionJson.Replace("LUMINESCE", "WRONG_CARD");
if(args.Length>2 && args[2]=="second_instance") {
 var node=System.Text.Json.Nodes.JsonNode.Parse(actionJson)!;
 node["NestedChoices"]![0]!["Cards"]![0]!["SourceOccurrence"]=1;
 node["NestedChoices"]![0]!["Cards"]![0]!["OptionOccurrence"]=1;
 actionJson=node.ToJsonString();
}
var actionPlan=JsonSerializer.Deserialize(actionJson,solver.GetType("CombatSolver.PlanAction")!,planOptions)!;
// Inject only the captured solver plan into the adapter cache. No game mutation:
// this removes timing-dependent search choices while testing real execution.
var queue=adapterType.GetField("_turnPlan",flags)!.GetValue(adapter)!;queue.GetType().GetMethod("Enqueue")!.Invoke(queue,new[]{actionPlan});
// The captured play may fall on any turn of its combat.
var planTurn=JsonDocument.Parse(actionJson).RootElement.GetProperty("Turn").GetInt32();
adapterType.GetField("_planTurn",flags)!.SetValue(adapter,planTurn);
adapterType.GetField("_planPolicy",flags)!.SetValue(adapter,"250|5000|true||60|true");
var endPlan=JsonSerializer.Deserialize("{\"Kind\":\"EndTurn\",\"Turn\":"+planTurn+"}",solver.GetType("CombatSolver.PlanAction")!,planOptions)!;
queue.GetType().GetMethod("Enqueue")!.Invoke(queue,new[]{endPlan});
var entry=solver.GetType("CombatSolver.Entry")!;
entry.GetProperty("Logger",flags)!.SetValue(null,Activator.CreateInstance(solver.GetType("CombatSolver.CombatSolverLog")!,flags,null,new object[]{Path.GetTempPath()},null));
object? expectedSecond = null;
for(int i=0;i<4;i++){
 var command=new Dictionary<string,object>{{"cmd","solver_step"},{"budget_ms",250},{"boss_budget_ms",5000},{"potions",true},{"beam_width",60},{"beam_portfolio",true},{"reuse_turn_plan",true}};
 foreach(var k in new[]{"decision_id","state_version","selection_revision"})if(frame.GetProperty("routing").TryGetProperty(k,out var v))command[k]=v.Clone();
 var response=Element(adapterType.GetMethod("Step",flags)!.Invoke(adapter,new object[]{Element(command),frame,sim})!);
 frame=Settle(response.GetProperty("frame"));
 if(args.Length>2 && args[2]=="second_instance") {
  if(i==0) {
   var selector=sim.GetType().GetField("_cardSelector",flags)!.GetValue(sim)!;
   expectedSecond=((System.Collections.IEnumerable)selector.GetType().GetProperty("PendingOptions",flags)!.GetValue(selector)!).Cast<object>().ElementAt(1);
  }
  if(i==1) {
   var selected=((System.Collections.IEnumerable)adapterType.GetField("_selectedCards",flags)!.GetValue(adapter)!).Cast<object>().Single();
   if(!ReferenceEquals(selected,expectedSecond)) throw new Exception("Wrong duplicate instance selected");
  }
 }
 if(i==2){var entities=frame.GetProperty("public").GetProperty("entities").EnumerateArray().ToArray();var hand=entities.Where(e=>e.TryGetProperty("zone",out var z)&&z.GetString()=="hand"&&e.TryGetProperty("content_id",out var id)&&id.GetString()=="CARD.LUMINESCE").Count();if(hand!=3)throw new Exception("Expected three Luminesce cards, got "+hand);}

 Console.WriteLine($"step {i} {frame.GetProperty("boundary")}");
}
Console.WriteLine("PASS: repeated hammer implicit and explicit choices consumed");
