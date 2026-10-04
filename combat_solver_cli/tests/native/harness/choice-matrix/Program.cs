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

solver.GetType("CombatSolver.Entry")!.GetProperty("Logger",flags)!.SetValue(null,
 Activator.CreateInstance(solver.GetType("CombatSolver.CombatSolverLog")!,flags,null,new object[]{Path.Combine(Path.GetTempPath(),"choice-matrix",Environment.ProcessId.ToString())},null));
var scenario=args[2];

var repeat=args.Length>3 && args[3].StartsWith("repeat_");
var explicitChoice=args.Length>3 && args[3].EndsWith("explicit");
var (character,target,effect)=scenario switch {
 "NIGHTMARE" => ("Silent","DEFEND_SILENT","Nightmare"),
 "SURVIVOR" => ("Silent","DEFEND_SILENT","Discard"),
 "ARMAMENTS" => ("Ironclad","DEFEND_IRONCLAD","Upgrade"),
 "BURNING_PACT" => ("Ironclad","DEFEND_IRONCLAD","Exhaust"),
 _ => throw new Exception("unknown scenario")
};
Send(new{cmd="start_run",character,seed="implicit-"+scenario,ascension=0});
Send(new{cmd="set_player",hp=100,max_hp=100,relics=Array.Empty<string>(),potions=Array.Empty<string>(),deck=explicitChoice ? new[]{scenario,target,target} : new[]{scenario,target}});
var frame=Settle(Send(new{cmd="enter_room",type="combat",encounter="SHRINKER_BEETLE_WEAK",decision_protocol=true}));
adapterType.GetMethod("PrepareEngine",flags)!.Invoke(adapter,null);
object Get(object obj,string name)=>obj.GetType().GetProperty(name,flags)?.GetValue(obj) ?? obj.GetType().GetField(name,flags)!.GetValue(obj)!;
var snapshot=sim.GetType().GetMethod("BuildPublicSnapshot",flags)!.Invoke(sim,null)!;
var cards=(System.Collections.IDictionary)Get(snapshot,"Cards");
// Snapshot contains both deck and combat entities; select the exact hand reference.
object HandCard(string id){var reference=frame.GetProperty("public").GetProperty("entities").EnumerateArray().First(e=>e.TryGetProperty("zone",out var z)&&z.GetString()=="hand"&&e.TryGetProperty("content_id",out var cid)&&cid.GetString()=="CARD."+id).GetProperty("ref").GetString();return cards.Keys.Cast<object>().Single(c=>cards[c]?.ToString()==reference);}
var keyMethod=solver.GetType("CombatSolver.CardChoiceSupport")!.GetMethods(flags).Single(m=>m.Name=="ChoiceCardKey"&&m.GetParameters().Length==1&&m.GetParameters()[0].ParameterType.Name=="CardModel");
string Key(object card)=>(string)keyMethod.Invoke(null,new[]{card})!;
var sourceCard=HandCard(scenario);var targetCard=HandCard(target);
if(repeat) sourceCard.GetType().GetProperty("BaseReplayCount",flags)!.SetValue(sourceCard,1);
var plan=System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(Path.Combine(root,"combat_solver_cli/tests/native/fixtures/decisions-action.json")))!;
plan["CardId"]=scenario;plan["CardStateKey"]=Key(sourceCard);plan["Turn"]=1;
plan["Choice"]!["Effect"]=effect;plan["Choice"]!["Cards"]![0]!["CardId"]=target;plan["Choice"]!["Cards"]![0]!["StateKey"]=Key(targetCard);
if(repeat) plan["NestedChoices"]=new System.Text.Json.Nodes.JsonArray(plan["Choice"]!.DeepClone());
var options=new JsonSerializerOptions{PropertyNameCaseInsensitive=true,Converters={new JsonStringEnumConverter()}};
var planType=solver.GetType("CombatSolver.PlanAction")!;
var action=JsonSerializer.Deserialize(plan.ToJsonString(),planType,options)!;
var queue=adapterType.GetField("_turnPlan",flags)!.GetValue(adapter)!;
void Enqueue(object a)=>queue.GetType().GetMethod("Enqueue")!.Invoke(queue,new[]{a});
Enqueue(action);
var end=System.Text.Json.Nodes.JsonNode.Parse(plan.ToJsonString())!;end["Kind"]="EndTurn";end["Choice"]=null;end["NestedChoices"]=null;end["CardId"]="";Enqueue(JsonSerializer.Deserialize(end.ToJsonString(),planType,options)!);
adapterType.GetField("_planTurn",flags)!.SetValue(adapter,1);
adapterType.GetField("_planPolicy",flags)!.SetValue(adapter,"250|5000|true||60|true");
for(int i=0;i<10;i++){
 var command=new Dictionary<string,object>{{"budget_ms",250},{"boss_budget_ms",5000},{"potions",true},{"beam_width",60},{"beam_portfolio",true},{"reuse_turn_plan",true}};
 foreach(var k in new[]{"decision_id","state_version","selection_revision"})if(frame.GetProperty("routing").TryGetProperty(k,out var v))command[k]=v.Clone();
 var response=Element(adapterType.GetMethod("Step",flags)!.Invoke(adapter,new object[]{Element(command),frame,sim})!);
 var chosen=frame.GetProperty("legal").GetProperty("candidates").EnumerateArray().Single(c=>c.GetProperty("candidate_ref").GetString()==response.GetProperty("candidate_ref").GetString()).GetProperty("verb").GetString();
 frame=Settle(response.GetProperty("frame"));
 if(chosen=="END_TURN"){
  if(scenario=="NIGHTMARE"){
   var hand=frame.GetProperty("public").GetProperty("entities").EnumerateArray().Count(e=>e.TryGetProperty("zone",out var z)&&z.GetString()=="hand"&&e.TryGetProperty("content_id",out var id)&&id.GetString()=="CARD."+target);
   if(hand<(repeat ? 6 : 3))throw new Exception("Nightmare failed to generate copies");
  }
  Console.WriteLine("PASS "+scenario);return;
 }
 if(frame.GetProperty("public").GetProperty("phase").GetString()=="combat"){
  if(scenario=="SURVIVOR"||scenario=="BURNING_PACT"){
   var zone=Get(Get(targetCard,"Pile"),"Type").ToString();
   if(zone!=(scenario=="SURVIVOR"?"Discard":"Exhaust"))throw new Exception("Wrong destination: "+zone);
  }
  if(scenario=="ARMAMENTS" && !frame.GetProperty("public").GetProperty("entities").EnumerateArray().Any(e=>e.TryGetProperty("zone",out var z)&&z.GetString()=="hand"&&e.TryGetProperty("content_id",out var id)&&id.GetString()=="CARD."+target&&e.GetProperty("upgraded").GetBoolean()))throw new Exception("Upgrade not applied");
 }
}
throw new Exception("Selection did not complete");
