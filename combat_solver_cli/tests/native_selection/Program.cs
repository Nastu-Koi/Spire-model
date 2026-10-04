using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Text.Json;
using CombatSolverCli;
using Sts2Headless;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Localization;

class Program
{
    const BindingFlags Flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static;
    static void Main(string[] args)
    {
        var root = Path.GetFullPath(args[0]);
        var config = JsonDocument.Parse(File.ReadAllText(args[1])).RootElement;
        var dirs = new[] { Path.Combine(root,"combat_solver_cli/bin/Debug/net9.0"),Path.Combine(root,"sts2-cli/lib"),Path.GetDirectoryName(config.GetProperty("solver_dll").GetString())! }
            .Concat(config.GetProperty("dependency_dirs").EnumerateArray().Select(x=>x.GetString()!));
        AssemblyLoadContext.Default.Resolving += (ctx,name) => { foreach(var dir in dirs) {var path=Path.Combine(dir,name.Name+".dll");if(File.Exists(path))return ctx.LoadFromAssemblyPath(path);}return null; };
        Directory.SetCurrentDirectory(Path.Combine(root,"sts2-cli"));
        Run(Assembly.LoadFrom(config.GetProperty("solver_dll").GetString()!));
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void Run(Assembly solver)
    {
        var sim = new RunSimulator();
        var handle = typeof(RunSimulator).Assembly.GetType("Sts2Headless.Program")!.GetMethod("HandleCommand",Flags)!;
        JsonElement Send(object request) => JsonSerializer.SerializeToElement(handle.Invoke(null,[sim,JsonSerializer.SerializeToElement(request)]));
        Send(new {cmd="start_run", character="Ironclad", seed="selection-branch-matrix", ascension=0});
        Send(new {cmd="set_player", hp=100,max_hp=100,relics=Array.Empty<string>(),potions=Array.Empty<string>(),deck=new[]{"DEFEND_IRONCLAD","STRIKE_IRONCLAD","BASH"}});
        var frame=Send(new {cmd="enter_room",type="combat",encounter="SHRINKER_BEETLE_WEAK",decision_protocol=true});
        for(int n=0;frame.GetProperty("boundary").GetString()=="waiting" && n<500;n++){Thread.Sleep(10);frame=Send(new{cmd="advance_to_boundary"});}
        var player=MegaCrit.Sts2.Core.Combat.CombatManager.Instance.DebugOnlyGetState()!.Players.Single();
        var keyMethod=solver.GetType("CombatSolver.CardChoiceSupport")!.GetMethods(Flags).Single(m=>m.Name=="ChoiceCardKey"&&m.GetParameters().Length==1&&m.GetParameters()[0].ParameterType==typeof(CardModel));
        string Key(CardModel card)=>(string)keyMethod.Invoke(null,[card])!;
        var selector=typeof(RunSimulator).GetField("_cardSelector",Flags)!.GetValue(sim)!;
        new SolverAdapter(solver).PrepareEngine();
        int passed=0;
        var hand=player.PlayerCombatState!.Hand.Cards.ToArray();
        object Choice(string pile,IEnumerable<CardModel> cards,string effect) => new { SourcePile=pile,Effect=effect,Cards=cards.Select(c=>new{CardId=c.Id.Entry,UpgradeLevel=c.CurrentUpgradeLevel,StateKey=Key(c),SourceOccurrence=0,OptionOccurrence=0}).ToArray()};
        var prefs=new CardSelectorPrefs(new LocString("cards","TEST.selectionScreenPrompt"),1);
        void Test(string name, string pile, CardModel[] expected, Func<Task> invoke, bool emptyNoPlan=false, int minimum=1)
        {
            using var tx=new ChoiceTransaction(emptyNoPlan ? [] : [Choice(pile,expected,name.Contains("upgrade") ? "Upgrade" : "unknown")],player,Key);
            NativeSelectionBridge.Activate(tx);
            try
            {
                var task=invoke();
                if(!task.IsCompleted && selector.GetType().GetProperty("PendingOptions")!.GetValue(selector) is List<CardModel> options)
                {
                    var selected=tx.Select(options,minimum,1);
                    selector.GetType().GetMethod("ResolvePending")!.Invoke(selector,[selected]);
                }
                if(!task.IsCompleted && typeof(RunSimulator).GetField("_pendingBundleTcs",Flags)!.GetValue(sim) is TaskCompletionSource<IEnumerable<CardModel>> bundle)
                {
                    var bundles=(IReadOnlyList<IReadOnlyList<CardModel>>)typeof(RunSimulator).GetField("_pendingBundles",Flags)!.GetValue(sim)!;
                    var selected=tx.Select(bundles.SelectMany(b=>b),0,int.MaxValue);
                    typeof(RunSimulator).GetField("_pendingBundleTcs",Flags)!.SetValue(sim,null);
                    typeof(RunSimulator).GetField("_pendingBundles",Flags)!.SetValue(sim,null);
                    bundle.SetResult(selected);
                }
                if(!task.Wait(TimeSpan.FromSeconds(3)))throw new Exception(name+" stalled");
                task.GetAwaiter().GetResult();tx.FinishAction();
                passed++;Console.WriteLine("PASS "+name);
            }
            finally {NativeSelectionBridge.Activate(null);}
        }
        Test("hand filtered automatic","Hand",[hand[0]],()=>CardSelectCmd.FromHand(null!,player,prefs,c=>ReferenceEquals(c,hand[0]),hand[1]));
        Test("hand explicit asynchronous","Hand",[hand[1]],()=>CardSelectCmd.FromHand(null!,player,prefs,null,hand[0]));
        Test("hand zero candidates","Hand",[],()=>CardSelectCmd.FromHand(null!,player,prefs,_=>false,hand[0]),true);
        var allPrefs=new CardSelectorPrefs(new LocString("cards","TEST.selectionScreenPrompt"),hand.Length);
        Test("hand multi automatic","Hand",hand,()=>CardSelectCmd.FromHand(null!,player,allPrefs,null,hand[0]));
        Test("hand upgrade","Hand",[hand[0]],()=>CardSelectCmd.FromHandForUpgrade(null!,player,hand[1]));
        Test("combat hand filtered overload","Hand",[hand[0]],()=>CardSelectCmd.FromCombatPile(null!,player.PlayerCombatState.Hand,player,prefs,c=>ReferenceEquals(c,hand[0])));
        foreach(var pile in new[]{PileType.Discard,PileType.Draw,PileType.Exhaust})
        {
            CardPileCmd.Add(hand[0],pile).GetAwaiter().GetResult();
            Test("combat "+pile+" wrapper",pile.ToString(),[hand[0]],()=>CardSelectCmd.FromCombatPile(null!,pile.GetPile(player),player,prefs));
            CardPileCmd.Add(hand[0],PileType.Hand).GetAwaiter().GetResult();
        }
        Test("generated grid automatic","None",[hand[0]],()=>CardSelectCmd.FromSimpleGrid(null!,new[]{hand[0]},player,prefs));
        Test("generated grid explicit","None",[hand[1]],()=>CardSelectCmd.FromSimpleGrid(null!,hand,player,prefs));
        Test("choose card automatic","None",[hand[0]],()=>CardSelectCmd.FromChooseACardScreen(null!,new[]{hand[0]},player,false));
        Test("choose card explicit","None",[hand[1]],()=>CardSelectCmd.FromChooseACardScreen(null!,hand,player,false));
        var deck=PileType.Deck.GetPile(player).Cards.ToArray();
        Test("deck upgrade","Deck",[deck[0]],()=>CardSelectCmd.FromDeckForUpgrade(player,prefs));
        // Generic/removal wrappers use the game's real filter, once per candidate.
        int filterCalls=0;
        Test("deck removal filtered","Deck",[deck[0]],()=>CardSelectCmd.FromDeckForRemoval(player,prefs,c=>{filterCalls++;return ReferenceEquals(c,deck[0]);}));
        if(filterCalls!=deck.Length)throw new Exception("Filter was evaluated more than once");
        Test("reward grid automatic","None",[hand[0]],()=>CardSelectCmd.FromSimpleGridForRewards(null!,[new CardCreationResult(hand[0])],player,prefs));
        Test("reward grid explicit","None",[hand[1]],()=>CardSelectCmd.FromSimpleGridForRewards(null!,hand.Select(c=>new CardCreationResult(c)).ToList(),player,prefs));
        Test("deck transformation","Deck",[deck[0]],()=>CardSelectCmd.FromDeckForTransformation(player,prefs,c=>new CardTransformation(c)));
        var enchant=ModelDb.GetById<EnchantmentModel>(new ModelId("ENCHANTMENT", "ADROIT"));
        Test("deck enchantment leaf","Deck",[deck[0]],()=>CardSelectCmd.FromDeckForEnchantment(new[]{deck[0]},enchant,1,prefs));
        Test("bundle asynchronous","None",[hand[0],hand[1]],()=>CardSelectCmd.FromChooseABundleScreen(player,new IReadOnlyList<CardModel>[] {new[]{hand[0],hand[1]},new[]{hand[2]}}));
        Test("bundle empty","None",[],()=>CardSelectCmd.FromChooseABundleScreen(player,Array.Empty<IReadOnlyList<CardModel>>()),true);
        // Synchronous reward API blocks on another thread; the transaction remains
        // bound to this execution, and the result is checked before it returns.
        using(var tx=new ChoiceTransaction([Choice("None",[hand[1]],"unknown")],player,Key))
        {
            NativeSelectionBridge.Activate(tx);
            try
            {
                var task=Task.Run(()=>selector.GetType().GetMethod("GetSelectedCardReward")!.Invoke(selector,
                    [hand.Select(c=>new CardCreationResult(c)).ToList(),Array.Empty<MegaCrit.Sts2.Core.Entities.CardRewardAlternatives.CardRewardAlternative>()]));
                if(!SpinWait.SpinUntil(()=>selector.GetType().GetProperty("HasPendingReward")!.GetValue(selector) is true,3000))throw new Exception("reward stalled");
                tx.Select(hand,0,1);
                selector.GetType().GetMethod("ResolveReward")!.Invoke(selector,[1]);
                task.GetAwaiter().GetResult();tx.FinishAction();passed++;
                Console.WriteLine("PASS reward cross-thread");
            }
            finally {NativeSelectionBridge.Activate(null);}
        }
        Test("choose card skipped","None",[],async () => {
            if(await CardSelectCmd.FromChooseACardScreen(null!,hand,player,true) != null)throw new Exception("Expected null skip");
        },minimum:0);
        foreach(var card in hand)CardPileCmd.Add(card,PileType.Discard).GetAwaiter().GetResult();
        Test("hand upgrade empty hand","Hand",[],async () => {
            if(await CardSelectCmd.FromHandForUpgrade(null!,player,hand[0]) != null)throw new Exception("Expected null upgrade");
        },true);
        foreach(var card in hand)CardPileCmd.Add(card,PileType.Hand).GetAwaiter().GetResult();
        foreach(var card in hand)card.UpgradeInternal();
        Test("hand upgrade all upgraded","Hand",[],async () => {
            if(await CardSelectCmd.FromHandForUpgrade(null!,player,hand[0]) != null)throw new Exception("Expected null upgrade");
        },true);
        Console.WriteLine($"PASS {passed} native selection branches");
    }
}
