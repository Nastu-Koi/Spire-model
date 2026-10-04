using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;

var root = Path.GetFullPath(args[0]);
AssemblyLoadContext.Default.Resolving += (ctx, name) => {
    var path = Path.Combine(root, "sts2-cli/lib", name.Name + ".dll");
    return File.Exists(path) ? ctx.LoadFromAssemblyPath(path) : null;
};
Directory.SetCurrentDirectory(Path.Combine(root, "sts2-cli"));
Parity.Run();

static class Parity
{
    public static void Run()
    {
        var sim = new Sts2Headless.RunSimulator();
        var response = sim.StartRun("Ironclad", 0, "merchant-potion-parity");
        if (response.GetValueOrDefault("type") as string == "error") throw new Exception(JsonSerializer.Serialize(response));
        var run = (MegaCrit.Sts2.Core.Runs.RunState)typeof(Sts2Headless.RunSimulator)
            .GetField("_runState", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(sim)!;
        var player = run.Players[0];
        var inventory = MegaCrit.Sts2.Core.Entities.Merchant.MerchantInventory.CreateForNormalMerchant(player);
        int checks = 0;
        foreach (var potion in inventory.PotionEntries)
        for (int i = 0; i < 20; i++)
        {
            var before = player.PlayerRng.Shops.ToSerializable();
            MegaCrit.Sts2.Core.TestSupport.TestMode.IsOn = false;
            try { potion.CalcCost(); }
            finally { MegaCrit.Sts2.Core.TestSupport.TestMode.IsOn = true; }
            var expectedCost = potion.Cost;
            var expectedRng = player.PlayerRng.Shops.ToSerializable();
            var expectedNext = player.PlayerRng.Shops.NextInt();
            player.PlayerRng.Shops.LoadFromSerializable(before);
            potion.CalcCost();
            var actualRng = player.PlayerRng.Shops.ToSerializable();
            if (actualRng.counter != before.counter + 1 || potion.Cost != expectedCost
                || player.PlayerRng.Shops.NextInt() != expectedNext
                || JsonSerializer.Serialize(actualRng) != JsonSerializer.Serialize(expectedRng))
                throw new Exception($"Merchant potion parity failed: headless cost={potion.Cost}, native cost={expectedCost}, headless RNG draws={actualRng.counter - before.counter}, native RNG draws={expectedRng.counter - before.counter}");
            checks++;
        }
        Console.WriteLine($"PASS {checks} potion repricings: price, RNG counter, RNG state and next draw match production");
    }
}
