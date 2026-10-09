using Spire.PublicHistory;

static void Check(bool value, string message)
{ if (!value) throw new Exception(message); }
static void Near(float? value, float expected, string message)
{ Check(value.HasValue && Math.Abs(value.Value - expected) < .00001f, message); }
var all = OddsHistory.RoomTypes.ToHashSet();
var noShop = all.Where(t => t != "Shop").ToHashSet();
var odds = new OddsHistory(true, 0);
Near(odds.PotionProbability(false, false), .4f, "new run potion prior");
Near(odds.PotionProbability(true, false), .525f, "elite increment");
odds.PotionGenerated(true, false);
Near(odds.PotionProbability(false, false), .3f, "generation updates before collection");
odds.PotionGenerated(true, true);
Near(odds.PotionProbability(false, false), .3f, "forced reward does not update accumulator");
odds.EnterAct(1);
Near(odds.PotionProbability(false, false), .3f, "potion accumulator persists across acts");
for (int i = 0; i < 12; i++) odds.PotionGenerated(false, false);
Near(odds.PotionProbability(false, false), 1, "effective probability clamps");
odds.PotionGenerated(true, false);
Near(odds.PotionProbability(false, false), 1, "accumulator itself must not clamp");

odds = new(true, 0);
Near(odds.RoomProbabilities(all)["Event"], .85f, "initial question odds");
odds.QuestionRoomGenerated("Event", all);
Near(odds.RoomProbabilities(all)["Monster"], .2f, "unrolled room increment");
Near(odds.RoomProbabilities(noShop)["Event"], .76f, "blacklisted mass falls back to event");
odds.QuestionRoomGenerated("Monster", noShop);
Near(odds.RoomProbabilities(all)["Monster"], .1f, "selected room resets");
Near(odds.RoomProbabilities(all)["Shop"], .06f, "excluded room does not increment");
for (int i = 0; i < 30; i++) odds.QuestionRoomGenerated("Event", all);
Near(odds.RoomProbabilities(all)["Monster"], 1, "native ordered cumulative clipping");
Near(odds.RoomProbabilities(all)["Treasure"], 0, "later mass vanishes");
odds.EnterAct(1);
Near(odds.RoomProbabilities(all)["Event"], .85f, "question odds reset each act");

var loaded = new OddsHistory(false, 1);
loaded.EnterAct(1);
Check(loaded.PotionProbability(false, false) == null && loaded.RoomProbabilities(all)["Event"] == null,
    "loading/reentering an act cannot invent a prior");
loaded.PotionGenerated(false, false);
Check(loaded.PotionProbability(false, false) == null, "unknown potion prefix stays unknown");
loaded.QuestionRoomGenerated("Monster", all);
Near(loaded.RoomProbabilities(all)["Monster"], .1f, "an observed room reset recovers its independently known probability");
Check(loaded.RoomProbabilities(all)["Event"] == null, "one room reset cannot invent the remaining probabilities");
Near(loaded.PotionProbability(false, true), 1, "public force is known even on midrun attach");
loaded.EnterAct(2);
Near(loaded.RoomProbabilities(all)["Event"], .85f, "public act transition restores room knowledge");
Check(loaded.PotionProbability(false, false) == null, "act transition cannot recover potion prefix");
var tutorial = new OddsHistory(true, 0, tutorial: true);
foreach (string type in new[] { "Event", "Event", "Monster" })
{
    Near(tutorial.RoomProbabilities(all)[type], 1, "public tutorial sequence");
    tutorial.QuestionRoomGenerated(type, all);
}
Near(tutorial.RoomProbabilities(all)["Event"], .85f, "tutorial does not change ordinary odds");
var attachedTutorial = new OddsHistory(false, 0, tutorial: true);
foreach (string type in new[] { "Event", "Event", "Monster" }) attachedTutorial.QuestionRoomGenerated(type, all);
attachedTutorial.EnterAct(1);
Near(attachedTutorial.RoomProbabilities(all)["Event"], .85f, "three observed questions bound the missing tutorial prefix");
var deadly = new OddsHistory(true, 0, deadlyEvents: true);
Near(deadly.RoomProbabilities(all)["Elite"], .1f, "public deadly events modifier");
deadly.QuestionRoomGenerated("Event", all);
Near(deadly.RoomProbabilities(all)["Treasure"], .06f, "deadly events doubles treasure increment");

// Reconstruct every prefix using the same public event log as online maintenance.
var log = new List<Action<OddsHistory>>();
var online = new OddsHistory(true, 0);
foreach (var apply in new Action<OddsHistory>[] {
    h => h.PotionGenerated(false, false), h => h.QuestionRoomGenerated("Event", all),
    h => h.QuestionRoomGenerated("Monster", noShop), h => h.PotionGenerated(true, true), h => h.EnterAct(1) })
{
    log.Add(apply); apply(online);
    var replay = new OddsHistory(true, 0);
    foreach (var e in log) e(replay);
    Check(online.RoomProbabilities(all).SequenceEqual(replay.RoomProbabilities(all)) &&
        online.PotionProbability(true, false) == replay.PotionProbability(true, false), "prefix replay parity");
}

// The knowledge tracker never sees random insertion/removal indices. Check all
// facts it retains against an independent hidden pile through 10,000 mutations.
var rng = new Random(8429);
for (int trial = 0; trial < 100; trial++)
{
    var actual = Enumerable.Range(0, 7).Select(_ => new object()).ToList();
    var known = new DrawKnowledge<object>();
    known.Invalidate(actual.Count);
    for (int step = 0; step < 100; step++)
    {
        int action = rng.Next(5);
        if (action < 3 || actual.Count == 0)
        {
            var card = new object();
            Placement placement = action == 0 ? Placement.Top : action == 1 ? Placement.Bottom : Placement.Unknown;
            known.Added(card, placement, actual.Count);
            actual.Insert(placement == Placement.Top ? 0 : placement == Placement.Bottom ? actual.Count : rng.Next(actual.Count + 1), card);
        }
        else if (action == 3)
        {
            int removed = rng.Next(actual.Count);
            known.Removed(actual[removed], actual.Count);
            actual.RemoveAt(removed);
        }
        else
        {
            actual = actual.OrderBy(_ => rng.Next()).ToList();
            known.Invalidate(actual.Count);
        }
        foreach (var (card, position) in known.KnownPositions(actual.Count))
            Check(ReferenceEquals(card, actual[position]), "history leaked or retained an uncertain card position");
    }
}
Console.WriteLine("Public history: odds, missing prefixes, resets, replay, and 10000 hidden-pile mutations passed.");
