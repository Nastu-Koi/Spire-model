using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Models;

namespace RunRecorder;

internal sealed record SelectionOffer(CardModel[] Cards, int Min, int Max, bool Cancelable, CardModel[][]? Bundles = null, CardRewardAlternative[]? Alternatives = null)
{
    // Keep the existing six-argument constructor for callers and saved fixtures.
    public SelectionMetadata? Metadata { get; init; }
}
