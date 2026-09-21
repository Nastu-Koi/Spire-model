using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Models;

namespace RunRecorder;

internal sealed record SelectionOffer(CardModel[] Cards, int Min, int Max, bool Cancelable, CardModel[][]? Bundles = null, CardRewardAlternative[]? Alternatives = null);
