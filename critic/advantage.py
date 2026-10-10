"""The critic's advantages for the rounds of a buffer.

Every decision a round's runs made outside fights is scored in batches, packed by
loader processes as the critic's training packs them; its value of the ±1 return is
`2 * p_win - 1`, and the temporal differences along each run become generalized
advantages (`model/advantage.py`, which also keeps the file they are written to
beside the round). A round already scored by this critic, at this λ, is left as it is.

python -m critic.advantage runs/iterate/buffer --critic runs/iterate/critic/iteration-0003
"""

import argparse
import json
from pathlib import Path

import torch

from model.advantage import CRITIC, LAMBDA, current, from_values, header, weights_digest, write

from .config import TrainingConfig
from .data import expand, load
from .model import load as load_critic
from .train import predict_rows


def identity(directory, scorer, lam=LAMBDA):
    """What advantages computed by this critic say of where they came from."""
    return dict(kind=CRITIC, lam=lam, critic=dict(
        version=scorer.version, weights=weights_digest(Path(directory) / "weights.pt")))


def score_rounds(rounds, critic, device="cpu", *, lam=LAMBDA, loader_workers=None):
    """Write the advantages of every round the saved critic at `critic` has not scored
    yet. Returns each round's header, in order."""
    scorer = load_critic(critic, device)
    settings = TrainingConfig(**scorer.manifest["training"])
    if loader_workers is not None:
        settings.loader_workers = loader_workers
    wanted = identity(critic, scorer, lam)
    headers = []
    for directory in rounds:
        if current(directory, wanted):
            headers.append(header(directory))
            continue
        rows, _ = load([directory])
        probabilities = predict_rows(scorer.net, scorer.vocabulary, rows, settings, scorer.net.device)
        values = (2.0 * probabilities[:, 0] - 1.0).tolist()
        headers.append(write(directory, wanted, from_values(
            [(row.run_id, row.macro, row.win) for row in rows], values, lam)))
        del rows
    return headers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rounds", type=Path, nargs="+", help="sampled rounds (round-N/) or a buffer directory of them")
    parser.add_argument("--critic", type=Path, required=True, help="saved critic directory")
    parser.add_argument("--buffer-iterations", type=int,
                        help="of a buffer directory, score the rounds of its last so many iterations")
    parser.add_argument("--lam", type=float, default=LAMBDA, help="λ of the generalized advantage")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()
    print(json.dumps(score_rounds(expand(arguments.rounds, arguments.buffer_iterations), arguments.critic,
                                  arguments.device, lam=arguments.lam), indent=1))
