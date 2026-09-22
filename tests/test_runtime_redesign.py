from copy import deepcopy
import json
import random

import pytest
import torch

from model.checkpoint import load_model, restore_training, save_checkpoint
from model.cli import vocabulary_for
from model.config import ModelConfig, TrainConfig
from model.data import validate_run
from model.model import PolicyValue
from model.policy import replay
from model.protocol import CHARACTERS, ProtocolError
from model.rollout import RolloutRunner, collect_round
from model.testing import SyntheticEngine, demonstration
from model.trainer import Learner


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def model_and_vocab():
    run = demonstration(count=4, select=2)
    config = ModelConfig.tiny()
    vocab = vocabulary_for([run], config.vocabulary_size)
    return PolicyValue(config), vocab, run


def test_resume_restores_both_optimizers_and_next_update(tmp_path):
    model, vocab, run = model_and_vocab()
    cfg = TrainConfig(precision='no', logical_batch_size=2, microbatch_size=2)
    learner = Learner(model, vocab, cfg)
    learner.bootstrap([run])
    path = tmp_path / 'checkpoint'
    save_checkpoint(path, model, vocab, learner.optimizer, learner.scheduler,
                    training=cfg, progress={'policy_version': learner.policy_version, 'updates': learner.updates})
    loaded, restored_vocab, manifest = load_model(path)
    restored = Learner(loaded, restored_vocab, cfg, policy_version=learner.policy_version)
    progress = restore_training(path, restored.optimizer, restored.scheduler)
    restored.updates = progress['updates']
    assert set(manifest['optimizer']) == {'adamw', 'muon'}
    random.seed(123)
    learner.bootstrap([run])
    random.seed(123)
    restored.bootstrap([run])
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
    assert learner.updates == restored.updates


def test_old_checkpoint_is_rejected_before_loading_weights(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'format': 1, 'model': {}}))
    with pytest.raises(ValueError, match='architecture v2'):
        load_model(tmp_path)


def test_current_checkpoint_atomic_replacement(tmp_path):
    model, vocab, _ = model_and_vocab()
    path = tmp_path / 'current'
    save_checkpoint(path, model, vocab)
    with torch.no_grad():
        model.bos.add_(1)
    save_checkpoint(path, model, vocab, overwrite=True)
    loaded, _, _ = load_model(path)
    torch.testing.assert_close(loaded.bos, model.bos)
    assert not list(tmp_path.glob('*.staging-*'))


def test_parallel_rollouts_replay_and_use_batches(tmp_path):
    model, vocab, _ = model_and_vocab()
    batches = []
    original = model.encode
    def capture(observations, vocabulary, **kwargs):
        batches.append(len(observations))
        return original(observations, vocabulary, **kwargs)
    model.encode = capture
    runner = RolloutRunner(model, vocab, precision='no')
    seeds = {c: [f'parallel-{c}'] for c in CHARACTERS}
    paths = collect_round(lambda: SyntheticEngine(4, 2), runner, seeds, tmp_path / 'parallel', workers=3)
    assert len(paths) == 5
    assert max(batches) > 1
    runs = [json.loads(p.read_text()) for p in paths]
    for run in runs:
        validate_run(run, on_policy=True)
        with torch.no_grad():
            for macro in run['macros']:
                lp, value, _ = replay(model, vocab, macro['steps'])
                assert float(lp) == pytest.approx(macro['old_log_prob'], abs=2e-5)
                assert float(value) == pytest.approx(macro['old_value'], abs=2e-5)
    learner = Learner(model, vocab, TrainConfig(precision='no', microbatch_size=3))
    assert learner.ppo(runs)


def test_failed_parallel_round_blocks_update(tmp_path):
    model, vocab, _ = model_and_vocab()
    class BadEngine(SyntheticEngine):
        def send(self, command):
            raise ProtocolError('injected failure')
    seeds = {c: [f'failure-{c}'] for c in CHARACTERS}
    runner = RolloutRunner(model, vocab)
    with pytest.raises(ProtocolError, match='Round incomplete'):
        collect_round(lambda: BadEngine(4, 2), runner, seeds, tmp_path / 'failed', workers=2)
    assert len(list((tmp_path / 'failed').glob('*.json'))) == 5


def test_small_profiles_are_valid():
    from pathlib import Path
    for path in Path('configs').glob('*.json'):
        profile = json.loads(path.read_text())
        if 'model' in profile:
            model = ModelConfig(**profile['model'])
            TrainConfig.from_dict(profile['training'])
            assert model.architecture_version == 2


def test_evaluation_seed_isolation():
    from pathlib import Path
    from model.seeds import validate_evaluation_seeds
    seeds = json.loads(Path('configs/evaluation-seeds.json').read_text())
    final = json.loads(Path('configs/final-test-seeds.json').read_text())
    validate_evaluation_seeds(seeds, {})
    assert not {s for group in seeds.values() for s in group} & {s for group in final.values() for s in group}
    with pytest.raises(ValueError, match='overlaps'):
        validate_evaluation_seeds(seeds, {'training_seeds': seeds['Ironclad'][:1]})
