import pytest
import torch

from model.config import ModelConfig
from model.model import PolicyValue
from model.representation import Vocabulary, symbols_from_frames
from model.testing import SyntheticEngine, demonstration


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(17)


@pytest.fixture
def demo():
    return demonstration(count=5, select=3)


@pytest.fixture
def setup(demo):
    cfg = ModelConfig.tiny()
    frames = [s["frame"] for m in demo["macros"] for s in m["steps"]]
    vocab = Vocabulary(symbols_from_frames(frames), cfg.vocabulary_size)
    model = PolicyValue(cfg)
    return model, vocab
