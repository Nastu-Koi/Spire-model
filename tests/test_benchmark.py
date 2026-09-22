import json

from model.benchmark import benchmark, capacity_report
from model.testing import demonstration


def test_statistics_skip_seed_metadata_and_journals(tmp_path):
    run = demonstration(count=4, select=2)
    (tmp_path / 'run.json').write_text(json.dumps(run))
    (tmp_path / 'journal.jsonl').write_text('not a run\n')
    (tmp_path / 'metadata').mkdir()
    (tmp_path / 'metadata' / 'seeds.json').write_text('{"Ironclad": ["seed"]}')
    report = capacity_report(tmp_path)
    assert report['runs'] == 1
    assert report['groups']['all']['actions']['max'] == 5
    assert report['groups']['all']['tokens']['count'] == 3


def test_tiny_benchmark_exercises_optimizer_without_checkpoint(tmp_path):
    import torch
    torch.set_num_threads(2)
    result = benchmark('configs/tiny.json', batch_size=2, steps=1, warmup=0, optimizer_steps=True)
    assert result['macros_per_second'] > 0
    assert result['optimizer'] == 'muon_adamw'
    assert result['peak_allocated_bytes'] is None
    assert result['parameters']['linear_layers'] == 0
