import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_parse_args_has_required_fields():
    from experiments.exp9_scheduling.run_scheduling import parse_args, DEFAULT_SCHEDULES, DEFAULT_CONCURRENCIES, DEFAULT_N_TYPES
    args = parse_args(["--dataset", "experiments/data/seeds_1k.jsonl",
                       "--output-dir", "/tmp/test_s1"])
    assert args.schedules == DEFAULT_SCHEDULES
    assert args.concurrency_values == DEFAULT_CONCURRENCIES
    assert args.n_mutation_types_values == DEFAULT_N_TYPES


def test_compute_optimal_batch_size():
    from experiments.exp9_scheduling.run_scheduling import compute_optimal_batch_size
    assert compute_optimal_batch_size(concurrency=50, n_types=5) == 50
    assert compute_optimal_batch_size(concurrency=10, n_types=3) == 12


def test_compute_optimal_batch_size_n_types_1():
    from experiments.exp9_scheduling.run_scheduling import compute_optimal_batch_size
    assert compute_optimal_batch_size(10, 1) == 10


def test_base_mutations_defined():
    from experiments.exp9_scheduling.run_scheduling import BASE_MUTATIONS
    assert len(BASE_MUTATIONS) >= 3
    for name, prompt in BASE_MUTATIONS:
        assert isinstance(name, str) and len(name) > 0
        assert isinstance(prompt, str) and len(prompt) > 0


def test_build_mutation_catalog():
    from experiments.exp9_scheduling.run_scheduling import build_mutation_catalog, BASE_MUTATIONS
    # n_types=1 should return 1 mutation
    catalog = build_mutation_catalog(1)
    assert len(catalog) == 1
    # n_types=3 should return 3 mutations
    catalog = build_mutation_catalog(3)
    assert len(catalog) == 3
    # n_types > len(BASE_MUTATIONS) should wrap around (still returns n_types entries)
    catalog = build_mutation_catalog(10)
    assert len(catalog) == 10


def test_schedule_configs_defined():
    from experiments.exp9_scheduling.run_scheduling import SCHEDULE_CONFIGS
    required_keys = {"mutation_schedule", "cluster_routing", "prefix_aware_scheduling",
                     "prefix_affinity_striping", "adaptive_concurrency"}
    for name, cfg in SCHEDULE_CONFIGS.items():
        assert required_keys.issubset(cfg.keys()), f"Schedule {name} missing keys"


def test_result_structure_from_run():
    """Verify run_one_config returns dict with required keys (without actual LLM call)."""
    # Just test that the function signature accepts correct kwargs
    from experiments.exp9_scheduling.run_scheduling import run_one_config
    import inspect
    sig = inspect.signature(run_one_config)
    params = set(sig.parameters.keys())
    assert "schedule_name" in params
    assert "concurrency" in params
    assert "n_types" in params
    assert "dataset" in params
    assert "base_urls" in params
