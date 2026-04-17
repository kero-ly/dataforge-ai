from argparse import Namespace

from experiments.exp5_quality.run_eval import _infer_method


def test_exp5_eval_infers_method_from_dataset_or_model():
    args = Namespace(method=None, eval_type="llm_judge", dataset="/tmp/raw_seed.jsonl", model_path="/tmp/model")
    assert _infer_method(args) == "raw_seed"

    args = Namespace(method=None, eval_type="mt_bench", dataset=None, model_path="/tmp/dataforge_full")
    assert _infer_method(args) == "dataforge_full"
