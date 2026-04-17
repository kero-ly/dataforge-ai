from dataforge.benchmark.tasks.if_eval_lite import _run_check


def test_if_eval_check_helpers():
    assert _run_check("Summary: cache hits help", {"type": "starts_with", "value": "Summary:"})
    assert _run_check("- a\n- b\n- c", {"type": "min_list_items", "value": 3})
    assert not _run_check("contains grape", {"type": "not_contains", "value": "grape"})
