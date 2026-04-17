from dataforge.registry import (
    get_assessment_suite,
    get_benchmark,
    register_assessment_suite,
    register_benchmark,
)


def test_register_and_get_assessment_suite():
    @register_assessment_suite("test-suite")
    class DemoSuite:
        pass

    assert get_assessment_suite("test-suite").__name__ == "DemoSuite"


def test_register_and_get_benchmark():
    @register_benchmark("test-benchmark")
    class DemoBenchmark:
        pass

    assert get_benchmark("test-benchmark").__name__ == "DemoBenchmark"
