# tests/test_fallback_client.py
"""Tests for the FallbackClient with auto-failover."""
from dataforge.clients.fallback import FallbackClient


class MockClient:
    """Mock LLM client that can be configured to fail."""

    def __init__(self, responses=None, fail_count=0):
        self.model = "mock-model"
        self.calls = 0
        self._responses = responses or ["default response"]
        self._fail_count = fail_count
        self._total_calls = 0

    async def generate(self, prompt, **kwargs):
        self._total_calls += 1
        if self._total_calls <= self._fail_count:
            raise ConnectionError(f"Simulated failure #{self._total_calls}")
        idx = self.calls % len(self._responses)
        self.calls += 1
        return self._responses[idx]


async def test_primary_success():
    """When primary works, fallback is never used."""
    primary = MockClient(responses=["primary ok"])
    fallback = MockClient(responses=["fallback ok"])
    client = FallbackClient(primary=primary, fallback=fallback)

    result = await client.generate("test")
    assert result == "primary ok"
    assert primary.calls == 1
    assert fallback.calls == 0
    assert not client.using_fallback


async def test_failover_after_max_failures():
    """After max_failures consecutive errors, switch to fallback."""
    primary = MockClient(fail_count=100)  # always fail
    fallback = MockClient(responses=["fallback ok"])
    client = FallbackClient(primary=primary, fallback=fallback, max_failures=2)

    # Call 1: primary fails (1/2), falls through to fallback
    result1 = await client.generate("test1")
    assert result1 == "fallback ok"

    # Call 2: primary fails again (2/2), triggers using_fallback
    result2 = await client.generate("test2")
    assert result2 == "fallback ok"
    assert client.using_fallback  # now 2 >= 2

    # Call 3: skips primary entirely, goes straight to fallback
    result3 = await client.generate("test3")
    assert result3 == "fallback ok"


async def test_primary_recovers():
    """After recovery_after successful fallback calls, retry primary."""
    primary = MockClient(fail_count=2)  # fails first 2 calls, then works
    fallback = MockClient(responses=["fallback ok"])
    client = FallbackClient(
        primary=primary, fallback=fallback,
        max_failures=2, recovery_after=3,
    )

    # Calls 1-2: primary fails, falls through to fallback each time
    await client.generate("t1")
    await client.generate("t2")
    assert client.using_fallback

    # Call 3: using_fallback=True, goes to fallback. After success,
    # _fallback_successes reaches 3 (= recovery_after), counters reset.
    await client.generate("t3")
    assert not client.using_fallback  # counters were reset

    # Call 4: tries primary again (which now works after 2 total fails)
    result = await client.generate("t4")
    assert result == "default response"


async def test_using_fallback_property():
    client = FallbackClient(
        primary=MockClient(), fallback=MockClient(),
        max_failures=3,
    )
    assert not client.using_fallback
    client._consecutive_failures = 3
    assert client.using_fallback


async def test_model_attribute():
    primary = MockClient()
    primary.model = "gpt-4o"
    fallback = MockClient()
    client = FallbackClient(primary=primary, fallback=fallback)
    assert client.model == "gpt-4o"


async def test_primary_reset_on_success():
    """Successful primary call resets failure counter."""
    primary = MockClient(responses=["ok"])
    fallback = MockClient()
    client = FallbackClient(primary=primary, fallback=fallback, max_failures=3)

    client._consecutive_failures = 2  # one more failure would trigger fallback
    result = await client.generate("test")
    assert result == "ok"
    assert client._consecutive_failures == 0  # reset on success
