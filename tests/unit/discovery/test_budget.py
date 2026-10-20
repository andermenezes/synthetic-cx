import pytest

from syntheticcx.discovery.budget import DiscoveryBudget
from syntheticcx.errors import ChannelError


@pytest.mark.parametrize(
    "kind,maximum",
    [("actions", 3), ("cycles", 24), ("observations", 24), ("waits", 4), ("open", 1)],
)
def test_budget_is_consumed_before_dispatch(kind, maximum):
    budget = DiscoveryBudget.create(30)
    for _ in range(maximum):
        budget.consume(kind)
    with pytest.raises(ChannelError):
        budget.consume(kind)


def test_cleanup_reserve_is_not_spendable():
    budget = DiscoveryBudget.create(1, reserve_seconds=2)
    with pytest.raises(ChannelError):
        budget.check()
