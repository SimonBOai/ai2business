"""Test-Environment for oneliner."""

from ai2business.macros import oneliner


def test_four_step_search() -> None:

    result = oneliner.TrendSearch.four_step_search(keyword_list=["2019", "2020"])
    assert list(result.keys()) == [
        "get_interest_over_time",
        "get_interest_by_region",
        "get_related_topics",
        "get_related_queries",
    ]
