""" Abstract """

from ai2business.macros import oneliner


def test_four_step_search() -> None:

    result = oneliner.TrendSearch.four_step_search(keyword_list=["2019", "2020"])
    assert list(result.keys()) == [
        "interest_over_time",
        "interest_by_region",
        "related_topics",
        "related_queries",
    ]
