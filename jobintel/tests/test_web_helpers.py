"""Small pure functions behind the web pages: link safety, placeholder employers, card text, pagination, sorting, role names, location parsing."""
import math

import pytest

from jobintel.web import cards
from jobintel.web.backend import Backend
from jobintel.web.forms import Params
from jobintel.web.ui import area_label, employer, safe_url, source_label
from jobintel.web.views.search import _location, friendly_warnings, fit_for, parse_search


@pytest.mark.parametrize("url,ok", [("https://www.linkedin.com/jobs/view/1", True), ("http://a.b/c?d=e", True), ("HTTPS://EXAMPLE.COM/x", True), ("javascript:alert(1)", False),
                                    ("data:text/html,<b>", False), ("//evil.example/x", False), ("/relative/path", False), ("https://", False), ("https://a.b/x y", False),
                                    ("https://a.b/\"onclick=1", False), ("", False), (None, False), ("ftp://a.b/c", False), ("https://a.b/" + "x" * 2100, False)])
def test_safe_url(url, ok):
    assert (safe_url(url) is not None) is ok


def test_employer_hides_generated_names_only():
    assert employer("Company-4c8196") is None and employer("") is None and employer(None) is None
    assert employer("Company-4c8196x") == "Company-4c8196x" and employer(" Acme Ltd ") == "Acme Ltd" and employer("Company Co") == "Company Co"


def test_labels():
    assert area_label("data_ai") == "Data and AI" and area_label("mystery_area") == "Mystery area" and source_label("linkedin") == "LinkedIn dataset"


def test_match_reason_is_built_only_from_engine_reasons():
    r = ["✓ 'python' matched", "✓ 'sql' matched", "✓ relevant passage found", "✓ skill: python", "✓ skill: sql", "✓ location matches Pune", "✓ experience band fits"]
    assert cards.match_reason(r) == "Lists python, sql · in Pune · experience fits the range you asked for"
    assert cards.match_reason(["✓ 'analyst' matched", "✓ relevant passage found"]) == "Mentions analyst"
    assert cards.match_reason([]) is None and cards.match_reason([], browse=True) == "Matches your filters"
    many = [f"✓ skill: s{i}" for i in range(7)]
    assert cards.match_reason(many).endswith("and 3 more")
    assert "✗" not in cards.match_reason(["✓ skill: a", "✗ skill missing: b"])


def test_location_text():
    assert cards.location_text("Pune", ["Pune"], "india") == "Pune, India"
    assert cards.location_text("Pune", ["Pune", "Mumbai", "Delhi"], "india") == "Pune and 2 more"
    assert cards.location_text(None, [], "india") == "India" and cards.location_text(None, [], None) is None


def _card(i, exp):
    return cards.Card(job_id=i, title=f"t{i}", company=None, location=None, experience=None, salary=None, salary_note=None, source="s", excerpt_html=None, exp_min=exp)


def test_sort_cards_is_stable_and_puts_unknown_last():
    cs = [_card(1, 5.0), _card(2, None), _card(3, 1.0), _card(4, 1.0), _card(5, 3.0)]
    assert [c.job_id for c in cards.sort_cards(cs, "exp_low")] == [3, 4, 5, 1, 2]
    assert [c.job_id for c in cards.sort_cards(cs, "exp_high")] == [1, 5, 3, 4, 2]
    assert cards.sort_cards(cs, "relevance") is cs


def test_paginate():
    items = list(range(25))
    p = cards.paginate(items, 3)
    assert p["items"] == [20, 21, 22, 23, 24] and p["pages"] == 3 and (p["first"], p["last"], p["total"]) == (21, 25, 25)
    assert cards.paginate(items, 99)["page"] == 3 and cards.paginate(items, 0)["page"] == 1
    e = cards.paginate([], 1)
    assert e["items"] == [] and e["pages"] == 1 and e["first"] == 0
    linked = cards.with_links(cards.paginate(items, 2), lambda n: f"/p{n}")
    assert linked["prev"] == "/p1" and linked["next"] == "/p3" and linked["numbers"][-1] == (3, "/p3")


@pytest.mark.parametrize("title,family", [("Senior Data Engineer (Azure)", "data engineer"), ("Data Engineer: Data Integration", "data engineer"), ("Lead Data Scientist", "data scientist"),
                                          ("Graduates: English Teacher Abroad (Conversational)", "english teacher abroad"), ("Senior Technical Lead (Data Engineer)", "data engineer"),
                                          ("Accountant", "accountant")])
def test_role_family(title, family):
    assert Backend.role_family(title) == family


def test_location_parsing():
    assert _location("Pune, India", ["India"]) == ("Pune", None)
    assert _location("india", ["India", "Canada"]) == (None, "india")
    assert _location("  ", []) == (None, None)


def test_friendly_warnings_hide_engine_wording():
    out = friendly_warnings(["terms not found in the index (ignored): ['zzz', 'qqq']", "no text terms in the query: showing filtered jobs ranked by the non-text signals"])
    assert out == ["These words do not appear in any posting, so they were ignored: zzz, qqq."]
    assert "index" not in " ".join(friendly_warnings(["none of the query terms occur in the index"]))


def test_parse_search_rules():
    p = Params({"q": "python", "area": "data_ai", "exp": "3-6", "city": "Pune", "sort": "exp_low", "page": "2"})
    prm = parse_search(p, ["emscad"], [])
    assert prm["filters"] == {"city": "Pune", "cluster": "data_ai", "min_experience": 3.0, "max_experience": 6.0} and prm["sort"] == "exp_low" and prm["page"] == 2 and not p.errors
    p = Params({"min_exp": "5", "max_exp": "2"})
    parse_search(p, [], [])
    assert any("larger than" in e for e in p.errors)
    p = Params({"area": "x'; drop table jobs;--", "exp": "99", "sort": "<b>"})
    prm = parse_search(p, [], [])
    assert "cluster" not in prm["filters"] and prm["sort"] == "relevance" and len(p.errors) == 3


class _P:                                                           # a stand-in for UserProfile
    def __init__(self, skills, years): self.skills, self.experience_years = skills, years


def test_fit_for_compares_only_stated_facts():
    f = fit_for(["sql", "excel", "tableau"], (1.0, 3.0), _P({"SQL": 1.0}, 2.0))
    assert f["have"] == ["sql"] and f["lack"] == ["excel", "tableau"] and "fit the 1 to 3 years" in f["experience"]
    assert "above that range" in fit_for([], (1.0, 3.0), _P({}, 9.0))["experience"]
    assert "asks for 5+" in fit_for(["a"], (5.0, None), _P({}, 2.0))["experience"]
    assert fit_for(["a"], (None, None), _P({}, 2.0))["experience"] is None
    assert fit_for([], (None, None), _P({}, 2.0)) is None and fit_for(["a"], (1.0, 2.0), None) is None
