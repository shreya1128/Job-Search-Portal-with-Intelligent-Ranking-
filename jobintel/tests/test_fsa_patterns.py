import pytest
from jobintel.processing import patterns as P
from jobintel.processing.fsa import AhoCorasick, TokenNFA, select_longest
from jobintel.processing.text import tokenize


def test_spec_example(extractor):
    r = extractor.extract("2 years of experience in Python")
    assert (r["experience"]["min"], r["experience"]["max"]) == (2, 2)
    assert [s["name"] for s in r["skills"]] == ["python"]


@pytest.mark.parametrize("text,lo,hi", [
    ("3-5 years of experience", 3, 5), ("minimum 5 years experience", 5, None), ("5+ yrs exp", 5, None),
    ("Two years' experience", 2, 2), ("at least two years experience", 2, None), ("up to 3 years exp", 0, 3),
    ("experience: 4 to 6 years", 4, 6), ("5-3 years of experience", 3, 5)])
def test_experience_variants(extractor, text, lo, hi):
    e = extractor.extract(text)["experience"]
    assert e and (e["min"], e["max"]) == (lo, hi)


@pytest.mark.parametrize("text", ["Founded 2 years ago", "5 years warranty on all products", "20121-3 years", "I have 3 apples"])
def test_experience_false_positives_rejected(extractor, text):
    assert extractor.extract(text)["experience"] is None


def test_degree_and_field(extractor):
    d = extractor.extract("Master of Science in statistics or Bachelor's degree in Computer Science")["education"]
    assert [(x["level"], x["field"]) for x in d] == [("master", "statistics"), ("bachelor", "computer science")]
    assert extractor.extract("B.Tech/MCA")["education"][0]["level"] == "bachelor"
    assert extractor.best_education(extractor.extract("Bachelor or Master degree")["education"]).startswith("bachelor")


@pytest.mark.parametrize("text,lo,hi,cur,per", [
    ("CTC 8-12 LPA", 800000, 1200000, "INR", "year"), ("Rs. 5,00,000 per annum", 500000, 500000, "INR", "year"),
    ("$1,800to$2,200Monthly", 1800, 2200, "USD", "month"), ("salary 50k-70k per year", 50000, 70000, None, "year")])
def test_salary(extractor, text, lo, hi, cur, per):
    s = extractor.extract(text)["salary"]
    assert (s["min"], s["max"], s["currency"], s["period"]) == (lo, hi, cur, per)


def test_salary_not_triggered_by_plain_numbers(extractor):
    assert extractor.extract("5 years of experience, team of 12 people, 3 openings")["salary"] is None


def test_certification_role_employment_level(extractor):
    r = extractor.extract("Senior Machine Learning Engineer, AWS certified solutions architect, valid driver's license. Full-time.")
    assert "Senior Machine Learning Engineer" in r["roles"]
    assert any("certified" in c.lower() for c in r["certifications"]) and any("license" in c.lower() for c in r["certifications"])
    assert r["employment_type"] == "full-time" and r["level"]["name"] == "senior"


def test_query_example(extractor):
    r = extractor.extract("Entry level machine learning jobs in Pune")
    assert r["level"]["name"] == "entry" and (r["level"]["min"], r["level"]["max"]) == (0, 2)
    assert [s["name"] for s in r["skills"]] == ["machine learning"] and r["locations"] == ["Pune"]


def test_ambiguous_skill_words(extractor):
    names = lambda t: [s["name"] for s in extractor.extract(t)["skills"]]
    assert "spark" in names("Experience with Spark and Hive") and "spark" not in names("a spark of creativity")
    assert "excel" not in names("We excel at teamwork") and "excel" in names("Proficient in Excel")
    assert "r" not in names("Part C and plan R") and "c" not in names("Part C")


def test_aho_corasick_maximal_munch_and_barriers():
    ac = AhoCorasick()
    for phrase, payload in {"machine learning": "ml", "learning": "l", "deep learning": "dl", "ci/cd": "cicd"}.items():
        from jobintel.processing.patterns import key_tokens
        ac.add(key_tokens(phrase), payload)
    ac.build()
    toks = tokenize("Machine Learning, deep-learning and CI/CD but machine, learning apart")
    got = [(toks[s].raw, p) for s, e, p in select_longest(ac.scan(toks))]
    assert ("Machine", "ml") in got and ("deep", "dl") in got and ("CI", "cicd") in got
    assert sum(1 for _, p in got if p == "ml") == 1          # 'machine, learning' is split by a comma barrier


def test_nfa_structure_and_dot_export():
    nfa = P.build_experience_nfa()
    assert nfa.n_states > 10 and nfa.n_transitions > 15 and any(t == "exp_explicit" for t, _ in nfa.accepting.values())
    dot = nfa.to_dot()
    assert dot.startswith('digraph "experience"') and "doublecircle" in dot
