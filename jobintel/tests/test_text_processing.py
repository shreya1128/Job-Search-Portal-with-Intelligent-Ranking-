from jobintel.processing.dedupe import NearDuplicateIndex, content_hash, hamming, simhash
from jobintel.processing.text import clean_description, fix_mojibake, index_tokens, strip_html, tokenize


def test_html_removed_but_structure_kept():
    out = clean_description("<p>We need <b>Python</b> &amp; SQL.</p><ul><li>3 years exp</li><li>B.Tech</li></ul><script>x()</script>")
    assert "<" not in out and "x()" not in out and "Python & SQL." in out and "• 3 years exp" in out


def test_mojibake_and_glued_sentences():
    assert fix_mojibake("CafÃ© Accomplished\u00c2\u00a0healthcare") == "Café Accomplished healthcare"
    c = clean_description("Research InstituteOur passion. We use JavaScript and LinkedIn. Requirements:Bachelor degree")
    assert "Institute Our" in c and "JavaScript" in c and "LinkedIn" in c and "Requirements: Bachelor" in c


def test_tokenizer_keeps_tech_tokens():
    norms = [t.norm for t in tokenize("C++/C#, .NET, node.js, B.Tech and 5,00,000")]
    assert {"c++", "c#", ".net", "node.js", "b.tech", "5,00,000"} <= set(norms)


def test_index_tokens_pipeline():
    t = index_tokens("Experienced Data Scientists building machine-learning models with Python and C++ (no degree required)")
    assert "cpp" in t and "python" in t and "scientist" in t and "no" in t        # negation kept, C++ normalised, plural stemmed
    assert "and" not in t and "with" not in t                                      # stop-words removed
    assert index_tokens("The Python", remove_stopwords=False)[0] == "the"


def test_simhash_near_vs_far_and_index():
    base = "senior data scientist python machine learning sql tensorflow docker aws kubernetes pandas numpy " * 3
    a, b = index_tokens(base + "lead"), index_tokens(base + "principal")
    c = index_tokens("registered nurse patient care hospital emergency department bls acls " * 3)
    sa, sb, sc = simhash(a), simhash(b), simhash(c)
    assert hamming(sa, sb) <= 3 < hamming(sa, sc)
    ix = NearDuplicateIndex()
    assert ix.add_or_match(1, sa) is None and ix.add_or_match(2, sb) == 1 and ix.add_or_match(3, sc) is None


def test_content_hash_normalisation():
    assert content_hash("Data  Scientist", "ACME, Inc.", "Hello!") == content_hash("data scientist", "acme inc", "hello")
