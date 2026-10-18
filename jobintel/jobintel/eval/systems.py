"""The systems that are compared.  Each system maps a query to a ranked list of job ids (+ the time it took).
Core (as in the brief): keyword baseline, language model, DFR, hybrid IR.  Extras: bigram LM, Jelinek-Mercer and large-mu Dirichlet,
DFR InL2, passage model, and hybrid ablations (no PageRank / text only / equal weights) - the evidence for 'does PageRank help?'."""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from jobintel.ir.engine import SearchEngine
from jobintel.ir.ranker import RankerConfig

System = Callable[[str], Tuple[List[int], float]]
POOL_SYSTEMS = ["keyword_baseline", "lm_dirichlet", "dfr_pl2", "hybrid", "hybrid_no_pagerank"]     # whose top-10 are labelled first (477 jobs / 20 queries);
# other systems (bigram LM, smoothing variants, ...) are added later with  make_judgment_sheet.py --extend labeled.xlsx --systems lm_bigram
MAIN_SYSTEMS = ["keyword_baseline", "lm_dirichlet", "dfr_pl2", "hybrid"]                                          # the comparison asked for in the brief


def build_systems(index, docstore, extractor=None, lexicon=None, k: int = 10, **filters) -> Dict[str, System]:
    mk = lambda cfg: SearchEngine(index, docstore, extractor, cfg, lexicon)
    base, jm, mu2000, inl2 = mk(RankerConfig()), mk(RankerConfig(lm_smoothing="jm")), mk(RankerConfig(mu=2000.0)), mk(RankerConfig(dfr_model="InL2"))
    zero = lambda *names: {n: 0.0 for n in names}

    def run(engine, model, weights=None) -> System:
        def f(q: str):
            r = engine.search(q, k=k, model=model, weights=weights, explain=False, **filters)
            return [x.job_id for x in r.results], float(r.timings_ms.get("total", 0.0))
        return f

    return {
        "keyword_baseline": run(base, "baseline"),
        "lm_dirichlet": run(base, "lm"),
        "lm_jm": run(jm, "lm"),
        "lm_dirichlet_mu2000": run(mu2000, "lm"),
        "lm_bigram": run(base, "lm_bigram"),
        "dfr_pl2": run(base, "dfr"),
        "dfr_inl2": run(inl2, "dfr"),
        "passage": run(base, "passage"),
        "hybrid": run(base, "hybrid"),
        "hybrid_no_pagerank": run(base, "hybrid", zero("pagerank")),
        "hybrid_text_only": run(base, "hybrid", zero("skill", "location", "experience", "pagerank")),
        "hybrid_equal_weights": run(base, "hybrid", {n: 1.0 for n in ("lm", "dfr", "passage", "skill", "location", "experience", "pagerank")}),
    }
