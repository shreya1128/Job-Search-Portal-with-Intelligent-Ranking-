#!/usr/bin/env python3
"""Write Graphviz .dot state diagrams of every automaton to docs/fsa/ (for the report)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import ROOT
from jobintel.processing import patterns as P

out = ROOT / "docs" / "fsa"
out.mkdir(parents=True, exist_ok=True)
for nfa in (P.build_experience_nfa(), P.build_degree_nfa(), P.build_salary_nfa(), P.build_certification_nfa(), P.build_role_nfa()):
    (out / f"{nfa.name}.dot").write_text(nfa.to_dot(), encoding="utf-8")
    print(f"{nfa.name:14s} states={nfa.n_states:3d} transitions={nfa.n_transitions:3d}  -> docs/fsa/{nfa.name}.dot")
