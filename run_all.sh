#!/bin/bash
# Regenerates every result of the paper (≈2–3 h on two cores).
set -e
mkdir -p out
python3 run_validation.py
python3 run_sweep.py
for v in "Euler method=euler" "N48 N=48" "N112 N=112" "W12 half_width=1.2"; do python3 run_sweep_variant.py $v; done
python3 run_sensitivity.py
python3 run_invariance.py
python3 run_physio.py
python3 make_tables.py; python3 make_inv_table.py
python3 make_fig1.py; python3 make_fig3.py; python3 make_fig4.py; python3 make_physio.py out/sweep.json
