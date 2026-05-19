python code/params.py

rm rdrgrm/*.txt
../../../src/CUDA/sim "inputs/params.json" "inputs/facets.fct" inputs/target.txt "rdrgrm"

python code/output.py
