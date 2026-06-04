python code/params.py

rm rdrgrm/*.txt
../../../src/CUDA/sim "inputs/params_5000.json" "inputs/facets.fct" inputs/target5000.txt "rdrgrm/5000"
#../../../src/CUDA/sim "inputs/params_2500.json" "inputs/facets.fct" inputs/target2500.txt "rdrgrm/2500"
#../../../src/CUDA/sim "inputs/params_1000.json" "inputs/facets.fct" inputs/target1000.txt "rdrgrm/1000"
#../../../src/CUDA/sim "inputs/params_0500.json" "inputs/facets.fct" inputs/target0500.txt "rdrgrm/0500"
../../../src/CUDA/sim "inputs/params_0250.json" "inputs/facets.fct" inputs/target0250.txt "rdrgrm/0250"

python code/output.py
