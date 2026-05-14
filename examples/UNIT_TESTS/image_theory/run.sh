python code/params.py

# run examples with varying facet size
for i in {0..9}; do
	printf "RUNNING TEST CASE $i ========================================================================"
	../../../src/CUDA/sim "inputs/co_params$i.json" "inputs/facets$i.fct" inputs/targets.txt "coRDR/rdr$i"
done

# run example with 20 m facets just as baseline
../../../src/CUDA/sim "inputs/co_params.json" "inputs/facets.fct" inputs/targets.txt "coRDR/rdr20m"

python code/output.py
