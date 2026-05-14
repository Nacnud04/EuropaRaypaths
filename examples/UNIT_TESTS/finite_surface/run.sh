python code/params.py

../../../src/CUDA/sim inputs/co_params.json inputs/facets.fct inputs/targets.txt coRDR

for i in {0..9}; do
	../../../src/CUDA/sim inputs/inco_params.json inputs/facets.fct inputs/targets.txt "incoRDR/rdr$i"
done

python code/output.py
