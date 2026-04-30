python code/params.py

for i in {0..99}; do
    ../../../src/CUDA/sim "inputs/params$i.json" "inputs/facets$i.fct" inputs/targets.txt "outputs/trc$i"
done

python code/output.py