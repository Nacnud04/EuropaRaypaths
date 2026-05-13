python code/params.py

# iterate over all 100 flat fresnel zone cases
#for i in {0..99}; do
#    ../../../src/CUDA/sim "inputs/params$i.json" "inputs/facets$i.fct" inputs/targets.txt "flatRDR/trc$i"
#done

# iterate over all 100 convex fresnel zone cases
for i in {0..99}; do
    ../../../src/CUDA/sim "inputs/params$i.json" "inputs/facets-conv-$i.fct" inputs/targets.txt "convexRDR/trc$i"
done

python code/output.py
