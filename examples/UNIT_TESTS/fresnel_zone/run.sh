#python code/params.py

# iterate over all 100 flat fresnel zone cases
#for i in {0..99}; do
#    ../../../src/CUDA/sim "inputs/params$i.json" "inputs/facets$i.fct" inputs/targets.txt "flatRDR/trc$i"
#done

# do 10 test cases for averaging of incoherent power returned
for j in {0..10}; do
    printf "TEST CASE %d ======================================\n" $j
    # iterate over all 100 flat fresnel zone cases (incoherent)
    for i in {0..99}; do
        ../../../src/CUDA/sim "inputs/inc-params$i.json" "inputs/facets$i.fct" inputs/targets.txt "inco_flatRDR/trc$i"
    done
    python code/accumulate.py $j
done

# iterate over all 100 convex fresnel zone cases
#for i in {0..99}; do
#    ../../../src/CUDA/sim "inputs/params$i.json" "inputs/facets-conv-$i.fct" inputs/targets.txt "convexRDR/trc$i"
#done

python code/output.py
