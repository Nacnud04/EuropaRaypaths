python code/params.py

# run coherent case
printf "================================\n"
printf "COHERENT CASE \n"
printf "================================\n"

# iterate over all 100 area cases
for i in {0..99}; do
    ../../../src/CUDA/sim "inputs/coh_params$i.json" "inputs/facets$i.fct" inputs/targets.txt "coh_outputs/trc$i"
done

# do 50 test cases for averaging of incoherent power returned
for j in {0..49}; do
    printf "================================\n"
    printf "Test case %d \n" $j
    printf "================================\n"
    # iterate over all 100 area cases
    for i in {0..99}; do
        ../../../src/CUDA/sim "inputs/incoh_params$i.json" "inputs/facets$i.fct" inputs/targets.txt "incoh_outputs/trc$i"
    done
    # accumulate results into npy file
    python code/accumulate.py $j
done

python code/output.py