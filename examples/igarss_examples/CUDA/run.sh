
echo -e "\n=== STEP 1: GENERATING PARAMETERS & FACETS ===\n"
python code/params.py

echo -e "\n=== STEP 2: SIMULATE =========================\n"
echo -e "--- STEP 2a: Simulate for flat surface -------\n"
../../../CUDA/sim params/params.json facets/flat.fct params/target.txt rdrgrm/flat
echo -e "\n\n--- STEP 2a: Simulate for ridge surface ------\n"
../../../CUDA/sim params/params.json facets/ridge.fct params/layer.txt rdrgrm/ridge

echo -e "\n=== STEP 3: GENERATE PLOTS ===================\n"
python code/output.py
python code/outputRidgeDouble.py