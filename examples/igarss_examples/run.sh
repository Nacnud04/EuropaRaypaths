
echo -e "\n=== STEP 1: GENERATING PARAMETERS & FACETS ===\n"
python code/png2csv.py
python code/params.py

echo -e "\n=== STEP 2: SIMULATE =========================\n"
echo -e "--- STEP 2a: Simulate for flat surface -------\n"
../../src/CUDA/sim params/params.json facets/flat.fct params/target.txt rdrgrm/flat
echo -e "\n\n--- STEP 2b: Simulate for doube ridge surface ------\n"
../../src/CUDA/sim params/params.json facets/ridge.fct params/target.txt rdrgrm/ridge
echo -e "\n\n--- STEP 2c: Simulate for DEM surface ------\n"
../../src/CUDA/sim params/dem.json facets/dem.fct params/dem_target.txt rdrgrm/dem

echo -e "\n=== STEP 3: GENERATE PLOTS ===================\n"
python code/output.py
python code/outputRidgeDouble.py
python code/outputDEM.py
