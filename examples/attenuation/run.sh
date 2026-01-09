echo -e "\n\n--------- GENERATING MODEL INPUTS -----------\n"
python generate.py

echo -e "\n\n--------- SIMULATING HALFSPACE --------------\n"
../../src/CUDA/sim params/halfspace.json facets/facets.fct params/targets.txt rdr_halfspace

echo -e "\n\n--------- SIMULATING WINDOW -----------------\n"
../../src/CUDA/sim params/window.json facets/facets.fct params/targets.txt rdr_window

echo -e "\n\n--------- GENERATING PLOTS ------------------\n"
python output.py
