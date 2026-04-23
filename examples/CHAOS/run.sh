printf "\n===============================================\n"
printf "Generate terrain\n"
python code/setup.py

printf "\n===============================================\n"
printf "Generate params.json\n"
python code/params.py

printf "\n===============================================\n"
printf "Simulate\n"
../../src/CUDA/sim inputs/params.json inputs/facets.fct inputs/targets.txt radargram/

printf "\n===============================================\n"
printf "Compile and focus output\n"
python code/output.py
