python code/params.py

../../../src/CUDA/sim "inputs/none.json" "../image_theory/inputs/facets.fct" "../image_theory/inputs/targets.txt" "outputs/None"
../../../src/CUDA/sim "inputs/blackman.json" "../image_theory/inputs/facets.fct" "../image_theory/inputs/targets.txt" "outputs/Blackman"
../../../src/CUDA/sim "inputs/nuttall.json" "../image_theory/inputs/facets.fct" "../image_theory/inputs/targets.txt" "outputs/Nuttall"

python code/output.py
