python "params.py"

echo "=============================="
echo " Running No Convolution Case "
echo "=============================="

../../src/CUDA/sim "inputs/NoConv.json" "inputs/facets.fct" "inputs/targets.txt" "NoConv"

echo "==================================="
echo " Running Circular Convolution Case "
echo "==================================="

../../src/CUDA/sim "inputs/CircConv.json" "inputs/facets.fct" "inputs/targets.txt" "CircConv"

echo "================================="
echo " Running Linear Convolution Case "
echo "================================="

../../src/CUDA/sim "inputs/LinConv.json" "inputs/facets.fct" "inputs/targets.txt" "LinConv"

python "assimilate.py"

python "plot.py"
