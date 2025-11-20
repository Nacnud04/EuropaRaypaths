python "params.py"

echo "=============================="
echo " Running No Convolution Case "
echo "=============================="

./../../CUDA/sim "NoConv.json" "facets.fct" "targets.txt" "NoConv"

echo "=============================="
echo " Running Circular Convolution Case "
echo "=============================="

./../../CUDA/sim "CircConv.json" "facets.fct" "targets.txt" "CircConv"

echo "=============================="
echo " Running Linear Convolution Case "
echo "=============================="

./../../CUDA/sim "LinConv.json" "facets.fct" "targets.txt" "LinConv"

python "assimilate.py"

python "plot.py"