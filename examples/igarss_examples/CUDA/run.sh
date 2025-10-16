
echo "Running code/params.py"
python code/params.py

echo "Running simulation"
../../../CUDA/sim params/params.json facets/flat.fct

echo "Running code/output.py"
python code/output.py