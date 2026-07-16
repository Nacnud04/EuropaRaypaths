python code/params.py

# do subsurface simulation
../../src/CUDA/sim data/params.json data/MOLA/KOR_F.fct data/Subsurface/KOR_T_MAPPED.txt rdrgrm_subsurf
# do surface simulation
../../src/CUDA/sim data/params.json data/MOLA/KOR_F.fct data/Subsurface/1Target.txt rdrgrm_surf

python code/output.py

python code/plots.py
