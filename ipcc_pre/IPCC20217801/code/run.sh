set -x
srun -p amd_256 -N 2 -n 2 ./slic $1
#srun -p amd_256 -N 1 ./slic $1
#srun -p amd_256 -N 1 ./SLIC
