set -x
#mpiicc -fp-model double -std=c++11 -qopenmp -Ofast -par-affinity=scatter -march=core-avx2 SLIC.cpp -o slic -D DEBUG -D CLEAR_DEBUG
mpiicc -fp-model double -std=c++11 -qopenmp -Ofast -par-affinity=scatter SLIC.cpp -o slic #-D DEBUG -D CLEAR_DEBUG
