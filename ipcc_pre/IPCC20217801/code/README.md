# load environment variables
source /public1/soft/modules/module.sh
module load mpi/intel/17.0.5

# compile with compile.sh
./compile.sh

# run 3 cases with different parameters
./run.sh 1 #run case 1
./run.sh 2 #run case 2
./run.sh 3 #run case 3


# result interpretation
Since we implement the program with 2 Nodes by MPI, there are 2 output of computing time.
The processor 0 is used to integrate the final result, so it takes longer time and its time is the final time.
The example output is as follows:

	process 1: Computing time=203 ms
	process 0: Computing time=256 ms
	There are 0 points' labels are different from original file.

we only consider the time of process 0, so the final time is 256 ms.

