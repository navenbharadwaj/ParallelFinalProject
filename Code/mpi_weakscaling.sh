mpicxx -std=c++11 -g -fopenmp -o parallel_fwd_pass parallel_fwd_pass.cpp
num_images=49984
export OMP_NUM_THREADS=1
echo '		{mpi_proc, omp_threads} = {1,1}'
mpiexec -np=1 ./parallel_fwd_pass 1000


export OMP_NUM_THREADS=1
echo '		{mpi_proc, omp_threads} = {2,1}'
mpiexec -np=2 ./parallel_fwd_pass 2000

export OMP_NUM_THREADS=1
echo '		{mpi_proc, omp_threads} = {4,1}'
mpiexec -np=4 ./parallel_fwd_pass 4000

export OMP_NUM_THREADS=1
echo '		{mpi_proc, omp_threads} = {8,1}'
mpiexec -np=8 ./parallel_fwd_pass 8000

export OMP_NUM_THREADS=1
echo '		{mpi_proc, omp_threads} = {16,1}'
mpiexec -np=16 ./parallel_fwd_pass 16000

export OMP_NUM_THREADS=1
echo '		{mpi_proc, omp_threads} = {32,1}'
mpiexec -np=32 ./parallel_fwd_pass 32000


