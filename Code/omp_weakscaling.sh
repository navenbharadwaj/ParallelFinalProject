mpicxx -std=c++11 -g -fopenmp -o parallel_fwd_pass parallel_fwd_pass.cpp
num_images=49984
export OMP_NUM_THREADS=1
echo '		{mpi_proc, omp_threads} = {1,1}'
mpiexec -np=1 ./parallel_fwd_pass 1000


export OMP_NUM_THREADS=2
echo '		{mpi_proc, omp_threads} = {1,2}'
mpiexec -np=1 ./parallel_fwd_pass 2000

export OMP_NUM_THREADS=4
echo '		{mpi_proc, omp_threads} = {1,4}'
mpiexec -np=1 ./parallel_fwd_pass 4000

export OMP_NUM_THREADS=8
echo '		{mpi_proc, omp_threads} = {1,8}'
mpiexec -np=1 ./parallel_fwd_pass 8000

export OMP_NUM_THREADS=16
echo '		{mpi_proc, omp_threads} = {1,16}'
mpiexec -np=1 ./parallel_fwd_pass 16000

export OMP_NUM_THREADS=32
echo '		{mpi_proc, omp_threads} = {1,32}'
mpiexec -np=1 ./parallel_fwd_pass 32000


