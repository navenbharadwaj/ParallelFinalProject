mpicxx -std=c++11 -g -fopenmp -o parallel_fwd_pass parallel_fwd_pass.cpp
num_images=49984
export OMP_NUM_THREADS=1
echo '		{mpi_proc, omp_threads} = {32,1}'
mpiexec -np=32 ./parallel_fwd_pass $num_images


export OMP_NUM_THREADS=2
echo '		{mpi_proc, omp_threads} = {16,2}'
mpiexec -np=16 ./parallel_fwd_pass $num_images

export OMP_NUM_THREADS=4
echo '		{mpi_proc, omp_threads} = {8,4}'
mpiexec -np=8 ./parallel_fwd_pass $num_images

export OMP_NUM_THREADS=8
echo '		{mpi_proc, omp_threads} = {4,8}'
mpiexec -np=4 ./parallel_fwd_pass $num_images

export OMP_NUM_THREADS=16
echo '		{mpi_proc, omp_threads} = {2,16}'
mpiexec -np=2 ./parallel_fwd_pass $num_images

export OMP_NUM_THREADS=32
echo '		{mpi_proc, omp_threads} = {1,32}'
mpiexec -np=1 ./parallel_fwd_pass $num_images


