mpicxx -std=c++11 -g -fopenmp -o parallel_fwd_pass parallel_fwd_pass.cpp
num_images=49984

export OMP_NUM_THREADS=2
echo '		{mpi_proc, omp_threads} = {1,2}'
mpiexec -np=1 ./parallel_fwd_pass $num_images

export OMP_NUM_THREADS=4
echo '		{mpi_proc, omp_threads} = {1,4}'
mpiexec -np=1 ./parallel_fwd_pass $num_images

export OMP_NUM_THREADS=8
echo '		{mpi_proc, omp_threads} = {1,8}'
mpiexec -np=1 ./parallel_fwd_pass $num_images

export OMP_NUM_THREADS=16
echo '		{mpi_proc, omp_threads} = {1,16}'
mpiexec -np=1 ./parallel_fwd_pass $num_images

export OMP_NUM_THREADS=32
echo '		{mpi_proc, omp_threads} = {1,32}'
mpiexec -np=1 ./parallel_fwd_pass $num_images 
