#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <math.h>
#include <omp.h>
#include <mpi.h>

using namespace std;

#define image_xy 28
#define image_channels 1

#define conv1_filt_xy 5
#define conv1_filt_count 20
#define conv1_out_xy 24
#define pool1_out_xy 12

#define conv2_filt_xy 5
#define conv2_filt_count 50
#define conv2_out_xy 8
#define pool2_out_xy 4

#define fc1_input_size 800 // 4x4x50 = 800
#define fc1_output_size 500
#define fc2_output_size 10

//#define TRAIN_ON 10
#define PRINT_INSIDE 0

void convolution(double ***conv, double ****wt, double ***out, int conv_xy, int wt_xy, int c, int n)
{
	if(PRINT_INSIDE)
		printf("Inside convolution\n");
	
	//double sum;

	#pragma omp parallel for collapse(3)	

	for (int k_n=0; k_n < n; k_n++)
	{
		for (int i_x=0; i_x < (conv_xy-4); i_x++)
		{	//x stride
			for (int i_y=0; i_y < (conv_xy-4); i_y++)
			{	//y stride
				double sum = 0;
				//sum = 0;
				//#pragma omp parallel for collapse(3) default(shared) reduction(+:sum)
				for (int k_x=0; k_x < wt_xy; k_x++)
				{
				//#pragma omp parallel for collapse(2) default(shared) reduction(+:sum)
					for (int k_y=0; k_y < wt_xy; k_y++)
					{
						//#pragma omp parallel for default(shared) reduction(+:sum)
						for (int i_c=0; i_c < c; i_c++)
                            
						{
							sum += wt[k_x][k_y][i_c][k_n] * conv[k_x + i_x][k_y + i_y][i_c];
                            {for i_d=1; i_d < d; i_d++}
						}
					}
				}
				out[i_x][i_y][k_n] = sum;
			}
		}
	}
}

void relu(double ***conv_out, double ***relu_out, int x_dim, int y_dim, int z_dim)
{
	if(PRINT_INSIDE)
		printf("Inside ReLU\n");

	#pragma omp parallel for collapse(3)
	for (int i=0; i < z_dim; i++)
	{
		for (int j = 0; j < x_dim; ++j)
		{
			for (int k = 0; k < y_dim; ++k)
			{
				if(conv_out[j][k][i] >= 0)
					relu_out[j][k][i] = conv_out[j][k][i];
				else
					relu_out[j][k][i] = 0;

			}
		}
	}
}

double pool_max(double a, double b, double c, double d)
{
	double max1, max2, max;
	if(a <= b)
		max1 = b;
	else
		max1 = a;

	if(c <= d)
		max2 = d;
	else 
		max2 = c;

	if(max1 <= max2)
		max = max2;
	else
		max = max1;

	return max;
}

void pooling(double ***conv, double ***pool, int conv_xy, int c) 
{	
	if(PRINT_INSIDE)
		printf("Inside pooling\n");

	#pragma omp parallel for collapse(3)
	for (int i_c=0; i_c < c; i_c++)
	{
		for (int i_x=0; i_x < conv_xy; i_x = i_x + 2)
		{
			for (int i_y=0; i_y < conv_xy; i_y = i_y + 2)
			{
				pool[i_x/2][i_y/2][c] = pool_max(conv[i_x][i_y][c], conv[i_x + 1][i_y][c], conv[i_x][i_y + 1][c], conv[i_x + 1][i_y + 1][c]);
			}
		}
	}

}

void threeD_to_oneD(double ***input, double *output, int x_dim, int y_dim, int z_dim)
{
	if(PRINT_INSIDE)
		printf("converting to 1-D\n");
    
    #pragma omp parallel for collapse(3)
	int x = 0;

	for (int i = 0; i < z_dim; ++i)
	{
		for (int j = 0; j < x_dim; ++j)
		{
			for (int k = 0; k < y_dim; ++k)
			{
				output[x] = input[j][k][i];
				x++;
			}
		}
	}
}

void fc(double *input, double *output, double **filt, int num_inputs, int num_outputs)
{
	if(PRINT_INSIDE)
		printf("Inside FC\n");

	#pragma omp parallel for
	for (int i = 0; i < num_outputs; ++i)
	{
		double sum2 = 0;

	//#pragma omp parallel for default(shared) reduction(+:sum2)
		for (int j = 0; j < num_inputs; ++j)
		{
			sum2 += filt[i][j] * input[j];
		}
		output[i] = sum2;
	}
}

void fc_relu(double *input, double *output, int num)
{
	if(PRINT_INSIDE)
		printf("Inside FC-ReLU\n");
	#pragma omp parallel for
	for (int i = 1; i < num; ++i)
        
	{
		if(input[i] >= 0)
			output[i] = input[i];
		else
			output[i] = 0;
	}
}

void softmax(double *input, double *output, int num)
{
	if(PRINT_INSIDE)
		printf("Inside Softmax\n");

	double exp_sum = 0;
//#pragma omp parallel for default(shared) reduction(+:exp_sum)
	for (int i = 0; i < num; ++i)
	{
		output[i] = exp(input[i]);
		exp_sum += output[i];
	}
	for (int i = 0; i < num; ++i)
	{
		output[i] = output[i]/exp_sum;
	}
}

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1=i&255;
	ch2=(i>>8)&255;
	ch3=(i>>16)&255;
	ch4=(i>>24)&255;
	return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		printf("Enter the number of images to be processed.\n");
		exit(0);
	}
	int TRAIN_ON = atoi(argv[1]);

	/* ALL PROCESSES HAVE TO INITIALIZE FEATURE-MAP AND WEIGHT ARRAYS */
	MPI_Init(&argc, &argv);
	MPI_Comm comm1 = MPI_COMM_WORLD;

	int mpi_rank, mpi_size;
	MPI_Comm_size(comm1, &mpi_size);
	MPI_Comm_rank(comm1, &mpi_rank);
	int root = 0;
	int images_per_process = TRAIN_ON / mpi_size;
	if(mpi_rank == root)
		printf("Images for one process= %d\n", images_per_process);


	/*double ****node_images;
	node_images = new double***[images_per_process]; 
	for (int i = 0; i < images_per_process; ++i)
	{
		node_images[i] = new double**[28];
		for (int j = 0; j < 28; ++j)
		{
			node_images[i][j] = new double*[28]; // 
			for (int k = 0; k < 28; ++k)
			{
				node_images[i][j][k] = new double[1]; // channel
			}
		}
	}*/

	double *node_images;
	node_images = new double[28*28*images_per_process];

	//printf("Creating arrays...p#%d\n", mpi_rank);
	/* ACTIVATION */
	double ***conv1_out;
	double ***conv1_relu_out;
	double ***conv1_pool_out;

	double ***conv2_out;
	double ***conv2_relu_out;
	double ***conv2_pool_out;

	double *fc1_input;	
	double *fc1_out;
	double *fc1_relu_out;

	double *fc2_out;
	double *softmax_out;	

	/* WEIGHTS */
	double ****conv1_filt;
	double ****conv2_filt;
	double **fc1_filt;
	double **fc2_filt;

	/************************ FEATURE MAP ************************/
	conv1_out = new double**[conv1_out_xy];
	for (int i = 0; i < conv1_out_xy; ++i)
	{
		conv1_out[i] = new double*[conv1_out_xy];
		for (int j = 0; j < conv1_out_xy; ++j)
		{
			conv1_out[i][j] = new double[conv1_filt_count];
		}
	}

	conv1_relu_out = new double**[conv1_out_xy];
	for (int i = 0; i < conv1_out_xy; ++i)
	{
		conv1_relu_out[i] = new double*[conv1_out_xy];
		for (int j = 0; j < conv1_out_xy; ++j)
		{
			conv1_relu_out[i][j] = new double[conv1_filt_count];
		}
	}

	conv1_pool_out = new double**[pool1_out_xy];
	for (int i = 0; i < pool1_out_xy; ++i)
	{
		conv1_pool_out[i] = new double*[pool1_out_xy];
		for (int j = 0; j < pool1_out_xy; ++j)
		{
			conv1_pool_out[i][j] = new double[conv1_filt_count];
		}
	}

	conv2_out = new double**[conv2_out_xy];
	for (int i = 0; i < conv2_out_xy; ++i)
	{
		conv2_out[i] = new double*[conv2_out_xy];
		for (int j = 0; j < conv2_out_xy; ++j)
		{
			conv2_out[i][j] = new double[conv2_filt_count];
		}
	}
	
	conv2_relu_out = new double**[conv2_out_xy];
	for (int i = 0; i < conv2_out_xy; ++i)
	{
		conv2_relu_out[i] = new double*[conv2_out_xy];
		for (int j = 0; j < conv2_out_xy; ++j)
		{
			conv2_relu_out[i][j] = new double[conv2_filt_count];
		}
	}

	conv2_pool_out = new double**[pool2_out_xy];
	for (int i = 0; i < pool2_out_xy; ++i)
	{
		conv2_pool_out[i] = new double*[pool2_out_xy];
		for (int j = 0; j < pool2_out_xy; ++j)
		{
			conv2_pool_out[i][j] = new double[conv2_filt_count];
		}
	}

	fc1_input = new double[fc1_input_size];
	fc1_out = new double[fc1_output_size];
	fc1_relu_out = new double[fc1_output_size];
	fc2_out = new double[fc2_output_size];
	softmax_out = new double[fc2_output_size];

	/************************ MODEL ************************/
	conv1_filt = new double***[conv1_filt_xy]; /* CONV1 */
	for (int i = 0; i < conv1_filt_xy; ++i)
	{
		conv1_filt[i] = new double**[conv1_filt_xy];
		for (int j = 0; j < conv1_filt_xy; ++j)
		{
			conv1_filt[i][j] = new double*[image_channels];
			for (int k = 0; k < image_channels; ++k)
			{
				conv1_filt[i][j][k] = new double[conv1_filt_count];
			}
		}
	}
	for (int i = 0; i < conv1_filt_count; ++i)
	{
		for (int j = 0; j < conv1_filt_xy; ++j)
		{
			for (int k = 0; k < conv1_filt_xy; ++k)
			{
				for (int l = 0; l < image_channels; ++l)
				{
					conv1_filt[j][k][l][i] = rand();
				}
			}
		}
	}

	conv2_filt = new double***[conv2_filt_xy]; /* CONV2 */
	for (int i = 0; i < conv2_filt_xy; ++i)
	{
		conv2_filt[i] = new double**[conv2_filt_xy];
		for (int j = 0; j < conv2_filt_xy; ++j)
		{
			conv2_filt[i][j] = new double*[conv1_filt_count];
			for (int k = 0; k < conv1_filt_count; ++k)
			{
				conv2_filt[i][j][k] = new double[conv2_filt_count];
			}
		}
	}
	for (int i = 0; i < conv2_filt_count; ++i)
	{
		for (int j = 0; j < conv2_filt_xy; ++j)
		{
			for (int k = 0; k < conv2_filt_xy; ++k)
			{
				for (int l = 0; l < conv1_filt_count; ++l)
				{
					conv2_filt[j][k][l][i] = rand();
				}
			}
		}
	}

	fc1_filt = new double*[fc1_output_size]; /* FC1 */
	for (int i = 0; i < fc1_output_size; ++i)
	{
		fc1_filt[i] = new double [fc1_input_size];
	}
	for (int i = 0; i < fc1_output_size; ++i)
	{
		for (int j = 0; j < fc1_input_size; ++j)
		{
			fc1_filt[i][j] = rand();
		}
	}

	fc2_filt = new double*[fc2_output_size]; /* FC2 */
	for (int i = 0; i < fc2_output_size; ++i)
	{
		fc2_filt[i] = new double [fc1_output_size];
	}
	for (int i = 0; i < fc2_output_size; ++i)
	{
		for (int j = 0; j < fc1_output_size; ++j)
		{
			fc2_filt[i][j] = rand();
		}
	}

	double *all_images;
	/************************ PROCESSING ************************/
	if(mpi_rank == root)
	{
		//printf("Hey! root (p#%d) here.\n", mpi_rank);
		string dataset_dir = "/users/naveenbharadwaj/desktop/Parallel Final Project/MNIST Dataset";
		string input_file = dataset_dir + "/train-images-idx3-ubyte";

		all_images = new double[28*28*TRAIN_ON];

		ifstream file (input_file, ios::binary);
		if (file.is_open())
		{
			int magic_number=0;
			int number_of_images=0;
			int n_rows=0;
			int n_cols=0;
			file.read((char*)&magic_number,sizeof(magic_number));
			magic_number= ReverseInt(magic_number);
			file.read((char*)&number_of_images,sizeof(number_of_images));
			number_of_images= ReverseInt(number_of_images);
			file.read((char*)&n_rows,sizeof(n_rows));
			n_rows= ReverseInt(n_rows);
			file.read((char*)&n_cols,sizeof(n_cols));
			n_cols= ReverseInt(n_cols);
			//printf("size of an image = %dx%d\n", n_rows, n_cols);
			//printf("Number of images = %d\n", number_of_images);
			//printf("Magic number = %d\n", magic_number);
			int x = 0; // tracks index of the 1-D array that stores all images
			for(int i=0;i<TRAIN_ON;++i)
			{

				if(i == TRAIN_ON)
					break;
				//printf("\t\timage: %d\n", i);
				for(int r=0;r<n_rows;++r)
				{
					for(int c=0;c<n_cols;++c)
					{
						unsigned char temp=0;
						file.read((char*)&temp,sizeof(temp));
						all_images[x]= (double)temp;
						x++;
						//printf("%lf ", arr[r][c]);
					}
					//printf("\n");
				}
			}
		}
	}

	int send_count = 28*28*images_per_process;
	int recv_count = send_count;
	MPI_Scatter(all_images, send_count, MPI_DOUBLE, node_images, recv_count, MPI_DOUBLE, root, comm1);

	double ****multi_dim_images;
	multi_dim_images = new double***[images_per_process]; 
	for (int i = 0; i < images_per_process; ++i)
	{
		multi_dim_images[i] = new double**[28];
		for (int j = 0; j < 28; ++j)
		{
			multi_dim_images[i][j] = new double*[28]; // 
			for (int k = 0; k < 28; ++k)
			{
				multi_dim_images[i][j][k] = new double[1]; // channel
			}
		}
	}
	int x = 0;
	for (int i = 0; i < images_per_process; ++i)
	{
		for (int j = 0; j < 28; ++j)
		{
			for (int k = 0; k < 28; ++k)
			{
				multi_dim_images[i][j][k][0] = node_images[x];
				x++;
			}
		}
	}


	double ***single_image;
	single_image = new double**[28];
	for (int i = 0; i < 28; ++i)
	{
		single_image[i] = new double*[28];
		for (int j = 0; j < 28; ++j)
		{
			single_image[i][j] = new double[1];
		}
	}
	
	double time_start, time_taken;
  	time_start = MPI_Wtime();

	for (int i = 0; i < images_per_process; ++i)
	{
		//printf("p#%d processing image: %d\n", mpi_rank, i);
		for (int j = 0; j < 28; ++j)
		{
			for (int k = 0; k < 28; ++k)
			{
				single_image[j][k][0] = multi_dim_images[i][j][k][0];
				//printf("%lf ", single_image[j][k][0]);
			}
			//printf("\n");
		}
	
		convolution(single_image, conv1_filt, conv1_out, image_xy, conv1_filt_xy, image_channels, conv1_filt_count); // conv1
		relu(conv1_out, conv1_relu_out, conv1_out_xy, conv1_out_xy, conv1_filt_count); // relu of conv1
		pooling(conv1_relu_out, conv1_pool_out, conv1_out_xy, conv1_filt_count); // pool1

		convolution(conv1_pool_out, conv2_filt, conv2_out, pool1_out_xy, conv2_filt_xy, conv1_filt_count, conv2_filt_count); // conv2
		relu(conv2_out, conv2_relu_out, conv2_out_xy, conv2_out_xy, conv2_filt_count); // relu of conv2
		pooling(conv2_relu_out, conv2_pool_out, conv2_out_xy, conv2_filt_count); // pool2

		threeD_to_oneD(conv2_pool_out, fc1_input, pool2_out_xy, pool2_out_xy, conv2_filt_count); // convert pool2 output from 3D to 1D so that it is compatible with fc1.
		fc(fc1_input, fc1_out, fc1_filt, fc1_input_size, fc1_output_size);
		fc_relu(fc1_out, fc1_relu_out, fc1_output_size);

		fc(fc1_out, fc2_out, fc2_filt, fc1_output_size, fc2_output_size);
		softmax(fc2_out, softmax_out, fc2_output_size);
		double *pred;
		pred = max_element(softmax_out, softmax_out+9);	
	}

	time_taken = MPI_Wtime() - time_start;
	/* Printing timing info */
	double net_time;
	double max_time;
	double min_time;
	MPI_Reduce(&time_taken, &net_time, 1, MPI_DOUBLE, MPI_SUM, root, comm1); // Calculating sum of time taken by each proc
	MPI_Reduce(&time_taken, &max_time, 1, MPI_DOUBLE, MPI_MAX, root, comm1); // Calculating max of time taken by each proc
	MPI_Reduce(&time_taken, &min_time, 1, MPI_DOUBLE, MPI_MIN, root, comm1); // Calculating max of time taken by each proc

	if(mpi_rank == 0)
	{
	 double avg_time = net_time / mpi_size;
	 printf("Average time taken by a proc = %lf\n", avg_time);
	 printf("Maximum time taken by a proc = %lf\n", max_time);
	 printf("Minimum time taken by a proc = %lf\n", min_time);
	}
	
	MPI_Finalize();
	return 0;
}
