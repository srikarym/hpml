#include <time.h>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdio.h>
#define BILLION 1E9
#define BLOCK_SIZE 4

using namespace std;

int C = 3, H = 1024, W = 1024;
int K = 64, FH = 3, FW = 3;


#define checkcuDNN(expression)                               \
{                                                            \
	cudnnStatus_t status = (expression);                     \
	if (status != CUDNN_STATUS_SUCCESS)                      \
	{														 \
		cerr << "Error on line " << __LINE__ << ": "      	 \
				<< cudnnGetErrorString(status) << endl; 	 \
		exit(EXIT_FAILURE);                               	 \
	}                                                        \
}



void print_stats(double ci, double td, double runtime, double th, double co, char* kernel) 
{
	cout << fixed << setprecision(6);
	cout << "I = checksum: " << ci << endl;
	cout << "Copy host->dev " << kernel << " " << td <<" sec" << endl;
	cout << "time " << kernel << " " <<runtime <<" sec" << endl;
	cout << "Copy dev->host " << kernel << " " << th <<" sec" << endl;
	cout << "CUDA O = checksum " << co << endl;
	cout << "" <<endl;
}


void init_3d_kernel(double *h_input) 
{

	for(int channel=0;channel<C;channel++)
	{
		for(int height=0; height<H; height++)
		{
			for(int width=0; width<W; width++)
			{
				h_input[(channel*W*H)+(height*W)+width]= channel * (width+height);
			}
		}
	}
}


void init_4d_kernel(double *h_filter) 
{

	for(int k=0;k<K;k++)
	{
		for(int channel=0;channel<C;channel++)
		{
			for(int height=0; height<FH; height++)
			{
				for(int width=0; width<FW; width++)
				{
					h_filter[(k*C*FW*FH)+(channel*FW*FH)+(height*FW)+width] = (channel+k)*(width+height);
				}
			}
		}
	}
}


double find_checksum(double * output, int K)
{
    double checksum = 0.0;
    for (int k=0; k<K; k++)
    {
        for(int row=0; row<H; row++)
        {
            for(int col=0; col<W; col++)
            {
                checksum+=output[(k*H*W)+(row*W)+col];
            }
        }
    }
    return checksum;
}



__global__ void tiled_conv_2d(double *in, int C, int H, int W,
									 double *filter, int K, int FH, int FW,
									 double *out) 
{
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int TW = BLOCK_SIZE + (FW / 2) * 2;
	int TH = BLOCK_SIZE + (FH / 2) * 2;
	extern __shared__ double tile[];

	if (k < K && i < H && j < W) 
	{

		int tj = threadIdx.y + FW / 2;
		int ti = threadIdx.x + FH / 2;

		for (int c = 0; c < C; ++c) 
		{
			
			tile[tj + ti * TW + c * TH * TW] = in[j + i * W + c * H * W];

			// Top
			if (threadIdx.x == 0) 
			{
				if (i > 0)
				{
					tile[tj + (ti - 1) * TW + c * TH * TW] =  in[j + (i - 1) * W + c * H * W];
				}
				else
				{
					tile[tj + (ti - 1) * TW + c * TH * TW] = 0.0;
				}

				// Top Left Corner
				if (threadIdx.y == 0) 
				{
					if (j > 0 && i > 0)
					{
						tile[(tj - 1) + (ti - 1) * TW + c * TH * TW] =  in[(j - 1) + (i - 1) * W + c * H * W];
					}
					else
					{
						tile[(tj - 1) + (ti - 1) * TW + c * TH * TW] = 0.0;
					}

				}
			}

			// Right
			if (threadIdx.y == BLOCK_SIZE - 1) 
			{
				if (j < W - 1)
				{
					tile[(tj + 1) + ti * TW + c * TH * TW] =in[(j + 1) + i * W + c * H * W];
				}

				else
				{
					tile[(tj + 1) + ti * TW + c * TH * TW] = 0.0;
				}

				// Top Right Corner
				if (threadIdx.x == 0) 
				{
					if (j < W - 1 && i > 0)
					{
						tile[(tj + 1) + (ti - 1) * TW + c * TH * TW] =  in[(j + 1) + (i - 1) * W + c * H * W];
					}
					else
					{
						tile[(tj + 1) + (ti - 1) * TW + c * TH * TW] = 0.0;
					}
				}
			}

			// Bottom
			if (threadIdx.x == BLOCK_SIZE - 1) 
			{
				if (i < H - 1)
				{
					tile[tj + (ti + 1) * TW + c * TH * TW] =   in[j + (i + 1) * W + c * H * W];
				}
				else
				{
					tile[tj + (ti + 1) * TW + c * TH * TW] = 0.0;
				}

				// Bottom Right Corner
				if (threadIdx.y == BLOCK_SIZE - 1) 
				{
					if (j < W - 1 && i < H - 1)
					{
						tile[(tj + 1) + (ti + 1) * TW + c * TH * TW] =  in[(j + 1) + (i + 1) * W + c * H * W];
					}
					else
					{
						tile[(tj + 1) + (ti + 1) * TW + c * TH * TW] = 0.0;

					}
				}
			}

			// Left
			if (threadIdx.y == 0) 
			{
				if (j > 0)
				{
					tile[(tj - 1) + ti * TW + c * TH * TW] =  in[(j - 1) + i * W + c * H * W];
				}
				else
				{
					tile[(tj - 1) + ti * TW + c * TH * TW] = 0.0;
				}

				// Bottom Left Corner
				if (threadIdx.x == BLOCK_SIZE - 1) 
				{
					if (j > 0 && i < H - 1)
					{
						tile[(tj - 1) + (ti + 1) * TW + c * TH * TW] = in[(j - 1) + (i + 1) * W + c * H * W];
					}
					else
					{
						tile[(tj - 1) + (ti + 1) * TW + c * TH * TW] = 0.0;
					}
				}
			}
		}

		__syncthreads();

		int out_idx = j + i * W + k * H * W;
		
		int row = i - (FH / 2), col = j - (FW / 2);
		double conv_val = 0.0;

		for (int c = 0; c < C; ++c) {
			for (int fh = 0; fh < FH; ++fh) 
			{
				for (int fw = 0; fw < FW; ++fw) 
				{

					if (col + fw >= 0 && col + fw < W && row + fh >= 0 && row + fh < H) 
					{
						int in_idx = (col + fw) + (row + fh) * W + c * H * W;

						int f_idx = (FW - 1 - fw) + (FH - 1 - fh) * FW + c * FH * FW + k * C * FH * FW;
						conv_val += in[in_idx] * filter[f_idx];
					}
					
				}
			}
		}

		out[out_idx] = conv_val;


	}
}

double c1() 
{
	
	int input_size = C * H * W * sizeof(double);
	int filter_size = K * C * FH * FW * sizeof(double);
	int output_size = H * W * K * sizeof(double);

	double *input = NULL, *filter = NULL, *output = NULL;
	struct timespec start, end;

	cudaMalloc(&input, input_size);
	cudaMalloc(&filter, filter_size);
	cudaMalloc(&output, output_size);

	double* h_input = (double*) malloc(input_size);
	double* h_filter = (double*) malloc(filter_size);
	double* h_output = (double*) malloc(output_size);
	char kernel[] = "kernel";


	init_3d_kernel(h_input);
	init_4d_kernel(h_filter);

	clock_gettime(CLOCK_MONOTONIC, &start);
	cudaMemcpy(input, h_input, input_size, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &end);

	double copy_to_device = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;


	cudaMemcpy(filter, h_filter, filter_size, cudaMemcpyHostToDevice);

	

	dim3 num_threads(BLOCK_SIZE, BLOCK_SIZE, K);
	dim3 num_blocks(H / BLOCK_SIZE, W / BLOCK_SIZE);


	int T = C * (BLOCK_SIZE + (FH / 2) * 2) * (BLOCK_SIZE + (FW / 2) * 2);
	int TS = T * sizeof(double);

	clock_gettime(CLOCK_MONOTONIC, &start);
	
	tiled_conv_2d<<<num_blocks, num_threads, TS>>>(input, C, H, W,filter, K, FH, FW,output);
	
	cudaDeviceSynchronize();
	
	clock_gettime(CLOCK_MONOTONIC, &end);

	double runtime = (end.tv_sec - start.tv_sec) +(end.tv_nsec - start.tv_nsec) / BILLION;


	clock_gettime(CLOCK_MONOTONIC, &start);
	cudaMemcpy(h_output, output, output_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &end);

	double copy_to_host = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;
	double checksum_I = find_checksum(h_input, C);
	double checksum_O = find_checksum(h_output, K);

	print_stats(checksum_I, copy_to_device, runtime , copy_to_host, checksum_O, kernel);

	free(h_input);
	free(h_filter);
	free(h_output);
	cudaFree(input);
	cudaFree(filter);
	cudaFree(output);

	return runtime;
}


double c2() 
{


	int input_size = C * H * W * sizeof(double);
	int filter_size = K * C * FH * FW * sizeof(double);
	int output_size = H * W * K * sizeof(double);

	double *input = NULL, *filter = NULL, *output = NULL;
	struct timespec start, end;

	cudaMalloc(&input, input_size);
	cudaMalloc(&filter, filter_size);
	cudaMalloc(&output, output_size);

	double* h_input = (double*) malloc(input_size);
	double* h_filter = (double*) malloc(filter_size);
	double* h_output = (double*) malloc(output_size);
	char kernel[] = "cudnn";

	init_3d_kernel(h_input);
	init_4d_kernel(h_filter);
	

	clock_gettime(CLOCK_MONOTONIC, &start);
	cudaMemcpy(input, h_input, input_size, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &end);

	double copy_to_device = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;


	cudaMemcpy(filter, h_filter, filter_size, cudaMemcpyHostToDevice);

	cudnnHandle_t cudnn;
	checkcuDNN(cudnnCreate(&cudnn));

	cudnnTensorDescriptor_t input_descriptor;
	checkcuDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkcuDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W));

	cudnnFilterDescriptor_t filter_descriptor;
	checkcuDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
	cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW);

	cudnnTensorDescriptor_t output_descriptor;
	checkcuDNN(cudnnCreateTensorDescriptor(&output_descriptor));
	checkcuDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W));

	cudnnConvolutionDescriptor_t convolution_descriptor;
	checkcuDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
	checkcuDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor, FH / 2, FW / 2, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

	cudnnConvolutionFwdAlgo_t convolution_algorithm;
	checkcuDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convolution_algorithm));

	size_t workspace_size = 0;
	checkcuDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, convolution_algorithm, &workspace_size));

	void  *workspace;
	cudaMalloc(&workspace, workspace_size);

	double alpha = 1.0, beta = 0.0;

	clock_gettime(CLOCK_MONOTONIC, &start);

	checkcuDNN(cudnnConvolutionForward(cudnn, &alpha, input_descriptor, input, filter_descriptor, filter, convolution_descriptor, convolution_algorithm, workspace, workspace_size, &beta, output_descriptor, output));

	clock_gettime(CLOCK_MONOTONIC, &end);

	double runtime = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;

	clock_gettime(CLOCK_MONOTONIC, &start);
	cudaMemcpy(h_output, output, output_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &end);

	double copy_to_host = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;
	double checksum_I = find_checksum(h_input, C);
	double checksum_O = find_checksum(h_output, K);

	print_stats(checksum_I, copy_to_device, runtime , copy_to_host, checksum_O, kernel);

	cudaFree(input);
	cudaFree(filter);
	cudaFree(output);
	cudaFree(workspace);

	free(h_input);
	free(h_filter);
	free(h_output);

	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyFilterDescriptor(filter_descriptor);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);
	cudnnDestroy(cudnn);

	return runtime;
}


int main() {
	

	int repetitions = 5;

	double timeConv = 0.0, timecuDNN = 0.0; 

	for (int i = 0; i < repetitions; i++)
	{

		cout << "Repetition "<< i+1 << " out of "<<repetitions<<endl;

		timeConv += c1();

		timecuDNN += c2();

	}

	timeConv /= repetitions;
	timecuDNN /= repetitions;

	printf("\n\n <Time>: Conv %lf sec cuDNN %lf sec\n", timeConv, timecuDNN);

	return 0;

}
