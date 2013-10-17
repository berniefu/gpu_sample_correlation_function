#define MIN 0
#define MAX 80
#define INTERVAL_NUM 40  // This three parameters determine the histogram range
#define SAMPLE_NUM 30
#define OBJECT_NUM 4096  // Number of objects to sample
#define TOTAL_NUM 1000000  // Total number of objects in the dataset
#define THREAD_LENGTH 16

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>

/* kernel routine starts with keyword __global__ */

__global__ void vecadd(double* data_x, double* data_y, double* data_z, int* result, int* final)
{
  int index = blockIdx.x * THREAD_LENGTH * THREAD_LENGTH + threadIdx.x * THREAD_LENGTH + threadIdx.y;
  // initializing
  for (int i = index * INTERVAL_NUM; i < index * INTERVAL_NUM + INTERVAL_NUM; ++i) {
    result[i] = 0;
  }

  double x = data_x[index];
  double y = data_y[index];
  double z = data_z[index];

  // calculation
  for (int i = 0; i < OBJECT_NUM; i++) {
    double distSqr = (data_x[i] - x) * (data_x[i] - x) +
                    (data_y[i] - y) * (data_y[i] - y) +
                    (data_z[i] - z) * (data_z[i] - z);
    int bin_num = (int)((sqrt(distSqr) - MIN - 0.00000001) * INTERVAL_NUM / (MAX - MIN));
    if (bin_num < INTERVAL_NUM)
      result[index * INTERVAL_NUM + bin_num]++;
  }
  
  // move to final result
  for (int i = 0; i < INTERVAL_NUM; ++i) {
    atomicAdd(final + i, result[index * INTERVAL_NUM + i]);
  }
}

using namespace std;

void generateData(int num_point, double box_size) {
  ofstream out("data_file", ios::out);
  for (int i = 0; i < num_point; ++i) {
    for (int j = 0; j < 3; j++) {
      out << (double)rand() / RAND_MAX * box_size << " ";
    }
    out << endl;
  }
  out.close();
}

int main(int argc, char * argv[])
{
  // generateData(1000000, 40.0);  // Use this to generate random data

  clock_t start = clock();
  srand(time(NULL));

  double*** data; // SAMPLE_NUM, OBJECT_NUM, dimensions(3)
  double** result; // SAMPLE_NUM, INTERVAL_NUM
  
  data = new double**[SAMPLE_NUM];
  for (int i = 0; i < SAMPLE_NUM; ++i) {
    data[i] = new double*[OBJECT_NUM];
    for (int j = 0; j < OBJECT_NUM; ++j) {
      data[i][j] = new double[3];
    }
  }
  
  result = new double*[SAMPLE_NUM];
  for (int i = 0; i < SAMPLE_NUM; ++i) {
    result[i] = new double[INTERVAL_NUM];
    for (int j = 0; j < INTERVAL_NUM; ++j) {
      result[i][j] = 0;
    }
  }

  // input data
  ifstream in("data_file", ios::in);
  double x, y, z;
  int num_object = 0;
  while ((in >> x) != NULL) {
    in >> y;
    in >> z;
    for (int i = 0; i < SAMPLE_NUM; ++i) {
      bool replace = false;
      int replace_index = -1;
      if (num_object < OBJECT_NUM) {
        replace = true;
        replace_index = num_object;
      } else {
        int draw = (int) (floor((double)rand() / RAND_MAX * (num_object + 1)));
        if (draw < OBJECT_NUM) {
          replace = true;
          replace_index = draw;
        }
      }
      if (replace) {
        data[i][replace_index][0] = x;
        data[i][replace_index][1] = y;
        data[i][replace_index][2] = z;
      }
    }
    num_object++;
  }

  cout << "After sampling: " << ((double)clock() - start) / CLOCKS_PER_SEC << endl;

  double *host_x, *host_y, *host_z;
  int *host_final;
  double *dev_x, *dev_y, *dev_z;
  int *dev_result, *dev_final;

  /* 1. allocate host memory */
  host_x = (double*)malloc( OBJECT_NUM*sizeof(double));
  host_y = (double*)malloc( OBJECT_NUM*sizeof(double) );
  host_z = (double*)malloc( OBJECT_NUM*sizeof(double) );
  host_final = (int*)malloc( INTERVAL_NUM*sizeof(int) );

  /* 2. allocate GPU memory */
  cudaMalloc( &dev_x, OBJECT_NUM*sizeof(double) );
  cudaMalloc( &dev_y, OBJECT_NUM*sizeof(double) ); 
  cudaMalloc( &dev_z, OBJECT_NUM*sizeof(double) ); 
  cudaMalloc( &dev_result, OBJECT_NUM*INTERVAL_NUM*sizeof(int) );
  cudaMalloc( &dev_final, INTERVAL_NUM*sizeof(int) );

  for (int s = 0; s < SAMPLE_NUM; ++s){
    for (int i = 0; i < INTERVAL_NUM; ++i)
      host_final[i] = 0;

    /* initialize input data */
    for (int i = 0 ; i < OBJECT_NUM ; i++) {
      host_x[i] = data[s][i][0];
      host_y[i] = data[s][i][1];
      host_z[i] = data[s][i][2];
    }

    /* 3. Copydata (host_x, host_y and host_z) to GPU */
    cudaMemcpy( dev_x, host_x, OBJECT_NUM*sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_y, host_y, OBJECT_NUM*sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_z, host_z, OBJECT_NUM*sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_final, host_final, INTERVAL_NUM*sizeof(int), cudaMemcpyHostToDevice );

    /* 4. call kernel routine to execute on GPU */
    /* launch 1 thread per vector-element, 1024 threads per block */
    dim3 threadsPerBlock(THREAD_LENGTH, THREAD_LENGTH);
    int numBlocks(OBJECT_NUM / threadsPerBlock.x / threadsPerBlock.y);
    vecadd<<<numBlocks, threadsPerBlock>>>( dev_x, dev_y, dev_z, dev_result, dev_final);

    cout << "After " << s << " samples: " << ((double)clock() - start) / CLOCKS_PER_SEC << endl;

    //!!!!! not n! but interval_num. gpu is wierd though...
    /* transfer results from GPU to CPU */
    cudaMemcpy( host_final, dev_final, INTERVAL_NUM*sizeof(int), cudaMemcpyDeviceToHost );
 
    for (int i = 0; i < INTERVAL_NUM; i++)
      result[s][i] = (host_final[i] / 2);  // Every pair is counted twice.
  }

  /* free host and GPU memory */
  free(host_x);  
  free(host_y);
  free(host_z);
  free(host_final);
  cudaFree(dev_x);
  cudaFree(dev_y);
  cudaFree(dev_z);
  cudaFree(dev_result);
  cudaFree(dev_final);
 
  for (int i = 0; i < INTERVAL_NUM; ++i) {
    double Ex = 0, Ex2 = 0;
    for (int j = 0; j < SAMPLE_NUM; ++j) {
      double reg = result[j][i] * TOTAL_NUM * TOTAL_NUM / OBJECT_NUM / OBJECT_NUM;
      Ex += reg;
      Ex2+= reg * reg;
    }
    Ex /= SAMPLE_NUM;
    Ex2/= SAMPLE_NUM;
    double var = Ex2 - Ex * Ex;
    double std = sqrt(var);
    double err95 = (Ex > 0.000001)? std / sqrt(SAMPLE_NUM - 1) * 2 / Ex : 0;
    cout << Ex << " " << err95 << endl;  // Output mean, and relative error for 95% confidence interval
  }

  cout << "End: " << ((double)clock() - start) / CLOCKS_PER_SEC << endl;

  return( 0 );
}
