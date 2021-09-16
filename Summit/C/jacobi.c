#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

// grid size
#define GRIDY 4096
#define GRIDX 4096

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

// smallest permitted change in temperature
#define MAX_TEMP_ERROR 0.02

// initialize grid and boundary conditions
void init(double T[][GRIDY + 2], double T_new[][GRIDY + 2]) {

  int i, j;

  for (i = 0; i <= GRIDX + 1; i++) {
    for (j = 0; j <= GRIDY + 1; j++) {
      T[i][j] = 0.0;
    }
  }

  // these boundary conditions never change throughout run

  // set left side to 0 and right to a linear increase
  for (i = 0; i <= GRIDX + 1; i++) {
    T[i][0] = 0.0;
    T[i][GRIDY + 1] = (128.0 / GRIDX) * i;
  }

  // set top to 0 and bottom to linear increase
  for (j = 0; j <= GRIDY + 1; j++) {
    T[0][j] = 0.0;
    T[GRIDX + 1][j] = (128.0 / GRIDY) * j;
  }
}

void kernel_serial(double T[][GRIDY + 2], double T_new[][GRIDY + 2],
                   int max_iterations, double dt) {

  int iteration = 0;

  // simulation iterations
  while (dt > MAX_TEMP_ERROR && iteration <= max_iterations) {

    // main computational kernel, average over neighbours in the grid
    for (int i = 1; i <= GRIDX; i++)
      for (int j = 1; j <= GRIDY; j++)
        T_new[i][j] =
            0.25 * (T[i + 1][j] + T[i - 1][j] + T[i][j + 1] + T[i][j - 1]);

    // reset dt
    dt = 0.0;

    // compute the largest change and copy T_new to T
    for (int i = 1; i <= GRIDX; i++) {
      for (int j = 1; j <= GRIDY; j++) {
        dt = MAX(fabs(T_new[i][j] - T[i][j]), dt);
        T[i][j] = T_new[i][j];
      }
    }

    // periodically print largest change
    if ((iteration % 100) == 0)
      printf("Iteration %4d, dt %.6f\n", iteration, dt);

    iteration++;
  }
}

void kernel_cpu_omp(double T[][GRIDY + 2], double T_new[][GRIDY + 2],
                    int max_iterations, double dt) {
  int iteration = 0;

  // simulation iterations
  while (dt > MAX_TEMP_ERROR && iteration <= max_iterations) {

// main computational kernel, average over neighbours in the grid
#pragma omp parallel for collapse(2)
    for (int i = 1; i <= GRIDX; i++)
      for (int j = 1; j <= GRIDY; j++)
        T_new[i][j] =
            0.25 * (T[i + 1][j] + T[i - 1][j] + T[i][j + 1] + T[i][j - 1]);

    // reset dt
    dt = 0.0;

// compute the largest change and copy T_new to T
#pragma omp parallel for reduction(max : dt) collapse(2)
    for (int i = 1; i <= GRIDX; i++) {
      for (int j = 1; j <= GRIDY; j++) {
        dt = MAX(fabs(T_new[i][j] - T[i][j]), dt);
        T[i][j] = T_new[i][j];
      }
    }

    // periodically print largest change
    if ((iteration % 100) == 0)
      printf("Iteration %4d, dt %.6f\n", iteration, dt);

    iteration++;
  }
}

void kernel_gpu_omp(double T[][GRIDY + 2], double T_new[][GRIDY + 2],
                    int max_iterations, double dt) {
  int iteration = 0;

  // simulation iterations
  while (dt > MAX_TEMP_ERROR && iteration <= max_iterations) {

// main computational kernel, average over neighbours in the grid
#pragma omp target map(T[:GRIDX + 2][:GRIDY + 2], T_new[:GRIDX + 2][:GRIDY + 2])
#pragma omp teams distribute parallel for collapse(2)
    for (int i = 1; i <= GRIDX; i++)
      for (int j = 1; j <= GRIDY; j++)
        T_new[i][j] =
            0.25 * (T[i + 1][j] + T[i - 1][j] + T[i][j + 1] + T[i][j - 1]);

    // reset dt
    dt = 0.0;

// compute the largest change and copy T_new to T
#pragma omp target map(dt)                                                     \
    map(T[:GRIDX + 2][:GRIDY + 2], T_new[:GRIDX + 2][:GRIDY + 2])
#pragma omp teams distribute parallel for collapse(2) reduction(max : dt)
    for (int i = 1; i <= GRIDX; i++) {
      for (int j = 1; j <= GRIDY; j++) {
        dt = MAX(fabs(T_new[i][j] - T[i][j]), dt);
        T[i][j] = T_new[i][j];
      }
    }

    // periodically print largest change
    if ((iteration % 100) == 0)
      printf("Iteration %4d, dt %.6f\n", iteration, dt);

    iteration++;
  }
}

void kernel_gpu_omp_data(double T[][GRIDY + 2], double T_new[][GRIDY + 2],
                         int max_iterations, double dt) {
  int iteration = 0;
/*  */
#pragma omp target enter data map(to: T[:GRIDX + 2][:GRIDY + 2]) \
                             map(alloc: T_new[:GRIDX][:GRIDY + 2])
  // simulation iterations
  while (dt > MAX_TEMP_ERROR && iteration <= max_iterations) {

// main computational kernel, average over neighbours in the grid
#pragma omp target
#pragma omp teams distribute parallel for collapse(2) 
    for (int i = 1; i <= GRIDX; i++)
      for (int j = 1; j <= GRIDY; j++)
        T_new[i][j] =
            0.25 * (T[i + 1][j] + T[i - 1][j] + T[i][j + 1] + T[i][j - 1]);

    // reset dt
    dt = 0.0;

// compute the largest change and copy T_new to T
#pragma omp target map(dt)
#pragma omp teams distribute parallel for collapse(2) reduction(max : dt)
    for (int i = 1; i <= GRIDX; i++) {
      for (int j = 1; j <= GRIDY; j++) {
        dt = MAX(fabs(T_new[i][j] - T[i][j]), dt);
        T[i][j] = T_new[i][j];
      }
    }

    // periodically print largest change
    if ((iteration % 100) == 0)
      printf("Iteration %4d, dt %.6f\n", iteration, dt);

    iteration++;
  }

#pragma omp target update from(T[:GRIDX + 2][:GRIDY + 2])
#pragma omp target exit data map(delete :T[:GRIDX + 2][:GRIDY + 2]) \
                             map(delete :T_new[:GRIDX][:GRIDY + 2])
}

int main(int argc, char *argv[]) {

  int max_iterations; // maximal number of iterations
  double dt = 100;    // largest change in temperature
//  struct timeval start_time, stop_time, elapsed_time; // timers

//  double T1_new[GRIDX + 2][GRIDY + 2]; // temperature grid
//  double T1[GRIDX + 2][GRIDY + 2];     // temperature grid from last iteration

  double  (*T_new)[GRIDY + 2];   // temperature grid
  double  (*T)[GRIDY + 2]; // temperature grid from last iteration

  int run_cpu=0, run_omp=0, run_omp_gpu=0, run_omp_gpu_data=0;

  if (argc != 3) {
    printf("Usage: %s number_of_iterations kernel_to_run\n", argv[0]);
    printf("kernel_to_run = CPU | CPU_OMP | GPU | GPU_DATA | ALL \n");
    exit(1);
  } else {
    max_iterations = atoi(argv[1]);
  }
  if (!strcmp(argv[2], "ALL")){
    printf("Running all kernels\n\n");
    run_cpu = 1;
    run_omp = 1;
    run_omp_gpu = 1;
    run_omp_gpu_data = 1;
  }
  else if (!strcmp(argv[2], "CPU"))
  {
    printf("Running CPU serial kernel\n\n");
    run_cpu = 1;
  }
  else if (!strcmp(argv[2], "CPU_OMP"))
  {
    printf("Running CPU OpenMP kernel\n\n");
    run_omp = 1;
  }
  else if (!strcmp(argv[2], "GPU"))
  {
    printf("Running GPU OpenMP kernel\n\n");
    run_omp_gpu = 1;
  }
  else if (!strcmp(argv[2], "GPU_DATA"))
  {
    printf("Running GPU OpenMP kernel with minimal data transfer\n\n");
    run_omp_gpu_data = 1;
  }
  else {
    printf("Usage: %s number_of_iterations kernel_to_run\n", argv[0]);
    printf("kernel_to_run = CPU | CPU_OMP | GPU | GPU_DATA | ALL \n");
    printf("Unrecognized kernel %s\n", argv[2]);
    exit(1);
  }
  

  T = (double (*)[GRIDY+2]) malloc((GRIDX+2) *(GRIDY+2) *sizeof(double));
  T_new = (double (*)[GRIDY+2]) malloc((GRIDX+2) *(GRIDY+2) *sizeof(double));

  if (T == NULL || T_new == NULL){
    printf("Error allocating storage for Temperature\n");
    exit(1);
  }

  if(run_cpu){
    init(T, T_new);
    double start = omp_get_wtime();
    kernel_serial(T, T_new, max_iterations, dt);
    double end = omp_get_wtime();
    printf("CPU serial kernel time = %.6f Seconds\n\n", end - start);
  }

  if(run_omp){
    init(T, T_new);
    double start = omp_get_wtime();
    kernel_cpu_omp(T, T_new, max_iterations, dt);
    double end = omp_get_wtime();
    printf("CPU OpenMP kernel time = %.6f Seconds\n\n", end - start);
  }

  if(run_omp_gpu){
    init(T, T_new);
    double start = omp_get_wtime();
    kernel_gpu_omp(T, T_new, max_iterations, dt);
    double end = omp_get_wtime();
    printf("GPU OpenMP offloading kernel time = %.6f Seconds\n\n", end - start);
  }

  if(run_omp_gpu_data){
    init(T, T_new);
    double start = omp_get_wtime();
    kernel_gpu_omp_data(T, T_new, max_iterations, dt);
    double end = omp_get_wtime();
    printf("GPU OpenMP offloading data kernel time = %.6f Seconds\n\n",
           end - start);
  }

  return 0;
}
