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
      T[i][j] = T_new[i][j] = 0.0;
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


void kernel_gpu_teams_parallel_omp(double T[][GRIDY + 2], double T_new[][GRIDY + 2],
                    int max_iterations, double dt) {
  int iteration = 0;

  // simulation iterations
  while (dt > MAX_TEMP_ERROR && iteration <= max_iterations) {

// main computational kernel, average over neighbours in the grid
#pragma omp target map(T[:GRIDX + 2][:GRIDY + 2], T_new[:GRIDX + 2][:GRIDY + 2])
#pragma omp teams distribute  parallel for simd collapse(2)
    for (int i = 1; i <= GRIDX; i++)
      for (int j = 1; j <= GRIDY; j++)
        T_new[i][j] =
            0.25 * (T[i + 1][j] + T[i - 1][j] + T[i][j + 1] + T[i][j - 1]);

    // reset dt
    dt = 0.0;

// compute the largest change and copy T_new to T
#pragma omp target map(dt)                                                     \
    map(T[:GRIDX + 2][:GRIDY + 2], T_new[:GRIDX + 2][:GRIDY + 2])
#pragma omp teams distribute parallel for simd collapse(2) reduction(max : dt)
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


int main(int argc, char *argv[])
{

  int max_iterations; // maximal number of iterations
  double dt = 100;    // largest change in temperature
                      //  struct timeval start_time, stop_time, elapsed_time; // timers

  //  double T1_new[GRIDX + 2][GRIDY + 2]; // temperature grid
  //  double T1[GRIDX + 2][GRIDY + 2];     // temperature grid from last iteration

  double(*T_new)[GRIDY + 2]; // temperature grid
  double(*T)[GRIDY + 2];     // temperature grid from last iteration

  if (argc != 2)
  {
    printf("Usage: %s number_of_iterations\n", argv[0]);
    exit(1);
  }
  else
  {
    max_iterations = atoi(argv[1]);
  }

  printf("Running GPU Teams Parallel OpenMP kernel\n\n");

  T = (double(*)[GRIDY + 2]) malloc((GRIDX + 2) * (GRIDY + 2) * sizeof(double));
  T_new = (double(*)[GRIDY + 2]) malloc((GRIDX + 2) * (GRIDY + 2) * sizeof(double));

  if (T == NULL || T_new == NULL)
  {
    printf("Error allocating storage for Temperature\n");
    exit(1);
  }

  init(T, T_new);
  double start = omp_get_wtime();
  kernel_gpu_teams_parallel_omp(T, T_new, max_iterations, dt);
  double end = omp_get_wtime();
  printf("GPU Teams Parallel OpenMP kernel time = %.6f Seconds\n\n", end - start);

  return 0;
}
