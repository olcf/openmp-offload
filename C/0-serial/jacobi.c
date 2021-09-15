#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

unsigned int n_cells;
unsigned int SIZE;

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

#define T(i, j) (T[(i)*n_cells + (j)])
#define T_new(i, j) (T_new[(i)*n_cells + (j)])
#define T_init(i, j) (T_init[(i)*n_cells + (j)])

// smallest permitted change in temperature
double MAX_RESIDUAL = 1.e-5;

// initialize grid and boundary conditions
void init(double *T, double *T_init) {

  static int first_time = 1;
  static int seed = 0;
  if (first_time == 1) {
    seed = time(0);
    first_time = 0;
  }
  srand(seed);

  for (unsigned i = 0; i <= n_cells + 1; i++) {
    for (unsigned j = 0; j <= n_cells + 1; j++) {
      T(i, j) = (double)rand() / (double)RAND_MAX;
      T_init(i, j) = T(i, j);
    }
  }
}

void kernel_serial(double *T, int max_iterations) {

  int iteration = 0;
  double residual = 1.e5;
  double *T_new;

  T_new = (double *)malloc(SIZE * sizeof(double));

  // simulation iterations
  while (residual > MAX_RESIDUAL && iteration <= max_iterations) {

    // main computational kernel, average over neighbours in the grid
    for (unsigned i = 1; i <= n_cells; i++)
      for (unsigned j = 1; j <= n_cells; j++)
        T_new(i, j) =
            0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));

    // reset dt
    residual = 0.0;

    // compute the largest change and copy T_new to T
    for (unsigned int i = 1; i <= n_cells; i++) {
      for (unsigned int j = 1; j <= n_cells; j++) {
        residual = MAX(fabs(T_new(i, j) - T(i, j)), residual);
        T(i, j) = T_new(i, j);
      }
    }
    iteration++;
  }
  printf("Serial Residual = %.9lf\n", residual);

  free(T_new);
}

int main(int argc, char *argv[]) {
  int max_iterations; // maximal number of iterations

  double *T;      // temperature grid
  double *T_init; // Initial temperature

  if (argc < 3) {
    printf("Usage: %s number_of_iterations number_of_cells\n", argv[0]);
    exit(1);
  } else {
    max_iterations = atoi(argv[1]);
    n_cells = atoi(argv[2]);
  }

  SIZE = (n_cells + 2) * (n_cells + 2);

  T = (double *)malloc(SIZE * sizeof(double));
  T_init = (double *)malloc(SIZE * sizeof(double));


  if (T == NULL || T_init == NULL) {
    printf("Error allocating storage for Temperature\n");
    exit(1);
  }

  init(T, T_init);

  double start = omp_get_wtime();
  kernel_serial(T, max_iterations);
  double end = omp_get_wtime();
  printf("CPU serial kernel time = %.6f Seconds\n\n", end - start);
}
