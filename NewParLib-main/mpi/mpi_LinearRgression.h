#include <mpi.h>
#include <iostream>
#include <vector>

void LinearRegression_fit(double** X, double* y, int n, int m, double* beta, int rank, int size) {
    int local_rows = n / size + (rank < n % size ? 1 : 0);
    int start_row = rank * (n / size) + std::min(rank, n % size);

    std::vector<double> local_XtX(m * m, 0);
    std::vector<double> XtX(m * m, 0);
    std::vector<double> local_XtY(m, 0);
    std::vector<double> XtY(m, 0);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = start_row; k < start_row + local_rows; k++) {
                local_XtX[i * m + j] += X[k][i] * X[k][j];
            }
        }
    }

    MPI_Reduce(local_XtX.data(), XtX.data(), m * m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    for (int i = 0; i < m; i++) {
        for (int k = start_row; k < start_row + local_rows; k++) {
            local_XtY[i] += X[k][i] * y[k];
        }
    }

    MPI_Reduce(local_XtY.data(), XtY.data(), m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Solve XtX * beta = XtY
        // Here, you can use any matrix solver such as Gauss-Jordan or LU Decomposition.
        // This is omitted for brevity.
    }

    MPI_Bcast(beta, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
