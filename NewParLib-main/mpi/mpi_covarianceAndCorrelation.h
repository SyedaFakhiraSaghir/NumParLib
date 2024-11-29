#include <mpi.h>
#include <cmath>

void covarianceMatrix(double** data, int n, int m, double** cov, int rank, int size) {
    int local_rows = n / size + (rank < n % size ? 1 : 0);
    int start_row = rank * (n / size) + std::min(rank, n % size);

    for (int i = 0; i < m; i++) {
        for (int j = i; j < m; j++) {
            double local_sum = 0;
            double local_mean_i = 0, local_mean_j = 0;

            for (int k = start_row; k < start_row + local_rows; k++) {
                local_mean_i += data[k][i];
                local_mean_j += data[k][j];
            }

            MPI_Allreduce(&local_mean_i, &local_mean_i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_mean_j, &local_mean_j, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            local_mean_i /= n;
            local_mean_j /= n;

            for (int k = start_row; k < start_row + local_rows; k++) {
                local_sum += (data[k][i] - local_mean_i) * (data[k][j] - local_mean_j);
            }

            MPI_Allreduce(&local_sum, &cov[i][j], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            cov[i][j] /= (n - 1);
        }
    }
}
