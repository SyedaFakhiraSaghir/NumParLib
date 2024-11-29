#include <mpi.h>
#include <vector>

void MatrixInversion_invert_MPI(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& A_inv, int n, int maxIter, double tol, int rank, int size) {
    A_inv = A;
    std::vector<std::vector<double>> I(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; i++) I[i][i] = 1.0;

    for (int iter = 0; iter < maxIter; iter++) {
        std::vector<std::vector<double>> temp(n, std::vector<double>(n, 0));

        for (int i = rank; i < n; i += size) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    temp[i][j] += A[i][k] * A_inv[k][j];
                }
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, temp.data(), n * n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double diff = 0.0;
        for (int i = rank; i < n; i += size) {
            for (int j = 0; j < n; j++) {
                diff += (temp[i][j] - I[i][j]) * (temp[i][j] - I[i][j]);
            }
        }

        double global_diff = 0.0;
        MPI_Allreduce(&diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (global_diff < tol) {
            break;
        }
    }
}
