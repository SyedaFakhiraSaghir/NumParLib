#include <mpi.h>
#include <vector>
#include <cmath>

double PowerMethod_compute_MPI(std::vector<std::vector<double>>& A, std::vector<double>& b, int maxIterations, double tolerance, int rank, int size) {
    int n = A.size();
    double lambda = 0.0;
    std::vector<double> x(n, 1.0);
    std::vector<double> Ax(n, 0.0);

    for (int iter = 0; iter < maxIterations; iter++) {
        std::fill(Ax.begin(), Ax.end(), 0.0);

        // Compute A * x using MPI parallelization
        for (int i = rank; i < n; i += size) {
            for (int j = 0; j < n; j++) {
                Ax[i] += A[i][j] * x[j];
            }
        }

        // Reduce the Ax vector across all processes
        std::vector<double> Ax_global(n, 0.0);
        MPI_Allreduce(Ax.data(), Ax_global.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        Ax = Ax_global;

        // Compute the norm of Ax
        double local_norm = 0.0;
        for (int i = rank; i < n; i += size) {
            local_norm += Ax[i] * Ax[i];
        }
        double global_norm = 0.0;
        MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        global_norm = sqrt(global_norm);

        // Normalize x
        for (int i = 0; i < n; i++) {
            x[i] = Ax[i] / global_norm;
        }

        // Compute lambda
        double local_lambda = 0.0;
        for (int i = rank; i < n; i += size) {
            local_lambda += x[i] * Ax[i];
        }
        MPI_Allreduce(&local_lambda, &lambda, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Check convergence
        if (rank == 0 && iter > 0 && fabs(lambda - lambda) < tolerance) {
            break;
        }
    }

    return lambda;
}
