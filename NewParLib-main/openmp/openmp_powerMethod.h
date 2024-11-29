#include <omp.h>
#include <vector>
#include <cmath>

double PowerMethod_compute_OpenMP(std::vector<std::vector<double>>& A, std::vector<double>& b, int maxIterations, double tolerance) {
    int n = A.size();
    double lambda = 0.0;
    std::vector<double> x(n, 1.0);
    std::vector<double> Ax(n, 0.0);

    for (int iter = 0; iter < maxIterations; iter++) {
        std::fill(Ax.begin(), Ax.end(), 0.0);

        // Parallelize matrix-vector multiplication
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Ax[i] += A[i][j] * x[j];
            }
        }

        // Compute the norm of Ax
        double norm = 0.0;
        #pragma omp parallel for reduction(+:norm)
        for (int i = 0; i < n; i++) {
            norm += Ax[i] * Ax[i];
        }
        norm = sqrt(norm);

        // Normalize x
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            x[i] = Ax[i] / norm;
        }

        // Compute lambda
        lambda = 0.0;
        #pragma omp parallel for reduction(+:lambda)
        for (int i = 0; i < n; i++) {
            lambda += x[i] * Ax[i];
        }

        // Check convergence
        if (iter > 0 && fabs(lambda - lambda) < tolerance) {
            break;
        }
    }

    return lambda;
}
