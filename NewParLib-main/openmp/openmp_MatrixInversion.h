#include <omp.h>
#include <vector>

void MatrixInversion_invert_OpenMP(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& A_inv, int n, int maxIter, double tol) {
    A_inv = A;
    std::vector<std::vector<double>> I(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; i++) I[i][i] = 1.0;

    for (int iter = 0; iter < maxIter; iter++) {
        std::vector<std::vector<double>> temp(n, std::vector<double>(n, 0));

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    temp[i][j] += A[i][k] * A_inv[k][j];
                }
            }
        }

        double diff = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:diff)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                diff += (temp[i][j] - I[i][j]) * (temp[i][j] - I[i][j]);
            }
        }

        if (diff < tol) {
            break;
        }
    }
}
