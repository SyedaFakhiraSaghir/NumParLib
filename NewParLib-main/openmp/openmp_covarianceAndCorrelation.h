#include <cmath>
#include <omp.h>

void covarianceMatrix(double** data, int n, int m, double** cov) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = i; j < m; j++) {
            double mean_i = 0, mean_j = 0;
            for (int k = 0; k < n; k++) {
                mean_i += data[k][i];
                mean_j += data[k][j];
            }
            mean_i /= n;
            mean_j /= n;

            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += (data[k][i] - mean_i) * (data[k][j] - mean_j);
            }

            cov[i][j] = sum / (n - 1);
            cov[j][i] = cov[i][j];
        }
    }
}
