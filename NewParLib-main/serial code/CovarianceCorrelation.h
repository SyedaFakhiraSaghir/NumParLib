#include <cmath>

class CovarianceCorrelation {
public:
    static void covarianceMatrix(double** data, int n, int m, double** cov) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double mean_i = 0.0, mean_j = 0.0;
                for (int k = 0; k < n; k++) {
                    mean_i += data[k][i];
                    mean_j += data[k][j];
                }
                mean_i /= n;
                mean_j /= n;

                double cov_ij = 0.0;
                for (int k = 0; k < n; k++) {
                    cov_ij += (data[k][i] - mean_i) * (data[k][j] - mean_j);
                }
                cov_ij /= (n - 1);
                cov[i][j] = cov_ij;
            }
        }
    }

static void correlationMatrix(double** data, int n, int m, double** corr) {
        double stddev[m];
        for (int i = 0; i < m; i++) {
            double mean_i = 0.0;
            for (int k = 0; k < n; k++) {
                mean_i += data[k][i];
            }
            mean_i /= n;

            double var_i = 0.0;
            for (int k = 0; k < n; k++) {
                var_i += (data[k][i] - mean_i) * (data[k][i] - mean_i);
            }
            var_i /= (n - 1);
            stddev[i] = sqrt(var_i);
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double mean_i = 0.0, mean_j = 0.0;
                for (int k = 0; k < n; k++) {
                    mean_i += data[k][i];
                    mean_j += data[k][j];
                }
                mean_i /= n;
                mean_j /= n;

                double corr_ij = 0.0;
                for (int k = 0; k < n; k++) {
                    corr_ij += (data[k][i] - mean_i) * (data[k][j] - mean_j);
                }
                corr_ij /= (n - 1);
                corr[i][j] = corr_ij / (stddev[i] * stddev[j]);
            }
        }
    }
};
