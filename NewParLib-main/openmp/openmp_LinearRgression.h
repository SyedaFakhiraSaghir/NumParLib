#include <omp.h>
#include <iostream>
#include <vector>

void LinearRegression_fit(double** X, double* y, int n, int m, double* beta) {
    std::vector<std::vector<double>> XtX(m, std::vector<double>(m, 0));
    std::vector<double> XtY(m, 0);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < n; k++) {
                XtX[i][j] += X[k][i] * X[k][j];
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            XtY[i] += X[k][i] * y[k];
        }
    }

    // Solve XtX * beta = XtY
    // Use any linear solver here. Example: Gauss-Jordan
}
