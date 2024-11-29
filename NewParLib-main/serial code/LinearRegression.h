#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

bool gauss_jordan(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        double maxEl = abs(matrix[i][i]);
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (abs(matrix[k][i]) > maxEl) {
                maxEl = abs(matrix[k][i]);
                maxRow = k;
            }
        }

        if (i != maxRow) {
            for (int j = 0; j < n; j++) {
                swap(matrix[i][j], matrix[maxRow][j]);
            }
        }

        for (int k = i + 1; k < n; k++) {
            double factor = matrix[k][i] / matrix[i][i];
            for (int j = 0; j < n; j++) {
                matrix[k][j] -= matrix[i][j] * factor;
            }
        }
    }

    for (int i = n - 1; i >= 0; i--) {
        for (int j = n - 1; j >= 0; j--) {
            if (i != j) {
                double factor = matrix[i][j] / matrix[j][j];
                for (int k = 0; k < n; k++) {
                    matrix[i][k] -= matrix[j][k] * factor;
                }
            }
        }
    }

    for (int i = 0; i < n; i++) {
        matrix[i][i] /= matrix[i][i];
    }

    return true;
}

void LinearRegression_fit(double** X, double* y, int n, int m, double* beta) {
    double** Xt = new double*[m];
    for (int i = 0; i < m; i++) {
        Xt[i] = new double[n];
    }

    double** XtX = new double*[m];
    for (int i = 0; i < m; i++) {
        XtX[i] = new double[m];
    }

    double* XtY = new double[m];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            Xt[i][j] = X[j][i];
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            XtX[i][j] = 0;
            for (int k = 0; k < n; k++) {
                XtX[i][j] += Xt[i][k] * X[k][j];
            }
        }
    }

    if (!gauss_jordan(XtX, m)) {
        cout << "Matrix inversion failed!" << endl;
        return;
    }

    for (int i = 0; i < m; i++) {
        XtY[i] = 0;
        for (int j = 0; j < n; j++) {
            XtY[i] += Xt[i][j] * y[j];
        }
    }

    for (int i = 0; i < m; i++) {
        beta[i] = 0;
        for (int j = 0; j < m; j++) {
            beta[i] += XtX[i][j] * XtY[j];
        }
    }

    for (int i = 0; i < m; i++) {
        delete[] Xt[i];
        delete[] XtX[i];
    }
    delete[] Xt;
    delete[] XtX;
    delete[] XtY;
}

void LinearRegression_predict(double** X, double* beta, int n, int m, double* y_pred) {
    for (int i = 0; i < n; i++) {
        y_pred[i] = 0;
        for (int j = 0; j < m; j++) {
            y_pred[i] += X[i][j] * beta[j];
        }
    }
}