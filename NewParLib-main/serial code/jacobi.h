#include <iostream>
#include <cmath>
using namespace std;


bool jacobi(double A[4][4], double b[4], double x[4], double tol = 1e-6) {
    double x_old[4]; // Array to store old values of x
    int max_iter = 100; // Maximum number of iterations

    for (int iter = 0; iter < max_iter; iter++) {
        // Copy current x values to x_old
        for (int i = 0; i < 4; i++) {
            x_old[i] = x[i];
        }

        // Iterate through each equation
        for (int i = 0; i < 4; i++) {
            double sum = b[i];
            for (int j = 0; j < N; j++) {
                if (j != i) {
                    sum -= A[i][j] * x_old[j];
                }
            }
            x[i] = sum / A[i][i]; // Update the solution for x[i]
        }

        // Compute the error (convergence check)
        double error = 0;
        for (int i = 0; i < 4; i++) {
            error += fabs(x[i] - x_old[i]); // Use fabs for floating-point comparison
        }
        if (error < tol) {
            return true; // Convergence achieved
        }
    }
    return false; // No convergence within max_iter
}

