#include <iostream>
#include <cmath>
using namespace std;
void qrFactorization(double A[4][4],double Q[4][4],double R[4][4]) {
    int n = 4;
    Q = A;
    for (int k = 0; k < n; k++) {
        R[k][k] = 0;
        for (int i = 0; i < n; i++)
            R[k][k] += Q[i][k] * Q[i][k];
        R[k][k] = std::sqrt(R[k][k]);
        for (int i = 0; i < n; i++)
            Q[i][k] /= R[k][k];
        for (int j = k + 1; j < n; j++) {
            R[k][j] = 0;
            for (int i = 0; i < n; i++)
                R[k][j] += Q[i][k] * Q[i][j];
            for (int i = 0; i < n; i++)
                Q[i][j] -= R[k][j] * Q[i][k];
        }
    }
}
