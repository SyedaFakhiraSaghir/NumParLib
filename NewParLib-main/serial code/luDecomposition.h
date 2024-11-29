#include <iostream>
using namespace std;
void luDecomposition(double A[4][4],double L[4][4],double U[4][4]) {
    int n = 4;
    for (int i = 0; i < n; i++) {
        for (int k = i; k < n; k++) {
            U[i][k] = A[i][k];
            for (int j = 0; j < i; j++)
                U[i][k] -= L[i][j] * U[j][k];
        }
        for (int k = i; k < n; k++) {
            if (i == k)
                L[i][i] = 1;
            else {
                L[k][i] = A[k][i];
                for (int j = 0; j < i; j++)
                    L[k][i] -= L[k][j] * U[j][i];
                L[k][i] /= U[i][i];
            }
        }
    }
}


