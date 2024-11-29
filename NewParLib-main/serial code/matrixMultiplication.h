#include<iostream>
void matrixMultiply(double A[4][4], double B[4][4], double C[4][4]) {
    int n = 4;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
}
