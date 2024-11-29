#include <iostream>
using namespace std;
void matrixTranspose(double A[4][4],double T[4][4]) {
    int n = 4;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            T[j][i] = A[i][j];
}

