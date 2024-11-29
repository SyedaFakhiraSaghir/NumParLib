#include <iostream>
#include"bisection.h"
#include"choleskyDecomposition.h"
#include"gaussSeidel.h"
#include "jacobi.h"
#include "luDecomposition.h"
#include "matrixMultiplication.h"
#include "matrixTransposition.h"
#include "newtonRaphson.h"
#include "qrFactorization.h"
using namespace std;
double func(double x) {
    return x * x - 4;
}
double funcNewtonPrime(double x){
	return 2*x;
}
double funcNewton(double x){
	return x * x - 2;
}
int main(){
    double A[4][4] = {
        {4, 12, -16, 0},
        {12, 37, -43, 0},
        {-16, -43, 98, 0},
        {0, 0, 0, 1}
    };
    double B[4][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };
    double C[4][4] = {0};
    double T[4][4] = {0};
    double L[4][4] = {0};
    double U[4][4] = {0};
    double Q[4][4] = {0};
    double R[4][4] = {0};
    double b[4] = {1, 2, 3, 4};
    double x[4] = {0};

    
   double root = bisection(func, 0.0, 5.0, 1e-6);
    cout << "Root found by Bisection Method: " << root <<endl;

    //Cholesky Decomposition
    if (choleskyDecomposition(A, L)) {
        cout << "Cholesky Decomposition successful." << endl;
    } else {
        cout << "Matrix is not positive definite." <<endl;
    }

    //Gauss-Seidel Method
    if (gaussSeidel(A, b, x)) {
        cout << "Solution by Gauss-Seidel Method: ";
        for (int i = 0; i < 4; i++) {
    cout << x[i] << " ";
}
        cout << endl;
    }

    //Jacobi Method
    if (jacobi(A, b, x)) {
        cout << "Solution by Jacobi Method: ";
        for (int i = 0; i < 4; i++) {
    cout << x[i] << " ";
}
        cout <<endl;
    }

    //LU Decomposition
    luDecomposition(A, L, U);
    cout << "LU Decomposition completed." <<endl;

    //Matrix Multiplication
    matrixMultiply(A, B, C);
    cout << "Matrix Multiplication Result:" << endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
           cout << C[i][j] << " ";
        }
       cout << endl;
    }

    //Matrix Transposition
    matrixTranspose(B, T);
    cout << "Matrix Transposition Result:" << endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            cout << T[i][j] << " ";
        }
        cout << endl;
    }

    double newtonRoot = newtonRaphson(funcNewton, funcNewtonPrime, 1.0, 1e-6);
    cout << "Root found by Newton-Raphson Method: " << newtonRoot <<endl;

    //QR Factorization
    qrFactorization(A, Q, R);
    cout << "QR Factorization completed." << endl;

    return 0;
}

