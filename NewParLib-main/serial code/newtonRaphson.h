#include<iostream>
#include<cmath>
double newtonRaphson(double (*f)(double), double (*f_prime)(double), double x0, double tol = 1e-6) {
    double x = x0;
    for (int i = 0; i < 4; i++) {
        double fx = f(x);
        double fpx = f_prime(x);
        if (abs(fpx) < tol) return x; // Avoid division by zero
        double x_new = x - fx / fpx;
        if (abs(x_new - x) < tol) return x_new;
        x = x_new;
    }
    return x;
}
