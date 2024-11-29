#include <iostream>
#include <cmath>
#define N 10
using namespace std;
double bisection(double (*func)(double), double a, double b, double tol = 1e-6) {
    double mid;
    for (int i = 0; i < N; i++) {
        mid = (a + b) / 2;
        if (abs(func(mid)) < tol) return mid;
        if (func(a) * func(mid) < 0)
            b = mid;
        else
            a = mid;
    }
    return mid;
}
