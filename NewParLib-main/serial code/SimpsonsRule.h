#include <cmath>

double SimpsonsRule_integrate(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = f(a) + f(b);
    for (int i = 1; i < n; i += 2) {
        sum += 4 * f(a + i * h);
    }
    for (int i = 2; i < n - 1; i += 2) {
        sum += 2 * f(a + i * h);
    }
    return sum * h / 3;
}
