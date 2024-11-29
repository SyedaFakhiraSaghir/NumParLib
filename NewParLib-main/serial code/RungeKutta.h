#include <iostream>
using namespace std;

void add_arrays(double result[], const double a[], const double b[], int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

void scale_array(double result[], const double a[], double scalar, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * scalar;
    }
}

void RungeKutta(
    double (*f)(double, const double[], int),
    double t0,
    double y0[],
    double h,
    int n,
    int size
) {
    double y[size], k1[size], k2[size], k3[size], k4[size];
    for (int i = 0; i < size; i++) {
        y[i] = y0[i];
    }

    for (int i = 0; i < n; i++) {
        double t = t0 + i * h;

        for (int j = 0; j < size; j++) {
            k1[j] = h * f(t, y, j);
        }

        double temp[size];
        scale_array(temp, k1, 0.5, size);
        add_arrays(temp, y, temp, size);
        for (int j = 0; j < size; j++) {
            k2[j] = h * f(t + h / 2, temp, j);
        }

        scale_array(temp, k2, 0.5, size);
        add_arrays(temp, y, temp, size);
        for (int j = 0; j < size; j++) {
            k3[j] = h * f(t + h / 2, temp, j);
        }

        scale_array(temp, k3, 1.0, size);
        add_arrays(temp, y, temp, size);
        for (int j = 0; j < size; j++) {
            k4[j] = h * f(t + h, temp, j);
        }

        for (int j = 0; j < size; j++) {
            y[j] = y[j] + (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6.0;
        }
    }

    cout << "Final y values: ";
    for (int i = 0; i < size; i++) {
        cout << y[i] << " ";
    }
    cout << endl;
}
