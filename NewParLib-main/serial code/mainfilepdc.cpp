#include <iostream>
#include <vector>
#include <cmath>
#include "RungeKutta.h"
#include "SimpsonsRule.h"
#include "TrapezodialRule.h"
#include "WaveEquationSolver.h"
#include "CovarianceCorrelation.h"
#include "HeatEquationSolver.h"
#include "LinearRegression.h"
#include "MatrixInversion.h"
#include "PowerMethod.h"

using namespace std;

// Example function for Runge-Kutta (changed to match the expected signature)
double exampleODE(double t, const double* y, int size) {
    return -y[0];  // dy/dt = -y
}

// Example function for Simpson's Rule
double exampleFunction(double x) {
    return x * x;  // f(x) = x^2
}

int main() {
    // Runge-Kutta Method
    cout << "Runge-Kutta Method:\n";
    const int size = 2;    // Example: 2D system
    double y0[size] = {1.0, 2.0};
    double t0 = 0.0, h = 0.1;
    int n = 10;
    RungeKutta(exampleODE, t0, y0, h, n, size);  // Corrected here

    cout << "\n\n";

    // Simpson's Rule
    cout << "Simpson's Rule:\n";
    double integral = SimpsonsRule_integrate(exampleFunction, 0.0, 1.0, 10);
    cout << "Integral result: " << integral << "\n\n";

    // Trapezoidal Rule
    cout << "Trapezoidal Rule:\n";
    integral = TrapezoidalRule_integrate(exampleFunction, 0.0, 1.0, 10);
    cout << "Integral result: " << integral << "\n\n";

    // Heat Equation Solver
    cout << "Heat Equation Solver:\n";
    int nx = 5, ny = 5, nt = 10;
    double alpha = 0.01, dt = 0.1;
    vector<vector<double>> u(nx, vector<double>(ny, 0.0));
    u[2][2] = 100.0;  // Initial heat source
    HeatEquationSolver_solve(alpha, u, nx, ny, dt, nt);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            cout << u[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";

    // Power Method (Eigenvalue Calculation)
    cout << "Power Method (Eigenvalue Calculation):\n";
    vector<vector<double>> A = {{4, 1}, {2, 3}};
    vector<double> b = {1, 1};
    double eigenvalue = PowerMethod_compute(A, b, 1000, 1e-6);
    cout << "Estimated eigenvalue: " << eigenvalue << "\n\n";

    // Matrix Inversion
    cout << "Matrix Inversion:\n";
    vector<vector<double>> A_inv(2, vector<double>(2));
    MatrixInversion_invert(A, A_inv, 2, 1000, 1e-6);
    for (const auto& row : A_inv) {
        for (double value : row) {
            cout << value << " ";
        }
        cout << "\n";
    }
    cout << "\n";

    // Linear Regression (Correct class instantiation)
    cout << "Linear Regression:\n";
    double X[3][2] = {{1, 1}, {2, 2}, {3, 3}};
    double y[3] = {1, 2, 3};
//
//    int n = 3;
      int m = 2;

    double** X_ptr = new double*[n];
    for (int i = 0; i < n; i++) {
        X_ptr[i] = X[i];
    }

    double beta[m];

    LinearRegression_fit(X_ptr, y, n, m, beta);

    cout << "Fitted coefficients (beta):\n";
    for (int i = 0; i < m; i++) {
        cout << "beta[" << i << "] = " << beta[i] << endl;
    }

    double y_pred[n];
    LinearRegression_predict(X_ptr, beta, n, m, y_pred);

    cout << "\nPredictions:\n";
    for (int i = 0; i < n; i++) {
        cout << "y_pred[" << i << "] = " << y_pred[i] << endl;
    }

    delete[] X_ptr;

    cout << "\n\n";

    // Covariance and Correlation Matrix Calculation
    cout << "Covariance and Correlation Matrix Calculation:\n";
    const int numDataPoints = 3;  // Renamed 'n' to avoid conflict

    double data[numDataPoints][2] = {
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0}
    };

    double* data_ptr[numDataPoints];
    for (int i = 0; i < numDataPoints; i++) {
        data_ptr[i] = data[i];
    }

    double cov[m][m];
    double corr[m][m];
    double* cov_ptr[m];
    double* corr_ptr[m];
    for (int i = 0; i < m; i++) {
        cov_ptr[i] = cov[i];
        corr_ptr[i] = corr[i];
    }

    CovarianceCorrelation::covarianceMatrix(data_ptr, numDataPoints, m, cov_ptr);
    CovarianceCorrelation::correlationMatrix(data_ptr, numDataPoints, m, corr_ptr);

    cout << "Covariance Matrix:\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            cout << fixed << setprecision(2) << cov[i][j] << " ";
        }
        cout << "\n";
    }

    cout << "Correlation Matrix:\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            cout << fixed << setprecision(2) << corr[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";

    // Wave Equation Solver (Correct class instantiation)
    cout << "Wave Equation Solver:\n";
    int ng = 10;  // Number of grid points in space
    int nts = 100; // Number of time steps
    double c = 0.5; // Wave speed
    double dts = 0.1; // Time step size
    double dx = 0.1; // Space step size

    // Create a 2D vector to store the solution (space x time)
    vector<vector<double>> uu(ng, vector<double>(nts, 0.0));

    // Set initial conditions (initial displacement and velocity)
    for (int i = 0; i < ng; i++) {
        uu[i][0] = 0.0;    // Initial displacement (t = 0)
        uu[i][1] = 0.0;    // Initial velocity (t = 1)
    }

    // Set initial displacement (you can modify this as needed)
    uu[ng / 2][0] = 1.0;  // A small displacement in the center

    // Solve the wave equation
    WaveEquationSolver_solve(uu, ng, nts, c, dts, dx);

    // Output the results (showing first 10 time steps)
    for (int t = 0; t < nts; t++) {
        cout << "Time step " << t << ": ";
        for (int i = 0; i < ng; i++) {
            cout << uu[i][t] << " ";
        }
        cout << endl;
    }
    return 0;
}
