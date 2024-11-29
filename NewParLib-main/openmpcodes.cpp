void HeatEquationSolver_solve(double alpha, std::vector<std::vector<double>>& u, int nx, int ny, double dt, int nt) {
    for (int t = 0; t < nt; t++) {
        std::vector<std::vector<double>> u_new = u;

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < nx - 1; i++) {
            for (int j = 1; j < ny - 1; j++) {
                u_new[i][j] = u[i][j] + alpha * dt * (
                    u[i + 1][j] - 2 * u[i][j] + u[i - 1][j] +
                    u[i][j + 1] - 2 * u[i][j] + u[i][j - 1]);
            }
        }

        u = u_new;
    }
}
void covarianceMatrix(double** data, int n, int m, double** cov) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = i; j < m; j++) {
            double mean_i = 0, mean_j = 0;
            for (int k = 0; k < n; k++) {
                mean_i += data[k][i];
                mean_j += data[k][j];
            }
            mean_i /= n;
            mean_j /= n;

            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += (data[k][i] - mean_i) * (data[k][j] - mean_j);
            }

            cov[i][j] = sum / (n - 1);
            cov[j][i] = cov[i][j];
        }
    }
}
void LinearRegression_fit(double** X, double* y, int n, int m, double* beta) {
    std::vector<std::vector<double>> XtX(m, std::vector<double>(m, 0));
    std::vector<double> XtY(m, 0);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < n; k++) {
                XtX[i][j] += X[k][i] * X[k][j];
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            XtY[i] += X[k][i] * y[k];
        }
    }

    // Solve XtX * beta = XtY
    // Use any linear solver here. Example: Gauss-Jordan
}
void MatrixInversion_invert_OpenMP(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& A_inv, int n, int maxIter, double tol) {
    A_inv = A;
    std::vector<std::vector<double>> I(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; i++) I[i][i] = 1.0;

    for (int iter = 0; iter < maxIter; iter++) {
        std::vector<std::vector<double>> temp(n, std::vector<double>(n, 0));

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    temp[i][j] += A[i][k] * A_inv[k][j];
                }
            }
        }

        double diff = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:diff)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                diff += (temp[i][j] - I[i][j]) * (temp[i][j] - I[i][j]);
            }
        }

        if (diff < tol) {
            break;
        }
    }
}

double PowerMethod_compute_OpenMP(std::vector<std::vector<double>>& A, std::vector<double>& b, int maxIterations, double tolerance) {
    int n = A.size();
    double lambda = 0.0;
    std::vector<double> x(n, 1.0);
    std::vector<double> Ax(n, 0.0);

    for (int iter = 0; iter < maxIterations; iter++) {
        std::fill(Ax.begin(), Ax.end(), 0.0);

        // Parallelize matrix-vector multiplication
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Ax[i] += A[i][j] * x[j];
            }
        }

        // Compute the norm of Ax
        double norm = 0.0;
        #pragma omp parallel for reduction(+:norm)
        for (int i = 0; i < n; i++) {
            norm += Ax[i] * Ax[i];
        }
        norm = sqrt(norm);

        // Normalize x
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            x[i] = Ax[i] / norm;
        }

        // Compute lambda
        lambda = 0.0;
        #pragma omp parallel for reduction(+:lambda)
        for (int i = 0; i < n; i++) {
            lambda += x[i] * Ax[i];
        }

        // Check convergence
        if (iter > 0 && fabs(lambda - lambda) < tolerance) {
            break;
        }
    }

    return lambda;
}
double TrapezoidalRule_integrate(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.5 * (f(a) + f(b));

    #pragma omp parallel for reduction(+:sum)
    for (int i = 1; i < n; i++) {
        sum += f(a + i * h);
    }

    return sum * h;
}