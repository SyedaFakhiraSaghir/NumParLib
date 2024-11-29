#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

// Function prototypes (already defined)
void covarianceMatrix(double** data, int n, int m, double** cov, int rank, int size);
void HeatEquationSolver_solve(double alpha, std::vector<std::vector<double>>& u, int nx, int ny, double dt, int nt, int rank, int size);
void LinearRegression_fit(double** X, double* y, int n, int m, double* beta, int rank, int size);
void MatrixInversion_invert_MPI(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& A_inv, int n, int maxIter, double tol, int rank, int size);
double PowerMethod_compute_MPI(std::vector<std::vector<double>>& A, std::vector<double>& b, int maxIterations, double tolerance, int rank, int size);
double TrapezoidalRule_integrate(double (*f)(double), double a, double b, int n, int rank, int size);

double exampleFunction(double x) {
    return x * x;  // Example function for integration
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv); // Initialize MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    // Example 1: Covariance Matrix Calculation
    int n = 1000;  // Number of data points
    int m = 10;    // Number of features
    double** X = new double*[n];
    for (int i = 0; i < n; ++i) {
        X[i] = new double[m];
    }
    double* y = new double[n];
    double** cov = new double*[m];
    for (int i = 0; i < m; ++i) {
        cov[i] = new double[m];
    }

    // Generate random data for testing
    srand(rank + time(0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            X[i][j] = rand() % 100;  // Example random data
        }
        y[i] = rand() % 100;  // Example target data
    }

    // Run covariance matrix calculation
    covarianceMatrix(X, n, m, cov, rank, size);
    if (rank == 0) {
        std::cout << "Covariance Matrix computed." << std::endl;
    }

    // Example 2: Heat Equation Solver (simplified 2D)
    int nx = 100, ny = 100, nt = 100;  // grid size and iterations
    double alpha = 0.01, dt = 0.01;
    std::vector<std::vector<double>> u(nx, std::vector<double>(ny, 0.0)); // Initialize the grid

    HeatEquationSolver_solve(alpha, u, nx, ny, dt, nt, rank, size);
    if (rank == 0) {
        std::cout << "Heat Equation solved." << std::endl;
    }

    // Example 3: Linear Regression (simple example)
    double* beta = new double[m];
    LinearRegression_fit(X, y, n, m, beta, rank, size);
    if (rank == 0) {
        std::cout << "Linear Regression fit completed." << std::endl;
    }

    // Example 4: Matrix Inversion using MPI
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 1.0));  // Example matrix
    std::vector<std::vector<double>> A_inv(n, std::vector<double>(n, 0.0));  // Inverted matrix
    int maxIter = 1000;
    double tol = 1e-6;
    
    MatrixInversion_invert_MPI(A, A_inv, n, maxIter, tol, rank, size);
    if (rank == 0) {
        std::cout << "Matrix Inversion completed." << std::endl;
    }

    // Example 5: Power Method to compute the largest eigenvalue
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 1.0));  // Example matrix
    std::vector<double> b(n, 1.0);  // Starting vector for power method
    double lambda = PowerMethod_compute_MPI(matrix, b, 1000, 1e-6, rank, size);
    if (rank == 0) {
        std::cout << "Largest Eigenvalue (Power Method) computed: " << lambda << std::endl;
    }

    // Example 6: Trapezoidal Rule Integration (parallel)
    double integral_result = TrapezoidalRule_integrate(exampleFunction, 0.0, 1.0, 1000, rank, size);
    if (rank == 0) {
        std::cout << "Integral computed using Trapezoidal Rule: " << integral_result << std::endl;
    }

    // Clean up dynamically allocated memory
    for (int i = 0; i < n; ++i) {
        delete[] X[i];
    }
    delete[] X;
    delete[] y;
    for (int i = 0; i < m; ++i) {
        delete[] cov[i];
    }
    delete[] cov;
    delete[] beta;

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
void covarianceMatrix(double** data, int n, int m, double** cov, int rank, int size) {
    int local_rows = n / size + (rank < n % size ? 1 : 0);
    int start_row = rank * (n / size) + std::min(rank, n % size);

    for (int i = 0; i < m; i++) {
        for (int j = i; j < m; j++) {
            double local_sum = 0;
            double local_mean_i = 0, local_mean_j = 0;

            for (int k = start_row; k < start_row + local_rows; k++) {
                local_mean_i += data[k][i];
                local_mean_j += data[k][j];
            }

            MPI_Allreduce(&local_mean_i, &local_mean_i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_mean_j, &local_mean_j, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            local_mean_i /= n;
            local_mean_j /= n;

            for (int k = start_row; k < start_row + local_rows; k++) {
                local_sum += (data[k][i] - local_mean_i) * (data[k][j] - local_mean_j);
            }

            MPI_Allreduce(&local_sum, &cov[i][j], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            cov[i][j] /= (n - 1);
        }
    }
}

void HeatEquationSolver_solve(double alpha, std::vector<std::vector<double>>& u, int nx, int ny, double dt, int nt, int rank, int size) {
    int local_nx = nx / size + (rank < nx % size ? 1 : 0);
    int start_row = rank * (nx / size) + std::min(rank, nx % size);

    for (int t = 0; t < nt; t++) {
        std::vector<std::vector<double>> u_new = u;

        for (int i = start_row; i < start_row + local_nx; i++) {
            for (int j = 1; j < ny - 1; j++) {
                if (i > 0 && i < nx - 1) {
                    u_new[i][j] = u[i][j] + alpha * dt * (
                        u[i + 1][j] - 2 * u[i][j] + u[i - 1][j] +
                        u[i][j + 1] - 2 * u[i][j] + u[i][j - 1]);
                }
            }
        }

        MPI_Allgather(MPI_IN_PLACE, local_nx * ny, MPI_DOUBLE, &u[0][0], local_nx * ny, MPI_DOUBLE, MPI_COMM_WORLD);

        u = u_new;
    }
}

void LinearRegression_fit(double** X, double* y, int n, int m, double* beta, int rank, int size) {
    int local_rows = n / size + (rank < n % size ? 1 : 0);
    int start_row = rank * (n / size) + std::min(rank, n % size);

    std::vector<double> local_XtX(m * m, 0);
    std::vector<double> XtX(m * m, 0);
    std::vector<double> local_XtY(m, 0);
    std::vector<double> XtY(m, 0);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = start_row; k < start_row + local_rows; k++) {
                local_XtX[i * m + j] += X[k][i] * X[k][j];
            }
        }
    }

    MPI_Reduce(local_XtX.data(), XtX.data(), m * m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    for (int i = 0; i < m; i++) {
        for (int k = start_row; k < start_row + local_rows; k++) {
            local_XtY[i] += X[k][i] * y[k];
        }
    }

    MPI_Reduce(local_XtY.data(), XtY.data(), m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Solve XtX * beta = XtY
        // Here, you can use any matrix solver such as Gauss-Jordan or LU Decomposition.
        // This is omitted for brevity.
    }

    MPI_Bcast(beta, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
void MatrixInversion_invert_MPI(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& A_inv, int n, int maxIter, double tol, int rank, int size) {
    A_inv = A;
    std::vector<std::vector<double>> I(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; i++) I[i][i] = 1.0;

    for (int iter = 0; iter < maxIter; iter++) {
        std::vector<std::vector<double>> temp(n, std::vector<double>(n, 0));

        for (int i = rank; i < n; i += size) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    temp[i][j] += A[i][k] * A_inv[k][j];
                }
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, temp.data(), n * n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double diff = 0.0;
        for (int i = rank; i < n; i += size) {
            for (int j = 0; j < n; j++) {
                diff += (temp[i][j] - I[i][j]) * (temp[i][j] - I[i][j]);
            }
        }

        double global_diff = 0.0;
        MPI_Allreduce(&diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (global_diff < tol) {
            break;
        }
    }
}
double PowerMethod_compute_MPI(std::vector<std::vector<double>>& A, std::vector<double>& b, int maxIterations, double tolerance, int rank, int size) {
    int n = A.size();
    double lambda = 0.0;
    std::vector<double> x(n, 1.0);
    std::vector<double> Ax(n, 0.0);

    for (int iter = 0; iter < maxIterations; iter++) {
        std::fill(Ax.begin(), Ax.end(), 0.0);

        // Compute A * x using MPI parallelization
        for (int i = rank; i < n; i += size) {
            for (int j = 0; j < n; j++) {
                Ax[i] += A[i][j] * x[j];
            }
        }

        // Reduce the Ax vector across all processes
        std::vector<double> Ax_global(n, 0.0);
        MPI_Allreduce(Ax.data(), Ax_global.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        Ax = Ax_global;

        // Compute the norm of Ax
        double local_norm = 0.0;
        for (int i = rank; i < n; i += size) {
            local_norm += Ax[i] * Ax[i];
        }
        double global_norm = 0.0;
        MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        global_norm = sqrt(global_norm);

        // Normalize x
        for (int i = 0; i < n; i++) {
            x[i] = Ax[i] / global_norm;
        }

        // Compute lambda
        double local_lambda = 0.0;
        for (int i = rank; i < n; i += size) {
            local_lambda += x[i] * Ax[i];
        }
        MPI_Allreduce(&local_lambda, &lambda, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Check convergence
        if (rank == 0 && iter > 0 && fabs(lambda - lambda) < tolerance) {
            break;
        }
    }

    return lambda;
}
double TrapezoidalRule_integrate(double (*f)(double), double a, double b, int n, int rank, int size) {
    double h = (b - a) / n;
    int local_n = n / size;
    double local_a = a + rank * local_n * h;
    double local_b = local_a + local_n * h;

    double local_sum = 0.5 * (f(local_a) + f(local_b));
    for (int i = 1; i < local_n; i++) {
        local_sum += f(local_a + i * h);
    }
    local_sum *= h;

    double total_sum = 0;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return total_sum;
}