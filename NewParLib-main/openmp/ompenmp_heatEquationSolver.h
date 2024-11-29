#include <vector>
#include <omp.h>

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
