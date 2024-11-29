
#include <vector>

void WaveEquationSolver_solve(std::vector<std::vector<double>>& u, int nx, int nt, double c, double dt, double dx) {
    for (int t = 1; t < nt; t++) {
        for (int i = 1; i < nx - 1; i++) {
            u[i][t] = 2 * (1 - c * c) * u[i][t - 1] - u[i][t - 2] + c * c * (u[i + 1][t - 1] - 2 * u[i][t - 1] + u[i - 1][t - 1]);
        }
    }
}
