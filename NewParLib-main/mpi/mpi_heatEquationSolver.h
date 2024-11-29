#include <mpi.h>
#include <vector>

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
