#include <mpi.h>
#include <cmath>

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
