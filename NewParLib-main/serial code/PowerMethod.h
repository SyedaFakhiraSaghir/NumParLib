#include <vector>
#include <cmath>

double PowerMethod_compute(std::vector<std::vector<double>>& A, std::vector<double>& b, int maxIterations, double tolerance) {
    int n = A.size();
    double lambda = 0.0;
    std::vector<double> x(n, 1.0);
    
    for (int iter = 0; iter < maxIterations; iter++) {
        std::vector<double> Ax(n, 0.0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Ax[i] += A[i][j] * x[j];
            }
        }
        
        double norm = 0.0;
        for (int i = 0; i < n; i++) {
            norm += Ax[i] * Ax[i];
        }
        norm = sqrt(norm);
        
        for (int i = 0; i < n; i++) {
            x[i] = Ax[i] / norm;
        }
        
        lambda = 0.0;
        for (int i = 0; i < n; i++) {
            lambda += x[i] * Ax[i];
        }
        
        if (fabs(lambda - lambda) < tolerance) {
            break;
        }
    }
    return lambda;
}
