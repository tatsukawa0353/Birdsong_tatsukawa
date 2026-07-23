#ifndef BIRDSONG_MODEL_LINEAR_EPSILON_H
#define BIRDSONG_MODEL_LINEAR_EPSILON_H

#include <fstream>
#include <string>
#include <vector>

struct LinearEpsilonParameters {
    double epsilon, b, c, f0, ps, x0, h, tau, M;
    double A, D;
    double alpha, beta;
};

struct LinearEpsilonSource {
    double x, y;
    LinearEpsilonParameters params;
};

class BirdsongModelLinearEpsilon {
public:
    BirdsongModelLinearEpsilon(
        double dt,
        double T_delay,
        double epsilon_start,
        double epsilon_slope,
        const std::string& output_filename,
        bool enable_right_source = true
    );

    void step();
    void saveData();

private:
    LinearEpsilonSource left, right;

    double gamma;
    double time;
    double dt;
    double epsilon_start;
    double epsilon_slope;

    std::vector<double> pi_history;
    int history_size;
    int current_pos;

    std::ofstream outfile;

    double epsilonAt(double evaluation_time) const;
    void calculate_derivatives(
        const LinearEpsilonSource& source,
        double pi_tilde,
        double& dx_dt,
        double& dy_dt
    ) const;
};

#endif
