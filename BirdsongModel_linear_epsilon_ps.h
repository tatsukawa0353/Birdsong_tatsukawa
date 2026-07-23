#ifndef BIRDSONG_MODEL_LINEAR_EPSILON_PS_H
#define BIRDSONG_MODEL_LINEAR_EPSILON_PS_H

#include <fstream>
#include <string>
#include <vector>

struct LinearEpsilonPsParameters {
    double epsilon, b, c, f0, ps, x0, h, tau, M;
    double A, D;
    double alpha, beta;
};

struct LinearEpsilonPsSource {
    double x, y;
    LinearEpsilonPsParameters params;
};

class BirdsongModelLinearEpsilonPs {
public:
    BirdsongModelLinearEpsilonPs(
        double dt,
        double T_delay,
        double epsilon_start,
        double epsilon_slope,
        double ps_start,
        double ps_slope,
        const std::string& output_filename
    );

    void step();
    void saveData();

private:
    LinearEpsilonPsSource left, right;

    double gamma;
    double time;
    double dt;
    double epsilon_start;
    double epsilon_slope;
    double ps_start;
    double ps_slope;

    std::vector<double> pi_history;
    int history_size;
    int current_pos;

    std::ofstream outfile;

    double epsilonAt(double evaluation_time) const;
    double psAt(double evaluation_time) const;
    void calculate_derivatives(
        const LinearEpsilonPsSource& source,
        double pi_tilde,
        double& dx_dt,
        double& dy_dt
    ) const;
};

#endif
