#ifndef BIRDSONG_MODEL_LINEAR_PS_H
#define BIRDSONG_MODEL_LINEAR_PS_H

#include <fstream>
#include <string>
#include <vector>

struct LinearPsParameters {
    double epsilon, b, c, f0, ps, x0, h, tau, M;
    double A, D;
    double alpha, beta;
};

struct LinearPsSource {
    double x, y;
    LinearPsParameters params;
};

class BirdsongModelLinearPs {
public:
    BirdsongModelLinearPs(
        double dt,
        double T_delay,
        double ps_start,
        double ps_slope,
        const std::string& output_filename,
        bool enable_right_source = true
    );

    void step();
    void saveData();

private:
    LinearPsSource left, right;

    double gamma;
    double time;
    double dt;
    double ps_start;
    double ps_slope;

    std::vector<double> pi_history;
    int history_size;
    int current_pos;

    std::ofstream outfile;

    double psAt(double evaluation_time) const;
    void calculate_derivatives(
        const LinearPsSource& source,
        double pi_tilde,
        double& dx_dt,
        double& dy_dt
    ) const;
};

#endif
