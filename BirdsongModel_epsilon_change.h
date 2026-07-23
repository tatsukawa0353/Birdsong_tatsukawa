#ifndef BIRDSONG_MODEL_EPSILON_CHANGE_H
#define BIRDSONG_MODEL_EPSILON_CHANGE_H

#include <fstream>
#include <string>
#include <vector>

struct Parameters {
    double epsilon, b, c, f0, ps, x0, h, tau, M;
    double A, D;
    double alpha, beta;
};

struct Source {
    double x, y;
    Parameters params;
};

enum class EpsilonPattern {
    Linear,
    Quadratic,
    Step,
    Sinusoidal
};

class BirdsongModel {
public:
    BirdsongModel(
        double dt,
        double T_delay,
        double total_time,
        EpsilonPattern epsilon_pattern,
        double epsilon_start,
        double epsilon_end,
        double sine_frequency,
        const std::string& output_filename
    );

    void step();
    void saveData();

private:
    Source left, right;

    double gamma;
    double time;
    double dt;
    double total_sim_time;
    double epsilon_start;
    double epsilon_end;
    double sine_frequency;
    EpsilonPattern epsilon_pattern;

    std::vector<double> pi_history;
    int history_size;
    int current_pos;

    std::ofstream outfile;

    double epsilonAt(double evaluation_time) const;
    void calculate_derivatives(
        const Source& source,
        double pi_tilde,
        double& dx_dt,
        double& dy_dt
    ) const;
};

#endif
