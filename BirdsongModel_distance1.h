#ifndef BIRDSONG_MODEL_DISTANCE1_H
#define BIRDSONG_MODEL_DISTANCE1_H

#include <fstream>
#include <string>
#include <vector>

struct Distance1Parameters {
    double epsilon, b, c, f0, ps, x0, h, tau, M;
    double A, D;
    double alpha, beta;
};

struct Distance1Source {
    double x, y;
    Distance1Parameters params;
};

class BirdsongModelDistance1 {
public:
    BirdsongModelDistance1(
        double dt,
        double delay,
        double ramp_start,
        double ramp_end,
        double epsilon_start,
        double epsilon_end,
        double ps_ramp_start,
        double ps_ramp_end,
        double ps_start,
        double ps_end,
        const std::string& output_filename,
        bool single_source = false
    );

    void step();
    void saveData();

private:
    Distance1Source left, right;
    double gamma;
    double time;
    double dt;
    double ramp_start;
    double ramp_end;
    double epsilon_start;
    double epsilon_end;
    double ps_ramp_start;
    double ps_ramp_end;
    double ps_start;
    double ps_end;
    std::vector<double> pi_history;
    int history_size;
    int current_pos;
    std::ofstream outfile;

    double interpolate(
        double evaluation_time,
        double change_start,
        double change_end,
        double value_start,
        double value_end
    ) const;
    double epsilonAt(double evaluation_time) const;
    double psAt(double evaluation_time) const;
    void setParameters(double evaluation_time);
    void setStageParameters(
        Distance1Source& left_source,
        Distance1Source& right_source,
        double evaluation_time
    ) const;
    void calculateDerivatives(
        const Distance1Source& source,
        double pi_tilde,
        double& dx_dt,
        double& dy_dt
    ) const;
    double calculatePiTilde(
        const Distance1Source& left_source,
        const Distance1Source& right_source,
        double pi_delayed
    ) const;
};

#endif
