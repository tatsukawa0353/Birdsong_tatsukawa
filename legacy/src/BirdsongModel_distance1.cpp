#include "BirdsongModel_distance1.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdexcept>

namespace {
constexpr double kClosedLeftF0 = 1.0e7;
constexpr double kOpenRightF0 = 4.9e4;
}

BirdsongModelDistance1::BirdsongModelDistance1(
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
    bool single_source
)
    : gamma(0.9),
      time(0.0),
      dt(dt),
      ramp_start(ramp_start),
      ramp_end(ramp_end),
      epsilon_start(epsilon_start),
      epsilon_end(epsilon_end),
      ps_ramp_start(ps_ramp_start),
      ps_ramp_end(ps_ramp_end),
      ps_start(ps_start),
      ps_end(ps_end),
      history_size(static_cast<int>(delay / dt)),
      current_pos(0) {
    if (dt <= 0.0 || history_size <= 0) {
        throw std::invalid_argument("dt must be positive and delay must be at least dt");
    }
    if (ramp_end <= ramp_start) {
        throw std::invalid_argument("ramp_end must be greater than ramp_start");
    }
    if (ps_ramp_end <= ps_ramp_start) {
        throw std::invalid_argument(
            "ps_ramp_end must be greater than ps_ramp_start"
        );
    }

    // The left syrinx is closed by muscle force; the right syrinx phonates.
    left.params = {
        epsilon_start, 2.0e4, 2.0e8, kClosedLeftF0, ps_start,
        0.04, 0.1, 1.0e-4, 5.0e-3, 1.0, 0.01, 1.2e6, 1.5e3
    };
    left.x = 0.0;
    left.y = 0.0;

    right.params = left.params;
    right.params.f0 = kOpenRightF0;
    right.x = 0.0;
    right.y = 0.0;

    if (single_source) {
        // Remove the left syrinx from the acoustic coupling completely.
        // Its internal state may still be integrated, but alpha=beta=0 means
        // it contributes neither to pi_tilde nor to the radiated pressure pi.
        left.params.alpha = 0.0;
        left.params.beta = 0.0;
        left.params.f0 = kOpenRightF0;
    }

    pi_history.assign(history_size, 0.0);
    outfile.open(output_filename);
    if (!outfile) {
        throw std::runtime_error("Could not open output file: " + output_filename);
    }
    outfile << std::setprecision(15);
    outfile
        << "time,pi,s_left,s_right,x_left,y_left,x_right,y_right,"
        << "epsilon_left,epsilon_right,ps_left,ps_right\n";
}

double BirdsongModelDistance1::interpolate(
    double evaluation_time,
    double change_start,
    double change_end,
    double value_start,
    double value_end
) const {
    const double fraction = std::clamp(
        (evaluation_time - change_start) / (change_end - change_start),
        0.0,
        1.0
    );
    return value_start + fraction * (value_end - value_start);
}

double BirdsongModelDistance1::epsilonAt(double evaluation_time) const {
    // Normalized exponential decrease.  k=3 gives a rapid initial decrease
    // and a gradual approach to the lower bound, while the normalization
    // makes both endpoint values exact.
    const double u = std::clamp(
        (evaluation_time - ramp_start) / (ramp_end - ramp_start),
        0.0,
        1.0
    );
    constexpr double decay_rate = 3.0;
    const double end_exponential = std::exp(-decay_rate);
    const double normalized_exponential =
        (std::exp(-decay_rate * u) - end_exponential)
        / (1.0 - end_exponential);
    return epsilon_end
        + (epsilon_start - epsilon_end) * normalized_exponential;
}

double BirdsongModelDistance1::psAt(double evaluation_time) const {
    return interpolate(
        evaluation_time,
        ps_ramp_start,
        ps_ramp_end,
        ps_start,
        ps_end
    );
}

void BirdsongModelDistance1::setStageParameters(
    Distance1Source& left_source,
    Distance1Source& right_source,
    double evaluation_time
) const {
    // Epsilon changes only on the right (phonating) side.  The closed left
    // side retains its initial epsilon throughout the simulation.
    left_source.params.epsilon = epsilon_start;
    right_source.params.epsilon = epsilonAt(evaluation_time);

    const double ps = psAt(evaluation_time);
    left_source.params.ps = ps;
    right_source.params.ps = ps;
}

void BirdsongModelDistance1::setParameters(double evaluation_time) {
    setStageParameters(left, right, evaluation_time);
}

void BirdsongModelDistance1::calculateDerivatives(
    const Distance1Source& source,
    double pi_tilde,
    double& dx_dt,
    double& dy_dt
) const {
    const double f =
        -source.params.epsilon * source.x
        - source.params.b * source.y
        - source.params.c * source.x * source.x * source.y
        - source.params.f0;
    const double p_tilde_g =
        source.params.ps
        + (source.params.D * source.y - source.params.A)
        * (source.params.ps - pi_tilde);

    dx_dt = source.y;
    dy_dt = f + p_tilde_g;
}

double BirdsongModelDistance1::calculatePiTilde(
    const Distance1Source& left_source,
    const Distance1Source& right_source,
    double pi_delayed
) const {
    return
        left_source.params.alpha
            * (left_source.x - left_source.params.tau * left_source.y)
        + left_source.params.beta * left_source.y
        + right_source.params.alpha
            * (right_source.x - right_source.params.tau * right_source.y)
        + right_source.params.beta * right_source.y
        - gamma * pi_delayed;
}

void BirdsongModelDistance1::step() {
    const double pi_delayed = pi_history[current_pos];

    setParameters(time);
    const double pi_tilde_start = calculatePiTilde(left, right, pi_delayed);
    double k1_x_l, k1_y_l, k1_x_r, k1_y_r;
    calculateDerivatives(left, pi_tilde_start, k1_x_l, k1_y_l);
    calculateDerivatives(right, pi_tilde_start, k1_x_r, k1_y_r);

    Distance1Source mid1_l = left;
    Distance1Source mid1_r = right;
    setStageParameters(mid1_l, mid1_r, time + dt / 2.0);
    mid1_l.x += k1_x_l * dt / 2.0;
    mid1_l.y += k1_y_l * dt / 2.0;
    mid1_r.x += k1_x_r * dt / 2.0;
    mid1_r.y += k1_y_r * dt / 2.0;
    const double pi_tilde_mid1 = calculatePiTilde(mid1_l, mid1_r, pi_delayed);
    double k2_x_l, k2_y_l, k2_x_r, k2_y_r;
    calculateDerivatives(mid1_l, pi_tilde_mid1, k2_x_l, k2_y_l);
    calculateDerivatives(mid1_r, pi_tilde_mid1, k2_x_r, k2_y_r);

    Distance1Source mid2_l = left;
    Distance1Source mid2_r = right;
    setStageParameters(mid2_l, mid2_r, time + dt / 2.0);
    mid2_l.x += k2_x_l * dt / 2.0;
    mid2_l.y += k2_y_l * dt / 2.0;
    mid2_r.x += k2_x_r * dt / 2.0;
    mid2_r.y += k2_y_r * dt / 2.0;
    const double pi_tilde_mid2 = calculatePiTilde(mid2_l, mid2_r, pi_delayed);
    double k3_x_l, k3_y_l, k3_x_r, k3_y_r;
    calculateDerivatives(mid2_l, pi_tilde_mid2, k3_x_l, k3_y_l);
    calculateDerivatives(mid2_r, pi_tilde_mid2, k3_x_r, k3_y_r);

    Distance1Source end_l = left;
    Distance1Source end_r = right;
    setStageParameters(end_l, end_r, time + dt);
    end_l.x += k3_x_l * dt;
    end_l.y += k3_y_l * dt;
    end_r.x += k3_x_r * dt;
    end_r.y += k3_y_r * dt;
    const double pi_tilde_end = calculatePiTilde(end_l, end_r, pi_delayed);
    double k4_x_l, k4_y_l, k4_x_r, k4_y_r;
    calculateDerivatives(end_l, pi_tilde_end, k4_x_l, k4_y_l);
    calculateDerivatives(end_r, pi_tilde_end, k4_x_r, k4_y_r);

    left.x += (k1_x_l + 2.0 * k2_x_l + 2.0 * k3_x_l + k4_x_l) * dt / 6.0;
    left.y += (k1_y_l + 2.0 * k2_y_l + 2.0 * k3_y_l + k4_y_l) * dt / 6.0;
    right.x += (k1_x_r + 2.0 * k2_x_r + 2.0 * k3_x_r + k4_x_r) * dt / 6.0;
    right.y += (k1_y_r + 2.0 * k2_y_r + 2.0 * k3_y_r + k4_y_r) * dt / 6.0;

    time += dt;
    setParameters(time);
    const double pi_tilde_next = calculatePiTilde(left, right, pi_delayed);
    double dx_l, dy_l, dx_r, dy_r;
    calculateDerivatives(left, pi_tilde_next, dx_l, dy_l);
    calculateDerivatives(right, pi_tilde_next, dx_r, dy_r);
    const double pi =
        pi_tilde_next
        - left.params.beta * left.params.tau * dy_l
        - right.params.beta * right.params.tau * dy_r;

    pi_history[current_pos] = pi;
    current_pos = (current_pos + 1) % history_size;

    const double s_left =
        left.params.alpha * (left.x - left.params.tau * left.y)
        + left.params.beta * (left.y - left.params.tau * dy_l);
    const double s_right =
        right.params.alpha * (right.x - right.params.tau * right.y)
        + right.params.beta * (right.y - right.params.tau * dy_r);

    outfile
        << time << ',' << pi << ',' << s_left << ',' << s_right << ','
        << left.x << ',' << left.y << ',' << right.x << ',' << right.y << ','
        << left.params.epsilon << ',' << right.params.epsilon << ','
        << left.params.ps << ',' << right.params.ps << '\n';
}

void BirdsongModelDistance1::saveData() {
    setParameters(time);
    outfile
        << time << ",0,0,0,"
        << left.x << ',' << left.y << ',' << right.x << ',' << right.y << ','
        << left.params.epsilon << ',' << right.params.epsilon << ','
        << left.params.ps << ',' << right.params.ps << '\n';
}
