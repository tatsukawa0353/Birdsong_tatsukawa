#include "BirdsongModel_linear_ps.h"

#include <cmath>
#include <iomanip>
#include <stdexcept>

namespace {
constexpr double kFixedEpsilon = 3.7e7;
constexpr double kClosedRightPs = 3.2e6;
}

BirdsongModelLinearPs::BirdsongModelLinearPs(
    double dt,
    double T_delay,
    double ps_start,
    double ps_slope,
    const std::string& output_filename,
    bool enable_right_source
)
    : gamma(0.9),
      time(0.0),
      dt(dt),
      ps_start(ps_start),
      ps_slope(ps_slope),
      history_size(static_cast<int>(T_delay / dt)),
      current_pos(0) {

    if (history_size <= 0) {
        throw std::invalid_argument("T_delay must be greater than or equal to dt.");
    }

    // Existing model parameters are retained. Epsilon is fixed at 3.7e7.
    // The left/open source uses ps(t), while the right/closed source keeps
    // the existing constant pressure, ps = 3.2e6.
    left.params = {
        kFixedEpsilon, 2.0e4, 2.0e8, 4.9e4, ps_start,
        0.04, 0.1, 1.0e-4, 5.0e-3, 1.0, 0.01, 1.2e6, 1.5e3
    };
    left.x = 0.0;
    left.y = 0.0;

    // The left syrinx is open for phonation. The right syrinx retains the
    // existing muscle-closure setting.
    right.params = left.params;
    right.params.f0 = 1.0e7;
    right.params.ps = kClosedRightPs;
    if (!enable_right_source) {
        right.params.alpha = 0.0;
        right.params.beta = 0.0;
    }
    right.x = 0.0;
    right.y = 0.0;

    pi_history.assign(history_size, 0.0);

    outfile.open(output_filename);
    if (!outfile) {
        throw std::runtime_error("Could not open output file: " + output_filename);
    }
    outfile << std::setprecision(15);
    outfile
        << "time,pi,x_left,y_left,x_right,y_right,"
        << "epsilon_left,epsilon_right,ps_left,ps_right\n";
}

double BirdsongModelLinearPs::psAt(double evaluation_time) const {
    return ps_start + ps_slope * evaluation_time;
}

// Equations (62)-(65) of the existing model.
void BirdsongModelLinearPs::calculate_derivatives(
    const LinearPsSource& source,
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

void BirdsongModelLinearPs::step() {
    // 1. Obtain p_i(t-T) from the existing circular delay buffer.
    const int past_pos =
        (current_pos - history_size + static_cast<int>(pi_history.size()))
        % static_cast<int>(pi_history.size());
    const double pi_delayed = pi_history[past_pos];

    // 2. Existing fourth-order Runge-Kutta structure.
    const double ps_k1 = psAt(time);
    left.params.ps = ps_k1;
    right.params.ps = kClosedRightPs;

    const double pi_tilde_start =
        left.params.alpha * (left.x - left.params.tau * left.y)
        + left.params.beta * left.y
        + right.params.alpha * (right.x - right.params.tau * right.y)
        + right.params.beta * right.y
        - gamma * pi_delayed;

    double k1_x_l, k1_y_l, k1_x_r, k1_y_r;
    calculate_derivatives(left, pi_tilde_start, k1_x_l, k1_y_l);
    calculate_derivatives(right, pi_tilde_start, k1_x_r, k1_y_r);

    LinearPsSource mid1_l = left;
    LinearPsSource mid1_r = right;
    const double ps_k2 = psAt(time + dt / 2.0);
    mid1_l.params.ps = ps_k2;
    mid1_r.params.ps = kClosedRightPs;
    mid1_l.x += k1_x_l * dt / 2.0;
    mid1_r.x += k1_x_r * dt / 2.0;
    mid1_l.y += k1_y_l * dt / 2.0;
    mid1_r.y += k1_y_r * dt / 2.0;

    const double pi_tilde_mid1 =
        mid1_l.params.alpha * (mid1_l.x - mid1_l.params.tau * mid1_l.y)
        + mid1_l.params.beta * mid1_l.y
        + mid1_r.params.alpha * (mid1_r.x - mid1_r.params.tau * mid1_r.y)
        + mid1_r.params.beta * mid1_r.y
        - gamma * pi_delayed;

    double k2_x_l, k2_y_l, k2_x_r, k2_y_r;
    calculate_derivatives(mid1_l, pi_tilde_mid1, k2_x_l, k2_y_l);
    calculate_derivatives(mid1_r, pi_tilde_mid1, k2_x_r, k2_y_r);

    LinearPsSource mid2_l = left;
    LinearPsSource mid2_r = right;
    const double ps_k3 = psAt(time + dt / 2.0);
    mid2_l.params.ps = ps_k3;
    mid2_r.params.ps = kClosedRightPs;
    mid2_l.x += k2_x_l * dt / 2.0;
    mid2_r.x += k2_x_r * dt / 2.0;
    mid2_l.y += k2_y_l * dt / 2.0;
    mid2_r.y += k2_y_r * dt / 2.0;

    const double pi_tilde_mid2 =
        mid2_l.params.alpha * (mid2_l.x - mid2_l.params.tau * mid2_l.y)
        + mid2_l.params.beta * mid2_l.y
        + mid2_r.params.alpha * (mid2_r.x - mid2_r.params.tau * mid2_r.y)
        + mid2_r.params.beta * mid2_r.y
        - gamma * pi_delayed;

    double k3_x_l, k3_y_l, k3_x_r, k3_y_r;
    calculate_derivatives(mid2_l, pi_tilde_mid2, k3_x_l, k3_y_l);
    calculate_derivatives(mid2_r, pi_tilde_mid2, k3_x_r, k3_y_r);

    LinearPsSource end_l = left;
    LinearPsSource end_r = right;
    const double ps_k4 = psAt(time + dt);
    end_l.params.ps = ps_k4;
    end_r.params.ps = kClosedRightPs;
    end_l.x += k3_x_l * dt;
    end_r.x += k3_x_r * dt;
    end_l.y += k3_y_l * dt;
    end_r.y += k3_y_r * dt;

    const double pi_tilde_end =
        end_l.params.alpha * (end_l.x - end_l.params.tau * end_l.y)
        + end_l.params.beta * end_l.y
        + end_r.params.alpha * (end_r.x - end_r.params.tau * end_r.y)
        + end_r.params.beta * end_r.y
        - gamma * pi_delayed;

    double k4_x_l, k4_y_l, k4_x_r, k4_y_r;
    calculate_derivatives(end_l, pi_tilde_end, k4_x_l, k4_y_l);
    calculate_derivatives(end_r, pi_tilde_end, k4_x_r, k4_y_r);

    left.x +=
        (k1_x_l + 2.0 * (k2_x_l + k3_x_l) + k4_x_l) * dt / 6.0;
    left.y +=
        (k1_y_l + 2.0 * (k2_y_l + k3_y_l) + k4_y_l) * dt / 6.0;
    right.x +=
        (k1_x_r + 2.0 * (k2_x_r + k3_x_r) + k4_x_r) * dt / 6.0;
    right.y +=
        (k1_y_r + 2.0 * (k2_y_r + k3_y_r) + k4_y_r) * dt / 6.0;

    time += dt;
    const double ps_next = psAt(time);
    left.params.ps = ps_next;
    right.params.ps = kClosedRightPs;

    // 3. Calculate pi after updating the state to t+dt.
    const double pi_tilde_next =
        left.params.alpha * (left.x - left.params.tau * left.y)
        + left.params.beta * left.y
        + right.params.alpha * (right.x - right.params.tau * right.y)
        + right.params.beta * right.y
        - gamma * pi_delayed;

    double dx_l, dy_l, dx_r, dy_r;
    calculate_derivatives(left, pi_tilde_next, dx_l, dy_l);
    calculate_derivatives(right, pi_tilde_next, dx_r, dy_r);

    const double pi =
        pi_tilde_next
        + left.params.beta * (-left.params.tau * dy_l)
        + right.params.beta * (-right.params.tau * dy_r);

    // 4. Save p_i(t+dt) in the existing circular delay buffer.
    pi_history[current_pos] = pi;
    current_pos = (current_pos + 1) % history_size;

    // 5. Save this time step.
    saveData();
}

void BirdsongModelLinearPs::saveData() {
    const int latest_pi_pos =
        (current_pos - 1 + history_size) % history_size;
    outfile
        << time << ","
        << pi_history[latest_pi_pos] << ","
        << left.x << ","
        << left.y << ","
        << right.x << ","
        << right.y << ","
        << left.params.epsilon << ","
        << right.params.epsilon << ","
        << left.params.ps << ","
        << right.params.ps << "\n";
}
