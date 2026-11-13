#include "BirdsongModel_distance1.h"

#include <cmath>
#include <iostream>

int main() {
    constexpr double dt = 1.0e-7;
    constexpr double trachea_length = 2.0;
    constexpr double sound_speed = 3.5e4;
    constexpr double delay = 2.0 * trachea_length / sound_speed;
    constexpr double total_time = 0.12;

    // Same parameter schedules as the current comparison result.
    constexpr double epsilon_ramp_start = 0.045;
    constexpr double epsilon_ramp_end = 0.12;
    constexpr double epsilon_start = 8.0e7;
    constexpr double epsilon_end = 3.5e7;
    constexpr double ps_ramp_start = 0.04;
    constexpr double ps_ramp_end = 0.12;
    constexpr double ps_start = 2.5e6;
    constexpr double ps_end = 3.0e6;
    constexpr const char* output_filename =
        "simulation_distance1_single_source.csv";

    BirdsongModelDistance1 model(
        dt,
        delay,
        epsilon_ramp_start,
        epsilon_ramp_end,
        epsilon_start,
        epsilon_end,
        ps_ramp_start,
        ps_ramp_end,
        ps_start,
        ps_end,
        output_filename,
        true
    );
    model.saveData();

    const int num_steps = static_cast<int>(std::llround(total_time / dt));
    for (int step = 0; step < num_steps; ++step) {
        model.step();
    }

    std::cout << "Saved single-source result: " << output_filename << '\n';
    return 0;
}
