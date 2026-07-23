#include "BirdsongModel_linear_epsilon.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct LinearEpsilonCase {
    double epsilon_start;
    double epsilon_total_change;
    std::string name;
    std::string output_filename;
};

int main() {
    const double dt = 1.0e-7;
    const double L = 2.0;
    const double c_sound = 3.5e4;
    const double T_delay = 2.0 * L / c_sound;
    const double total_time = 0.4;

    // Linear epsilon schedule:
    // epsilon(t) = epsilon_start + epsilon_slope * t
    // epsilon_slope = epsilon_total_change / total_time
    //
    // The previous epsilon-change simulations are intentionally left in
    // main_epsilon_change.cpp. This executable is for the new linear-only study.
    const std::vector<LinearEpsilonCase> cases = {
        // 1. Initial-value effect with no temporal change.
        {
            2.7e7,
            0.0,
            "initial epsilon 2.7e7 constant",
            "simulation_linear_epsilon_start2.7e7_delta0.0e7.csv"
        },
        {
            3.2e7,
            0.0,
            "initial epsilon 3.2e7 constant",
            "simulation_linear_epsilon_start3.2e7_delta0.0e7.csv"
        },
        {
            3.7e7,
            0.0,
            "initial epsilon 3.7e7 constant",
            "simulation_linear_epsilon_start3.7e7_delta0.0e7.csv"
        },

        // 2. Slope effect around epsilon_start = 3.2e7.
        // Total changes are deliberately gentler than the previous 2.5e7 change.
        {
            3.2e7,
            -0.5e7,
            "initial epsilon 3.2e7 total change -0.5e7",
            "simulation_linear_epsilon_start3.2e7_delta-0.5e7.csv"
        },
        {
            3.2e7,
            -0.25e7,
            "initial epsilon 3.2e7 total change -0.25e7",
            "simulation_linear_epsilon_start3.2e7_delta-0.25e7.csv"
        },
        {
            3.2e7,
            0.25e7,
            "initial epsilon 3.2e7 total change +0.25e7",
            "simulation_linear_epsilon_start3.2e7_delta+0.25e7.csv"
        },
        {
            3.2e7,
            0.5e7,
            "initial epsilon 3.2e7 total change +0.5e7",
            "simulation_linear_epsilon_start3.2e7_delta+0.5e7.csv"
        },

        // 3. Combined edge cases inside the main 2.7e7--3.7e7 region.
        {
            2.7e7,
            0.5e7,
            "initial epsilon 2.7e7 total change +0.5e7",
            "simulation_linear_epsilon_start2.7e7_delta+0.5e7.csv"
        },
        {
            3.7e7,
            -0.5e7,
            "initial epsilon 3.7e7 total change -0.5e7",
            "simulation_linear_epsilon_start3.7e7_delta-0.5e7.csv"
        }
    };

    std::ofstream condition_file("simulation_linear_epsilon_conditions.csv");
    if (!condition_file) {
        std::cerr << "Could not open simulation_linear_epsilon_conditions.csv\n";
        return 1;
    }
    condition_file << std::setprecision(15);
    condition_file
        << "case_index,name,epsilon_start,epsilon_total_change,"
        << "epsilon_slope,epsilon_end,output_filename\n";

    const int num_steps = static_cast<int>(std::llround(total_time / dt));

    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const LinearEpsilonCase& simulation_case = cases[case_index];
        const double epsilon_slope =
            simulation_case.epsilon_total_change / total_time;
        const double epsilon_end =
            simulation_case.epsilon_start + simulation_case.epsilon_total_change;

        condition_file
            << case_index + 1 << ","
            << simulation_case.name << ","
            << simulation_case.epsilon_start << ","
            << simulation_case.epsilon_total_change << ","
            << epsilon_slope << ","
            << epsilon_end << ","
            << simulation_case.output_filename << "\n";

        std::cout
            << "Starting " << simulation_case.name
            << " (" << case_index + 1 << "/" << cases.size() << ")...\n";

        BirdsongModelLinearEpsilon model(
            dt,
            T_delay,
            simulation_case.epsilon_start,
            epsilon_slope,
            simulation_case.output_filename
        );
        model.saveData();

        for (int i = 0; i < num_steps; ++i) {
            model.step();
            if (i % 100000 == 0) {
                std::cout
                    << "Progress: "
                    << (100.0 * static_cast<double>(i) / num_steps)
                    << "%\r"
                    << std::flush;
            }
        }

        std::cout
            << "Progress: 100%\nSaved: "
            << simulation_case.output_filename << "\n";
    }

    std::cout << "All linear-epsilon simulations finished.\n";
    return 0;
}
