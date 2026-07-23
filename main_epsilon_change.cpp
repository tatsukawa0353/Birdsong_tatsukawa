#include "BirdsongModel_epsilon_change.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

struct SimulationCase {
    EpsilonPattern pattern;
    double epsilon_start;
    double epsilon_end;
    double sine_frequency;
    std::string name;
    std::string output_filename;
};

int main() {
    const double dt = 1.0e-7;
    const double L = 2.0;
    const double c_sound = 3.5e4;
    const double T_delay = 2.0 * L / c_sound;
    const double total_time = 0.4;

    const double epsilon_min = 2.5e7;
    const double epsilon_max = 5.0e7;

    const std::vector<SimulationCase> cases = {
        {
            EpsilonPattern::Linear,
            epsilon_max,
            epsilon_min,
            0.0,
            "linear decrease",
            "simulation_epsilon_linear_decrease.csv"
        },
        {
            EpsilonPattern::Linear,
            epsilon_min,
            epsilon_max,
            0.0,
            "linear increase",
            "simulation_epsilon_linear_increase.csv"
        },
        {
            EpsilonPattern::Quadratic,
            epsilon_max,
            epsilon_min,
            0.0,
            "quadratic decrease",
            "simulation_epsilon_quadratic_decrease.csv"
        },
        {
            EpsilonPattern::Quadratic,
            epsilon_min,
            epsilon_max,
            0.0,
            "quadratic increase",
            "simulation_epsilon_quadratic_increase.csv"
        },
        {
            EpsilonPattern::Step,
            epsilon_max,
            epsilon_min,
            0.0,
            "step decrease",
            "simulation_epsilon_step_decrease.csv"
        },
        {
            EpsilonPattern::Step,
            epsilon_min,
            epsilon_max,
            0.0,
            "step increase",
            "simulation_epsilon_step_increase.csv"
        },
        {
            EpsilonPattern::Sinusoidal,
            3.2e7,
            3.2e7,
            10.0 / 3.0,
            "sinusoidal f=3.333333 Hz",
            "simulation_epsilon_sine_f3.333333Hz.csv"
        },
        {
            EpsilonPattern::Sinusoidal,
            3.2e7,
            3.2e7,
            20.0 / 3.0,
            "sinusoidal f=6.666667 Hz",
            "simulation_epsilon_sine_f6.666667Hz.csv"
        },
        {
            EpsilonPattern::Sinusoidal,
            3.2e7,
            3.2e7,
            10.0,
            "sinusoidal f=10 Hz",
            "simulation_epsilon_sine_f10Hz.csv"
        }
    };

    const int num_steps = static_cast<int>(std::llround(total_time / dt));

    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const SimulationCase& simulation_case = cases[case_index];
        std::cout
            << "Starting " << simulation_case.name
            << " (" << case_index + 1 << "/" << cases.size() << ")...\n";

        BirdsongModel model(
            dt,
            T_delay,
            total_time,
            simulation_case.pattern,
            simulation_case.epsilon_start,
            simulation_case.epsilon_end,
            simulation_case.sine_frequency,
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

    std::cout << "All epsilon-change simulations finished.\n";
    return 0;
}
