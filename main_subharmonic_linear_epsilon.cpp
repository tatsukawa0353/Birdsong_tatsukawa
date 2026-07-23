#include "BirdsongModel_linear_epsilon.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct LinearEpsilonCase {
    double epsilon_start;
    double epsilon_total_change;
    std::string name;
    std::string output_filename;
};

std::string format_epsilon_label(double epsilon) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << epsilon / 1.0e7;
    std::string label = stream.str();
    while (!label.empty() && label.back() == '0') {
        label.pop_back();
    }
    if (!label.empty() && label.back() == '.') {
        label.push_back('0');
    }
    return label + "e7";
}

std::string format_delta_label(double delta) {
    std::ostringstream stream;
    if (delta > 0.0) {
        stream << "+";
    }
    stream << std::fixed << std::setprecision(2) << delta / 1.0e7;
    std::string label = stream.str();
    while (!label.empty() && label.back() == '0') {
        label.pop_back();
    }
    if (!label.empty() && label.back() == '.') {
        label.push_back('0');
    }
    return label + "e7";
}

int main() {
    const double dt = 1.0e-7;
    const double L = 2.0;
    const double c_sound = 3.5e4;
    const double T_delay = 2.0 * L / c_sound;
    const double total_time = 0.4;
    const std::filesystem::path result_directory =
        "subharmonic_linear_epsilon_results";
    const std::filesystem::path csv_directory = result_directory / "csv";
    std::filesystem::create_directories(csv_directory);

    // Linear epsilon schedule:
    // epsilon(t) = epsilon_start + epsilon_slope * t
    // epsilon_slope = epsilon_total_change / total_time
    //
    // This executable focuses on linearly changing epsilon with ps = 3.2e6.
    // Start points are spaced to match the existing constant conditions as
    // closely as possible: roughly 0.3e7--0.35e7 apart.
    const std::vector<double> epsilon_starts = {
        2.5e7,
        2.8e7,
        3.1e7,
        3.4e7,
        3.7e7,
        4.0e7,
        4.35e7,
        4.7e7,
        5.0e7
    };
    const std::vector<double> epsilon_total_changes = {
        0.0,
        0.65e7,
        -0.65e7,
        1.3e7,
        -1.3e7
    };
    std::vector<LinearEpsilonCase> cases;
    cases.reserve(epsilon_starts.size() * epsilon_total_changes.size());
    for (const double epsilon_start : epsilon_starts) {
        for (const double epsilon_total_change : epsilon_total_changes) {
            const std::string start_label = format_epsilon_label(epsilon_start);
            const std::string delta_label =
                format_delta_label(epsilon_total_change);
            const double epsilon_end = epsilon_start + epsilon_total_change;

            std::string change_description = "constant";
            if (epsilon_total_change > 0.0) {
                change_description =
                    epsilon_total_change == 0.65e7 ? "slow increase" : "fast increase";
            } else if (epsilon_total_change < 0.0) {
                change_description =
                    epsilon_total_change == -0.65e7 ? "slow decrease" : "fast decrease";
            }

            cases.push_back({
                epsilon_start,
                epsilon_total_change,
                change_description + " epsilon " + start_label
                    + " to " + format_epsilon_label(epsilon_end),
                (csv_directory / (
                    "simulation_subharmonic_linear_epsilon_start" + start_label
                    + "_delta" + delta_label + ".csv"
                )).string()
            });
        }
    }

    const std::filesystem::path condition_path =
        result_directory / "simulation_subharmonic_linear_epsilon_conditions.csv";
    std::ofstream condition_file(condition_path);
    if (!condition_file) {
        std::cerr
            << "Could not open " << condition_path << "\n";
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

    std::cout << "All subharmonic-region linear-epsilon simulations finished.\n";
    return 0;
}
