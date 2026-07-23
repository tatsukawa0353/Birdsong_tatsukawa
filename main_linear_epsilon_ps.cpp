#include "BirdsongModel_linear_epsilon_ps.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct ChangePattern {
    std::string key;
    std::string description;
    double epsilon_total_change;
    double ps_total_change;
};

struct LinearEpsilonPsCase {
    double epsilon_start;
    double epsilon_total_change;
    double ps_start;
    double ps_total_change;
    std::string pattern_key;
    std::string name;
    std::string output_filename;
};

std::string format_scaled_label(double value, double scale, const std::string& suffix) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value / scale;
    std::string label = stream.str();
    while (!label.empty() && label.back() == '0') {
        label.pop_back();
    }
    if (!label.empty() && label.back() == '.') {
        label.push_back('0');
    }
    return label + suffix;
}

std::string format_delta_label(double value, double scale, const std::string& suffix) {
    std::ostringstream stream;
    if (value > 0.0) {
        stream << "+";
    }
    stream << std::fixed << std::setprecision(2) << value / scale;
    std::string label = stream.str();
    while (!label.empty() && label.back() == '0') {
        label.pop_back();
    }
    if (!label.empty() && label.back() == '.') {
        label.push_back('0');
    }
    return label + suffix;
}

int main() {
    const double dt = 1.0e-7;
    const double L = 2.0;
    const double c_sound = 3.5e4;
    const double T_delay = 2.0 * L / c_sound;
    const double total_time = 0.4;
    const std::filesystem::path result_directory = "linear_epsilon_ps_results";
    const std::filesystem::path csv_directory = result_directory / "csv";
    std::filesystem::create_directories(csv_directory);

    // Combined linear schedules:
    // epsilon(t) = epsilon_start + epsilon_slope * t
    // ps_left(t) = ps_start + ps_slope * t
    //
    // The right/closed source keeps ps_right = 3.2e6.
    // We combine only matching change-pattern categories:
    // constant, slow increase/decrease, fast increase/decrease.
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
    const std::vector<double> ps_starts = {
        2.5e6,
        2.8e6,
        3.1e6,
        3.4e6,
        3.7e6,
        4.0e6
    };
    const std::vector<ChangePattern> patterns = {
        {"constant", "constant", 0.0, 0.0},
        {"slow_increase", "slow increase", 0.65e7, 0.35e6},
        {"slow_decrease", "slow decrease", -0.65e7, -0.35e6},
        {"fast_increase", "fast increase", 1.3e7, 0.7e6},
        {"fast_decrease", "fast decrease", -1.3e7, -0.7e6}
    };

    std::vector<LinearEpsilonPsCase> cases;
    cases.reserve(epsilon_starts.size() * ps_starts.size() * patterns.size());
    for (const double epsilon_start : epsilon_starts) {
        for (const double ps_start : ps_starts) {
            for (const ChangePattern& pattern : patterns) {
                const std::string epsilon_start_label =
                    format_scaled_label(epsilon_start, 1.0e7, "e7");
                const std::string epsilon_delta_label =
                    format_delta_label(pattern.epsilon_total_change, 1.0e7, "e7");
                const std::string ps_start_label =
                    format_scaled_label(ps_start, 1.0e6, "e6");
                const std::string ps_delta_label =
                    format_delta_label(pattern.ps_total_change, 1.0e6, "e6");
                const double epsilon_end =
                    epsilon_start + pattern.epsilon_total_change;
                const double ps_end = ps_start + pattern.ps_total_change;

                cases.push_back({
                    epsilon_start,
                    pattern.epsilon_total_change,
                    ps_start,
                    pattern.ps_total_change,
                    pattern.key,
                    pattern.description
                        + " epsilon " + epsilon_start_label + " to "
                        + format_scaled_label(epsilon_end, 1.0e7, "e7")
                        + ", ps " + ps_start_label + " to "
                        + format_scaled_label(ps_end, 1.0e6, "e6"),
                    (csv_directory / (
                        "simulation_linear_epsilon_ps_"
                        "epsstart" + epsilon_start_label
                        + "_epsdelta" + epsilon_delta_label
                        + "_psstart" + ps_start_label
                        + "_psdelta" + ps_delta_label
                        + ".csv"
                    )).string()
                });
            }
        }
    }

    const std::filesystem::path condition_path =
        result_directory / "simulation_linear_epsilon_ps_conditions.csv";
    std::ofstream condition_file(condition_path);
    if (!condition_file) {
        std::cerr << "Could not open " << condition_path << "\n";
        return 1;
    }
    condition_file << std::setprecision(15);
    condition_file
        << "case_index,pattern_key,name,epsilon_start,epsilon_total_change,"
        << "epsilon_slope,epsilon_end,ps_start,ps_total_change,ps_slope,"
        << "ps_end,ps_right_fixed,output_filename\n";

    const int num_steps = static_cast<int>(std::llround(total_time / dt));

    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const LinearEpsilonPsCase& simulation_case = cases[case_index];
        const double epsilon_slope =
            simulation_case.epsilon_total_change / total_time;
        const double epsilon_end =
            simulation_case.epsilon_start + simulation_case.epsilon_total_change;
        const double ps_slope = simulation_case.ps_total_change / total_time;
        const double ps_end =
            simulation_case.ps_start + simulation_case.ps_total_change;

        condition_file
            << case_index + 1 << ","
            << simulation_case.pattern_key << ","
            << simulation_case.name << ","
            << simulation_case.epsilon_start << ","
            << simulation_case.epsilon_total_change << ","
            << epsilon_slope << ","
            << epsilon_end << ","
            << simulation_case.ps_start << ","
            << simulation_case.ps_total_change << ","
            << ps_slope << ","
            << ps_end << ","
            << 3.2e6 << ","
            << simulation_case.output_filename << "\n";

        std::cout
            << "Starting " << simulation_case.name
            << " (" << case_index + 1 << "/" << cases.size() << ")...\n";

        BirdsongModelLinearEpsilonPs model(
            dt,
            T_delay,
            simulation_case.epsilon_start,
            epsilon_slope,
            simulation_case.ps_start,
            ps_slope,
            simulation_case.output_filename
        );
        model.saveData();

        for (int i = 0; i < num_steps; ++i) {
            model.step();
            if (i % 400000 == 0) {
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

    std::cout << "All linear-epsilon-ps simulations finished.\n";
    return 0;
}
