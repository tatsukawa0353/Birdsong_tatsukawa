#include "BirdsongModel_linear_ps.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct LinearPsCase {
    double ps_start;
    double ps_total_change;
    std::string name;
    std::string output_filename;
};

std::string format_ps_label(double ps) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << ps / 1.0e6;
    std::string label = stream.str();
    while (!label.empty() && label.back() == '0') {
        label.pop_back();
    }
    if (!label.empty() && label.back() == '.') {
        label.push_back('0');
    }
    return label + "e6";
}

std::string format_delta_label(double delta) {
    std::ostringstream stream;
    if (delta > 0.0) {
        stream << "+";
    }
    stream << std::fixed << std::setprecision(2) << delta / 1.0e6;
    std::string label = stream.str();
    while (!label.empty() && label.back() == '0') {
        label.pop_back();
    }
    if (!label.empty() && label.back() == '.') {
        label.push_back('0');
    }
    return label + "e6";
}

int main() {
    const double dt = 1.0e-7;
    const double L = 2.0;
    const double c_sound = 3.5e4;
    const double T_delay = 2.0 * L / c_sound;
    const double total_time = 0.4;
    const std::filesystem::path result_directory =
        "single_source_linear_ps_results";
    const std::filesystem::path csv_directory = result_directory / "csv";
    std::filesystem::create_directories(csv_directory);

    const std::vector<double> ps_starts = {
        2.5e6,
        2.8e6,
        3.1e6,
        3.4e6,
        3.7e6,
        4.0e6
    };
    const std::vector<double> ps_total_changes = {
        0.0,
        0.35e6,
        -0.35e6,
        0.7e6,
        -0.7e6
    };

    std::vector<LinearPsCase> cases;
    cases.reserve(ps_starts.size() * ps_total_changes.size());
    for (const double ps_start : ps_starts) {
        for (const double ps_total_change : ps_total_changes) {
            const std::string start_label = format_ps_label(ps_start);
            const std::string delta_label = format_delta_label(ps_total_change);
            const double ps_end = ps_start + ps_total_change;

            std::string change_description = "constant";
            if (ps_total_change > 0.0) {
                change_description =
                    ps_total_change == 0.35e6 ? "slow increase" : "fast increase";
            } else if (ps_total_change < 0.0) {
                change_description =
                    ps_total_change == -0.35e6 ? "slow decrease" : "fast decrease";
            }

            cases.push_back({
                ps_start,
                ps_total_change,
                change_description + " ps " + start_label
                    + " to " + format_ps_label(ps_end),
                (csv_directory / (
                    "simulation_single_source_linear_ps_start" + start_label
                    + "_delta" + delta_label + ".csv"
                )).string()
            });
        }
    }

    const std::filesystem::path condition_path =
        result_directory / "simulation_single_source_linear_ps_conditions.csv";
    std::ofstream condition_file(condition_path);
    if (!condition_file) {
        std::cerr << "Could not open " << condition_path << "\n";
        return 1;
    }
    condition_file << std::setprecision(15);
    condition_file
        << "case_index,name,ps_start,ps_total_change,"
        << "ps_slope,ps_end,epsilon_fixed,ps_right_fixed,right_alpha,right_beta,"
        << "output_filename\n";

    const int num_steps = static_cast<int>(std::llround(total_time / dt));

    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const LinearPsCase& simulation_case = cases[case_index];
        const double ps_slope = simulation_case.ps_total_change / total_time;
        const double ps_end =
            simulation_case.ps_start + simulation_case.ps_total_change;

        condition_file
            << case_index + 1 << ","
            << simulation_case.name << ","
            << simulation_case.ps_start << ","
            << simulation_case.ps_total_change << ","
            << ps_slope << ","
            << ps_end << ","
            << 3.7e7 << ","
            << 3.2e6 << ","
            << 0.0 << ","
            << 0.0 << ","
            << simulation_case.output_filename << "\n";

        std::cout
            << "Starting " << simulation_case.name
            << " (" << case_index + 1 << "/" << cases.size() << ")...\n";

        BirdsongModelLinearPs model(
            dt,
            T_delay,
            simulation_case.ps_start,
            ps_slope,
            simulation_case.output_filename,
            false
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

    std::cout << "All single-source linear-ps simulations finished.\n";
    return 0;
}
