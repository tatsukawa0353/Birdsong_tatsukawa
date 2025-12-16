#include "BirdsongModel_2.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h> 
#include <unistd.h>
using namespace std;

int main() {
    // シミュレーション設定
    const double dt = 1.0e-7;           // 時間ステップ (秒)
    const double L = 2.0;               // 声道の長さ (cm) [cite: 293]
    const double c_sound = 3.5e4;       // 音速 (cm/s) [cite: 293]
    const double T_delay = 2.0 * L / c_sound; // 音の往復時間 (秒)
    const double total_time = 0.12;      // シミュレーション総時間 (秒)

    //出力フォルダ名定義
    const string output_folder = "simulation_results_1_f0=4.9e4_x0=0.02/";

    // フォルダ自動作成
    struct stat st;
    if (stat(output_folder.c_str(), &st) != 0) {
        // Unix/Linux系の場合 (WSL環境など)
         if (mkdir(output_folder.c_str(), 0777) != 0) {
             cerr << "Warning: Could not create directory " << output_folder << endl;
             // フォルダ作成に失敗しても続行する (ファイル書き込み時にエラーになる可能性あり)
         } else {
             cout << "Created output directory: " << output_folder << endl;
         }
    }

    //パラメータ掃引設定
    vector<double> epsilon_values;
    for (int i = 0; i < 10; i++) {
        epsilon_values.push_back( (2.0 + i * (28.0/9.0)) * 1e7 );//パターン1，2 
        //epsilon_values.push_back( (0.50 + i * (0.50)) * 1e7 );  //low parameters
    } 
    vector<double> ps_values;
    for (int i = 0; i < 10; i++) {
        ps_values.push_back( (1.0 + i * (5.0/9.0)) * 1e6 );//パターン1，2
        //ps_values.push_back( (1.60 + i * (2.00/45.00)) * 1e6 ); //low parameters 使わない
    }

    cout << "Starting parameter sweep (" << epsilon_values.size() << " x " << ps_values.size() << " = " << epsilon_values.size() * ps_values.size() << " simulations)" << endl;
    
// --- 2重ループで全組み合わせを実行 ---
    for (double current_epsilon : epsilon_values) {
        for (double current_ps : ps_values) {

            // --- ファイル名生成時にフォルダ名を追加 ---
            std::stringstream ss;
            ss << output_folder // フォルダ名を追加
               << "sim_output_eps_" << std::scientific << std::setprecision(2) << current_epsilon
               << "_ps_" << std::scientific << std::setprecision(2) << current_ps << ".csv";
            std::string output_filename = ss.str();
            // ----------------------------------------------

            std::cout << "Running simulation for epsilon=" << current_epsilon << ", ps=" << current_ps << " -> " << output_filename << std::endl;


    // モデルのインスタンスを作成
    {
    BirdsongModel model(dt, T_delay, total_time, current_epsilon, current_ps, output_filename);

    model.saveData();

    cout << "Simulation starting..." << endl;
    
    // シミュレーションループ
    int num_steps = static_cast<int>(total_time / dt);
    for (int i = 0; i < num_steps; i++) {
        model.step();
        if (i % 10000 == 0) { // 進捗表示
            cout << "Progress: " << (100.0 * i / num_steps) << "%\r";
        }
    }
    }
    }
}
    cout << "\nParameter sweep finished. Results saved in " << output_folder << endl;
   
    return 0;
}