#include "BirdsongModel_2.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>

using namespace std;

int main() {
    // シミュレーション設定
    const double dt = 1.0e-7;           // 時間ステップ (秒)
    const double L = 2.0;               // 声道の長さ (cm) [cite: 293]
    const double c_sound = 3.5e4;       // 音速 (cm/s) [cite: 293]
    const double T_delay = 2.0 * L / c_sound; // 音の往復時間 (秒)
    const double total_time = 0.12;      // シミュレーション総時間 (秒)

    //出力フォルダ名定義
    const string output_folder = "simulation_results_1/"

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
    

    // モデルのインスタンスを作成
    BirdsongModel model(dt, T_delay, total_time);

    model.saveData();

    cout << "Simulation starting..." << endl;
    
    // シミュレーションループ
    int num_steps = static_cast<int>(total_time / dt);
    for (int i = 0; i < num_steps; ++i) {
        model.step();
        if (i % 10000 == 0) { // 進捗表示
            cout << "Progress: " << (100.0 * i / num_steps) << "%\r";
        }
    }

    cout << "\nSimulation finished. Data saved to simulation_output_test1.csv" << endl;

    return 0;
}