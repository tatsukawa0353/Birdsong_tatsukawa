#include "BirdsongModel.h"
#include <iostream>

using namespace std;

int main() {
    // シミュレーション設定
    const double dt = 1.0e-7;           // 時間ステップ (秒)
    const double L = 2.0;               // 声道の長さ (cm) [cite: 293]
    const double c_sound = 3.5e4;       // 音速 (cm/s) [cite: 293]
    const double T_delay = 2.0 * L / c_sound; // 音の往復時間 (秒)
    const double total_time = 0.12;      // シミュレーション総時間 (秒)

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