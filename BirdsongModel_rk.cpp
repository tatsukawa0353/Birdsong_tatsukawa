//BirdsongModel ここでは論文紹介の論文を再現します

#include "BirdsongModel.h"
#include <iostream>

BirdsongModel::BirdsongModel(double dt, double T_delay, double total_time)
 : time(0.0), dt(dt), total_sim_time(total_time) {

    // パラメータを論文の Table I から設定
    // 左音源 (left)
    left.params = {2.4e8, 2.0e4, 2.0e8, 4.9e4, 6.0e6, 0.04, 0.1, 1.0e-4, 5.0e-3, 1.0, 5.0e-3, 1.2e6, 1.5e3};
    left.x = 0.0; // 初期位置
    left.y = 0.0; // 初期速度
    
    // Fig. 5(a) のためのパラメータ設定
    //epsilon_start = 3.6e8;
    //epsilon_end = 2.6e8;
   // left.params.epsilon = epsilon_start; // 左音源のepsilonを開始値で初期化

    // 右音源 (right) 
    right.params = {1.4e8, 2.0e4, 2.0e8, 4.9e4, 6.0e6, 0.04, 0.1, 1.0e-4, 5.0e-3, 1.0, 5.0e-3, 1.2e6, 1.5e3};
    //right.params.f0 = 1.0e12;→Fig5の再現
    right.x = 0.0;
    right.y = 0.0;

    gamma = 0.0; // 反射係数

    // 時間遅延バッファの初期化
    history_size = static_cast<int>(T_delay / dt);
    pi_history.assign(history_size, 0.0); // 過去のpiを0で初期化
    current_pos = 0;

    // 出力ファイルを開く
    outfile.open("simulation_output_2(a).csv");
    outfile << "time,pi,x_left,y_left,x_right,y_right\n";
}

// 論文の式(62), (63)等に対応する微分係数の計算
void BirdsongModel::calculate_derivatives(const Source& s, double pi_tilde, double& dx_dt, double& dy_dt) const {
    // f_j(x_j, y_j) の計算 [式(64)]
    double f = -s.params.epsilon * s.x - s.params.b * s.y - s.params.c * s.x * s.x * s.y - s.params.f0;
    
    // p_tilde_gj の計算 [式(65)]
    double p_tilde_g = s.params.ps + (s.params.D * s.y - s.params.A) * (s.params.ps - pi_tilde);

    dx_dt = s.y; // 式(62)
    dy_dt = f + p_tilde_g; // 式(62)
}

void BirdsongModel::step() {
 //時間に応じてepsilonを変化させる ---
    //double progress = time / total_sim_time;
    //if (progress > 1.0) { progress = 1.0; } // 1.0を超えないようにする
    
    // epsilonを開始値から終了値へ線形に変化させる
    //double current_epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress;
    //left.params.epsilon = current_epsilon;

    // 1. p_i(t - T) を履歴から取得
    int past_pos = (current_pos - history_size + pi_history.size()) % pi_history.size();
    double pi_delayed = pi_history[past_pos];

    // 2. p_tilde_i(t) の計算 式(66)
    double pi_tilde = left.params.alpha * (left.x - left.params.tau * left.y) + left.params.beta * left.y +
                      right.params.alpha * (right.x - right.params.tau * right.y) + right.params.beta * right.y -
                      gamma * pi_delayed;

    // 3. 時刻tにおける微分係数(速度と加速度)を4次のルンゲクッタ法で計算
    double k1_x_l, k1_y_l, k1_x_r, k1_y_r;
    calculate_derivatives(left, pi_tilde, k1_x_l, k1_y_l);
    calculate_derivatives(right, pi_tilde, k1_x_r, k1_y_r);

    double k2_x_l, k2_y_l, k2_x_r, k2_y_r;
    calculate_derivatives(left, pi_tilde, k2_x_l, k2_y_l);
    calculate_derivatives(right, pi_tilde, k2_x_r, k2_y_r);

    double k3_x_l, k3_y_l, k3_x_r, k3_y_r;
    calculate_derivatives(left, pi_tilde, k3_x_l, k3_y_l);
    calculate_derivatives(right, pi_tilde, k3_x_r, k3_y_r);

    double k4_x_l, k4_y_l, k4_x_r, k4_y_r;
    calculate_derivatives(left, pi_tilde, k4_x_l, k4_y_l);
    calculate_derivatives(right, pi_tilde, k4_x_r, k4_y_r);

    // 4. p_i(t) の計算 
    // 時刻tの加速度(dy_l, dy_r)を使って、時刻tのpiを計算
    double pi = pi_tilde + left.params.beta * (-left.params.tau * dy_l) + 
                         right.params.beta * (-right.params.tau * dy_r);

    left.x += (k1_x_l + 2.0 * (k2_x_l + k3_x_l) + k4_x_l ) / 6.0;
    left.y += (k1_y_l + 2.0 * (k2_y_l + k3_y_l) + k4_y_l ) / 6.0;
    right.x += (k1_x_r + 2.0 * (k2_x_r + k3_x_r) + k4_x_r ) / 6.0;
    right.y += (k1_y_r + 2.0 * (k2_y_r + k3_y_r) + k4_y_r ) / 6.0;

    // 6. p_i(t) を履歴に保存
    pi_history[current_pos] = pi;
    current_pos = (current_pos + 1) % history_size;

    time += dt;

    // 7. 定期的にデータを保存
    if (static_cast<int>(time / dt) % 10 == 0) { // 10ステップごとに保存
        saveData();
    }
}

void BirdsongModel::saveData() {
    outfile << time << "," << pi_history[current_pos-1] << "," << left.x << "," << left.y << "," << right.x << "," << right.y << "\n";
}