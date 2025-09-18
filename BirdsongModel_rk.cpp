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

    // 1. p_i(t - T) を履歴から取得
    int past_pos = (current_pos - history_size + pi_history.size()) % pi_history.size();
    double pi_delayed = pi_history[past_pos];

    // 2. p_tilde_i(t) の計算 式(66)
    double pi_tilde = left.params.alpha * (left.x - left.params.tau * left.y) + left.params.beta * left.y +
                      right.params.alpha * (right.x - right.params.tau * right.y) + right.params.beta * right.y -
                      gamma * pi_delayed;

    // 3. 時刻tにおける微分係数(速度と加速度)を計算
    double dx_l, dy_l, dx_r, dy_r;
    calculate_derivatives(left, pi_tilde, dx_l, dy_l);
    calculate_derivatives(right, pi_tilde, dx_r, dy_r);

    // 4. p_i(t) の計算,時刻tの加速度(dy_l, dy_r)を使って、時刻tのpiを計算
    double pi = pi_tilde + left.params.beta * (-left.params.tau * dy_l) + 
                         right.params.beta * (-right.params.tau * dy_r);

    // 5．4次ルンゲクッタ法で x と y を t+dt の値に更新
    //k1
    double k1_x_l, k1_y_l, k1_x_r, k1_y_r;
    calculate_derivatives(left, pi_tilde, k1_x_l, k1_y_l);
    calculate_derivatives(right, pi_tilde, k1_x_r, k1_y_r);

    //k2
    Source mid1_l = left, mid1_r = right;
    mid1_l.x += k1_x_l * dt / 2;
    mid1_r.x += k1_x_r * dt / 2;
    mid1_l.y += k1_y_l * dt / 2;
    mid1_r.y += k1_y_r * dt / 2;

    // 中間点1のpi_tildeを計算
    double pi_tilde_mid1 = mid1_l.params.alpha * (mid1_l.x - mid1_l.params.tau * mid1_l.y) + mid1_l.params.beta * mid1_l.y +
                           mid1_r.params.alpha * (mid1_r.x - mid1_r.params.tau * mid1_r.y) + mid1_r.params.beta * mid1_r.y -
                           gamma * pi_delayed;
    //↑を用いて↓計算
    double k2_x_l, k2_y_l, k2_x_r, k2_y_r;
    calculate_derivatives(mid1_l, pi_tilde_mid1, k2_x_l, k2_y_l);
    calculate_derivatives(mid1_r, pi_tilde_mid1, k2_x_r, k2_y_r);

    //k3
    Source mid2_l = left, mid2_r = right;
    mid2_l.x += k2_x_l * dt / 2;
    mid2_r.x += k2_x_r * dt / 2;
    mid2_l.y += k2_y_l * dt / 2;
    mid2_r.y += k2_y_r * dt / 2;

    // 中間点2のpi_tildeを計算
    double pi_tilde_mid2 = mid2_l.params.alpha * (mid2_l.x - mid2_l.params.tau * mid2_l.y) + mid2_l.params.beta * mid2_l.y +
                           mid2_r.params.alpha * (mid2_r.x - mid2_r.params.tau * mid2_r.y) + mid2_r.params.beta * mid2_r.y -
                           gamma * pi_delayed;
    //↑を用いて↓計算
    double k3_x_l, k3_y_l, k3_x_r, k3_y_r;
    calculate_derivatives(mid2_l, pi_tilde_mid2, k3_x_l, k3_y_l);
    calculate_derivatives(mid2_r, pi_tilde_mid2, k3_x_r, k3_y_r);

    //k4
    Source mid3_l = left, mid3_r = right;
    end_l.x += k3_x_l * dt;
    end_r.x += k3_x_r * dt;
    end_l.y += k3_y_l * dt;
    end_r.y += k3_y_r * dt;

    // 終点のpi_tildeを再計算
    double pi_tilde_end = end_l.params.alpha * (end_l.x - end_l.params.tau * end_l.y) + end_l.params.beta * end_l.y +
                          end_r.params.alpha * (end_r.x - end_r.params.tau * end_r.y) + end_r.params.beta * end_r.y -
                          gamma * pi_delayed;

    //↑を用いて↓計算
    double k4_x_l, k4_y_l, k4_x_r, k4_y_r;
    calculate_derivatives(end_l, pi_tilde, k4_x_l, k4_y_l);
    calculate_derivatives(end_r, pi_tilde, k4_x_r, k4_y_r);
 
    //最終的に
    left.x += (k1_x_l + 2.0 * (k2_x_l + k3_x_l) + k4_x_l ) * dt / 6.0;
    left.y += (k1_y_l + 2.0 * (k2_y_l + k3_y_l) + k4_y_l ) * dt / 6.0;
    right.x += (k1_x_r + 2.0 * (k2_x_r + k3_x_r) + k4_x_r ) * dt / 6.0;
    right.y += (k1_y_r + 2.0 * (k2_y_r + k3_y_r) + k4_y_r ) * dt / 6.0;
   
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