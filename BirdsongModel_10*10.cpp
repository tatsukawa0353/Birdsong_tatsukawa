//BirdsongModel εとpsの10×10パターンの音出力用

#include "BirdsongModel_2.h"
#include <iostream>
#include <cmath>
#include <iomanip>

BirdsongModel::BirdsongModel(double dt, double T_delay, double total_time,
                             double initial_epsilon, double initial_ps, const std::string& output_filename)
 : time(0.0), dt(dt), total_sim_time(total_time) {

    // パラメータ設定
    // 左音源 (left)
    left.params = {initial_epsilon, 2.0e4, 2.0e8, 4.9e4, initial_ps, 0.04, 0.1, 1.0e-4, 5.0e-3, 1.0, 0.01, 1.2e6, 1.5e3};
    left.x = 0.0; // 初期位置
    left.y = 0.0; // 初期速度

    // 右音源 (right)  ~Fig.5の場合はミュートする~
    //right.params = {initial_epsilon, 2.0e4, 2.0e8, 4.9e4, initial_ps, 0.04, 0.1, 1.0e-4, 5.0e-3, 1.0, 0.01, 0, 0};//パターン2 one bronchus で使用
    right.params = left.params;//パターン1で使用
    right.params.f0 = 0.1e7; //筋肉に力を入れて絞めるイメージ　パターン1 two bronchiで使用 0.1 0.4 0.7 1.0
    right.x = 0.0;
    right.y = 0.0;

    gamma = 0.9; // 反射係数

    // 時間遅延バッファの初期化
    history_size = static_cast<int>(T_delay / dt);
    pi_history.assign(history_size, 0.0); // 過去のpiを0で初期化
    current_pos = 0;

    // 出力ファイルを開く
    outfile.open(output_filename);
    outfile << "time,pi,x_left,y_left,x_right,y_right\n";
}

// 【追加】デストラクタ: ファイルを確実に閉じる
BirdsongModel::~BirdsongModel() {
    if (outfile.is_open()) {
        outfile.close();
    }
}

// 論文の式(62)~(65)に対応する計算
void BirdsongModel::calculate_derivatives(const Source& s, double pi_tilde, double& dx_dt, double& dy_dt) const {
    // f_j(x_j, y_j) 式(64)
    double f = -s.params.epsilon * s.x - s.params.b * s.y - s.params.c * s.x * s.x * s.y - s.params.f0;
    
    // p_tilde_gj 式(65)
    double p_tilde_g = s.params.ps + (s.params.D * s.y - s.params.A) * (s.params.ps - pi_tilde);

    dx_dt = s.y; // 式(62)
    dy_dt = f + p_tilde_g; // 式(62)
}

//t+dtの状態を計算するとする．
void BirdsongModel::step() {
    // 1. p_i(t - T) を履歴から取得
    int past_pos = (current_pos - history_size + pi_history.size()) % pi_history.size();
    double pi_delayed = pi_history[past_pos];

    // 2．4次ルンゲクッタ法で x と y を t+dt の値に更新 
    //k1
    double pi_tilde_start = left.params.alpha * (left.x - left.params.tau * left.y) + left.params.beta * left.y +
                      right.params.alpha * (right.x - right.params.tau * right.y) + right.params.beta * right.y -
                      gamma * pi_delayed;

    double k1_x_l, k1_y_l, k1_x_r, k1_y_r;
    calculate_derivatives(left, pi_tilde_start, k1_x_l, k1_y_l);
    calculate_derivatives(right, pi_tilde_start, k1_x_r, k1_y_r);

    //k2
    Source mid1_l = left, mid1_r = right;
    mid1_l.x += k1_x_l * dt / 2.0;
    mid1_r.x += k1_x_r * dt / 2.0;
    mid1_l.y += k1_y_l * dt / 2.0;
    mid1_r.y += k1_y_r * dt / 2.0;

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
    mid2_l.x += k2_x_l * dt / 2.0;
    mid2_r.x += k2_x_r * dt / 2.0;
    mid2_l.y += k2_y_l * dt / 2.0;
    mid2_r.y += k2_y_r * dt / 2.0;

    // 中間点2のpi_tildeを計算
    double pi_tilde_mid2 = mid2_l.params.alpha * (mid2_l.x - mid2_l.params.tau * mid2_l.y) + mid2_l.params.beta * mid2_l.y +
                           mid2_r.params.alpha * (mid2_r.x - mid2_r.params.tau * mid2_r.y) + mid2_r.params.beta * mid2_r.y -
                           gamma * pi_delayed;
    //↑を用いて↓計算
    double k3_x_l, k3_y_l, k3_x_r, k3_y_r;
    calculate_derivatives(mid2_l, pi_tilde_mid2, k3_x_l, k3_y_l);
    calculate_derivatives(mid2_r, pi_tilde_mid2, k3_x_r, k3_y_r);

    //k4
    Source end_l = left, end_r = right;
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
    calculate_derivatives(end_l, pi_tilde_end, k4_x_l, k4_y_l);
    calculate_derivatives(end_r, pi_tilde_end, k4_x_r, k4_y_r);
 
    //最終的に
    left.x += (k1_x_l + 2.0 * (k2_x_l + k3_x_l) + k4_x_l ) * dt / 6.0;
    left.y += (k1_y_l + 2.0 * (k2_y_l + k3_y_l) + k4_y_l ) * dt / 6.0;
    right.x += (k1_x_r + 2.0 * (k2_x_r + k3_x_r) + k4_x_r ) * dt / 6.0;
    right.y += (k1_y_r + 2.0 * (k2_y_r + k3_y_r) + k4_y_r ) * dt / 6.0;

    //時刻更新
    time += dt;

    // 3. piの計算を、状態がt+dtに更新された後に行う
    double pi_tilde_next = left.params.alpha * (left.x - left.params.tau * left.y) + left.params.beta * left.y +
                           right.params.alpha * (right.x - right.params.tau * right.y) + right.params.beta * right.y -
                           gamma * pi_delayed;

    double dx_l, dy_l, dx_r, dy_r;
    calculate_derivatives(left, pi_tilde_next, dx_l, dy_l);
    calculate_derivatives(right, pi_tilde_next, dx_r, dy_r);

    double pi = pi_tilde_next + left.params.beta * (-left.params.tau * dy_l) + 
                         right.params.beta * (-right.params.tau * dy_r);

    // 4. p_i(t+dt) を履歴に保存
    pi_history[current_pos] = pi;
    current_pos = (current_pos + 1) % history_size;

    // 5. データを保存
        saveData();
    //}
}

void BirdsongModel::saveData() {
    // 現在の時刻(t+dt)と、それに対応する最新のpiを保存
    int latest_pi_pos = (current_pos - 1 + history_size) % history_size;
    outfile << time << "," << pi_history[latest_pi_pos] << "," << left.x << "," << left.y << "," << right.x << "," << right.y << "\n";
}