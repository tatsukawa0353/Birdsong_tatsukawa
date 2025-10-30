#ifndef BIRDSONG_MODEL_H
#define BIRDSONG_MODEL_H

#include <vector>
#include <fstream>
#include <string>

using namespace std;

// パラメータ（TABLE I)
struct Parameters {
    double epsilon, b, c, f0, ps, x0, h, tau, M;
    double A, D;
    double alpha, beta;
};

// 左右どちらかの音源の状態を保持する構造体
struct Source {
    double x, y; // 位置と速度
    Parameters params;
};

class BirdsongModel {
public:
    // コンストラクタ: シミュレーションの時間設定とパラメータ初期化
    BirdsongModel(double dt, double T_delay, double total_time);
   
    // 1ステップシミュレーションを進める
    void step();

    // 結果をファイルに出力
    void saveData();

private:
    Source left, right; // 左右の音源
    
    double gamma; // 声道の反射係数
    double time;  // 現在のシミュレーション時刻
    double dt;    // 時間ステップ

    // 時間遅延 p_i(t - T) を管理するためのバッファ
    vector<double> pi_history;
    int history_size;
    int current_pos;

    // 結果を保存するファイルストリーム
    ofstream outfile;

   // 時間変化するパラメータのための変数
    double total_sim_time;
    double epsilon_start;
    double epsilon_end;
    double ps_start;
    double ps_end;

    // 論文の式(62)-(67)に対応する計算
    void calculate_derivatives(const Source& s, double pi_tilde, double& dx_dt, double& dy_dt) const;
};

#endif // BIRDSONG_MODEL_H