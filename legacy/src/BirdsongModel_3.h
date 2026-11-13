#ifndef BIRDSONG_MODEL_3_H
#define BIRDSONG_MODEL_3_H

#include <vector>
#include <fstream>
#include <string> 

using namespace std;

// パラメータ(TABLE I) 
struct Parameters {
    double epsilon, b, c, f0, ps, x0, h, tau, M;
    double A, D;
    double alpha, beta;
};

// 音源状態構造体 
struct Source {
    double x, y;
    Parameters params;
};

class BirdsongModel {
public:
    // コンストラクタ: epsilon, ps, 出力ファイル名を引数で受け取る
    BirdsongModel(double dt, double T_delay, double total_time,
                  double initial_epsilon_left, double initial_epsilon_right, double initial_ps, const std::string& output_filename);

    // デストラクタを追加 (ファイルを閉じるため)
    ~BirdsongModel();

    void step();
    void saveData();

private:
    Source left, right;
    
    double gamma;
    double time;
    double dt;

    vector<double> pi_history;
    int history_size;
    int current_pos;

    ofstream outfile; // ファイルストリームはメンバー変数として保持

       // 時間変化するパラメータのための変数
    double total_sim_time;
    double epsilon_start;
    double epsilon_end;
    double ps_start;
    double ps_end;

    void calculate_derivatives(const Source& s, double pi_tilde, double& dx_dt, double& dy_dt) const;

    // 【削除】線形補間用の関数も、遅延項を使い回すなら不要
    // double get_interpolated_pi(double target_time) const;
};

#endif // BIRDSONG_MODEL_3_H