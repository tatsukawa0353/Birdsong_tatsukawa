// RK4 equations copied from BirdsongModel_10*10.cpp; see reports/model_audit.md.
#include <vector>
#include <cmath>
#include <algorithm>
#include <new>
struct Parameters { double epsilon,b,c,f0,ps,x0,h,tau,M,A,D,alpha,beta; };
struct Source { double x,y; Parameters params; };
struct Model {
    Source left, right;
    double gamma=0.9, time=0, dt=1e-7, last_pi=0;
    int history_size=static_cast<int>((4.0/3.5e4)/1e-7), current_pos=0;
    std::vector<double> pi_history;
    Model() : pi_history(history_size,0.0) {}
    void calculate_derivatives(const Source&,double,double&,double&) const;
    void step();
};
// 論文の式(62)~(65)に対応する計算
void Model::calculate_derivatives(const Source& s, double pi_tilde, double& dx_dt, double& dy_dt) const {
    // f_j(x_j, y_j) 式(64)
    double f = -s.params.epsilon * s.x - s.params.b * s.y - s.params.c * s.x * s.x * s.y - s.params.f0;
    
    // p_tilde_gj 式(65)
    double p_tilde_g = s.params.ps + (s.params.D * s.y - s.params.A) * (s.params.ps - pi_tilde);

    dx_dt = s.y; // 式(62)
    dy_dt = f + p_tilde_g; // 式(62)
}

//t+dtの状態を計算するとする．
void Model::step() {
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
        last_pi = pi;
    //}
}


// Opaque C ABI: no exceptions cross the Python boundary. Caller owns buffers.
extern "C" {
void* bird_create() { try { return new Model(); } catch (...) { return nullptr; } }
void bird_destroy(void* p) { delete static_cast<Model*>(p); }
int bird_history_size(void* p) { return static_cast<Model*>(p)->history_size; }
int bird_reset(void* p, const double* controls, const double* state,
               const double* history, int count, int condition) {
    auto& m=*static_cast<Model*>(p);
    if (count!=m.history_size || condition<0 || condition>2) return -1;
    m.left.params={controls[0],2e4,2e8,controls[3],controls[2],0.04,0.1,1e-4,5e-3,1.0,0.01,1.2e6,1.5e3};
    m.right.params=m.left.params;
    m.right.params.epsilon=controls[1]; m.right.params.f0=controls[4];
    if (condition==0) { m.right.params.alpha=0; m.right.params.beta=0; }
    m.left.x=state[0]; m.left.y=state[1]; m.right.x=state[2]; m.right.y=state[3];
    std::copy(history,history+count,m.pi_history.begin());
    m.current_pos=0; m.time=0; m.last_pi=history[count-1]; return 0;
}
int bird_controls(void* p, const double* c) {
    auto& m=*static_cast<Model*>(p);
    m.left.params.epsilon=c[0]; m.right.params.epsilon=c[1];
    m.left.params.ps=c[2]; m.right.params.ps=c[2];
    m.left.params.f0=c[3]; m.right.params.f0=c[4]; return 0;
}
int bird_advance(void* p, int n, double* out) {
    auto& m=*static_cast<Model*>(p);
    for(int i=0;i<n;++i) {
        m.step();
        if (!std::isfinite(m.last_pi) || !std::isfinite(m.left.x) ||
            !std::isfinite(m.right.x) || !std::isfinite(m.left.y) || !std::isfinite(m.right.y)) return -1;
        out[i]=m.last_pi;
    }
    return 0;
}
void bird_state(void* p, double* out) {
    auto& m=*static_cast<Model*>(p);
    out[0]=m.time; out[1]=m.last_pi; out[2]=m.left.x; out[3]=m.left.y;
    out[4]=m.right.x; out[5]=m.right.y;
}
// Offline verification of vibration, measured at every integration step.
int bird_advance_rms(void* p, int n, double* out, double* rms) {
    auto& m=*static_cast<Model*>(p);
    double sums[4]={0,0,0,0};
    for(int i=0;i<n;++i) {
        m.step();
        const double state[4]={m.left.x,m.left.y,m.right.x,m.right.y};
        if (!std::isfinite(m.last_pi)) return -1;
        for(int j=0;j<4;++j) {
            if(!std::isfinite(state[j])) return -1;
            sums[j]+=state[j]*state[j];
        }
        out[i]=m.last_pi;
    }
    for(int j=0;j<4;++j) rms[j]=std::sqrt(sums[j]/n);
    return 0;
}
}
