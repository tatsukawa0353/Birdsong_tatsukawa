// Read-only diagnostics on an actual libbirdsong.so Model instance.
// Physics advances exclusively through the function pointer supplied by Python.
#include "native/model.cpp"
#include <limits>
extern "C" {
void issue27_read(void* p,double* o) {
    const auto& m=*static_cast<Model*>(p);
    const auto& l=m.left;const auto& r=m.right;
    double h=0;for(double v:m.pi_history)h=std::max(h,std::abs(v));
    const double v[]={l.params.alpha,l.params.beta,r.params.alpha,r.params.beta,
        l.params.f0,r.params.f0,l.params.epsilon,r.params.epsilon,l.params.ps,r.params.ps,
        l.params.tau,r.params.tau,l.params.x0,r.params.x0,l.params.D,r.params.D,
        m.dt,m.gamma,double(m.history_size),h,m.time,m.last_pi,l.x,l.y,r.x,r.y,
        l.params.alpha*(l.x-l.params.tau*l.y)+l.params.beta*l.y,
        r.params.alpha*(r.x-r.params.tau*r.y)+r.params.beta*r.y,
        m.pi_history[m.current_pos],double(m.current_pos)};
    std::copy(v,v+30,o);
}
int issue27_diagnostic_advance(void* p,int n,void* actual_fn,double* audio,
                              double* stats,double* samples,int stride) {
    using Fn=int(*)(void*,int,double*);const auto advance=reinterpret_cast<Fn>(actual_fn);
    auto& m=*static_cast<Model*>(p);const int k=12;
    double mean[k]={},m2[k]={},lo[k],hi[k],err=0;
    for(int j=0;j<k;++j){lo[j]=std::numeric_limits<double>::infinity();hi[j]=-lo[j];}
    stats[49]=0;
    for(int i=0;i<n;++i) {
        const double delayed=m.pi_history[m.current_pos];
        if(advance(p,1,audio+i))return -1; // Original loaded bird_advance, never m.step here.
        stats[49]=i+1;
        const auto& l=m.left;const auto& r=m.right;
        const double sl=l.params.alpha*(l.x-l.params.tau*l.y)+l.params.beta*l.y;
        const double sr=r.params.alpha*(r.x-r.params.tau*r.y)+r.params.beta*r.y;
        const double feedback=-m.gamma*delayed;
        const double pit=l.params.alpha*(l.x-l.params.tau*l.y)+l.params.beta*l.y+
            r.params.alpha*(r.x-r.params.tau*r.y)+r.params.beta*r.y-m.gamma*delayed;
        double dxl,dyl,dxr,dyr;m.calculate_derivatives(l,pit,dxl,dyl);m.calculate_derivatives(r,pit,dxr,dyr);
        const double ol=sl-l.params.beta*l.params.tau*dyl;
        const double oright=sr-r.params.beta*r.params.tau*dyr;
        const double v[]={l.x,l.y,r.x,r.y,sl,sr,feedback,pit,ol,oright,delayed,audio[i]};
        err=std::max(err,std::abs(ol+oright+feedback-audio[i]));
        for(int j=0;j<k;++j){
            if(!std::isfinite(v[j]))return -2;
            double delta=v[j]-mean[j];mean[j]+=delta/(i+1);m2[j]+=delta*(v[j]-mean[j]);
            lo[j]=std::min(lo[j],v[j]);hi[j]=std::max(hi[j],v[j]);
            if((i+1)%stride==0)samples[(i/stride)*k+j]=v[j];
        }
    }
    for(int j=0;j<k;++j){stats[j]=mean[j];stats[k+j]=std::sqrt(std::max(0.,m2[j]/n));stats[2*k+j]=lo[j];stats[3*k+j]=hi[j];}
    stats[48]=err;return 0;
}
}
