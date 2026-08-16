#ifndef PIECEWISE_LINEAR_2D_H
#define PIECEWISE_LINEAR_2D_H
//==============================================================================
// piecewise_linear_2d.h  --  bilinear-interpolated 2D warping sampler
//
// Mirrors pbrt-v4 PiecewiseLinear2D<Dimension> (util/sampling.h A.5).
// Original algorithm from Mitsuba (J. Mueller et al.).
//
// Build  : CDF uses trapezoidal rule over bilinear patches
// Sample : analytic inverse per patch (quadratic formula, no inner bisect)
// Eval   : bilinear interpolation
// Dim>0  : optional extra parameter axes blended via recursive lookup
//
// Dimension==0 (unconditional):
//   PiecewiseLinear2D<0> d(data, nx, ny);
//   PLSample s = d.Sample(u0, u1);
//   float    v = d.Eval(px, py);
//
// Dimension>0 (conditional):
//   float pv0[]={0.f,1.f}; const float* pvs[1]={pv0}; int res[1]={2};
//   PiecewiseLinear2D<1> d(data, nx, ny, res, pvs);
//   PLSample s = d.Sample(u0, u1, p0);
//==============================================================================

#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstdint>

struct PLSample { float px=0.f, py=0.f, pdf=0.f; };

namespace pl2d_detail {
inline float Clamp(float v,float lo,float hi){return v<lo?lo:(v>hi?hi:v);}
inline float SafeSqrt(float x){return std::sqrt(x>0.f?x:0.f);}
inline float FMA(float a,float b,float c){return std::fma(a,b,c);}
template<typename P>
inline uint32_t FindInterval(uint32_t n,P p){
    uint32_t lo=0,hi=n;
    while(lo+1<hi){uint32_t m=(lo+hi)>>1;if(p(m))lo=m;else hi=m;}
    return lo;
}
} // namespace pl2d_detail

template<size_t Dimension=0>
class PiecewiseLinear2D {
    // MSVC forbids zero-size arrays; clamp to 1
    static constexpr size_t AS=(Dimension!=0)?Dimension:1;
public:
    PiecewiseLinear2D()=default;

    // Dimension==0 constructor
    PiecewiseLinear2D(const float* data,int nx,int ny,
                      bool normalize=true,bool build_cdf=true){
        static_assert(Dimension==0,"Use param_res/param_values ctor for Dimension>0");
        BuildImpl(data,nx,ny,nullptr,nullptr,normalize,build_cdf);
    }

    // Dimension>0 constructor
    PiecewiseLinear2D(const float* data,int nx,int ny,
                      const int*          param_res,
                      const float* const* param_values,
                      bool normalize=true,bool build_cdf=true){
        static_assert(Dimension>0,"Use simple ctor for Dimension==0");
        BuildImpl(data,nx,ny,param_res,param_values,normalize,build_cdf);
    }

    // Sample: u0,u1 in [0,1]; params = Dimension extra floats
    template<typename... Ts>
    PLSample Sample(float u0,float u1,Ts... params) const {
        static_assert(sizeof...(Ts)==Dimension,"Sample: wrong number of extra params");
        static constexpr float kOme=1.f-1.19209e-7f;
        u0=pl2d_detail::Clamp(u0,1.f-kOme,kOme);
        u1=pl2d_detail::Clamp(u1,1.f-kOme,kOme);

        float pw[2*AS]={};
        uint32_t soff=0;
        FillPW(soff,pw,params...);

        const uint32_t nx=m_nx,ny=m_ny,sz=nx*ny;

        // sample marginal row
        uint32_t mb=(Dimension!=0)?(soff*ny):0u;
        auto fm=[&](uint32_t i){return LookImpl<Dimension>(m_mcdf.data(),mb+i,ny,pw);};
        uint32_t row=pl2d_detail::FindInterval(ny,[&](uint32_t i){return fm(i)<u1;});
        u1-=fm(row);

        uint32_t off=row*nx+((Dimension!=0)?soff*sz:0u);
        float r0=LookImpl<Dimension>(m_ccdf.data(),off+nx-1,sz,pw);
        float r1=LookImpl<Dimension>(m_ccdf.data(),off+(nx*2-1),sz,pw);

        bool ky=std::abs(r0-r1)<1e-4f*(r0+r1);
        u1=ky?(2.f*u1):(r0-pl2d_detail::SafeSqrt(r0*r0-2.f*u1*(r0-r1)));
        u1/=ky?(r0+r1):(r0-r1);

        // sample conditional column
        u0*=(1.f-u1)*r0+u1*r1;
        auto fc=[&](uint32_t i){
            float v0=LookImpl<Dimension>(m_ccdf.data(),off+i,sz,pw);
            float v1=LookImpl<Dimension>(m_ccdf.data()+nx,off+i,sz,pw);
            return (1.f-u1)*v0+u1*v1;
        };
        uint32_t col=pl2d_detail::FindInterval(nx,[&](uint32_t i){return fc(i)<u0;});
        u0-=fc(col); off+=col;

        float v00=LookImpl<Dimension>(m_data.data(),     off,sz,pw);
        float v10=LookImpl<Dimension>(m_data.data()+1,   off,sz,pw);
        float v01=LookImpl<Dimension>(m_data.data()+nx,  off,sz,pw);
        float v11=LookImpl<Dimension>(m_data.data()+nx+1,off,sz,pw);

        float c0=pl2d_detail::FMA(1.f-u1,v00,u1*v01);
        float c1=pl2d_detail::FMA(1.f-u1,v10,u1*v11);

        bool kx=std::abs(c0-c1)<1e-4f*(c0+c1);
        u0=kx?(2.f*u0):(c0-pl2d_detail::SafeSqrt(c0*c0-2.f*u0*(c0-c1)));
        u0/=kx?(c0+c1):(c0-c1);

        PLSample s;
        s.px=(col+u0)*m_px; s.py=(row+u1)*m_py;
        s.pdf=((1.f-u0)*c0+u0*c1)*m_invx*m_invy;
        return s;
    }

    // Eval: bilinear interpolation of normalized density at (px,py)
    template<typename... Ts>
    float Eval(float px,float py,Ts... params) const {
        static_assert(sizeof...(Ts)==Dimension,"Eval: wrong number of extra params");
        float pw[2*AS]={};
        uint32_t soff=0;
        FillPW(soff,pw,params...);

        float x=px*m_invx, y=py*m_invy;
        int ix=std::min((int)x,(int)m_nx-2);
        int iy=std::min((int)y,(int)m_ny-2);
        float wx1=x-(float)ix, wx0=1.f-wx1;
        float wy1=y-(float)iy, wy0=1.f-wy1;

        uint32_t idx=(uint32_t)(ix+iy*(int)m_nx);
        uint32_t sz=m_nx*m_ny;
        if(Dimension!=0) idx+=soff*sz;

        float v00=LookImpl<Dimension>(m_data.data(),     idx,sz,pw);
        float v10=LookImpl<Dimension>(m_data.data()+1,   idx,sz,pw);
        float v01=LookImpl<Dimension>(m_data.data()+m_nx,idx,sz,pw);
        float v11=LookImpl<Dimension>(m_data.data()+m_nx+1,idx,sz,pw);

        return pl2d_detail::FMA(wy0,pl2d_detail::FMA(wx0,v00,wx1*v10),
                                wy1*pl2d_detail::FMA(wx0,v01,wx1*v11))
               *m_invx*m_invy;
    }

    // Invert: given a point (px,py) produced by Sample(), recover the original
    // uniform sample u and its pdf.  Mirrors pbrt-v4 PLSample Invert().
    template<typename... Ts>
    PLSample Invert(float px,float py,Ts... params) const {
        static_assert(sizeof...(Ts)==Dimension,"Invert: wrong number of extra params");
        float pw[2*AS]={};
        uint32_t soff=0;
        FillPW(soff,pw,params...);

        // convert to grid coords
        float sx=px*m_invx, sy=py*m_invy;
        int ix=std::min((int)sx,(int)m_nx-2);
        int iy=std::min((int)sy,(int)m_ny-2);
        sx-=(float)ix; sy-=(float)iy;          // fractional within patch

        uint32_t sz=m_nx*m_ny;
        uint32_t off=(uint32_t)(ix+iy*(int)m_nx);
        if(Dimension!=0) off+=soff*sz;

        // corner values
        float v00=LookImpl<Dimension>(m_data.data(),     off,   sz,pw);
        float v10=LookImpl<Dimension>(m_data.data()+1,   off,   sz,pw);
        float v01=LookImpl<Dimension>(m_data.data()+m_nx,off,   sz,pw);
        float v11=LookImpl<Dimension>(m_data.data()+m_nx+1,off, sz,pw);

        float w0y=1.f-sy, w1y=sy;
        float c0=pl2d_detail::FMA(w0y,v00,w1y*v01);
        float c1=pl2d_detail::FMA(w0y,v10,w1y*v11);
        float pdf=pl2d_detail::FMA(1.f-sx,c0,sx*c1);

        // invert X: undo the quadratic solve
        float ux=sx*(c0+0.5f*sx*(c1-c0));

        // add conditional CDF base
        float cdf_v0=LookImpl<Dimension>(m_ccdf.data(),     off,sz,pw);
        float cdf_v1=LookImpl<Dimension>(m_ccdf.data()+m_nx,off,sz,pw);
        ux+=(1.f-sy)*cdf_v0+sy*cdf_v1;

        // normalize by row total (r0, r1)
        uint32_t roff=(uint32_t)(iy*(int)m_nx);
        if(Dimension!=0) roff+=soff*sz;
        float r0=LookImpl<Dimension>(m_ccdf.data(),roff+m_nx-1,  sz,pw);
        float r1=LookImpl<Dimension>(m_ccdf.data(),roff+(m_nx*2-1),sz,pw);
        ux/=(1.f-sy)*r0+sy*r1;

        // invert Y: undo the marginal quadratic solve
        float uy=sy*(r0+0.5f*sy*(r1-r0));

        // add marginal CDF base
        uint32_t moff=(uint32_t)iy;
        if(Dimension!=0) moff+=soff*m_ny;
        uy+=LookImpl<Dimension>(m_mcdf.data(),moff,m_ny,pw);

        PLSample s;
        s.px=pl2d_detail::Clamp(ux,0.f,1.f);
        s.py=pl2d_detail::Clamp(uy,0.f,1.f);
        s.pdf=pdf*m_invx*m_invy;
        return s;
    }

    bool IsEmpty()const{return m_data.empty();}
    int  XSize()const{return(int)m_nx;}
    int  YSize()const{return(int)m_ny;}

    // ---- GPU upload accessors -------------------------------------------
    // Read-only access to the already-built flat tables/parameter axes, for
    // flattening one PiecewiseLinear2D instance into device buffers (see
    // gpu/optix/pbrt_gpu_builder.h's getOrBuildMeasuredTable()). No CDF
    // construction happens on the GPU - these just expose what BuildImpl()
    // already computed CPU-side. Pure const getters, no behavior change.
    const uint32_t* ParamRes()    const { return m_ps; }
    const uint32_t* ParamStride() const { return m_pst; }
    const std::vector<float>& ParamValues(size_t axis) const { return m_pv[axis]; }
    const std::vector<float>& Data() const { return m_data; }
    const std::vector<float>& Mcdf() const { return m_mcdf; }
    const std::vector<float>& Ccdf() const { return m_ccdf; }

private:
    void BuildImpl(const float* data,int xSize,int ySize,
                   const int* pr,const float* const* pv,
                   bool normalize,bool build_cdf)
    {
        assert(xSize>=2&&ySize>=2);
        assert(!build_cdf||normalize);
        m_nx=(uint32_t)xSize; m_ny=(uint32_t)ySize;
        m_px=1.f/(xSize-1);  m_py=1.f/(ySize-1);
        m_invx=(float)(xSize-1); m_invy=(float)(ySize-1);

        uint32_t slices=1;
        for(int i=(int)Dimension-1;i>=0;--i){
            assert(pr[i]>=1);
            m_ps[i]=(uint32_t)pr[i];
            m_pst[i]=(pr[i]>1)?slices:0;
            m_pv[i].assign(pv[i],pv[i]+pr[i]);
            slices*=m_ps[i];
        }

        const uint32_t nv=m_nx*m_ny;
        m_data.resize((size_t)slices*nv);

        if(build_cdf){
            m_mcdf.resize((size_t)slices*m_ny);
            m_ccdf.resize((size_t)slices*nv);
            float* mc=m_mcdf.data(),*cc=m_ccdf.data(),*dout=m_data.data();

            for(uint32_t sl=0;sl<slices;++sl){
                // conditional CDF per row (trapezoidal rule)
                for(uint32_t y=0;y<m_ny;++y){
                    double sum=0.0;
                    size_t i=(size_t)y*m_nx;
                    cc[i]=0.f;
                    for(uint32_t x=0;x<m_nx-1;++x,++i){
                        sum+=0.5*((double)data[i]+(double)data[i+1]);
                        cc[i+1]=(float)sum;
                    }
                }
                // marginal CDF (trapezoidal over row totals)
                mc[0]=0.f;
                {
                    double sum=0.0;
                    for(uint32_t y=0;y<m_ny-1;++y){
                        sum+=0.5*((double)cc[(y+1)*m_nx-1]+(double)cc[(y+2)*m_nx-1]);
                        mc[y+1]=(float)sum;
                    }
                }
                float norm=mc[m_ny-1];
                float inv=(norm>0.f)?1.f/norm:0.f;
                for(size_t i=0;i<nv;++i)    cc[i]  *=inv;
                for(uint32_t i=0;i<m_ny;++i) mc[i]  *=inv;
                for(size_t i=0;i<nv;++i)    dout[i] =data[i]*inv;
                mc+=m_ny; cc+=nv; dout+=nv; data+=nv;
            }
        } else {
            float* dout=m_data.data();
            for(uint32_t sl=0;sl<slices;++sl){
                float inv=1.f;
                if(normalize){
                    double sum=0.0;
                    for(uint32_t y=0;y<m_ny-1;++y){
                        size_t i=(size_t)y*m_nx;
                        for(uint32_t x=0;x<m_nx-1;++x,++i)
                            sum+=0.25*(data[i]+data[i+1]+data[i+m_nx]+data[i+m_nx+1]);
                    }
                    inv=(float)(1.0/sum);
                } else {
                    inv=1.f/(m_invx*m_invy);
                }
                for(size_t k=0;k<nv;++k) dout[k]=data[k]*inv;
                data+=nv; dout+=nv;
            }
        }
    }

    // base case: no params
    void FillPW(uint32_t&,float*)const{}

    // recursive: one param per call, dim = Dimension-1-remaining
    template<typename T,typename... Ts>
    void FillPW(uint32_t& soff,float* pw,T first,Ts... rest)const{
        constexpr size_t dim=Dimension-1-sizeof...(Ts);
        float p=(float)first;
        if(m_ps[dim]==1){pw[2*dim]=1.f;pw[2*dim+1]=0.f;}
        else{
            uint32_t idx=pl2d_detail::FindInterval(m_ps[dim],
                [&](uint32_t i){return m_pv[dim][i]<=p;});
            if(idx+1>=m_ps[dim]) idx=m_ps[dim]-2;  // clamp: idx+1 must be valid
            float p0=m_pv[dim][idx],p1=m_pv[dim][idx+1];
            pw[2*dim+1]=pl2d_detail::Clamp((p-p0)/(p1-p0),0.f,1.f);
            pw[2*dim]=1.f-pw[2*dim+1];
            soff+=m_pst[dim]*idx;
        }
        FillPW(soff,pw,rest...);
    }

    // recursive slice blending (mirrors pbrt-v4 lookup<Dim>)
    template<size_t Dim>
    float LookImpl(const float* data,uint32_t i0,uint32_t sz,const float* pw)const{
        if constexpr(Dim==0){ return data[i0]; }
        else{
            uint32_t i1=i0+m_pst[Dim-1]*sz;
            float v0=LookImpl<Dim-1>(data,i0,sz,pw);
            float v1=LookImpl<Dim-1>(data,i1,sz,pw);
            return pl2d_detail::FMA(v0,pw[2*Dim-2],v1*pw[2*Dim-1]);
        }
    }

    uint32_t m_nx=0,m_ny=0;
    float    m_px=0.f,m_py=0.f,m_invx=0.f,m_invy=0.f;
    uint32_t           m_ps [AS]={};
    uint32_t           m_pst[AS]={};
    std::vector<float> m_pv [AS];
    std::vector<float> m_data,m_mcdf,m_ccdf;
};

#endif // PIECEWISE_LINEAR_2D_H
