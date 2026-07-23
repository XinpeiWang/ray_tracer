#pragma once

// GGX Microfacet (Trowbridge-Reitz) -- mirrors pbrt-v4 PBRT_CPU_GPU
#ifndef CPU_GPU
#   if defined(__CUDACC__)
#       define CPU_GPU __host__ __device__ __forceinline__
#   else
#       define CPU_GPU inline
#   endif
#endif
#if !defined(__CUDACC__)
#   include <cmath>
#endif

// Shading-frame helpers (local frame: surface normal = +Z)
template<typename T> CPU_GPU T GGX_Cos2Theta(T,T y,T z){return z*z;}
template<typename T> CPU_GPU T GGX_Sin2Theta(T x,T y,T z){T c=z*z;return c<T(1)?T(1)-c:T(0);}
template<typename T> CPU_GPU T GGX_Tan2Theta(T x,T y,T z){T c2=GGX_Cos2Theta(x,y,z);return c2>T(0)?GGX_Sin2Theta(x,y,z)/c2:T(0);}
template<typename T> CPU_GPU T GGX_SinTheta(T x,T y,T z){
#if defined(__CUDACC__)
  return sqrtf(GGX_Sin2Theta(x,y,z));
#else
  return std::sqrt(GGX_Sin2Theta(x,y,z));
#endif
}
template<typename T> CPU_GPU T GGX_CosPhi(T x,T y,T z){T s=GGX_SinTheta(x,y,z);return s==T(0)?T(1):x/s;}
template<typename T> CPU_GPU T GGX_SinPhi(T x,T y,T z){T s=GGX_SinTheta(x,y,z);return s==T(0)?T(0):y/s;}
template<typename T> CPU_GPU T GGX_AbsCosTheta(T x,T y,T z){
#if defined(__CUDACC__)
  return fabsf(z);
#else
  return std::fabs(z);
#endif
}

// TrowbridgeReitz NDF
template<typename T>
struct TrowbridgeReitz {
  T alpha_x, alpha_y;
  CPU_GPU TrowbridgeReitz(T ax,T ay):alpha_x(ax),alpha_y(ay){
    if(!EffectivelySmooth()){
#if defined(__CUDACC__)
      alpha_x=fmaxf(alpha_x,T(1e-4f));alpha_y=fmaxf(alpha_y,T(1e-4f));
#else
      alpha_x=std::max(alpha_x,T(1e-4));alpha_y=std::max(alpha_y,T(1e-4));
#endif
    }
  }
  CPU_GPU bool EffectivelySmooth()const{
#if defined(__CUDACC__)
    return fmaxf(alpha_x,alpha_y)<T(1e-3f);
#else
    return std::max(alpha_x,alpha_y)<T(1e-3);
#endif
  }
  CPU_GPU T D(T wx,T wy,T wz)const{
    T t2=GGX_Tan2Theta(wx,wy,wz);
#if defined(__CUDACC__)
    if(isinf(t2))return T(0);
#else
    if(std::isinf(t2))return T(0);
#endif
    T c4=GGX_Cos2Theta(wx,wy,wz)*GGX_Cos2Theta(wx,wy,wz);
    if(c4<T(1e-16))return T(0);
    T cp=GGX_CosPhi(wx,wy,wz),sp=GGX_SinPhi(wx,wy,wz);
    T e=t2*((cp*cp)/(alpha_x*alpha_x)+(sp*sp)/(alpha_y*alpha_y));
    const T Pi=T(3.14159265358979323846);
    return T(1)/(Pi*alpha_x*alpha_y*c4*(T(1)+e)*(T(1)+e));
  }
  CPU_GPU T Lambda(T wx,T wy,T wz)const{
    T t2=GGX_Tan2Theta(wx,wy,wz);
#if defined(__CUDACC__)
    if(isinf(t2))return T(0);
#else
    if(std::isinf(t2))return T(0);
#endif
    T cp=GGX_CosPhi(wx,wy,wz),sp=GGX_SinPhi(wx,wy,wz);
    T a2=(cp*alpha_x)*(cp*alpha_x)+(sp*alpha_y)*(sp*alpha_y);
#if defined(__CUDACC__)
    return (sqrtf(T(1)+a2*t2)-T(1))/T(2);
#else
    return (std::sqrt(T(1)+a2*t2)-T(1))/T(2);
#endif
  }
  CPU_GPU T G1(T wx,T wy,T wz)const{return T(1)/(T(1)+Lambda(wx,wy,wz));}
  CPU_GPU T G(T wox,T woy,T woz,T wix,T wiy,T wiz)const{return T(1)/(T(1)+Lambda(wox,woy,woz)+Lambda(wix,wiy,wiz));}

  // Visible-normal weighted D: D(w,wm) = G1(w)/|cosTheta_w| * D(wm) * |dot(w,wm)|
  // (pbrt-v4: Float D(Vector3f w, Vector3f wm))
  CPU_GPU T D_visible(T wx,T wy,T wz, T wmx,T wmy,T wmz)const{
    T dot_w_wm = wx*wmx+wy*wmy+wz*wmz;
#if defined(__CUDACC__)
    return G1(wx,wy,wz)/GGX_AbsCosTheta(wx,wy,wz)*D(wmx,wmy,wmz)*fabsf(dot_w_wm);
#else
    return G1(wx,wy,wz)/GGX_AbsCosTheta(wx,wy,wz)*D(wmx,wmy,wmz)*std::fabs(dot_w_wm);
#endif
  }

  // PDF for visible-normal sampling: PDF(w,wm) = D_visible(w,wm)
  // (pbrt-v4: Float PDF(Vector3f w, Vector3f wm))
  CPU_GPU T PDF(T wx,T wy,T wz, T wmx,T wmy,T wmz)const{
    return D_visible(wx,wy,wz, wmx,wmy,wmz);
  }

  // Sample visible microfacet normal (VNDF) given outgoing direction w and
  // two uniform random numbers u1,u2 in [0,1).
  // Returns sampled half-vector (wmx,wmy,wmz) in local shading frame.
  // (pbrt-v4: Vector3f Sample_wm(Vector3f w, Point2f u))
  CPU_GPU void Sample_wm(T wx,T wy,T wz, T u1,T u2,
                         T &wmx,T &wmy,T &wmz)const{
    // Transform w to hemispherical configuration (stretch by alpha)
    T whx=alpha_x*wx, why=alpha_y*wy, whz=wz;
    T wh_len=whx*whx+why*why+whz*whz;
#if defined(__CUDACC__)
    T inv=T(1)/sqrtf(wh_len);
#else
    T inv=T(1)/std::sqrt(wh_len);
#endif
    whx*=inv; why*=inv; whz*=inv;
    if(whz<T(0)){whx=-whx;why=-why;whz=-whz;}

    // Build orthonormal basis (T1,T2,wh)
    T T1x,T1y,T1z, T2x,T2y,T2z;
    if(whz<T(0.99999)){
      // T1 = normalize(cross((0,0,1), wh))
      T len=whx*whx+why*why;
#if defined(__CUDACC__)
      T ilen=T(1)/sqrtf(len);
#else
      T ilen=T(1)/std::sqrt(len);
#endif
      T1x=-why*ilen; T1y=whx*ilen; T1z=T(0);
    } else {
      T1x=T(1); T1y=T(0); T1z=T(0);
    }
    // T2 = cross(wh, T1)
    T2x=why*T1z-whz*T1y;
    T2y=whz*T1x-whx*T1z;
    T2z=whx*T1y-why*T1x;

    // Sample uniform disk polar -> (px,py)
    T r,theta;
#if defined(__CUDACC__)
    r=sqrtf(u1); theta=T(2)*T(3.14159265358979323846)*u2;
    T px=r*cosf(theta), py=r*sinf(theta);
#else
    r=std::sqrt(u1); theta=T(2)*T(3.14159265358979323846)*u2;
    T px=r*std::cos(theta), py=r*std::sin(theta);
#endif

    // Warp for visible normal sampling
#if defined(__CUDACC__)
    T h=sqrtf(T(1)-px*px);
#else
    T h=std::sqrt(T(1)-px*px);
#endif
    py=((T(1)+whz)/T(2))*py+(T(1)-((T(1)+whz)/T(2)))*h;

    // Reproject to hemisphere
    T pz_sq=T(1)-px*px-py*py;
    T pz=pz_sq>T(0)?
#if defined(__CUDACC__)
      sqrtf(pz_sq)
#else
      std::sqrt(pz_sq)
#endif
      :T(0);

    // nh in local basis, then transform back to ellipsoid config
    T nhx=px*T1x+py*T2x+pz*whx;
    T nhy=px*T1y+py*T2y+pz*why;
    T nhz=px*T1z+py*T2z+pz*whz;

    // Stretch back and normalize
    wmx=alpha_x*nhx; wmy=alpha_y*nhy;
    wmz=nhz>T(1e-6)?nhz:T(1e-6);
    T wm_len=wmx*wmx+wmy*wmy+wmz*wmz;
#if defined(__CUDACC__)
    T winv=T(1)/sqrtf(wm_len);
#else
    T winv=T(1)/std::sqrt(wm_len);
#endif
    wmx*=winv; wmy*=winv; wmz*=winv;
  }

  // Regularize (pbrt-v4: void Regularize()) -- bump low alphas for subsurface use
  CPU_GPU void Regularize(){
    if(alpha_x<T(0.3))alpha_x=alpha_x<T(0.1)?T(0.1):T(0.3);
    if(alpha_y<T(0.3))alpha_y=alpha_y<T(0.1)?T(0.1):T(0.3);
  }

  CPU_GPU static T RoughnessToAlpha(T r){
#if defined(__CUDACC__)
    return sqrtf(r);
#else
    return std::sqrt(r);
#endif
  }
};

// GGX conductor BRDF scalar value (multiply by albedo/color outside)
// f = D(wm)*G(wo,wi)/(4*|cosO|*|cosI|)  Fresnel=1 (perfect conductor approx)
template<typename T>
CPU_GPU T GGX_conductor_brdf(T wox,T woy,T woz,T wix,T wiy,T wiz,T ax,T ay){
  T cosO=GGX_AbsCosTheta(wox,woy,woz),cosI=GGX_AbsCosTheta(wix,wiy,wiz);
  if(cosO==T(0)||cosI==T(0))return T(0);
  T hmx=wox+wix,hmy=woy+wiy,hmz=woz+wiz;
  T hl=hmx*hmx+hmy*hmy+hmz*hmz;
  if(hl==T(0))return T(0);
#if defined(__CUDACC__)
  T inv=T(1)/sqrtf(hl);
#else
  T inv=T(1)/std::sqrt(hl);
#endif
  hmx*=inv;hmy*=inv;hmz*=inv;
  TrowbridgeReitz<T> d(ax,ay);
  return d.D(hmx,hmy,hmz)*d.G(wox,woy,woz,wix,wiy,wiz)/(T(4)*cosO*cosI);
}