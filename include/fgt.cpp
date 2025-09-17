#ifndef FGT_CPP
#define FGT_CPP

#include <chrono>
#include <random>
#include <cassert>

template <class Real, unsigned int VecLen> inline void Transpose(sctl::Vec<Real,VecLen> (&v)[VecLen]) noexcept { // generic
  Real v_[VecLen][VecLen];
  for (unsigned int i = 0; i < VecLen; i++) {
    for (unsigned int j = 0; j < VecLen; j++) {
      v_[i][j] = v[i].get().v[j];
    }
  }
  for (unsigned int i = 0; i < VecLen; i++) {
    for (unsigned int j = 0; j < VecLen; j++) {
      v[i].get().v[j] = v_[j][i];
    }
  }
}

#ifdef __SSE4_2__
template <> inline void Transpose<double,2>(sctl::Vec<double,2> (&v)[2]) noexcept {
  auto& r0 = v[0].get().v;
  auto& r1 = v[1].get().v;

  __m128d t0 = _mm_unpacklo_pd(r0, r1);
  __m128d t1 = _mm_unpackhi_pd(r0, r1);

  r0 = t0;
  r1 = t1;
}

template <> inline void Transpose<float,4>(sctl::Vec<float,4> (&v)[4]) noexcept {
  auto& r0 = v[0].get().v;
  auto& r1 = v[1].get().v;
  auto& r2 = v[2].get().v;
  auto& r3 = v[3].get().v;

  __m128 a0 = _mm_unpacklo_ps(r0, r1);
  __m128 a1 = _mm_unpackhi_ps(r0, r1);
  __m128 b0 = _mm_unpacklo_ps(r2, r3);
  __m128 b1 = _mm_unpackhi_ps(r2, r3);

  r0 = _mm_movelh_ps(a0, b0);
  r1 = _mm_movehl_ps(b0, a0);
  r2 = _mm_movelh_ps(a1, b1);
  r3 = _mm_movehl_ps(b1, a1);
}
#endif

#ifdef __AVX__
template <> inline void Transpose<double,4>(sctl::Vec<double,4> (&v)[4]) noexcept {
  auto& r0 = v[0].get().v;
  auto& r1 = v[1].get().v;
  auto& r2 = v[2].get().v;
  auto& r3 = v[3].get().v;

  __m256d t0 = _mm256_shuffle_pd(r0, r1, 0x0);
  __m256d t2 = _mm256_shuffle_pd(r0, r1, 0xF);
  __m256d t1 = _mm256_shuffle_pd(r2, r3, 0x0);
  __m256d t3 = _mm256_shuffle_pd(r2, r3, 0xF);

  r0 = _mm256_permute2f128_pd(t0, t1, 0x20);
  r1 = _mm256_permute2f128_pd(t2, t3, 0x20);
  r2 = _mm256_permute2f128_pd(t0, t1, 0x31);
  r3 = _mm256_permute2f128_pd(t2, t3, 0x31);
}

template <> inline void Transpose<float,8>(sctl::Vec<float,8> (&v)[8]) noexcept {
  auto& r0 = v[0].get().v;
  auto& r1 = v[1].get().v;
  auto& r2 = v[2].get().v;
  auto& r3 = v[3].get().v;
  auto& r4 = v[4].get().v;
  auto& r5 = v[5].get().v;
  auto& r6 = v[6].get().v;
  auto& r7 = v[7].get().v;

  const __m256 a0 = _mm256_unpacklo_ps(r0, r1);
  const __m256 a1 = _mm256_unpackhi_ps(r0, r1);
  const __m256 a2 = _mm256_unpacklo_ps(r2, r3);
  const __m256 a3 = _mm256_unpackhi_ps(r2, r3);
  const __m256 a4 = _mm256_unpacklo_ps(r4, r5);
  const __m256 a5 = _mm256_unpackhi_ps(r4, r5);
  const __m256 a6 = _mm256_unpacklo_ps(r6, r7);
  const __m256 a7 = _mm256_unpackhi_ps(r6, r7);

  const __m256 t0 = _mm256_shuffle_ps(a0, a2, _MM_SHUFFLE(1,0,1,0));
  const __m256 t1 = _mm256_shuffle_ps(a0, a2, _MM_SHUFFLE(3,2,3,2));
  const __m256 t2 = _mm256_shuffle_ps(a1, a3, _MM_SHUFFLE(1,0,1,0));
  const __m256 t3 = _mm256_shuffle_ps(a1, a3, _MM_SHUFFLE(3,2,3,2));
  const __m256 t4 = _mm256_shuffle_ps(a4, a6, _MM_SHUFFLE(1,0,1,0));
  const __m256 t5 = _mm256_shuffle_ps(a4, a6, _MM_SHUFFLE(3,2,3,2));
  const __m256 t6 = _mm256_shuffle_ps(a5, a7, _MM_SHUFFLE(1,0,1,0));
  const __m256 t7 = _mm256_shuffle_ps(a5, a7, _MM_SHUFFLE(3,2,3,2));

  r0 = _mm256_permute2f128_ps(t0, t4, 0x20);
  r1 = _mm256_permute2f128_ps(t1, t5, 0x20);
  r2 = _mm256_permute2f128_ps(t2, t6, 0x20);
  r3 = _mm256_permute2f128_ps(t3, t7, 0x20);
  r4 = _mm256_permute2f128_ps(t0, t4, 0x31);
  r5 = _mm256_permute2f128_ps(t1, t5, 0x31);
  r6 = _mm256_permute2f128_ps(t2, t6, 0x31);
  r7 = _mm256_permute2f128_ps(t3, t7, 0x31);
}
#endif

#ifdef __AVX512F__
template <> inline void Transpose<double,8>(sctl::Vec<double,8> (&v)[8]) noexcept {
  auto& r0 = v[0].get().v;
  auto& r1 = v[1].get().v;
  auto& r2 = v[2].get().v;
  auto& r3 = v[3].get().v;
  auto& r4 = v[4].get().v;
  auto& r5 = v[5].get().v;
  auto& r6 = v[6].get().v;
  auto& r7 = v[7].get().v;

  // 1) Unpack within 64-bit lanes (build 2x2 blocks)
  __m512d t0 = _mm512_unpacklo_pd(r0, r1);
  __m512d t1 = _mm512_unpackhi_pd(r0, r1);
  __m512d t2 = _mm512_unpacklo_pd(r2, r3);
  __m512d t3 = _mm512_unpackhi_pd(r2, r3);
  __m512d t4 = _mm512_unpacklo_pd(r4, r5);
  __m512d t5 = _mm512_unpackhi_pd(r4, r5);
  __m512d t6 = _mm512_unpacklo_pd(r6, r7);
  __m512d t7 = _mm512_unpackhi_pd(r6, r7);

  // 2) Shuffle 128-bit lanes to form 4x4 blocks
  // imm8=0x88 selects low 128-bit lanes; imm8=0xDD selects high 128-bit lanes
  __m512d s0 = _mm512_shuffle_f64x2(t0, t2, 0x88);
  __m512d s1 = _mm512_shuffle_f64x2(t1, t3, 0x88);
  __m512d s2 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
  __m512d s3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
  __m512d s4 = _mm512_shuffle_f64x2(t4, t6, 0x88);
  __m512d s5 = _mm512_shuffle_f64x2(t5, t7, 0x88);
  __m512d s6 = _mm512_shuffle_f64x2(t4, t6, 0xDD);
  __m512d s7 = _mm512_shuffle_f64x2(t5, t7, 0xDD);

  // 3) Final 8x8 assembly by shuffling 128-bit lanes between the 4x4 blocks
  (r0) = _mm512_shuffle_f64x2(s0, s4, 0x88);
  (r1) = _mm512_shuffle_f64x2(s1, s5, 0x88);
  (r2) = _mm512_shuffle_f64x2(s2, s6, 0x88);
  (r3) = _mm512_shuffle_f64x2(s3, s7, 0x88);
  (r4) = _mm512_shuffle_f64x2(s0, s4, 0xDD);
  (r5) = _mm512_shuffle_f64x2(s1, s5, 0xDD);
  (r6) = _mm512_shuffle_f64x2(s2, s6, 0xDD);
  (r7) = _mm512_shuffle_f64x2(s3, s7, 0xDD);
}

template <> inline void Transpose<float,16>(sctl::Vec<float,16> (&v)[16]) noexcept {
  auto& r0 = v[0].get().v;
  auto& r1 = v[1].get().v;
  auto& r2 = v[2].get().v;
  auto& r3 = v[3].get().v;
  auto& r4 = v[4].get().v;
  auto& r5 = v[5].get().v;
  auto& r6 = v[6].get().v;
  auto& r7 = v[7].get().v;
  auto& r8 = v[8].get().v;
  auto& r9 = v[9].get().v;
  auto& r10= v[10].get().v;
  auto& r11= v[11].get().v;
  auto& r12= v[12].get().v;
  auto& r13= v[13].get().v;
  auto& r14= v[14].get().v;
  auto& r15= v[15].get().v;

  // Stage 1: interleave 32-bit elements (within 128-bit lanes)
  __m512 t0  = _mm512_unpacklo_ps(r0 , r1 );  __m512 t1  = _mm512_unpackhi_ps(r0 , r1 );
  __m512 t2  = _mm512_unpacklo_ps(r2 , r3 );  __m512 t3  = _mm512_unpackhi_ps(r2 , r3 );
  __m512 t4  = _mm512_unpacklo_ps(r4 , r5 );  __m512 t5  = _mm512_unpackhi_ps(r4 , r5 );
  __m512 t6  = _mm512_unpacklo_ps(r6 , r7 );  __m512 t7  = _mm512_unpackhi_ps(r6 , r7 );
  __m512 t8  = _mm512_unpacklo_ps(r8 , r9 );  __m512 t9  = _mm512_unpackhi_ps(r8 , r9 );
  __m512 t10 = _mm512_unpacklo_ps(r10, r11);  __m512 t11 = _mm512_unpackhi_ps(r10, r11);
  __m512 t12 = _mm512_unpacklo_ps(r12, r13);  __m512 t13 = _mm512_unpackhi_ps(r12, r13);
  __m512 t14 = _mm512_unpacklo_ps(r14, r15);  __m512 t15 = _mm512_unpackhi_ps(r14, r15);

  // Stage 2: gather 2-wide groups into 4-wide (still within 128-bit lanes)
  // imm 0x44 = [a0,a1, b0,b1]; imm 0xEE = [a2,a3, b2,b3]
  __m512 s0  = _mm512_shuffle_ps(t0 , t2 , 0x44);  __m512 s1  = _mm512_shuffle_ps(t0 , t2 , 0xEE);
  __m512 s2  = _mm512_shuffle_ps(t1 , t3 , 0x44);  __m512 s3  = _mm512_shuffle_ps(t1 , t3 , 0xEE);
  __m512 s4  = _mm512_shuffle_ps(t4 , t6 , 0x44);  __m512 s5  = _mm512_shuffle_ps(t4 , t6 , 0xEE);
  __m512 s6  = _mm512_shuffle_ps(t5 , t7 , 0x44);  __m512 s7  = _mm512_shuffle_ps(t5 , t7 , 0xEE);
  __m512 s8  = _mm512_shuffle_ps(t8 , t10, 0x44);  __m512 s9  = _mm512_shuffle_ps(t8 , t10, 0xEE);
  __m512 s10 = _mm512_shuffle_ps(t9 , t11, 0x44);  __m512 s11 = _mm512_shuffle_ps(t9 , t11, 0xEE);
  __m512 s12 = _mm512_shuffle_ps(t12, t14, 0x44);  __m512 s13 = _mm512_shuffle_ps(t12, t14, 0xEE);
  __m512 s14 = _mm512_shuffle_ps(t13, t15, 0x44);  __m512 s15 = _mm512_shuffle_ps(t13, t15, 0xEE);

  // Stage 3: move 128-bit lanes across the 512-bit vectors
  // imm 0x88 = [a0,a1, b0,b1] (low halves), 0xDD = [a2,a3, b2,b3] (high halves)
  __m512 u0  = _mm512_shuffle_f32x4(s0 , s4 , 0x88);
  __m512 u1  = _mm512_shuffle_f32x4(s1 , s5 , 0x88);
  __m512 u2  = _mm512_shuffle_f32x4(s2 , s6 , 0x88);
  __m512 u3  = _mm512_shuffle_f32x4(s3 , s7 , 0x88);
  __m512 u4  = _mm512_shuffle_f32x4(s0 , s4 , 0xDD);
  __m512 u5  = _mm512_shuffle_f32x4(s1 , s5 , 0xDD);
  __m512 u6  = _mm512_shuffle_f32x4(s2 , s6 , 0xDD);
  __m512 u7  = _mm512_shuffle_f32x4(s3 , s7 , 0xDD);

  __m512 u8  = _mm512_shuffle_f32x4(s8 , s12, 0x88);
  __m512 u9  = _mm512_shuffle_f32x4(s9 , s13, 0x88);
  __m512 u10 = _mm512_shuffle_f32x4(s10, s14, 0x88);
  __m512 u11 = _mm512_shuffle_f32x4(s11, s15, 0x88);
  __m512 u12 = _mm512_shuffle_f32x4(s8 , s12, 0xDD);
  __m512 u13 = _mm512_shuffle_f32x4(s9 , s13, 0xDD);
  __m512 u14 = _mm512_shuffle_f32x4(s10, s14, 0xDD);
  __m512 u15 = _mm512_shuffle_f32x4(s11, s15, 0xDD);

  // Final combine: produce the 16 transposed rows
  r0  = _mm512_shuffle_f32x4(u0 , u8 , 0x88);
  r1  = _mm512_shuffle_f32x4(u1 , u9 , 0x88);
  r2  = _mm512_shuffle_f32x4(u2 , u10, 0x88);
  r3  = _mm512_shuffle_f32x4(u3 , u11, 0x88);
  r4  = _mm512_shuffle_f32x4(u4 , u12, 0x88);
  r5  = _mm512_shuffle_f32x4(u5 , u13, 0x88);
  r6  = _mm512_shuffle_f32x4(u6 , u14, 0x88);
  r7  = _mm512_shuffle_f32x4(u7 , u15, 0x88);

  r8  = _mm512_shuffle_f32x4(u0 , u8 , 0xDD);
  r9  = _mm512_shuffle_f32x4(u1 , u9 , 0xDD);
  r10 = _mm512_shuffle_f32x4(u2 , u10, 0xDD);
  r11 = _mm512_shuffle_f32x4(u3 , u11, 0xDD);
  r12 = _mm512_shuffle_f32x4(u4 , u12, 0xDD);
  r13 = _mm512_shuffle_f32x4(u5 , u13, 0xDD);
  r14 = _mm512_shuffle_f32x4(u6 , u14, 0xDD);
  r15 = _mm512_shuffle_f32x4(u7 , u15, 0xDD);
}
#endif

template <class Real, unsigned int N> void test_transpose() {
  sctl::Vec<Real,N> v[N];
  for (unsigned int i = 0; i < N; i++) {
    for (unsigned int j = 0; j < N; j++) {
      v[i].get().v[j] = i*N + j + 1;
    }
  }
  Transpose<Real,N>(v);
  for (unsigned int i = 0; i < N; i++) {
    for (unsigned int j = 0; j < N; j++) {
      if (v[i].get().v[j] != j*N + i + 1) {
        std::cout << "Transpose<" << typeid(Real).name() << "," << N << "> failed at ("
                  << i << "," << j << "): " << v[i].get().v[j] << " != " << j*N + i + 1 << std::endl;
        std::exit(1);
        return;
      }
    }
  }
}

template <class Real, unsigned int order = 4>
class FGT_Coeffs {
  static_assert(order >= 1 && order <= 15, "order must be in [1,15]");

public:

  static constexpr const std::array<Real, order>& get_tr() {
    static_assert(order >= 2 && order <= 15, "order must be in [2,15]");
    if constexpr (order == 2) return tr2;
    else if constexpr (order == 3) return tr3;
    else if constexpr (order == 4) return tr4;
    else if constexpr (order == 5) return tr5;
    else if constexpr (order == 6) return tr6;
    else if constexpr (order == 7) return tr7;
    else if constexpr (order == 8) return tr8;
    else if constexpr (order == 9) return tr9;
    else if constexpr (order == 10) return tr10;
    else if constexpr (order == 11) return tr11;
    else if constexpr (order == 12) return tr12;
    else if constexpr (order == 13) return tr13;
    else if constexpr (order == 14) return tr14;
    else return tr15;
  }
  static constexpr const std::array<Real, order>& get_ti() {
    static_assert(order >= 2 && order <= 15, "order must be in [2,15]");
    if constexpr (order == 2) return ti2;
    else if constexpr (order == 3) return ti3;
    else if constexpr (order == 4) return ti4;
    else if constexpr (order == 5) return ti5;
    else if constexpr (order == 6) return ti6;
    else if constexpr (order == 7) return ti7;
    else if constexpr (order == 8) return ti8;
    else if constexpr (order == 9) return ti9;
    else if constexpr (order == 10) return ti10;
    else if constexpr (order == 11) return ti11;
    else if constexpr (order == 12) return ti12;
    else if constexpr (order == 13) return ti13;
    else if constexpr (order == 14) return ti14;
    else return ti15;
  }
  static constexpr const std::array<Real, order>& get_wr() {
    static_assert(order >= 2 && order <= 15, "order must be in [2,15]");
    if constexpr (order == 2) return wr2;
    else if constexpr (order == 3) return wr3;
    else if constexpr (order == 4) return wr4;
    else if constexpr (order == 5) return wr5;
    else if constexpr (order == 6) return wr6;
    else if constexpr (order == 7) return wr7;
    else if constexpr (order == 8) return wr8;
    else if constexpr (order == 9) return wr9;
    else if constexpr (order == 10) return wr10;
    else if constexpr (order == 11) return wr11;
    else if constexpr (order == 12) return wr12;
    else if constexpr (order == 13) return wr13;
    else if constexpr (order == 14) return wr14;
    else return wr15;
  }
  static constexpr const std::array<Real, order>& get_wi() {
    static_assert(order >= 2 && order <= 15, "order must be in [2,15]");
    if constexpr (order == 2) return wi2;
    else if constexpr (order == 3) return wi3;
    else if constexpr (order == 4) return wi4;
    else if constexpr (order == 5) return wi5;
    else if constexpr (order == 6) return wi6;
    else if constexpr (order == 7) return wi7;
    else if constexpr (order == 8) return wi8;
    else if constexpr (order == 9) return wi9;
    else if constexpr (order == 10) return wi10;
    else if constexpr (order == 11) return wi11;
    else if constexpr (order == 12) return wi12;
    else if constexpr (order == 13) return wi13;
    else if constexpr (order == 14) return wi14;
    else return wi15;
  }


  // The following are actual maximum absolute errors:
  // n = 2     error = 3.335321088237908e-2.
  // n = 3     error = 4.318718985100389e-3.
  // n = 4     error = 5.318838249204205e-4.
  // n = 5     error = 6.365696251409148e-5.
  // n = 6     error = 7.478518927683808e-6.
  // n = 7     error = 8.672276035071036e-7.
  // n = 8     error = 9.96015296905739e-8.
  // n = 9     error = 1.1354696738408165e-8.
  // n = 10    error = 1.2868426324530446e-9.
  // n = 11    error = 1.4514256463371566e-10.
  // n = 12    error = 1.63069557856943e-11.
  // n = 13    error = 1.8260948309034575e-12.
  // n = 14    error = 1.9184653865522705e-13.
  // n = 15    error = 3.186340080674199e-14.

  static constexpr std::array<Real,2> tr2 = {
    (Real) 1.7331182112288734e+00,
    (Real) 1.7331182112288734e+00};
  static constexpr std::array<Real,2> ti2 = {
    (Real) 1.20547925590969e+00,
    (Real)-1.20547925590969e+00};
  static constexpr std::array<Real,2> wr2 = {
    (Real) 4.8332339455881046e-01,
    (Real) 4.8332339455881046e-01};
  static constexpr std::array<Real,2> wi2 = {
    (Real) 9.216605141213353e-01,
    (Real) -9.216605141213353e-01};

  static constexpr std::array<Real,3> tr3 = {
    (Real) 2.0984269549539922e+00,
    (Real) 2.0984269549539922e+00,
    (Real) 2.158903804700515e+00};
  static constexpr std::array<Real,3> ti3 = {
    (Real)-2.0986325746287107e+00,
    (Real) 2.0986325746287107e+00,
    (Real)-1.0320006208641968e-145};
  static constexpr std::array<Real,3> wr3 = {
    (Real)-4.101430763332758e-01,
    (Real)-4.101430763332758e-01,
    (Real) 1.8246048716516519e+00};
  static constexpr std::array<Real,3> wi3 = {
    (Real)-4.9897572754518255e-01,
    (Real) 4.9897572754518255e-01,
    (Real)-1.7299518579916967e-145};

  static constexpr std::array<Real,4> tr4 = {
    (Real) 2.4031501789520147e+00,
    (Real) 2.4031501789520147e+00,
    (Real) 2.4863139794167175e+00,
    (Real) 2.4863139794167175e+00};
  static constexpr std::array<Real,4> ti4 = {
    (Real)-2.836160689145866e+00,
    (Real) 2.836160689145866e+00,
    (Real)-8.973783261287895e-01,
    (Real) 8.973783261287895e-01};
  static constexpr std::array<Real,4> wr4 = {
    (Real)-3.265153953752501e-01,
    (Real)-3.265153953752501e-01,
    (Real) 8.262494534627899e-01,
    (Real) 8.262494534627899e-01};
  static constexpr std::array<Real,4> wi4 = {
    (Real) 1.1193658182522465e-01,
    (Real)-1.1193658182522465e-01,
    (Real)-1.7813106716738372e+00,
    (Real) 1.7813106716738372e+00};

  static constexpr std::array<Real,5> tr5 = {
    (Real) 2.67033102889893e+00,
    (Real) 2.67033102889893e+00,
    (Real) 2.764687968339535e+00,
    (Real) 2.764687968339535e+00,
    (Real) 2.7923185416690517e+00};
  static constexpr std::array<Real,5> ti5 = {
    (Real)-3.4770510222742175e+00,
    (Real) 3.4770510222742175e+00,
    (Real)-1.6422830177863512e+00,
    (Real) 1.6422830177863512e+00,
    (Real)-1.0742534035470357e-143};
  static constexpr std::array<Real,5> wr5 = {
    (Real)-2.5050015281834995e-04,
    (Real)-2.5050015281834995e-04,
    (Real)-1.1020837362280733e+00,
    (Real)-1.1020837362280733e+00,
    (Real) 3.2047321297242974e+00};
  static constexpr std::array<Real,5> wi5 = {
    (Real) 1.6846643425657054e-01,
    (Real)-1.6846643425657054e-01,
    (Real)-1.2242858312963845e+00,
    (Real) 1.2242858312963845e+00,
    (Real)-4.301414339148198e-143};

  static constexpr std::array<Real,6> tr6 = {
    (Real) 2.9112538664082526e+00,
    (Real) 2.9112538664082526e+00,
    (Real) 3.011893759050851e+00,
    (Real) 3.011893759050851e+00,
    (Real) 3.05520644832387e+00,
    (Real) 3.05520644832387e+00};
  static constexpr std::array<Real,6> ti6 = {
    (Real)-4.050712865259695e+00,
    (Real) 4.050712865259695e+00,
    (Real)-2.2916845694400143e+00,
    (Real) 2.2916845694400143e+00,
    (Real)-7.467354852979078e-01,
    (Real) 7.467354852979078e-01};
  static constexpr std::array<Real,6> wr6 = {
    (Real) 7.348997894647497e-02,
    (Real) 7.348997894647497e-02,
    (Real)-1.1101005457846775e+00,
    (Real)-1.1101005457846775e+00,
    (Real) 1.5366068275787388e+00,
    (Real) 1.5366068275787388e+00};
  static constexpr std::array<Real,6> wi6 = {
    (Real) 2.397066894335284e-02,
    (Real)-2.397066894335284e-02,
    (Real) 4.072148062881761e-01,
    (Real)-4.072148062881761e-01,
    (Real)-3.476044990645759e+00,
    (Real) 3.476044990645759e+00};

  static constexpr std::array<Real,7> tr7 = {
    (Real) 3.132461457144918e+00,
    (Real) 3.132461457144918e+00,
    (Real) 3.236888988781277e+00,
    (Real) 3.236888988781277e+00,
    (Real) 3.290148251426203e+00,
    (Real) 3.290148251426203e+00,
    (Real) 3.3068220128578854e+00};
  static constexpr std::array<Real,7> ti7 = {
    (Real)-4.574236474132155e+00,
    (Real) 4.574236474132155e+00,
    (Real)-2.874114667841827e+00,
    (Real) 2.874114667841827e+00,
    (Real)-1.3989392862653773e+00,
    (Real) 1.3989392862653773e+00,
    (Real)-6.101907007600904e-142};
  static constexpr std::array<Real,7> wr7 = {
    (Real) 1.958296204172131e-02,
    (Real) 1.958296204172131e-02,
    (Real)-9.600360651345459e-03,
    (Real)-9.600360651345459e-03,
    (Real)-2.5225049391191634e+00,
    (Real)-2.5225049391191634e+00,
    (Real) 6.025045542685179e+00};
  static constexpr std::array<Real,7> wi7 = {
    (Real)-2.7674983882176752e-02,
    (Real) 2.7674983882176752e-02,
    (Real) 7.616056105902165e-01,
    (Real)-7.616056105902165e-01,
    (Real)-2.6842033371961653e+00,
    (Real) 2.6842033371961653e+00,
    (Real)-5.569387056695978e-141};

  static constexpr std::array<Real,8> tr8 = {
    (Real) 3.3381533922638518e+00,
    (Real) 3.3381533922638518e+00,
    (Real) 3.444956252295488e+00,
    (Real) 3.444956252295488e+00,
    (Real) 3.504972109607807e+00,
    (Real) 3.504972109607807e+00,
    (Real) 3.532764209043209e+00,
    (Real) 3.532764209043209e+00};
  static constexpr std::array<Real,8> ti8 = {
    (Real)-5.058564429214023e+00,
    (Real) 5.058564429214023e+00,
    (Real)-3.4062853988427206e+00,
    (Real) 3.4062853988427206e+00,
    (Real)-1.9846958893854647e+00,
    (Real) 1.9846958893854647e+00,
    (Real)-6.531189708628177e-01,
    (Real) 6.531189708628177e-01};
  static constexpr std::array<Real,8> wr8 = {
    (Real)-8.859838517355943e-03,
    (Real)-8.859838517355943e-03,
    (Real) 4.232617530807562e-01,
    (Real) 4.232617530807562e-01,
    (Real)-2.8916860824724746e+00,
    (Real)-2.8916860824724746e+00,
    (Real) 2.9772841181083094e+00,
    (Real) 2.9772841181083094e+00};
  static constexpr std::array<Real,8> wi8 = {
    (Real)-1.1303556223177774e-02,
    (Real) 1.1303556223177774e-02,
    (Real) 1.6079405672156089e-01,
    (Real)-1.6079405672156089e-01,
    (Real) 1.1144740526974480e+00,
    (Real)-1.1144740526974480e+00,
    (Real)-6.911045539781778e+00,
    (Real) 6.911045539781778e+00};

  static constexpr std::array<Real,9> tr9 = {
    (Real) 3.531217406750906e+00,
    (Real) 3.531217406750906e+00,
    (Real) 3.6395352440653563e+00,
    (Real) 3.6395352440653563e+00,
    (Real) 3.704359812941743e+00,
    (Real) 3.704359812941743e+00,
    (Real) 3.740042891419225e+00,
    (Real) 3.740042891419225e+00,
    (Real) 3.7514993957716185e+00};
  static constexpr std::array<Real,9> ti9 = {
    (Real)-5.51118999364326e+00,
    (Real) 5.51118999364326e+00,
    (Real)-3.898977530472683e+00,
    (Real) 3.898977530472683e+00,
    (Real)-2.5204661364164487e+00,
    (Real) 2.5204661364164487e+00,
    (Real)-1.2403180611094913e+00,
    (Real) 1.2403180611094913e+00,
    (Real)-2.489802256734023e-140};
  static constexpr std::array<Real,9> wr9 = {
    (Real)-5.489679480132527e-03,
    (Real)-5.489679480132527e-03,
    (Real) 1.633437278732229e-01,
    (Real) 1.633437278732229e-01,
    (Real)-1.601994537954573e-02,
    (Real)-1.601994537954573e-02,
    (Real)-5.513675422918792e+00,
    (Real)-5.513675422918792e+00,
    (Real) 1.174368265116519e+01};
  static constexpr std::array<Real,9> wi9 = {
    (Real) 2.211091351713169e-03,
    (Real)-2.211091351713169e-03,
    (Real)-1.9307687809702626e-01,
    (Real) 1.9307687809702626e-01,
    (Real) 2.3456134744873234e+00,
    (Real)-2.3456134744873234e+00,
    (Real)-5.719512665903741e+00,
    (Real) 5.719512665903741e+00,
    (Real)-5.103188243883734e-139};

  static constexpr std::array<Real,10> tr10 = {
    (Real) 3.7137409438111786e+00,
    (Real) 3.7137409438111786e+00,
    (Real) 3.823021035348068e+00,
    (Real) 3.823021035348068e+00,
    (Real) 3.8913847331359337e+00,
    (Real) 3.8913847331359337e+00,
    (Real) 3.9329144143851247e+00,
    (Real) 3.9329144143851247e+00,
    (Real) 3.952715352245921e+00,
    (Real) 3.952715352245921e+00};
  static constexpr std::array<Real,10> ti10 = {
    (Real)-5.937504424255872e+00,
    (Real) 5.937504424255872e+00,
    (Real)-4.359620953695430e+00,
    (Real) 4.359620953695430e+00,
    (Real)-3.0168837702182794e+00,
    (Real) 3.0168837702182794e+00,
    (Real)-1.7778523146913043e+00,
    (Real) 1.7778523146913043e+00,
    (Real)-5.877263792685181e-01,
    (Real) 5.877263792685181e-01};
  static constexpr std::array<Real,10> wr10 = {
    (Real) 2.5662332920047595e-04,
    (Real) 2.5662332920047595e-04,
    (Real)-6.898207847105345e-02,
    (Real)-6.898207847105345e-02,
    (Real) 1.5233704946872504e+00,
    (Real) 1.5233704946872504e+00,
    (Real)-6.866656071865811e+00,
    (Real)-6.866656071865811e+00,
    (Real) 5.912011031676992e+00,
    (Real) 5.912011031676992e+00};
  static constexpr std::array<Real,10> wi10 = {
    (Real) 2.370379176426328e-03,
    (Real)-2.370379176426328e-03,
    (Real)-1.1532579752456389e-01,
    (Real) 1.1532579752456389e-01,
    (Real) 5.93120389584223e-01,
    (Real)-5.93120389584223e-01,
    (Real) 2.738126203872661e+00,
    (Real)-2.738126203872661e+00,
    (Real)-1.3946007798501306e+01,
    (Real) 1.3946007798501306e+01};

  static constexpr std::array<Real,11> tr11 = {
    (Real) 3.8872914367868026e+00,
    (Real) 3.8872914367868026e+00,
    (Real) 3.9971642659757305e+00,
    (Real) 3.9971642659757305e+00,
    (Real) 4.068198585952093e+00,
    (Real) 4.068198585952093e+00,
    (Real) 4.114199074195780e+00,
    (Real) 4.114199074195780e+00,
    (Real) 4.140327976506981e+00,
    (Real) 4.140327976506981e+00,
    (Real) 4.148822025045313e+00};
  static constexpr std::array<Real,11> ti11 = {
    (Real)-6.341536634302268e+00,
    (Real) 6.341536634302268e+00,
    (Real)-4.793589998343800e+00,
    (Real) 4.793589998343800e+00,
    (Real)-3.481290357380952e+00,
    (Real) 3.481290357380952e+00,
    (Real)-2.2762333546720495e+00,
    (Real) 2.2762333546720495e+00,
    (Real)-1.1261197234791214e+00,
    (Real) 1.1261197234791214e+00,
    (Real)-8.107465261282890e-139};
  static constexpr std::array<Real,11> wr11 = {
    (Real) 9.302713931207854e-04,
    (Real) 9.302713931207854e-04,
    (Real)-6.691577921412374e-02,
    (Real)-6.691577921412374e-02,
    (Real) 7.082999599416703e-01,
    (Real) 7.082999599416703e-01,
    (Real) 2.2717407434215864e-03,
    (Real) 2.2717407434215864e-03,
    (Real)-1.184842460988801e+01,
    (Real)-1.184842460988801e+01,
    (Real) 2.3407676834192984e+01};
  static constexpr std::array<Real,11> wi11 = {
    (Real) 1.5150671870006586e-04,
    (Real)-1.5150671870006586e-04,
    (Real) 1.479992873938146e-02,
    (Real)-1.479992873938146e-02,
    (Real)-7.966058935968946e-01,
    (Real) 7.966058935968946e-01,
    (Real) 6.230585869013911e+00,
    (Real)-6.230585869013911e+00,
    (Real)-1.2084620412072155e+01,
    (Real) 1.2084620412072155e+01,
    (Real)-3.699307694969999e-137};

  static constexpr std::array<Real,12> tr12 = {
    (Real) 4.053081638818847e+00,
    (Real) 4.053081638818847e+00,
    (Real) 4.163292523931315e+00,
    (Real) 4.163292523931315e+00,
    (Real) 4.236380668269799e+00,
    (Real) 4.236380668269799e+00,
    (Real) 4.285883519203109e+00,
    (Real) 4.285883519203109e+00,
    (Real) 4.316956058176300e+00,
    (Real) 4.316956058176300e+00,
    (Real) 4.331994580762299e+00,
    (Real) 4.331994580762299e+00};
  static constexpr std::array<Real,12> ti12 = {
    (Real)-6.726389354885087e+00,
    (Real) 6.726389354885087e+00,
    (Real)-5.204918817728363e+00,
    (Real) 5.204918817728363e+00,
    (Real)-3.919006378087133e+00,
    (Real) 3.919006378087133e+00,
    (Real)-2.742725504393537e+00,
    (Real) 2.742725504393537e+00,
    (Real)-1.625578797140899e+00,
    (Real) 1.625578797140899e+00,
    (Real)-5.387282045904411e-01,
    (Real) 5.387282045904411e-01};
  static constexpr std::array<Real,12> wr12 = {
    (Real) 1.5144683242261037e-04,
    (Real) 1.5144683242261037e-04,
    (Real)-3.2573758994071606e-03,
    (Real)-3.2573758994071606e-03,
    (Real)-3.133482962671488e-01,
    (Real)-3.133482962671488e-01,
    (Real) 4.517601770729534e+00,
    (Real) 4.517601770729534e+00,
    (Real)-1.5634471881369052e+01,
    (Real)-1.5634471881369052e+01,
    (Real) 1.1933324335965498e+01,
    (Real) 1.1933324335965498e+01};
  static constexpr std::array<Real,12> wi12 = {
    (Real)-3.339163152745438e-04,
    (Real) 3.339163152745438e-04,
    (Real) 3.360984393724251e-02,
    (Real)-3.360984393724251e-02,
    (Real)-5.784710887457339e-01,
    (Real) 5.784710887457339e-01,
    (Real) 1.7567413050479161e+00,
    (Real)-1.7567413050479161e+00,
    (Real) 6.388994399849236e+00,
    (Real)-6.388994399849236e+00,
    (Real)-2.8461484722324673e+01,
    (Real) 2.8461484722324673e+01};

  static constexpr std::array<Real,13> tr13 = {
    (Real) 4.212072888612011e+00,
    (Real) 4.212072888612011e+00,
    (Real) 4.322442111693796e+00,
    (Real) 4.322442111693796e+00,
    (Real) 4.397132764973097e+00,
    (Real) 4.397132764973097e+00,
    (Real) 4.449432088445341e+00,
    (Real) 4.449432088445341e+00,
    (Real) 4.484456223531104e+00,
    (Real) 4.484456223531104e+00,
    (Real) 4.504682173779635e+00,
    (Real) 4.504682173779635e+00,
    (Real) 4.511304100411463e+00};
  static constexpr std::array<Real,13> ti13 = {
    (Real)-7.094511131110056e+00,
    (Real) 7.094511131110056e+00,
    (Real)-5.5967256964293135e+00,
    (Real) 5.5967256964293135e+00,
    (Real)-4.3340330576122685e+00,
    (Real) 4.3340330576122685e+00,
    (Real)-3.1825983829324436e+00,
    (Real) 3.1825983829324436e+00,
    (Real)-2.093296988107494e+00,
    (Real) 2.093296988107494e+00,
    (Real)-1.0387477528198807e+00,
    (Real) 1.0387477528198807e+00,
    (Real)-1.946271217096134e-137};
  static constexpr std::array<Real,13> wr13 = {
    (Real)-1.088590738941445e-04,
    (Real)-1.088590738941445e-04,
    (Real) 1.4851258406863969e-02,
    (Real) 1.4851258406863969e-02,
    (Real)-3.8295981771766585e-01,
    (Real)-3.8295981771766585e-01,
    (Real) 2.366190116485650e+00,
    (Real) 2.366190116485650e+00,
    (Real) 1.0532674002236929e-01,
    (Real) 1.0532674002236929e-01,
    (Real)-2.528906494296514e+01,
    (Real)-2.528906494296514e+01,
    (Real) 4.737153100968546e+01};
  static constexpr std::array<Real,13> wi13 = {
    (Real)-8.909269226699887e-05,
    (Real) 8.909269226699887e-05,
    (Real) 6.360684878173006e-03,
    (Real)-6.360684878173006e-03,
    (Real) 6.020156318438565e-02,
    (Real)-6.020156318438565e-02,
    (Real)-2.617802351651345e+00,
    (Real) 2.617802351651345e+00,
    (Real) 1.5378641326633803e+01,
    (Real)-1.5378641326633803e+01,
    (Real)-2.5483924158931163e+01,
    (Real) 2.5483924158931163e+01,
    (Real)-1.9640639885045305e-135};

  static constexpr std::array<Real,14> tr14 = {
    (Real) 4.365042631141287e+00,
    (Real) 4.365042631141287e+00,
    (Real) 4.475441050761311e+00,
    (Real) 4.475441050761311e+00,
    (Real) 4.551395833420572e+00,
    (Real) 4.551395833420572e+00,
    (Real) 4.605962555007532e+00,
    (Real) 4.605962555007532e+00,
    (Real) 4.644203317703821e+00,
    (Real) 4.644203317703821e+00,
    (Real) 4.668631963496593e+00,
    (Real) 4.668631963496593e+00,
    (Real) 4.680559309369279e+00,
    (Real) 4.680559309369279e+00};
  static constexpr std::array<Real,14> ti14 = {
    (Real)-7.447873761481417e+00,
    (Real) 7.447873761481417e+00,
    (Real)-5.971479081962183e+00,
    (Real) 5.971479081962183e+00,
    (Real)-4.729468781179055e+00,
    (Real) 4.729468781179055e+00,
    (Real)-3.599818821607615e+00,
    (Real) 3.599818821607615e+00,
    (Real)-2.5344967473668443e+00,
    (Real) 2.5344967473668443e+00,
    (Real)-1.5071841416389609e+00,
    (Real) 1.5071841416389609e+00,
    (Real)-5.002473792064709e-01,
    (Real) 5.002473792064709e-01};
  static constexpr std::array<Real,14> wr14 = {
    (Real)-4.312686860326849e-05,
    (Real)-4.312686860326849e-05,
    (Real) 4.895994418347001e-03,
    (Real) 4.895994418347001e-03,
    (Real)-3.8928298888518086e-02,
    (Real)-3.8928298888518086e-02,
    (Real)-1.1178860093239285e+00,
    (Real)-1.1178860093239285e+00,
    (Real) 1.210763264171696e+01,
    (Real) 1.210763264171696e+01,
    (Real)-3.482673218478756e+01,
    (Real)-3.482673218478756e+01,
    (Real) 2.4371060983733205e+01,
    (Real) 2.4371060983733205e+01};
  static constexpr std::array<Real,14> wi14 = {
    (Real) 3.135245805051113e-05,
    (Real)-3.135245805051113e-05,
    (Real)-5.728997321364097e-03,
    (Real) 5.728997321364097e-03,
    (Real) 2.1608061400262654e-01,
    (Real)-2.1608061400262654e-01,
    (Real)-2.156246303539091e+00,
    (Real) 2.156246303539091e+00,
    (Real) 4.669272126055485e+00,
    (Real)-4.669272126055485e+00,
    (Real) 1.4495960280961373e+01,
    (Real)-1.4495960280961373e+01,
    (Real)-5.859325680724801e+01,
    (Real) 5.859325680724801e+01};

  static constexpr std::array<Real,15> tr15 = {
    (Real) 4.512630253056927e+00,
    (Real) 4.512630253056927e+00,
    (Real) 4.6229637917090685e+00,
    (Real) 4.6229637917090685e+00,
    (Real) 4.699923805511177e+00,
    (Real) 4.699923805511177e+00,
    (Real) 4.756352200736327e+00,
    (Real) 4.756352200736327e+00,
    (Real) 4.797250390237187e+00,
    (Real) 4.797250390237187e+00,
    (Real) 4.825143774452592e+00,
    (Real) 4.825143774452592e+00,
    (Real) 4.8414122824812775e+00,
    (Real) 4.8414122824812775e+00,
    (Real) 4.846762266475932e+00};
  static constexpr std::array<Real,15> ti15 = {
    (Real)-7.788092428868649e+00,
    (Real) 7.788092428868649e+00,
    (Real)-6.331171979887532e+00,
    (Real) 6.331171979887532e+00,
    (Real)-5.107770372984256e+00,
    (Real) 5.107770372984256e+00,
    (Real)-3.99746153171612e+00,
    (Real) 3.99746153171612e+00,
    (Real)-2.9531122929723455e+00,
    (Real) 2.9531122929723455e+00,
    (Real)-1.9492148782836751e+00,
    (Real) 1.9492148782836751e+00,
    (Real)-9.690733083695756e-01,
    (Real) 9.690733083695756e-01,
    (Real)-3.133654013490132e-136};
  static constexpr std::array<Real,15> wr15 = {
    (Real) 7.335203392844229e-06,
    (Real) 7.335203392844229e-06,
    (Real)-1.843939078852822e-03,
    (Real)-1.843939078852822e-03,
    (Real) 1.050569643090327e-01,
    (Real) 1.050569643090327e-01,
    (Real)-1.5804513487442309e+00,
    (Real)-1.5804513487442309e+00,
    (Real) 6.900969159056043e+00,
    (Real) 6.900969159056043e+00,
    (Real) 4.505734066831291e-01,
    (Real) 4.505734066831291e-01,
    (Real)-5.383958457431412e+01,
    (Real)-5.383958457431412e+01,
    (Real) 9.693054599377125e+01};
  static constexpr std::array<Real,15> wi15 = {
    (Real) 1.8601153697741973e-05,
    (Real)-1.8601153697741973e-05,
    (Real)-2.901698664732016e-03,
    (Real) 2.901698664732016e-03,
    (Real) 5.8307601124232736e-02,
    (Real)-5.8307601124232736e-02,
    (Real) 2.013803027616615e-01,
    (Real)-2.013803027616615e-01,
    (Real)-7.60065519744115e+00,
    (Real) 7.60065519744115e+00,
    (Real) 3.641276572752900e+01,
    (Real)-3.641276572752900e+01,
    (Real)-5.3766000425687395e+01,
    (Real) 5.3766000425687395e+01,
    (Real)-6.966986293383512e-134};
};



template <class Real, unsigned int order>
void FGT<Real,order>::fgt1d(long nv, long nx, Real x1, Real dx, const std::vector<Real>& alpha, Real delta, std::vector<Real>& u) {
  ensure_size(u, static_cast<std::size_t>(nv * nx));
  fgt1d_strided(u, alpha, nv, nx, 1, x1, dx, delta);

  sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, nv * nx * order * 22 * 1);
}

template <class Real, unsigned int order>
void FGT<Real,order>::fgt2d(long nx, long ny, Real x1, Real y1, Real dx, Real dy, const std::vector<Real>& alpha, Real delta, std::vector<Real>& u) {
  static std::vector<Real> tmp1; // TODO: not thread-safe
  ensure_size(tmp1, static_cast<std::size_t>(nx * ny));
  ensure_size(u,    static_cast<std::size_t>(nx * ny));

  fgt1d_strided(tmp1, alpha, ny, nx, 1, x1, dx, delta); // along x
  fgt1d_strided(u,    tmp1,  1, ny, nx, y1, dy, delta); // along y

  sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, ny * nx * order * 22 * 2);
}

template <class Real, unsigned int order>
void FGT<Real,order>::fgt3d(long nx, long ny, long nz, Real x1, Real y1, Real z1, Real dx, Real dy, Real dz, const std::vector<Real>& alpha, Real delta, std::vector<Real>& u) {
  static std::vector<Real> tmp1, tmp2; // TODO: not thread-safe
  ensure_size(tmp1, static_cast<std::size_t>(nx * ny * nz));
  ensure_size(tmp2, static_cast<std::size_t>(nx * ny * nz));
  ensure_size(u,    static_cast<std::size_t>(nx * ny * nz));

  fgt1d_strided(tmp1, alpha, nz * ny, nx, 1, x1, dx, delta); // x
  fgt1d_strided(tmp2, tmp1,  nz,      ny, nx, y1, dy, delta); // y
  fgt1d_strided(u,    tmp2,  1,       nz, ny*nx, z1, dz, delta); // z

  sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, nz * ny * nx * order * 22 * 3);
}



// -----------------------------------------------------------------------------
// Timing function (substitute for Fortran's CPU_TIME)
// -----------------------------------------------------------------------------
static double cpu_time() {
  using clock = std::chrono::high_resolution_clock;
  static auto const start = clock::now();
  auto now = clock::now();
  std::chrono::duration<double> elapsed = now - start;
  return elapsed.count();
}

template <class Real, unsigned int order>
void FGT<Real,order>::test(int ndim, long nx, long ny, long nz) {
  // Initialize the random generator (seed = 86456 as in Fortran)
  static std::mt19937_64 rng(86456); // Mersenne Twister 64-bit
  static std::uniform_real_distribution<double> dist(0.0, 1.0);
  const auto uniformReal = []() { return dist(rng); };

  int ntotal = 0;
  if      (ndim == 1) ntotal = nx;
  else if (ndim == 2) ntotal = nx * ny;
  else if (ndim == 3) ntotal = nx * ny * nz;
  else {
    std::cerr << "Error: ndim must be 1, 2, or 3.\n";
    std::abort();
  }

  // Allocate coordinate grids xs, ys, zs
  std::vector<double> xs(nx), ys(ny), zs(nz);

  // Step sizes
  double hx = 1.0 / double(nx);
  double hy = (ny > 0)? 1.0 / double(ny) : 0.0;
  double hz = (nz > 0)? 1.0 / double(nz) : 0.0;

  for (int i = 0; i < nx; i++) {
    xs[i] = i * hx;
  }
  for (int i = 0; i < ny; i++) {
    ys[i] = i * hy;
  }
  for (int i = 0; i < nz; i++) {
    zs[i] = i * hz;
  }

  double delta = 1.0e0;

  // Allocate alpha, ucomp depending on ndim
  // For 1D: alpha1(nx), ucomp1(nx)
  // For 2D: alpha2(nx,ny), ucomp2(nx,ny)
  // For 3D: alpha3(nx,ny,nz), ucomp3(nx,ny,nz)
  std::vector<double> alpha, ucomp;
  if (ndim == 1) {
    alpha.resize(nx);
    ucomp.resize(nx);
  } else if (ndim == 2) {
    alpha.resize(nx * ny);
    ucomp.resize(nx * ny);
  } else if (ndim == 3) {
    alpha.resize(nx * ny * nz);
    ucomp.resize(nx * ny * nz);
  }

  if (ndim == 1) {
    for (int i = 0; i < nx; i++) {
      alpha[i] = uniformReal() - 0.5;
    }
  } else if (ndim == 2) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        alpha[i + j*nx] = uniformReal() - 0.5;
      }
    }
  } else {
    for (int k = 0; k < nz; k++) {
      for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
          alpha[i + j*nx + k*(nx*ny)] = uniformReal() - 0.5;
        }
      }
    }
  }

  // For convenience, define x1=xs[0], y1=ys[0], z1=zs[0]
  double x1 = (nx > 0) ? xs[0] : 0.0;
  double y1 = (ny > 0) ? ys[0] : 0.0;
  double z1 = (nz > 0) ? zs[0] : 0.0;

  int ntest = 5;
  for (int i = 0; i < ntest; i++) {
    // Run the FGT
    double tstart = cpu_time();
    if (ndim == 1) {
      int nv = 1;
      FGT<double>::fgt1d(nv, nx, x1, hx, alpha, delta, ucomp);
    }
    else if (ndim == 2) {
      FGT<double>::fgt2d(nx, ny, x1, y1, hx, hy, alpha, delta, ucomp);
    }
    else {
      FGT<double>::fgt3d(nx, ny, nz, x1, y1, z1, hx, hy, hz,
          alpha, delta, ucomp);
    }
    double tend = cpu_time();

    std::cout << "time on FGT = " << (tend - tstart) << "\n";
    double nth = double(ntotal) / (tend - tstart);
    std::cout << "throughput = " << nth << "\n";

    // -------------------------------------------------------------------------
    // Check the accuracy on ncheck random grid points
    // Compute exact solution by direct summation of Gaussian
    // -------------------------------------------------------------------------
    int ncheck = 100;
    double dd = 0.0;  // sum of squared errors
    double dn = 0.0;  // sum of squared exact

    for (int ic = 0; ic < ncheck; ic++) {
      // pick random indices in each dimension
      int ix = int(std::floor(uniformReal() * nx));
      if (ix < 0) ix = 0;
      if (ix >= nx) ix = nx - 1;

      int iy = 0, iz = 0;
      if (ndim >= 2) {
        iy = int(std::floor(uniformReal() * ny));
        if (iy < 0) iy = 0;
        if (iy >= ny) iy = ny - 1;
      }
      if (ndim == 3) {
        iz = int(std::floor(uniformReal() * nz));
        if (iz < 0) iz = 0;
        if (iz >= nz) iz = nz - 1;
      }

      // exact solution
      double uexact = 0.0;
      double x = xs[ix];
      double y = (ndim >= 2) ? ys[iy] : 0.0;
      double z = (ndim == 3) ? zs[iz] : 0.0;

      if (ndim == 1) {
        for (int i = 0; i < nx; i++) {
          double r2 = (x - xs[i])*(x - xs[i]);
          uexact += alpha[i] * std::exp(-r2 / delta);
        }
        double diff = ucomp[ix] - uexact;
        dd += diff * diff;
      }
      else if (ndim == 2) {
        for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
            double r2 = (x - xs[i])*(x - xs[i]) +
              (y - ys[j])*(y - ys[j]);
            uexact += alpha[i + j*nx] * std::exp(-r2 / delta);
          }
        }
        double diff = ucomp[ix + iy*nx] - uexact;
        dd += diff * diff;
      }
      else { // ndim == 3
        for (int k = 0; k < nz; k++) {
          for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
              double r2 = (x - xs[i])*(x - xs[i]) +
                (y - ys[j])*(y - ys[j]) +
                (z - zs[k])*(z - zs[k]);
              uexact += alpha[i + j*nx + k*(nx*ny)]
                * std::exp(-r2 / delta);
            }
          }
        }
        double diff = ucomp[ix + iy*nx + iz*(nx*ny)] - uexact;
        dd += diff * diff;
      }

      dn += uexact * uexact;
    }

    double rerr = (dn > 0.0) ? std::sqrt(dd / dn) : 0.0;
    std::cout << "relative l2 error = " << rerr << "\n\n\n";
  }
}



template <class Real, unsigned int order>
inline void FGT<Real,order>::ensure_size(std::vector<Real>& v, std::size_t n) noexcept {
  if (v.size() < n) v.resize(n);
}

template <class Real, unsigned int order>
void FGT<Real,order>::fgt1d_strided(std::vector<Real>& u, const std::vector<Real>& alpha, long nv, long nx, long stride, Real x1, Real dx, Real delta) noexcept {
  SCTL_ASSERT(static_cast<size_t>(nv * nx * stride) <= alpha.size());
  SCTL_ASSERT(static_cast<size_t>(nv * nx * stride) <= u.size());

  const auto& tr = FGT_Coeffs<Real,order>::get_tr();
  const auto& ti = FGT_Coeffs<Real,order>::get_ti();

  Real zer[order], zei[order];
  const Real dx_delta2 = dx / std::sqrt(delta);
  for (unsigned int i = 0; i < order; i++) {
    const Real e = std::exp(-tr[i] * dx_delta2);
    const Real t = ti[i] * dx_delta2;
    zer[i] = e * std::cos(t);
    zei[i] =-e * std::sin(t);
  }

  // Wrap std::vector memory in sctl::Vector without copying
  sctl::Vector<Real>       u_(nv * nx * stride, sctl::Ptr2Itr<Real>((Real*)u.data(),     nv * nx * stride), false);
  const sctl::Vector<Real> a_(nv * nx * stride, sctl::Ptr2Itr<Real>((Real*)alpha.data(), nv * nx * stride), false);

  if (stride == 1) {                  // vectorize across nv
    fgt1d_vec_stride1(u_, a_, nv, nx, zer, zei);
  } else if (stride >= VecLen) {      // vectorize across stride
    fgt1d_vec(u_, a_, nv, nx, stride, stride, zer, zei);
  } else {                            // no-vectorize fallback
    fgt1d_novec(u_, a_, nv, nx, stride, stride, 0, zer, zei);
  }
}

template <class Real, unsigned int order>
template <unsigned int q0>
void FGT<Real,order>::fgt1d_novec(sctl::Vector<Real>& u, const sctl::Vector<Real>& alpha, long nv, long nx, long q, long stride, long offset, const Real zer[order], const Real zei[order]) noexcept {
  if constexpr (q0 == 0) {
    switch (q) {
      case  1: return fgt1d_novec< 1>(u, alpha, nv, nx, q, stride, offset, zer, zei);
      case  2: return fgt1d_novec< 2>(u, alpha, nv, nx, q, stride, offset, zer, zei);
      case  3: return fgt1d_novec< 3>(u, alpha, nv, nx, q, stride, offset, zer, zei);
      case  4: return fgt1d_novec< 4>(u, alpha, nv, nx, q, stride, offset, zer, zei);
      case  5: return fgt1d_novec< 5>(u, alpha, nv, nx, q, stride, offset, zer, zei);
      case  6: return fgt1d_novec< 6>(u, alpha, nv, nx, q, stride, offset, zer, zei);
      case  7: return fgt1d_novec< 7>(u, alpha, nv, nx, q, stride, offset, zer, zei);
      case  8: return fgt1d_novec< 8>(u, alpha, nv, nx, q, stride, offset, zer, zei);
      case  9: return fgt1d_novec< 9>(u, alpha, nv, nx, q, stride, offset, zer, zei);
      case 10: return fgt1d_novec<10>(u, alpha, nv, nx, q, stride, offset, zer, zei);
      case 11: return fgt1d_novec<11>(u, alpha, nv, nx, q, stride, offset, zer, zei);
      case 12: return fgt1d_novec<12>(u, alpha, nv, nx, q, stride, offset, zer, zei);
      case 13: return fgt1d_novec<13>(u, alpha, nv, nx, q, stride, offset, zer, zei);
      case 14: return fgt1d_novec<14>(u, alpha, nv, nx, q, stride, offset, zer, zei);
      case 15: return fgt1d_novec<15>(u, alpha, nv, nx, q, stride, offset, zer, zei);
      default: SCTL_ERROR("q must be in [1,15]");
    }
    return;
  }

  const auto& wr0 = FGT_Coeffs<Real,order>::get_wr();
  const auto& wi0 = FGT_Coeffs<Real,order>::get_wi();

  SCTL_ASSERT(q0 >= 1 && q0 <= stride);
  for (long l = 0; l < nv; ++l) {
    const Real* RESTRICT a0 = &alpha[l * nx * stride + offset];
    Real*       RESTRICT u0 = &u    [l * nx * stride + offset];

    Real hr[(q0?q0:1) * order], hi[(q0?q0:1) * order];
    for (unsigned int k = 0; k < order * q0; ++k) hr[k] = hi[k] = Real(0);
    for (long j = 0; j < nx; ++j) { // forward
      for (unsigned int i = 0; i < q0; ++i) {
        Real sum = Real(0);
        const Real a = a0[j * stride + i];
        for (unsigned int k = 0; k < order; ++k) {
          const unsigned int off = i * order + k;
          const Real hrp = hr[off];
          const Real hir = hi[off];
          const Real hrn = hrp * zer[k] - hir * zei[k] + a;
          const Real hin = hrp * zei[k] + hir * zer[k];
          hr[off] = hrn;
          hi[off] = hin;
          sum += hrn * wr0[k] - hin * wi0[k];
        }
        u0[j * stride + i] = sum;
      }
    }

    for (unsigned int k = 0; k < order * q0; ++k) hr[k] = hi[k] = Real(0);
    for (long j = nx - 1; j >= 0; --j) { // backward
      for (unsigned int i = 0; i < q0; ++i) {
        Real sum = Real(0);
        const Real a = a0[j * stride + i];
        for (unsigned int k = 0; k < order; ++k) {
          const unsigned int off = i * order + k;
          sum += hr[off] * wr0[k] - hi[off] * wi0[k];
          const Real hrp = hr[off] + a;
          const Real hir = hi[off];
          hr[off] = hrp * zer[k] - hir * zei[k];
          hi[off] = hrp * zei[k] + hir * zer[k];
        }
        u0[j * stride + i] += sum;
      }
    }
  }

  if (0) { // complex arithmetic version (slower)
    std::complex<Real> z[order], w[order];
    for (unsigned int k = 0; k < order; k++) {
      z[k] = std::complex<Real>(zer[k], zei[k]);
      w[k] = std::complex<Real>(wr0[k], wi0[k]);
    }

    SCTL_ASSERT(q0 >= 1 && q0 <= stride);
    for (long l = 0; l < nv; l++) {
      const Real* RESTRICT alpha_ = &alpha[l * nx * stride + offset];
      Real*       RESTRICT u_     =     &u[l * nx * stride + offset];

      std::complex<Real> h[(q0?q0:1) * order];
      for (unsigned int k = 0; k < q0 * order; k++) h[k] = Real(0);
      for (long j = 0; j < nx; j++) { // forward
        for (unsigned int i = 0; i < q0; i++) {
          std::complex<Real> sum = Real(0);
          for (unsigned int k = 0; k < order; k++) {
            h[k * q0 + i] = h[k * q0 + i] * z[k] + alpha_[j * stride + i];
            sum += h[k * q0 + i] * w[k];
          }
          u_[j * stride + i] = sum.real();
        }
      }

      for (unsigned int k = 0; k < q0 * order; k++) h[k] = Real(0);
      for (long j = nx - 1; j >= 0; j--) { // backward
        for (unsigned int i = 0; i < q0; i++) {
          std::complex<Real> sum = Real(0);
          for (unsigned int k = 0; k < order; k++) {
            sum += h[k * q0 + i] * w[k];
            h[k * q0 + i] = (h[k * q0 + i] + alpha_[j * stride + i]) * z[k];
          }
          u_[j * stride + i] += sum.real();
        }
      }
    }
  }
}

template <class Real, unsigned int order>
void FGT<Real,order>::fgt1d_vec(sctl::Vector<Real>& u, const sctl::Vector<Real>& alpha, long nv, long nx, long q, long stride, const Real zer[order], const Real zei[order]) noexcept {
  const auto& wr0 = FGT_Coeffs<Real,order>::get_wr();
  const auto& wi0 = FGT_Coeffs<Real,order>::get_wi();

  VecType zr[order], zi[order], wr[order], wi[order];
  for (unsigned int k = 0; k < order; k++) {
    zr[k] = zer[k];
    zi[k] = zei[k];
    wr[k] = wr0[k];
    wi[k] = wi0[k];
  }

  const long q0 = q - q % VecLen;
  for (long l = 0; l < nv; l++) {
    for (long i = 0; i < q0; i += VecLen) {
      const Real* alpha_ = &alpha[l * nx * stride + i];
      Real*       u_     = &u    [l * nx * stride + i];
      VecType hr[order], hi[order];

      // forward
      for (unsigned int k = 0; k < order; k++) hr[k] = hi[k] = VecType::Zero();
      for (long j = 0; j < nx; j++) {
        const VecType aa = VecType::Load(alpha_ + j * stride);
        VecType uu = VecType::Zero();

        for (unsigned int k = 0; k < order; k++) {
          const VecType hrp = hr[k];
          hr[k] = hrp * zr[k] - hi[k] * zi[k] + aa;
          hi[k] = hrp * zi[k] + hi[k] * zr[k];
          uu += hr[k] * wr[k] - hi[k] * wi[k];
        }

        uu.Store(u_ + j * stride);
      }

      // backward
      for (unsigned int k = 0; k < order; k++) hr[k] = hi[k] = VecType::Zero();
      for (long j = nx - 1; j >= 0; j--) {
        const VecType aa = VecType::Load(alpha_ + j * stride);
        const VecType u0 = VecType::Load(u_     + j * stride);
        VecType uu = VecType::Zero();

        for (unsigned int k = 0; k < order; k++) {
          uu += hr[k] * wr[k] - hi[k] * wi[k];
          const VecType hrp = hr[k] + aa;
          hr[k] = hrp * zr[k] - hi[k] * zi[k];
          hi[k] = hrp * zi[k] + hi[k] * zr[k];
        }

        (u0 + uu).Store(u_ + j * stride);
      }
    }
  }

  if (q0 < q) fgt1d_novec(u, alpha, nv, nx, q - q0, stride, q0, zer, zei);
}

template <class Real, unsigned int order>
void FGT<Real,order>::fgt1d_vec_stride1(sctl::Vector<Real>& u, const sctl::Vector<Real>& alpha, long nv, long nx, const Real zer[order], const Real zei[order]) noexcept {
  const auto& wr0 = FGT_Coeffs<Real,order>::get_wr();
  const auto& wi0 = FGT_Coeffs<Real,order>::get_wi();

  VecType zr[order], zi[order], wr[order], wi[order];
  for (unsigned int k = 0; k < order; k++) {
    zr[k] = zer[k];
    zi[k] = zei[k];
    wr[k] = wr0[k];
    wi[k] = wi0[k];
  }

  const long nv0 = nv - nv % VecLen;
  const long nx0 = nx - nx % VecLen;
  for (long i = 0; i < nv0; i += VecLen) {
    const Real* RESTRICT alpha_ = &alpha[i * nx];
    Real*       RESTRICT u_     = &u    [i * nx];

    // forward
    VecType hr[order], hi[order];
    for (unsigned int k = 0; k < order; k++) hr[k] = hi[k] = VecType::Zero();

    for (long j = 0; j < nx0; j += VecLen) {
      VecType uu[VecLen], aa[VecLen];
      for (unsigned int k = 0; k < VecLen; k++) { // load alpha
        uu[k] = VecType::Zero();
        aa[k] = VecType::Load((alpha_ + j) + nx * k);
      }
      Transpose<Real, VecLen>(aa);

      for (unsigned int l = 0; l < VecLen; l++) {
        for (unsigned int k = 0; k < order; k++) {
          const VecType hrp = hr[k];
          hr[k] = hr[k] * zr[k] - hi[k] * zi[k] + aa[l];
          hi[k] = hrp   * zi[k] + hi[k] * zr[k];
          uu[l] += hr[k] * wr[k] - hi[k] * wi[k];
        }
      }

      Transpose<Real, VecLen>(uu);
      for (unsigned int k = 0; k < VecLen; k++) uu[k].Store((u_ + j) + nx * k);
    }
    { // j = nx0,...,nx-1
      VecType uu[VecLen], aa[VecLen];
      for (int k = 0; k < static_cast<int>(nx - nx0); k++) { // load alpha
        uu[k] = VecType::Zero();

        Real buf[VecLen];
        for (unsigned int t = 0; t < VecLen; t++) buf[t] = alpha_[nx0 + k + nx * t];
        aa[k] = VecType::Load(buf);
      }

      for (int l = 0; l < static_cast<int>(nx - nx0); l++) {
        for (unsigned int k = 0; k < order; k++) {
          const VecType hrp = hr[k];
          hr[k] = hr[k] * zr[k] - hi[k] * zi[k] + aa[l];
          hi[k] = hrp   * zi[k] + hi[k] * zr[k];
          uu[l] += hr[k] * wr[k] - hi[k] * wi[k];
        }
      }

      for (int k = 0; k < static_cast<int>(nx - nx0); k++) { // store u
        Real buf[VecLen];
        uu[k].Store(buf);
        for (unsigned int t = 0; t < VecLen; t++) u_[nx0 + k + nx * t] = buf[t];
      }
    }

    // backward
    for (unsigned int k = 0; k < order; k++) hr[k] = hi[k] = VecType::Zero();
    { // j = nx-1,...,nx0
      VecType uu[VecLen], aa[VecLen];
      for (int k = 0; k < static_cast<int>(nx - nx0); k++) { // load u, alpha
        uu[k] = VecType::Zero();

        Real buf[VecLen];
        for (unsigned int t = 0; t < VecLen; t++) buf[t] = alpha_[nx0 + k + nx * t];
        aa[k] = VecType::Load(buf);
      }

      for (int l = static_cast<int>(nx - nx0 - 1); l >= 0; l--) {
        for (unsigned int k = 0; k < order; k++) {
          uu[l] += hr[k] * wr[k] - hi[k] * wi[k];
          const VecType hrp = hr[k] + aa[l];
          hr[k] = hrp * zr[k] - hi[k] * zi[k];
          hi[k] = hrp * zi[k] + hi[k] * zr[k];
        }
      }

      for (int k = 0; k < static_cast<int>(nx - nx0); k++) { // store u
        Real buf[VecLen];
        uu[k].Store(buf);
        for (unsigned int t = 0; t < VecLen; t++) u_[nx0 + k + nx * t] += buf[t];
      }
    }
    for (long j = nx0 - VecLen; j >= 0; j -= VecLen) {
      VecType uu[VecLen], u0[VecLen], aa[VecLen];
      for (unsigned int k = 0; k < VecLen; k++) { // load u, alpha
        uu[k] = VecType::Zero();
        u0[k] = VecType::Load((u_     + j) + nx * k);
        aa[k] = VecType::Load((alpha_ + j) + nx * k);
      }
      Transpose<Real, VecLen>(aa);

      for (int l = VecLen - 1; l >= 0; l--) {
        for (unsigned int k = 0; k < order; k++) {
          uu[l] += hr[k] * wr[k] - hi[k] * wi[k];
          const VecType hrp = hr[k] + aa[l];
          hr[k] = hrp * zr[k] - hi[k] * zi[k];
          hi[k] = hrp * zi[k] + hi[k] * zr[k];
        }
      }

      Transpose<Real, VecLen>(uu);
      for (unsigned int k = 0; k < VecLen; k++) (u0[k] + uu[k]).Store((u_ + j) + nx * k);
    }
  }

  //if (nv0 < nv) fgt1d_novec<1>(u, alpha, nv - nv0, nx, 1, 1, nv0 * nx, zer, zei);
  if (nv0 < nv) fgt1d_vec_stride1_(u, alpha, nv - nv0, nx, nv0 * nx, zer, zei);
}

template <class Real, unsigned int order>
void FGT<Real,order>::fgt1d_vec_stride1_(sctl::Vector<Real>& u, const sctl::Vector<Real>& alpha, long nv, long nx, long offset, const Real zer[order], const Real zei[order]) noexcept {
  const unsigned int order0 = (order + VecLen - 1) / VecLen * VecLen; // round up to multiple of VecLen
  const auto& wr0 = FGT_Coeffs<Real,order>::get_wr();
  const auto& wi0 = FGT_Coeffs<Real,order>::get_wi();

  VecType zr[order0/VecLen], zi[order0/VecLen], wr[order0/VecLen], wi[order0/VecLen];
  { // Set zr,zi,wr,wi with zero padding
    Real zr_buf[order0], zi_buf[order0], wr_buf[order0], wi_buf[order0];
    for (unsigned int k = order; k < order0; k++) zr_buf[k] = zi_buf[k] = wr_buf[k] = wi_buf[k] = Real(0);
    for (unsigned int k = 0; k < order; k++) {
      zr_buf[k] = zer[k];
      zi_buf[k] = zei[k];
      wr_buf[k] = wr0[k];
      wi_buf[k] = wi0[k];
    }
    for (unsigned int k = 0; k < order0/VecLen; k++) {
      zr[k] = VecType::Load(&zr_buf[k*VecLen]);
      zi[k] = VecType::Load(&zi_buf[k*VecLen]);
      wr[k] = VecType::Load(&wr_buf[k*VecLen]);
      wi[k] = VecType::Load(&wi_buf[k*VecLen]);
    }
  }

  const long nx0 = nx - nx % VecLen;
  for (long l = 0; l < nv; ++l) {
    const Real* RESTRICT a0 = &alpha[l * nx + offset];
    Real*       RESTRICT u0 = &u    [l * nx + offset];

    VecType sum[VecLen], hr[order0/VecLen], hi[order0/VecLen];
    for (unsigned int k = 0; k < order0/VecLen; ++k) hr[k] = hi[k] = VecType::Zero();

    for (long j = 0; j < nx0; j += VecLen) { // forward
      for (unsigned int k = 0; k < VecLen; ++k) sum[k] = VecType::Zero();
      for (unsigned int j_ = 0; j_ < VecLen; ++j_) {
        const VecType a = VecType::Load1(a0 + j + j_);
        for (unsigned int k = 0; k < order0/VecLen; ++k) {
          const VecType hrp = hr[k], hir = hi[k];
          hr[k] = hrp * zr[k] - hir * zi[k] + a;
          hi[k] = hrp * zi[k] + hir * zr[k];
          sum[j_] += hr[k] * wr[k] - hi[k] * wi[k];
        }
      }
      Transpose<Real,VecLen>(sum);
      for (unsigned int k = 1; k < VecLen; ++k) sum[0] += sum[k];
      sum[0].Store(u0 + j);
    }
    if (nx0 < nx) { // j = nx0,...,nx-1
      const long j = nx0;
      for (unsigned int k = 0; k < VecLen; ++k) sum[k] = VecType::Zero();
      for (int j_ = 0; j_ < static_cast<int>(nx - nx0); ++j_) {
        const VecType aa = VecType::Load1(a0 + j + j_);
        for (unsigned int k = 0; k < order0/VecLen; ++k) {
          const VecType hrp = hr[k], hir = hi[k];
          hr[k] = hrp * zr[k] - hir * zi[k] + aa;
          hi[k] = hrp * zi[k] + hir * zr[k];
          sum[j_] += hr[k] * wr[k] - hi[k] * wi[k];
        }
      }
      Transpose<Real,VecLen>(sum);
      for (unsigned int k = 1; k < VecLen; ++k) sum[0] += sum[k];

      Real buf[VecLen];
      sum[0].Store(buf);
      for (int k = 0; k < static_cast<int>(nx - nx0); ++k) u0[nx0 + k] = buf[k];
    }

    for (unsigned int k = 0; k < order0/VecLen; ++k) hr[k] = hi[k] = VecType::Zero();

    if (nx0 < nx) { // j = nx-1,...,nx0
      const long j = nx - 1;
      for (unsigned int k = 0; k < VecLen; ++k) sum[k] = VecType::Zero();
      for (int j_ = 0; j_ < static_cast<int>(nx - nx0); ++j_) {
        const VecType a = a0[j - j_];
        for (unsigned int k = 0; k < order0/VecLen; ++k) {
          sum[j_] += hr[k] * wr[k] - hi[k] * wi[k];
          const VecType hrp = hr[k] + a, hir = hi[k];
          hr[k] = hrp * zr[k] - hir * zi[k];
          hi[k] = hrp * zi[k] + hir * zr[k];
        }
      }
      Transpose<Real,VecLen>(sum);
      for (unsigned int k = 1; k < VecLen; ++k) sum[0] += sum[k];

      Real buf[VecLen];
      sum[0].Store(buf);
      for (int k = 0; k < static_cast<int>(nx - nx0); ++k) u0[j - k] += buf[k];
    }
    for (long j = nx0 - VecLen; j >= 0; j -= VecLen) { // backward
      const VecType uu = VecType::Load(u0 + j);
      for (unsigned int k = 0; k < VecLen; ++k) sum[k] = VecType::Zero();
      for (int j_ = VecLen - 1; j_ >= 0; --j_) {
        const VecType a = a0[j + j_];
        for (unsigned int k = 0; k < order0/VecLen; ++k) {
          sum[j_] += hr[k] * wr[k] - hi[k] * wi[k];
          const VecType hrp = hr[k] + a, hir = hi[k];
          hr[k] = hrp * zr[k] - hir * zi[k];
          hi[k] = hrp * zi[k] + hir * zr[k];
        }
      }
      Transpose<Real,VecLen>(sum);
      for (unsigned int k = 1; k < VecLen; ++k) sum[0] += sum[k];
      (sum[0] + uu).Store(u0 + j);
    }
  }
}

#endif // FGT_IMPL_HPP

