#ifndef FGT_HPP
#define FGT_HPP

#include "sctl.hpp"
#include <array>
#include <vector>
#include <complex>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <typeinfo>
#include <iostream>

#if defined(__clang__) || defined(__GNUC__)
  #define RESTRICT __restrict__
#elif defined(_MSC_VER)
  #define RESTRICT __restrict
#else
  #define RESTRICT
#endif

/*
 * This is done using only SIMD operations and no extra memory allocation.
 * Transpose a square matrix of SIMD vectors in place.
 *
 * @tparam Real   Floating point type (float or double).
 * @tparam VecLen Number of elements in each SIMD vector.
 *
 * @param[in,out] v Array of VecLen SIMD vectors, each of length VecLen.
 */
template <class Real, unsigned int VecLen>
inline void Transpose(sctl::Vec<Real,VecLen> (&v)[VecLen]) noexcept;

// Test function for Transpose
template <class Real, unsigned int N>
void test_transpose();

/*
 * Tensor product Fast Gaussian Transform (FGT) in 1D, 2D, and 3D using sum of exponentials approximation.
 *
 * @tparam Real  Floating point type (float or double).
 * @tparam order Number of exponentials in the approximation (max 15).
 */
template <class Real, unsigned int order = 4>
class FGT {
  using VecType = sctl::Vec<Real>;
  static constexpr unsigned int VecLen = static_cast<unsigned int>(VecType::Size());
  static_assert(order >= 1 && order <= 15, "order must be in [1,15]");

public:
  // 1D: alpha,u size = nv * nx
  static void fgt1d(long nv, long nx, Real x1, Real dx, const std::vector<Real>& alpha, Real delta, std::vector<Real>& u);

  // 2D: alpha,u size = nx * ny
  static void fgt2d(long nx, long ny, Real x1, Real y1, Real dx, Real dy, const std::vector<Real>& alpha, Real delta, std::vector<Real>& u);

  // 3D: alpha,u size = nx * ny * nz
  static void fgt3d(long nx, long ny, long nz, Real x1, Real y1, Real z1, Real dx, Real dy, Real dz, const std::vector<Real>& alpha, Real delta, std::vector<Real>& u);

  // Test function for FGT
  static void test(int ndim, long nx, long ny, long nz);

private:
  static inline void ensure_size(std::vector<Real>& v, std::size_t n) noexcept;

  /*
   * Compute FGT along the nx dimension of an nv×nx×stride array (row-major).
   * u, alpha have size nv*nx*stride.
   */
  static void fgt1d_strided(std::vector<Real>& u, const std::vector<Real>& alpha, long nv, long nx, long stride, Real x1, Real dx, Real delta) noexcept;

  /*
   * Compute FGT along the nx dimension of an nv×nx×stride array (row-major) without vectorization.
   *
   * @tparam q0 Number of points in the last dimension to process (1 to 15)
   * @param[out] u Output array of size nv*nx*stride in row-major order with FGT computed along nx dimension in the nv*nx*q slice.
   * @param[in] alpha Input array of size nv*nx*stride in row-major order.
   * @param[in] nv Number of vectors.
   * @param[in] nx Number of points in x dimension.
   * @param[in] q Number of points in the last dimension (q > 0).
   * @param[in] stride Stride in the last dimension (stride >= q).
   * @param[in] offset Offset in the last dimension (0 <= offset <= stride-q).
   * @param[in] zer Real parts the exponentials evaluated at dx.
   * @param[in] zei Imaginary parts the exponentials evaluated at dx.
   */
  template <unsigned int q0=0>
  static void fgt1d_novec(sctl::Vector<Real>& u, const sctl::Vector<Real>& alpha, long nv, long nx, long q, long stride, long offset, const Real zer[order], const Real zei[order]) noexcept;

  /*
   * Compute FGT along the nx dimension of an nv×nx×stride array (row-major) and vectorize across stride dimension.
   *
   * @param[out] u Output array of size nv*nx*stride in row-major order with FGT computed along nx dimension in the nv*nx*q slice.
   * @param[in] alpha Input array of size nv*nx*stride in row-major order.
   * @param[in] nv Number of vectors.
   * @param[in] nx Number of points in x dimension.
   * @param[in] q Number of points in the last dimension (q > 0).
   * @param[in] stride Stride in the last dimension (stride >= q).
   * @param[in] zer Real parts the exponentials evaluated at dx.
   * @param[in] zei Imaginary parts the exponentials evaluated at dx.
   */
  static void fgt1d_vec(sctl::Vector<Real>& u, const sctl::Vector<Real>& alpha, long nv, long nx, long q, long stride, const Real zer[order], const Real zei[order]) noexcept;

  /*
   * Compute FGT along the nx dimension of an nv×nx array (row-major) and vectorize across nv dimension.
   *
   * @param[out] u Output array of size nv*nx in row-major order with FGT computed along nx dimension.
   * @param[in] alpha Input array of size nv*nx in row-major order.
   * @param[in] nv Number of vectors.
   * @param[in] nx Number of points in x dimension.
   * @param[in] zer Real parts the exponentials evaluated at dx.
   * @param[in] zei Imaginary parts the exponentials evaluated at dx.
   */
  static void fgt1d_vec_stride1(sctl::Vector<Real>& u, const sctl::Vector<Real>& alpha, long nv, long nx, const Real zer[order], const Real zei[order]) noexcept;

  /*
   * Compute FGT along the nx dimension of an nv×nx array (row-major) and vectorize across nx dimension.
   *
   * @param[out] u Output array of size nv*nx in row-major order with FGT computed along nx dimension.
   * @param[in] alpha Input array of size nv*nx in row-major order.
   * @param[in] nv Number of vectors.
   * @param[in] nx Number of points in x dimension.
   * @param[in] zer Real parts the exponentials evaluated at dx.
   * @param[in] zei Imaginary parts the exponentials evaluated at dx.
   */
  static void fgt1d_vec_stride1_(sctl::Vector<Real>& u, const sctl::Vector<Real>& alpha, long nv, long nx, long offset, const Real zer[order], const Real zei[order]) noexcept;
};

#include "fgt.cpp"

#endif // FGT_HPP

