/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Luc Berger-Vergiat (lberge@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef __KOKKOSBLASDEVICE_AXPY_IMPL_HPP__
#define __KOKKOSBLASDEVICE_AXPY_IMPL_HPP__

namespace KokkosBlas {
namespace Experimental {
namespace Device {

/// \brief SerialAxpy callable from device
struct SerialAxpy {
  template <typename XViewType, typename YViewType, typename alphaViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const alphaViewType &alpha,
                                           const XViewType &X,
                                           const YViewType &Y);
};




///
/// Serial Internal Impl
/// ====================
struct SerialAxpyInternal {
  template <typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const int m, const ScalarType alpha,
      const ValueType* KOKKOS_RESTRICT X,
      const int xs0,
      /* */ ValueType* KOKKOS_RESTRICT Y,
      const int ys0) {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
    for (int i = 0; i < m; ++i) Y[i * ys0] += alpha * X[i * xs0];

    return 0;
  }
};

///
/// Serial Impl
/// ===========
template <typename XViewType, typename YViewType, typename alphaViewType>
KOKKOS_INLINE_FUNCTION int SerialAxpy::invoke(const alphaViewType& alpha,
                                              const XViewType& X,
                                              const YViewType& Y) {
#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  static_assert(Kokkos::is_view<XViewType>::value,
                "KokkosBatched::axpy: XViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<YViewType>::value,
                "KokkosBatched::axpy: YViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<alphaViewType>::value,
                "KokkosBatched::axpy: alphaViewType is not a Kokkos::View.");
  static_assert(XViewType::Rank == 1,
                "KokkosBatched::axpy: XViewType must have rank 2.");
  static_assert(YViewType::Rank == 1,
                "KokkosBatched::axpy: YViewType must have rank 2.");
  static_assert(alphaViewType::Rank == 1,
                "KokkosBatched::axpy: alphaViewType must have rank 1.");

  // Check compatibility of dimensions at run time.
  if (X.extent(0) != Y.extent(0)) {
    KOKKOS_IMPL_DO_NOT_USE_PRINTF(
        "KokkosBatched::axpy: Dimensions of X and Y do not match: X: %d, "
        "Y: %d\n",
        (int)X.extent(0), (int)Y.extent(0));
    return 1;
  }
  if (alpha.extent(0) != 1) {
    KOKKOS_IMPL_DO_NOT_USE_PRINTF(
        "KokkosBatched::axpy: Dimension of alpha is not 1: %d\n",
        (int)alpha.extent(0));
    return 1;
  }
#endif

  return SerialAxpyInternal::template invoke<
      typename alphaViewType::non_const_value_type,
      typename XViewType::non_const_value_type>(
      X.extent(0), *(alpha.data()), X.data(),
      X.stride_0(), Y.data(), Y.stride_0());
}

} // namespace Device
} // namespace Experimental
} // namespace KokkosBlas

#endif // __KOKKOSBLASDEVICE_AXPY_IMPL_HPP__
