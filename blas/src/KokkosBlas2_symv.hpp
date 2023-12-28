//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#ifndef KOKKOSBLAS2_SYMV_HPP_
#define KOKKOSBLAS2_SYMV_HPP_

/// \file KokkosBlas2_gemv.hpp
/// \brief BLAS 2 kernels specifically optimized for typical
///   Tpetra::MultiVector use cases.

#include <KokkosBlas2_symv_spec.hpp>
#include <KokkosKernels_helpers.hpp>
#include <KokkosKernels_Error.hpp>
#include <sstream>

namespace KokkosBlas {

/// \brief Dense symmetric matrix-vector multiply: y = beta*y + alpha*A*x.
///
/// \tparam AViewType Input symmetric matrix, as a rank 1 Kokkos::View
/// \tparam XViewType Input vector, as a rank 1 Kokkos::View
/// \tparam YViewType Output vector, as a nonconst rank 1 Kokkos::View
/// \tparam AlphaCoeffType Type of input coefficient alpha
/// \tparam BetaCoeffType Type of input coefficient beta
///
/// \param space [in] execution space instance on which to run the
///   kernel. This may contain information about which stream to
///   run on.
/// \param uplo [in] "U", "u" for upper triangular, "L", "l" for lower
///   triangular.  All characters after the first are
///   ignored.  This works just like the BLAS routines.
/// \param alpha [in] Input coefficient of A*x
/// \param A [in] Input symmetric matrix, as a rank 1 Kokkos::View
/// \param x [in] Input vector, as a rank 1 Kokkos::View
/// \param beta [in] Input coefficient of y
/// \param y [in/out] Output vector, as a nonconst rank 1 Kokkos::View
template <class ExecutionSpace, class AViewType, class XViewType,
          class YViewType>
void symv(const ExecutionSpace& space, const char uplo_[],
          typename AViewType::const_value_type& alpha, const AViewType& A,
          const XViewType& x, typename YViewType::const_value_type& beta,
          const YViewType& y) {
  static_assert(Kokkos::is_execution_space_v<ExecutionSpace>,
                "KokkosBlas::symv: ExecutionSpace must be a valid Kokkos "
                "execution space.");
  static_assert(Kokkos::is_view<AViewType>::value,
                "KokkosBlas::symv: AViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<XViewType>::value,
                "KokkosBlas::symv: XViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<YViewType>::value,
                "KokkosBlas::symv: YViewType must be a Kokkos::View.");
  static_assert(static_cast<int>(AViewType::rank()) == 2,
                "KokkosBlas::symv: AViewType must have rank 1.");
  static_assert(static_cast<int>(XViewType::rank()) == 1,
                "KokkosBlas::symv: XViewType must have rank 1.");
  static_assert(static_cast<int>(YViewType::rank()) == 1,
                "KokkosBlas::symv: YViewType must have rank 1.");
  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename AViewType::memory_space>::accessible,
      "KokkosBlas::symv: AViewType must be accessible from ExecutionSpace");
  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename XViewType::memory_space>::accessible,
      "KokkosBlas::symv: XViewType must be accessible from ExecutionSpace");
  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename YViewType::memory_space>::accessible,
      "KokkosBlas::symv: YViewType must be accessible from ExecutionSpace");

  // Check compatibility of dimensions at run time.
  if((A.extent(0) != A.extent(1)) || (A.extent(0) != x.extent(0))
     || (A.extent(1) != y.extent(0))) {
    std::ostringstream os;
    os << "KokkosBlas::symv: A.extent(0), A.extent(1), x.extent(0) and "
       << "y.extent(0) should all be equal, however their respective "
       << "values are: " << A.extent(0) << ", " << A.extent(1) << ", "
       << x.extent(0) << " and " << y.extent(0);
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }

  // Make the uplo parameter lower case for easy comparison.
  std::string uplo(1, uplo_[0]);
  std::tolower(static_cast<unsigned char>(uplo[0]));
  if((uplo.compare("u") == 0)
     || (uplo.compare("l") == 0)) {
    std::ostringstream os;
    os << "KokkosBlas::symv: uplo must be set to U, u, L or l, instead "
       << "it is set to " << uplo[0];
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }

  using ALayout = typename AViewType::array_layout;

  // Minimize the number of Impl::GEMV instantiations, by
  // standardizing on particular View specializations for its template
  // parameters.
  using AVT = Kokkos::View<typename AViewType::const_value_type**, ALayout,
			   typename AViewType::device_type,
			   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using XVT = Kokkos::View<typename XViewType::const_value_type*,
			   typename KokkosKernels::Impl::GetUnifiedLayoutPreferring<
			     XViewType, ALayout>::array_layout,
			   typename XViewType::device_type,
			   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using YVT = Kokkos::View<typename YViewType::non_const_value_type*,
			   typename KokkosKernels::Impl::GetUnifiedLayoutPreferring<
			     YViewType, ALayout>::array_layout,
			   typename YViewType::device_type,
			   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  // Degenerate case is essentially same as scal - use fallback impl
  // to avoid potential (unlikely?) circular dependence issues by including
  // other KokkosBlas headers
  bool useFallback = A.extent(0) == 0 || A.extent(1) == 0;
  // If A is LayoutRight and we have the BLAS, cuBLAS or rocBLAS TPL, use
  // fallback because those only support LayoutLeft

  if (useFallback) {
    const bool eti_spec_avail =
        KokkosBlas::Impl::symv_eti_spec_avail<ExecutionSpace, AVT, XVT,
                                              YVT>::value;
    using fallback_impl_type = Impl::SYMV<ExecutionSpace, AVT, XVT, YVT,
					  false, eti_spec_avail>;
    fallback_impl_type::symv(space, uplo_, alpha, A, x, beta, y);
  } else {
    using impl_type = Impl::SYMV<ExecutionSpace, AVT, XVT, YVT>;
    impl_type::symv(space, uplo_, alpha, A, x, beta, y);
  }
}

/// \brief Dense symmetric  matrix-vector multiply: y = beta*y + alpha*A*x.
///
/// \tparam AViewType Input symmetric matrix, as a rank 1 Kokkos::View
/// \tparam XViewType Input vector, as a rank 1 Kokkos::View
/// \tparam YViewType Output vector, as a nonconst rank 1 Kokkos::View
/// \tparam AlphaCoeffType Type of input coefficient alpha
/// \tparam BetaCoeffType Type of input coefficient beta
///
/// \param uplo [in] "U", "u" for upper triangular and "L", "l"
///   for lower triangular. All characters after the first are
///   ignored.  This works just like the BLAS routines.
/// \param alpha [in] Input coefficient of A*x
/// \param A [in] Input symmetric matrix, as a rank 2 Kokkos::View
/// \param x [in] Input vector, as a rank 1 Kokkos::View
/// \param beta [in] Input coefficient of y
/// \param y [in/out] Output vector, as a nonconst rank 1 Kokkos::View
template <class AViewType, class XViewType, class YViewType>
void symv(const char uplo[], typename AViewType::const_value_type& alpha,
          const AViewType& A, const XViewType& x,
          typename YViewType::const_value_type& beta, const YViewType& y) {
  symv(typename AViewType::execution_space{}, uplo, alpha, A, x, beta, y);
}

}  // namespace KokkosBlas
#endif  // KOKKOSBLAS2_SYMV_HPP_
