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

#ifndef KOKKOSBLAS2_SYMV_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS2_SYMV_TPL_SPEC_DECL_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
#include "KokkosBlas_Host_tpl.hpp"

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS2_SYMV_DETERMINE_ARGS(LAYOUTA)                             \
  bool A_is_lr      = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;     \
  const int M       = static_cast<int>(A_is_lr ? A.extent(1) : A.extent(0)); \
  const int N       = static_cast<int>(A_is_lr ? A.extent(0) : A.extent(1)); \
  constexpr int one = 1;                                                     \
  const int LDA     = A_is_lr ? A.stride(0) : A.stride(1);                   \
                                                                             \
  char transa;                                                               \
  if ((trans[0] == 'N') || (trans[0] == 'n'))                                \
    transa = A_is_lr ? 'T' : 'N';                                            \
  else if ((trans[0] == 'T') || (trans[0] == 't'))                           \
    transa = A_is_lr ? 'N' : 'T';                                            \
  else {                                                                     \
    if (A_is_lr)                                                             \
      throw std::runtime_error(                                              \
          "Error: HostBlas::symv conjugate transpose requires LayoutLeft "   \
          "views.");                                                         \
    transa = 'C';                                                            \
  }

#define KOKKOSBLAS2_DSYMV_BLAS(LAYOUTA, LAYOUTX, LAYOUTY, MEM_SPACE,       \
                               ETI_SPEC_AVAIL)                             \
  template <class ExecSpace>                                               \
  struct SYMV<                                                             \
      ExecSpace,                                                           \
      Kokkos::View<const double**, LAYOUTA,                                \
                   Kokkos::Device<ExecSpace, MEM_SPACE>,                   \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      Kokkos::View<const double*, LAYOUTX,                                 \
                   Kokkos::Device<ExecSpace, MEM_SPACE>,                   \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      Kokkos::View<double*, LAYOUTY, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,              \
      true, ETI_SPEC_AVAIL> {                                              \
    typedef double SCALAR;                                                 \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                          \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >         \
        AViewType;                                                         \
    typedef Kokkos::View<const SCALAR*, LAYOUTX,                           \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >         \
        XViewType;                                                         \
    typedef Kokkos::View<SCALAR*, LAYOUTY,                                 \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >         \
        YViewType;                                                         \
                                                                           \
    static void symv(const ExecSpace& /* space */, const char uplo[],      \
                     typename AViewType::const_value_type& alpha,          \
                     const AViewType& A, const XViewType& X,               \
                     typename YViewType::const_value_type& beta,           \
                     const YViewType& Y) {                                 \
      Kokkos::Profiling::pushRegion("KokkosBlas::symv[TPL_BLAS,double]");  \
      KOKKOSBLAS2_SYMV_DETERMINE_ARGS(LAYOUTA);                            \
      HostBlas<double>::symv(uplo, N, alpha, A.data(), LDA, X.data(), one, \
                             beta, Y.data(), one);                         \
      Kokkos::Profiling::popRegion();                                      \
    }                                                                      \
  };

#define KOKKOSBLAS2_SSYMV_BLAS(LAYOUTA, LAYOUTX, LAYOUTY, MEM_SPACE,           \
                               ETI_SPEC_AVAIL)                                 \
  template <class ExecSpace>                                                   \
  struct SYMV<                                                                 \
      ExecSpace,                                                               \
      Kokkos::View<const float**, LAYOUTA,                                     \
                   Kokkos::Device<ExecSpace, MEM_SPACE>,                       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const float*, LAYOUTX,                                      \
                   Kokkos::Device<ExecSpace, MEM_SPACE>,                       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<float*, LAYOUTY, Kokkos::Device<ExecSpace, MEM_SPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef float SCALAR;                                                      \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTX,                               \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        XViewType;                                                             \
    typedef Kokkos::View<SCALAR*, LAYOUTY,                                     \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        YViewType;                                                             \
                                                                               \
    static void symv(const ExecSpace& /* space */, const char uplo[],          \
                     typename AViewType::const_value_type& alpha,              \
                     const AViewType& A, const XViewType& X,                   \
                     typename YViewType::const_value_type& beta,               \
                     const YViewType& Y) {                                     \
      Kokkos::Profiling::pushRegion("KokkosBlas::symv[TPL_BLAS,float]");       \
      KOKKOSBLAS2_SYMV_DETERMINE_ARGS(LAYOUTA);                                \
      HostBlas<float>::symv(uplo, N, alpha, A.data(), LDA, X.data(), one,      \
                            beta, Y.data(), one);                              \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS2_ZSYMV_BLAS(LAYOUTA, LAYOUTX, LAYOUTY, MEM_SPACE,         \
                               ETI_SPEC_AVAIL)                               \
  template <class ExecSpace>                                                 \
  struct SYMV<ExecSpace,                                                     \
              Kokkos::View<const Kokkos::complex<double>**, LAYOUTA,         \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
              Kokkos::View<const Kokkos::complex<double>*, LAYOUTX,          \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
              Kokkos::View<Kokkos::complex<double>*, LAYOUTY,                \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
              true, ETI_SPEC_AVAIL> {                                        \
    typedef Kokkos::complex<double> SCALAR;                                  \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                            \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,               \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >           \
        AViewType;                                                           \
    typedef Kokkos::View<const SCALAR*, LAYOUTX,                             \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,               \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >           \
        XViewType;                                                           \
    typedef Kokkos::View<SCALAR*, LAYOUTY,                                   \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,               \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >           \
        YViewType;                                                           \
                                                                             \
    static void symv(const ExecSpace& /* space */, const char uplo[],        \
                     typename AViewType::const_value_type& alpha,            \
                     const AViewType& A, const XViewType& X,                 \
                     typename YViewType::const_value_type& beta,             \
                     const YViewType& Y) {                                   \
      Kokkos::Profiling::pushRegion(                                         \
          "KokkosBlas::symv[TPL_BLAS,complex<double>]");                     \
      KOKKOSBLAS2_SYMV_DETERMINE_ARGS(LAYOUTA);                              \
      const std::complex<double> alpha_val = alpha, beta_val = beta;         \
      HostBlas<std::complex<double> >::symv(                                 \
          uplo, N, alpha_val,                                                \
          reinterpret_cast<const std::complex<double>*>(A.data()), LDA,      \
          reinterpret_cast<const std::complex<double>*>(X.data()), one,      \
          beta_val, reinterpret_cast<std::complex<double>*>(Y.data()), one); \
      Kokkos::Profiling::popRegion();                                        \
    }                                                                        \
  };

#define KOKKOSBLAS2_CSYMV_BLAS(LAYOUTA, LAYOUTX, LAYOUTY, MEM_SPACE,        \
                               ETI_SPEC_AVAIL)                              \
  template <class ExecSpace>                                                \
  struct SYMV<ExecSpace,                                                    \
              Kokkos::View<const Kokkos::complex<float>**, LAYOUTA,         \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,            \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,       \
              Kokkos::View<const Kokkos::complex<float>*, LAYOUTX,          \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,            \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,       \
              Kokkos::View<Kokkos::complex<float>*, LAYOUTY,                \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,            \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,       \
              true, ETI_SPEC_AVAIL> {                                       \
    typedef Kokkos::complex<float> SCALAR;                                  \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                           \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        AViewType;                                                          \
    typedef Kokkos::View<const SCALAR*, LAYOUTX,                            \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        XViewType;                                                          \
    typedef Kokkos::View<SCALAR*, LAYOUTY,                                  \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        YViewType;                                                          \
                                                                            \
    static void symv(const ExecSpace& /* space */, const char uplo[],       \
                     typename AViewType::const_value_type& alpha,           \
                     const AViewType& A, const XViewType& X,                \
                     typename YViewType::const_value_type& beta,            \
                     const YViewType& Y) {                                  \
      Kokkos::Profiling::pushRegion(                                        \
          "KokkosBlas::symv[TPL_BLAS,complex<float>]");                     \
      KOKKOSBLAS2_SYMV_DETERMINE_ARGS(LAYOUTA);                             \
      const std::complex<float> alpha_val = alpha, beta_val = beta;         \
      HostBlas<std::complex<float> >::symv(                                 \
          uplo, N, alpha_val,                                               \
          reinterpret_cast<const std::complex<float>*>(A.data()), LDA,      \
          reinterpret_cast<const std::complex<float>*>(X.data()), one,      \
          beta_val, reinterpret_cast<std::complex<float>*>(Y.data()), one); \
      Kokkos::Profiling::popRegion();                                       \
    }                                                                       \
  };

KOKKOSBLAS2_DSYMV_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                       Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS2_DSYMV_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                       Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS2_DSYMV_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight,
                       Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS2_DSYMV_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight,
                       Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS2_SSYMV_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                       Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS2_SSYMV_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                       Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS2_SSYMV_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight,
                       Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS2_SSYMV_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight,
                       Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS2_ZSYMV_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                       Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS2_ZSYMV_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                       Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS2_ZSYMV_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight,
                       Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS2_ZSYMV_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight,
                       Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS2_CSYMV_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                       Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS2_CSYMV_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                       Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS2_CSYMV_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight,
                       Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS2_CSYMV_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight,
                       Kokkos::LayoutRight, Kokkos::HostSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_BLAS

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS2_SYMV_CUBLAS_DETERMINE_ARGS(LAYOUTA)                      \
  bool A_is_lr      = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;     \
  const int M       = static_cast<int>(A_is_lr ? A.extent(1) : A.extent(0)); \
  const int N       = static_cast<int>(A_is_lr ? A.extent(0) : A.extent(1)); \
  constexpr int one = 1;                                                     \
  const int LDA     = A_is_lr ? A.stride(0) : A.stride(1);                   \
                                                                             \
  cublasOperation_t transa;                                                  \
  if ((trans[0] == 'N') || (trans[0] == 'n'))                                \
    transa = A_is_lr ? CUBLAS_OP_T : CUBLAS_OP_N;                            \
  else if ((trans[0] == 'T') || (trans[0] == 't'))                           \
    transa = A_is_lr ? CUBLAS_OP_N : CUBLAS_OP_T;                            \
  else {                                                                     \
    if (A_is_lr)                                                             \
      throw std::runtime_error(                                              \
          "Error: cublas symv conjugate transpose requires LayoutLeft "      \
          "views.");                                                         \
    transa = CUBLAS_OP_C;                                                    \
  }

#define KOKKOSBLAS2_DSYMV_CUBLAS(LAYOUTA, LAYOUTX, LAYOUTY, MEM_SPACE,         \
                                 ETI_SPEC_AVAIL)                               \
  template <class ExecSpace>                                                   \
  struct SYMV<                                                                 \
      ExecSpace,                                                               \
      Kokkos::View<const double**, LAYOUTA,                                    \
                   Kokkos::Device<ExecSpace, MEM_SPACE>,                       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const double*, LAYOUTX,                                     \
                   Kokkos::Device<ExecSpace, MEM_SPACE>,                       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<double*, LAYOUTY, Kokkos::Device<ExecSpace, MEM_SPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef double SCALAR;                                                     \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTX,                               \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        XViewType;                                                             \
    typedef Kokkos::View<SCALAR*, LAYOUTY,                                     \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        YViewType;                                                             \
                                                                               \
    static void symv(const ExecSpace& space, const char kk_uplo[],             \
                     typename AViewType::const_value_type& alpha,              \
                     const AViewType& A, const XViewType& X,                   \
                     typename YViewType::const_value_type& beta,               \
                     const YViewType& Y) {                                     \
      Kokkos::Profiling::pushRegion("KokkosBlas::symv[TPL_CUBLAS,double]");    \
      cublasFillMode_t uplo = fill_mode_kk_to_cublas(kk_uplo);                 \
      KOKKOSBLAS2_SYMV_CUBLAS_DETERMINE_ARGS(LAYOUTA);                         \
      KokkosBlas::Impl::CudaBlasSingleton& s =                                 \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                    \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasSetStream(s.handle, space.cuda_stream()));                     \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasDsymv(s.handle, uplo, N, &alpha,      \
                                               A.data(), LDA, X.data(), one,   \
                                               &beta, Y.data(), one));         \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasSetStream(s.handle, NULL));           \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS2_SSYMV_CUBLAS(LAYOUTA, LAYOUTX, LAYOUTY, MEM_SPACE,         \
                                 ETI_SPEC_AVAIL)                               \
  template <class ExecSpace>                                                   \
  struct SYMV<                                                                 \
      ExecSpace,                                                               \
      Kokkos::View<const float**, LAYOUTA,                                     \
                   Kokkos::Device<ExecSpace, MEM_SPACE>,                       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const float*, LAYOUTX,                                      \
                   Kokkos::Device<ExecSpace, MEM_SPACE>,                       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<float*, LAYOUTY, Kokkos::Device<ExecSpace, MEM_SPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef float SCALAR;                                                      \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTX,                               \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        XViewType;                                                             \
    typedef Kokkos::View<SCALAR*, LAYOUTY,                                     \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        YViewType;                                                             \
                                                                               \
    static void symv(const ExecSpace& space, const char kk_uplo[],               \
                     typename AViewType::const_value_type& alpha,              \
                     const AViewType& A, const XViewType& X,                   \
                     typename YViewType::const_value_type& beta,               \
                     const YViewType& Y) {                                     \
      Kokkos::Profiling::pushRegion("KokkosBlas::symv[TPL_CUBLAS,float]");     \
      cublasFillMode_t uplo = fill_mode_kk_to_cublas(kk_uplo);                 \
      KOKKOSBLAS2_SYMV_CUBLAS_DETERMINE_ARGS(LAYOUTA);                         \
      KokkosBlas::Impl::CudaBlasSingleton& s =                                 \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                    \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasSetStream(s.handle, space.cuda_stream()));                     \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasSsymv(s.handle, uplo, N, &alpha, \
                                               A.data(), LDA, X.data(), one,   \
                                               &beta, Y.data(), one));         \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasSetStream(s.handle, NULL));           \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS2_ZSYMV_CUBLAS(LAYOUTA, LAYOUTX, LAYOUTY, MEM_SPACE,         \
                                 ETI_SPEC_AVAIL)                               \
  template <class ExecSpace>                                                   \
  struct SYMV<ExecSpace,                                                       \
              Kokkos::View<const Kokkos::complex<double>**, LAYOUTA,           \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,               \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,          \
              Kokkos::View<const Kokkos::complex<double>*, LAYOUTX,            \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,               \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,          \
              Kokkos::View<Kokkos::complex<double>*, LAYOUTY,                  \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,               \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,          \
              true, ETI_SPEC_AVAIL> {                                          \
    typedef Kokkos::complex<double> SCALAR;                                    \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTX,                               \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        XViewType;                                                             \
    typedef Kokkos::View<SCALAR*, LAYOUTY,                                     \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        YViewType;                                                             \
                                                                               \
    static void symv(const ExecSpace& space, const char kk_uplo[],               \
                     typename AViewType::const_value_type& alpha,              \
                     const AViewType& A, const XViewType& X,                   \
                     typename YViewType::const_value_type& beta,               \
                     const YViewType& Y) {                                     \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::symv[TPL_CUBLAS,complex<double>]");                     \
      cublasFillMode_t uplo = fill_mode_kk_to_cublas(kk_uplo);                 \
      KOKKOSBLAS2_SYMV_CUBLAS_DETERMINE_ARGS(LAYOUTA);                         \
      KokkosBlas::Impl::CudaBlasSingleton& s =                                 \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                    \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasSetStream(s.handle, space.cuda_stream()));                     \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasZsymv(s.handle, uplo, N,                                  \
                      reinterpret_cast<const cuDoubleComplex*>(&alpha),        \
                      reinterpret_cast<const cuDoubleComplex*>(A.data()), LDA, \
                      reinterpret_cast<const cuDoubleComplex*>(X.data()), one, \
                      reinterpret_cast<const cuDoubleComplex*>(&beta),         \
                      reinterpret_cast<cuDoubleComplex*>(Y.data()), one));     \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasSetStream(s.handle, NULL));           \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS2_CSYMV_CUBLAS(LAYOUTA, LAYOUTX, LAYOUTY, MEM_SPACE,        \
                                 ETI_SPEC_AVAIL)                              \
  template <class ExecSpace>                                                  \
  struct SYMV<ExecSpace,                                                      \
              Kokkos::View<const Kokkos::complex<float>**, LAYOUTA,           \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              Kokkos::View<const Kokkos::complex<float>*, LAYOUTX,            \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              Kokkos::View<Kokkos::complex<float>*, LAYOUTY,                  \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              true, ETI_SPEC_AVAIL> {                                         \
    typedef Kokkos::complex<float> SCALAR;                                    \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                             \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<const SCALAR*, LAYOUTX,                              \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        XViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUTY,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        YViewType;                                                            \
                                                                              \
    static void symv(const ExecSpace& space, const char kk_uplo[],              \
                     typename AViewType::const_value_type& alpha,             \
                     const AViewType& A, const XViewType& X,                  \
                     typename YViewType::const_value_type& beta,              \
                     const YViewType& Y) {                                    \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::symv[TPL_CUBLAS,complex<float>]");                     \
      cublasFillMode_t uplo = fill_mode_kk_to_cublas(kk_uplo);                \
      KOKKOSBLAS2_SYMV_CUBLAS_DETERMINE_ARGS(LAYOUTA);                        \
      KokkosBlas::Impl::CudaBlasSingleton& s =                                \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                   \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                           \
          cublasSetStream(s.handle, space.cuda_stream()));                    \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasCsymv(                               \
          s.handle, uplo, N, reinterpret_cast<const cuComplex*>(&alpha), \
          reinterpret_cast<const cuComplex*>(A.data()), LDA,                  \
          reinterpret_cast<const cuComplex*>(X.data()), one,                  \
          reinterpret_cast<const cuComplex*>(&beta),                          \
          reinterpret_cast<cuComplex*>(Y.data()), one));                      \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasSetStream(s.handle, NULL));          \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

KOKKOSBLAS2_DSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                         Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS2_DSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                         Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS2_DSYMV_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight,
                         Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS2_DSYMV_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight,
                         Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS2_SSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                         Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS2_SSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                         Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS2_SSYMV_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight,
                         Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS2_SSYMV_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight,
                         Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS2_ZSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                         Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS2_ZSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                         Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS2_ZSYMV_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight,
                         Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS2_ZSYMV_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight,
                         Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS2_CSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                         Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS2_CSYMV_CUBLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                         Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSBLAS2_CSYMV_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight,
                         Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSBLAS2_CSYMV_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight,
                         Kokkos::LayoutRight, Kokkos::CudaSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUBLAS

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS2_SYMV_ROCBLAS_DETERMINE_ARGS(LAYOUT)                      \
  bool A_is_lr      = std::is_same<Kokkos::LayoutRight, LAYOUT>::value;      \
  const int M       = static_cast<int>(A_is_lr ? A.extent(1) : A.extent(0)); \
  const int N       = static_cast<int>(A_is_lr ? A.extent(0) : A.extent(1)); \
  constexpr int one = 1;                                                     \
  const int LDA     = A_is_lr ? A.stride(0) : A.stride(1);                   \
                                                                             \
  rocblas_operation transa;                                                  \
  if ((trans[0] == 'N') || (trans[0] == 'n'))                                \
    transa = A_is_lr ? rocblas_operation_transpose : rocblas_operation_none; \
  else if ((trans[0] == 'T') || (trans[0] == 't'))                           \
    transa = A_is_lr ? rocblas_operation_none : rocblas_operation_transpose; \
  else {                                                                     \
    if (A_is_lr)                                                             \
      throw std::runtime_error(                                              \
          "Error: rocBLAS Xsymv conjugate transpose requires LayoutLeft "    \
          "matrix.");                                                        \
    transa = rocblas_operation_conjugate_transpose;                          \
  }

#define KOKKOSBLAS2_DSYMV_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)         \
  template <class ExecSpace>                                                 \
  struct SYMV<                                                               \
      ExecSpace,                                                             \
      Kokkos::View<const double**, LAYOUT,                                   \
                   Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                   \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<const double*, LAYOUT,                                    \
                   Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                   \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<double*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      true, ETI_SPEC_AVAIL> {                                                \
    typedef double SCALAR;                                                   \
    typedef Kokkos::View<const SCALAR**, LAYOUT,                             \
                         Kokkos::Device<Kokkos::HIP, MEM_SPACE>,             \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >           \
        AViewType;                                                           \
    typedef Kokkos::View<const SCALAR*, LAYOUT,                              \
                         Kokkos::Device<Kokkos::HIP, MEM_SPACE>,             \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >           \
        XViewType;                                                           \
    typedef Kokkos::View<SCALAR*, LAYOUT,                                    \
                         Kokkos::Device<Kokkos::HIP, MEM_SPACE>,             \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >           \
        YViewType;                                                           \
                                                                             \
    static void symv(const ExecSpace& space, const char kk_uplo[],             \
                     typename AViewType::const_value_type& alpha,            \
                     const AViewType& A, const XViewType& X,                 \
                     typename YViewType::const_value_type& beta,             \
                     const YViewType& Y) {                                   \
      Kokkos::Profiling::pushRegion("KokkosBlas::symv[TPL_ROCBLAS,double]"); \
      rocblas_fill uplo = fill_mode_kk_to_rocblas(kk_uplo);		     \
      KOKKOSBLAS2_SYMV_ROCBLAS_DETERMINE_ARGS(LAYOUT);                       \
      KokkosBlas::Impl::RocBlasSingleton& s =                                \
          KokkosBlas::Impl::RocBlasSingleton::singleton();                   \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(                                         \
          rocblas_set_stream(s.handle, space.hip_stream()));                 \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(                                         \
          rocblas_dsymv(s.handle, uplo, N, &alpha, A.data(), LDA,       \
                        X.data(), one, &beta, Y.data(), one));               \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_set_stream(s.handle, NULL));     \
      Kokkos::Profiling::popRegion();                                        \
    }                                                                        \
  };

#define KOKKOSBLAS2_SSYMV_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)        \
  template <class ExecSpace>                                                \
  struct SYMV<                                                              \
      ExecSpace,                                                            \
      Kokkos::View<const float**, LAYOUT,                                   \
                   Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,               \
      Kokkos::View<const float*, LAYOUT,                                    \
                   Kokkos::Device<Kokkos::HIP, MEM_SPACE>,                  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,               \
      Kokkos::View<float*, LAYOUT, Kokkos::Device<Kokkos::HIP, MEM_SPACE>,  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,               \
      true, ETI_SPEC_AVAIL> {                                               \
    typedef float SCALAR;                                                   \
    typedef Kokkos::View<const SCALAR**, LAYOUT,                            \
                         Kokkos::Device<Kokkos::HIP, MEM_SPACE>,            \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        AViewType;                                                          \
    typedef Kokkos::View<const SCALAR*, LAYOUT,                             \
                         Kokkos::Device<Kokkos::HIP, MEM_SPACE>,            \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        XViewType;                                                          \
    typedef Kokkos::View<SCALAR*, LAYOUT,                                   \
                         Kokkos::Device<Kokkos::HIP, MEM_SPACE>,            \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        YViewType;                                                          \
                                                                            \
    static void symv(const ExecSpace& space, const char kk_uplo[],            \
                     typename AViewType::const_value_type& alpha,           \
                     const AViewType& A, const XViewType& X,                \
                     typename YViewType::const_value_type& beta,            \
                     const YViewType& Y) {                                  \
      Kokkos::Profiling::pushRegion("KokkosBlas::symv[TPL_ROCBLAS,float]"); \
      rocblas_fill uplo = fill_mode_kk_to_rocblas(kk_uplo);		     \
      KOKKOSBLAS2_SYMV_ROCBLAS_DETERMINE_ARGS(LAYOUT);                      \
      KokkosBlas::Impl::RocBlasSingleton& s =                               \
          KokkosBlas::Impl::RocBlasSingleton::singleton();                  \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(                                        \
          rocblas_set_stream(s.handle, space.hip_stream()));                \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(                                        \
          rocblas_ssymv(s.handle, uplo, N, &alpha, A.data(), LDA,      \
                        X.data(), one, &beta, Y.data(), one));              \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_set_stream(s.handle, NULL));    \
      Kokkos::Profiling::popRegion();                                       \
    }                                                                       \
  };

#define KOKKOSBLAS2_ZSYMV_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)      \
  template <class ExecSpace>                                              \
  struct SYMV<ExecSpace,                                                  \
              Kokkos::View<const Kokkos::complex<double>**, LAYOUT,       \
                           Kokkos::Device<Kokkos::HIP, MEM_SPACE>,        \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,     \
              Kokkos::View<const Kokkos::complex<double>*, LAYOUT,        \
                           Kokkos::Device<Kokkos::HIP, MEM_SPACE>,        \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,     \
              Kokkos::View<Kokkos::complex<double>*, LAYOUT,              \
                           Kokkos::Device<Kokkos::HIP, MEM_SPACE>,        \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,     \
              true, ETI_SPEC_AVAIL> {                                     \
    typedef Kokkos::complex<double> SCALAR;                               \
    typedef Kokkos::View<const SCALAR**, LAYOUT,                          \
                         Kokkos::Device<Kokkos::HIP, MEM_SPACE>,          \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >        \
        AViewType;                                                        \
    typedef Kokkos::View<const SCALAR*, LAYOUT,                           \
                         Kokkos::Device<Kokkos::HIP, MEM_SPACE>,          \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >        \
        XViewType;                                                        \
    typedef Kokkos::View<SCALAR*, LAYOUT,                                 \
                         Kokkos::Device<Kokkos::HIP, MEM_SPACE>,          \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >        \
        YViewType;                                                        \
                                                                          \
    static void symv(const ExecSpace& space, const char kk_uplo[],          \
                     typename AViewType::const_value_type& alpha,         \
                     const AViewType& A, const XViewType& X,              \
                     typename YViewType::const_value_type& beta,          \
                     const YViewType& Y) {                                \
      Kokkos::Profiling::pushRegion(                                      \
          "KokkosBlas::symv[TPL_ROCBLAS,complex<double>]");               \
      rocblas_fill uplo = fill_mode_kk_to_rocblas(kk_uplo);		     \
      KOKKOSBLAS2_SYMV_ROCBLAS_DETERMINE_ARGS(LAYOUT);                    \
      KokkosBlas::Impl::RocBlasSingleton& s =                             \
          KokkosBlas::Impl::RocBlasSingleton::singleton();                \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(                                      \
          rocblas_set_stream(s.handle, space.hip_stream()));              \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_zsymv(                        \
          s.handle, uplo, N,                                         \
          reinterpret_cast<const rocblas_double_complex*>(&alpha),        \
          reinterpret_cast<const rocblas_double_complex*>(A.data()), LDA, \
          reinterpret_cast<const rocblas_double_complex*>(X.data()), one, \
          reinterpret_cast<const rocblas_double_complex*>(&beta),         \
          reinterpret_cast<rocblas_double_complex*>(Y.data()), one));     \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_set_stream(s.handle, NULL));  \
      Kokkos::Profiling::popRegion();                                     \
    }                                                                     \
  };

#define KOKKOSBLAS2_CSYMV_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)     \
  template <class ExecSpace>                                             \
  struct SYMV<ExecSpace,                                                 \
              Kokkos::View<const Kokkos::complex<float>**, LAYOUT,       \
                           Kokkos::Device<Kokkos::HIP, MEM_SPACE>,       \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,    \
              Kokkos::View<const Kokkos::complex<float>*, LAYOUT,        \
                           Kokkos::Device<Kokkos::HIP, MEM_SPACE>,       \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,    \
              Kokkos::View<Kokkos::complex<float>*, LAYOUT,              \
                           Kokkos::Device<Kokkos::HIP, MEM_SPACE>,       \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,    \
              true, ETI_SPEC_AVAIL> {                                    \
    typedef Kokkos::complex<float> SCALAR;                               \
    typedef Kokkos::View<const SCALAR**, LAYOUT,                         \
                         Kokkos::Device<Kokkos::HIP, MEM_SPACE>,         \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >       \
        AViewType;                                                       \
    typedef Kokkos::View<const SCALAR*, LAYOUT,                          \
                         Kokkos::Device<Kokkos::HIP, MEM_SPACE>,         \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >       \
        XViewType;                                                       \
    typedef Kokkos::View<SCALAR*, LAYOUT,                                \
                         Kokkos::Device<Kokkos::HIP, MEM_SPACE>,         \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >       \
        YViewType;                                                       \
                                                                         \
    static void symv(const ExecSpace& space, const char kk_uplo[],         \
                     typename AViewType::const_value_type& alpha,        \
                     const AViewType& A, const XViewType& X,             \
                     typename YViewType::const_value_type& beta,         \
                     const YViewType& Y) {                               \
      Kokkos::Profiling::pushRegion(                                     \
          "KokkosBlas::symv[TPL_ROCBLAS,complex<float>]");               \
      rocblas_fill uplo = fill_mode_kk_to_rocblas(kk_uplo);		     \
      KOKKOSBLAS2_SYMV_ROCBLAS_DETERMINE_ARGS(LAYOUT);                   \
      KokkosBlas::Impl::RocBlasSingleton& s =                            \
          KokkosBlas::Impl::RocBlasSingleton::singleton();               \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(                                     \
          rocblas_set_stream(s.handle, space.hip_stream()));             \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_csymv(                       \
          s.handle, uplo, N,                                        \
          reinterpret_cast<const rocblas_float_complex*>(&alpha),        \
          reinterpret_cast<const rocblas_float_complex*>(A.data()), LDA, \
          reinterpret_cast<const rocblas_float_complex*>(X.data()), one, \
          reinterpret_cast<const rocblas_float_complex*>(&beta),         \
          reinterpret_cast<rocblas_float_complex*>(Y.data()), one));     \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_set_stream(s.handle, NULL)); \
      Kokkos::Profiling::popRegion();                                    \
    }                                                                    \
  };

KOKKOSBLAS2_DSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS2_DSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)
KOKKOSBLAS2_DSYMV_ROCBLAS(Kokkos::LayoutRight, Kokkos::HIPSpace, true)
KOKKOSBLAS2_DSYMV_ROCBLAS(Kokkos::LayoutRight, Kokkos::HIPSpace, false)

KOKKOSBLAS2_SSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS2_SSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)
KOKKOSBLAS2_SSYMV_ROCBLAS(Kokkos::LayoutRight, Kokkos::HIPSpace, true)
KOKKOSBLAS2_SSYMV_ROCBLAS(Kokkos::LayoutRight, Kokkos::HIPSpace, false)

KOKKOSBLAS2_ZSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS2_ZSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)
KOKKOSBLAS2_ZSYMV_ROCBLAS(Kokkos::LayoutRight, Kokkos::HIPSpace, true)
KOKKOSBLAS2_ZSYMV_ROCBLAS(Kokkos::LayoutRight, Kokkos::HIPSpace, false)

KOKKOSBLAS2_CSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, true)
KOKKOSBLAS2_CSYMV_ROCBLAS(Kokkos::LayoutLeft, Kokkos::HIPSpace, false)
KOKKOSBLAS2_CSYMV_ROCBLAS(Kokkos::LayoutRight, Kokkos::HIPSpace, true)
KOKKOSBLAS2_CSYMV_ROCBLAS(Kokkos::LayoutRight, Kokkos::HIPSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCBLAS

// ONEMKL
#if defined(KOKKOSKERNELS_ENABLE_TPL_MKL) && defined(KOKKOS_ENABLE_SYCL)
#include <mkl.h>
#include <oneapi/mkl/blas.hpp>
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

template <typename T, bool is_complex = false>
struct kokkos_to_std_type_map {
  using type = T;
};

// e.g., map Kokkos::complex<float> to std::complex<float>
template <typename T>
struct kokkos_to_std_type_map<T, true> {
  using type = std::complex<typename Kokkos::ArithTraits<T>::mag_type>;
};

#define KOKKOSBLAS2_SYMV_ONEMKL(SCALAR, LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)     \
  template <class ExecSpace>                                                   \
  struct SYMV<                                                                 \
      ExecSpace,                                                               \
      Kokkos::View<const SCALAR**, LAYOUT,                                     \
                   Kokkos::Device<Kokkos::Experimental::SYCL, MEM_SPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const SCALAR*, LAYOUT,                                      \
                   Kokkos::Device<Kokkos::Experimental::SYCL, MEM_SPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<SCALAR*, LAYOUT,                                            \
                   Kokkos::Device<Kokkos::Experimental::SYCL, MEM_SPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    using device_type = Kokkos::Device<ExecSpace, MEM_SPACE>;                  \
    using mem_traits  = Kokkos::MemoryTraits<Kokkos::Unmanaged>;               \
    using AViewType =                                                          \
        Kokkos::View<const SCALAR**, LAYOUT, device_type, mem_traits>;         \
    using XViewType =                                                          \
        Kokkos::View<const SCALAR*, LAYOUT, device_type, mem_traits>;          \
    using YViewType = Kokkos::View<SCALAR*, LAYOUT, device_type, mem_traits>;  \
                                                                               \
    static void symv(const ExecSpace& exec, const char kk_uplo[],             \
                     typename AViewType::const_value_type& alpha,              \
                     const AViewType& A, const XViewType& X,                   \
                     typename YViewType::const_value_type& beta,               \
                     const YViewType& Y) {                                     \
      const std::int64_t N = A.extent(0);                                      \
      oneapi::mkl::uplo uplo = fill_mode_kk_to_onemkl(kk_uplo[0]);             \
      std::string label      = "KokkosBlas::symv[TPL_ONEMKL," +                \
                          Kokkos::ArithTraits<SCALAR>::name() + "]";           \
                                                                               \
      Kokkos::Profiling::pushRegion(label);                                    \
      using mag_type = kokkos_to_std_type_map<                                 \
          SCALAR, Kokkos::ArithTraits<SCALAR>::is_complex>::type;              \
      const mag_type* a = reinterpret_cast<const mag_type*>(A.data());         \
      const mag_type* x = reinterpret_cast<const mag_type*>(X.data());         \
      mag_type* y       = reinterpret_cast<mag_type*>(Y.data());               \
      if constexpr (std::is_same_v<Kokkos::LayoutRight, LAYOUT>) {	       \
	const std::int64_t LDA = A.stride(0);				\
        oneapi::mkl::blas::row_major::symv(exec.sycl_queue(), uplo, N,     \
                                           alpha, a, LDA, x, 1, beta, y, 1);   \
      } else {                                                                 \
	const std::int64_t LDA = A.stride(1);				\
        oneapi::mkl::blas::column_major::symv(                                 \
            exec.sycl_queue(), uplo, N, alpha, a, LDA, x, 1, beta, y, 1);  \
      }                                                                        \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

KOKKOSBLAS2_SYMV_ONEMKL(float, Kokkos::LayoutLeft,
                        Kokkos::Experimental::SYCLDeviceUSMSpace, true)
KOKKOSBLAS2_SYMV_ONEMKL(float, Kokkos::LayoutRight,
                        Kokkos::Experimental::SYCLDeviceUSMSpace, true)
KOKKOSBLAS2_SYMV_ONEMKL(double, Kokkos::LayoutLeft,
                        Kokkos::Experimental::SYCLDeviceUSMSpace, true)
KOKKOSBLAS2_SYMV_ONEMKL(double, Kokkos::LayoutRight,
                        Kokkos::Experimental::SYCLDeviceUSMSpace, true)
KOKKOSBLAS2_SYMV_ONEMKL(Kokkos::complex<float>, Kokkos::LayoutLeft,
                        Kokkos::Experimental::SYCLDeviceUSMSpace, true)
KOKKOSBLAS2_SYMV_ONEMKL(Kokkos::complex<float>, Kokkos::LayoutRight,
                        Kokkos::Experimental::SYCLDeviceUSMSpace, true)
KOKKOSBLAS2_SYMV_ONEMKL(Kokkos::complex<double>, Kokkos::LayoutLeft,
                        Kokkos::Experimental::SYCLDeviceUSMSpace, true)
KOKKOSBLAS2_SYMV_ONEMKL(Kokkos::complex<double>, Kokkos::LayoutRight,
                        Kokkos::Experimental::SYCLDeviceUSMSpace, true)
}  // namespace Impl
}  // namespace KokkosBlas
#endif

#endif
