// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSPARSE_SPMV_MV_TPL_SPEC_AVAIL_HPP_
#define KOKKOSPARSE_SPMV_MV_TPL_SPEC_AVAIL_HPP_

namespace KokkosSparse {
namespace Impl {

// Specialization struct which defines whether a specialization exists
template <class ExecutionSpace, class Handle, class AMatrix, class XVector, class YVector,
          const bool integerScalarType = std::is_integral_v<typename AMatrix::non_const_value_type>>
struct spmv_mv_tpl_spec_avail {
  enum : bool { value = false };
};

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#define KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(SCALAR, ORDINAL, OFFSET, XL, YL, MEMSPACE)                      \
  template <>                                                                                                        \
  struct spmv_mv_tpl_spec_avail<                                                                                     \
      Kokkos::Cuda, KokkosSparse::Impl::SPMVHandleImpl<Kokkos::Cuda, MEMSPACE, SCALAR, OFFSET, ORDINAL>,             \
      KokkosSparse::CrsMatrix<const SCALAR, const ORDINAL, Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                   \
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>, const OFFSET>,                                \
      Kokkos::View<const SCALAR**, XL, Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                                       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>,                                  \
      Kokkos::View<SCALAR**, YL, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>> { \
    enum : bool { value = true };                                                                                    \
  };

/* cusparseSpMM also produces incorrect results for some inputs in CUDA 11.6.1.
 * (CUSPARSE_VERSION 11702).
 * ALG1 and ALG3 produce completely incorrect results for one set of inputs.
 * ALG2 works for that case, but has low numerical accuracy in another case.
 */
#if defined(CUSPARSE_VERSION) && (CUSPARSE_VERSION != 11702)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(double, int, int, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(double, int, int, Kokkos::LayoutRight, Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)

KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(float, int, int, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(float, int, int, Kokkos::LayoutRight, Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)

KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<double>, int, int, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<double>, int, int, Kokkos::LayoutRight, Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)

KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<float>, int, int, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<float>, int, int, Kokkos::LayoutRight, Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)

KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(double, int, int, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(double, int, int, Kokkos::LayoutRight, Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)

KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(float, int, int, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(float, int, int, Kokkos::LayoutRight, Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)

KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<double>, int, int, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<double>, int, int, Kokkos::LayoutRight, Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)

KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<float>, int, int, Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<float>, int, int, Kokkos::LayoutRight, Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)

#if defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::Experimental::half_t, int, int, Kokkos::LayoutLeft,
                                             Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::Experimental::half_t, int, int, Kokkos::LayoutRight,
                                             Kokkos::LayoutLeft, Kokkos::CudaSpace)

KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::Experimental::half_t, int, int, Kokkos::LayoutLeft,
                                             Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::Experimental::half_t, int, int, Kokkos::LayoutRight,
                                             Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)

#endif
#endif  // defined(CUSPARSE_VERSION) && (CUSPARSE_VERSION != 11702)
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
#define KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_ROCSPARSE(SCALAR, XL, YL, MEMSPACE)                                     \
  template <>                                                                                                       \
  struct spmv_mv_tpl_spec_avail<                                                                                    \
      Kokkos::HIP, KokkosSparse::Impl::SPMVHandleImpl<Kokkos::HIP, MEMSPACE, SCALAR, rocsparse_int, rocsparse_int>, \
      KokkosSparse::CrsMatrix<const SCALAR, const rocsparse_int, Kokkos::Device<Kokkos::HIP, MEMSPACE>,             \
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>, const rocsparse_int>,                        \
      Kokkos::View<const SCALAR**, XL, Kokkos::Device<Kokkos::HIP, MEMSPACE>,                                       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>,                                 \
      Kokkos::View<SCALAR**, YL, Kokkos::Device<Kokkos::HIP, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>> { \
    enum : bool { value = true };                                                                                   \
  };

#define AVAIL_ROCSPARSE_SCALAR_MEMSPACE(SCALAR, MEMSPACE)                                                  \
  KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_ROCSPARSE(SCALAR, Kokkos::LayoutLeft, Kokkos::LayoutLeft, MEMSPACE)  \
  KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_ROCSPARSE(SCALAR, Kokkos::LayoutLeft, Kokkos::LayoutRight, MEMSPACE) \
  KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_ROCSPARSE(SCALAR, Kokkos::LayoutRight, Kokkos::LayoutLeft, MEMSPACE) \
  KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_ROCSPARSE(SCALAR, Kokkos::LayoutRight, Kokkos::LayoutRight, MEMSPACE)

#define AVAIL_ROCSPARSE_SCALAR(SCALAR)                      \
  AVAIL_ROCSPARSE_SCALAR_MEMSPACE(SCALAR, Kokkos::HIPSpace) \
  AVAIL_ROCSPARSE_SCALAR_MEMSPACE(SCALAR, Kokkos::HIPManagedSpace)

AVAIL_ROCSPARSE_SCALAR(float)
AVAIL_ROCSPARSE_SCALAR(double)
AVAIL_ROCSPARSE_SCALAR(Kokkos::complex<float>)
AVAIL_ROCSPARSE_SCALAR(Kokkos::complex<double>)

#undef AVAIL_ROCSPARSE_SCALAR_MEMSPACE
#undef AVAIL_ROCSPARSE_SCALAR
#undef KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_ROCSPARSE

#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
#define KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_MKL(SCALAR, EXECSPACE)                                              \
  template <>                                                                                                \
  struct spmv_tpl_spec_avail<                                                                                \
      EXECSPACE, KokkosSparse::Impl::SPMVHandleImpl<EXECSPACE, Kokkos::HostSpace, SCALAR, MKL_INT, MKL_INT>, \
      KokkosSparse::CrsMatrix<const SCALAR, const MKL_INT, Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,     \
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>, const MKL_INT>,                       \
      Kokkos::View<const SCALAR**, Kokkos::LayoutLeft, Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,         \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>,                          \
      Kokkos::View<SCALAR**, Kokkos::LayoutLeft, Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,               \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>> {                                               \
    enum : bool { value = true };                                                                            \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_MKL(float, Kokkos::Serial)
KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_MKL(double, Kokkos::Serial)
KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_MKL(Kokkos::complex<float>, Kokkos::Serial)
KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_MKL(Kokkos::complex<double>, Kokkos::Serial)
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_MKL(float, Kokkos::OpenMP)
KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_MKL(double, Kokkos::OpenMP)
KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_MKL(Kokkos::complex<float>, Kokkos::OpenMP)
KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_MKL(Kokkos::complex<double>, Kokkos::OpenMP)
#endif

#if defined(KOKKOS_ENABLE_SYCL)
#define KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_ONEMKL(SCALAR, ORDINAL, MEMSPACE)                                       \
  template <>                                                                                                    \
  struct spmv_tpl_spec_avail<                                                                                    \
      Kokkos::Experimental::SYCL,                                                                                \
      KokkosSparse::Impl::SPMVHandleImpl<Kokkos::Experimental::SYCL, MEMSPACE, SCALAR, ORDINAL, ORDINAL>,        \
      KokkosSparse::CrsMatrix<const SCALAR, const ORDINAL, Kokkos::Device<Kokkos::Experimental::SYCL, MEMSPACE>, \
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>, const ORDINAL>,                           \
      Kokkos::View<const SCALAR**, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Experimental::SYCL, MEMSPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>,                              \
      Kokkos::View<SCALAR**, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Experimental::SYCL, MEMSPACE>,           \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>> {                                                   \
    enum : bool { value = true };                                                                                \
  };

// intel-oneapi-mkl/2023.2.0: spmv with complex data types produce:
// oneapi::mkl::sparse::gemv: unimplemented functionality: currently does not
// support complex data types.
// TODO: Revisit with later versions and selectively enable this if it's
// working.

KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_ONEMKL(float, std::int32_t, Kokkos::Experimental::SYCLDeviceUSMSpace)
KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_ONEMKL(double, std::int32_t, Kokkos::Experimental::SYCLDeviceUSMSpace)
/*
KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_ONEMKL(
    Kokkos::complex<float>, std::int32_t,
    Kokkos::Experimental::SYCLDeviceUSMSpace)
KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_ONEMKL(
    Kokkos::complex<double>, std::int32_t,
    Kokkos::Experimental::SYCLDeviceUSMSpace)
*/

KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_ONEMKL(float, std::int64_t, Kokkos::Experimental::SYCLDeviceUSMSpace)
KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_ONEMKL(double, std::int64_t, Kokkos::Experimental::SYCLDeviceUSMSpace)
/*
KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_ONEMKL(
    Kokkos::complex<float>, std::int64_t,
    Kokkos::Experimental::SYCLDeviceUSMSpace)
KOKKOSSPARSE_SPMV_TPL_SPEC_AVAIL_ONEMKL(
    Kokkos::complex<double>, std::int64_t,
    Kokkos::Experimental::SYCLDeviceUSMSpace)
*/
#endif

#endif  // KOKKOSKERNELS_ENABLE_TPL_MKL

}  // namespace Impl
}  // namespace KokkosSparse

#endif  // KOKKOSPARSE_SPMV_MV_TPL_SPEC_AVAIL_HPP_
