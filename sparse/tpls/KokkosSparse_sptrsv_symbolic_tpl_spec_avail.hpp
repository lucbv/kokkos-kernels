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

#ifndef KOKKOSPARSE_SPTRSV_SYMBOLIC_TPL_SPEC_AVAIL_HPP_
#define KOKKOSPARSE_SPTRSV_SYMBOLIC_TPL_SPEC_AVAIL_HPP_

namespace KokkosSparse {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class KernelHandle, class RowMapType, class EntriesType,
          class ValuesType>
struct sptrsv_symbolic_tpl_spec_avail {
  enum : bool { value = false };
};

// cuSPARSE
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE

#define KOKKOSSPARSE_SPTRSV_SYMBOLIC_TPL_SPEC_AVAIL_CUSPARSE(SCALAR, LAYOUT,   \
                                                             MEMSPACE)         \
  template <>                                                                  \
  struct sptrsv_symbolic_tpl_spec_avail<                                       \
      KokkosKernels::Experimental::KokkosKernelsHandle<                        \
          const int, const int, const SCALAR, Kokkos::Cuda, MEMSPACE,          \
          MEMSPACE>,                                                           \
      Kokkos::View<                                                            \
          const int *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>,         \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >,    \
      Kokkos::View<                                                            \
          const int *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>,         \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >,    \
      Kokkos::View<                                                            \
          const SCALAR *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>,      \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> > > { \
    enum : bool { value = true };                                              \
  };

KOKKOSSPARSE_SPTRSV_SYMBOLIC_TPL_SPEC_AVAIL_CUSPARSE(float, Kokkos::LayoutLeft,
                                                     Kokkos::CudaSpace)
KOKKOSSPARSE_SPTRSV_SYMBOLIC_TPL_SPEC_AVAIL_CUSPARSE(double, Kokkos::LayoutLeft,
                                                     Kokkos::CudaSpace)
KOKKOSSPARSE_SPTRSV_SYMBOLIC_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<float>,
                                                     Kokkos::LayoutLeft,
                                                     Kokkos::CudaSpace)
KOKKOSSPARSE_SPTRSV_SYMBOLIC_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<double>,
                                                     Kokkos::LayoutLeft,
                                                     Kokkos::CudaSpace)

KOKKOSSPARSE_SPTRSV_SYMBOLIC_TPL_SPEC_AVAIL_CUSPARSE(float, Kokkos::LayoutLeft,
                                                     Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPTRSV_SYMBOLIC_TPL_SPEC_AVAIL_CUSPARSE(double, Kokkos::LayoutLeft,
                                                     Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPTRSV_SYMBOLIC_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<float>,
                                                     Kokkos::LayoutLeft,
                                                     Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPTRSV_SYMBOLIC_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<double>,
                                                     Kokkos::LayoutLeft,
                                                     Kokkos::CudaUVMSpace)

#endif  // KOKKOSKERNELS_ENABLE_TPL_CUSPARSE

}  // namespace Impl
}  // namespace KokkosSparse

#endif
