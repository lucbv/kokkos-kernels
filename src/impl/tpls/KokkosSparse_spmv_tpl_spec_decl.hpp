/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOSPARSE_SPMV_TPL_SPEC_DECL_HPP_
#define KOKKOSPARSE_SPMV_TPL_SPEC_DECL_HPP_

namespace KokkosSparse {
namespace Impl {

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
template<>
struct SPMV<double const,  int const, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Unmanaged>, int,
            double const*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Unmanaged|Kokkos::RandomAccess>,
            double*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, false> {
  using AMatrix = CrsMatrix<double const, int const, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Unmanaged>, int const>;
  using XVector = Kokkos::View<double const*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Unmanaged|Kokkos::RandomAccess> >;
  using YVector = Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, false>;

  using coefficient_type = typename YVector::non_const_value_type;

  static void spmv (const char mode[],
                    const coefficient_type& alpha,
                    const AMatrix& A,
                    const XVector& x,
                    const coefficient_type& beta,
                    const YVector& y) {
    printf("Using the fake cuSPARSE spmv tpl specialization!\n");
  }
};

template<>
struct SPMV<double const,  int const, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Unmanaged>, int const,
            double const*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Unmanaged|Kokkos::RandomAccess>,
            double*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, true> {
  using AMatrix = CrsMatrix<double const, int const, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Unmanaged>, int const>;
  using XVector = Kokkos::View<double const*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Unmanaged|Kokkos::RandomAccess> >;
  using YVector = Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, true>;

  using coefficient_type = typename YVector::non_const_value_type;

  static void spmv (const char mode[],
                    const coefficient_type& alpha,
                    const AMatrix& A,
                    const XVector& x,
                    const coefficient_type& beta,
                    const YVector& y) {
    printf("Using the fake cuSPARSE spmv tpl specialization!\n");
  }
};
#endif // KOKKOSKERNELS_ENABLE_TPL_CUSPARSE

}
}

#endif // KOKKOSPARSE_SPMV_TPL_SPEC_DECL_HPP_
