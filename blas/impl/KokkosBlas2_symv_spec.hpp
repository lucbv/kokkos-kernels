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
#ifndef KOKKOSBLAS2_SYMV_SPEC_HPP_
#define KOKKOSBLAS2_SYMV_SPEC_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_InnerProductSpaceTraits.hpp"

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include <KokkosBlas2_symv_impl.hpp>
#endif

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class ExecutionSpace, class XMV, class YMV, class ZMV>
struct symv_eti_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

//
// Macro for declaration of full specialization availability
// KokkosBlas::Impl::SYMV.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _INST macro below) across one or
// more .cpp files.
//
#define KOKKOSBLAS2_SYMV_ETI_SPEC_AVAIL(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  template <>                                                                  \
  struct symv_eti_spec_avail<                                                  \
      EXEC_SPACE,                                                              \
      Kokkos::View<const SCALAR**, LAYOUT,                                     \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const SCALAR*, LAYOUT,                                      \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {               \
    enum : bool { value = true };                                              \
  };

// Include the actual specialization declarations
#include <KokkosBlas2_symv_tpl_spec_avail.hpp>
#include <generated_specializations_hpp/KokkosBlas2_symv_eti_spec_avail.hpp>

namespace KokkosBlas {
namespace Impl {

//
// symv
//

// Implementation of KokkosBlas::symv.
template <
    class ExecutionSpace, class AViewType, class XViewType, class YViewType,
    bool tpl_spec_avail = symv_tpl_spec_avail<ExecutionSpace, AViewType,
                                              XViewType, YViewType>::value,
    bool eti_spec_avail = symv_eti_spec_avail<ExecutionSpace, AViewType,
                                              XViewType, YViewType>::value>
struct SYMV {
  static void symv(const ExecutionSpace& space, const char trans[],
                   typename AViewType::const_value_type& alpha,
                   const AViewType& A, const XViewType& x,
                   typename YViewType::const_value_type& beta,
                   const YViewType& y)
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
  {
    static_assert(Kokkos::is_view<AViewType>::value,
                  "SYMV: AViewType must be a Kokkos::View.");
    static_assert(Kokkos::is_view<XViewType>::value,
                  "SYMV: XViewType must be a Kokkos::View.");
    static_assert(Kokkos::is_view<YViewType>::value,
                  "SYMV: YViewType must be a Kokkos::View.");
    static_assert(static_cast<int>(AViewType::rank) == 2,
                  "SYMV: AViewType must have rank 1.");
    static_assert(static_cast<int>(XViewType::rank) == 1,
                  "SYMV: XViewType must have rank 1.");
    static_assert(static_cast<int>(YViewType::rank) == 1,
                  "SYMV: YViewType must have rank 1.");
    Kokkos::Profiling::pushRegion(KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
                                      ? "KokkosBlas::symv[ETI]"
                                      : "KokkosBlas::symv[noETI]");
    typedef typename AViewType::size_type size_type;
    const size_type numRows = A.extent(0);

    // Prefer int as the index type, but use a larger type if needed.
    if (numRows < static_cast<size_type>(INT_MAX)) {
      generalSymvImpl<AViewType, XViewType, YViewType, int>(space, trans, alpha,
                                                            A, x, beta, y);
    } else {
      generalSymvImpl<AViewType, XViewType, YViewType, int64_t>(
          space, trans, alpha, A, x, beta, y);
    }
    Kokkos::Profiling::popRegion();
  }
#else
      ;
#endif  //! defined(KOKKOSKERNELS_ETI_ONLY) ||
        //! KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
};

}  // namespace Impl
}  // namespace KokkosBlas

//
// Macro for declaration of full specialization of
// KokkosBlas::Impl::SYMV.  This is NOT for users!!!
// All the declarations of full specializations go in this header
// file.  We may spread out definitions (see _DEF macro below) across
// one or more .cpp files.
//

#define KOKKOSBLAS2_SYMV_ETI_SPEC_DECL(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  extern template struct SYMV<                                                \
      EXEC_SPACE,                                                             \
      Kokkos::View<const SCALAR**, LAYOUT,                                    \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<const SCALAR*, LAYOUT,                                     \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      false, true>;

#define KOKKOSBLAS2_SYMV_ETI_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  template struct SYMV<                                                       \
      EXEC_SPACE,                                                             \
      Kokkos::View<const SCALAR**, LAYOUT,                                    \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<const SCALAR*, LAYOUT,                                     \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      false, true>;

#include <KokkosBlas2_symv_tpl_spec_decl.hpp>

#endif  // KOKKOSBLAS2_SYMV_SPEC_HPP_
