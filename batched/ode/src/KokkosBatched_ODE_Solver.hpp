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

#ifndef __KOKKOSBATCHED_ODE_SOLVER_HPP__
#define __KOKKOSBATCHED_ODE_SOLVER_HPP__

#include "KokkosBatched_ODE_RungeKuttaTables.hpp"
#include "KokkosBatched_ODE_Args.hpp"
#include "KokkosBatched_ODE_RKSolve.hpp"

namespace KokkosBatched {
namespace Experimental {
namespace ODE {

enum class ODE_solver_type : unsigned {
  RK    = 0,
  BDFS  = 1,
  ADAMS = 2
};


// clang-format off
/// \brief Generic interface for the ODESolvers
// clang-format on
template <ODE_solver_type type>
struct ODESolver {

  KOKKOS_FUNCTION static void invoke() {
    // Can switch to a Kokkos::abort("msg") call
    printf("This ODE solver type is not supported!\n");
  }

};

template <>
struct ODESolver<ODE_solver_type::RK> {

  template <typename ODEType, typename vec_view, typename stack_type>
  KOKKOS_FUNCTION static ODESolverStatus invoke(ODEType& ode, ODEArgs& args,
                                                vec_view& y, vec_view& y0,
                                                vec_view& f, vec_view& ytmp,
                                                stack_type& stack,
                                                const double t0,
                                                const double tn) {

    ButcherTableau<1, 1> table;
    return SerialRKSolve::invoke(table, ode, args, y, y0, f, ytmp, stack, t0, tn);
  }

};

} // namespace ODE
} // namespace Experimental
} // namespace KokkosBatched

#include "KokkosBatched_ODE_BDFS_Impl.hpp"

#endif // __KOKKOSBATCHED_ODE_SOLVER_HPP__
