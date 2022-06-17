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

#ifndef __KOKKOSBATCHED_ODE_BDFS_IMPL_HPP__
#define __KOKKOSBATCHED_ODE_BDFS_IMPL_HPP__

#include "KokkosBatched_ODE_Newton.hpp"

namespace KokkosBatched {
namespace Experimental {
namespace ODE {

template<typename ODEType>
struct BDF1_system {
  using scalar_type = typename ODEType::scalar_type;
  using vec_type    = typename ODEType::vec_type;
  using mat_type    = typename ODEType::mat_type;

  ODEType ode;
  scalar_type t, dt;

  BDF1_system(ODEType ode_, const scalar_type t_, const scalar_type dt_)
    : ode(ode_), t(t_), dt(dt_) {};

  KOKKOS_FUNCTION void jacobian(const vec_type y,
                                mat_type jac) const {
    // For BDF1 the Jacobian is computed as:
    // J = I - dt*ode.jacobian()

    ode.jacobian(t, y, jac);
    for(int i=0; i < jac.extent(0); ++i) {
      for(int j=0; j < jac.extent(1); ++j) {
        jac(i,j) = -dt*jac(i,j);
        if(i==j) {
          jac(i,j) += static_cast<scalar_type>(1);
        }
      }
    }
  }

  KOKKOS_FUNCTION void residual(const vec_type y, const vec_type y_n,
                                vec_type& r) const {

    // For BDF1 R= y - y_n - dt*(dy/dt)
    ode.derivatives(t, y, r);
    for(int i=0; i < r.extent(0); ++i) {
      r(i) = y(i) - y_n(i) - dt*r(i);
    }
  }
};

template <>
struct ODESolver<ODE_solver_type::BDFS> {

  template <typename ODEType, typename stack_type>
  KOKKOS_FUNCTION static ODESolverStatus invoke(ODEType& ode, ODEArgs& args,
                                                typename ODEType::vec_view& y,
                                                typename ODEType::vec_view& y0,
                                                typename ODEType::vec_view& f,
                                                typename ODEType::vec_view& ytmp,
                                                stack_type& stack,
                                                typename ODEType::scalar_type const t0,
                                                typename ODEType::scalar_type const tn) {
    using scalar_type = typename ODEType::scalar_type;
    using vec_type    = typename ODEType::vec_type;
    using mat_type    = typename ODEType::mat_type;
    using norm_type   = typename Kokkos::ArithTraits<scalar_type>::mag_type;
    using handle_type = NewtonHandle<norm_type>;

    scalar_type t  = t0;
    scalar_type dt = (tn - t0) / args.num_substeps;
    Kokkos::deep_copy(y, y0);

    for(int step = 0; step < args.num_substeps; ++step) {
      // Create a BDF1_system that is in charge
      // of computing the residual and jacobian
      // needed by the Newton solver.
      BDF1_system sys(ode, t, dt);

      handle_type handle;
      NewtonFunctor<BDF1_system<ODEType>, mat_type, vec_type, vec_type, handle_type>
        newton(sys, y, f, handle);

      t += dt;
    }

    return ODESolverStatus::SUCCESS;
  }

}; // ODESolver for BDFS

}
}
}

#endif // __KOKKOSBATCHED_ODE_BDFS_IMPL_HPP__
