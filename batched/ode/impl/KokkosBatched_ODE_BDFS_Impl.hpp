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

// BDF1 or Backward Euler
// is the simplest implicity
// time integrator. It uses
// the following formula:
// y_{n+1}-y_n=dt*f(t_{n+1}, y_{n+1})
template<typename ODEType>
struct BDF1_system {
  using scalar_type = typename ODEType::scalar_type;
  using vec_type    = typename ODEType::vec_type;
  using mat_type    = typename ODEType::mat_type;

  ODEType ode;
  vec_type y_n;
  scalar_type t, dt;

  BDF1_system(ODEType ode_, const scalar_type t_, const scalar_type dt_)
    : ode(ode_), y_n(vec_type("history", ode.num_equations())), t(t_), dt(dt_) {};

  KOKKOS_FUNCTION void jacobian(const scalar_type /*t*/,
				const vec_type& y,
                                const mat_type& jac) const {
    // For BDF1 the Jacobian is computed as:
    // J = I - dt*ode.jacobian()

    ode.jacobian(t, y, jac);
    for(int i=0; i < static_cast<int>(jac.extent(0)); ++i) {
      for(int j=0; j < static_cast<int>(jac.extent(1)); ++j) {
        jac(i,j) = -dt*jac(i,j);
        if(i==j) {
          jac(i,j) += static_cast<scalar_type>(1);
        }
      }
    }
  }

  KOKKOS_FUNCTION void derivatives(const scalar_type /*t*/,
                                   const vec_type& y,
				   const vec_type& r) const {

    // For BDF1 R= y - y_n - dt*(dy/dt)
    ode.derivatives(t, y, r);
    for(int i=0; i < static_cast<int>(r.extent(0)); ++i) {
      r(i) = y(i) - y_n(i) - dt*r(i);
    }
  }

  KOKKOS_FUNCTION void set_history(const vec_type& hist) const {
    Kokkos::deep_copy(y_n, hist);
  }
};

// BDF1  uses the following formula:
// y_{n+2}-4/3*y_{n+1}+1/3*y_n=2/3*dt*f(t_{n+1}, y_{n+1})
template<typename ODEType>
struct BDF2_system {
  using scalar_type = typename ODEType::scalar_type;
  using vec_type    = typename ODEType::vec_type;
  using mat_type    = typename ODEType::mat_type;

  Kokkos::Array<scalar_type, 3> coeffs{{-4.0/3.0, 1.0/3.0, 2.0/3.0}};

  ODEType ode;
  vec_type y_n;
  scalar_type t, dt;

  BDF2_system(ODEType ode_, const scalar_type t_, const scalar_type dt_)
    : ode(ode_), y_n(vec_type("history", ode.num_equations())), t(t_), dt(dt_) {};

  KOKKOS_FUNCTION void jacobian(const scalar_type /*t*/,
				const vec_type& y,
                                const mat_type& jac) const {
    // For BDF2 the Jacobian is computed as:
    // J = I - 2/3*dt*ode.jacobian()

    ode.jacobian(t, y, jac);
    for(int i=0; i < static_cast<int>(jac.extent(0)); ++i) {
      for(int j=0; j < static_cast<int>(jac.extent(1)); ++j) {
        jac(i,j) = -dt*static_cast<scalar_type>(2.0/3.0)*jac(i,j);
        if(i==j) {
          jac(i,j) += static_cast<scalar_type>(1);
        }
      }
    }
  }

  KOKKOS_FUNCTION void derivatives(const scalar_type /*t*/,
                                   const vec_type& y,
				   const vec_type& r) const {

    // For BDF1 R= y -4/3*y_{n+1} + 1/3*y_n - dt*2/3*(dy/dt)
    ode.derivatives(t, y, r);
    for(int i=0; i < static_cast<int>(r.extent(0)); ++i) {
      r(i) = y(i) - 4.0/3.0*y_n(i,0) + 1.0/3.0*y_n(i,0) - 2.0/3.0*dt*r(i);
    }
  }

  KOKKOS_FUNCTION void set_history(const vec_type& hist) const {
    Kokkos::deep_copy(y_n, hist);
  }
};

template <>
struct ODESolver<ODE_solver_type::BDFS> {

  template <typename ODEType>
  KOKKOS_FUNCTION static ODESolverStatus invoke(ODEType& ode, ODEArgs& args,
                                                typename ODEType::vec_type& y,
                                                typename ODEType::vec_type& y0,
                                                typename ODEType::vec_type& f,
                                                typename ODEType::vec_type& /* ytmp */,
                                                typename ODEType::scalar_type const t0,
                                                typename ODEType::scalar_type const tn) {
    using execution_space = typename ODEType::execution_space;
    using scalar_type     = typename ODEType::scalar_type;
    using vec_type        = typename ODEType::vec_type;
    using mat_type        = typename ODEType::mat_type;
    using norm_type       = Kokkos::View<typename Kokkos::ArithTraits<scalar_type>::mag_type*, execution_space>;
    using handle_type     = NewtonHandle<norm_type>;

    scalar_type t  = t0;
    scalar_type dt = (tn - t0) / args.num_substeps;
    Kokkos::deep_copy(y, y0);
    printf("Solution at t=0, y=%g\n", y(0));

    for(int step = 0; step < args.num_substeps; ++step) {
      // Create a BDF1_system that is in charge
      // of computing the residual and jacobian
      // needed by the Newton solver.
      BDF1_system sys(ode, t, dt);
      sys.set_history(y);

      handle_type handle;
      handle.debug_mode = false;
      NewtonFunctor<BDF1_system<ODEType>, mat_type, vec_type, vec_type, handle_type>
        newton(sys, y, f, handle);
      newton(0);

      t += dt;
      printf("Solution at t=%f, y=%g\n", t, y(0));
    }

    return ODESolverStatus::SUCCESS;
  }

}; // ODESolver for BDFS

}
}
}

#endif // __KOKKOSBATCHED_ODE_BDFS_IMPL_HPP__
