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

#ifndef __KOKKOSBATCHED_ODE_NEWTON_HPP__
#define __KOKKOSBATCHED_ODE_NEWTON_HPP__

#include "Kokkos_Core.hpp"
#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_LU_Serial_Impl.hpp"
#include "KokkosBatched_Scale_Decl.hpp"
#include "KokkosBatched_Gesv.hpp"
#include "KokkosBlasDevice_nrm2_impl.hpp"
#include "KokkosBlasDevice_axpy_impl.hpp"
#include "KokkosBlasDevice_scale_impl.hpp"

enum class NewtonSolverStatus { Converged = 0, LinearSolveFailure, MaxIters };

std::ostream& operator<<(std::ostream& os, NewtonSolverStatus& status) {
switch (status) {
 case NewtonSolverStatus::Converged: os << "Newton Solver Converged!"; break;
 case NewtonSolverStatus::LinearSolveFailure:
os << "Newton: Linear Solver Failure";
break;
 case NewtonSolverStatus::MaxIters:
os << "Newton reached maximum iterations without convergence.";
break;
}
return os;
}

/// \brief NewtonHandle
///
/// This handle is used to pass information between the Newton Solver and
/// the calling code.
///
/// \tparam: NormViewType: Type of view used to store the residual convergence
/// history

template <class NormViewType>
struct NewtonHandle {
public:
using norm_type = typename NormViewType::non_const_value_type;
NormViewType lastResidual;  // Residual of last successful iteration
typename NormViewType::HostMirror lastResidualHost;

// NormViewType  residual_norms;
// TODO: Making these public for now. Should make private and access
// via setters and getters?
int maxIters;           // Maximum number of Newton steps
norm_type relativeTol;  // Relative convergence tolerance
bool debug_mode;        // Returns extra verbose output if true.

public:
NewtonHandle(int _maxIters = 25, double _relativeTol = 1.0e-6,
               bool _debug = false)
  : lastResidual("ending Residual norm", 1),
    lastResidualHost("end res norm host", 1),
    maxIters(_maxIters),
    relativeTol(_relativeTol),
    debug_mode(_debug) {}

  norm_type get_end_residual() const {
Kokkos::deep_copy(lastResidualHost, lastResidual);
return lastResidualHost(0);
}
    };

/// \brief Newton Functor:
/// Solves the nonlinear system F(x) = 0
/// where F is a map from R^n to R^m.
/// \tparam LHSFunc: Functor on (J,x) where matrix J is output with
/// the value of the Jacobian evaluated at x.
/// \tparam RHSFunc: Functor on (y,x) where y is output with the value
/// of F at x.
/// \tparam Matrix: 3D view-type of LHSFunc output (extent(0) = 1)
/// \tparam XVector: 2D view-type (extent(0) = 1)
/// \tparam YVector: 2D view-type (extent(0) = 1)
/// \param
/// \param X [in]: Input vector X, a rank 2 view
/// \param Y [in/out]: Output vector Y, a rank 2 view
///
/// No nested parallel_for is used inside of the function.
///
template <class System, class Matrix, class XVector,
          class YVector, class NewtonHandleType>
struct NewtonFunctor {
using execution_space = typename YVector::execution_space;
using yvalue_type     = typename YVector::non_const_value_type;
using norm_type       = typename NewtonHandleType::norm_type;

  System sys;
  XVector x;
  YVector rhs;
  NewtonHandleType handle;
  int N;

  Matrix J, tmp;
  XVector update;
  Kokkos::View<yvalue_type*, execution_space> norm, alpha;

  NewtonFunctor(System _sys, XVector _x, YVector _rhs,
                  NewtonHandleType& _handle)
    : sys(_sys),
      x(_x),
      rhs(_rhs),
      handle(_handle) {
    J      = Matrix("Jacobian", x.extent(0), x.extent(0));
    tmp    = Matrix("Jacobian", x.extent(0), x.extent(0) + 4);
    update = XVector("update", x.extent(0));
    norm   = Kokkos::View<norm_type*, execution_space>("Newton norm", 1);
    alpha  = Kokkos::View<yvalue_type*, execution_space>("alpha", 1);
    N      = x.extent(0);
  }

  KOKKOS_INLINE_FUNCTION
  NewtonSolverStatus operator()(int /* idx */) const {
    alpha(0)               = Kokkos::ArithTraits<yvalue_type>::one();
    handle.lastResidual(0) = -1;  // init to dummy value

    // Iterate until maxIts or the tolerance is reached
    for (int it = 0; it < handle.maxIters; ++it) {
      // compute initial rhs
      sys.derivatives(0, x, rhs);

      // Solve the following linearized
      // problem at each step: J*update=rhs
      // with J=du/dx, rhs=f(u_n+update)-f(u_n)
      KokkosBlas::Experimental::Device::SerialNrm2::invoke(rhs, norm);
      handle.lastResidual(0) = norm(0);

      if (handle.debug_mode) {
        KOKKOS_IMPL_DO_NOT_USE_PRINTF(
            "NewtonFunctor: Iteration: %d  Current res norm is: %e \n Current "
            "soln is:\n",
            // it, norm(0));
            it, (double)handle.lastResidual(0));
        for (int k = 0; k < N; k++) {
          KOKKOS_IMPL_DO_NOT_USE_PRINTF("%f \n", x(k));
          // KOKKOS_IMPL_DO_NOT_USE_PRINTF("%f \n", x(0, k));
        }
      }

      if (norm(0) < handle.relativeTol) {
        // Problem solved, exit the functor
        if (handle.debug_mode) {
          KOKKOS_IMPL_DO_NOT_USE_PRINTF(
              "NewtonFunctor: Newton solver converged! Ending norm is: %e \n "
              "Solution x is: "
              "\n",
              norm(0));
          for (int k = 0; k < N; k++) {
            KOKKOS_IMPL_DO_NOT_USE_PRINTF("%f \n", x(k));
            // KOKKOS_IMPL_DO_NOT_USE_PRINTF("%f \n", x(0, k));
          }
        }
        return NewtonSolverStatus::Converged;
      }

      // compute LHS
      sys.jacobian(0, x, J);

      // solve linear problem
      int linSolverStat = KokkosBatched::SerialGesv<
        KokkosBatched::Gesv::StaticPivoting>::invoke(J, update, rhs, tmp);
      KokkosBlas::Experimental::Device::SerialScale::invoke(-1, update);

      // update state of variables if needed
      sys.update_state(update);

      if (handle.debug_mode) {
        KOKKOS_IMPL_DO_NOT_USE_PRINTF(
                                      "NewtonFunctor: Print linear solve solution: \n");
        for (int k = 0; k < N; k++) {
          KOKKOS_IMPL_DO_NOT_USE_PRINTF("%f \n", -update(k));
        }
      }
      if (linSolverStat == 1) {
        KOKKOS_IMPL_DO_NOT_USE_PRINTF(
                                      "NewtonFunctor: Linear solve gesv returned failure! \n");
        return NewtonSolverStatus::LinearSolveFailure;
      }

      // update solution // x = x + alpha*update
      KokkosBlas::Experimental::Device::SerialAxpy::invoke(alpha, update, x);
    }
    return NewtonSolverStatus::MaxIters;
  }  // End solve functor.
};

#endif // __KOKKOSBATCHED_ODE_NEWTON_HPP__
