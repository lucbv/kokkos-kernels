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

#ifndef __KOKKOSBATCHED_ODE_NEWTON_HPP__
#define __KOKKOSBATCHED_ODE_NEWTON_HPP__

#include "Kokkos_Core.hpp"
#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_LU_Serial_Impl.hpp"
#include "KokkosBatched_Gesv.hpp"
#include "KokkosBlas1_nrm2.hpp"
#include "KokkosBlas1_scal.hpp"
#include "KokkosBlas1_axpby.hpp"

namespace KokkosBlas {
namespace Impl {

/// \brief Newton Functor:
/// Solves the nonlinear system F(x) = 0
/// where F is a map from R^n to R^n.
/// \tparam System: Struct that allows the evaluation
///         of the residual and jacobian using the
///         residual() and jacobian() methods.
/// \tparam Matrix: rank-2 view-type
/// \tparam XVector: rank-1 view-type
/// \tparam YVector: rank-1 view-type
/// \param
/// \param X [in]: Input vector X, a rank 1 view
/// \param Y [in/out]: Output vector Y, a rank 1 view
///
/// No nested parallel_for is used inside of the function.
///
template <class System, class Matrix, class XVector, class YVector>
struct NewtonFunctor {
  using execution_space = typename YVector::execution_space;
  using yvalue_type     = typename YVector::non_const_value_type;
  using KAT_yvalue      = typename Kokkos::ArithTraits<yvalue_type>;
  using norm_type       = typename KAT_yvalue::mag_type;
  using KAT_norm        = typename Kokkos::ArithTraits<norm_type>;

  System sys;
  XVector x;
  YVector rhs;

  Matrix J, tmp;
  XVector update;

  KOKKOS_FUNCTION
  NewtonFunctor(System _sys, XVector _x, YVector _rhs, Matrix J_, Matrix tmp_, XVector update_)
    : sys(_sys), x(_x), rhs(_rhs), J(J_), tmp(tmp_), update(update_) {}

  KOKKOS_INLINE_FUNCTION
  int solve(const int maxIters = 25, const norm_type relativeTol = 1e-6, const bool debug_mode = false) const {
    norm_type norm    = KAT_norm::zero();
    yvalue_type alpha = KAT_yvalue::one();
    norm_type residual_norm = -KAT_norm::one();

    // Iterate until maxIts or the tolerance is reached
    for (int it = 0; it < maxIters; ++it) {
      // compute initial rhs
      sys.residual(x, rhs);
      if (debug_mode) {
        KOKKOS_IMPL_DO_NOT_USE_PRINTF("NewtonFunctor: r=");
        for (int k = 0; k < rhs.extent_int(0); k++) {
          KOKKOS_IMPL_DO_NOT_USE_PRINTF("%f \n", rhs(k));
        }
      }

      // Solve the following linearized
      // problem at each step: J*update=-rhs
      // with J=du/dx, rhs=f(u_n+update)-f(u_n)
      norm = KokkosBlas::serial_nrm2(rhs);
      residual_norm = norm;

      if (debug_mode) {
        KOKKOS_IMPL_DO_NOT_USE_PRINTF(
            "NewtonFunctor: Iteration: %d  Current res norm is: %e \n Current "
            "soln is:\n",
            it, (double)residual_norm);
        for (int k = 0; k < x.extent_int(0); k++) {
          KOKKOS_IMPL_DO_NOT_USE_PRINTF("%f \n", x(k));
        }
      }

      if (norm < relativeTol) {
        // Problem solved, exit the functor
        if (debug_mode) {
          KOKKOS_IMPL_DO_NOT_USE_PRINTF(
              "NewtonFunctor: Newton solver converged! Ending norm is: %e \n "
              "Solution x is: "
              "\n",
              norm);
          for (int k = 0; k < x.extent_int(0); k++) {
            KOKKOS_IMPL_DO_NOT_USE_PRINTF("%f \n", x(k));
          }
        }
        return 0;
      }

      // compute LHS
      sys.jacobian(x, J);

      // solve linear problem
      int linSolverStat = KokkosBatched::SerialGesv<
          KokkosBatched::Gesv::StaticPivoting>::invoke(J, update, rhs, tmp);
      KokkosBlas::SerialScale::invoke(-1, update);

      if (debug_mode) {
        KOKKOS_IMPL_DO_NOT_USE_PRINTF(
            "NewtonFunctor: Print linear solve solution: \n");
        for (int k = 0; k < update.extent_int(0); k++) {
          KOKKOS_IMPL_DO_NOT_USE_PRINTF("%f \n", update(k));
        }
      }
      if (linSolverStat == 1) {
        KOKKOS_IMPL_DO_NOT_USE_PRINTF(
            "NewtonFunctor: Linear solve gesv returned failure! \n");
        return -1;
      }

      // update solution // x = x + alpha*update
      KokkosBlas::serial_axpy(alpha, update, x);
      if (debug_mode) {
        KOKKOS_IMPL_DO_NOT_USE_PRINTF(
            "NewtonFunctor: Print updated solution: \n");
        for (int k = 0; k < x.extent_int(0); k++) {
          KOKKOS_IMPL_DO_NOT_USE_PRINTF("%f \n", x(k));
        }
      }
    }
    return maxIters;
  }  // End solve functor.
};

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // __KOKKOSBATCHED_ODE_NEWTON_HPP__
