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

#ifndef KOKKOSODE_NEWTON_IMPL_HPP
#define KOKKOSODE_NEWTON_IMPL_HPP

#include "Kokkos_Core.hpp"
#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_LU_Serial_Impl.hpp"
#include "KokkosBatched_Trsm_Decl.hpp"
#include "KokkosBatched_Copy_Decl.hpp"
#include "KokkosBatched_Gesv.hpp"
#include "KokkosBlas1_nrm2.hpp"
#include "KokkosBlas1_scal.hpp"
#include "KokkosBlas1_axpby.hpp"

#include "KokkosODE_Types.hpp"

namespace KokkosODE {
namespace Impl {

template <class system_type, class mat_type, class ini_vec_type, class rhs_vec_type, class update_type,
          class scale_type>
KOKKOS_FUNCTION KokkosODE::Experimental::newton_solver_status NewtonSolve(
    system_type& sys, const KokkosODE::Experimental::Newton_params& params, mat_type& J, mat_type& tmp,
    ini_vec_type& y0, rhs_vec_type& rhs, update_type& update, const scale_type& scale) {
  using newton_solver_status = KokkosODE::Experimental::newton_solver_status;
  using value_type           = typename ini_vec_type::non_const_value_type;

  // Define the type returned by nrm2 to store
  // the norm of the residual.
  using norm_type =
      typename Kokkos::Details::InnerProductSpaceTraits<typename ini_vec_type::non_const_value_type>::mag_type;
  sys.residual(y0, rhs);
  const norm_type norm0 = KokkosBlas::serial_nrm2(rhs);
  norm_type norm_old    = Kokkos::ArithTraits<norm_type>::zero();
  norm_type norm_new    = Kokkos::ArithTraits<norm_type>::zero();
  norm_type rate        = Kokkos::ArithTraits<norm_type>::zero();

  // LBV - 07/24/2023: for now assume that we take
  // a full Newton step. Eventually this value can
  // be computed using a line search algorithm to
  // improve convergence for difficult problems.
  const value_type alpha = Kokkos::ArithTraits<value_type>::one();

  // YVV: TODO give it its own buffer and split stuff out, for now just-reusing part
  // of the the static pivoting buffer
  auto dfdy = Kokkos::subview(tmp, Kokkos::ALL,
                              Kokkos::pair<int, int>(0, y0.extent_int(0)));

  // Iterate until maxIts or the tolerance is reached
  for (int it = 0; it < params.max_iters; ++it) {
    // compute initial rhs
    // sys.residual(y0, rhs);
    int lin_solver_stat = 0;

    // Solve the following linearized
    // problem at each iteration: J*update=-rhs
    // with J=du/dx, rhs=f(u_n+update)-f(u_n)

    // compute LHS
    sys.jacobian(y0, dfdy, J);

    { // solve linear problem
      // J = I - c * dfdy, re-use dfdy as much as possible!
      if (sys.compute_jac || sys.compute_dfdy) {
	// printf("...computing jac and LU factorization!\n");
	lin_solver_stat = KokkosBatched::SerialLU<
          KokkosBatched::Algo::Level3::Unblocked>::invoke(J);
	sys.compute_jac = false;
	sys.compute_dfdy = false;
      }
 
      // TODO partial pivoting, or make static pivoting more robust.
      if (lin_solver_stat == 0) {
	// copy rhs into update, update will modified in place
	lin_solver_stat =
          KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose,
                                    1>::invoke(rhs, update);
      }
                                    
      if (lin_solver_stat == 0) {
	lin_solver_stat = KokkosBatched::SerialTrsm<
          KokkosBatched::Side::Left, KokkosBatched::Uplo::Lower,
          KokkosBatched::Trans::NoTranspose, KokkosBatched::Diag::Unit,
          KokkosBatched::Algo::Level3::Unblocked>::invoke(1.0, J, update);
      }

      if (lin_solver_stat == 0) {
	lin_solver_stat = KokkosBatched::SerialTrsm<
          KokkosBatched::Side::Left, KokkosBatched::Uplo::Upper,
          KokkosBatched::Trans::NoTranspose, KokkosBatched::Diag::NonUnit,
          KokkosBatched::Algo::Level3::Unblocked>::invoke(1.0, J, update);
      }
      if (lin_solver_stat == 1) {
	Kokkos::printf("NewtonFunctor: Linear solve gesv returned failure! \n");
	return newton_solver_status::LIN_SOLVE_FAIL;
      }
    }

    // update solution // y0 = y0 - alpha * update
    for (int eqIdx = 0; eqIdx < sys.neqs; ++eqIdx) {
      y0(eqIdx) -= alpha * update(eqIdx);
    }

    // Compute rms norm of the scaled update
    norm_new = 0;
    for (int eqIdx = 0; eqIdx < sys.neqs; ++eqIdx) {
      norm_new += (update(eqIdx) * update(eqIdx)) / (scale(eqIdx) * scale(eqIdx));
    }
    norm_new = Kokkos::sqrt(norm_new / sys.neqs);

    constexpr double safety_factor = 0.1;

    if (it == 0 && norm_new < safety_factor) {
      return newton_solver_status::NLS_SUCCESS;
    }

    rate = (it > 0 && norm_old > Kokkos::ArithTraits<norm_type>::zero()) ? norm_new / norm_old : 1;

    const auto norm_k = KokkosBlas::serial_nrm2(rhs);

    if (it == 0) {

      sys.residual(y0, rhs);

      constexpr double relative_residual_tol = 1e-6;
      const bool small_relative_residual =
          norm_k < relative_residual_tol * norm0;

      if (small_relative_residual) {
        return newton_solver_status::NLS_SUCCESS;
      }
    } else if (rate * norm_new < safety_factor) {
      return newton_solver_status::NLS_SUCCESS;
    } else {
      // if it >=1, estimate if we will hit max iters based on current
      // rate
      const auto iters_left = params.max_iters - (it + 1);
      if (Kokkos::pow(rate, iters_left) * norm_new > safety_factor) {
        return newton_solver_status::MAX_ITER;
      }
    }

    // we already updated for it == 0 check don't waste another update..
    // this also avoids updating if we aren't going to converge...
    if (it != 0){
      sys.residual(y0, rhs);
    }

    norm_old = norm_new;
  }
  return newton_solver_status::MAX_ITER;
}

}  // namespace Impl
}  // namespace KokkosODE

#endif  // KOKKOSODE_NEWTON_IMPL_HPP
