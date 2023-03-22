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

#ifndef KOKKOSBLAS_BDF_IMPL_HPP
#define KOKKOSBLAS_BDF_IMPL_HPP

#include "Kokkos_Core.hpp"

namespace KokkosBlas {
namespace Impl {

template <int order>
struct BDFTable{};

template <>
struct BDFTable<1>{
  static constexpr int order = 1;

  static constexpr double coeff_f = 1;
  Kokkos::Array<double, 1> coeffs{{-1}};
};

template <>
struct BDFTable<2>{
  static constexpr int order = 2;

  static constexpr double coeff_f = 2.0 / 3.0;
  Kokkos::Array<double, 2> coeffs{{-4.0 / 3.0, 1.0 / 3.0}};
};

template <>
struct BDFTable<3>{
  static constexpr int order = 3;

  static constexpr double coeff_f = 6.0 / 11.0;
  Kokkos::Array<double, 3> coeffs{{-18.0 / 11.0, 9.0 / 11.0, -2.0 / 11.0}};
};

template <>
struct BDFTable<4>{
  static constexpr int order = 4;

  static constexpr double coeff_f = 12.0 / 25.0;
  Kokkos::Array<double, 4> coeffs{{-48.0 / 25.0, 36.0 / 25.0, -16.0 / 25.0, 3.0 / 25.0}};
};

template <>
struct BDFTable<5>{
  static constexpr int order = 5;

  static constexpr double coeff_f = 60.0 / 137.0;
  Kokkos::Array<double, 5> coeffs{{-300.0 / 137.0, 300.0 / 137.0, -200.0 / 137.0, 75.0 / 137.0, -12.0 / 137.0}};
};

template <>
struct BDFTable<6>{
  static constexpr int order = 6;

  static constexpr double coeff_f = 60.0 / 147.0;
  Kokkos::Array<double, 6> coeffs{{-360.0 / 147.0, 450.0 / 147.0, -400.0 / 147.0, 225.0 / 147.0, -72.0 / 147.0, 10.0 / 147.0}};
};

template <class ode_type, class table_type, class scalar_type, class vec_type, class mv_type, class mat_type>
struct nonlinear_system{

  ode_type ode;
  table_type table;
  mv_type history;
  scalar_type t, dt;

  const int neqs;

  nonlinear_system(const ode_type& ode_, const table_type& table_,
		   const scalar_type t_, const scalar_type dt_,
		   const mv_type& history_)
    : ode(ode_), table(table_), history(history_), t(t_), dt(dt_), neqs(ode_.neqs) {}

  KOKKOS_FUNCTION
  void residual(const vec_type& y, const vec_type& r) const {
    // r = f
    ode.evaluate_function(y, r);

    // r = -coeff*dt*r = -coeff*dt*f
    for(int eqIdx = 0; eqIdx < neqs; ++eqIdx) {
      r(eqIdx) = -table.coeff_f*dt*r(eqIdx);
    }

    // r += y --> r = y - coeff*dt*f
    for(int eqIdx = 0; eqIdx < neqs; ++eqIdx) {
      r(eqIdx) += y(eqIdx);
    }

    // r += coeffs(i)*hist(i) --> r = y - coeff*dt*f + sum coeff(i)*hist(i)
    for(int vecIdx = 0; vecIdx < table.order; ++vecIdx) {
      for(int eqIdx = 0; eqIdx < neqs; ++eqIdx) {
	r(eqIdx) += table.coeffs[vecIdx]*history(eqIdx, vecIdx);
      }
    }
  }

  KOKKOS_FUNCTION
  void jacobian(const vec_type& y, const mat_type& jacobian) const {
    // jacobian = df/dy
    ode.evaluate_derivatives(y, jacobian);

    // jacobian *= coeff*dt*jacobian
    // jacobian += I
    for(int rowIdx = 0; rowIdx < neqs; ++rowIdx) {
      for(int colIdx = 0; colIdx < neqs; ++colIdx) {
	jacobian(rowIdx, colIdx) = table.coeff_f*dt*jacobian(rowIdx, colIdx);
      }

      // After all other updates, add Identity to jacobian
      jacobian(rowIdx, rowIdx) += 1;
    }
  }
};


template <class ode_type, class table_type, class scalar_type, class vec_type, class mv_type, class mat_type>
KOKKOS_FUNCTION
void BDFStep(ode_type& ode, const table_type& table, scalar_type t, scalar_type dt,
	     const mv_type& history, const vec_type& y, const mat_type& /*jacobian*/) {

    using nls_type = nonlinear_system<ode_type, table_type, scalar_type,
				      vec_type, mv_type, mat_type>;
    using handle_type = NewtonHandle<vec_type>;
    using newton_type = NewtonFunctor<nls_type, mat_type, vec_type, vec_type, handle_type>;

    handle_type handle;
    handle.debug_mode = false;
    vec_type rhs("right hand side", 1);

    // Create a nonlinear-system from the ode
    // and the chosen time integrator
    nls_type nls(ode, table, t, dt, history);
    newton_type newton_solver(nls, y, rhs, handle);
    newton_solver.solve();
}

template <class ode_type, class table_type, class scalar_type, class vec_type, class mv_type, class mat_type>
KOKKOS_FUNCTION
void BDFInit(ode_type& ode, const table_type& table, scalar_type t, scalar_type dt,
	     const mv_type& history, const vec_type& y0, const vec_type& y, const vec_type& tmp, const mv_type& kvecs) {

  // Depending on the BDF order required
  // the history vectors need to be computed
  // before the BDF integrator can start.
  // Using BDF of lower order with small
  // dt to ensure the accuracy is sufficient
  // and then increasing order is one option
  // Alternatively using explicit RK methods
  // of the appropriate order can also be used
  // to start BDF.

  switch (table.order) {

  case 6:

  case 5:
    // Use four steps of RKF-4,5
    for(int histIdx = 0; histIdx < 4; ++histIdx) {
      ButcherTableau<4, 5> explicit_table;
      RKSolve<ode_type, ButcherTableau<4, 5>, vec_type, mv_type, double>(ode, explicit_table, t + histIdx*dt,
									 t + (histIdx + 1)*dt, dt, 1, y0, y, tmp, kvecs);
      for(int eqIdx = 0; eqIdx < ode.neqs; ++eqIdx) {
	history(eqIdx, histIdx + 1) = y(eqIdx);
      }
      Kokkos::deep_copy(y0, y);
    }
    break;

  case 4:
    // Use three steps of RKF-4,5
    for(int histIdx = 0; histIdx < 3; ++histIdx) {
      ButcherTableau<4, 5> explicit_table;
      RKSolve<ode_type, ButcherTableau<4, 5>, vec_type, mv_type, double>(ode, explicit_table, t + histIdx*dt,
									 t + (histIdx + 1)*dt, dt, 1, y0, y, tmp, kvecs);
      for(int eqIdx = 0; eqIdx < ode.neqs; ++eqIdx) {
	history(eqIdx, histIdx + 1) = y(eqIdx);
      }
      Kokkos::deep_copy(y0, y);
    }
    break;

  case 3:
    // Use two steps of Bogacki-Shampine
    for(int histIdx = 0; histIdx < 2; ++histIdx) {
      ButcherTableau<2, 3> explicit_table;
      RKSolve<ode_type, ButcherTableau<2, 3>, vec_type, mv_type, double>(ode, explicit_table, t + histIdx*dt,
									 t + (histIdx + 1)*dt, dt, 1, y0, y, tmp, kvecs);
      for(int eqIdx = 0; eqIdx < ode.neqs; ++eqIdx) {
	history(eqIdx, histIdx + 1) = y(eqIdx);
      }
      Kokkos::deep_copy(y0, y);
    }
    break;

  case 2:
    // Use one step of Euler-Heun
    {
    ButcherTableau<1, 1> explicit_table;
    RKSolve<ode_type, ButcherTableau<1, 1>, vec_type, mv_type, double>(ode, explicit_table, t, t+dt, dt, 1, y0, y, tmp, kvecs);
    for(int eqIdx = 0; eqIdx < ode.neqs; ++eqIdx) {
      history(eqIdx, 1) = y(eqIdx);
    }
    }
    break;

  case 1:
    //No history required beyond initial conditions
    break;

  default:
    Kokkos::abort("BDF methods must have order 1, 2, 3, 4, 5 or 6!");
  }

}

} //namespace Impl
} // namespace KokkosBlas

#endif // KOKKOSBLAS_BDF_IMPL_HPP
