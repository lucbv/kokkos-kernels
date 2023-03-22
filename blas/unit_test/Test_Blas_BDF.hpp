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
#include <gtest/gtest.h>

#include <KokkosBlas_BDF_impl.hpp>
#include <KokkosBlas_Newton_impl.hpp>
#include <KokkosKernels_TestUtils.hpp>

namespace Test {
namespace BDF {

// Logistic equation
// dy/dt=y(1-y)
//
// solution y = 1/(1+exp(-t))
// y(0)=0.5
//
// Using BDF1 to integrate:
// y-y_n=dt*y*(1-y)
//
// Residual: r = y - y_n - dt*y*(1-y)
// Jacobian: J = 1 - dt + 2*dt*y
template <typename scalar_type, typename execution_space>
struct LogisticEquation {
  using vec_type = Kokkos::View<scalar_type*, execution_space>;
  using mat_type = Kokkos::View<scalar_type**, execution_space>;

  const int neqs = 1;
  scalar_type dt;
  vec_type state;

  LogisticEquation(const scalar_type dt_, vec_type initial_state)
      : dt(dt_), state(initial_state) {}

  KOKKOS_FUNCTION void evaluate_function(const vec_type& y, const vec_type& f) const {
    f(0) = y(0) * (1 - y(0));
  }

  KOKKOS_FUNCTION void evaluate_derivatives(const vec_type& y, const mat_type& dfdy) const {
    dfdy(0, 0) = 1 - 2*y(0);
  }

  KOKKOS_FUNCTION void residual(const vec_type& y, const vec_type& dydt) const {
    dydt(0) = y(0) - state(0) - dt * y(0) * (1 - y(0));
  }

  KOKKOS_FUNCTION void jacobian(const vec_type& y, const mat_type& jac) const {
    jac(0, 0) = 1 - dt + 2 * dt * y(0);
  }

  KOKKOS_FUNCTION scalar_type expected_val(const scalar_type t) const {
    using Kokkos::exp;

    return static_cast<scalar_type>(1 / (1 + exp(-t)));
  }

  KOKKOS_FUNCTION int num_equations() const { return neqs; }
};

template <class solver>
struct NewtonWrapper {
  solver newton_solver;

  NewtonWrapper(solver newton_solver_) : newton_solver(newton_solver_){};

  KOKKOS_INLINE_FUNCTION
  void operator()(const int /* system_index */) const { newton_solver.solve(); }
};

template <typename execution_space, typename scalar_type>
int test_logistic() {
  using vec_type    = typename Kokkos::View<scalar_type*, execution_space>;
  using mat_type    = typename Kokkos::View<scalar_type**, execution_space>;
  using norm_type   = typename Kokkos::View<scalar_type*, execution_space>;
  using handle_type = KokkosBlas::Impl::NewtonHandle<norm_type>;
  using system_type = LogisticEquation<scalar_type, execution_space>;
  using newton_type =
      KokkosBlas::Impl::NewtonFunctor<system_type, mat_type, vec_type, vec_type,
                                      handle_type>;
  using table_type  = KokkosBlas::Impl::BDFTable<1>;
  using nls_type    = KokkosBlas::Impl::nonlinear_system<system_type, table_type,
							 scalar_type, vec_type,
							 mat_type, mat_type>;

  {
    // Create the non-linear system and initialize data
    vec_type state("state", 1);
    Kokkos::deep_copy(state, 0.5);
    system_type ode(0.1, state);

    vec_type x("solution vector", 1), rhs("right hand side vector", 1);
    Kokkos::deep_copy(x, 0.5);

    // Create the solver and wrapper
    handle_type handle;
    handle.debug_mode = false;
    newton_type newton_solver(ode, x, rhs, handle);
    NewtonWrapper<newton_type> wrapper(newton_solver);

    // Launch the problem in a parallel_for
    Kokkos::RangePolicy<execution_space> my_policy(0, 1);
    Kokkos::parallel_for(my_policy, wrapper);

    // Get the solution back and test it
    auto x_h = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(x_h, x);
    printf("Non-linear problem solution:\n");
    printf("  [%f]\n", x_h(0));
  }

  // Now do the same but using the nonlinear system struct
  // which computes residual and jacobian based on function
  // and derivative evaluations from the system.
  {
    const scalar_type dt  = 0.1;
    mat_type jac("Jacobian", 1, 1);
    mat_type history("history", 1, 1);
    Kokkos::deep_copy(history, 0.5);
    vec_type state("state", 1);
    Kokkos::deep_copy(state, 0.5);
    system_type ode(dt, state);
    table_type table;

    nls_type nls(ode, table, 0, dt, history);
    vec_type x("solution vector", 1), rhs("right hand side vector", 1);
    Kokkos::deep_copy(x, 0.5);

    // Create the solver and wrapper
    handle_type handle;
    handle.debug_mode = false;
    KokkosBlas::Impl::NewtonFunctor<nls_type, mat_type, vec_type, vec_type,
				    handle_type> newton_solver(nls, x, rhs, handle);
    newton_solver.solve();

    // Get the solution back and test it
    auto x_h = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(x_h, x);
    printf("Non-linear problem solution:\n");
    printf("  [%f]\n", x_h(0));
  }

  // Now do the same thing with a direct call
  // to the BDFStep function which is closer
  // to what users want to do!
  {
    const scalar_type dt  = 0.1;
    mat_type jac("Jacobian", 1, 1);
    mat_type history("history", 1, 1);
    Kokkos::deep_copy(history, 0.5);
    vec_type state("state", 1);
    Kokkos::deep_copy(state, 0.5);
    system_type ode(dt, state);
    table_type table;
    scalar_type t = 0;

    vec_type y("solution vector", 1);
    Kokkos::deep_copy(y, 0.5);
    KokkosBlas::Impl::BDFStep(ode, table, t, dt, history, y, jac);

    // Get the solution back and test it
    auto y_h = Kokkos::create_mirror_view(y);
    Kokkos::deep_copy(y_h, y);
    printf("Non-linear problem solution:\n");
    printf("  [%f]\n", y_h(0));
  }

  return 0;
}

}  // namespace BDF
}  // namespace Test

template <class scalar_type>
int test_bdf() {
  Test::BDF::test_logistic<TestExecSpace, scalar_type>();

  return 1;
}

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, bdf_serial) { test_bdf<double>(); }
#endif
