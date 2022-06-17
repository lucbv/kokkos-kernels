#ifdef KOKKOSKERNELS_INST_DOUBLE

#include <gtest/gtest.h>

#include <KokkosBatched_ODE_Newton.hpp>
#include <KokkosBatched_ODE_TestProblems.hpp>

namespace KokkosBatched {
namespace Experimental {
namespace ODE {

// Logistic equation
// dy/dt=y(1-y)
//
// solution y = 1/(1+exp(-t))
// y(0)=0.5
//
// Using BDF1 to integrate:
// y-y_n=dt*y*(1-y)
//
// Residual: r = y - dt*y*(1-y) - y_n
// Jacobian: J = 1 - dt + 2*dt*y
template <typename scalar_type, typename execution_space>
struct TestODE {
  using vec_type = Kokkos::View<scalar_type*, execution_space>;
  using mat_type = Kokkos::View<scalar_type**, execution_space>;

  const int neqs = 1;
  scalar_type dt;
  vec_type state;

  TestODE(const scalar_type dt_, vec_type initial_state)
    : dt(dt_), state(initial_state) {}

  KOKKOS_FUNCTION void derivatives(const scalar_type /*t*/,
                                   const vec_type& y, const vec_type& dydt) const {
    dydt(0) = y(0) - state(0) - dt*y(0)*(1 - y(0));
  }

  KOKKOS_FUNCTION void jacobian(const scalar_type /*t*/, const vec_type& y, const mat_type& jac) const {
    jac(0,0) = 1 - dt + 2*dt*y(0);
  }

  KOKKOS_FUNCTION void update_state(const vec_type /*y*/) const {
    state(0) = state(0);
  }

  KOKKOS_FUNCTION scalar_type expected_val(const scalar_type t) const {
    using Kokkos::exp;

    return static_cast<scalar_type>(1 / (1 + exp(-t)));
  }

  KOKKOS_FUNCTION int num_equations() const { return neqs; }
};


template <typename execution_space, typename scalar_type>
int test_newton_solver() {
  using vec_type  = typename Kokkos::View<scalar_type*,  execution_space>;
  using mat_type  = typename Kokkos::View<scalar_type**, execution_space>;
  using norm_type = typename Kokkos::View<scalar_type*,  execution_space>;
  using handle_type = NewtonHandle<norm_type>;
  using system_type = TestODE<scalar_type, execution_space>;

  vec_type state("state", 1);
  Kokkos::deep_copy(state, 0.5);
  system_type ode(0.1, state);

  vec_type x("solution vector", 1), rhs("right hand side vector", 1);
  Kokkos::deep_copy(x, 0.5);

  handle_type handle;
  handle.debug_mode=true;
  NewtonFunctor<system_type, mat_type, vec_type, vec_type, handle_type> newton_solver(ode, x, rhs, handle);
  newton_solver(0);

  auto x_h = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(x_h, x);
  printf("Non-linear problem solution:\n");
  printf("  [%f]\n", x_h(0));

  return 0;
}

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, ODE_Newton) {
  test_newton_solver<TestExecSpace, double>();
}
#endif

}
}
}

#endif
