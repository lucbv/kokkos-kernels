#ifdef KOKKOSKERNELS_INST_DOUBLE

#include <gtest/gtest.h>

#include <KokkosBatched_ODE_Newton.hpp>
#include <KokkosBatched_ODE_TestProblems.hpp>

namespace KokkosBatched {
namespace Experimental {
namespace ODE {

template <typename execution_space, typename scalar_type>
int test_newton_solver() {
  using vec_type  = typename Kokkos::View<scalar_type*,  execution_space>;
  using mat_type  = typename Kokkos::View<scalar_type**, execution_space>;
  using norm_type = typename Kokkos::View<scalar_type*,  execution_space>;
  using handle_type = NewtonHandle<norm_type>;

  LucP1 ode{};

  vec_type x("solution vector", 2), rhs("right hand side vector", 2);

  { // Set initial values on host
    auto x_h = Kokkos::create_mirror_view(x);
    x_h(0) = 3; x_h(1) = 2;
    Kokkos::deep_copy(x, x_h);
    auto rhs_h = Kokkos::create_mirror_view(rhs);
    rhs_h(0) = 2; rhs(1) = 3;
    Kokkos::deep_copy(rhs, rhs_h);
  }
  printf("x: [%f, %f]\n", x(0), x(1));
  printf("rhs: [%f, %f]\n", rhs(0), rhs(1));

  handle_type handle;
  NewtonFunctor<LucP1, mat_type, vec_type, vec_type, handle_type> newton_solver(ode, x, rhs, handle);
  newton_solver(0);

  auto x_h = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(x_h, x);
  printf("Non-linear problem solution:\n");
  printf("  [%f]\n", x_h(0));
  printf("  [%f]\n", x_h(1));

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
