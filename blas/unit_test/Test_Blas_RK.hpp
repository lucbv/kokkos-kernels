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
#include "KokkosBlas_RungeKutta_impl.hpp"
#include "KokkosBlas_RungeKuttaTables_impl.hpp"

namespace Test {

// damped harmonic undriven oscillator
// m y'' + c y' + k y = 0
// solution: y=A * exp(-xi * omega_0 * t) * sin(sqrt(1-xi^2) * omega_0 * t + phi)
// omega_0 = sqrt(k/m); xi = c / sqrt(4*m*k)
// A and phi depend on y(0) and y'(0);
// Change of variables: x(t) = y(t)*exp(-c/(2m)*t) = y(t)*exp(-xi * omega_0 * t)
// Change of variables: X = [x ]
//                          [x']
// Leads to X' = A*X  with A = [ 0  1]
//                             [-d  0]
// with d = k/m - (c/(2m)^2) = (1 - xi^2)*omega_0^2
struct duho {

  constexpr static int neqs = 2;
  const double m, c, k, d;
  const double a11 = 0, a12 = 1, a21, a22;

  duho(const double m_, const double c_, const double k_) : m(m_), c(c_), k(k_), d(k_ / m_ - (c_*c_) / (4*m_*m_)), a21(-k / m), a22(-c / m) {};

  template <class vec_type1, class vec_type2>
  KOKKOS_FUNCTION
  void evaluate_function(const double /*t*/, const double /*dt*/, const vec_type1& y, const vec_type2& f) {
    f(0) = a11*y(0) + a12*y(1);
    f(1) = a21*y(0) + a22*y(1);
  }

  template <class vec_type>
  KOKKOS_FUNCTION
  void solution(const double t, const vec_type& y0, const vec_type& y) {
    using KAT = Kokkos::ArithTraits<double>;

    const double omega_0     = KAT::sqrt(k / m);
    const double xi          = c / (2*KAT::sqrt(k * m));
    const double numerator   = KAT::sqrt(1 - xi) * omega_0 * y0(0);
    const double denominator = y0(1) + xi * omega_0 * y0(0);
    const double phi         = KAT::atan(numerator / denominator);
    const double A           = y0(0) / KAT::sin(phi);

    y(0) = A * KAT::exp(-xi * omega_0 * t) * KAT::sin(KAT::sqrt(1 - xi) * omega_0 * t + phi);
    y(1) = -xi * omega_0 * y(0) + KAT::sqrt(1 - xi) * omega_0 * A * KAT::exp(-xi * omega_0 * t) * KAT::cos(KAT::sqrt(1 - xi) * omega_0 * t + phi);
  }

}; // duho

template <class ode_type, class table_type, class vec_type, class mv_type, class scalar_type>
void test_method(const std::string label, ode_type& my_ode,
		 const scalar_type& tstart, const scalar_type& tend, scalar_type& dt,
		 const int max_steps, vec_type& y_old, vec_type& y_new,
		 const Kokkos::View<double**, Kokkos::HostSpace>& ks,
		 const Kokkos::View<double*, Kokkos::HostSpace>& sol) {

  table_type table;
  vec_type tmp("tmp vector", my_ode.neqs);
  mv_type kstack("k stack", my_ode.neqs, table.nstages);
  KokkosBlas::Impl::RKSolve<ode_type, table_type, vec_type, mv_type, double>(my_ode, table, tstart, tend, dt, max_steps, y_old, y_new, tmp, kstack);

  auto y_new_h = Kokkos::create_mirror_view(y_new);
  Kokkos::deep_copy(y_new_h, y_new);
  auto kstack_h = Kokkos::create_mirror_view(kstack);
  Kokkos::deep_copy(kstack_h, kstack);

  std::cout << "\n" << label << std::endl;
  for(int stageIdx = 0; stageIdx < table.nstages; ++stageIdx) {
    EXPECT_NEAR_KK(ks(0, stageIdx), kstack_h(0, stageIdx), 1e-8);
    EXPECT_NEAR_KK(ks(1, stageIdx), kstack_h(1, stageIdx), 1e-8);
    std::cout << "  k" << stageIdx << "={" << kstack_h(0, stageIdx) << ", " << kstack_h(1, stageIdx) << "}" << std::endl;
  }
  EXPECT_NEAR_KK(sol(0), y_new_h(0), 1e-8);
  EXPECT_NEAR_KK(sol(1), y_new_h(1), 1e-8);
  std::cout << "  y={" << y_new_h(0) << ", " << y_new_h(1) << "}" << std::endl;

} // test_method

template <class execution_space>
void test_RK() {
  using vec_type   = Kokkos::View<double*,  execution_space>;
  using mv_type    = Kokkos::View<double**, execution_space>;

  duho my_oscillator(1, 1, 4);
  const int neqs    = my_oscillator.neqs;
  
  vec_type y("solution", neqs), f("function", neqs);
  y(0) = 1; y(1) = 0;

  constexpr double tstart = 0, tend = 10;
  constexpr int max_steps = 1000;
  double dt = (tend - tstart) / max_steps;
  vec_type y_new("y new", neqs), y_old("y old", neqs);

  // Since y_old_h will be reused to set initial conditions
  // for each method tested we do not want to use
  // create_mirror_view which would not do a copy
  // when y_old is in HostSpace.
  typename vec_type::HostMirror y_old_h = Kokkos::create_mirror(y_old);
  y_old_h(0) = 1; y_old_h(1) = 0;

  // We perform a single step using a RK method
  // and check the values for ki and y_new against
  // expected values.
  {
    Kokkos::deep_copy(y_old, y_old_h);
    double ks_raw[2] = {0, -4};
    Kokkos::View<double**, Kokkos::HostSpace> ks(ks_raw, 2, 1);
    double sol_raw[2] = {1, -0.04};
    Kokkos::View<double*, Kokkos::HostSpace> sol(sol_raw, 2);
    test_method<duho, KokkosBlas::Impl::ButcherTableau<0, 0>, vec_type, mv_type, double>("Euler-Forward", my_oscillator, tstart, tend, dt, 1, y_old, y_new, ks, sol);
  }

  {
    Kokkos::deep_copy(y_old, y_old_h);
    double ks_raw[4] = {0, -0.04,
			-4, -3.96};
    Kokkos::View<double**, Kokkos::HostSpace> ks(ks_raw, 2, 2);
    double sol_raw[2] = {0.9998, -0.0398};
    Kokkos::View<double*, Kokkos::HostSpace> sol(sol_raw, 2);
    test_method<duho, KokkosBlas::Impl::ButcherTableau<1, 1>, vec_type, mv_type, double>("Euler-Heun", my_oscillator, tstart, tend, dt, 1, y_old, y_new, ks, sol);
  }

  {
    Kokkos::deep_copy(y_old, y_old_h);
    double ks_raw[6]  = {0, -0.02, -0.03980078,
			 -4, -3.98, -3.95940234};
    Kokkos::View<double**, Kokkos::HostSpace> ks(ks_raw, 2, 3);
    double sol_raw[2] = {0.9998, -0.03979999};
    Kokkos::View<double*, Kokkos::HostSpace> sol(sol_raw, 2);
    test_method<duho, KokkosBlas::Impl::ButcherTableau<1, 2>, vec_type, mv_type, double>("RKF-12", my_oscillator, tstart, tend, dt, 1, y_old, y_new, ks, sol);
  }

  {
    Kokkos::deep_copy(y_old, y_old_h);
    double ks_raw[8]  = {0, -0.02, -0.02985, -0.039798,
			 -4, -3.98, -3.96955, -3.95940467};
    Kokkos::View<double**, Kokkos::HostSpace> ks(ks_raw, 2, 4);
    double sol_raw[2] = {0.99980067, -0.039798};
    Kokkos::View<double*, Kokkos::HostSpace> sol(sol_raw, 2);
    test_method<duho, KokkosBlas::Impl::ButcherTableau<2, 3>, vec_type, mv_type, double>("RKBS", my_oscillator, tstart, tend, dt, 1, y_old, y_new, ks, sol);
  }

  {
    Kokkos::deep_copy(y_old, y_old_h);
    double ks_raw[12] = {0, -0.01, -0.01497188, -0.03674986, -0.03979499, -0.0199505,
			 -4, -3.99, -3.98491562, -3.96257222, -3.95941166, -3.97984883};
    Kokkos::View<double**, Kokkos::HostSpace> ks(ks_raw, 2, 6);
    double sol_raw[2] = { 0.99980067, -0.03979801};
    Kokkos::View<double*, Kokkos::HostSpace> sol(sol_raw, 2);
    test_method<duho, KokkosBlas::Impl::ButcherTableau<4, 5>, vec_type, mv_type, double>("RKF-45", my_oscillator, tstart, tend, dt, 1, y_old, y_new, ks, sol);
  }

  {
    Kokkos::deep_copy(y_old, y_old_h);
    double ks_raw[12] = {0, -0.008, -0.011982, -0.02392735, -0.03979862, -0.03484563,
			 -4, -3.992, -3.987946, -3.97578551, -3.95940328, -3.96454357};
    Kokkos::View<double**, Kokkos::HostSpace> ks(ks_raw, 2, 6);
    double sol_raw[2] = { 0.99980067, -0.03979801};
    Kokkos::View<double*, Kokkos::HostSpace> sol(sol_raw, 2);
    test_method<duho, KokkosBlas::Impl::ButcherTableau<4, 5, 1>, vec_type, mv_type, double>("Cash-Karp", my_oscillator, tstart, tend, dt, 1, y_old, y_new, ks, sol);
  }

  vec_type y_ref("reference value", neqs);
  y_old(0) = 1; y_old(1) = 0;
  my_oscillator.solution(tstart + dt, y_old, y_ref);

  std::cout << "\nAnalytical solution" << std::endl;
  std::cout << "  y={" << y_ref(0) << ", " << y_ref(1) << "}" << std::endl;

}

} // namespace Test

int test_RK() {
  Test::test_RK<TestExecSpace>();

  return 1;
}

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, RKSolve_serial) { test_RK(); }
#endif
