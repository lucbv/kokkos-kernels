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

#include "KokkosBlas_RungeKuttaTables_impl.hpp"
#include "KokkosBlas_RungeKutta_impl.hpp"

namespace {
// R1 = 1e-6*1.85e10 * exp(-15618 / T) * (reac) ( 1 – (1- 10^-9) reac)
// d(reac)/dt = -R1
// d(prod)/dt = R1
struct chem_model_1 {

  constexpr static int neqs = 2;
  constexpr static double alpha = 1e-6*1.85e10;
  constexpr static double beta  = 15618;
  constexpr static double gamma = 1 - 10^-9;

  const double tstart, tend, T0, T1;

  chem_model_1(const double tstart_ = 0, const double tend_ = 300,
	       const double T0_ = 300, const double T1_ = 800) : tstart(tstart_), tend(tend_), T0(T0_), T1(T1_) {};

  template <class vec_type1, class vec_type2>
  KOKKOS_FUNCTION
  void evaluate_function(const double t, const double /*dt*/, const vec_type1& y, const vec_type2& f) const {
    // First compute the temperature
    // using linear ramp from T0 to T1
    // between tstart and tend.
    double T = (T1 - T0) * (t - tstart) / (tend - tstart) + T0;

    // Evaluate the chemical reaction rate
    f(0) = -alpha * Kokkos::exp(-beta / T) * y(0) * (1 - gamma * y(0));
    f(1) = -f(0);
  }

};

template <class ode_type, class table_type, class vec_type, class mv_type, class scalar_type>
struct RKSolve_wrapper {

  ode_type my_ode;
  table_type table;
  scalar_type tstart, tend, dt;
  int max_steps;
  vec_type y_old, y_new, tmp;
  mv_type kstack;

  RKSolve_wrapper(const ode_type& my_ode_, const table_type& table_,
		  const scalar_type tstart_, const scalar_type tend_, const scalar_type dt_,
		  const int max_steps_, const vec_type& y_old_, const vec_type& y_new_,
		  const vec_type& tmp_, const mv_type& kstack_) :
    my_ode(my_ode_), table(table_), tstart(tstart_), tend(tend_), dt(dt_), max_steps(max_steps_),
    y_old(y_old_), y_new(y_new_), tmp(tmp_), kstack(kstack_) {}

  KOKKOS_FUNCTION
  void operator() (const int idx) const {

    // Take subviews to create the local problem
    auto local_y_old  = Kokkos::subview( y_old, Kokkos::pair(2*idx, 2*idx + 1));
    auto local_y_new  = Kokkos::subview( y_new, Kokkos::pair(2*idx, 2*idx + 1));
    auto local_tmp    = Kokkos::subview(   tmp, Kokkos::pair(2*idx, 2*idx + 1));
    auto local_kstack = Kokkos::subview(kstack, Kokkos::pair(2*idx, 2*idx + 1), Kokkos::ALL());

    // Run Runge-Kutta time integrator
    KokkosBlas::Impl::RKSolve<ode_type, table_type, vec_type, mv_type, double>(my_ode, table, tstart, tend, dt, max_steps,
									       local_y_old, local_y_new, local_tmp, local_kstack);
  }
};

} // namespace (anonymous)

int main(int /*argc*/, char** /*argv*/) {

  Kokkos::initialize();
  {

  using execution_space = Kokkos::DefaultExecutionSpace;
  using vec_type   = Kokkos::View<double*,  execution_space>;
  using mv_type    = Kokkos::View<double**, execution_space>;
  using table_type = KokkosBlas::Impl::ButcherTableau<4, 5, 1>;

  constexpr int num_odes = 10000;
  chem_model_1 chem_model;
  const int neqs = chem_model.neqs;
  const int max_steps = 15000;
  const double dt = 0.1;

  table_type table;
  vec_type tmp("tmp vector", neqs*num_odes);
  mv_type kstack("k stack", neqs*num_odes, table.nstages);

  // Set initial conditions
  vec_type y_new("solution", neqs*num_odes);
  vec_type y_old("initial conditions", neqs*num_odes);
  auto y_old_h = Kokkos::create_mirror(y_old);
  y_old_h(0) = 1; y_old_h(1) = 0;
  Kokkos::deep_copy(y_old, y_old_h);
  Kokkos::deep_copy(y_new, y_old_h);

  Kokkos::RangePolicy<execution_space> my_policy(0, num_odes);
  RKSolve_wrapper solve_wrapper(chem_model, table, chem_model.tstart, chem_model.tend,
				dt, max_steps, y_old, y_new, tmp, kstack);

  Kokkos::Timer time;
  time.reset();
  Kokkos::parallel_for(my_policy, solve_wrapper);
  double run_time = time.seconds();

  auto y_new_h = Kokkos::create_mirror(y_new);
  Kokkos::deep_copy(y_new_h, y_new);
  std::cout << "\nChem model 1" << std::endl;
  std::cout << "  t0=" << chem_model.tstart << ", tn=" << chem_model.tend << std::endl;
  std::cout << "  T0=" << chem_model.T0 << ", Tn=" << chem_model.T1 << std::endl;
  std::cout << "  dt=" << dt << std::endl;
  std::cout << "  y(t0)={" << y_old_h(0) << ", " << y_old_h(1) << "}" << std::endl;
  std::cout << "  y(tn)={" << y_new_h(0) << ", " << y_new_h(1) << "}" << std::endl;
  std::cout << "  num odes: " << num_odes << std::endl;
  std::cout << "  time elapsed: " << run_time << std::endl;

  }
  Kokkos::finalize();
}
