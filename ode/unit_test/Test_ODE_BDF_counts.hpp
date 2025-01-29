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
#include "KokkosKernels_TestUtils.hpp"

#include "KokkosODE_BDF.hpp"
#include "Test_ODE_TestProblems.hpp"

namespace Test {

template <class Device, class OdeType>
void BDF_Count(const Device, const OdeType myODE, const double relTol, const double absTol,
	       const int /*expected_count*/) {
  using execution_space = typename Device::execution_space;
  using vec_type        = Kokkos::View<double*, Device>;
  using mat_type        = Kokkos::View<double**, Device>;
  using count_type      = Kokkos::View<int*, execution_space>;

  constexpr int neqs = myODE.neqs;

  constexpr double tstart = myODE.tstart(), tend = myODE.tend();
  constexpr int num_steps      = myODE.numsteps();
  constexpr int max_substeps = 1000;

  vec_type y("solution", neqs), f("function", neqs);
  vec_type y_new("y new", neqs), y_old("y old", neqs);
  count_type count("time step count", 1);
  Kokkos::deep_copy(count, max_substeps);

  auto y_h                              = Kokkos::create_mirror_view(y);
  typename vec_type::HostMirror y_old_h = Kokkos::create_mirror(y_old);
  auto y_ref_h                          = Kokkos::create_mirror(y);
  for (int dofIdx = 0; dofIdx < neqs; ++dofIdx) {
    y_h(dofIdx)     = myODE.expected_val(tstart, dofIdx);
    y_old_h(dofIdx) = y_h(dofIdx);
    y_ref_h(dofIdx) = myODE.expected_val(tend, dofIdx);
  }
  Kokkos::deep_copy(y, y_h);

  mat_type temp("buffer1", myODE.neqs, 23 + 2 * myODE.neqs + 4), temp2("buffer2", 6, 7);

  Kokkos::RangePolicy<execution_space> my_policy(0, 1);
  Kokkos::deep_copy(y_old, y_old_h);
  Kokkos::deep_copy(y_new, y_old_h);
  BDF_Solve_wrapper solve_wrapper(myODE, tstart, tend, (tend - tstart) / num_steps,
				  10*(tend - tstart) / num_steps, y_old, y_new, temp, temp2, count);
  Kokkos::parallel_for(my_policy, solve_wrapper);
  Kokkos::fence();

  auto y_new_h = Kokkos::create_mirror(y_new);
  Kokkos::deep_copy(y_new_h, y_new);

  typename count_type::HostMirror count_h = Kokkos::create_mirror_view(count);
  Kokkos::deep_copy(count_h, count);

  double error = 0.0;
  for (int eqIdx = 0; eqIdx < neqs; ++eqIdx) {
    error += Kokkos::pow(y_ref_h(eqIdx) - y_new_h(eqIdx), 2.0) /
             Kokkos::pow(absTol + relTol * Kokkos::abs(y_new_h(eqIdx)), 2.0);
  }
  error = Kokkos::sqrt(error / neqs);

  std::cout << std::string(OdeType::name) << " took " << count_h(0) << " steps" << std::endl;

  EXPECT_LE(error, 1.0);
  // EXPECT_LE(count_h(0), expected_count);
}  // BDF_Count

}  // namespace Test

void test_BDF_count() {
  //    BDF_Count(Device,       OdeType,                      relTol, absTol, /*expected_count*/)
  Test::BDF_Count(TestDevice(), TestProblem::DegreeOnePoly(), 1.0e-6, 1e-12, 2);
  Test::BDF_Count(TestDevice(), TestProblem::DegreeTwoPoly(), 1.0e-6, 1e-12, 2);
  Test::BDF_Count(TestDevice(), TestProblem::DegreeThreePoly(), 1.0e-6, 1e-12, 2);
  Test::BDF_Count(TestDevice(), TestProblem::DegreeFivePoly(), 1.0e-6, 1e-12, 5);
  Test::BDF_Count(TestDevice(), TestProblem::Exponential(0.7), 2.0e-6, 1e-12, 4);
  Test::BDF_Count(TestDevice(), TestProblem::SpringMassDamper(1001., 1000.), 1.0e-4, 0.0, 272);
  Test::BDF_Count(TestDevice(), TestProblem::CosExp(-10., 2., 1.), 5.3e-5, 0.0, 25);
  Test::BDF_Count(TestDevice(), TestProblem::StiffChemicalDecayProcess(1e4, 1.), 4e-9, 1.8e-10, 2786);
  // Test::BDF_Count(TestDevice(), TestProblem::Tracer(10.0), 0.0, 1e-3, 10);
  Test::BDF_Count(TestDevice(), TestProblem::EnrightB5(), 1.3e-2, 0.0, 90);
  Test::BDF_Count(TestDevice(), TestProblem::EnrightC1(), 1.e-5, 1e-14, 90);
  Test::BDF_Count(TestDevice(), TestProblem::EnrightC5(), 1.e-4, 1e-14, 97);
  Test::BDF_Count(TestDevice(), TestProblem::EnrightC5(), 1.e-5, 1e-14, 97);
  Test::BDF_Count(TestDevice(), TestProblem::EnrightD2(), 2.e-4, 0.0, 590);
  Test::BDF_Count(TestDevice(), TestProblem::EnrightD4(), 1.e-5, 1.e-9, 932);
  Test::BDF_Count(TestDevice(), TestProblem::KKStiffChemistry(), 1e-5, 0.0, 1);
}

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, BDF_Count) { test_BDF_count(); }
#endif
