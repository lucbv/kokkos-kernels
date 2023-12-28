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
#include <vector>
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <KokkosBlas2_symv.hpp>
#include <KokkosBlas2_gemv.hpp>

#include <KokkosKernels_TestUtils.hpp>

namespace Test {

template <class Scalar, class Ordinal, class Offset, class Layout, class Device>
void symv_deterministic_test() {
  using execution_space = typename Device::execution_space;
  // using memory_space    = typename Device::memory_space;

  using mat_type = Kokkos::View<Scalar**, Layout, execution_space>;
  using vec_type = Kokkos::View<Scalar*, Layout, execution_space>;
  using KAT_S    = Kokkos::ArithTraits<Scalar>;

  execution_space space{};
  constexpr Ordinal numRows = 5;

  mat_type symA("symmetric matrix", numRows, numRows);
  typename mat_type::HostMirror symA_h = Kokkos::create_mirror_view(symA);

  vec_type X("X vector", numRows), Y("Y vector", numRows);
  typename vec_type::HostMirror X_h    = Kokkos::create_mirror_view(X);
  typename vec_type::HostMirror Y_h    = Kokkos::create_mirror_view(Y);
  typename vec_type::HostMirror Yini_h = Kokkos::create_mirror(Y);

  X_h(0) = 3; X_h(1) = 5; X_h(2) = 6; X_h(3) = 2; X_h(4) = 10;
  Yini_h(0) = 7; Yini_h(1) = 8; Yini_h(2) = 5; Yini_h(3) = 2; Yini_h(4) = 5;

  Kokkos::deep_copy(symA, symA_h);
  Kokkos::deep_copy(X, X_h);
  Kokkos::deep_copy(Y, Yini_h);

  std::string uplos = "UL";
  std::vector<Scalar> betas ({KAT_S::zero(), KAT_S::one(), KAT_S::one()+KAT_S::one()});
  std::vector<Scalar> alphas({KAT_S::zero(), KAT_S::one(), KAT_S::one()+KAT_S::one()});

  {
    // Filling the full matrix symmetrically
    symA_h(0, 0) = 74; symA_h(0, 1) = 80; symA_h(0, 2) = 12; symA_h(0, 3) = 48; symA_h(0, 4) =  9;
    symA_h(1, 0) = 80; symA_h(1, 1) = 79; symA_h(1, 2) = 43; symA_h(1, 3) = 73; symA_h(1, 4) = 24;
    symA_h(2, 0) = 12; symA_h(2, 1) = 43; symA_h(2, 2) = 97; symA_h(2, 3) = 11; symA_h(2, 4) = 17;
    symA_h(3, 0) = 48; symA_h(3, 1) = 73; symA_h(3, 2) = 11; symA_h(3, 3) = 56; symA_h(3, 4) = 35;
    symA_h(4, 0) =  9; symA_h(4, 1) = 24; symA_h(4, 2) = 17; symA_h(4, 3) = 35; symA_h(4, 4) = 37;

    // Compute the vector A*X
    typename vec_type::HostMirror AX_h("AX vector values", numRows);
    AX_h(0) = 880; AX_h(1) = 1279; AX_h(2) = 1025; AX_h(3) = 1037; AX_h(4) = 689;

    for(char uplo : uplos) {
      for(Scalar beta : betas) {
	for(Scalar alpha : alphas) {
	  Kokkos::deep_copy(Y, Yini_h);
	  KokkosBlas::symv(space, &uplo, alpha, symA, X, beta, Y);

	  Kokkos::deep_copy(Y_h, Y);
	  for(int rowIdx = 0; rowIdx < numRows; ++rowIdx) {
	    EXPECT_NEAR_KK_REL(Y_h(rowIdx),
			       beta*Yini_h(rowIdx) + alpha*AX_h(rowIdx),
			       10*KAT_S::eps(),
			       "SYMV: Failed");
	  }
	}
      }
    }
  }

  {
    // Filling the full matrix un-symmetrically
    // This means that "U" and "L" modes will
    // actually return different results!
    symA_h(0, 0) = 74; symA_h(0, 1) = 80; symA_h(0, 2) = 12; symA_h(0, 3) = 48; symA_h(0, 4) =  9;
    symA_h(1, 0) = 21; symA_h(1, 1) = 79; symA_h(1, 2) = 43; symA_h(1, 3) = 73; symA_h(1, 4) = 24;
    symA_h(2, 0) = 90; symA_h(2, 1) = 14; symA_h(2, 2) = 97; symA_h(2, 3) = 11; symA_h(2, 4) = 17;
    symA_h(3, 0) = 17; symA_h(3, 1) = 43; symA_h(3, 2) = 26; symA_h(3, 3) = 56; symA_h(3, 4) = 35;
    symA_h(4, 0) = 18; symA_h(4, 1) = 37; symA_h(4, 2) = 15; symA_h(4, 3) = 58; symA_h(4, 4) = 37;
    Kokkos::deep_copy(symA, symA_h);

    // Compute the vector A*X
    // For "U" mode
    typename vec_type::HostMirror AXu_h("AXu vector values", numRows);
    AXu_h(0) = 880; AXu_h(1) = 1279; AXu_h(2) = 1025; AXu_h(3) = 1037; AXu_h(4) = 689;

    for(Scalar beta : betas) {
      for(Scalar alpha : alphas) {
	Kokkos::deep_copy(Y, Yini_h);
	KokkosBlas::symv(space, "U", alpha, symA, X, beta, Y);

	Kokkos::deep_copy(Y_h, Y);
	for(int rowIdx = 0; rowIdx < numRows; ++rowIdx) {
	  EXPECT_NEAR_KK_REL(Y_h(rowIdx),
			     beta*Yini_h(rowIdx) + alpha*AXu_h(rowIdx),
			     10*KAT_S::eps(),
			     "SYMV: Failed");
	}
      }
    }

    // Compute the vector A*X
    // For "L" mode
    typename vec_type::HostMirror AXl_h("AXl vector values", numRows);
    AXl_h(0) = 1081; AXl_h(1) = 998; AXl_h(2) = 1124; AXl_h(3) = 1114; AXl_h(4) = 815;

    for(Scalar beta : betas) {
      for(Scalar alpha : alphas) {
	Kokkos::deep_copy(Y, Yini_h);
	KokkosBlas::symv(space, "L", alpha, symA, X, beta, Y);

	Kokkos::deep_copy(Y_h, Y);
	for(int rowIdx = 0; rowIdx < numRows; ++rowIdx) {
	  EXPECT_NEAR_KK_REL(Y_h(rowIdx),
			     beta*Yini_h(rowIdx) + alpha*AXl_h(rowIdx),
			     10*KAT_S::eps(),
			     "SYMV: Failed");
	}
      }
    }
  }
}

}

template <class Scalar, class Ordinal, class Offset, class Device>
void test_symv() {

#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  Test::symv_deterministic_test<Scalar, Ordinal, Offset, Kokkos::LayoutLeft, Device>();
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  Test::symv_deterministic_test<Scalar, Ordinal, Offset, Kokkos::LayoutRight, Device>();
#endif
}

#define KOKKOSKERNELS_EXECUTE_TEST(SCALAR, ORDINAL, OFFSET, DEVICE)            \
  TEST_F(TestCategory,                                                         \
         symv##_##SCALAR##_##ORDINAL##_##OFFSET##_##DEVICE) {                  \
    test_symv<SCALAR, ORDINAL, OFFSET, DEVICE>();                              \
  }

#define NO_TEST_COMPLEX

#include <Test_Common_Test_All_Type_Combos.hpp>

#undef KOKKOSKERNELS_EXECUTE_TEST
#undef NO_TEST_COMPLEX
