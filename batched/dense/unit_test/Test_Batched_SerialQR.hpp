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
/// \author Luc Berger-Vergiat (lberg@sandia.gov)
/// \author Cameron Smith (smithc11@rpi.edu)

#include "gtest/gtest.h"

#include <KokkosBatched_QR_Decl.hpp>      // KokkosBatched::QR
#include "KokkosBatched_ApplyQ_Decl.hpp"  // KokkosBatched::ApplyQ
#include "KokkosBatched_QR_FormQ_Serial_Internal.hpp"
#include <KokkosBatched_Util.hpp>  // KokkosBlas::Algo
#include <Kokkos_Core.hpp>

template <class Device, class Scalar, class AlgoTagType>
void testQR() {
  // Analytical test with a rectangular matrix
  //     [3,  5]        [-0.60, -0.80,  0.00]        [-5, -3]
  // A = [4,  0]    Q = [-0.80, -0.48, -0.36]    R = [ 0,  5]
  //     [0, -3]        [ 0.00, -0.60, -0.80]        [ 0,  0]
  //
  // Expected outputs:
  //                         [ -5,  -3]
  // tau = [5/8, 10/18]  A = [1/2,   5]
  //                         [  0, 1/3]
  //

  using MatrixViewType    = Kokkos::View<double**>;
  using ColVectorViewType = Kokkos::View<double*>;
  using ColWorkViewType   = Kokkos::View<double*>;

  constexpr int m = 3, n = 2;

  MatrixViewType A("A", m, n), B("B", m, n);
  ColVectorViewType t("t", n);
  ColWorkViewType w("w", n);

  typename MatrixViewType::HostMirror A_h = Kokkos::create_mirror_view(A);
  A_h(0, 0)                               = 3;
  A_h(0, 1)                               = 5;
  A_h(1, 0)                               = 4;
  A_h(1, 1)                               = 0;
  A_h(2, 0)                               = 0;
  A_h(2, 1)                               = -3;

  Kokkos::deep_copy(A, A_h);
  Kokkos::deep_copy(B, A_h);

  Kokkos::parallel_for(
      "serialQR", 1, KOKKOS_LAMBDA(int) {
        // compute the QR factorization of A and store the results in A and t
        // (tau) - see the lapack dgeqp3(...) documentation:
        // www.netlib.org/lapack/explore-html-3.6.1/dd/d9a/group__double_g_ecomputational_ga1b0500f49e03d2771b797c6e88adabbb.html
        KokkosBatched::SerialQR<AlgoTagType>::invoke(A, t, w);
      });

  Kokkos::fence();
  Kokkos::deep_copy(A_h, A);
  typename ColVectorViewType::HostMirror tau = Kokkos::create_mirror_view(t);
  Kokkos::deep_copy(tau, t);

  EXPECT_DOUBLE_EQ(A_h(0, 0), -5);
  EXPECT_DOUBLE_EQ(A_h(0, 1), -3);
  EXPECT_DOUBLE_EQ(A_h(1, 0), 0.5);
  EXPECT_DOUBLE_EQ(A_h(1, 1), 5);
  EXPECT_DOUBLE_EQ(A_h(2, 0), 0);
  EXPECT_DOUBLE_EQ(A_h(2, 1), 1. / 3.);

  EXPECT_DOUBLE_EQ(tau(0), 5. / 8.);
  EXPECT_DOUBLE_EQ(tau(1), 10. / 18.);

  Kokkos::parallel_for(
      "serialApplyQ", 1, KOKKOS_LAMBDA(int) {
        KokkosBatched::SerialApplyQ<Side::Left, Trans::Transpose, Algo::ApplyQ::Unblocked>::invoke(A, t, B, w);
      });
  typename MatrixViewType::HostMirror B_h = Kokkos::create_mirror_view(A);
  Kokkos::deep_copy(B_h, B);

  EXPECT_DOUBLE_EQ(B_h(0, 0), -5);
  EXPECT_DOUBLE_EQ(B_h(0, 1), -3);
  EXPECT_DOUBLE_EQ(B_h(1, 0), 0);
  EXPECT_DOUBLE_EQ(B_h(1, 1), 5);
  EXPECT_DOUBLE_EQ(B_h(2, 0), 0);
  EXPECT_DOUBLE_EQ(B_h(2, 1), 0);

  // Kokkos::parallel_for("serialFormQ", 1, KOKKOS_LAMBDA(int) {
  //     KokkosBatched::SerialQR_FormQ_Internal::invoke(m, n, A, as0, as1, t, ts, B, bs0, bs1, w);
  //   });
}

#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F(TestCategory, batched_scalar_serial_qr_float) {
  typedef KokkosBlas::Algo::QR::Unblocked AlgoTagType;
  testQR<TestDevice, float, AlgoTagType>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, batched_scalar_serial_qr_double) {
  typedef KokkosBlas::Algo::QR::Unblocked AlgoTagType;
  testQR<TestDevice, double, AlgoTagType>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT)
TEST_F(TestCategory, batched_scalar_serial_qr_scomplex) {
  typedef KokkosBlas::Algo::QR::Unblocked AlgoTagType;
  testQR<TestDevice, Kokkos::complex<float>, AlgoTagType>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
TEST_F(TestCategory, batched_scalar_serial_qr_dcomplex) {
  typedef KokkosBlas::Algo::QR::Unblocked AlgoTagType;
  testQR<TestDevice, Kokkos::complex<double>, AlgoTagType>();
}
#endif
