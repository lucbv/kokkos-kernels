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
/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

#include "KokkosBlas1_set.hpp"
#include "KokkosBatched_Copy_Decl.hpp"
#include "KokkosBlas2_team_gemv_spec.hpp"
#include "KokkosBatched_Trsv_Decl.hpp"
#include "KokkosBatched_QR_Decl.hpp"
#include "KokkosBatched_ApplyQ_Decl.hpp"

#include "KokkosKernels_TestUtils.hpp"

using namespace KokkosBatched;

namespace Test {

template <typename DeviceType, typename MatrixViewType, typename VectorViewType,
          typename WorkViewType, typename AlgoTagType>
struct Functor_TestBatchedTeamVectorQR {
  using execution_space = typename DeviceType::execution_space;
  MatrixViewType _a;
  VectorViewType _x, _b, _t;
  WorkViewType _w;

  KOKKOS_INLINE_FUNCTION
  Functor_TestBatchedTeamVectorQR(const MatrixViewType &a,
                                  const VectorViewType &x,
                                  const VectorViewType &b,
                                  const VectorViewType &t,
                                  const WorkViewType &w)
      : _a(a), _x(x), _b(b), _t(t), _w(w) {}

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType &member) const {
    typedef typename MatrixViewType::non_const_value_type value_type;
    const value_type one(1), zero(0), add_this(10);

    const int k = member.league_rank();
    auto aa     = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto bb     = Kokkos::subview(_b, k, Kokkos::ALL());
    auto xx     = Kokkos::subview(_x, k, Kokkos::ALL());
    auto tt     = Kokkos::subview(_t, k, Kokkos::ALL());
    auto ww     = Kokkos::subview(_w, k, Kokkos::ALL());

    // make diagonal dominant
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, aa.extent(0)),
                         [&](const int &i) { aa(i, i) += add_this; });

    /// xx = 1
    KokkosBlas::TeamVectorSet<MemberType>::invoke(member, one, xx);
    member.team_barrier();

    /// bb = AA*xx
    KokkosBlas::TeamVectorGemv<MemberType, Trans::NoTranspose,
                               Algo::Gemv::Unblocked>::invoke(member, one, aa,
                                                              xx, zero, bb);
    member.team_barrier();

    /// AA = QR
    TeamVectorQR<MemberType, AlgoTagType>::invoke(member, aa, tt, ww);
    member.team_barrier();

    /// xx = bb;
    TeamVectorCopy<MemberType, Trans::NoTranspose, 1>::invoke(member, bb, xx);
    member.team_barrier();

    /// xx = Q^{T}xx;
    TeamVectorApplyQ<MemberType, Side::Left, Trans::Transpose,
                     Algo::ApplyQ::Unblocked>::invoke(member, aa, tt, xx, ww);
    member.team_barrier();

    /// xx = R^{-1} xx
    TeamVectorTrsv<MemberType, Uplo::Upper, Trans::NoTranspose, Diag::NonUnit,
                   Algo::Trsv::Unblocked>::invoke(member, one, aa, xx);
  }

  inline void run() {
    typedef typename MatrixViewType::non_const_value_type value_type;
    std::string name_region("KokkosBatched::Test::TeamVectorQR");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());

    const int league_size = _a.extent(0);
    Kokkos::TeamPolicy<execution_space> policy(league_size, Kokkos::AUTO);

    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();
  }
};

template <typename DeviceType, typename MatrixViewType, typename VectorViewType,
          typename WorkViewType, typename AlgoTagType>
struct Functor_TestBatchedTeamVectorQR_rectangular {
  using execution_space = typename DeviceType::execution_space;
  MatrixViewType _a;
  MatrixViewType _b;
  VectorViewType _t;
  WorkViewType _w;

  KOKKOS_INLINE_FUNCTION
  Functor_TestBatchedTeamVectorQR_rectangular(const MatrixViewType &a,
                                              const MatrixViewType &b,
                                              const VectorViewType &t,
                                              const WorkViewType &w)
      : _a(a), _b(b), _t(t), _w(w) {}

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType &member) const {
    using value_type = typename MatrixViewType::non_const_value_type;
    // const value_type one = Kokkos::ArithTraits<value_type>::one();
    const value_type zero  = Kokkos::ArithTraits<value_type>::zero();
    const value_type add_this(10);

    const int k = member.league_rank();
    auto aa     = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto bb     = Kokkos::subview(_b, k, Kokkos::ALL(), Kokkos::ALL());
    auto tt     = Kokkos::subview(_t, k, Kokkos::ALL());
    auto ww     = Kokkos::subview(_w, k, Kokkos::ALL());

    // make diagonal dominant
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, aa.extent(0)),
                         [&](const int &i) { aa(i, i) += add_this; });

    /// AA = QR
    TeamVectorQR<MemberType, AlgoTagType>::invoke(member, aa, tt, ww);
    member.team_barrier();

    // assign upper tridiagonal part to bb
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, aa.extent(0)),
                         [&](const int &i) {
                           for (int j = 0; j < aa.extent_int(1); ++j)
                             bb(i, j) = (i <= j ? aa(i, j) : zero);
                         });
    member.team_barrier();

    // Multiply with Q
    TeamVectorApplyQ<MemberType, Side::Left, Trans::NoTranspose,
                     Algo::ApplyQ::Unblocked>::invoke(member, aa, tt, bb, ww);
    member.team_barrier();
  }

  inline void run() {
    using value_type = typename MatrixViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::TeamVectorQR_rectangular");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());

    const int league_size = _a.extent(0);
    Kokkos::TeamPolicy<execution_space> policy(league_size, Kokkos::AUTO);

    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();
  }
};

template <typename DeviceType, typename MatrixViewType, typename VectorViewType,
          typename WorkViewType, typename AlgoTagType>
struct Functor_TestBatchedTeamVectorQR_analytic {
  using execution_space = typename DeviceType::execution_space;

  struct computeQR{};
  struct QtimesR{};
  struct QtimesI{};

  MatrixViewType _a;
  MatrixViewType _b;
  VectorViewType _t;
  WorkViewType _w;

  const int numRows;
  const int numCols;

  typename MatrixViewType::HostMirror a_h;
  typename MatrixViewType::HostMirror b_h;

  KOKKOS_INLINE_FUNCTION
  Functor_TestBatchedTeamVectorQR_analytic(const MatrixViewType &a,
					   const MatrixViewType &b,
					   const VectorViewType &t,
					   const WorkViewType &w)
    : _a(a), _b(b), _t(t), _w(w), numRows(a.extent(1)), numCols(a.extent(2)) {
    a_h = Kokkos::create_mirror_view(_a);
    b_h = Kokkos::create_mirror_view(_b);
  }

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const computeQR&, const MemberType &member) const {

    const int k = member.league_rank();
    auto aa     = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto bb     = Kokkos::subview(_b, k, Kokkos::ALL(), Kokkos::ALL());
    auto tt     = Kokkos::subview(_t, k, Kokkos::ALL());
    auto ww     = Kokkos::subview(_w, k, Kokkos::ALL());

    /// AA = QR
    TeamVectorQR<MemberType, AlgoTagType>::invoke(member, aa, tt, ww);
    member.team_barrier();
  }

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const QtimesI&, const MemberType &member) const {
    using value_type = typename MatrixViewType::non_const_value_type;
    const value_type one = Kokkos::ArithTraits<value_type>::one();
    const value_type zero  = Kokkos::ArithTraits<value_type>::zero();

    const int k = member.league_rank();
    auto aa     = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto bb     = Kokkos::subview(_b, k, Kokkos::ALL(), Kokkos::ALL());
    auto tt     = Kokkos::subview(_t, k, Kokkos::ALL());
    auto ww     = Kokkos::subview(_w, k, Kokkos::ALL());

    // assign I to bb
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, aa.extent(0)),
                         [&](const int &i) {
                           for (int j = 0; j < aa.extent_int(1); ++j)
                             bb(i, j) = (i == j ? one : zero);
                         });
    member.team_barrier();

    // Multiply with Q
    TeamVectorApplyQ<MemberType, Side::Left, Trans::NoTranspose,
                     Algo::ApplyQ::Unblocked>::invoke(member, aa, tt, bb, ww);
    member.team_barrier();
  }

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const QtimesR&, const MemberType &member) const {
    using value_type = typename MatrixViewType::non_const_value_type;
    const value_type zero  = Kokkos::ArithTraits<value_type>::zero();

    const int k = member.league_rank();
    auto aa     = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto bb     = Kokkos::subview(_b, k, Kokkos::ALL(), Kokkos::ALL());
    auto tt     = Kokkos::subview(_t, k, Kokkos::ALL());
    auto ww     = Kokkos::subview(_w, k, Kokkos::ALL());

    // assign I to bb
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, aa.extent(0)),
                         [&](const int &i) {
                           for (int j = 0; j < aa.extent_int(1); ++j)
                             bb(i, j) = (i <= j ? aa(i, j) : zero);
                         });
    member.team_barrier();

    // Multiply with Q
    TeamVectorApplyQ<MemberType, Side::Left, Trans::NoTranspose,
                     Algo::ApplyQ::Unblocked>::invoke(member, aa, tt, bb, ww);
    member.team_barrier();
  }

  inline void compute_factors(const std::vector<double>& Rref) {
    using value_type = typename MatrixViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::TeamVectorQR_analytical::computeQR");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());

    const int league_size = _a.extent(0);
    Kokkos::TeamPolicy<execution_space, computeQR> policy(league_size, Kokkos::AUTO);

    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();

    Kokkos::deep_copy(a_h, _a);
    const int factor  = (std::signbit(a_h(0,0,0)) == std::signbit(Rref[0])) ? 1 : -1;
    const int matIdx  = static_cast<int>(_a.extent(0)/2);
    for(int i = 0; i < numRows; ++i) {
      for(int j = i; j < numCols; ++j) {
	if(Kokkos::abs(a_h(matIdx, i, j) - factor*Rref[numCols*i + j]) > 10e-6) {
	  std::cout << "R(" << i << ", " << j << ")=" << a_h(matIdx, i, j)
		    << ", Rref(" << i << ", " << j << ")=" << Rref[numCols*i + j] << std::endl;
	}
      }
    }
  }

  inline void multiplyQandI(const std::vector<double>& Qref) {
    using value_type = typename MatrixViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::TeamVectorQR_analytical::QtimesI");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());

    const int league_size = _a.extent(0);
    Kokkos::TeamPolicy<execution_space, QtimesI> policy(league_size, Kokkos::AUTO);

    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();

    Kokkos::deep_copy(b_h, _b);
    const int factor = (std::signbit(b_h(0,0,0)) == std::signbit(Qref[0])) ? 1 : -1;
    const int matIdx = static_cast<int>(_a.extent(0)/2);
    for(int i = 0; i < numRows; ++i) {
      for(int j = 0; j < numCols; ++j) {
	if(Kokkos::abs(b_h(matIdx, i, j) - factor*Qref[numCols*i + j]) > 10e-6) {
	  std::cout << "Q(" << i << ", " << j << ")=" << b_h(matIdx, i, j)
		    << ", Qref(" << i << ", " << j << ")=" << Qref[numCols*i + j]
		    << ", diff=" << Kokkos::abs(b_h(matIdx, i, j) - factor*Qref[numCols*i + j]) << std::endl;
	}
      }
    }
  }

  inline void multiplyQandR(const std::vector<double>& Aref) {
    using value_type = typename MatrixViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::TeamVectorQR_analytical::QtimesR");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());

    const int league_size = _a.extent(0);
    Kokkos::TeamPolicy<execution_space, QtimesR> policy(league_size, Kokkos::AUTO);

    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();

    Kokkos::deep_copy(b_h, _b);
    const int matIdx = static_cast<int>(_a.extent(0)/2);
    for(int i = 0; i < numRows; ++i) {
      for(int j = 0; j < numCols; ++j) {
	if(Kokkos::abs(b_h(matIdx, i, j) - Aref[numCols*i + j]) > 10e-6) {
	  std::cout << "A(" << i << ", " << j << ")=" << b_h(matIdx, i, j)
		    << ", Aref(" << i << ", " << j << ")=" << Aref[numCols*i + j] << std::endl;
	}
      }
    }
  }
};

template <typename DeviceType, typename MatrixViewType, typename VectorViewType,
          typename WorkViewType, typename AlgoTagType>
void impl_test_batched_qr(const int N, const int BlkSize) {
  typedef typename MatrixViewType::non_const_value_type value_type;
  typedef Kokkos::ArithTraits<value_type> ats;
  const value_type one(1);
  /// randomized input testing views
  MatrixViewType a("a", N, BlkSize, BlkSize);
  VectorViewType x("x", N, BlkSize);
  VectorViewType b("b", N, BlkSize);
  VectorViewType t("t", N, BlkSize);
  WorkViewType w("w", N, BlkSize);

  Kokkos::fence();

  Kokkos::Random_XorShift64_Pool<typename DeviceType::execution_space> random(
      13718);
  Kokkos::fill_random(a, random, value_type(1.0));

  Kokkos::fence();

  Functor_TestBatchedTeamVectorQR<DeviceType, MatrixViewType, VectorViewType,
                                  WorkViewType, AlgoTagType>(a, x, b, t, w)
      .run();

  Kokkos::fence();

  /// for comparison send it to host
  typename VectorViewType::HostMirror x_host = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(x_host, x);

  /// check x = 1; this eps is about 1e-14
  typedef typename ats::mag_type mag_type;
  const mag_type eps = 1e3 * ats::epsilon();

  for (int k = 0; k < N; ++k) {
    for (int i = 0; i < BlkSize; ++i) {
      const mag_type sum  = ats::abs(x_host(k, i));
      const mag_type diff = ats::abs(x_host(k, i) - one);
      EXPECT_NEAR_KK(diff / sum, mag_type(0), eps);
    }
  }
}

template <typename DeviceType, typename MatrixViewType, typename VectorViewType,
          typename WorkViewType, typename AlgoTagType>
void impl_test_batched_qr_rectangular(const int N, const int BlkSize1,
                                      const int BlkSize2) {
  typedef typename MatrixViewType::non_const_value_type value_type;
  typedef Kokkos::ArithTraits<value_type> ats;
  // const value_type one = Kokkos::ArithTraits<value_type>::one();

  /// randomized input testing views
  MatrixViewType a("a", N, BlkSize1, BlkSize2);
  MatrixViewType b("b", N, BlkSize1, BlkSize2);
  VectorViewType t("t", N, BlkSize2);
  WorkViewType w("w", N, 20 * BlkSize1);

  Kokkos::fence();

  Kokkos::Random_XorShift64_Pool<typename DeviceType::execution_space> random(
      13718);
  Kokkos::fill_random(a, random, value_type(1.0));

  Kokkos::fence();

  Functor_TestBatchedTeamVectorQR_rectangular<
      DeviceType, MatrixViewType, VectorViewType, WorkViewType, AlgoTagType>(
      a, b, t, w)
      .run();

  Kokkos::fence();

  /// for comparison send it to host
  typename MatrixViewType::HostMirror a_host = Kokkos::create_mirror_view(a);
  Kokkos::deep_copy(a_host, a);
  typename MatrixViewType::HostMirror b_host = Kokkos::create_mirror_view(b);
  Kokkos::deep_copy(b_host, b);

  /// check x = 1; this eps is about 1e-14
  typedef typename ats::mag_type mag_type;
  const mag_type eps = 1e3 * ats::epsilon();

  for (int k = 0; k < N; ++k) {
    for (int i = 0; i < BlkSize1; ++i) {
      for (int j = 0; j < BlkSize2; ++j) {
        const mag_type diff = ats::abs(a_host(k, i, j) - b_host(k, i, j));
        printf("k = %d, i = %d, j= %d, a %e b %e diff %e \n", k, i, j,
               a_host(k, i, j), b_host(k, i, j), diff);
        EXPECT_NEAR_KK(diff, mag_type(0), eps);
      }
    }
  }
}

template <typename DeviceType, typename MatrixViewType, typename VectorViewType,
          typename WorkViewType, typename AlgoTagType>
void impl_test_batched_qr_analytic(const int N) {
  // A = [12  -51    4]
  //     [ 6  167  -68]
  //     [-4   24  -41]
  //
  // Q = [ 6/7  -69/175  58/175]
  //     [ 3/7  158/175  -6/175]
  //     [-2/7    6/25   33/35 ]
  //
  // R = [14   21  -14]
  //     [ 0  175  -70]
  //     [ 0    0  -35]
  {
    MatrixViewType a("a", N, 3, 3);
    MatrixViewType b("b", N, 3, 3);
    VectorViewType t("t", N, 3);
    WorkViewType w("w", N, 20 * 3);

    std::vector<double> Aref = {12, -51, 4, 6, 167, -68, -4, 24, -41};
    std::vector<double> Qref = {6./7., -69./175, 58./175, 3./7, 158./175, -6./175, -2./7, 6./35, 33./35};
    std::vector<double> Rref = {14, 21, -14, 0, 175, -70, 0, 0, -35};

    typename MatrixViewType::HostMirror a_h = Kokkos::create_mirror_view(a);
    typename MatrixViewType::HostMirror b_h = Kokkos::create_mirror_view(b);
    for(int matIdx = 0; matIdx < N; ++matIdx) {
      a_h(matIdx, 0, 0) = 12; a_h(matIdx, 0, 1) = -51; a_h(matIdx, 0, 2) =   4;
      a_h(matIdx, 1, 0) =  6; a_h(matIdx, 1, 1) = 167; a_h(matIdx, 1, 2) = -68;
      a_h(matIdx, 2, 0) = -4; a_h(matIdx, 2, 1) =  24; a_h(matIdx, 2, 2) = -41;
    }
    Kokkos::deep_copy(a, a_h);

    Functor_TestBatchedTeamVectorQR_analytic<
      DeviceType, MatrixViewType, VectorViewType, WorkViewType, AlgoTagType> tester(a, b, t, w);

    tester.compute_factors(Rref);

    tester.multiplyQandI(Qref);

    tester.multiplyQandR(Aref);
  }

  // A = [12  -51    4  -2     24]
  //     [ 6  167  -68  34     12]
  //     [-4   24  -41  20.5   -8]
  //     [ 0    0    0   3     16]
  //     [ 0    0    0   4    -12]
  //
  // Q = [ 6/7  -69/175  58/175    0     0]
  //     [ 3/7  158/175  -6/175    0     0]
  //     [-2/7    6/25   33/35     0     0]
  //     [   0        0       0  3/5   4/5]
  //     [   0        0       0  4/5  -3/5]
  //
  // R = [ 14   21  -14    7    28]
  //     [  0  175  -70   35     0]
  //     [  0    0  -35   17.5   0]
  //     [  0    0    0    5     0]
  //     [  0    0    0    0    20]
  {
    MatrixViewType a("a", N, 5, 5);
    MatrixViewType b("b", N, 5, 5);
    VectorViewType t("t", N, 5);
    WorkViewType w("w", N, 20 * 5);

    std::vector<double> Aref = {12, -51,   4,   -2,  24,
				 6, 167, -68,   34,  12,
				-4,  24, -41, 20.5,  -8,
				 0,   0,   0,    3,  16,
				 0,   0,   0,    4, -12};
    std::vector<double> Qref = { 6./7, -69./175, 58./175,    0,     0,
				 3./7, 158./175, -6./175,    0,     0,
				-2./7,    6./35,  33./35,    0,     0,
				    0,        0,       0, 3./5,  4./5,
				    0,        0,       0, 4./5, -3./5};
    std::vector<double> Rref = {14, 21, -14,    7, 28,
				0, 175, -70,   35,  0,
				0,   0, -35, 17.5,  0,
				0,   0,   0,    5,  0,
				0,   0,   0,    0, 20};

    typename MatrixViewType::HostMirror a_h = Kokkos::create_mirror_view(a);
    typename MatrixViewType::HostMirror b_h = Kokkos::create_mirror_view(b);
    for(int matIdx = 0; matIdx < N; ++matIdx) {
      for(int i = 0; i < a.extent_int(1); ++i) {
	for(int j = 0; j < a.extent_int(2); ++j) {
	  a_h(matIdx, i, j) = Aref[i*a.extent_int(2) + j];
	}
      }
    }
    Kokkos::deep_copy(a, a_h);

    Functor_TestBatchedTeamVectorQR_analytic<
      DeviceType, MatrixViewType, VectorViewType, WorkViewType, AlgoTagType> tester(a, b, t, w);

    tester.compute_factors(Rref);

    tester.multiplyQandI(Qref);

    tester.multiplyQandR(Aref);
  }
}

}  // namespace Test

template <typename DeviceType, typename ValueType, typename AlgoTagType>
int test_batched_qr() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    typedef Kokkos::View<ValueType ***, Kokkos::LayoutLeft, DeviceType>
        MatrixViewType;
    typedef Kokkos::View<ValueType **, Kokkos::LayoutLeft, DeviceType>
        VectorViewType;
    typedef Kokkos::View<ValueType **, Kokkos::LayoutRight, DeviceType>
        WorkViewType;
    Test::impl_test_batched_qr<DeviceType, MatrixViewType, VectorViewType,
                               WorkViewType, AlgoTagType>(0, 10);
    // for (int i = 1; i < 10; ++i) {
    //   // printf("Testing: LayoutLeft,  Blksize %d\n", i);
    //   printf("Testing: LayoutLeft,  Blksize %d\n", i);
    //   Test::impl_test_batched_qr<DeviceType, MatrixViewType, VectorViewType,
    //                              WorkViewType, AlgoTagType>(1024, i);
    //   printf("Testing: LayoutLeft,  Blksize %d %d\n", 2 * i, i);
    //   Test::impl_test_batched_qr_rectangular<DeviceType, MatrixViewType,
    // 					     VectorViewType, WorkViewType,
    // 					     AlgoTagType>(1024, i, i);
    // }
    Test::impl_test_batched_qr_analytic<DeviceType, MatrixViewType,
					VectorViewType, WorkViewType,
					AlgoTagType>(10);
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    typedef Kokkos::View<ValueType ***, Kokkos::LayoutRight, DeviceType>
        MatrixViewType;
    typedef Kokkos::View<ValueType **, Kokkos::LayoutRight, DeviceType>
        VectorViewType;
    typedef Kokkos::View<ValueType **, Kokkos::LayoutRight, DeviceType>
        WorkViewType;
    Test::impl_test_batched_qr<DeviceType, MatrixViewType, VectorViewType,
                               WorkViewType, AlgoTagType>(0, 10);
    // for (int i = 1; i < 10; ++i) {
    //   // printf("Testing: LayoutRight, Blksize %d\n", i);
    //   printf("Testing: LayoutRight, Blksize %d\n", i);
    //   Test::impl_test_batched_qr<DeviceType, MatrixViewType, VectorViewType,
    //                              WorkViewType, AlgoTagType>(1024, i);
    //   Test::impl_test_batched_qr_rectangular<DeviceType, MatrixViewType,
    //                                          VectorViewType, WorkViewType,
    //                                          AlgoTagType>(1024, i, i);
    // }
    Test::impl_test_batched_qr_analytic<DeviceType, MatrixViewType,
					VectorViewType, WorkViewType,
					AlgoTagType>(10);
  }
#endif

  return 0;
}
