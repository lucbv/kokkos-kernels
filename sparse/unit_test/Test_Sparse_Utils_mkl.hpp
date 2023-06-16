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
// Note: Luc Berger-Vergiat 06/15/23
//       Only include this test if
//       compiling with MKL enabled.
#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "KokkosSparse_Utils_mkl.hpp"
#include "KokkosSparse_CrsMatrix.hpp"

// Adding oneMKL SYCL specific tests
#ifdef KOKKOS_ENABLE_SYCL
void test_onemkl_matrix_creation() {
  using execution_space = Kokkos::Experimental::SYCL;
  using memory_space    = Kokkos::Experimental::SYCLDeviceUSMSpace;
  using device_type     = Kokkos::Device<execution_space, memory_space>;
  bool check = true;

  constexpr int num_rows = 3;
  constexpr int num_cols = 3;
  constexpr int nnz = 5;

  using matrix_type = KokkosSparse::CrsMatrix<double, int, device_type, void, int>;
  using rowptr_type = typename matrix_type::row_map_type;
  using colind_type = typename matrix_type::index_type;
  using values_type = typename matrix_type::values_type;

  rowptr_type rowptr("row ptr", num_rows + 1);
  colind_type colinds("column indices", nnz);
  values_type values("values", nnz);

  matrix_type A("A", num_rows, num_cols, nnz, values, colinds, rowptr);

  execution_space exec{};

  oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
  oneapi::mkl::sparse::init_matrix_handle(&handle);
  oneapi::mkl::sparse::set_csr_data(exec.sycl_queue(), handle, A.numRows(), A.numCols(), oneapi::mkl::index_base::zero,
				    A.graph.row_map.data(), A.graph.entries.data(), A.values.data());
  oneapi::mkl::sparse::release_matrix_handle(exec.sycl_queue(), &handle);

  EXPECT_TRUE(check == true);
}

TEST_F(TestCategory, sparse_onemkl_matrix_creation) { test_onemkl_matrix_creation(); }
#endif

#endif  // check for MKL
