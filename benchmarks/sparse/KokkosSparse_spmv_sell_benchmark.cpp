// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

// Headers needed to create initial data
// and to check results at the end
#include <KokkosKernels_IOUtils.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include "KokkosKernels_default_types.hpp"
#include "KokkosKernels_TestStringUtils.hpp"
#include "KokkosKernels_benchmark_utilities.hpp"

// Headers for benchmark library
#include <benchmark/benchmark.h>
#include "Benchmark_Context.hpp"

// Headers for spmv
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>

namespace {

struct spmv_parameters {
  int N, offset, numvecs;
  std::string mode;
  std::string filename;
  std::string alg;
  std::string tpl;

  spmv_parameters(const int N_) : N(N_), offset(0), numvecs(1), mode(""), filename(""), alg(""), tpl("") {}
};

void print_options() {
  std::cerr << "Options\n" << std::endl;

  std::cerr << benchmark::list_common_options();

  std::cerr << "\t[Optional] --mode        :: whether to run a suite of "
            << "automated test or manually define one (auto, manual)" << std::endl;
  std::cerr << "\t[Optional] --repeat      :: how many times to repeat overall "
            << "test" << std::endl;
  std::cerr << "  -n [N]          :: generate a semi-random banded (band size "
               "0.01xN)\n"
               "NxN matrix with average of 10 entries per row."
            << std::endl;
  std::cerr << "\t[Optional] --alg           :: the algorithm to run (default, "
               "native, merge)"
            << std::endl;
  std::cerr << "\t[Optional] --TPL       :: when available and compatible with "
               "alg, a TPL can be used (cusparse, rocsparse, MKL)"
            << std::endl;
  std::cerr << "  -f [file]       : Read in Matrix Market formatted text file"
            << " 'file'." << std::endl;
  std::cerr << "  --offset [O]    : Subtract O from every index.\n"
            << "                    Useful in case the matrix market file is "
               "not 0 based."
            << std::endl;
  std::cerr << "  --num_vecs      : The number of vectors stored in X and Y" << std::endl;
}  // print_options

void parse_inputs(int argc, char** argv, spmv_parameters& params) {
  for (int i = 1; i < argc; ++i) {
    if (benchmark::check_arg_int(i, argc, argv, "-n", params.N)) {
      ++i;
    } else if (benchmark::check_arg_str(i, argc, argv, "--mode", params.mode)) {
      if ((params.mode != "") && (params.mode != "auto") && (params.mode != "manual")) {
        throw std::runtime_error("--mode can only be an empty string, `auto` or `manual`!");
      }
      ++i;
    } else if (benchmark::check_arg_str(i, argc, argv, "--alg", params.alg)) {
      if ((params.alg != "") && (params.alg != "default") && (params.alg != "native") && (params.alg != "merge")) {
        throw std::runtime_error(
            "--alg can only be an empty string, `default`, `native` or "
            "`merge`!");
      }
      ++i;
    } else if (benchmark::check_arg_str(i, argc, argv, "--TPL", params.tpl)) {
      ++i;
    } else if (benchmark::check_arg_str(i, argc, argv, "-f", params.filename)) {
      ++i;
    } else if (benchmark::check_arg_int(i, argc, argv, "--offset", params.offset)) {
      ++i;
    } else if (benchmark::check_arg_int(i, argc, argv, "--num_vecs", params.numvecs)) {
      ++i;
    } else {
      print_options();
      KK_USER_REQUIRE_MSG(false, "Unrecognized command line argument #" << i << ": " << argv[i]);
    }
  }
}  // parse_inputs

template <class execution_space>
void run_spmv(benchmark::State& state, const spmv_parameters& inputs) {
  using matrix_type  = KokkosSparse::CrsMatrix<double, int, execution_space, void, int>;
  using mv_type      = Kokkos::View<double*, execution_space>;
  using handle_t     = KokkosSparse::SPMVHandle<execution_space, matrix_type, mv_type, mv_type>;
  using sell_matrix_type = KokkosSparse::Experimental::SellMatrix<double, int, execution_space, void, int>;

  KokkosSparse::SPMVAlgorithm spmv_alg;
  if ((inputs.alg == "default") || (inputs.alg == "")) {
    spmv_alg = KokkosSparse::SPMVAlgorithm::SPMV_DEFAULT;
  } else if (inputs.alg == "native") {
    spmv_alg = KokkosSparse::SPMVAlgorithm::SPMV_NATIVE;
  } else if (inputs.alg == "merge") {
    spmv_alg = KokkosSparse::SPMVAlgorithm::SPMV_MERGE_PATH;
  } else {
    throw std::runtime_error("invalid spmv algorithm");
  }
  handle_t handle(spmv_alg);

  // Create test matrix
  srand(17312837);
  matrix_type A;
  int nnz = 10 * inputs.N;
  if (inputs.filename == "") {
    A = KokkosSparse::Impl::kk_generate_sparse_matrix<matrix_type>(inputs.N, inputs.N, nnz, 0, 0.01 * inputs.N);
  } else {
    A = KokkosSparse::Impl::read_kokkos_crst_matrix<matrix_type>(inputs.filename.c_str());
  }

  int max_row_length = 0;
  Kokkos::parallel_reduce("Max Row Length", Kokkos::RangePolicy(0, A.numRows()),
			  KOKKOS_LAMBDA(const int& rowIdx, int& lmax_row_length) {
			    int row_length = A.graph.row_map(rowIdx + 1) - A.graph.row_map(rowIdx);
			    lmax_row_length = (lmax_row_length < row_length) ? row_length : lmax_row_length;
			  }, Kokkos::Max<int>(max_row_length));

  const int padded_nnz = max_row_length * A.numRows();
  std::cout << "A statistics\n"
	    << "   num rows:           " << A.numRows() << "\n"
	    << "   num cols:           " << A.numCols() << "\n"
	    << "   nnz:                " << A.nnz() << "\n"
	    << "   max row:            " << max_row_length << "\n"
	    << "   padded nnz:         " << padded_nnz << "\n"
	    << "   num rows per slice: " << A.numRows() << "\n"
	    << "   num slices:         " << 1 << std::endl;

  typename sell_matrix_type::offsets_type slice_offsets("sell slice offsets", 2);
  typename sell_matrix_type::entries_type entries("sell indices", padded_nnz);
  typename sell_matrix_type::values_type values("sell values", padded_nnz);
  Kokkos::deep_copy(entries, -1);

  // Fill the sell matrix data
  Kokkos::parallel_for("Fill Sell Matrix", Kokkos::RangePolicy(0, A.numRows()),
		       KOKKOS_LAMBDA(const int& rowIdx) {
			 for(int entryIdx = A.graph.row_map(rowIdx); entryIdx < A.graph.row_map(rowIdx + 1); ++entryIdx) {
			   int entry_count = entryIdx - A.graph.row_map(rowIdx);
			   entries(entry_count * A.numRows() + rowIdx) = A.graph.entries(entryIdx);
			   values(entry_count * A.numRows() + rowIdx)  = A.values(entryIdx);
			 }
			 if (rowIdx == 0) slice_offsets(1) = A.numRows();
		       });

  sell_matrix_type sellA(A.numRows(), A.numCols(), A.nnz(), padded_nnz, A.numRows(), slice_offsets, entries, values);

  // create input vectors
  mv_type x("X", A.numRows());
  mv_type y("Y", A.numCols());

  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);
  Kokkos::fill_random(x, rand_pool, 10);
  Kokkos::fill_random(y, rand_pool, 10);
  Kokkos::fence();

  // Run the actual experiments
  for (auto _ : state) {
    KokkosSparse::Experimental::spmv(KokkosSparse::NoTranspose, 1.0, sellA, x, 0.0, y);
    Kokkos::fence();
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);

  benchmark::Initialize(&argc, argv);
  benchmark::SetDefaultTimeUnit(benchmark::kMillisecond);
  KokkosKernelsBenchmark::add_benchmark_context(true);

  benchmark::CommonInputParams common_params;
  benchmark::parse_common_options(argc, argv, common_params);

  std::string bench_name = "KokkosSparse_spmv_sell";

  // Set input parameters, default to random 100000x100000
  spmv_parameters inputs(100000);
  parse_inputs(argc, argv, inputs);

  if ((inputs.mode == "") || (inputs.mode == "auto")) {
    for (int n : {10000, 20000, 40000, 100000, 250000, 1000000}) {
      for (int nv : {1, 2, 3, 4, 10}) {
        inputs.N       = n;
        inputs.numvecs = nv;
        KokkosKernelsBenchmark::register_benchmark_real_time(bench_name.c_str(),
                                                             run_spmv<Kokkos::DefaultExecutionSpace>, {"n", "nv"},
                                                             {inputs.N, inputs.numvecs}, common_params.repeat, inputs);
      }
    }
  } else {
    // Google benchmark will report the wrong n if an input file matrix is used.
    KokkosKernelsBenchmark::register_benchmark_real_time(bench_name.c_str(), run_spmv<Kokkos::DefaultExecutionSpace>,
                                                         {"n"}, {inputs.N}, common_params.repeat, inputs);
  }

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  Kokkos::finalize();

  return 0;
}
