/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <cstdio>

#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <limits.h>
#include <cmath>
#include <unordered_map>

#ifdef HAVE_CUSPARSE
#include <cusparse.h>
#endif

#include <Kokkos_Core.hpp>
#include <matrix_market.hpp>

#include <KokkosKernels_SPMV.hpp>
#include <Kokkos_SPMV.hpp>
#include <Kokkos_SPMV_Inspector.hpp>
#include <CuSparse_SPMV.hpp>
#include <MKL_SPMV.hpp>

#include <KokkosSparse_spmv.hpp>

#ifdef _OPENMP
#include <OpenMPStatic_SPMV.hpp>
#include <OpenMPDynamic_SPMV.hpp>
#include <OpenMPSmartStatic_SPMV.hpp>
#endif

enum {KOKKOS, MKL, CUSPARSE, KK_KERNELS, KK_KERNELS_INSP, KK_INSP, OMP_STATIC, OMP_DYNAMIC, OMP_INSP};
enum {AUTO, DYNAMIC, STATIC};
enum {REF, NEW};

#ifdef INT64
typedef long long int LocalOrdinalType;
#else
typedef int LocalOrdinalType;
#endif


template< typename ScalarType , typename OrdinalType>
int SparseMatrix_generate(OrdinalType nrows, OrdinalType ncols, OrdinalType &nnz, OrdinalType varianz_nel_row, OrdinalType width_row, ScalarType* &values, OrdinalType* &rowPtr, OrdinalType* &colInd)
{
  rowPtr = new OrdinalType[nrows+1];

  OrdinalType elements_per_row = nnz/nrows;
  srand(13721);
  rowPtr[0] = 0;
  for(int row=0;row<nrows;row++)
  {
    int varianz = (1.0*rand()/INT_MAX-0.5)*varianz_nel_row;
    rowPtr[row+1] = rowPtr[row] + elements_per_row+varianz;
  }
  nnz = rowPtr[nrows];
  values = new ScalarType[nnz];
  colInd = new OrdinalType[nnz];
  for(int row=0;row<nrows;row++)
  {
         for(int k=rowPtr[row];k<rowPtr[row+1];k++)
         {
                int pos = (1.0*rand()/INT_MAX-0.5)*width_row+row;
                if(pos<0) pos+=ncols;
                if(pos>=ncols) pos-=ncols;
                colInd[k]= pos;
                values[k] = 100.0*rand()/INT_MAX-50.0;
         }
  }
  return nnz;
}


template<class execution_space>
int64_t spmv_launch_parameters(int64_t numRows,
                               int64_t nnz,
                               int numVecs,
                               int64_t rows_per_thread,
                               int& team_size,
                               int& vector_length) {
  int64_t rows_per_team;
  int64_t nnz_per_row = nnz/numRows;

  if(nnz_per_row < 1) nnz_per_row = 1;

  if(vector_length < 1) {
    vector_length = 1;
    #ifdef KOKKOS_ENABLE_CUDA
    if(std::is_same<Kokkos::Cuda, execution_space>::value)
      {
        while(vector_length < 32 && vector_length*6 < nnz_per_row) {
          vector_length *= 2;
        }
      }
    else
    #endif
      {
        while(vector_length < numVecs && vector_length < 4) {
          vector_length *= 2;
        }
      }
  }

  // Determine rows per thread
  if(rows_per_thread < 1) {
    #ifdef KOKKOS_ENABLE_CUDA
    if(std::is_same<Kokkos::Cuda,execution_space>::value)
      rows_per_thread = 1;
    else
    #endif
    {
      if(nnz_per_row < 20 && nnz > 5000000 ) {
        rows_per_thread = 256;
      } else
        rows_per_thread = 64;
    }
  }

  #ifdef KOKKOS_ENABLE_CUDA
  if(team_size < 1) {
    if(std::is_same<Kokkos::Cuda,execution_space>::value)
    { team_size = 256/vector_length; }
    else
    { team_size = 1; }
  }
  #endif

  rows_per_team = rows_per_thread * team_size;

  if(rows_per_team < 0) {
    int64_t nnz_per_team = 4096;
    int64_t conc = execution_space::concurrency();
    while((conc * nnz_per_team * 4> nnz)&&(nnz_per_team>256)) nnz_per_team/=2;
    rows_per_team = (nnz_per_team+nnz_per_row - 1)/nnz_per_row;
  }


  return rows_per_team;
}

template<typename Scalar>
int test_spmv_mv(int numRows, int numCols, int nnz, const int numVecs,
                 const double alpha, const double beta, int test, const char* filename,
                 const bool binaryfile, int rows_per_thread, int team_size, int vector_length,
                 int idx_offset, int schedule, int loop, const int verbose) {
  typedef KokkosSparse::CrsMatrix<Scalar,int,Kokkos::DefaultExecutionSpace,void,int> matrix_type;
  typedef typename matrix_type::non_const_value_type   value_type;
  typedef typename matrix_type::device_type            device_type;
  typedef typename matrix_type::memory_traits          memory_traits;
  typedef typename Kokkos::View<value_type**, Kokkos::LayoutLeft,  device_type, memory_traits> multivector_type;
  typedef typename multivector_type::HostMirror h_multivector_type;

  Scalar* val = NULL;
  int* row = NULL;
  int* col = NULL;

  if(verbose) { printf("Generating random CrsMatrix\n"); }
  srand(17312837);
  if(filename==NULL) {
    nnz = SparseMatrix_generate<Scalar,int>(numRows,numCols,nnz,nnz/numRows*0.2,numRows*0.01,val,row,col);
  } else {
    if(!binaryfile) {
      nnz = SparseMatrix_MatrixMarket_read<Scalar,int>(filename,numRows,numCols,nnz,val,row,col,false,idx_offset);
    } else {
      nnz = SparseMatrix_ReadBinaryFormat<Scalar,int>(filename,numRows,numCols,nnz,val,row,col);
    }
  }

  matrix_type A("CRS::A", numRows, numCols, nnz, val, row, col, false);

  multivector_type x1("X1", numCols, numVecs);
  multivector_type y1("Y1", numRows, numVecs);
  h_multivector_type h_x = Kokkos::create_mirror_view(x1);
  h_multivector_type h_y = Kokkos::create_mirror_view(y1);
  h_multivector_type h_y_compare = Kokkos::create_mirror(y1);

  typename matrix_type::StaticCrsGraphType::HostMirror h_graph = Kokkos::create_mirror(A.graph);
  typename matrix_type::values_type::HostMirror h_values = Kokkos::create_mirror_view(A.values);
  Kokkos::deep_copy(h_values, A.values);

  Kokkos::fence();
  if(verbose) { printf("Loading values in left and right hand side multivectors\n"); }
  for(int vecIdx = 0; vecIdx < numVecs; ++vecIdx) {
    for(int i = 0; i < numCols; i++) {
      h_x(i, vecIdx) = (value_type) (1.0*(rand()%40)-20.);
    }
    for(int i = 0; i < numRows; i++) {
      h_y(i, vecIdx) = (value_type) (1.0*(rand()%40)-20.);
    }
  }

  Kokkos::fence();
  if(verbose) { printf("Computing serial spmv for error check\n"); }
  // Error Check Gold Values
  for(int i = 0; i < numRows; i++) {
    int start = h_graph.row_map(i);
    int end = h_graph.row_map(i + 1);
    for(int vecIdx = 0; vecIdx < numVecs; ++vecIdx) {
      h_y_compare(i, vecIdx) = beta*h_y(i, vecIdx);
    }

    for(int j = start; j < end; j++) {
      int idx = h_graph.entries(j);
      for(int vecIdx = 0; vecIdx < numVecs; ++vecIdx) {
        h_y_compare(i, vecIdx) += alpha*h_values(j)*h_x(idx, vecIdx);
      }
    }
  }

  Kokkos::fence();
  if(verbose) { printf("Copy data to device\n"); }
  Kokkos::deep_copy(x1, h_x);
  Kokkos::deep_copy(y1, h_y);
  Kokkos::deep_copy(A.graph.entries,h_graph.entries);
  Kokkos::deep_copy(A.values,h_values);

  Kokkos::fence();
  if(verbose) { printf("Perform initial spmv to check for errors\n"); }
  KokkosSparse::spmv("N", alpha, A, x1, beta, y1);

  Kokkos::fence();
  if(verbose) { printf("Copy results of matvec to host before correctness check\n"); }
  // Error Check
  Kokkos::deep_copy(h_y, y1);

  Kokkos::fence();
  if(verbose) { printf("Check correctness against serial spmv results\n"); }
  Scalar error = 0.0;
  Scalar sum = 0.0;
  for(int vecIdx = 0; vecIdx < numVecs; ++vecIdx) {
    for(int i = 0; i < numRows; i++) {
      error += (h_y_compare(i, vecIdx) - h_y(i, vecIdx))*(h_y_compare(i, vecIdx) - h_y(i, vecIdx));
      sum += h_y_compare(i, vecIdx)*h_y_compare(i, vecIdx);
    }
  }

  int num_errors = 0;
  double total_error = 0;
  double total_sum = 0;
  num_errors += (error/(sum==0?1:sum))>1e-5?1:0;
  total_error += error;
  total_sum += sum;

  Kokkos::fence();
  if(verbose) { printf("Test performance of new spmv_mv implementation\n"); }
  // Benchmark new impl
  double min_new_time = 1.0e32;
  double max_new_time = 0.0;
  double ave_new_time = 0.0;
  for(int i = 0; i < loop; i++) {
    Kokkos::Timer timer;
    KokkosSparse::spmv("N", alpha, A, x1, beta, y1);
    Kokkos::fence();
    double time = timer.seconds();
    ave_new_time += time;
    if(time > max_new_time) max_new_time = time;
    if(time < min_new_time) min_new_time = time;
  }

  // Performance Output
  double matrix_size = 1.0*((nnz*(sizeof(Scalar)+sizeof(int)) + numRows*sizeof(int)))/1024/1024;
  double vector_size = 2.0*numRows*numVecs*sizeof(Scalar)/1024/1024;
  double vector_readwrite = (nnz+numCols)*numVecs*sizeof(Scalar)/1024/1024;

  double problem_size = matrix_size+vector_size;
  printf("NNZ NumRows NumCols ProblemSize(MB) AveBandwidth(GB/s) MinBandwidth(GB/s) MaxBandwidth(GB/s) AveGFlop MinGFlop MaxGFlop aveTime(ms) maxTime(ms) minTime(ms) numErrors\n");
  printf("%i %i %i %6.2lf ( %6.2lf %6.2lf %6.2lf ) ( %6.3lf %6.3lf %6.3lf ) ( %6.3lf %6.3lf %6.3lf ) %i RESULT\n",nnz, numRows,numCols,problem_size,
          (matrix_size+vector_readwrite)/ave_new_time*loop/1024, (matrix_size+vector_readwrite)/max_new_time/1024,(matrix_size+vector_readwrite)/min_new_time/1024,
          2.0*nnz*loop/ave_new_time/1e9, 2.0*nnz/max_new_time/1e9, 2.0*nnz/min_new_time/1e9,
          ave_new_time/loop*1000, max_new_time*1000, min_new_time*1000,
          num_errors);
  return (int)total_error;
}

void print_help() {
  printf("SPMV benchmark code written by Christian Trott.\n");
  printf("OpenMP implementations written by Simon Hammond (Sandia National Laboratories).\n\n");
  printf("Options:\n");
  printf("  -s [N]          : generate a semi-random banded (band size 0.01xN) NxN matrix\n");
  printf("                    with average of 10 entries per row.\n");
  printf("  -v              : Set the number of vectors stored in the rhs.\n");
  printf("  --test [OPTION] : Use different kernel implementations\n");
  printf("                    Options:\n");
  printf("                      kk,kk-kernels          (Kokkos/Trilinos)\n");
  printf("                      kk-insp                (Kokkos Structure Inspection)\n");
#ifdef _OPENMP
  printf("                      omp-dynamic,omp-static (Standard OpenMP)\n");
  printf("                      omp-insp               (OpenMP Structure Inspection)\n");
#endif
  printf("                      mkl,cusparse           (Vendor Libraries)\n\n");
  printf("  --schedule [SCH]: Set schedule for kk variant (static,dynamic,auto [ default ]).\n");
  printf("  -f [file]       : Read in Matrix Market formatted text file 'file'.\n");
  printf("  -fb [file]      : Read in binary Matrix files 'file'.\n");
  printf("  --write-binary  : In combination with -f, generate binary files.\n");
  printf("  --offset [O]    : Subtract O from every index.\n");
  printf("                    Useful in case the matrix market file is not 0 based.\n\n");
  printf("  -rpt [K]        : Number of Rows assigned to a thread.\n");
  printf("  -ts [T]         : Number of threads per team.\n");
  printf("  -vl [V]         : Vector-length (i.e. how many Cuda threads are a Kokkos 'thread').\n");
  printf("  -l [LOOP]       : How many spmv to run to aggregate average time. \n");
}

int main(int argc, char **argv)
{
 long long int size = 110503; // a prime number
 int numVecs = 4;
 int test=KOKKOS;
 //int type=-1;
 char* filename = NULL;
 bool binaryfile = false;
 bool write_binary = false;
 bool verbose = false;

 int rows_per_thread = -1;
 int vector_length = -1;
 int team_size = -1;
 int idx_offset = 0;
 int schedule=AUTO;
 int loop = 100;

 double alpha = 2.0, beta = 3.0;

 if(argc == 1) {
   print_help();
   return 0;
 }

 for(int i=0;i<argc;i++)
 {
  if((strcmp(argv[i],"-s")==0)) {size=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"-v")==0)) {numVecs=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"--alpha")==0)) {alpha=atof(argv[++i]); continue;}
  if((strcmp(argv[i],"--beta")==0))  {beta=atof(argv[++i]); continue;}
  if((strcmp(argv[i],"--test")==0)) {
    i++;
    if((strcmp(argv[i],"mkl")==0))
      test = MKL;
    if((strcmp(argv[i],"kk")==0))
      test = KOKKOS;
    if((strcmp(argv[i],"cusparse")==0))
      test = CUSPARSE;
    if((strcmp(argv[i],"kk-kernels")==0))
      test = KK_KERNELS;
    if((strcmp(argv[i],"kk-kernels-insp")==0))
      test = KK_KERNELS_INSP;
    if((strcmp(argv[i],"kk-insp")==0))
      test = KK_INSP;
#ifdef _OPENMP
    if((strcmp(argv[i],"omp-static") == 0))
      test = OMP_STATIC;
    if((strcmp(argv[i], "omp-dynamic") == 0))
      test = OMP_DYNAMIC;
    if((strcmp(argv[i], "omp-insp") == 0))
      test = OMP_INSP;
#endif
    continue;
  }
  if((strcmp(argv[i],"-f")==0)) {filename = argv[++i]; continue;}
  if((strcmp(argv[i],"-fb")==0)) {filename = argv[++i]; binaryfile = true; continue;}
  if((strcmp(argv[i],"-rpt")==0)) {rows_per_thread=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"-ts")==0)) {team_size=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"-vl")==0)) {vector_length=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"--offset")==0)) {idx_offset=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"--write-binary")==0)) {write_binary=true;}
  if((strcmp(argv[i],"--verbose")==0)) {verbose=true; continue;}
  if((strcmp(argv[i],"-l")==0)) {loop=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"--schedule")==0)) {
    i++;
    if((strcmp(argv[i],"auto")==0))
      schedule = AUTO;
    if((strcmp(argv[i],"dynamic")==0))
      schedule = DYNAMIC;
    if((strcmp(argv[i],"static")==0))
      schedule = STATIC;
    continue;
  }
  if((strcmp(argv[i],"--help")==0) || (strcmp(argv[i],"-h")==0)) {
    print_help();
    return 0;
  }
 }

 if(write_binary) {
   double* val = NULL;
   int* row = NULL;
   int* col = NULL;
   int numRows,numCols,nnz;
   SparseMatrix_WriteBinaryFormat<double,int>(filename, numRows, numCols, nnz,
                                              val, row, col, true, idx_offset);
   return 0;
 }

 Kokkos::initialize(argc,argv);

 printf("Problem parameters: matrix size=%d, number of rhs=%d\n", (int) size, numVecs);
 int total_errors = test_spmv_mv<double>(size, size, size*10, numVecs,
                                         alpha, beta, test, filename, binaryfile,
					 rows_per_thread, team_size, vector_length,
					 idx_offset, schedule, loop, verbose);

 if(total_errors == 0)
   printf("Kokkos::MultiVector Test: Passed\n");
 else
   printf("Kokkos::MultiVector Test: Failed\n");


  Kokkos::finalize();
}
