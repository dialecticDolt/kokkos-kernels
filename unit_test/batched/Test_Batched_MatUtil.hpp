/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

#include "KokkosBatched_Set_Decl.hpp"
#include "KokkosBatched_Set_Impl.hpp"

#include "KokkosBatched_Scale_Decl.hpp"
#include "KokkosBatched_Scale_Impl.hpp"

#include "KokkosKernels_TestUtils.hpp"

using namespace KokkosBatched::Experimental;

namespace Test {
        
  enum : int  { BatchedSet = 0,
                BatchedScale = 1 };
  
  struct KokkosKernelTag {};
  struct NaiveTag {};
  
  template<typename DeviceType, 
           typename ViewType, 
           typename ScalarType, 
           typename AlgoTagType, 
           int TestID>
  struct Functor {
          
    ScalarType _alpha;
    ViewType _a;

    KOKKOS_INLINE_FUNCTION
    Functor(const ScalarType alpha, 
            const ViewType &a) 
      : _alpha(alpha), _a(a) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator()(const KokkosKernelTag &, const int i) const {
      auto A = Kokkos::subview(_a, i, Kokkos::ALL(), Kokkos::ALL());
      switch (TestID) {
      case BatchedSet:   SerialSet  ::invoke(_alpha, A); break;
      case BatchedScale: SerialScale::invoke(_alpha, A); break;
      }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const NaiveTag &, const int i) const {
      auto A = Kokkos::subview(_a, i, Kokkos::ALL(), Kokkos::ALL());
      const int m = A.dimension_0(), n = A.dimension_1();
      switch (TestID) {
      case BatchedSet: {
        for (int i=0;i<m;++i) 
          for (int j=0;j<n;++j)
            A(i,j)  = _alpha;
        break;
      }
      case BatchedScale: {
        for (int i=0;i<m;++i) 
          for (int j=0;j<n;++j)
            A(i,j) *= _alpha;
        break;
      }
      }
    }

    inline
    int run() {
      Kokkos::RangePolicy<DeviceType,AlgoTagType> policy(0, _a.dimension_0());
      Kokkos::parallel_for(policy, *this);
    }      
  };

  template<typename DeviceType,
           typename ViewType, 
           typename ScalarType,
           int TestID>
  void impl_test_batched_matutil(const int N, const int BlkSize) {

    /// typedefs
    typedef typename ViewType::value_type value_type;
    typedef ScalarType scalar_type;
    typedef Kokkos::Details::ArithTraits<value_type> ats;

    /// radomized input testing views 
    const scalar_type alpha = 11.1;
    ViewType a("a", N, BlkSize, BlkSize);
    ViewType b("b", N, BlkSize, BlkSize);

    Kokkos::Random_XorShift64_Pool<typename DeviceType::execution_space> random(13718);
    Kokkos::fill_random(a, random, value_type(1.0));
    Kokkos::deep_copy(b, a);

    /// test body
    Functor<DeviceType,ViewType,ScalarType,NaiveTag,       TestID>(alpha, a).run();
    Functor<DeviceType,ViewType,ScalarType,KokkosKernelTag,TestID>(alpha, b).run();

    /// for comparison send it to host
    typename ViewType::HostMirror a_host = Kokkos::create_mirror_view(a);
    typename ViewType::HostMirror b_host = Kokkos::create_mirror_view(b);

    Kokkos::deep_copy(a_host, a);
    Kokkos::deep_copy(b_host, b);
      
    /// check a = b
    typename ats::mag_type eps = std::numeric_limits<typename ats::mag_type>::epsilon();
    for (int k=0;k<N;++k) 
      for (int i=0;i<BlkSize;++i) 
        for (int j=0;j<BlkSize;++j) 
          EXPECT_NEAR_KK( b_host(k,i,j), a_host(k,i,j), eps);
  }
}

template<typename DeviceType, 
         typename ValueType, 
         typename ScalarType,
         int TestID>
int test_batched_matutil() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) 
  {
    typedef Kokkos::View<ValueType***,Kokkos::LayoutLeft,DeviceType> ViewType;
    Test::impl_test_batched_matutil<DeviceType,ViewType,ScalarType,TestID>(     0, 10);
    Test::impl_test_batched_matutil<DeviceType,ViewType,ScalarType,TestID>(    10, 15);
    Test::impl_test_batched_matutil<DeviceType,ViewType,ScalarType,TestID>(  1024,  9);
    Test::impl_test_batched_matutil<DeviceType,ViewType,ScalarType,TestID>(132231,  3);
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) 
  {
    typedef Kokkos::View<ValueType***,Kokkos::LayoutRight,DeviceType> ViewType;
    Test::impl_test_batched_matutil<DeviceType,ViewType,ScalarType,TestID>(     0, 10);
    Test::impl_test_batched_matutil<DeviceType,ViewType,ScalarType,TestID>(    10, 15);
    Test::impl_test_batched_matutil<DeviceType,ViewType,ScalarType,TestID>(  1024,  9);
    Test::impl_test_batched_matutil<DeviceType,ViewType,ScalarType,TestID>(132231,  3);
  }
#endif
  
  return 0;
}
  
#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F( TestCategory, batched_set_float_float_set ) {
  test_batched_matutil<TestExecSpace,float,float,0>();
}
TEST_F( TestCategory, batched_set_float_float_scale ) {
  test_batched_matutil<TestExecSpace,float,float,1>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F( TestCategory, batched_set_double_double_set ) {
  test_batched_matutil<TestExecSpace,double,double,0>();
}
TEST_F( TestCategory, batched_set_double_double_scale ) {
  test_batched_matutil<TestExecSpace,double,double,1>();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
TEST_F( TestCategory, batched_set_dcomplex_dcomplex_set ) {
  test_batched_matutil<TestExecSpace,Kokkos::complex<double>,Kokkos::complex<double>,0>();
}
TEST_F( TestCategory, batched_set_dcomplex_dcomplex_scale ) {
  test_batched_matutil<TestExecSpace,Kokkos::complex<double>,Kokkos::complex<double>,1>();
}
TEST_F( TestCategory, batched_set_dcomplex_double_set ) {
  test_batched_matutil<TestExecSpace,Kokkos::complex<double>,double,0>();
}
TEST_F( TestCategory, batched_set_dcomplex_double_scale ) {
  test_batched_matutil<TestExecSpace,Kokkos::complex<double>,double,1>();
}
#endif
