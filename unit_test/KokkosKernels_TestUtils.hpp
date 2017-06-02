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


namespace Test {
  template<class ViewType, bool strided = std::is_same<typename ViewType::array_layout, Kokkos::LayoutStride>::value>
  struct multivector_layout_adapter;

  template<class ViewType>
  struct multivector_layout_adapter<ViewType,true> {
    typedef typename ViewType::value_type Scalar;
    typedef typename ViewType::device_type Device;
    typedef Kokkos::View<Scalar**[2], Kokkos::LayoutRight, Device> BaseTypeRight;
    typedef Kokkos::View<Scalar**, typename ViewType::array_layout, Device> BaseTypeDefault;
    typedef typename std::conditional<
                std::is_same<typename ViewType::array_layout,Kokkos::LayoutStride>::value,
                BaseTypeRight, BaseTypeDefault>::type BaseType;

    static ViewType view(const BaseType& v) {
      return Kokkos::subview(v,Kokkos::ALL,Kokkos::ALL,0);
    };
  };

  template<class ViewType>
  struct multivector_layout_adapter<ViewType,false> {
    typedef typename ViewType::value_type Scalar;
    typedef typename ViewType::device_type Device;
    typedef Kokkos::View<Scalar**[2], Kokkos::LayoutRight, Device> BaseTypeRight;
    typedef Kokkos::View<Scalar**, typename ViewType::array_layout, Device> BaseTypeDefault;
    typedef typename std::conditional<
                std::is_same<typename ViewType::array_layout,Kokkos::LayoutStride>::value,
                BaseTypeRight, BaseTypeDefault>::type BaseType;

    static ViewType view(const BaseType& v) {
      return Kokkos::subview(v,Kokkos::ALL,Kokkos::ALL);
    };
  };

  template<class Scalar1, class Scalar2, class Scalar3>
  void EXPECT_NEAR_KK(Scalar1 val1, Scalar2 val2, Scalar3 tol) {
    typedef Kokkos::Details::ArithTraits<Scalar1> AT1;
    typedef Kokkos::Details::ArithTraits<Scalar2> AT2;
    typedef Kokkos::Details::ArithTraits<Scalar3> AT3;
    EXPECT_NEAR(double(AT1::abs(val1)),double(AT2::abs(val2)),double(AT3::abs(tol)));
  }
}

