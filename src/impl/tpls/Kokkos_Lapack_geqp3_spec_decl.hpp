#ifndef KOKKOS_LAPACK_GEQP3_TPL_SPEC_DECL_HPP_
#define KOKKOS_LAPACK_GEQP3_TPL_SPEC_DECL_HPP_

#ifdef KOKKOSKERNELES_ENABLE_TPL_BLAS
#include "KokkosBlas_host_tpl.hpp"
#ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACK
#include "KokkosLapack_host_tpl.hpp"

namespace KokkosLapack {
    namespace Impl {

    #define KOKKOSLAPACK_DGEQP3_LAPACK(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQP3< \
        Kokkos::View<double**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<int*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<double*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef double SCALAR; \
        typedef int ORDINAl; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<ORDINAL*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > PViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        \
        static void geqp3(AViewType& A, PViewType& p, TauViewType& tau){ \
        Kokkos::Profiling::pushRegion("KokkosLapack::geqp3[TPL_BLAS, double]");\
        const int M = A.extent(0); \
        const int N = A.extent(1); \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            if(!A_is_lr){ \
                HostLapack<double>::geqp3(   \
                        Kokkos::LayoutLeft, \
                        M, N,                \
                        A.data(), LDA        \
                        p.data(),            \
                        tau.data(),          \
                        ); \
            } \
            else{ \
                HostLapack<double>::geqp3(   \
                        Kokkos::LayoutRight, \
                        M, N,                \
                        A.data(), LDA        \
                        p.data(),            \
                        tau.data(),          \
                        ); \
            } \
            Kokkos::Profiling::popRegion(); \
        } \
    };



    #define KOKKOSLAPACK_SGEQP3_LAPACK(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQP3< \
        Kokkos::View<float**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<int*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<float*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef float SCALAR; \
        typedef int ORDINAl; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<ORDINAL*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > PViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        \
        static void geqp3(AViewType& A, PViewType& p, TauViewType& tau){ \
        Kokkos::Profiling::pushRegion("KokkosLapack::geqp3[TPL_BLAS, double]");\
        const int M = A.extent(0); \
        const int N = A.extent(1); \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            if(!A_is_lr){ \
                HostLapack<float>::geqp3(   \
                        Kokkos::LayoutLeft, \
                        M, N,                \
                        A.data(), LDA        \
                        p.data(),            \
                        tau.data(),          \
                        ); \
            } \
            else{ \
                HostLapack<float>::geqp3(   \
                        Kokkos::LayoutRight, \
                        M, N,                \
                        A.data(), LDA        \
                        p.data(),            \
                        tau.data(),          \
                        ); \
            } \
            Kokkos::Profiling::popRegion(); \
        } \
    };


    #define KOKKOSLAPACK_ZGEQP3_LAPACK(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQP3< \
        Kokkos::View<Kokkos::complex<double>**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<int*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<Kokkos::complex<double>*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef Kokkos::complex<double> SCALAR; \
        typedef int ORDINAl; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<ORDINAL*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > PViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        \
        static void geqp3(AViewType& A, PViewType& p, TauViewType& tau){ \
        Kokkos::Profiling::pushRegion("KokkosLapack::geqp3[TPL_BLAS, Complex<double>]");\
        const int M = A.extent(0); \
        const int N = A.extent(1); \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            if(!A_is_lr){ \
                HostLapack<std::complex<double>>::geqp3(   \
                        Kokkos::LayoutLeft, \
                        M, N,                \
                        reinterpret_cast<std::complex<double>*>(A.data()), LDA        \
                        p.data(),            \
                        reinterpret_cast<std::complex<double>*>(tau.data()),          \
                        ); \
            } \
            else{ \
                HostLapack<std::complex<double>>::geqp3(   \
                        Kokkos::LayoutRight, \
                        M, N,                \
                        reinterpret_cast<std::complex<double>*>(A.data()), LDA        \
                        p.data(),            \
                        reinterpret_cast<std::complex<double>*>(tau.data()),          \
                        ); \
            } \
            Kokkos::Profiling::popRegion(); \
        } \
    };


    #define KOKKOSLAPACK_CGEQP3_LAPACK(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQP3< \
        Kokkos::View<Kokkos::complex<float>**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<int*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<Kokkos::complex<float>*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef Kokkos::complex<float> SCALAR; \
        typedef int ORDINAl; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<ORDINAL*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > PViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        \
        static void geqp3(AViewType& A, PViewType& p, TauViewType& tau){ \
        Kokkos::Profiling::pushRegion("KokkosLapack::geqp3[TPL_BLAS, Complex<float>]");\
        const int M = A.extent(0); \
        const int N = A.extent(1); \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            if(!A_is_lr){ \
                HostLapack<std::complex<float>>::geqp3(   \
                        Kokkos::LayoutLeft, \
                        M, N,                \
                        reinterpret_cast<std::complex<float>*>(A.data()), LDA        \
                        p.data(),            \
                        reinterpret_cast<std::complex<float>*>(tau.data()),          \
                        ); \
            } \
            else{ \
                HostLapack<std::complex<float>>::geqp3(   \
                        Kokkos::LayoutRight, \
                        M, N,                \
                        reinterpret_cast<std::complex<float>*>(A.data()), LDA        \
                        p.data(),            \
                        reinterpret_cast<std::complex<float>*>(tau.data()),          \
                        ); \
            } \
            Kokkos::Profiling::popRegion(); \
        } \
    };

    KOKKOSLAPACK_DGEQP3_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSLAPACK_DGEQP3_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSLAPACK_DGEQP3_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSLAPACK_DGEQP3_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSLAPACK_SGEQP3_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSLAPACK_SGEQP3_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSLAPACK_SGEQP3_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSLAPACK_SGEQP3_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSLAPACK_ZGEQP3_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSLAPACK_ZGEQP3_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSLAPACK_ZGEQP3_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSLAPACK_ZGEQP3_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSLAPACK_CGEQP3_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSLAPACK_CGEQP3_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSLAPACK_CGEQP3_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSLAPACK_CGEQP3_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

    } // namespace Impl
} //namespace KokkosLapack
#endif

//cuBLAS
/*
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include<KokkosBlas_tpl_spec.hpp>

namespace KokkosLapack{
    namespace Impl{

    #define KOKKOSLAPACK_DGEQP3_CUBLAS(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) 


    } //namespace Impl
} //namespace KokkosLapack
*/

//Magma

/*
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
#include<KokkosMagma_tpl_spec.hpp>

namespace KokkosLapack{
    namespace Impl{

    #define KOKKOSLAPACK_DGEQP3_MAGMA(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL)


    } //namespace Impl
} //namespace KokkosLapack

*/
