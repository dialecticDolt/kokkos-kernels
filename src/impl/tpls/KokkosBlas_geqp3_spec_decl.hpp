#ifndef KOKKOS_LAPACK_GEQP3_TPL_SPEC_DECL_HPP_
#define KOKKOS_LAPACK_GEQP3_TPL_SPEC_DECL_HPP_

#ifdef KOKKOSKERNELES_ENABLE_TPL_BLAS
#include "KokkosBlas_host_tpl.hpp"
#ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACK
#include "KokkosLapack_host_tpl.hpp"

namespace KokkosBlas {
    namespace Impl {

    #define KOKKOSBLAS_DGEQP3_BLAS(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
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
                        A_is_lr, \
                        M, N,                \
                        A.data(), LDA        \
                        p.data(),            \
                        tau.data(),          \
                        ); \
            } \
            else{ \
                HostLapack<double>::geqp3(   \
                        A_is_lr, \
                        M, N,                \
                        A.data(), LDA        \
                        p.data(),            \
                        tau.data(),          \
                        ); \
            } \
            Kokkos::Profiling::popRegion(); \
        } \
    };



    #define KOKKOSBLAS_SGEQP3_BLAS(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
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
                        A_is_lr, \
                        M, N,                \
                        A.data(), LDA        \
                        p.data(),            \
                        tau.data(),          \
                        ); \
            } \
            else{ \
                HostLapack<float>::geqp3(   \
                        A_is_lr, \
                        M, N,                \
                        A.data(), LDA        \
                        p.data(),            \
                        tau.data(),          \
                        ); \
            } \
            Kokkos::Profiling::popRegion(); \
        } \
    };


    #define KOKKOSBLAS_ZGEQP3_BLAS(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
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
                        A_is_lr, \
                        M, N,                \
                        reinterpret_cast<std::complex<double>*>(A.data()), LDA        \
                        p.data(),            \
                        reinterpret_cast<std::complex<double>*>(tau.data()),          \
                        ); \
            } \
            else{ \
                HostLapack<std::complex<double>>::geqp3(   \
                        A_is_lr, \
                        M, N,                \
                        reinterpret_cast<std::complex<double>*>(A.data()), LDA        \
                        p.data(),            \
                        reinterpret_cast<std::complex<double>*>(tau.data()),          \
                        ); \
            } \
            Kokkos::Profiling::popRegion(); \
        } \
    };


    #define KOKKOSBLAS_CGEQP3_BLAS(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
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
                        A_is_lr, \
                        M, N,                \
                        reinterpret_cast<std::complex<float>*>(A.data()), LDA        \
                        p.data(),            \
                        reinterpret_cast<std::complex<float>*>(tau.data()),          \
                        ); \
            } \
            else{ \
                HostLapack<std::complex<float>>::geqp3(   \
                        A_is_lr, \
                        M, N,                \
                        reinterpret_cast<std::complex<float>*>(A.data()), LDA        \
                        p.data(),            \
                        reinterpret_cast<std::complex<float>*>(tau.data()),          \
                        ); \
            } \
            Kokkos::Profiling::popRegion(); \
        } \
    };

    KOKKOSBLAS_DGEQP3_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSBLAS_DGEQP3_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSBLAS_DGEQP3_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_DGEQP3_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSBLAS_SGEQP3_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSBLAS_SGEQP3_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSBLAS_SGEQP3_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_SGEQP3_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSBLAS_ZGEQP3_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSBLAS_ZGEQP3_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSBLAS_ZGEQP3_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_ZGEQP3_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSBLAS_CGEQP3_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSBLAS_CGEQP3_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSBLAS_CGEQP3_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_CGEQP3_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

    } // namespace Impl
} //namespace KokkosBlas
#endif


//Magma

/*
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
#include<KokkosMagma_tpl_spec.hpp>

namespace KokkosBlas{
    namespace Impl{

    #define KOKKOSBLAS_DGEQP3_MAGMA(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL)


    } //namespace Impl
} //namespace KokkosBlas

*/
