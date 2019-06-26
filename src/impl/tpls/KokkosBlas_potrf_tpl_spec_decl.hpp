#ifndef KOKKOSBLAS_POTRF_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS_POTRF_TPL_SPEC_DECL_HPP_

#if defined( KOKKOSKERNELS_ENABLE_TPL_BLAS )
#include "KokkosBlas_Host_tpl.hpp"
#include <stdio.h>

namespace KokkosBlas {
    namespace Impl {

    #define KOKKOSBLAS_DPOTRF_BLAS(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct POTRF< \
        Kokkos::View<double**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef double SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        \
        static void potrf(const char uplo, AViewType& A){ \
            Kokkos::Profiling::pushRegion("KokkosBlas::potrf[TPL_BLAS, double]");\
            int M = A.extent(0); \
            int N = A.extent(1); \
            bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
            const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            HostBlas<double>::potrf(A_is_lr, uplo, N, \
                                    A.data(), LDA); \
            Kokkos::Profiling::popRegion(); \
        } \
    };


    #define KOKKOSBLAS_SPOTRF_BLAS(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct POTRF< \
        Kokkos::View<float**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef float SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        \
        static void potrf(const char uplo, AViewType& A){ \
            Kokkos::Profiling::pushRegion("KokkosBlas::potrf[TPL_BLAS, float]");\
            int M = A.extent(0); \
            int N = A.extent(1); \
            bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
            const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            HostBlas<float>::potrf(A_is_lr, uplo, N, \
                                    A.data(), LDA); \
            Kokkos::Profiling::popRegion(); \
        } \
    };


    #define KOKKOSBLAS_CPOTRF_BLAS(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct POTRF< \
        Kokkos::View<Kokkos::complex<float>**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef Kokkos::complex<float> SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        \
        static void potrf(const char uplo, AViewType& A){ \
            Kokkos::Profiling::pushRegion("KokkosBlas::potrf[TPL_BLAS, Kokkos::complex<float>]");\
            int N = A.extent(1); \
            bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
            const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            HostBlas<std::complex<float>>::potrf(A_is_lr, uplo, N, \
                                reinterpret_cast<std::complex<float>>( A.data() ), LDA); \
            Kokkos::Profiling::popRegion(); \
        } \
    };


    #define KOKKOSBLAS_CPOTRF_BLAS(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct POTRF< \
        Kokkos::View<Kokkos::complex<double>**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef Kokkos::complex<double> SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        \
        static void potrf(const char uplo, AViewType& A){ \
            Kokkos::Profiling::pushRegion("KokkosBlas::potrf[TPL_BLAS, Kokkos::complex<double>]");\
            int N = A.extent(1); \
            bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
            const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            HostBlas<std::complex<double>>::potrf(A_is_lr, uplo, N, \
                                reinterpret_cast<std::complex<double>>( A.data() ), LDA); \
            Kokkos::Profiling::popRegion(); \
        } \
    };

    KOKKOSBLAS_DPOTRF_BLAS(Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
    KOKKOSBLAS_DPOTRF_BLAS(Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
    KOKKOSBLAS_DPOTRF_BLAS(Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_DPOTRF_BLAS(Kokkos::LayoutRight, Kokkos::HostSpace, false)

    KOKKOSBLAS_SPOTRF_BLAS(Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
    KOKKOSBLAS_SPOTRF_BLAS(Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
    KOKKOSBLAS_SPOTRF_BLAS(Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_SPOTRF_BLAS(Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSBLAS_ZPOTRF_BLAS(Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
    KOKKOSBLAS_ZPOTRF_BLAS(Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
    KOKKOSBLAS_ZPOTRF_BLAS(Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_ZPOTRF_BLAS(Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSBLAS_CPOTRF_BLAS(Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
    KOKKOSBLAS_CPOTRF_BLAS(Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
    KOKKOSBLAS_CPOTRF_BLAS(Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_CPOTRF_BLAS(Kokkos::LayoutRight, Kokkos::HostSpace, false)

    } // namespace Impl
} //namespace KokkosBlas

#endif //ENABLE BLAS/LAPACK


//Magma

#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
#include<KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas{
    namespace Impl{

    #define KOKKOSBLAS_DPOTRF_MAGMA(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct POTRF< \
        Kokkos::View<double**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef double SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        \
        static void potrf(const char uplo, AViewType& A){ \
            Kokkos::Profiling::pushRegion("KokkosBlas::potrf[TPL_MAGMA, double]");\
            KokkosBlas::Impl::MagmaSingleton & s = KokkosBlas::Impl::MagmaSingleton::singleton(); \
            magma_int_t N = A.extent(1); \
            bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
            const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            magma_uplo_t m_uplo  = (uplo   == 'U' || uplo   == 'u' ) ? MagmaUpper : MagmaLower; \
            magma_int_t info = 0; \
            magma_dpotrf_gpu(m_uplo, N,\
                    reinterpret_cast<magmaDouble_ptr>( A.data() ), LDA,  &info);\
            Kokkos::Profiling::popRegion(); \
        } \
    };

    #define KOKKOSBLAS_SPOTRF_MAGMA(LAYOUTA, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct POTRF< \
        Kokkos::View<float**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef float SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        \
        static void potrf(const char uplo, AViewType& A){ \
            Kokkos::Profiling::pushRegion("KokkosBlas::potrf[TPL_MAGMA, float]");\
            KokkosBlas::Impl::MagmaSingleton & s = KokkosBlas::Impl::MagmaSingleton::singleton(); \
            magma_int_t N = A.extent(1); \
            bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
            const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            magma_uplo_t m_uplo  = (uplo   == 'U' || uplo   == 'u' ) ? MagmaUpper : MagmaLower; \
            magma_int_t info = 0; \
            magma_spotrf_gpu(m_uplo, N,\
                    reinterpret_cast<magmaFloat_ptr>( A.data() ), LDA,  &info);\
            Kokkos::Profiling::popRegion(); \
        } \
    };

    KOKKOSBLAS_DPOTRF_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, true) 
    KOKKOSBLAS_DPOTRF_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

    KOKKOSBLAS_SPOTRF_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, true) 
    KOKKOSBLAS_SPOTRF_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

    } //namespace Impl
} //namespace KokkosBlas

#endif //IF MAGMA

#endif //KOKKOSBLAS_POTRF_TPL_SPEC_DECL_HPP_

