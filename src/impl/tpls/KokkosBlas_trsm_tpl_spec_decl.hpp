#ifndef KOKKOSBLAS_TRSM_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS_TRSM_TPL_SPEC_DECL_HPP_

#if defined( KOKKOSKERNELS_ENABLE_TPL_BLAS )
#include "KokkosBlas_Host_tpl.hpp"
#include <stdio.h>

namespace KokkosBlas {
    namespace Impl {

    #define KOKKOSBLAS_DTRSM_BLAS(LAYOUTA, LAYOUTB, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct TRSM< \
        Kokkos::View<const double**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<double**, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef double SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType; \
        \
        static void trsm(const char side, const char uplo, \
                         const char transa, const char diag, \
                         typename AViewType::const_value_type& alpha,\
                         AViewType& A, BViewType& B){ \
            Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_BLAS, double]");\
            int M = A.extent(0); \
            int N = A.extent(1); \
            bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
            bool B_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTB>::value; \
            const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            const int BST = B_is_lr?B.stride(0):B.stride(1), LDB = BST == 0 ? 1:BST; \
            HostBlas<double>::trsm(A_is_lr, side, uplo, transa, diag, M, N, alpha, \
                                   A.data(), LDA, B.data(), LDB); \
            Kokkos::Profiling::popRegion(); \
        } \
    };


    #define KOKKOSBLAS_STRSM_BLAS(LAYOUTA, LAYOUTB, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct TRSM< \
        Kokkos::View<const float**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<float**, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef float SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType; \
        \
        static void trsm(const char side, const char uplo, \
                         const char transa, const char diag, \
                         typename AViewType::const_value_type& alpha,\
                         AViewType& A, BViewType& B){ \
            Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_BLAS, float]");\
            int M = A.extent(0); \
            int N = A.extent(1); \
            bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
            bool B_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTB>::value; \
            const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            const int BST = B_is_lr?B.stride(0):B.stride(1), LDB = BST == 0 ? 1:BST; \
            HostBlas<double>::trsm(A_is_lr, side, uplo, transa, diag, M, N, alpha, \
                                   A.data(), LDA, B.data(), LDB); \
            Kokkos::Profiling::popRegion(); \
        } \
    };
    

    #define KOKKOSBLAS_ZTRSM_BLAS(LAYOUTA, LAYOUTB, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct TRSM< \
        Kokkos::View<const Kokkos::complex<double>**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<Kokkos::complex<double>**, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef Kokkos::complex<double> SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType; \
        \
        static void trsm(const char side, const char uplo, \
                         const char transa, const char diag, \
                         typename AViewType::const_value_type& alpha,\
                         AViewType& A, BViewType& B){ \
            Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_BLAS, complex<double>]");\
            int M = A.extent(0); \
            int N = A.extent(1); \
            bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
            bool B_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTB>::value; \
            const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            const int BST = B_is_lr?B.stride(0):B.stride(1), LDB = BST == 0 ? 1:BST; \
            HostBlas<std::complex<double>>::trsm(A_is_lr, side, uplo, transa, diag, M, N, alpha, \
                                   reinterpret_cast<const std::complex<double>*>( A.data() ), LDA,  \
                                   reinterpret_cast<const std::complex<double>*>( B.data() ), LDB); \
            Kokkos::Profiling::popRegion(); \
        } \
    };


    #define KOKKOSBLAS_CTRSM_BLAS(LAYOUTA, LAYOUTB, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct TRSM< \
        Kokkos::View<const Kokkos::complex<float>**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<Kokkos::complex<float>**, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef Kokkos::complex<float> SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<SCALAR**, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > BViewType; \
        \
        static void trsm(const char side, const char uplo, \
                         const char transa, const char diag, \
                         typename AViewType::const_value_type& alpha,\
                         AViewType& A, BViewType& B){ \
            Kokkos::Profiling::pushRegion("KokkosBlas::trsm[TPL_BLAS, complex<double>]");\
            int M = A.extent(0); \
            int N = A.extent(1); \
            bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
            bool B_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTB>::value; \
            const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            const int BST = B_is_lr?B.stride(0):B.stride(1), LDB = BST == 0 ? 1:BST; \
            HostBlas<std::complex<float>>::trsm(A_is_lr, side, uplo, transa, diag, M, N, alpha, \
                                   reinterpret_cast<const std::complex<float>*>( A.data() ), LDA,  \
                                   reinterpret_cast<const std::complex<float>*>( B.data() ), LDB); \
            Kokkos::Profiling::popRegion(); \
        } \
    };


    KOKKOSBLAS_DTRSM_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
    KOKKOSBLAS_DTRSM_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
    KOKKOSBLAS_DTRSM_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_DTRSM_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

    KOKKOSBLAS_STRSM_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
    KOKKOSBLAS_STRSM_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
    KOKKOSBLAS_STRSM_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_STRSM_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSBLAS_ZTRSM_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
    KOKKOSBLAS_ZTRSM_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
    KOKKOSBLAS_ZTRSM_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_ZTRSM_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSBLAS_CTRSM_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
    KOKKOSBLAS_CTRSM_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
    KOKKOSBLAS_CTRSM_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_CTRSM_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

    } // namespace Impl
} //namespace KokkosBlas

#endif //ENABLE BLAS/LAPACK


//Magma

/*
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
#include<KokkosMagma_tpl_spec.hpp>

namespace KokkosBlas{
    namespace Impl{

    #define KOKKOSBLAS_DTRSM_MAGMA(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL)


    } //namespace Impl
} //namespace KokkosBlas

*/

#endif //KOKKOSBLAS_TRSM_TPL_SPEC_DECL_HPP_

