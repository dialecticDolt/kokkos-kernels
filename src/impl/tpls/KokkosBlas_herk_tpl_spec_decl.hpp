#ifndef KOKKOSBLAS_HERK_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS_HERK_TPL_SPEC_DECL_HPP_

#if defined( KOKKOSKERNELS_ENABLE_TPL_BLAS )
#include "KokkosBlas_Host_tpl.hpp"
#include <stdio.h>

namespace KokkosBlas {
    namespace Impl {

    #define KOKKOSBLAS_DHERK_BLAS(LAYOUTA, LAYOUTB, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct HERK< \
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
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
        \
        static void herk(const char uplo, const char trans, \
                         typename AViewType::const_value_type& alpha,\
                         AViewType& A,\
                         typename CViewType::const_value_type& beta, \
                         CViewType& C){ \
            Kokkos::Profiling::pushRegion("KokkosBlas::herk[TPL_BLAS, double]");\
            int N, k; \
            if (trans == 'T' || trans == 't'){ \
                N = A.extent(1); \
                k = A.extent(0); \
            } \
            else  {\
                N = A.extent(0); \
                k = A.extent(1); \
            } \
            bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
            bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTB>::value; \
            const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
            HostBlas<double>::syrk(A_is_lr, uplo, trans, N, k, alpha, \
                                   A.data(), LDA, beta, \
                                   C.data(), LDC); \
            Kokkos::Profiling::popRegion(); \
        } \
    };


    #define KOKKOSBLAS_SHERK_BLAS(LAYOUTA, LAYOUTB, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct HERK< \
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
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
        \
        static void herk(const char uplo, const char trans, \
                         typename AViewType::const_value_type& alpha,\
                         AViewType& A,\
                         typename CViewType::const_value_type& beta, \
                         CViewType& C){ \
            Kokkos::Profiling::pushRegion("KokkosBlas::herk[TPL_BLAS, float]");\
            int N, k; \
            if (trans == 'T' || trans == 't'){ \
                N = A.extent(1); \
                k = A.extent(0); \
            } \
            else  {\
                N = A.extent(0); \
                k = A.extent(1); \
            } \
            bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
            bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTB>::value; \
            const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
            HostBlas<float>::syrk(A_is_lr, uplo, trans, N, k, alpha, \
                                   A.data(), LDA, beta, \
                                   C.data(), LDC); \
            Kokkos::Profiling::popRegion(); \
        } \
    };


    #define KOKKOSBLAS_CHERK_BLAS(LAYOUTA, LAYOUTB, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct HERK< \
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
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
        \
        static void herk(const char uplo, const char trans, \
                         typename AViewType::const_value_type& alpha,\
                         AViewType& A,\
                         typename BViewType::const_value_type& beta, \
                         CViewType& C){ \
            Kokkos::Profiling::pushRegion("KokkosBlas::herk[TPL_BLAS, Kokkos:complex<float>]");\
            int N, k; \
            if (trans == 'T' || trans == 't'){ \
                N = A.extent(1); \
                k = A.extent(0); \
            } \
            else  {\
                N = A.extent(0); \
                k = A.extent(1); \
            } \
            bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
            bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTB>::value; \
            const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
            HostBlas<std::complex<float>>::herk(A_is_lr, uplo, trans, N, k, alpha, \
                            reinterpret_cast<const std::complex<float>>( A.data() ), LDA, beta, \
                            reinterpret_cast<std::complex<float>>( C.data() ), LDC); \
            Kokkos::Profiling::popRegion(); \
        } \
    };


    #define KOKKOSBLAS_CHERK_BLAS(LAYOUTA, LAYOUTB, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct HERK< \
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
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
        \
        static void herk(const char uplo, const char trans, \
                         typename AViewType::const_value_type& alpha,\
                         AViewType& A,\
                         typename BViewType::const_value_type& beta, \
                         CViewType& C){ \
            Kokkos::Profiling::pushRegion("KokkosBlas::herk[TPL_BLAS, Kokkos:complex<double>]");\
            int N, k; \
            if (trans == 'T' || trans == 't'){ \
                N = A.extent(1); \
                k = A.extent(0); \
            } \
            else  {\
                N = A.extent(0); \
                k = A.extent(1); \
            } \
            bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
            bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTB>::value; \
            const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
            HostBlas<std::complex<double>>::herk(A_is_lr, uplo, trans, N, k, alpha, \
                            reinterpret_cast<const std::complex<double>>( A.data() ), LDA, beta, \
                            reinterpret_cast<std::complex<double>>( C.data() ), LDC); \
            Kokkos::Profiling::popRegion(); \
        } \
    };

    KOKKOSBLAS_DHERK_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
    KOKKOSBLAS_DHERK_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
    KOKKOSBLAS_DHERK_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_DHERK_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

    KOKKOSBLAS_SHERK_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
    KOKKOSBLAS_SHERK_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
    KOKKOSBLAS_SHERK_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_SHERK_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSBLAS_ZHERK_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
    KOKKOSBLAS_ZHERK_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
    KOKKOSBLAS_ZHERK_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_ZHERK_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSBLAS_CHERK_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true)
    KOKKOSBLAS_CHERK_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
    KOKKOSBLAS_CHERK_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_CHERK_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

    } // namespace Impl
} //namespace KokkosBlas

#endif //ENABLE BLAS/LAPACK


//Magma

#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
#include<KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas{
    namespace Impl{

    #define KOKKOSBLAS_DHERK_MAGMA(LAYOUTA, LAYOUTB, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct HERK< \
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
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
        \
        static void herk(const char uplo, const char trans, \
                         typename AViewType::const_value_type& alpha,\
                         AViewType& A, \
                         typename CViewType::const_value_type& beta, \
                         CViewType& C){ \
            Kokkos::Profiling::pushRegion("KokkosBlas::herk[TPL_MAGMA, double]");\
            KokkosBlas::Impl::MagmaSingleton & s = KokkosBlas::Impl::MagmaSingleton::singleton(); \
            int N, k; \
            if (trans == 'T' || trans == 't'){ \
                N = A.extent(1); \
                k = A.extent(0); \
            } \
            else { \
                N = A.extent(0); \
                k = A.extent(1); \
            } \
            bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
            bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTB>::value; \
            const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
            int device = 0; \
            magma_queue_t queue; \
            magma_queue_create(device, &queue); \
            magma_uplo_t m_uplo  = (uplo   == 'U' || uplo   == 'u' ) ? MagmaUpper : MagmaLower; \
            magma_trans_t m_trans = (trans == 'T' || trans == 't' ) ? MagmaTrans : MagmaNoTrans; \
            magma_dsyrk(m_uplo, m_trans, N, k, alpha, \
                    reinterpret_cast<magmaDouble_const_ptr>( A.data() ), LDA, beta,  \
                    reinterpret_cast<magmaDouble_ptr>( C.data() ), LDC, queue); \
            magma_queue_destroy(queue);\
            Kokkos::Profiling::popRegion(); \
        } \
    };

    #define KOKKOSBLAS_SHERK_MAGMA(LAYOUTA, LAYOUTB, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct HERK< \
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
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
        \
        static void herk(const char uplo, const char trans, \
                         typename AViewType::const_value_type& alpha,\
                         AViewType& A, \
                         typename CViewType::const_value_type& beta, \
                         CViewType& C){ \
            Kokkos::Profiling::pushRegion("KokkosBlas::herk[TPL_MAGMA, float]");\
            KokkosBlas::Impl::MagmaSingleton & s = KokkosBlas::Impl::MagmaSingleton::singleton(); \
            int N, k; \
            if (trans == 'T' || trans == 't'){ \
                N = A.extent(1); \
                k = A.extent(0); \
            } \
            else  {\
                N = A.extent(0); \
                k = A.extent(1); \
            } \
            bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
            bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTB>::value; \
            const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
            const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
            int device = 0; \
            magma_queue_t queue; \
            magma_queue_create(device, &queue); \
            magma_uplo_t m_uplo  = (uplo   == 'U' || uplo   == 'u' ) ? MagmaUpper : MagmaLower; \
            magma_trans_t m_trans = (trans == 'T' || trans == 't' ) ? MagmaTrans : MagmaNoTrans; \
            magma_dsyrk(m_uplo, m_trans, N, k, alpha, \
                    reinterpret_cast<magmaFloat_const_ptr>( A.data() ), LDA, beta,  \
                    reinterpret_cast<magmaFloat_ptr>( C.data() ), LDC, queue); \
            magma_queue_destroy(queue);\
            Kokkos::Profiling::popRegion(); \
        } \
    };


    KOKKOSBLAS_DHERK_MAGMA(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, true) 
    KOKKOSBLAS_DHERK_MAGMA(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

    KOKKOSBLAS_SHERK_MAGMA(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, true) 
    KOKKOSBLAS_SHERK_MAGMA(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

    } //namespace Impl
} //namespace KokkosBlas

#endif //IF MAGMA

#endif //KOKKOSBLAS_HERK_TPL_SPEC_DECL_HPP_

