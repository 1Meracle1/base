#ifndef CPUID_H
#define CPUID_H

#include "../types.h"
#include <ostream>

namespace simd
{
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#else
#warning "cpuid intrinsics are not available."
#endif

struct SupportedFeatures
{
    bool sse2  = false;
    bool sse3  = false;
    bool sse41 = false;
    bool avx   = false;
    bool fma   = false;
    bool avx2  = false;

    inline SupportedFeatures() noexcept
    {
        sse2 = false;
        avx  = false;
        avx2 = false;

        u32 eax = 0, ebx = 0, ecx = 0, edx = 0;

        // Get max supported leaf
        get_cpuid(0, 0, &eax, &ebx, &ecx, &edx);
        u32 max_leaf = eax;

        if (max_leaf >= 1)
        {
            get_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
            // SSE2: EDX bit 26
            sse2 = (edx & (1 << 26)) != 0; // SSE3: CPUID.1:ECX[bit 0]
            sse3 = (ecx & (1 << 0)) != 0;
            // SSE4.1: CPUID.1:ECX[bit 19]
            sse41 = (ecx & (1 << 19)) != 0;

            // AVX requires OSXSAVE (ECX bit 27) and AVX support (ECX bit 28)
            // and OS support for YMM state (XCR0 bit 2)
            bool osxsave          = (ecx & (1 << 27)) != 0;
            bool cpu_supports_avx = (ecx & (1 << 28)) != 0;

            if (osxsave && cpu_supports_avx)
            {
                u64 xcr0 = xgetbv(0); // Read XCR0
                // Check if bits 1 (SSE state) and 2 (YMM state) are set
                avx = (xcr0 & 0x6) == 0x6;
                // FMA support
                fma = avx && (ecx & (1 << 12)) != 0;
            }
        }

        if (max_leaf >= 7)
        {
            // AVX2 requires AVX support and AVX2 support (Leaf 7, Subleaf 0, EBX bit 5)
            get_cpuid(7, 0, &eax, &ebx, &ecx, &edx);
            bool cpu_supports_avx2 = (ebx & (1 << 5)) != 0;

            if (avx && cpu_supports_avx2)
            {
                avx2 = true;
            }
        }
        if (avx2)
            avx = true;
    }

  private:
    void get_cpuid(u32 leaf, u32 subleaf, u32* eax, u32* ebx, u32* ecx, u32* edx)
    {
#if defined(_MSC_VER)
        int regs[4];
        __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
        *eax = regs[0];
        *ebx = regs[1];
        *ecx = regs[2];
        *edx = regs[3];
#elif defined(__GNUC__) || defined(__clang__)
        __cpuid_count(static_cast<unsigned int>(leaf), static_cast<unsigned int>(subleaf), *eax, *ebx, *ecx, *edx);
#else
        // Fallback or error for unsupported compilers
        *eax = *ebx = *ecx = *edx = 0;
#endif
    }

    // OS support for AVX state saving
    u64 xgetbv(u32 xcr_idx)
    {
#if defined(_MSC_VER)
        return _xgetbv(xcr_idx);
#elif defined(__GNUC__) || defined(__clang__)
        u32 eax_val, edx_val;
        __asm__ volatile("xgetbv" : "=a"(eax_val), "=d"(edx_val) : "c"(xcr_idx));
        return (static_cast<u64>(edx_val) << 32) | eax_val;
#else
        return 0; // Not supported or unknown
#endif
    }
};

inline SupportedFeatures supported_features()
{
    static SupportedFeatures res{};
    return res;
}

inline std::ostream& operator<<(std::ostream& os, const SupportedFeatures& supported_features)
{
    os << "Supported SIMD features:\n\tsse2: " << std::boolalpha << supported_features.sse2
       << "\n\tsse3: " << supported_features.sse3 << "\n\tsse4.1: " << supported_features.sse41
       << "\n\tavx: " << supported_features.avx << "\n\tfma: " << supported_features.fma
       << "\n\tavx2: " << supported_features.avx2;
    return os;
}
} // namespace simd

#endif