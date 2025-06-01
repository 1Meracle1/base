#ifndef X86_SSE3_H
#define X86_SSE3_H

#include "x86_sse2.h"

namespace simd
{
// --- SSE3: f64 ---
template <> struct RegisterTrait<f64, SSE3InstructionSet> : RegisterTrait<f64, SSE2InstructionSet> // Inherit common ops
{
    // SSE3 specific or overrides
    static register_type hadd(register_type a, register_type b) { return _mm_hadd_pd(a, b); }
    static register_type hsub(register_type a, register_type b) { return _mm_hsub_pd(a, b); }
    static register_type addsub(register_type a, register_type b)
    {
        return _mm_addsub_pd(a, b); // a0-b0, a1+b1
    }
};

// --- SSE3: f32 ---
template <> struct RegisterTrait<f32, SSE3InstructionSet> : RegisterTrait<f32, SSE2InstructionSet> // Inherit common ops
{
    // SSE3 specific or overrides
    static register_type hadd(register_type a, register_type b) { return _mm_hadd_ps(a, b); }
    static register_type hsub(register_type a, register_type b) { return _mm_hsub_ps(a, b); }
    static register_type addsub(register_type a, register_type b)
    {
        return _mm_addsub_ps(a, b); // a0-b0, a1+b1, a2-b2, a3+b3
    }
};

// For integer types, SSE3 primarily adds _mm_lddqu_si128.
// Other integer operations are largely the same as SSE2 unless SSSE3 is considered.
// Since we are doing full specializations, we'll copy and potentially adjust load_unaligned.

// Helper for SSE3 Integer Base (override unaligned load)
struct SSE3IntegerBase : SSEIntegerBase
{
    static __m128i load_unaligned(const void* p) { return _mm_lddqu_si128(reinterpret_cast<const __m128i*>(p)); }
};

// --- SSE3: u8 ---
template <> struct RegisterTrait<u8, SSE3InstructionSet> : RegisterTrait<u8, SSE2InstructionSet>
{
    static register_type load_unaligned(const element_type* p)
    {
        return _mm_lddqu_si128(reinterpret_cast<const __m128i*>(p));
    }
};
// --- SSE3: i8 ---
template <> struct RegisterTrait<i8, SSE3InstructionSet> : RegisterTrait<i8, SSE2InstructionSet>
{
    static register_type load_unaligned(const element_type* p)
    {
        return _mm_lddqu_si128(reinterpret_cast<const __m128i*>(p));
    }
};
// --- SSE3: u16 ---
template <> struct RegisterTrait<u16, SSE3InstructionSet> : RegisterTrait<u16, SSE2InstructionSet>
{
    static register_type load_unaligned(const element_type* p)
    {
        return _mm_lddqu_si128(reinterpret_cast<const __m128i*>(p));
    }
};
// --- SSE3: i16 ---
template <> struct RegisterTrait<i16, SSE3InstructionSet> : RegisterTrait<i16, SSE2InstructionSet>
{
    static register_type load_unaligned(const element_type* p)
    {
        return _mm_lddqu_si128(reinterpret_cast<const __m128i*>(p));
    }
};
// --- SSE3: u32 ---
template <> struct RegisterTrait<u32, SSE3InstructionSet> : RegisterTrait<u32, SSE2InstructionSet>
{
    static register_type load_unaligned(const element_type* p)
    {
        return _mm_lddqu_si128(reinterpret_cast<const __m128i*>(p));
    }
};
// --- SSE3: i32 ---
template <> struct RegisterTrait<i32, SSE3InstructionSet> : RegisterTrait<i32, SSE2InstructionSet>
{
    static register_type load_unaligned(const element_type* p)
    {
        return _mm_lddqu_si128(reinterpret_cast<const __m128i*>(p));
    }
};
// --- SSE3: u64 ---
template <> struct RegisterTrait<u64, SSE3InstructionSet> : RegisterTrait<u64, SSE2InstructionSet>
{
    static register_type load_unaligned(const element_type* p)
    {
        return _mm_lddqu_si128(reinterpret_cast<const __m128i*>(p));
    }
};
// --- SSE3: i64 ---
template <> struct RegisterTrait<i64, SSE3InstructionSet> : RegisterTrait<i64, SSE2InstructionSet>
{
    static register_type load_unaligned(const element_type* p)
    {
        return _mm_lddqu_si128(reinterpret_cast<const __m128i*>(p));
    }
};
} // namespace simd

#endif