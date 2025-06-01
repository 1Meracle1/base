#ifndef X86_SSE41
#define X86_SSE41

#include "x86_sse3.h"

namespace simd
{
// --- SSE4.1: f64 ---
template <>
struct RegisterTrait<f64, SSE41InstructionSet>
    : RegisterTrait<f64, SSE3InstructionSet> // Inherit common ops
{
    // SSE4.1 specific or overrides
    static register_type floor(register_type a) { return _mm_floor_pd(a); }
    static register_type ceil(register_type a) { return _mm_ceil_pd(a); }
    static register_type round_nearest(register_type a) {
        return _mm_round_pd(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
    static register_type truncate(register_type a) {
        return _mm_round_pd(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
    static register_type blendv(register_type a, register_type b,
                                register_type mask) {
        return _mm_blendv_pd(a, b, mask);
    }
};

// --- SSE4.1: f32 ---
template <>
struct RegisterTrait<f32, SSE41InstructionSet>
    : RegisterTrait<f32, SSE3InstructionSet> // Inherit common ops
{
    // SSE4.1 specific or overrides
    static register_type floor(register_type a) { return _mm_floor_ps(a); }
    static register_type ceil(register_type a) { return _mm_ceil_ps(a); }
    static register_type round_nearest(register_type a) {
        return _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
    static register_type truncate(register_type a) {
        return _mm_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
    static register_type blendv(register_type a, register_type b,
                                register_type mask) {
        return _mm_blendv_ps(a, b, mask);
    }
};


// --- SSE4.1: Integer Base (for blendv_epi8) ---
struct SSE41IntegerBase : SSE3IntegerBase {
     static __m128i blendv(__m128i a, __m128i b, __m128i mask) { // val_false, val_true, mask
        return _mm_blendv_epi8(a, b, mask);
    }
};


// --- SSE4.1: u8 ---
template <>
struct RegisterTrait<u8, SSE41InstructionSet> : RegisterTrait<u8, SSE3InstructionSet> {
    // SSE4.1 min/max_epu8 is same as SSE2. blendv_epi8 is new.
    static register_type blendv(register_type val_false, register_type val_true, register_type mask) {
        return _mm_blendv_epi8(val_false, val_true, mask);
    }
};

// --- SSE4.1: i8 ---
template <>
struct RegisterTrait<i8, SSE41InstructionSet> : RegisterTrait<i8, SSE3InstructionSet> {
    static register_type min(register_type a, register_type b) {
        return _mm_min_epi8(a, b);
    }
    static register_type max(register_type a, register_type b) {
        return _mm_max_epi8(a, b);
    }
    static register_type blendv(register_type val_false, register_type val_true, register_type mask) {
        return _mm_blendv_epi8(val_false, val_true, mask);
    }
};

// --- SSE4.1: u16 ---
template <>
struct RegisterTrait<u16, SSE41InstructionSet> : RegisterTrait<u16, SSE3InstructionSet> {
    static register_type min(register_type a, register_type b) {
        return _mm_min_epu16(a, b);
    }
    static register_type max(register_type a, register_type b) {
        return _mm_max_epu16(a, b);
    }
    static register_type blendv(register_type val_false, register_type val_true, register_type mask) {
        return _mm_blendv_epi8(val_false, val_true, mask); // Use epi8 version, mask needs to be per-byte
    }
};

// --- SSE4.1: i16 ---
template <>
struct RegisterTrait<i16, SSE41InstructionSet> : RegisterTrait<i16, SSE3InstructionSet> {
    // SSE2 already has min/max_epi16.
    static register_type blendv(register_type val_false, register_type val_true, register_type mask) {
        return _mm_blendv_epi8(val_false, val_true, mask);
    }
};

// --- SSE4.1: u32 ---
template <>
struct RegisterTrait<u32, SSE41InstructionSet> : RegisterTrait<u32, SSE3InstructionSet> {
    static register_type mullo(register_type a, register_type b) {
        return _mm_mullo_epi32(a, b);
    }
    static register_type min(register_type a, register_type b) {
        return _mm_min_epu32(a, b);
    }
    static register_type max(register_type a, register_type b) {
        return _mm_max_epu32(a, b);
    }
    static register_type blendv(register_type val_false, register_type val_true, register_type mask) {
        return _mm_blendv_epi8(val_false, val_true, mask);
    }
};

// --- SSE4.1: i32 ---
template <>
struct RegisterTrait<i32, SSE41InstructionSet> : RegisterTrait<i32, SSE3InstructionSet> {
    static register_type mullo(register_type a, register_type b) {
        return _mm_mullo_epi32(a, b);
    }
    static register_type mul_widening_to_i64(register_type a, register_type b) { // i32*i32 -> i64
        return _mm_mul_epi32(a,b); // result has 2 i64s
    }
    static register_type min(register_type a, register_type b) {
        return _mm_min_epi32(a, b);
    }
    static register_type max(register_type a, register_type b) {
        return _mm_max_epi32(a, b);
    }
    static register_type blendv(register_type val_false, register_type val_true, register_type mask) {
        return _mm_blendv_epi8(val_false, val_true, mask);
    }
};

// --- SSE4.1: u64 ---
template <>
struct RegisterTrait<u64, SSE41InstructionSet> : RegisterTrait<u64, SSE3InstructionSet> {
    static register_type cmpeq(register_type a, register_type b) {
        return _mm_cmpeq_epi64(a, b);
    }
    // cmpgt_epu64 still tricky. min/max_epu64 not available.
    static register_type blendv(register_type val_false, register_type val_true, register_type mask) {
        return _mm_blendv_epi8(val_false, val_true, mask);
    }
};

// --- SSE4.1: i64 ---
template <>
struct RegisterTrait<i64, SSE41InstructionSet> : RegisterTrait<i64, SSE3InstructionSet> {
    static register_type cmpeq(register_type a, register_type b) {
        return _mm_cmpeq_epi64(a, b);
    }
    // cmpgt_epi64 is SSE4.2. min/max_epi64 not available.
    static register_type blendv(register_type val_false, register_type val_true, register_type mask) {
        return _mm_blendv_epi8(val_false, val_true, mask);
    }
};
} // namespace simd

#endif