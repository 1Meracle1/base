#ifndef X86_AVX_H
#define X86_AVX_H

#include "instruction_sets.h"
#include "../types.h"
#include <immintrin.h>

namespace simd
{
template <> struct RegisterTrait<f64, AVXInstructionSet>
{
    using element_type             = f64;
    using register_type            = __m256d;
    static constexpr u64 length    = 4;
    static constexpr u64 alignment = 32;

    // set/load/store
    static register_type set1(f32 v) { return _mm256_set1_pd(v); }
    static register_type setzero() { return _mm256_setzero_pd(); }
    static register_type load_unaligned(const element_type* p) { return _mm256_loadu_pd(p); }
    static void          store_unaligned(element_type* p, register_type v) { _mm256_storeu_pd(p, v); }
    static register_type load_aligned(const element_type* p) { return _mm256_load_pd(p); }
    static void          store_aligned(element_type* p, register_type v) { _mm256_store_pd(p, v); }

    // arithmetic
    static register_type add(register_type a, register_type b) { return _mm256_add_pd(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm256_sub_pd(a, b); }
    static register_type mul(register_type a, register_type b) { return _mm256_mul_pd(a, b); }
    static register_type div(register_type a, register_type b) { return _mm256_div_pd(a, b); }

    static register_type sqrt(register_type a) { return _mm256_sqrt_pd(a); }
    static register_type rsqrt(register_type a)
    {
        return _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(a)); // emulated as rsqrt_pd not available on AVX1
    }
    static register_type rcp(register_type a)
    {
        return _mm256_div_pd(_mm256_set1_pd(1.0), a); // emulated as rcp_pd not available on AVX1
    }
    static register_type abs(register_type a)
    {
        // clears the sign bit (MSB of each float)
        // 0x7FFFFFFF = 2147483647
        __m256i sign_mask = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL);
        return _mm256_and_pd(a, _mm256_castsi256_pd(sign_mask));
    }
    static register_type floor(register_type a) { return _mm256_floor_pd(a); }
    static register_type ceil(register_type a) { return _mm256_ceil_pd(a); }
    static register_type round_nearest(register_type a)
    {
        return _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
    static register_type truncate(register_type a)
    {
        return _mm256_round_pd(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC); // round towards zero
    }

    static register_type min(register_type a, register_type b) { return _mm256_min_pd(a, b); }
    static register_type max(register_type a, register_type b) { return _mm256_max_pd(a, b); }

    // bitwise
    static register_type bitwise_and(register_type a, register_type b) { return _mm256_and_pd(a, b); }
    static register_type bitwise_or(register_type a, register_type b) { return _mm256_or_pd(a, b); }
    static register_type bitwise_xor(register_type a, register_type b) { return _mm256_xor_pd(a, b); }
    static register_type bitwise_andnot(register_type val_to_keep, register_type val_to_invert_mask)
    {
        // computes: val_to_keep & (~val_to_invert_mask)
        return _mm256_andnot_pd(val_to_invert_mask, val_to_keep);
    }

    // comparison (result is a mask in a register_type)
    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
    static register_type cmpneq(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_NEQ_UQ); }
    static register_type cmplt(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_LT_OS); }
    static register_type cmple(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_LE_OS); }
    static register_type cmpgt(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_GT_OS); }
    static register_type cmpge(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_GE_OS); }
    static register_type cmpord(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_ORD_Q); }
    static register_type cmpunord(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_UNORD_Q); }

    // blend / select
    static register_type blendv(register_type a, register_type b, register_type mask)
    {
        // mask selects from b if MSB of mask element is 1, else from a
        return _mm256_blendv_pd(a, b, mask);
    }

    // mask extraction
    static int movemask(register_type v) { return _mm256_movemask_pd(v); }

    // horizontal ops
    static register_type hadd(register_type a, register_type b) { return _mm256_hadd_pd(a, b); }
    static register_type hsub(register_type a, register_type b) { return _mm256_hsub_pd(a, b); }
};

template <> struct RegisterTrait<f32, AVXInstructionSet>
{
    using element_type             = f32;
    using register_type            = __m256;
    static constexpr u64 length    = 8;
    static constexpr u64 alignment = 32;

    // set/load/store
    static register_type set1(f32 v) { return _mm256_set1_ps(v); }
    static register_type setzero() { return _mm256_setzero_ps(); }
    static register_type load_unaligned(const element_type* p) { return _mm256_loadu_ps(p); }
    static void          store_unaligned(element_type* p, register_type v) { _mm256_storeu_ps(p, v); }
    static register_type load_aligned(const element_type* p) { return _mm256_load_ps(p); }
    static void          store_aligned(element_type* p, register_type v) { _mm256_store_ps(p, v); }

    // arithmetic
    static register_type add(register_type a, register_type b) { return _mm256_add_ps(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm256_sub_ps(a, b); }
    static register_type mul(register_type a, register_type b) { return _mm256_mul_ps(a, b); }
    static register_type div(register_type a, register_type b) { return _mm256_div_ps(a, b); }

    static register_type sqrt(register_type a) { return _mm256_sqrt_ps(a); }
    static register_type rsqrt(register_type a)
    {
        return _mm256_rsqrt_ps(a); // approximation
    }
    static register_type rcp(register_type a)
    {
        return _mm256_rcp_ps(a); // approximation
    }
    static register_type abs(register_type a)
    {
        // clears the sign bit (MSB of each float)
        // 0x7FFFFFFF = 2147483647
        __m256i sign_mask = _mm256_set1_epi32(0x7FFFFFFF);
        return _mm256_and_ps(a, _mm256_castsi256_ps(sign_mask));
    }
    static register_type floor(register_type a) { return _mm256_floor_ps(a); }
    static register_type ceil(register_type a) { return _mm256_ceil_ps(a); }
    static register_type round_nearest(register_type a)
    {
        return _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
    static register_type truncate(register_type a)
    {
        return _mm256_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC); // round towards zero
    }

    static register_type min(register_type a, register_type b) { return _mm256_min_ps(a, b); }
    static register_type max(register_type a, register_type b) { return _mm256_max_ps(a, b); }

    // bitwise
    static register_type bitwise_and(register_type a, register_type b) { return _mm256_and_ps(a, b); }
    static register_type bitwise_or(register_type a, register_type b) { return _mm256_or_ps(a, b); }
    static register_type bitwise_xor(register_type a, register_type b) { return _mm256_xor_ps(a, b); }
    static register_type bitwise_andnot(register_type val_to_keep, register_type val_to_invert_mask)
    {
        // computes: val_to_keep & (~val_to_invert_mask)
        return _mm256_andnot_ps(val_to_invert_mask, val_to_keep);
    }

    // comparison (result is a mask in a register_type)
    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
    static register_type cmpneq(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_NEQ_UQ); }
    static register_type cmplt(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_LT_OS); }
    static register_type cmple(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_LE_OS); }
    static register_type cmpgt(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_GT_OS); }
    static register_type cmpge(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_GE_OS); }
    static register_type cmpord(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_ORD_Q); }
    static register_type cmpunord(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_UNORD_Q); }

    // blend / select
    static register_type blendv(register_type a, register_type b, register_type mask)
    {
        // mask selects from b if MSB of mask element is 1, else from a
        return _mm256_blendv_ps(a, b, mask);
    }

    // mask extraction
    static int movemask(register_type v) { return _mm256_movemask_ps(v); }

    // horizontal ops
    static register_type hadd(register_type a, register_type b) { return _mm256_hadd_ps(a, b); }
    static register_type hsub(register_type a, register_type b) { return _mm256_hsub_ps(a, b); }
};
} // namespace simd

#endif