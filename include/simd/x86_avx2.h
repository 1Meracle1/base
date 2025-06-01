#ifndef X86_AVX2_H
#define X86_AVX2_H

#include "instruction_sets.h"
#include "../types.h"
#include <immintrin.h>

namespace simd
{

template <> struct RegisterTrait<f64, AVX2InstructionSet>
{
    using element_type                = f64;
    using register_type               = __m256d;
    static constexpr size_t length    = 4;
    static constexpr size_t alignment = 32;

    // Load / Store / Set
    static register_type set1(element_type v) { return _mm256_set1_pd(v); }
    static register_type setzero() { return _mm256_setzero_pd(); }
    static register_type set(element_type v3, element_type v2, element_type v1, element_type v0)
    {
        return _mm256_set_pd(v3, v2, v1, v0);
    }
    static register_type load_unaligned(const element_type* p) { return _mm256_loadu_pd(p); }
    static void          store_unaligned(element_type* p, register_type v) { _mm256_storeu_pd(p, v); }
    static register_type load_aligned(const element_type* p) { return _mm256_load_pd(p); }
    static void          store_aligned(element_type* p, register_type v) { _mm256_store_pd(p, v); }

    // Arithmetic
    static register_type add(register_type a, register_type b) { return _mm256_add_pd(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm256_sub_pd(a, b); }
    static register_type mul(register_type a, register_type b) { return _mm256_mul_pd(a, b); }
    static register_type div(register_type a, register_type b) { return _mm256_div_pd(a, b); }

    // Mathematical Functions
    static register_type sqrt(register_type a) { return _mm256_sqrt_pd(a); }
    static register_type rsqrt(register_type a)
    {
        return _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(a));
    }
    static register_type rcp(register_type a)
    {
        return _mm256_div_pd(_mm256_set1_pd(1.0), a);
    }
    static register_type abs(register_type a)
    {
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
        return _mm256_round_pd(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }

    // Min / Max
    static register_type min(register_type a, register_type b) { return _mm256_min_pd(a, b); }
    static register_type max(register_type a, register_type b) { return _mm256_max_pd(a, b); }

    // Bitwise
    static register_type bitwise_and(register_type a, register_type b) { return _mm256_and_pd(a, b); }
    static register_type bitwise_or(register_type a, register_type b) { return _mm256_or_pd(a, b); }
    static register_type bitwise_xor(register_type a, register_type b) { return _mm256_xor_pd(a, b); }
    static register_type bitwise_andnot(register_type val_to_keep, register_type val_to_invert_mask)
    {
        return _mm256_andnot_pd(val_to_invert_mask, val_to_keep);
    }

    // Comparison
    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
    static register_type cmpneq(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_NEQ_UQ); }
    static register_type cmplt(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_LT_OS); }
    static register_type cmple(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_LE_OS); }
    static register_type cmpgt(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_GT_OS); }
    static register_type cmpge(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_GE_OS); }

    // Blend / Select
    static register_type blendv(register_type a, register_type b, register_type mask)
    {
        return _mm256_blendv_pd(a, b, mask);
    }

    // Mask Extraction
    static int movemask(register_type v) { return _mm256_movemask_pd(v); }

    // Horizontal Operations
    static register_type hadd(register_type a, register_type b) { return _mm256_hadd_pd(a, b); }
    static register_type hsub(register_type a, register_type b) { return _mm256_hsub_pd(a, b); }

    // Fused Multiply-Add (FMA)
    static register_type fmadd(register_type a, register_type b, register_type c) { return _mm256_fmadd_pd(a, b, c); }
    static register_type fmsub(register_type a, register_type b, register_type c) { return _mm256_fmsub_pd(a, b, c); }
    static register_type fnmadd(register_type a, register_type b, register_type c) { return _mm256_fnmadd_pd(a, b, c); }
    static register_type fnmsub(register_type a, register_type b, register_type c) { return _mm256_fnmsub_pd(a, b, c); }
};

template <> struct RegisterTrait<f32, AVX2InstructionSet>
{
    using element_type                = f32;
    using register_type               = __m256;
    static constexpr size_t length    = 8;
    static constexpr size_t alignment = 32;

    // Load / Store / Set
    static register_type set1(element_type v) { return _mm256_set1_ps(v); }
    static register_type setzero() { return _mm256_setzero_ps(); }
    static register_type set(element_type v7,
                             element_type v6,
                             element_type v5,
                             element_type v4,
                             element_type v3,
                             element_type v2,
                             element_type v1,
                             element_type v0)
    {
        return _mm256_set_ps(v7, v6, v5, v4, v3, v2, v1, v0);
    }
    static register_type load_unaligned(const element_type* p) { return _mm256_loadu_ps(p); }
    static void          store_unaligned(element_type* p, register_type v) { _mm256_storeu_ps(p, v); }
    static register_type load_aligned(const element_type* p) { return _mm256_load_ps(p); }
    static void          store_aligned(element_type* p, register_type v) { _mm256_store_ps(p, v); }

    // Arithmetic
    static register_type add(register_type a, register_type b) { return _mm256_add_ps(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm256_sub_ps(a, b); }
    static register_type mul(register_type a, register_type b) { return _mm256_mul_ps(a, b); }
    static register_type div(register_type a, register_type b) { return _mm256_div_ps(a, b); }

    // Mathematical Functions
    static register_type sqrt(register_type a) { return _mm256_sqrt_ps(a); }
    static register_type rsqrt(register_type a) { return _mm256_rsqrt_ps(a); }
    static register_type rcp(register_type a) { return _mm256_rcp_ps(a); }
    static register_type abs(register_type a)
    {
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
        return _mm256_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }

    // Min / Max
    static register_type min(register_type a, register_type b) { return _mm256_min_ps(a, b); }
    static register_type max(register_type a, register_type b) { return _mm256_max_ps(a, b); }

    // Bitwise
    static register_type bitwise_and(register_type a, register_type b) { return _mm256_and_ps(a, b); }
    static register_type bitwise_or(register_type a, register_type b) { return _mm256_or_ps(a, b); }
    static register_type bitwise_xor(register_type a, register_type b) { return _mm256_xor_ps(a, b); }
    static register_type bitwise_andnot(register_type val_to_keep, register_type val_to_invert_mask)
    {
        return _mm256_andnot_ps(val_to_invert_mask, val_to_keep);
    }

    // Comparison
    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
    static register_type cmpneq(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_NEQ_UQ); }
    static register_type cmplt(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_LT_OS); }
    static register_type cmple(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_LE_OS); }
    static register_type cmpgt(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_GT_OS); }
    static register_type cmpge(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_GE_OS); }

    // Blend / Select
    static register_type blendv(register_type a, register_type b, register_type mask)
    {
        return _mm256_blendv_ps(a, b, mask);
    }

    // Mask Extraction
    static int movemask(register_type v) { return _mm256_movemask_ps(v); }

    // Horizontal Operations
    static register_type hadd(register_type a, register_type b) { return _mm256_hadd_ps(a, b); }
    static register_type hsub(register_type a, register_type b) { return _mm256_hsub_ps(a, b); }

    // Fused Multiply-Add (FMA)
    static register_type fmadd(register_type a, register_type b, register_type c) { return _mm256_fmadd_ps(a, b, c); }
    static register_type fmsub(register_type a, register_type b, register_type c) { return _mm256_fmsub_ps(a, b, c); }
    static register_type fnmadd(register_type a, register_type b, register_type c) { return _mm256_fnmadd_ps(a, b, c); }
    static register_type fnmsub(register_type a, register_type b, register_type c) { return _mm256_fnmsub_ps(a, b, c); }
    static register_type fmaddsub(register_type a, register_type b, register_type c)
    {
        return _mm256_fmaddsub_ps(a, b, c);
    }
    static register_type fmsubadd(register_type a, register_type b, register_type c)
    {
        return _mm256_fmsubadd_ps(a, b, c);
    }
};

// --- Integer AVX2 Traits ---

// Common bitwise operations for integer types
template <typename RegType> struct IntegerBitwiseOps
{
    static RegType bitwise_and(RegType a, RegType b) { return _mm256_and_si256(a, b); }
    static RegType bitwise_or(RegType a, RegType b) { return _mm256_or_si256(a, b); }
    static RegType bitwise_xor(RegType a, RegType b) { return _mm256_xor_si256(a, b); }
    static RegType bitwise_andnot(RegType val_to_keep, RegType val_to_invert_mask)
    {
        return _mm256_andnot_si256(val_to_invert_mask, val_to_keep);
    }
    // Generic blend using epi8, mask from comparison (all 0s or all 1s per element)
    static RegType blendv(RegType a, RegType b, RegType mask) { return _mm256_blendv_epi8(a, b, mask); }
};

template <> struct RegisterTrait<u8, AVX2InstructionSet> : IntegerBitwiseOps<__m256i>
{
    using element_type                = u8;
    using register_type               = __m256i;
    static constexpr size_t length    = 32;
    static constexpr size_t alignment = 32;

    static register_type set1(element_type v) { return _mm256_set1_epi8(static_cast<char>(v)); }
    static register_type setzero() { return _mm256_setzero_si256(); }
    // Note: _mm256_set_epi8 takes 32 char arguments in reverse order.
    // Example: static register_type set(char c31, ..., char c0) { return _mm256_set_epi8(c31, ..., c0); }
    static register_type load_unaligned(const element_type* p)
    {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store_unaligned(element_type* p, register_type v)
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    static register_type load_aligned(const element_type* p)
    {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store_aligned(element_type* p, register_type v)
    {
        _mm256_store_si256(reinterpret_cast<__m256i*>(p), v);
    }

    static register_type add(register_type a, register_type b) { return _mm256_add_epi8(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm256_sub_epi8(a, b); }
    // No direct mullo_epi8. Widening multiply:
    static __m256i mul_widening_to_u16(register_type a, register_type b)
    {
        __m256i a_lo   = _mm256_unpacklo_epi8(a, _mm256_setzero_si256()); // u16
        __m256i b_lo   = _mm256_unpacklo_epi8(b, _mm256_setzero_si256()); // u16
        __m256i a_hi   = _mm256_unpackhi_epi8(a, _mm256_setzero_si256()); // u16
        __m256i b_hi   = _mm256_unpackhi_epi8(b, _mm256_setzero_si256()); // u16
        __m256i res_lo = _mm256_mullo_epi16(a_lo, b_lo);
        __m256i res_hi = _mm256_mullo_epi16(a_hi, b_hi);
        // Combine res_lo and res_hi if needed, or return as two __m256i for 16-bit results
        // This example returns the lower half's multiplication (16 results)
        // For a full 32 results, it's more complex or requires two return values.
        // Let's assume this means "multiply corresponding elements, result is 16-bit wide"
        // This is not ideal. A better approach for mul(u8,u8)->u16 might be needed.
        // For now, this is a placeholder for a more specific widening multiply.
        // A common use is _mm256_maddubs_epi16 for u8*s8 -> s16 then horizontal add.
        // For simple u8*u8 -> u16 (lower 16 elements):
        return _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(a)), // lower 16 u8 -> u16
                                  _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b)));
        // And for upper 16 elements:
        // _mm256_mullo_epi16(
        //    _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a, 1)),
        //    _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b, 1))
        // );
        // This API expects one register_type. So mul for u8 is complex.
    }

    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmpeq_epi8(a, b); }
    static register_type cmpgt_unsigned(register_type a, register_type b)
    {
        // (a > b) unsigned is equivalent to (max(a,b) == a) && (a != b)
        // or using bias: (a^0x80 > b^0x80) signed
        const __m256i bias = _mm256_set1_epi8(static_cast<char>(0x80));
        return _mm256_cmpgt_epi8(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias));
    }
    static register_type min(register_type a, register_type b) { return _mm256_min_epu8(a, b); }
    static register_type max(register_type a, register_type b) { return _mm256_max_epu8(a, b); }
    static int           movemask(register_type v) { return _mm256_movemask_epi8(v); }
};

template <> struct RegisterTrait<i8, AVX2InstructionSet> : IntegerBitwiseOps<__m256i>
{
    using element_type                = i8;
    using register_type               = __m256i;
    static constexpr size_t length    = 32;
    static constexpr size_t alignment = 32;

    static register_type set1(element_type v) { return _mm256_set1_epi8(v); }
    static register_type setzero() { return _mm256_setzero_si256(); }
    static register_type load_unaligned(const element_type* p)
    {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store_unaligned(element_type* p, register_type v)
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    static register_type load_aligned(const element_type* p)
    {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store_aligned(element_type* p, register_type v)
    {
        _mm256_store_si256(reinterpret_cast<__m256i*>(p), v);
    }

    static register_type add(register_type a, register_type b) { return _mm256_add_epi8(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm256_sub_epi8(a, b); }
    // No direct mullo_epi8. Widening multiply:
    static __m256i mul_widening_to_i16(register_type a, register_type b)
    {
        // Similar to u8, this is complex for a single register_type return.
        // Placeholder for i8*i8 -> i16 (lower 16 elements)
        return _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(a)),
                                  _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b)));
    }

    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmpeq_epi8(a, b); }
    static register_type cmpgt_signed(register_type a, register_type b) { return _mm256_cmpgt_epi8(a, b); }
    static register_type min(register_type a, register_type b) { return _mm256_min_epi8(a, b); }
    static register_type max(register_type a, register_type b) { return _mm256_max_epi8(a, b); }
    static register_type abs(register_type a) { return _mm256_abs_epi8(a); }
    static int           movemask(register_type v) { return _mm256_movemask_epi8(v); }
};

template <> struct RegisterTrait<u16, AVX2InstructionSet> : IntegerBitwiseOps<__m256i>
{
    using element_type                = u16;
    using register_type               = __m256i;
    static constexpr size_t length    = 16;
    static constexpr size_t alignment = 32;

    static register_type set1(element_type v) { return _mm256_set1_epi16(static_cast<short>(v)); }
    static register_type setzero() { return _mm256_setzero_si256(); }
    static register_type load_unaligned(const element_type* p)
    {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store_unaligned(element_type* p, register_type v)
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    static register_type load_aligned(const element_type* p)
    {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store_aligned(element_type* p, register_type v)
    {
        _mm256_store_si256(reinterpret_cast<__m256i*>(p), v);
    }

    static register_type add(register_type a, register_type b) { return _mm256_add_epi16(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm256_sub_epi16(a, b); }
    static register_type mullo(register_type a, register_type b) { return _mm256_mullo_epi16(a, b); }
    static register_type mulhi(register_type a, register_type b) { return _mm256_mulhi_epu16(a, b); }

    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmpeq_epi16(a, b); }
    static register_type cmpgt_unsigned(register_type a, register_type b)
    {
        const __m256i bias = _mm256_set1_epi16(static_cast<short>(0x8000));
        return _mm256_cmpgt_epi16(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias));
    }
    static register_type min(register_type a, register_type b) { return _mm256_min_epu16(a, b); }
    static register_type max(register_type a, register_type b) { return _mm256_max_epu16(a, b); }

    static register_type slli(register_type v, int count) { return _mm256_slli_epi16(v, count); }
    static register_type srli(register_type v, int count) { return _mm256_srli_epi16(v, count); }
    static int           movemask(register_type v)
    {
        // v has 0x0000 or 0xFFFF in each 16-bit lane.
        // Extract MSB (sign bit effectively) of each 16-bit element.
        __m256i signs = _mm256_srai_epi16(v, 15); // 0x0000 or 0xFFFF
        // Pack 16-bit elements to 8-bit elements.
        // 0x0000 -> 0x00, 0xFFFF -> 0xFF (saturated)
        __m128i lo_16       = _mm256_castsi256_si128(signs);
        __m128i hi_16       = _mm256_extracti128_si256(signs, 1);
        __m128i packed_lo_8 = _mm_packs_epi16(lo_16, _mm_setzero_si128()); // Lower 8 results
        __m128i packed_hi_8 = _mm_packs_epi16(hi_16, _mm_setzero_si128()); // Upper 8 results
        // Combine and movemask
        __m256i packed_16_to_8 = _mm256_set_m128i(
            packed_hi_8, packed_lo_8); // This order might be reversed depending on desired final mask bit order
                                       // Intel typically has lane 0 as LSB.
                                       // _mm256_inserti128_si256(_mm256_castsi128_si256(packed_lo_8), packed_hi_8, 1);
        packed_16_to_8 =
            _mm256_permute4x64_epi64(packed_16_to_8, _MM_SHUFFLE(3, 1, 2, 0)); // Ensure correct order for final mask
                                                                               // This might be overly complex.
                                                                               // A simpler conceptual way:
        int mask_lo = _mm_movemask_epi8(packed_lo_8);                          // 8 LSBs
        int mask_hi = _mm_movemask_epi8(packed_hi_8);                          // 8 MSBs
        return mask_lo | (mask_hi << 8);
    }
};

template <> struct RegisterTrait<i16, AVX2InstructionSet> : IntegerBitwiseOps<__m256i>
{
    using element_type                = i16;
    using register_type               = __m256i;
    static constexpr size_t length    = 16;
    static constexpr size_t alignment = 32;

    static register_type set1(element_type v) { return _mm256_set1_epi16(v); }
    static register_type setzero() { return _mm256_setzero_si256(); }
    static register_type load_unaligned(const element_type* p)
    {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store_unaligned(element_type* p, register_type v)
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    static register_type load_aligned(const element_type* p)
    {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store_aligned(element_type* p, register_type v)
    {
        _mm256_store_si256(reinterpret_cast<__m256i*>(p), v);
    }

    static register_type add(register_type a, register_type b) { return _mm256_add_epi16(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm256_sub_epi16(a, b); }
    static register_type mullo(register_type a, register_type b) { return _mm256_mullo_epi16(a, b); }
    static register_type mulhi(register_type a, register_type b) { return _mm256_mulhi_epi16(a, b); }

    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmpeq_epi16(a, b); }
    static register_type cmpgt_signed(register_type a, register_type b) { return _mm256_cmpgt_epi16(a, b); }
    static register_type min(register_type a, register_type b) { return _mm256_min_epi16(a, b); }
    static register_type max(register_type a, register_type b) { return _mm256_max_epi16(a, b); }
    static register_type abs(register_type a) { return _mm256_abs_epi16(a); }

    static register_type slli(register_type v, int count) { return _mm256_slli_epi16(v, count); }
    static register_type srli(register_type v, int count) { return _mm256_srli_epi16(v, count); }
    static register_type srai(register_type v, int count) { return _mm256_srai_epi16(v, count); }
    static int           movemask(register_type v)
    {
        // Same logic as u16
        __m256i signs       = _mm256_srai_epi16(v, 15);
        __m128i lo_16       = _mm256_castsi256_si128(signs);
        __m128i hi_16       = _mm256_extracti128_si256(signs, 1);
        __m128i packed_lo_8 = _mm_packs_epi16(lo_16, _mm_setzero_si128());
        __m128i packed_hi_8 = _mm_packs_epi16(hi_16, _mm_setzero_si128());
        int     mask_lo     = _mm_movemask_epi8(packed_lo_8);
        int     mask_hi     = _mm_movemask_epi8(packed_hi_8);
        return mask_lo | (mask_hi << 8);
    }
};

template <> struct RegisterTrait<u32, AVX2InstructionSet> : IntegerBitwiseOps<__m256i>
{
    using element_type                = u32;
    using register_type               = __m256i;
    static constexpr size_t length    = 8;
    static constexpr size_t alignment = 32;

    static register_type set1(element_type v) { return _mm256_set1_epi32(static_cast<int>(v)); }
    static register_type setzero() { return _mm256_setzero_si256(); }
    static register_type load_unaligned(const element_type* p)
    {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store_unaligned(element_type* p, register_type v)
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    static register_type load_aligned(const element_type* p)
    {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store_aligned(element_type* p, register_type v)
    {
        _mm256_store_si256(reinterpret_cast<__m256i*>(p), v);
    }

    static register_type add(register_type a, register_type b) { return _mm256_add_epi32(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm256_sub_epi32(a, b); }
    static register_type mullo(register_type a, register_type b) { return _mm256_mullo_epi32(a, b); }
    // For u32 * u32 -> u64, use _mm256_mul_epu32 (results in 4 u64s)

    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmpeq_epi32(a, b); }
    static register_type cmpgt_unsigned(register_type a, register_type b)
    {
        const __m256i bias = _mm256_set1_epi32(0x80000000);
        return _mm256_cmpgt_epi32(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias));
    }
    static register_type min(register_type a, register_type b) { return _mm256_min_epu32(a, b); }
    static register_type max(register_type a, register_type b) { return _mm256_max_epu32(a, b); }

    static register_type slli(register_type v, int count) { return _mm256_slli_epi32(v, count); }
    static register_type srli(register_type v, int count) { return _mm256_srli_epi32(v, count); }
    static register_type sllv(register_type v, register_type counts) { return _mm256_sllv_epi32(v, counts); }
    static register_type srlv(register_type v, register_type counts) { return _mm256_srlv_epi32(v, counts); }
    static int           movemask(register_type v) { return _mm256_movemask_ps(_mm256_castsi256_ps(v)); }
};

template <> struct RegisterTrait<i32, AVX2InstructionSet> : IntegerBitwiseOps<__m256i>
{
    using element_type                = i32;
    using register_type               = __m256i;
    static constexpr size_t length    = 8;
    static constexpr size_t alignment = 32;

    static register_type set1(element_type v) { return _mm256_set1_epi32(v); }
    static register_type setzero() { return _mm256_setzero_si256(); }
    static register_type load_unaligned(const element_type* p)
    {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store_unaligned(element_type* p, register_type v)
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    static register_type load_aligned(const element_type* p)
    {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store_aligned(element_type* p, register_type v)
    {
        _mm256_store_si256(reinterpret_cast<__m256i*>(p), v);
    }

    static register_type add(register_type a, register_type b) { return _mm256_add_epi32(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm256_sub_epi32(a, b); }
    static register_type mullo(register_type a, register_type b) { return _mm256_mullo_epi32(a, b); }
    // For i32 * i32 -> i64, use _mm256_mul_epi32 (results in 4 i64s)

    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmpeq_epi32(a, b); }
    static register_type cmpgt_signed(register_type a, register_type b) { return _mm256_cmpgt_epi32(a, b); }
    static register_type min(register_type a, register_type b) { return _mm256_min_epi32(a, b); }
    static register_type max(register_type a, register_type b) { return _mm256_max_epi32(a, b); }
    static register_type abs(register_type a) { return _mm256_abs_epi32(a); }

    static register_type slli(register_type v, int count) { return _mm256_slli_epi32(v, count); }
    static register_type srli(register_type v, int count) { return _mm256_srli_epi32(v, count); }
    static register_type srai(register_type v, int count) { return _mm256_srai_epi32(v, count); }
    static register_type sllv(register_type v, register_type counts) { return _mm256_sllv_epi32(v, counts); }
    static register_type srlv(register_type v, register_type counts) { return _mm256_srlv_epi32(v, counts); }
    static register_type srav(register_type v, register_type counts) { return _mm256_srav_epi32(v, counts); }
    static int           movemask(register_type v) { return _mm256_movemask_ps(_mm256_castsi256_ps(v)); }
};

template <> struct RegisterTrait<u64, AVX2InstructionSet> : IntegerBitwiseOps<__m256i>
{
    using element_type                = u64;
    using register_type               = __m256i;
    static constexpr size_t length    = 4;
    static constexpr size_t alignment = 32;

    static register_type set1(element_type v) { return _mm256_set1_epi64x(static_cast<long long>(v)); }
    static register_type setzero() { return _mm256_setzero_si256(); }
    static register_type load_unaligned(const element_type* p)
    {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store_unaligned(element_type* p, register_type v)
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    static register_type load_aligned(const element_type* p)
    {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store_aligned(element_type* p, register_type v)
    {
        _mm256_store_si256(reinterpret_cast<__m256i*>(p), v);
    }

    static register_type add(register_type a, register_type b) { return _mm256_add_epi64(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm256_sub_epi64(a, b); }
    static register_type mullo(register_type a, register_type b)
    {
        // Emulated: (a_lo*b_lo) + ((a_lo*b_hi + a_hi*b_lo) << 32)
        // _mm256_mul_epu32 multiplies pairs of 32-bit integers into 64-bit results.
        // It operates on adjacent 32-bit elements.
        // a = [a3_hi a3_lo | a2_hi a2_lo | a1_hi a1_lo | a0_hi a0_lo]
        // To get a_lo and b_lo for each 64-bit element:
        __m256i a_lo  = _mm256_shuffle_epi32(a, _MM_SHUFFLE(3, 1, 3, 1)); // a_lo parts at even 32-bit positions
        __m256i b_lo  = _mm256_shuffle_epi32(b, _MM_SHUFFLE(3, 1, 3, 1));
        __m256i ab_lo = _mm256_mul_epu32(a_lo, b_lo); // (a0_lo*b0_lo), (a1_lo*b1_lo), ...

        __m256i a_hi = _mm256_shuffle_epi32(a, _MM_SHUFFLE(2, 0, 2, 0)); // a_hi parts
        __m256i b_hi = _mm256_shuffle_epi32(b, _MM_SHUFFLE(2, 0, 2, 0));

        __m256i alo_bhi = _mm256_mul_epu32(a_lo, b_hi);
        __m256i ahi_blo = _mm256_mul_epu32(a_hi, b_lo);

        __m256i mid_sum = _mm256_add_epi64(alo_bhi, ahi_blo);
        mid_sum         = _mm256_slli_epi64(mid_sum, 32);
        return _mm256_add_epi64(ab_lo, mid_sum);
    }

    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmpeq_epi64(a, b); }
    static register_type cmpgt_unsigned(register_type a, register_type b)
    {
        const __m256i bias = _mm256_set1_epi64x(0x8000000000000000LL);
        return _mm256_cmpgt_epi64(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias));
    }
    static register_type min(register_type a, register_type b)
    {
        // Emulated: (a < b) ? a : b
        register_type mask = cmpgt_unsigned(b, a); // mask is 1 where a < b
        return blendv(b, a, mask);                 // if mask[i] is 1, select a[i], else b[i]
    }
    static register_type max(register_type a, register_type b)
    {
        // Emulated: (a > b) ? a : b
        register_type mask = cmpgt_unsigned(a, b); // mask is 1 where a > b
        return blendv(b, a, mask);                 // if mask[i] is 1, select a[i], else b[i]
    }

    static register_type slli(register_type v, int count) { return _mm256_slli_epi64(v, count); }
    static register_type srli(register_type v, int count) { return _mm256_srli_epi64(v, count); }
    static register_type sllv(register_type v, register_type counts) { return _mm256_sllv_epi64(v, counts); }
    static register_type srlv(register_type v, register_type counts) { return _mm256_srlv_epi64(v, counts); }
    static int           movemask(register_type v) { return _mm256_movemask_pd(_mm256_castsi256_pd(v)); }
};

template <> struct RegisterTrait<i64, AVX2InstructionSet> : IntegerBitwiseOps<__m256i>
{
    using element_type                = i64;
    using register_type               = __m256i;
    static constexpr size_t length    = 4;
    static constexpr size_t alignment = 32;

    static register_type set1(element_type v) { return _mm256_set1_epi64x(v); }
    static register_type setzero() { return _mm256_setzero_si256(); }
    static register_type load_unaligned(const element_type* p)
    {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store_unaligned(element_type* p, register_type v)
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    static register_type load_aligned(const element_type* p)
    {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store_aligned(element_type* p, register_type v)
    {
        _mm256_store_si256(reinterpret_cast<__m256i*>(p), v);
    }

    static register_type add(register_type a, register_type b) { return _mm256_add_epi64(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm256_sub_epi64(a, b); }
    static register_type mullo(register_type a, register_type b)
    {
        // Same emulation as u64, as low bits are the same for signed/unsigned mul.
        __m256i a_lo  = _mm256_shuffle_epi32(a, _MM_SHUFFLE(3, 1, 3, 1));
        __m256i b_lo  = _mm256_shuffle_epi32(b, _MM_SHUFFLE(3, 1, 3, 1));
        __m256i ab_lo = _mm256_mul_epu32(a_lo, b_lo); // Use epu32 for 32x32->64

        __m256i a_hi = _mm256_shuffle_epi32(a, _MM_SHUFFLE(2, 0, 2, 0));
        __m256i b_hi = _mm256_shuffle_epi32(b, _MM_SHUFFLE(2, 0, 2, 0));

        __m256i alo_bhi = _mm256_mul_epu32(a_lo, b_hi);
        __m256i ahi_blo = _mm256_mul_epu32(a_hi, b_lo);

        __m256i mid_sum = _mm256_add_epi64(alo_bhi, ahi_blo);
        mid_sum         = _mm256_slli_epi64(mid_sum, 32);
        return _mm256_add_epi64(ab_lo, mid_sum);
    }

    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmpeq_epi64(a, b); }
    static register_type cmpgt_signed(register_type a, register_type b) { return _mm256_cmpgt_epi64(a, b); }
    static register_type min(register_type a, register_type b)
    {
        register_type mask = cmpgt_signed(b, a); // mask is 1 where a < b
        return blendv(b, a, mask);
    }
    static register_type max(register_type a, register_type b)
    {
        register_type mask = cmpgt_signed(a, b); // mask is 1 where a > b
        return blendv(b, a, mask);
    }
    static register_type abs(register_type val)
    {
        // Emulated: (val < 0) ? -val : val
        // -val = (0-val) or (~val + 1)
        // A common way: mask = val >> 63 (arithmetic); (val ^ mask) - mask
        __m256i mask    = _mm256_cmpgt_epi64(_mm256_setzero_si256(), val); // All 1s if val < 0
        __m256i xor_val = _mm256_xor_si256(val, mask);                     // if val < 0, ~val. else val.
        return _mm256_sub_epi64(xor_val, mask); // if val < 0, ~val - (-1) = ~val + 1. else val - 0.
    }

    static register_type slli(register_type v, int count) { return _mm256_slli_epi64(v, count); }
    static register_type srli(register_type v, int count) { return _mm256_srli_epi64(v, count); }
    // srai_epi64 is not directly available in AVX2. Emulation is complex.
    // For simplicity, it's omitted here. If needed, a robust emulation involves
    // checking sign and ORing with shifted sign bits.
    // A simpler (but possibly less optimal or specific) srai can be done if count is small
    // or by using srav_epi64 if available (AVX512VL).
    // For now, omitting srai_epi64 and srav_epi64.
    static register_type sllv(register_type v, register_type counts) { return _mm256_sllv_epi64(v, counts); }
    static register_type srlv(register_type v, register_type counts) { return _mm256_srlv_epi64(v, counts); }
    static int           movemask(register_type v) { return _mm256_movemask_pd(_mm256_castsi256_pd(v)); }
};
} // namespace simd

#endif