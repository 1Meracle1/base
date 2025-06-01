#ifndef X86_SSE2_H
#define X86_SSE2_H

#include "instruction_sets.h"
#include "../types.h"
#include <array>
#include <immintrin.h>

namespace simd
{
namespace detail
{
template <typename RegType, typename ElementType, int Bits> struct SSEIntegerAbs
{
    static RegType abs(RegType val)
    {
        if constexpr (Bits == 8)
        {
            RegType mask =
                _mm_srai_epi16(_mm_slli_epi16(val, 8), 8); // sign extend each byte to 16 bits then shift back to get
                                                           // 0x00 or 0xFF per byte A bit more direct for epi8 if we had
                                                           // _mm_srai_epi8 Alternative for epi8:
            RegType neg_mask = _mm_cmplt_epi8(val, _mm_setzero_si128()); // 0xFF if negative, 0x00 otherwise
            RegType neg_val  = _mm_sub_epi8(_mm_setzero_si128(), val);
            return _mm_or_si128(_mm_and_si128(neg_mask, neg_val), _mm_andnot_si128(neg_mask, val)); // blend
        }
        else if constexpr (Bits == 16)
        {
            RegType mask = _mm_srai_epi16(val, 15);
            return _mm_sub_epi16(_mm_xor_si128(val, mask), mask);
        }
        else if constexpr (Bits == 32)
        {
            RegType mask = _mm_srai_epi32(val, 31);
            return _mm_sub_epi32(_mm_xor_si128(val, mask), mask);
        }
        else if constexpr (Bits == 64)
        { // Emulated for i64
            RegType mask =
                _mm_cmpgt_epi64(_mm_setzero_si128(), val); // SSE4.2 for _mm_cmpgt_epi64.
                                                           // For SSE4.1 and below, this is harder.
                                                           // A simpler way for abs_epi64 without cmpgt_epi64:
            RegType sign_bits =
                _mm_srai_epi32(_mm_shuffle_epi32(val, _MM_SHUFFLE(3, 3, 1, 1)), 31); // Get sign bits of high dwords
            sign_bits       = _mm_unpacklo_epi32(sign_bits, sign_bits);              // duplicate to make 64-bit masks
            RegType xor_val = _mm_xor_si128(val, sign_bits);
            return _mm_sub_epi64(xor_val, sign_bits);
        }
        return val; // Should not happen
    }
};
} // namespace detail

// --- SSE2: f64 ---
template <> struct RegisterTrait<f64, SSE2InstructionSet>
{
    using element_type                = f64;
    using register_type               = __m128d;
    static constexpr size_t length    = 2;
    static constexpr size_t alignment = 16;

    static register_type set1(element_type v) { return _mm_set1_pd(v); }
    static register_type setzero() { return _mm_setzero_pd(); }
    static register_type set(element_type v1, element_type v0) { return _mm_set_pd(v1, v0); }
    static register_type load_unaligned(const element_type* p) { return _mm_loadu_pd(p); }
    static void          store_unaligned(element_type* p, register_type v) { _mm_storeu_pd(p, v); }
    static register_type load_aligned(const element_type* p) { return _mm_load_pd(p); }
    static void          store_aligned(element_type* p, register_type v) { _mm_store_pd(p, v); }

    static register_type add(register_type a, register_type b) { return _mm_add_pd(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm_sub_pd(a, b); }
    static register_type mul(register_type a, register_type b) { return _mm_mul_pd(a, b); }
    static register_type div(register_type a, register_type b) { return _mm_div_pd(a, b); }
    static register_type sqrt(register_type a) { return _mm_sqrt_pd(a); }
    static register_type rsqrt(register_type a)
    { // Composed
        return _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(a));
    }
    static register_type rcp(register_type a)
    { // Composed
        return _mm_div_pd(_mm_set1_pd(1.0), a);
    }
    static register_type abs(register_type a)
    {
        // 0x7FFFFFFFFFFFFFFFLL
        return _mm_and_pd(a, _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
    }
    // floor, ceil, round_nearest, truncate require SSE4.1

    static register_type min(register_type a, register_type b) { return _mm_min_pd(a, b); }
    static register_type max(register_type a, register_type b) { return _mm_max_pd(a, b); }

    static register_type bitwise_and(register_type a, register_type b) { return _mm_and_pd(a, b); }
    static register_type bitwise_or(register_type a, register_type b) { return _mm_or_pd(a, b); }
    static register_type bitwise_xor(register_type a, register_type b) { return _mm_xor_pd(a, b); }
    static register_type bitwise_andnot(register_type val_to_keep, register_type val_to_invert_mask)
    {
        return _mm_andnot_pd(val_to_invert_mask, val_to_keep);
    }

    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_pd(a, b); }
    static register_type cmpneq(register_type a, register_type b) { return _mm_cmpneq_pd(a, b); }
    static register_type cmplt(register_type a, register_type b) { return _mm_cmplt_pd(a, b); }
    static register_type cmple(register_type a, register_type b) { return _mm_cmple_pd(a, b); }
    static register_type cmpgt(register_type a, register_type b) { return _mm_cmpgt_pd(a, b); }
    static register_type cmpge(register_type a, register_type b) { return _mm_cmpge_pd(a, b); }
    // blendv requires SSE4.1
    static int movemask(register_type v) { return _mm_movemask_pd(v); }
    // hadd/hsub require SSE3
};

// --- SSE2: f32 ---
template <> struct RegisterTrait<f32, SSE2InstructionSet>
{
    using element_type                = f32;
    using register_type               = __m128;
    static constexpr size_t length    = 4;
    static constexpr size_t alignment = 16;

    static register_type set1(element_type v) { return _mm_set1_ps(v); }
    static register_type setzero() { return _mm_setzero_ps(); }
    static register_type set(element_type v3, element_type v2, element_type v1, element_type v0)
    {
        return _mm_set_ps(v3, v2, v1, v0);
    }
    static register_type load_unaligned(const element_type* p) { return _mm_loadu_ps(p); }
    static void          store_unaligned(element_type* p, register_type v) { _mm_storeu_ps(p, v); }
    static register_type load_aligned(const element_type* p) { return _mm_load_ps(p); }
    static void          store_aligned(element_type* p, register_type v) { _mm_store_ps(p, v); }

    static register_type add(register_type a, register_type b) { return _mm_add_ps(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm_sub_ps(a, b); }
    static register_type mul(register_type a, register_type b) { return _mm_mul_ps(a, b); }
    static register_type div(register_type a, register_type b) { return _mm_div_ps(a, b); }
    static register_type sqrt(register_type a) { return _mm_sqrt_ps(a); }
    static register_type rsqrt(register_type a) { return _mm_rsqrt_ps(a); }
    static register_type rcp(register_type a) { return _mm_rcp_ps(a); }
    static register_type abs(register_type a) { return _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF))); }
    // floor, ceil, round_nearest, truncate require SSE4.1

    static register_type min(register_type a, register_type b) { return _mm_min_ps(a, b); }
    static register_type max(register_type a, register_type b) { return _mm_max_ps(a, b); }

    static register_type bitwise_and(register_type a, register_type b) { return _mm_and_ps(a, b); }
    static register_type bitwise_or(register_type a, register_type b) { return _mm_or_ps(a, b); }
    static register_type bitwise_xor(register_type a, register_type b) { return _mm_xor_ps(a, b); }
    static register_type bitwise_andnot(register_type val_to_keep, register_type val_to_invert_mask)
    {
        return _mm_andnot_ps(val_to_invert_mask, val_to_keep);
    }

    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_ps(a, b); }
    static register_type cmpneq(register_type a, register_type b) { return _mm_cmpneq_ps(a, b); }
    static register_type cmplt(register_type a, register_type b) { return _mm_cmplt_ps(a, b); }
    static register_type cmple(register_type a, register_type b) { return _mm_cmple_ps(a, b); }
    static register_type cmpgt(register_type a, register_type b) { return _mm_cmpgt_ps(a, b); }
    static register_type cmpge(register_type a, register_type b) { return _mm_cmpge_ps(a, b); }
    // blendv requires SSE4.1
    static int movemask(register_type v) { return _mm_movemask_ps(v); }
    // hadd/hsub require SSE3, dp requires SSE4.1
};

// --- SSE2: Integer Base (Bitwise ops are common) ---
struct SSEIntegerBase
{
    using register_type = __m128i;
    static register_type setzero() { return _mm_setzero_si128(); }
    static register_type load_unaligned(const void* p)
    { // void* for generic load
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    }
    static void store_unaligned(void* p, register_type v) { _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v); }
    static register_type load_aligned(const void* p) { return _mm_load_si128(reinterpret_cast<const __m128i*>(p)); }
    static void          store_aligned(void* p, register_type v) { _mm_store_si128(reinterpret_cast<__m128i*>(p), v); }

    static register_type bitwise_and(register_type a, register_type b) { return _mm_and_si128(a, b); }
    static register_type bitwise_or(register_type a, register_type b) { return _mm_or_si128(a, b); }
    static register_type bitwise_xor(register_type a, register_type b) { return _mm_xor_si128(a, b); }
    static register_type bitwise_andnot(register_type val_to_keep, register_type val_to_invert_mask)
    {
        return _mm_andnot_si128(val_to_invert_mask, val_to_keep);
    }
    // blendv_epi8 requires SSE4.1. For SSE2, a select would be:
    // static register_type blendv(register_type a, register_type b, register_type mask) {
    //    return bitwise_or(bitwise_andnot(mask, a), bitwise_and(mask,b));
    // }
    // This generic blend works if mask has all 0s or all 1s per byte.
    static register_type blendv_generic(register_type a, register_type b, register_type mask)
    {
        return _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b));
    }
};

// --- SSE2: u8 ---
template <> struct RegisterTrait<u8, SSE2InstructionSet> : SSEIntegerBase
{
    using element_type                = u8;
    static constexpr size_t length    = 16;
    static constexpr size_t alignment = 16;

    static register_type set1(element_type v) { return _mm_set1_epi8(static_cast<char>(v)); }
    static register_type add(register_type a, register_type b) { return _mm_add_epi8(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm_sub_epi8(a, b); }
    // No direct mullo_epi8. Widening multiply:
    static register_type mul_widening_to_u16_low(register_type a, register_type b)
    {                                                              // First 8 results
        __m128i a_exp = _mm_unpacklo_epi8(a, _mm_setzero_si128()); // u16
        __m128i b_exp = _mm_unpacklo_epi8(b, _mm_setzero_si128()); // u16
        return _mm_mullo_epi16(a_exp, b_exp);
    }
    static register_type mul_widening_to_u16_high(register_type a, register_type b)
    {                                                              // Next 8 results
        __m128i a_exp = _mm_unpackhi_epi8(a, _mm_setzero_si128()); // u16
        __m128i b_exp = _mm_unpackhi_epi8(b, _mm_setzero_si128()); // u16
        return _mm_mullo_epi16(a_exp, b_exp);
    }

    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_epi8(a, b); }
    static register_type cmpgt_unsigned(register_type a, register_type b)
    {
        // (a > b) unsigned: (a^0x80 > b^0x80) signed
        const __m128i bias = _mm_set1_epi8(static_cast<char>(0x80));
        return _mm_cmpgt_epi8(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias));
    }
    static register_type min(register_type a, register_type b) { return _mm_min_epu8(a, b); }
    static register_type max(register_type a, register_type b) { return _mm_max_epu8(a, b); }
    static int           movemask(register_type v) { return _mm_movemask_epi8(v); }
    static register_type blendv(register_type val_false, register_type val_true, register_type mask)
    {
        return blendv_generic(val_false, val_true, mask);
    }
};

// --- SSE2: i8 ---
template <> struct RegisterTrait<i8, SSE2InstructionSet> : SSEIntegerBase
{
    using element_type                = i8;
    static constexpr size_t length    = 16;
    static constexpr size_t alignment = 16;

    static register_type set1(element_type v) { return _mm_set1_epi8(v); }
    static register_type add(register_type a, register_type b) { return _mm_add_epi8(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm_sub_epi8(a, b); }
    static register_type mul_widening_to_i16_low(register_type a, register_type b)
    {
        __m128i a_exp = _mm_unpacklo_epi8(a, _mm_srai_epi16(_mm_slli_epi16(a, 8), 15)); // sign extend
        __m128i b_exp = _mm_unpacklo_epi8(b, _mm_srai_epi16(_mm_slli_epi16(b, 8), 15));
        return _mm_mullo_epi16(a_exp, b_exp);
    }
    static register_type mul_widening_to_i16_high(register_type a, register_type b)
    {
        __m128i a_exp = _mm_unpackhi_epi8(a, _mm_srai_epi16(_mm_slli_epi16(a, 8), 15));
        __m128i b_exp = _mm_unpackhi_epi8(b, _mm_srai_epi16(_mm_slli_epi16(b, 8), 15));
        return _mm_mullo_epi16(a_exp, b_exp);
    }

    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_epi8(a, b); }
    static register_type cmpgt_signed(register_type a, register_type b) { return _mm_cmpgt_epi8(a, b); }
    static register_type min(register_type a, register_type b)
    {
        // SSE4.1 for _mm_min_epi8
        // Emulation for SSE2: (a < b) ? a : b  => (b > a) ? a : b
        register_type mask = _mm_cmpgt_epi8(b, a); // mask is FF where b > a (a < b)
        return blendv_generic(b, a, mask);
    }
    static register_type max(register_type a, register_type b)
    {
        // SSE4.1 for _mm_max_epi8
        // Emulation for SSE2: (a > b) ? a : b
        register_type mask = _mm_cmpgt_epi8(a, b); // mask is FF where a > b
        return blendv_generic(b, a, mask);
    }
    static register_type abs(register_type a)
    {
        // register_type mask =
        //     _mm_srai_epi16(_mm_slli_epi16(a, 8), 8); // sign extend each byte to 16 bits then shift back to get 0x00
        //     or
        //                                              // 0xFF per byte A bit more direct for epi8 if we had
        //                                              _mm_srai_epi8
        //                                              // Alternative for epi8:
        register_type neg_mask = _mm_cmplt_epi8(a, _mm_setzero_si128()); // 0xFF if negative, 0x00 otherwise
        register_type neg_val  = _mm_sub_epi8(_mm_setzero_si128(), a);
        return _mm_or_si128(_mm_and_si128(neg_mask, neg_val), _mm_andnot_si128(neg_mask, a)); // blend
    }
    static int           movemask(register_type v) { return _mm_movemask_epi8(v); }
    static register_type blendv(register_type val_false, register_type val_true, register_type mask)
    {
        return blendv_generic(val_false, val_true, mask);
    }
};

// --- SSE2: u16 ---
template <> struct RegisterTrait<u16, SSE2InstructionSet> : SSEIntegerBase
{
    using element_type                = u16;
    static constexpr size_t length    = 8;
    static constexpr size_t alignment = 16;

    static register_type set1(element_type v) { return _mm_set1_epi16(static_cast<short>(v)); }
    static register_type add(register_type a, register_type b) { return _mm_add_epi16(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm_sub_epi16(a, b); }
    static register_type mullo(register_type a, register_type b) { return _mm_mullo_epi16(a, b); }
    static register_type mulhi(register_type a, register_type b) { return _mm_mulhi_epu16(a, b); }

    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_epi16(a, b); }
    static register_type cmpgt_unsigned(register_type a, register_type b)
    {
        const __m128i bias = _mm_set1_epi16(static_cast<short>(0x8000));
        return _mm_cmpgt_epi16(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias));
    }
    static register_type min(register_type a, register_type b)
    {
        // SSE4.1 for _mm_min_epu16
        register_type mask = cmpgt_unsigned(b, a); // mask is FFFF where a < b
        return blendv_generic(b, a, mask);
    }
    static register_type max(register_type a, register_type b)
    {
        // SSE4.1 for _mm_max_epu16
        register_type mask = cmpgt_unsigned(a, b); // mask is FFFF where a > b
        return blendv_generic(b, a, mask);
    }

    static register_type slli(register_type v, int count) { return _mm_slli_epi16(v, count); }
    static register_type srli(register_type v, int count) { return _mm_srli_epi16(v, count); }
    static int           movemask(register_type v_cmp)
    {                                                                       // v_cmp has 0xFFFF or 0x0000
        __m128i signs        = _mm_srai_epi16(v_cmp, 15);                   // 0xFFFF or 0x0000
        __m128i packed_bytes = _mm_packs_epi16(signs, _mm_setzero_si128()); // pack to 8 bytes
        return _mm_movemask_epi8(packed_bytes) & 0xFF;                      // only 8 bits valid
    }
    static register_type blendv(register_type val_false, register_type val_true, register_type mask)
    {
        return blendv_generic(val_false, val_true, mask);
    }
};

// --- SSE2: i16 ---
template <> struct RegisterTrait<i16, SSE2InstructionSet> : SSEIntegerBase
{
    using element_type                = i16;
    static constexpr size_t length    = 8;
    static constexpr size_t alignment = 16;

    static register_type set1(element_type v) { return _mm_set1_epi16(v); }
    static register_type add(register_type a, register_type b) { return _mm_add_epi16(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm_sub_epi16(a, b); }
    static register_type mullo(register_type a, register_type b) { return _mm_mullo_epi16(a, b); }
    static register_type mulhi(register_type a, register_type b) { return _mm_mulhi_epi16(a, b); }

    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_epi16(a, b); }
    static register_type cmpgt_signed(register_type a, register_type b) { return _mm_cmpgt_epi16(a, b); }
    static register_type min(register_type a, register_type b)
    {
        return _mm_min_epi16(a, b);
    }
    static register_type max(register_type a, register_type b)
    {
        return _mm_max_epi16(a, b);
    }
    static register_type abs(register_type a)
    {
        register_type mask = _mm_srai_epi16(a, 15);
        return _mm_sub_epi16(_mm_xor_si128(a, mask), mask);
    }

    static register_type slli(register_type v, int count) { return _mm_slli_epi16(v, count); }
    static register_type srli(register_type v, int count) // logical
    {
        return _mm_srli_epi16(v, count);
    }
    static register_type srai(register_type v, int count) // arithmetic
    {
        return _mm_srai_epi16(v, count);
    }
    static int movemask(register_type v_cmp)
    {
        __m128i signs        = _mm_srai_epi16(v_cmp, 15);
        __m128i packed_bytes = _mm_packs_epi16(signs, _mm_setzero_si128());
        return _mm_movemask_epi8(packed_bytes) & 0xFF;
    }
    static register_type blendv(register_type val_false, register_type val_true, register_type mask)
    {
        return blendv_generic(val_false, val_true, mask);
    }
};

// --- SSE2: u32 ---
template <> struct RegisterTrait<u32, SSE2InstructionSet> : SSEIntegerBase
{
    using element_type                = u32;
    static constexpr size_t length    = 4;
    static constexpr size_t alignment = 16;

    static register_type set1(element_type v) { return _mm_set1_epi32(static_cast<int>(v)); }
    static register_type add(register_type a, register_type b) { return _mm_add_epi32(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm_sub_epi32(a, b); }
    // mullo_epi32 is SSE4.1. _mm_mul_epu32 does u32*u32->u64
    static register_type mul_widening_to_u64(register_type a, register_type b)
    {
        return _mm_mul_epu32(a, b); // result has 2 u64s
    }

    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_epi32(a, b); }
    static register_type cmpgt_unsigned(register_type a, register_type b)
    {
        const __m128i bias = _mm_set1_epi32(0x80000000);
        return _mm_cmpgt_epi32(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias));
    }
    static register_type min(register_type a, register_type b)
    { 
        // SSE4.1 for _mm_min_epu32
        register_type mask = cmpgt_unsigned(b, a);
        return blendv_generic(b, a, mask);
    }
    static register_type max(register_type a, register_type b)
    { 
        // SSE4.1 for _mm_max_epu32
        register_type mask = cmpgt_unsigned(a, b);
        return blendv_generic(b, a, mask);
    }

    static register_type slli(register_type v, int count) { return _mm_slli_epi32(v, count); }
    static register_type srli(register_type v, int count) { return _mm_srli_epi32(v, count); }
    // sllv/srlv are AVX2
    static int           movemask(register_type v_cmp) { return _mm_movemask_ps(_mm_castsi128_ps(v_cmp)); }
    static register_type blendv(register_type val_false, register_type val_true, register_type mask)
    {
        return blendv_generic(val_false, val_true, mask);
    }
};

// --- SSE2: i32 ---
template <> struct RegisterTrait<i32, SSE2InstructionSet> : SSEIntegerBase
{
    using element_type                = i32;
    static constexpr size_t length    = 4;
    static constexpr size_t alignment = 16;

    static register_type set1(element_type v) { return _mm_set1_epi32(v); }
    static register_type add(register_type a, register_type b) { return _mm_add_epi32(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm_sub_epi32(a, b); }
    // mullo_epi32 is SSE4.1. _mm_mul_epi32 (SSE4.1) does i32*i32->i64
    // For SSE2, _mm_mul_epu32 can be used for low part if careful with sign extension or if result fits.

    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_epi32(a, b); }
    static register_type cmpgt_signed(register_type a, register_type b) { return _mm_cmpgt_epi32(a, b); }
    static register_type min(register_type a, register_type b)
    { 
        // SSE4.1 for _mm_min_epi32
        register_type mask = _mm_cmpgt_epi32(b, a);
        return blendv_generic(b, a, mask);
    }
    static register_type max(register_type a, register_type b)
    { 
        // SSE4.1 for _mm_max_epi32
        register_type mask = _mm_cmpgt_epi32(a, b);
        return blendv_generic(b, a, mask);
    }
    static register_type abs(register_type a)
    {
        register_type mask = _mm_srai_epi32(a, 31);
        return _mm_sub_epi32(_mm_xor_si128(a, mask), mask);
    }

    static register_type slli(register_type v, int count) { return _mm_slli_epi32(v, count); }
    static register_type srli(register_type v, int count) { return _mm_srli_epi32(v, count); }
    static register_type srai(register_type v, int count) { return _mm_srai_epi32(v, count); }
    // sllv/srlv/srav are AVX2
    static int           movemask(register_type v_cmp) { return _mm_movemask_ps(_mm_castsi128_ps(v_cmp)); }
    static register_type blendv(register_type val_false, register_type val_true, register_type mask)
    {
        return blendv_generic(val_false, val_true, mask);
    }
};

// --- SSE2: u64 ---
template <> struct RegisterTrait<u64, SSE2InstructionSet> : SSEIntegerBase
{
    using element_type                = u64;
    static constexpr size_t length    = 2;
    static constexpr size_t alignment = 16;

    static register_type set1(element_type v) { return _mm_set1_epi64x(static_cast<long long>(v)); }
    static register_type add(register_type a, register_type b) { return _mm_add_epi64(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm_sub_epi64(a, b); }
    // No mullo_epi64. Emulation is complex.

    // cmpeq_epi64 is SSE4.1. cmpgt_epi64 is SSE4.2.
    // Emulation for cmpeq_epi64 for SSE2:
    static register_type cmpeq(register_type a, register_type b)
    {
        __m128i tmp = _mm_xor_si128(a, b); // 0 if equal
        // Check if both 32-bit halves are zero
        __m128i tmp_hi = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(2, 3, 0, 1)); // swap hi/lo 32-bit parts
        tmp            = _mm_or_si128(tmp, tmp_hi);                 // if 64-bit is 0, then this is 0 in lower 32 bits
        tmp            = _mm_cmpeq_epi32(tmp, _mm_setzero_si128()); // 0xFFFFFFFF if 64-bit was 0
        return _mm_shuffle_epi32(tmp, _MM_SHUFFLE(0, 0, 0, 0));     // Broadcast to both 64-bit lanes
    }
    // cmpgt_unsigned for u64 is very hard without SSE4.2/AVX2 features.

    static register_type slli(register_type v, int count) { return _mm_slli_epi64(v, count); }
    static register_type srli(register_type v, int count) { return _mm_srli_epi64(v, count); }
    // sllv/srlv are AVX2
    static int           movemask(register_type v_cmp) { return _mm_movemask_pd(_mm_castsi128_pd(v_cmp)); }
    static register_type blendv(register_type val_false, register_type val_true, register_type mask)
    {
        return blendv_generic(val_false, val_true, mask);
    }
};

// --- SSE2: i64 ---
template <> struct RegisterTrait<i64, SSE2InstructionSet> : SSEIntegerBase
{
    using element_type                = i64;
    static constexpr size_t length    = 2;
    static constexpr size_t alignment = 16;

    static register_type set1(element_type v) { return _mm_set1_epi64x(v); }
    static register_type add(register_type a, register_type b) { return _mm_add_epi64(a, b); }
    static register_type sub(register_type a, register_type b) { return _mm_sub_epi64(a, b); }
    // No mullo_epi64.

    static register_type cmpeq(register_type a, register_type b)
    {
        // Same emulation as u64 for SSE2
        __m128i tmp    = _mm_xor_si128(a, b);
        __m128i tmp_hi = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(2, 3, 0, 1));
        tmp            = _mm_or_si128(tmp, tmp_hi);
        tmp            = _mm_cmpeq_epi32(tmp, _mm_setzero_si128());
        return _mm_shuffle_epi32(tmp, _MM_SHUFFLE(0, 0, 0, 0));
    }
    // cmpgt_signed for i64 is hard without SSE4.2.
    static register_type abs(register_type a)
    {
        // Emulated for i64
        // register_type mask =
        //     _mm_cmpgt_epi64(_mm_setzero_si128(), a); // SSE4.2 for _mm_cmpgt_epi64.
        //                                              // For SSE4.1 and below, this is harder.
        //                                              // A simpler way for abs_epi64 without cmpgt_epi64:
        register_type sign_bits =
            _mm_srai_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(3, 3, 1, 1)), 31); // Get sign bits of high dwords
        sign_bits             = _mm_unpacklo_epi32(sign_bits, sign_bits);      // duplicate to make 64-bit masks
        register_type xor_val = _mm_xor_si128(a, sign_bits);
        return _mm_sub_epi64(xor_val, sign_bits);
    }

    static register_type slli(register_type v, int count) { return _mm_slli_epi64(v, count); }
    static register_type srli(register_type v, int count) { return _mm_srli_epi64(v, count); }
    // srai_epi64 is not available. sllv/srlv/srav are AVX2.
    static int           movemask(register_type v_cmp) { return _mm_movemask_pd(_mm_castsi128_pd(v_cmp)); }
    static register_type blendv(register_type val_false, register_type val_true, register_type mask)
    {
        return blendv_generic(val_false, val_true, mask);
    }
};
} // namespace simd

#endif