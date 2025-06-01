#ifndef VECTOR_H
#define VECTOR_H

#include "defines.h"
#include "types.h"
#include "assert.h"
#include <immintrin.h>
#include <array>
#include <cmath>
#include <cstring>
#include <ostream>
#include <type_traits>

/*
Inspiration:
https://www.openmymind.net/SIMD-With-Zig/
https://ziglang.org/documentation/0.14.1/#Vectors
https://github.com/xtensor-stack/xsimd

fn firstIndexOf(haystack: []const u8, needle: u8) ?usize {
  const vector_len = 8;


  // {111, 111, 111, 111, 111, 111, 111, 111}
  const vector_needles: @Vector(vector_len, u8) = @splat(@as(u8, needle));

  // Because we're implementing our own std.simd.firstTrue
  // we can move the following two vectors, indexes and null
  // outside the loop and re-use them.

  // {0, 1, 2, 3, 4, 5, 6, 7}
  const indexes = std.simd.iota(u8, vector_len);

  // {255, 255, 255, 255, 255, 255, 255, 255}
  const nulls: @Vector(vector_len, u8) = @splat(@as(u8, 255));

  var pos: usize = 0;
  var left = haystack.len;
  while (left > 0) {
    if (left < vector_len) {
      // fallback to a normal scan when our input (or what's left of
      // it is smaller than our vector_len)
      return std.mem.indexOfScalarPos(u8, haystack, pos, needle);
    }

    const h: @Vector(vector_len, u8) = haystack[pos..][0..vector_len].*;
    const matches = h == vector_needles;

    if (@reduce(.Or, matches)) {
      // we found a match, we just need to find its index
      const result = @select(u8, matches, indexes, nulls);

      // we have to add pos to this value, since this is merely
      // the index within this vector_len chunk (e.g. 0-7).
      return @reduce(.Min, result) + pos;
    }

    pos += vector_len;
    left -= vector_len;
  }
  return null;
}
*/
namespace simd
{
// MARK: CPUID
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#else
#warning "CPUID detection not fully implemented for this compiler."
#endif

struct SupportedFeatures
{
    bool sse2 = false;
    bool avx  = false;
    bool avx2 = false;

    inline SupportedFeatures() noexcept
    {
        sse2 = false;
        avx  = false;
        avx2 = false;

        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

        // Get max supported leaf
        get_cpuid(0, 0, &eax, &ebx, &ecx, &edx);
        unsigned int max_leaf = eax;

        if (max_leaf >= 1)
        {
            get_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
            // SSE2: EDX bit 26
            sse2 = (edx & (1 << 26)) != 0;

            // AVX requires OSXSAVE (ECX bit 27) and AVX support (ECX bit 28)
            // and OS support for YMM state (XCR0 bit 2)
            bool osxsave          = (ecx & (1 << 27)) != 0;
            bool cpu_supports_avx = (ecx & (1 << 28)) != 0;

            if (osxsave && cpu_supports_avx)
            {
                uint64_t xcr0 = xgetbv(0); // Read XCR0
                // Check if bits 1 (SSE state) and 2 (YMM state) are set
                avx = (xcr0 & 0x6) == 0x6;
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
    void get_cpuid(unsigned int  leaf,
                   unsigned int  subleaf,
                   unsigned int* eax,
                   unsigned int* ebx,
                   unsigned int* ecx,
                   unsigned int* edx)
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
    uint64_t xgetbv(unsigned int xcr_idx)
    {
#if defined(_MSC_VER)
        return _xgetbv(xcr_idx);
#elif defined(__GNUC__) || defined(__clang__)
        uint32_t eax_val, edx_val;
        __asm__ volatile("xgetbv" : "=a"(eax_val), "=d"(edx_val) : "c"(xcr_idx));
        return (static_cast<uint64_t>(edx_val) << 32) | eax_val;
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

template <typename T>
concept Scalar = std::is_scalar_v<T>;

struct ScalarInstructions
{
};
struct SSE2Instructions
{
};
struct SSE3Instructions
{
};
struct SSE4Instructions
{
};
struct AVXInstructions
{
};
struct AVX2Instructions
{
};

template <typename T, typename InstructionSet> struct RegisterTrait;

template <typename T> struct RegisterTrait<T, ScalarInstructions>;

// -----------------------------------------------------------------------------------------------
// MARK: SSE2 traits
// -----------------------------------------------------------------------------------------------
// #if defined(__SSE2__)
template <> struct RegisterTrait<f64, SSE2Instructions>
{
    using element_type             = f64;
    using register_type            = __m128d;
    static constexpr u64 length    = 2;
    static constexpr u64 alignment = 16;
    // unary
    static register_type set1(f32 v) { return _mm_set1_pd(v); }
    static register_type setzero() { return _mm_setzero_pd(); }
    static register_type load_unaligned(const element_type* p) { return _mm_loadu_pd(p); }
    static void          store_unaligned(element_type* p, register_type v) { _mm_storeu_pd(p, v); }
    static register_type load_aligned(const element_type* p) { return _mm_load_pd(p); }
    static void          store_aligned(element_type* p, register_type v) { _mm_store_pd(p, v); }
    // binary
    static register_type add(register_type a, register_type b) { return _mm_add_pd(a, b); }
    static register_type mul(register_type a, register_type b) { return _mm_mul_pd(a, b); }
    static register_type sqrt(register_type a) { return _mm_sqrt_pd(a); }
    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_pd(a, b); }
    static i32           mask(register_type v) { return _mm_movemask_pd(v); }
};

template <> struct RegisterTrait<f32, SSE2Instructions>
{
    using element_type             = f32;
    using register_type            = __m128;
    static constexpr u64 length    = 4;
    static constexpr u64 alignment = 16;
    // unary
    static register_type set1(element_type v) { return _mm_set1_ps(v); }
    static register_type setzero() { return _mm_setzero_ps(); }
    static register_type load_unaligned(const element_type* p) { return _mm_loadu_ps(p); }
    static void          store_unaligned(element_type* p, register_type v) { _mm_storeu_ps(p, v); }
    static register_type load_aligned(const element_type* p) { return _mm_load_ps(p); }
    static void          store_aligned(element_type* p, register_type v) { _mm_store_ps(p, v); }
    // binary
    static register_type add(register_type a, register_type b) { return _mm_add_ps(a, b); }
    static register_type mul(register_type a, register_type b) { return _mm_mul_ps(a, b); }
    static register_type sqrt(register_type a) { return _mm_sqrt_ps(a); }
    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_ps(a, b); }
    static i32           mask(register_type v) { return _mm_movemask_ps(v); }
};

template <> struct RegisterTrait<i32, SSE2Instructions>
{
    using element_type             = i32;
    using register_type            = __m128i;
    static constexpr u64 length    = 4;
    static constexpr u64 alignment = 16;
    // unary
    static register_type set1(element_type v) { return _mm_set1_epi32(v); }
    static register_type setzero() { return _mm_setzero_si128(); }
    static register_type load_unaligned(const element_type* p) { return _mm_loadu_si128(cast(const register_type*) p); }
    static void store_unaligned(element_type* p, register_type v) { _mm_storeu_si128(cast(register_type*) p, v); }
    static register_type load_aligned(const element_type* p) { return _mm_load_si128(cast(const register_type*) p); }
    static void          store_aligned(element_type* p, register_type v) { _mm_store_si128(cast(register_type*) p, v); }
    //
    static i32           mask(register_type v) { return _mm_movemask_ps(_mm_castsi128_ps(v)); }
    // binary
    static register_type add(register_type a, register_type b) { return _mm_add_epi32(a, b); }
    static register_type mul(register_type a, register_type b)
    {
#if defined(__SSE4_1__)
        return _mm_mullo_epi32(a, b);
#else
        alignas(alignment) std::array<element_type, length> arr_a, arr_b, arr_res;
        store_unaligned(arr_a.data(), a);
        store_unaligned(arr_b.data(), b);
        for (u64 i = 0; i < length; ++i)
            arr_res[i] = arr_a[i] * arr_b[i];
        return load_unaligned(arr_res.data());
#endif
    }
    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_epi32(a, b); }
    static register_type cmpor(register_type a, register_type b) { return _mm_or_si128(a, b); }
};

template <> struct RegisterTrait<u32, SSE2Instructions>
{
    using element_type             = u32;
    using register_type            = __m128i;
    static constexpr u64 length    = 4;
    static constexpr u64 alignment = 16;
    // unary
    static register_type set1(element_type v) { return _mm_set1_epi32(v); }
    static register_type setzero() { return _mm_setzero_si128(); }
    static register_type load_unaligned(const element_type* p) { return _mm_loadu_si128(cast(const register_type*) p); }
    static void store_unaligned(element_type* p, register_type v) { _mm_storeu_si128(cast(register_type*) p, v); }
    static register_type load_aligned(const element_type* p) { return _mm_load_si128(cast(const register_type*) p); }
    static void          store_aligned(element_type* p, register_type v) { _mm_store_si128(cast(register_type*) p, v); }
    // binary
    static register_type add(register_type a, register_type b) { return _mm_add_epi32(a, b); }

    static register_type mul(register_type a, register_type b)
    {
#if defined(__SSE4_1__)
        return _mm_mullo_epi32(a, b);
#else
        alignas(alignment) std::array<element_type, length> arr_a, arr_b, arr_res;
        store_unaligned(arr_a.data(), a);
        store_unaligned(arr_b.data(), b);
        for (u64 i = 0; i < length; ++i)
            arr_res[i] = arr_a[i] * arr_b[i];
        return load_unaligned(arr_res.data());
#endif
    }
    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_epi32(a, b); }
    static i32           mask(register_type v) { return _mm_movemask_ps(_mm_castsi128_ps(v)); }
};

template <> struct RegisterTrait<i8, SSE2Instructions>
{
    using element_type             = i8;
    using register_type            = __m128i;
    static constexpr u64 length    = 16;
    static constexpr u64 alignment = 16;
    // unary
    static register_type set1(element_type v) { return _mm_set1_epi8(v); }
    static register_type setzero() { return _mm_setzero_si128(); }
    static register_type load_unaligned(const element_type* p) { return _mm_loadu_si128(cast(const register_type*) p); }
    static void store_unaligned(element_type* p, register_type v) { _mm_storeu_si128(cast(register_type*) p, v); }
    static register_type load_aligned(const element_type* p) { return _mm_load_si128(cast(const register_type*) p); }
    static void          store_aligned(element_type* p, register_type v) { _mm_store_si128(cast(register_type*) p, v); }
    // binary
    static register_type add(register_type a, register_type b) { return _mm_add_epi8(a, b); }

    // static register_type mul(register_type a, register_type b)
    // {
    // #if defined(__SSE4_1__)
    //     return _mm_mullo_epi(a, b);
    // #else
    //     alignas(alignment) std::array<element_type, length> arr_a, arr_b, arr_res;
    //     store_unaligned(arr_a.data(), a);
    //     store_unaligned(arr_b.data(), b);
    //     for (u64 i = 0; i < length; ++i) arr_res[i] = arr_a[i] * arr_b[i];
    //     return load_unaligned(arr_res.data());
    // #endif
    // }
    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_epi8(a, b); }
    static i32           mask(register_type v) { return _mm_movemask_epi8(v); }
};

template <> struct RegisterTrait<u8, SSE2Instructions>
{
    using element_type             = u8;
    using register_type            = __m128i;
    static constexpr u64 length    = 16;
    static constexpr u64 alignment = 16;
    // unary
    static register_type set1(element_type v) { return _mm_set1_epi8(v); }
    static register_type setzero() { return _mm_setzero_si128(); }
    static register_type load_unaligned(const element_type* p) { return _mm_loadu_si128(cast(const register_type*) p); }
    static void store_unaligned(element_type* p, register_type v) { _mm_storeu_si128(cast(register_type*) p, v); }
    static register_type load_aligned(const element_type* p) { return _mm_load_si128(cast(const register_type*) p); }
    static void          store_aligned(element_type* p, register_type v) { _mm_store_si128(cast(register_type*) p, v); }
    // binary
    static register_type add(register_type a, register_type b) { return _mm_add_epi8(a, b); }

    // static register_type mul(register_type a, register_type b)
    // {
    // #if defined(__SSE4_1__)
    //     return _mm_mullo_epi8(a, b);
    // #else
    //     alignas(alignment) std::array<element_type, length> arr_a, arr_b, arr_res;
    //     store_unaligned(arr_a.data(), a);
    //     store_unaligned(arr_b.data(), b);
    //     for (u64 i = 0; i < length; ++i) arr_res[i] = arr_a[i] * arr_b[i];
    //     return load_unaligned(arr_res.data());
    // #endif
    // }
    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_epi8(a, b); }
    static i32           mask(register_type v) { return _mm_movemask_epi8(v); }
};

// -----------------------------------------------------------------------------------------------
// MARK: AVX Traits
// -----------------------------------------------------------------------------------------------
// #ifdef __AVX__
template <> struct RegisterTrait<f64, AVXInstructions>
{
    using element_type             = f64;
    using register_type            = __m256d;
    static constexpr u64 length    = 4;
    static constexpr u64 alignment = 32;
    // unary
    static register_type set1(f32 v) { return _mm256_set1_pd(v); }
    static register_type setzero() { return _mm256_setzero_pd(); }
    static register_type load_unaligned(const element_type* p) { return _mm256_loadu_pd(p); }
    static void          store_unaligned(element_type* p, register_type v) { _mm256_storeu_pd(p, v); }
    static register_type load_aligned(const element_type* p) { return _mm256_load_pd(p); }
    static void          store_aligned(element_type* p, register_type v) { _mm256_store_pd(p, v); }
    // binary
    static register_type add(register_type a, register_type b) { return _mm256_add_pd(a, b); }
    static register_type mul(register_type a, register_type b) { return _mm256_mul_pd(a, b); }
    static register_type sqrt(register_type a) { return _mm256_sqrt_pd(a); }
    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
    static i32           mask(register_type v) { return _mm256_movemask_pd(v); }
};

template <> struct RegisterTrait<f32, AVXInstructions>
{
    using element_type             = f32;
    using register_type            = __m256;
    static constexpr u64 length    = 8;
    static constexpr u64 alignment = 32;
    // unary
    static register_type set1(element_type v) { return _mm256_set1_ps(v); }
    static register_type setzero() { return _mm256_setzero_ps(); }
    static register_type load_unaligned(const element_type* p) { return _mm256_loadu_ps(p); }
    static void          store_unaligned(element_type* p, register_type v) { _mm256_storeu_ps(p, v); }
    static register_type load_aligned(const element_type* p) { return _mm256_load_ps(p); }
    static void          store_aligned(element_type* p, register_type v) { _mm256_store_ps(p, v); }
    // binary
    static register_type add(register_type a, register_type b) { return _mm256_add_ps(a, b); }
    static register_type mul(register_type a, register_type b) { return _mm256_mul_ps(a, b); }
    static register_type sqrt(register_type a) { return _mm256_sqrt_ps(a); }
    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
    static i32           mask(register_type v) { return _mm256_movemask_ps(v); }
};

// -----------------------------------------------------------------------------------------------
// MARK: AVX2 Traits
// -----------------------------------------------------------------------------------------------

template <> struct RegisterTrait<i8, AVX2Instructions>
{
    using element_type             = i8;
    using register_type            = __m256i;
    static constexpr u64 length    = 32;
    static constexpr u64 alignment = 32;
    // unary
    static register_type set1(element_type v) { return _mm256_set1_epi8(v); }
    static register_type setzero() { return _mm256_setzero_si256(); }
    static register_type load_unaligned(const element_type* p)
    {
        return _mm256_loadu_si256(cast(const register_type*) p);
    }
    static void store_unaligned(element_type* p, register_type v) { _mm256_storeu_si256(cast(register_type*) p, v); }
    static register_type load_aligned(const element_type* p) { return _mm256_load_si256(cast(const register_type*) p); }
    static void store_aligned(element_type* p, register_type v) { _mm256_store_si256(cast(register_type*) p, v); }
    // binary
    static register_type add(register_type a, register_type b) { return _mm256_add_epi8(a, b); }

    // static register_type mul(register_type a, register_type b)
    // {
    // }
    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmpeq_epi8(a, b); }
    static i32           mask(register_type v) { return _mm256_movemask_epi8(v); }
};

template <> struct RegisterTrait<u8, AVX2Instructions>
{
    using element_type             = u8;
    using register_type            = __m256i;
    static constexpr u64 length    = 32;
    static constexpr u64 alignment = 32;
    // unary
    static register_type set1(element_type v) { return _mm256_set1_epi8(v); }
    static register_type setzero() { return _mm256_setzero_si256(); }
    static register_type load_unaligned(const element_type* p)
    {
        return _mm256_loadu_si256(cast(const register_type*) p);
    }
    static void store_unaligned(element_type* p, register_type v) { _mm256_storeu_si256(cast(register_type*) p, v); }
    static register_type load_aligned(const element_type* p) { return _mm256_load_si256(cast(const register_type*) p); }
    static void store_aligned(element_type* p, register_type v) { _mm256_store_si256(cast(register_type*) p, v); }
    // binary
    static register_type add(register_type a, register_type b) { return _mm256_add_epi8(a, b); }

    // static register_type mul(register_type a, register_type b)
    // {
    // }
    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmpeq_epi8(a, b); }
    static i32           mask(register_type v) { return _mm256_movemask_epi8(v); }
};

template <> struct RegisterTrait<i32, AVX2Instructions>
{
    using element_type             = i32;
    using register_type            = __m256i;
    static constexpr u64 length    = 8;
    static constexpr u64 alignment = 32;
    // unary
    static register_type set1(element_type v) { return _mm256_set1_epi32(v); }
    static register_type setzero() { return _mm256_setzero_ps(); }
    static register_type load_unaligned(const element_type* p)
    {
        return _mm256_loadu_si256(cast(const register_type*) p);
    }
    static void store_unaligned(element_type* p, register_type v) { _mm256_storeu_si256(cast(register_type*) p, v); }
    static register_type load_aligned(const element_type* p) { return _mm256_load_si256(cast(const register_type*) p); }
    static void store_aligned(element_type* p, register_type v) { _mm256_store_si256(cast(register_type*) p, v); }
    // binary
    static register_type add(register_type a, register_type b) { return _mm256_add_epi32(a, b); }
    static register_type mul(register_type a, register_type b) { return _mm256_mul_epi32(a, b); }
    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmpeq_epi32(a, b); }
};

template <typename T, typename Tag>
concept RegisterTraitRequirement = requires { typename RegisterTrait<T, Tag>::element_type; };

template <typename T, u64 N, typename Tag>
concept RegisterTraitLengthRequirement = (RegisterTraitRequirement<T, Tag> && RegisterTrait<T, Tag>::length == N);

inline u8 bit_scan_first_set_bit(i32 mask)
{
    u8 res;
#if defined(__GNUC__) || defined(__clang__)
    res = cast(u8) __builtin_ctz(mask);
    return res;
#elif defined(_MSC_VER)
    _BitScanForward(&res, cast(u64) mask);
    return res;
#else
    u8 count = 0;
    while ((mask & 1) == 0)
    {
        mask >>= 1;
        count++;
    }
    return count;
#endif
}

template <typename T, typename InstructionSet> struct InstructionSetLength
{
    inline static bool fits(u64 length)
    {
        if constexpr (requires { RegisterTrait<T, InstructionSet>::length != 0; })
        {
            return length >= RegisterTrait<T, InstructionSet>::length;
        }
        else
        {
            return false;
        }
    }
};

// -----------------------------------------------------------------------------------------------
// MARK: Vector
// -----------------------------------------------------------------------------------------------

template <typename T, typename InstructionSet> struct alignas(RegisterTrait<T, InstructionSet>::alignment) Vector
{
    using register_trait = RegisterTrait<T, InstructionSet>;
    using register_type  = typename register_trait::register_type;
    using element_type   = T;

    static_assert(RegisterTraitRequirement<T, InstructionSet>);
    static_assert(register_trait::length > 0);

    static constexpr u64 length    = register_trait::length;
    static constexpr u64 alignment = register_trait::alignment;

    register_type m_data;

    Vector() noexcept
        : m_data(register_trait::setzero())
    {
    }

    explicit Vector(register_type reg) noexcept
        : m_data(reg)
    {
    }

    Vector(T (&arr)[length]) noexcept
        : m_data(register_trait::load_unaligned(arr))
    {
        static_assert(length == length);
    }

    static Vector splat(T scalyr) { return Vector(register_trait::set1(scalyr)); }

    static Vector load_unaligned(const T* ptr) { return Vector(register_trait::load_unaligned(ptr)); }
    void          store_unaligned(T* ptr) const { register_trait::store_unaligned(ptr, m_data); }
    static Vector load_aligned(const T* ptr) { return Vector(register_trait::load_aligned(ptr)); }
    void          store_aligned(T* ptr) const { register_trait::store_aligned(ptr, m_data); }

    T operator[](u64 i) const
    {
        Assert(i < length);
        alignas(alignment) std::array<T, length> temp{};
        register_trait::store_aligned(temp.data(), m_data);
        return temp[i];
    }

    Vector operator+(const Vector& other) const { return Vector(register_trait::add(m_data, other.m_data)); }
    Vector operator*(const Vector& other) const { return Vector(register_trait::mul(m_data, other.m_data)); }

    friend Vector sqrt(const Vector& v)
    {
        if constexpr (std::is_invocable_r_v<register_type, decltype(register_trait::sqrt), register_type>)
        {
            return Vector(register_trait::sqrt(v.m_data));
        }
        else
        {
            alignas(alignment) std::array<T, length> elements;
            v.store_aligned(elements.data());

            for (u64 i = 0; i < length; ++i)
            {
                if constexpr (std::is_floating_point_v<T>)
                {
                    elements[i] = static_cast<T>(std::sqrt(elements[i]));
                }
                else if constexpr (std::is_integral_v<T>)
                {
                    // integer sqrt is different - floor(sqrt(x))
                    // assert(false, "integer sqrt not implemented for SIMD fallback")
                    elements[i] = static_cast<T>(std::sqrt(static_cast<double>(elements[i])));
                }
            }
            return Vector(register_trait::load_aligned(elements.data()));
        }
    }

    Vector operator==(const Vector& other) const { return Vector(register_trait::cmpeq(m_data, other.m_data)); }

    i32 mask() const { return register_trait::mask(m_data); }

    friend std::ostream& operator<<(std::ostream& os, const Vector& v)
    {
        alignas(alignment) std::array<T, length> temp;
        register_trait::load_aligned(temp.data(), v.m_data());
        os << "Vector(" << length << ", [";
        for (u64 i = 0; i < length; ++i)
        {
            if constexpr (std::is_same_v<T, i8> || std::is_same_v<T, u8>)
            {
                os << cast(i32) temp[i];
            }
            else
            {
                os << temp[i];
            }
            if (i < length - 1)
                os << ", ";
        }
        os << "])";
        return os;
    }
};

template <typename T, typename InstructionSet> inline i64 first_index_of(const T* ptr, u64 length, T needle)
{
    using Vector           = simd::Vector<u8, InstructionSet>;
    constexpr u64 v_length = Vector::length;
    auto          v_needle = Vector::splat(needle);
    u64           pos      = 0;
    while (pos + v_length <= length)
    {
        auto v_haystack = Vector::load_unaligned(ptr + pos);
        auto matches    = v_haystack == v_needle;
        auto mask       = matches.mask();
        if (mask != 0)
        {
            auto match_offset_in_v = simd::bit_scan_first_set_bit(mask);
            return pos + match_offset_in_v;
        }
        pos += v_length;
    }
    for (; pos < length; ++pos)
    {
        if (ptr[pos] == needle)
            return pos;
    }
    return -1;
}

template <typename T> inline i64 first_index_of(const T* ptr, u64 length, T needle)
{
    if (length == 0)
        return -1;
    if constexpr (RegisterTraitRequirement<T, AVX2Instructions>)
    {
        if (supported_features().avx2 && InstructionSetLength<T, AVX2Instructions>::fits(length))
        {
            return first_index_of<T, AVX2Instructions>(ptr, length, needle);
        }
    }
    if constexpr (RegisterTraitRequirement<T, AVXInstructions>)
    {
        if (supported_features().avx && InstructionSetLength<T, AVXInstructions>::fits(length))
        {
            return first_index_of<T, AVXInstructions>(ptr, length, needle);
        }
    }
    if constexpr (RegisterTraitRequirement<T, SSE2Instructions>)
    {
        if (supported_features().sse2 && InstructionSetLength<T, SSE2Instructions>::fits(length))
        {
            return first_index_of<T, SSE2Instructions>(ptr, length, needle);
        }
    }
    for (u64 i = 0; i < length; ++i)
    {
        if (ptr[i] == needle)
            return i;
    }
    return -1;
}
} // namespace simd

// inline i64 firstIndexOfVectorized(Slice<u8> haystack, Slice<u8> needle)
// {
//     if (haystack.len() < needle.len() || haystack.empty() || needle.empty())
//     {
//         return -1;
//     }
//     constexpr auto v_length = Vector<u8>::length;
//     auto           v_needle = Vector<u8>::splat(needle);
//     u64    pos      = 0;
//     while (pos + v_length <= haystack.len())
//     {
//         auto v_haystack = Vector<u8>::load_unaligned(haystack.slice_from(pos).data());
//         auto matches    = v_haystack == v_needle;
//         auto mask       = matches.mask();
//         if (mask != 0)
//         {
//             auto match_offset_in_v = bit_scan_first_set_bit(mask);
//             return pos + match_offset_in_v;
//         }
//         pos += v_length;
//     }
//     for (; pos < haystack.len(); ++pos)
//     {
//         if (haystack[pos] == needle)
//             return pos;
//     }
//     return -1;
// }

#endif