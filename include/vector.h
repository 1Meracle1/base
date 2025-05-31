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

struct dummy_tag
{
};
struct sse2_tag
{
};
struct avx_tag
{
};

template <typename T, typename Tag> struct RegisterTrait;

template <typename T> struct RegisterTrait<T, dummy_tag>
{
    using element_type                     = T;
    static constexpr std::size_t length    = 0;
    static constexpr std::size_t alignment = 0;
};

// -----------------------------------------------------------------------------------------------
// MARK: SSE2 traits
// -----------------------------------------------------------------------------------------------
// #if defined(__SSE2__)
template <> struct RegisterTrait<f64, sse2_tag>
{
    using element_type                     = f64;
    using register_type                    = __m128d;
    static constexpr std::size_t length    = 2;
    static constexpr std::size_t alignment = 16;
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
    static register_type sqrt(register_type a) { return _mm_sqrt_ps(a); }
    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_pd(a, b); }
    static i32           mask(register_type v) { return _mm_movemask_pd(v); }
};

template <> struct RegisterTrait<f32, sse2_tag>
{
    using element_type                     = f32;
    using register_type                    = __m128;
    static constexpr std::size_t length    = 4;
    static constexpr std::size_t alignment = 16;
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

template <> struct RegisterTrait<i32, sse2_tag>
{
    using element_type                     = i32;
    using register_type                    = __m128i;
    static constexpr std::size_t length    = 4;
    static constexpr std::size_t alignment = 16;
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
        for (std::size_t i = 0; i < length; ++i)
            arr_res[i] = arr_a[i] * arr_b[i];
        return load_unaligned(arr_res.data());
#endif
    }
    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_epi32(a, b); }
    static i32           mask(register_type v) { return _mm_movemask_ps(_mm_castsi128_ps(v)); }
};

template <> struct RegisterTrait<u32, sse2_tag>
{
    using element_type                     = u32;
    using register_type                    = __m128i;
    static constexpr std::size_t length    = 4;
    static constexpr std::size_t alignment = 16;
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
        for (std::size_t i = 0; i < length; ++i)
            arr_res[i] = arr_a[i] * arr_b[i];
        return load_unaligned(arr_res.data());
#endif
    }
    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_epi32(a, b); }
    static i32           mask(register_type v) { return _mm_movemask_ps(_mm_castsi128_ps(v)); }
};

template <> struct RegisterTrait<i8, sse2_tag>
{
    using element_type                     = i8;
    using register_type                    = __m128i;
    static constexpr std::size_t length    = 16;
    static constexpr std::size_t alignment = 16;
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
    //     for (std::size_t i = 0; i < length; ++i) arr_res[i] = arr_a[i] * arr_b[i];
    //     return load_unaligned(arr_res.data());
    // #endif
    // }
    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_epi8(a, b); }
    static i32           mask(register_type v) { return _mm_movemask_epi8(v); }
};

template <> struct RegisterTrait<u8, sse2_tag>
{
    using element_type                     = u8;
    using register_type                    = __m128i;
    static constexpr std::size_t length    = 16;
    static constexpr std::size_t alignment = 16;
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
    //     for (std::size_t i = 0; i < length; ++i) arr_res[i] = arr_a[i] * arr_b[i];
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
template <> struct RegisterTrait<f64, avx_tag>
{
    using element_type                     = f64;
    using register_type                    = __m256d;
    static constexpr std::size_t length    = 4;
    static constexpr std::size_t alignment = 32;
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
    static register_type sqrt(register_type a) { return _mm256_sqrt_ps(a); }
    static register_type cmpeq(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
    static i32           mask(register_type v) { return _mm256_movemask_pd(v); }
};

template <> struct RegisterTrait<f32, avx_tag>
{
    using element_type                     = f32;
    using register_type                    = __m256;
    static constexpr std::size_t length    = 8;
    static constexpr std::size_t alignment = 32;
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

template <> struct RegisterTrait<i8, avx_tag>
{
    using element_type                     = i8;
    using register_type                    = __m256i;
    static constexpr std::size_t length    = 32;
    static constexpr std::size_t alignment = 32;
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

template <> struct RegisterTrait<u8, avx_tag>
{
    using element_type                     = u8;
    using register_type                    = __m256i;
    static constexpr std::size_t length    = 32;
    static constexpr std::size_t alignment = 32;
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

// -----------------------------------------------------------------------------------------------
// MARK: AVX2 Traits
// -----------------------------------------------------------------------------------------------
#ifdef __AVX2__
template <> struct RegisterTrait<i32, avx_tag>
{
    using element_type                     = i32;
    using register_type                    = __m256i;
    static constexpr std::size_t length    = 8;
    static constexpr std::size_t alignment = 32;
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
#else
template <> struct RegisterTrait<i32, avx_tag> : RegisterTrait<i32, sse2_tag>
{
};
#endif
// #endif
// #endif

template <typename T, typename Tag>
concept RegisterTraitRequirement = requires { typename RegisterTrait<T, Tag>::element_type; };

template <typename T, std::size_t N, typename Tag>
concept RegisterTraitLengthRequirement = (RegisterTraitRequirement<T, Tag> && RegisterTrait<T, Tag>::length == N);

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
        u32 eax, ebx, ecx, edx;
        get_cpuid(0, 0, &eax, &ebx, &ecx, &edx);
        u32 max_leaf = eax;
        if (max_leaf >= 1)
        {
            get_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
            sse2 = (edx & (1 << 26)) != 0;
        }
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
        if (subleaf == 0)
        {
            __cpuid(regs, static_cast<int>(leaf));
        }
        else
        {
            __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
        }
        *eax = regs[0];
        *ebx = regs[1];
        *ecx = regs[2];
        *edx = regs[3];
#elif defined(__GNUC__) || defined(__clang__)
        __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
#else
// TODO: cases when intrinsics are not available
#endif
    }

    // XGETBV instruction to check OS support for AVX state saving
    uint64_t xgetbv(unsigned int xcr)
    {
#if defined(_MSC_VER)
        return _xgetbv(xcr);
#elif defined(__GNUC__) || defined(__clang__)
        uint32_t eax, edx;
        __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(xcr));
        return (static_cast<uint64_t>(edx) << 32) | eax;
#else
        return 0; // Not supported or unknown
#endif
    }
};

static SupportedFeatures supported_features{};

#ifdef __SSE2__
constexpr bool has_sse2 = true;
#else
constexpr bool has_sse2 = false;
#endif
#ifdef __AVX__
constexpr bool has_avx = true;
#else
constexpr bool has_avx = false;
#endif

template <typename T, std::size_t N> auto tag_from_type_and_length()
{
    if constexpr (has_avx && RegisterTraitLengthRequirement<T, N, avx_tag>)
    {
        return std::type_identity<avx_tag>{};
    }
    else if constexpr (has_sse2 && RegisterTraitLengthRequirement<T, N, sse2_tag>)
    {
        return std::type_identity<sse2_tag>{};
    }
    else
    {
        return std::type_identity<dummy_tag>{};
    }
};

template <typename T, std::size_t N>
using AvailableRegisterTrait = RegisterTrait<T, typename decltype(tag_from_type_and_length<T, N>())::type>;

template <typename T> constexpr std::size_t max_length_available()
{
    if constexpr (has_avx && RegisterTraitRequirement<T, avx_tag>)
    {
        return RegisterTrait<T, avx_tag>::length;
    }
    else if constexpr (has_sse2 && RegisterTraitRequirement<T, sse2_tag>)
    {
        return RegisterTrait<T, sse2_tag>::length;
    }
    else
    {
        return 0;
    }
}

inline u64 bit_scan_first_set_bit(i32 mask)
{
    u64 res;
#if defined(__GNUC__) || defined(__clang__)
    res = cast(u64) __builtin_ctz(mask);
#elif defined(_MSC_VER)
    _BitScanForward(&res, cast(u64) mask);
#else
    static_assert(false, "not implemented for this compiler");
#endif
    return res;
}

// -----------------------------------------------------------------------------------------------
// MARK: Vector
// -----------------------------------------------------------------------------------------------

template <typename T, std::size_t N> struct alignas(AvailableRegisterTrait<T, N>::alignment) Vector
{
    using register_trait = AvailableRegisterTrait<T, N>;
    using register_type  = register_trait::register_type;

    static constexpr std::size_t length    = register_trait::length;
    static constexpr std::size_t alignment = register_trait::alignment;

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

    T operator[](std::size_t i) const
    {
        Assert(i < length);
        alignas(alignment) std::array<T, length> temp{};
        // TODO: on which alignments we have to still use unaligned store?
        register_trait::store_unaligned(temp.data(), m_data);
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
            // static_assert(std::is_floating_point_v<T>, "sqrt works only for floating point types.");
            alignas(alignment) std::array<T, length> arr_v, arr_res;
            // TODO: on which alignments we have to still use unaligned load?
            register_trait::load_aligned(arr_v.data(), v.m_data());
            for (std::size_t i = 0; i < length; ++i)
            {
                arr_res[i] = cast(T) std::sqrt(cast(f64) arr_v[i]);
            }
            // TODO: on which alignments we have to still use unaligned load?
            return Vector(register_trait::load_unaligned(arr_res.data()));
        }
    }

    Vector operator==(const Vector& other) const { return Vector(register_trait::cmpeq(m_data, other.m_data)); }

    i32 mask() const { return register_trait::mask(m_data); }

    friend std::ostream& operator<<(std::ostream& os, const Vector& v)
    {
        alignas(alignment) std::array<T, length> temp;
        // TODO: on which alignments we have to still use unaligned load?
        register_trait::load_aligned(temp.data(), v.m_data());
        os << "Vector(" << length << ", [";
        for (std::size_t i = 0; i < length; ++i)
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
} // namespace simd

// -----------------------------------------------------------------------------------------------
// MARK: Examples
// -----------------------------------------------------------------------------------------------

#include "slice.h"

inline i64 firstIndexOfVectorized(Slice<u8> haystack, u8 needle)
{
    constexpr auto v_length = simd::max_length_available<u8>();
    auto           v_needle = simd::Vector<u8, v_length>::splat(needle);
    std::size_t    pos      = 0;
    while (pos + v_length <= haystack.len())
    {
        auto v_haystack = simd::Vector<u8, v_length>::load_unaligned(haystack.slice_from(pos).data());
        auto matches    = v_haystack == v_needle;
        auto mask       = matches.mask();
        if (mask != 0)
        {
            auto match_offset_in_v = simd::bit_scan_first_set_bit(mask);
            return pos + match_offset_in_v;
        }
        pos += v_length;
    }
    for (; pos < haystack.len(); ++pos)
    {
        if (haystack[pos] == needle)
            return pos;
    }
    return -1;
}

inline i64 firstIndexOf(Slice<u8> haystack, u8 needle)
{
    for (std::size_t pos = 0; pos < haystack.len(); ++pos)
    {
        if (haystack[pos] == needle)
            return pos;
    }
    return -1;
}

// inline i64 firstIndexOfVectorized(Slice<u8> haystack, Slice<u8> needle)
// {
//     if (haystack.len() < needle.len() || haystack.empty() || needle.empty())
//     {
//         return -1;
//     }
//     constexpr auto v_length = Vector<u8>::length;
//     auto           v_needle = Vector<u8>::splat(needle);
//     std::size_t    pos      = 0;
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

/*
// speedup of firstIndexOfVectorized is 1.5 on SSE2, 2.6 on AVX1

MeasureTimeStats2 stats_seq_search;
MeasureTimeStats2 stats_vectorized_search;
for (std::size_t iteration = 0; iteration < 10000; ++iteration)
{
    for (i32 ch = 33; ch < 127; ++ch)
    {
        u8 c = cast(u8) ch;

        stats_vectorized_search.start();
        auto maybe_index_vec = firstIndexOfVectorized(data, c);
        stats_vectorized_search.end();

        stats_seq_search.start();
        auto maybe_index = firstIndexOf(data, c);
        stats_seq_search.end();

        Assert(maybe_index == maybe_index_vec);
    }
}
stats_vectorized_search.print_summary_with_reference_ms(
    ByteSliceFromCstr("vectorized search of single character vs sequential search"), stats_seq_search);
*/

#endif