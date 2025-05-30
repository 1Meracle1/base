#ifndef VECTOR_H
#define VECTOR_H

#include "defines.h"
#include "types.h"
#include "assert.h"
#include <array>
#include <cmath>
#include <cstring>
#include <emmintrin.h>
#include <immintrin.h>
#include <initializer_list>
#include <ostream>
#include <type_traits>
#include <xmmintrin.h>

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

// clang-format off
  struct SSE2_Tag {};
  struct AVX_Tag {};
  struct Scalar_Tag {};

  template <typename RegisterType> struct RegisterAlignment;
  template<> struct RegisterAlignment<__m128>  { static constexpr std::size_t value = 16; };
  template<> struct RegisterAlignment<__m128i> { static constexpr std::size_t value = 16; };
  template<> struct RegisterAlignment<__m128d> { static constexpr std::size_t value = 16; };
  template<> struct RegisterAlignment<__m256>  { static constexpr std::size_t value = 32; };
  template<> struct RegisterAlignment<__m256i> { static constexpr std::size_t value = 32; };
  template<> struct RegisterAlignment<__m256d> { static constexpr std::size_t value = 32; };

template <typename T, typename Tag> struct RegisterTrait;

// MARK: SSE2 traits
#if defined(__SSE2__)
    template <> struct RegisterTrait<f64, SSE2_Tag>
    {
        using element_type                     = f64;
        using register_type                    = __m128d;
        static constexpr std::size_t length    = 2;
        static constexpr std::size_t alignment = RegisterAlignment<register_type>::value;
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

    template <> struct RegisterTrait<f32, SSE2_Tag>
    {
        using element_type                     = f32;
        using register_type                    = __m128;
        static constexpr std::size_t length    = 4;
        static constexpr std::size_t alignment = RegisterAlignment<register_type>::value;
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

    template <> struct RegisterTrait<i32, SSE2_Tag>
    {
        using element_type                     = i32;
        using register_type                    = __m128i;
        static constexpr std::size_t length    = 4;
        static constexpr std::size_t alignment = RegisterAlignment<register_type>::value;
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

    template <> struct RegisterTrait<u32, SSE2_Tag>
    {
        using element_type                     = u32;
        using register_type                    = __m128i;
        static constexpr std::size_t length    = 4;
        static constexpr std::size_t alignment = RegisterAlignment<register_type>::value;
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

    template <> struct RegisterTrait<i8, SSE2_Tag>
    {
        using element_type                     = i8;
        using register_type                    = __m128i;
        static constexpr std::size_t length    = 16;
        static constexpr std::size_t alignment = RegisterAlignment<register_type>::value;
        // unary
        static register_type set1(element_type v) { return _mm_set1_epi8(v); }
        static register_type setzero() { return _mm_setzero_si128(); }
        static register_type load_unaligned(const element_type* p) { return _mm_loadu_si128(cast(const register_type*) p); }
        static void          store_unaligned(element_type* p, register_type v) { _mm_storeu_si128(cast(register_type*) p, v); }
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

    template <> struct RegisterTrait<u8, SSE2_Tag>
    {
        using element_type                     = u8;
        using register_type                    = __m128i;
        static constexpr std::size_t length    = 16;
        static constexpr std::size_t alignment = RegisterAlignment<register_type>::value;
        // unary
        static register_type set1(element_type v) { return _mm_set1_epi8(v); }
        static register_type setzero() { return _mm_setzero_si128(); }
        static register_type load_unaligned(const element_type* p) { return _mm_loadu_si128(cast(const register_type*) p); }
        static void          store_unaligned(element_type* p, register_type v) { _mm_storeu_si128(cast(register_type*) p, v); }
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

    // MARK: AVX Traits
    #ifdef __AVX__
        template <> struct RegisterTrait<f64, AVX_Tag>
        {
            using element_type                     = f64;
            using register_type                    = __m256d;
            static constexpr std::size_t length    = 4;
            static constexpr std::size_t alignment = RegisterAlignment<register_type>::value;
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

        template <> struct RegisterTrait<f32, AVX_Tag>
        {
            using element_type                     = f32;
            using register_type                    = __m256;
            static constexpr std::size_t length    = 8;
            static constexpr std::size_t alignment = RegisterAlignment<register_type>::value;
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

    template <> struct RegisterTrait<i8, AVX_Tag>
    {
        using element_type                     = i8;
        using register_type                    = __m256i;
        static constexpr std::size_t length    = 32;
        static constexpr std::size_t alignment = RegisterAlignment<register_type>::value;
        // unary
        static register_type set1(element_type v) { return _mm256_set1_epi8(v); }
        static register_type setzero() { return _mm256_setzero_si256(); }
        static register_type load_unaligned(const element_type* p) { return _mm256_loadu_si256(cast(const register_type*) p); }
        static void          store_unaligned(element_type* p, register_type v) { _mm256_storeu_si256(cast(register_type*) p, v); }
        static register_type load_aligned(const element_type* p) { return _mm256_load_si256(cast(const register_type*) p); }
        static void          store_aligned(element_type* p, register_type v) { _mm256_store_si256(cast(register_type*) p, v); }
        // binary
        static register_type add(register_type a, register_type b) { return _mm256_add_epi8(a, b); }

        // static register_type mul(register_type a, register_type b)
        // {
        // }
        static register_type cmpeq(register_type a, register_type b) { return _mm256_cmpeq_epi8(a, b); }
        static i32           mask(register_type v) { return _mm256_movemask_epi8(v); }
    };

    template <> struct RegisterTrait<u8, AVX_Tag>
    {
        using element_type                     = u8;
        using register_type                    = __m256i;
        static constexpr std::size_t length    = 32;
        static constexpr std::size_t alignment = RegisterAlignment<register_type>::value;
        // unary
        static register_type set1(element_type v) { return _mm256_set1_epi8(v); }
        static register_type setzero() { return _mm256_setzero_si256(); }
        static register_type load_unaligned(const element_type* p) { return _mm256_loadu_si256(cast(const register_type*) p); }
        static void          store_unaligned(element_type* p, register_type v) { _mm256_storeu_si256(cast(register_type*) p, v); }
        static register_type load_aligned(const element_type* p) { return _mm256_load_si256(cast(const register_type*) p); }
        static void          store_aligned(element_type* p, register_type v) { _mm256_store_si256(cast(register_type*) p, v); }
        // binary
        static register_type add(register_type a, register_type b) { return _mm256_add_epi8(a, b); }

        // static register_type mul(register_type a, register_type b)
        // {
        // }
        static register_type cmpeq(register_type a, register_type b) { return _mm256_cmpeq_epi8(a, b); }
        static i32           mask(register_type v) { return _mm256_movemask_epi8(v); }
    };

        // MARK: AVX2 Traits
        #ifdef __AVX2__
            template <> struct RegisterTrait<i32, AVX_Tag>
            {
                using element_type                     = i32;
                using register_type                    = __m256i;
                static constexpr std::size_t length    = 8;
                static constexpr std::size_t alignment = RegisterAlignment<register_type>::value;
                // unary
                static register_type set1(element_type v) { return _mm256_set1_epi32(v); }
                static register_type setzero() { return _mm256_setzero_ps(); }
                static register_type load_unaligned(const element_type* p) { return _mm256_loadu_si256(cast(const register_type*)p); }
                static void          store_unaligned(element_type* p, register_type v) { _mm256_storeu_si256(cast(register_type*)p, v); }
                static register_type load_aligned(const element_type* p) { return _mm256_load_si256(cast(const register_type*)p); }
                static void          store_aligned(element_type* p, register_type v) { _mm256_store_si256(cast(register_type*)p, v); }
                // binary
                static register_type add(register_type a, register_type b) { return _mm256_add_epi32(a, b); }
                static register_type mul(register_type a, register_type b) { return _mm256_mul_epi32(a, b); }
                static register_type cmpeq(register_type a, register_type b) { return _mm256_cmpeq_epi32(a, b); }
            };
        #else
            template <> struct RegisterTrait<i32, AVX_Tag>: RegisterTrait<i32, SSE2_Tag>
            {
            };
        #endif
    #endif
#endif
// clang-format on

template <typename T, std::size_t N = 4> struct ScalyrRegisterType
{
    using element_type = T;
    std::array<element_type, N> data{};
    ScalyrRegisterType() = default;
    explicit ScalyrRegisterType(element_type v) { data.fill(v); }
    ScalyrRegisterType(const element_type* p) { std::copy(p, p + N, data.begin()); }
};

template <> struct RegisterTrait<f32, Scalar_Tag>
{
    static constexpr std::size_t length    = 4;
    using register_type                    = ScalyrRegisterType<f32, length>;
    static constexpr std::size_t alignment = alignof(f32);
    // unary
    static register_type set1(f32 v) { return register_type(v); }
    static register_type setzero() { return register_type(); }
    static register_type load_unaligned(const f32* p) { return register_type(p); }
    static void          store_unaligned(f32* p, register_type v) { std::copy(v.data.begin(), v.data.end(), p); }
    static register_type load_aligned(const f32* p) { return load_unaligned(p); }
    static void          store_aligned(f32* p, register_type v) { store_unaligned(p, v); }
    // binary
    // clang-format off
    static register_type add(register_type a, register_type b) { register_type res; for(std::size_t i = 0; i < length; ++i) { res.data[i] = a.data[i] + b.data[i]; } return res; }
    static register_type mul(register_type a, register_type b) { register_type res; for(std::size_t i = 0; i < length; ++i) { res.data[i] = a.data[i] * b.data[i]; } return res; }
    static register_type sqrt(register_type a) { register_type res; for(std::size_t i = 0; i < length; ++i) { res.data[i] = std::sqrt(a.data[i]); } return res; }
    static register_type cmpeq(register_type a, register_type b) { register_type res; f32 all_ones; std::memset(&all_ones, 0xFF, sizeof(f32)); for(std::size_t i = 0; i < length; ++i) { res.data[i] = (a.data[i] == b.data[i]) ? all_ones : 0.0f; } return res; }
    // clang-format on
};

struct SSE2_Policy
{
    using Tag = SSE2_Tag;

    static constexpr bool        is_parallel = true;
    static constexpr const char* name        = "SSE2";
};

struct AVX_Policy
{
    using Tag = AVX_Tag;

    static constexpr bool        is_parallel = true;
    static constexpr const char* name        = "AVX";
};

struct Scalar_Policy
{
    using Tag = Scalar_Tag;

    static constexpr bool        is_parallel = false;
    static constexpr const char* name        = "Scalar";
};

#if defined(__AVX__)
using BestPolicy = AVX_Policy;
#elif defined(__SSE2__)
using BestPolicy = SSE2_Policy;
#else
using BestPolicy = Scalyr_Policy;
#endif

} // namespace simd

inline u64 bit_scan(i32 mask)
{
    u64 res;
#if defined(__GNUC__) || defined(__clang__)
    res = cast(u64) __builtin_ctz(mask);
#elif defined(_MSC_VER)
    _BitScanForward(&res, cast(u64) mask);
#else
    static_assert(false, "bit_scan not implemented for this compiler");
#endif
    return res;
}

template <typename T, typename Policy = simd::BestPolicy>
struct alignas(simd::RegisterTrait<T, typename Policy::Tag>::alignment) Vector
{
    using ops                              = simd::RegisterTrait<T, typename Policy::Tag>;
    using register_type                    = typename ops::register_type;
    static constexpr std::size_t length    = ops::length;
    static constexpr std::size_t alignment = ops::alignment;

    register_type data;

    Vector() noexcept
        : data(ops::setzero())
    {
    }
    explicit Vector(T scalyr) noexcept
        : data(ops::set1(scalyr))
    {
    }
    Vector(register_type reg) noexcept
        : data(reg)
    {
    }
    // Vector(std::initializer_list<T> init_list) noexcept
    //     : data(ops::setzero())
    // {
    //     Assert(init_list.size() <= length);
    //     alignas(alignment) std::array<T, length> temp{};
    //     std::size_t i = 0;
    //     for(auto it = init_list.begin(), itEnd = init_list.end(); it != itEnd; ++it)
    //     {
    //         if (i < length)
    //             temp[i] = *it;
    //         ++i;
    //     }
    //     data = ops::load_unaligned(temp.data());
    // }
    template <std::size_t N>
    Vector(T (&arr)[N]) noexcept
        : data(ops::load_unaligned(arr))
    {
        static_assert(N == length);
    }

    static Vector splat(T scalyr) { return Vector(ops::set1(scalyr)); }

    static Vector load_unaligned(const T* ptr) { return Vector(ops::load_unaligned(ptr)); }
    void          store_unaligned(T* ptr) const { ops::store_unaligned(ptr, data); }
    static Vector load_aligned(const T* ptr) { return Vector(ops::load_aligned(ptr)); }
    void          store_aligned(T* ptr) const { ops::store_aligned(ptr, data); }

    T operator[](std::size_t i) const
    {
        Assert(i < length);
        alignas(alignment) std::array<T, length> temp{};
        // TODO: on which alignments we have to still use unaligned store?
        ops::store_unaligned(temp.data(), data);
        return temp[i];
    }

    Vector operator+(const Vector& other) const { return Vector(ops::add(data, other.data)); }
    Vector operator*(const Vector& other) const { return Vector(ops::mul(data, other.data)); }

    friend Vector sqrt(const Vector& v)
    {
        if constexpr (std::is_invocable_r_v<register_type, decltype(ops::sqrt), register_type>)
        {
            return Vector(ops::sqrt(v.data));
        }
        else
        {
            // static_assert(std::is_floating_point_v<T>, "sqrt works only for floating point types.");
            alignas(alignment) std::array<T, length> arr_v, arr_res;
            // TODO: on which alignments we have to still use unaligned load?
            ops::load_aligned(arr_v.data(), v.data());
            for (std::size_t i = 0; i < length; ++i)
            {
                arr_res[i] = cast(T) std::sqrt(cast(f64) arr_v[i]);
            }
            // TODO: on which alignments we have to still use unaligned load?
            return Vector(ops::load_unaligned(arr_res.data()));
        }
    }

    Vector operator==(const Vector& other) const { return Vector(ops::cmpeq(data, other.data)); }

    i32 mask() const { return ops::mask(data); }

    friend std::ostream& operator<<(std::ostream& os, const Vector& v)
    {
        alignas(alignment) std::array<T, length> temp;
        // TODO: on which alignments we have to still use unaligned load?
        ops::load_aligned(temp.data(), v.data());
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


#include "slice.h"

inline i64 firstIndexOfVectorized(Slice<u8> haystack, u8 needle)
{
    constexpr auto v_length = Vector<u8>::length;
    auto           v_needle = Vector<u8>::splat(needle);
    std::size_t    pos      = 0;
    while (pos + v_length <= haystack.len())
    {
        auto v_haystack = Vector<u8>::load_unaligned(haystack.slice_from(pos).data());
        auto matches    = v_haystack == v_needle;
        auto mask       = matches.mask();
        if (mask != 0)
        {
            auto match_offset_in_v = bit_scan(mask);
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