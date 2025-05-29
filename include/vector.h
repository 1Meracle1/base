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
// clang-format on

template <typename T, typename Tag> struct RegisterTrait;

#if defined(__SSE2__)
template <> struct RegisterTrait<f32, SSE2_Tag>
{
    using element_type                     = f32;
    using register_type                    = __m128;
    static constexpr std::size_t length    = 4;
    static constexpr std::size_t alignment = RegisterAlignment<register_type>::value;
    // unary
    static register_type set1(f32 v) { return _mm_set1_ps(v); }
    static register_type setzero() { return _mm_setzero_ps(); }
    static register_type load_unaligned(const f32* p) { return _mm_loadu_ps(p); }
    static void          store_unaligned(f32* p, register_type v) { _mm_storeu_ps(p, v); }
    static register_type load_aligned(const f32* p) { return _mm_load_ps(p); }
    static void          store_aligned(f32* p, register_type v) { _mm_store_ps(p, v); }
    // binary
    static register_type add(register_type a, register_type b) { return _mm_add_ps(a, b); }
    static register_type mul(register_type a, register_type b) { return _mm_mul_ps(a, b); }
    static register_type sqrt(register_type a) { return _mm_sqrt_ps(a); }
    static register_type cmpeq(register_type a, register_type b) { return _mm_cmpeq_ps(a, b); }
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
};
#endif

#ifdef __AVX__
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
};
#ifdef __AVX2__
// clang-format off
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
    Vector(std::initializer_list<T> init_list) noexcept
        : data(ops::setzero())
    {
        Assert(init_list.size() <= length);
        alignas(alignment) std::array<T, length> temp{};
        for (std::size_t i = 0; i < init_list.size(); ++i)
        {
            if (i < length)
                temp[i] = init_list[i];
        }
        data = ops::load_unaligned(temp.data());
    }
    template <std::size_t N>
    Vector(T (&arr)[N]) noexcept
        : data(ops::load_unaligned(arr))
    {
        static_assert(N == length);
    }

    static Vector load_unaligned(const T* ptr) { return Vector(ops::load_unaligned(ptr)); }
    void          store_unaligned(T* ptr) const { ops::store_unaligned(ptr, data); }
    static Vector load_aligned(const T* ptr) { return Vector(ops::load_aligned(ptr)); }
    void          store_aligned(T* ptr) const { ops::store_aligned(ptr, data); }

    T operator[](std::size_t i) const
    {
        Assert(i < length);
        alignas(alignment) std::array<T, length> temp{};
        // TODO: on which alignments we have to still use unaligned store?
        ops::store_aligned(temp.data(), data);
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
            return Vector(ops::load_aligned(arr_res.data()));
        }
    }

    Vector operator==(const Vector& other) const { return Vector(ops::cmpeq(data, other.data)); }

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

#endif