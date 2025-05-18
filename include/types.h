#ifndef TYPES_H
#define TYPES_H

#include <cstdint>

using i8    = std::int8_t;
using u8    = std::uint8_t;
using i32   = std::int32_t;
using i64   = std::int64_t;
using u32   = std::uint32_t;
using u64   = std::uint64_t;
using f32   = float;
using f64   = double;
using isize = i64;
using usize = u64;
using byte  = u8;

using rune = i32;
#define RUNE_INVALID cast(rune)(0xfffd)
#define RUNE_MAX     cast(rune)(0x0010ffff)
#define RUNE_BOM     cast(rune)(0xfeff)
#define RUNE_EOF     cast(rune)(-1)

using cstring = const char*;
using rawptr  = void*;

#define cast(Type) (Type)

#endif