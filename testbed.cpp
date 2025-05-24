#include "include/array.h"
#include "include/defines.h"
#include "include/memory.h"
#include "include/defer.h"
#include "include/assert.h"
#include "include/slice.h"
#include "include/string.h"
#include "include/parse.h"
#include "include/time.h"
#include <cstdio>
#include <iostream>

int main()
{
    MeasureTimeMicro("entire program duration");

    auto start = time_now();

    VirtualArena arena{};
    Allocator*   allocator = &arena;

    auto diff = time_diff_micro(start, time_now());
    std::cout << "arena allocation took " << diff << " micros\n";

    // {
    //     String str    = String::from_raw(allocator, "-12345");
    //     i64    number = 0;
    //     Assert(parse_int(str, number) == ParseIntFromStringError::None);
    //     Assert(number == -12345);
    // }

    // {
    //     String str    = String::from_raw(allocator, "-12345");
    //     u64    number = 0;
    //     Assert(parse_int(str, number) == ParseIntFromStringError::UnexpectedNegativeSign);
    // }

    // {
    //     String str    = String::from_raw(allocator, "12345");
    //     u64    number = 0;
    //     Assert(parse_int(str, number) == ParseIntFromStringError::None);
    //     Assert(number == 12345);
    // }

    // {
    //     String str    = String::from_raw(allocator, "12345");
    //     u8    number = 0;
    //     Assert(parse_int(str, number) == ParseIntFromStringError::Overflow);
    // }
    {
        MeasureTimeMicro("parsing of negative float with fractional part");
        Slice<const char> cstr   = "-1.234";
        Slice<u8>         str    = cstr.chop_zero_termination().reinterpret_elements_as<u8>();
        f64               number = 0;
        Assert(parse_float(str, number) == ParseFloatFromStringError::None);
    }
    {
        MeasureTimeMicro("parsing of positive float with fractional part");
        Slice<const char> cstr   = "1,234";
        Slice<u8>         str    = cstr.chop_zero_termination().reinterpret_elements_as<u8>();
        f64               number = 0;
        Assert(parse_float(str, number, ',') == ParseFloatFromStringError::None);
    }
    {
        MeasureTimeMicro("parsing of positive float without fractional part");
        Slice<const char> cstr   = "1";
        Slice<u8>         str    = cstr.chop_zero_termination().reinterpret_elements_as<u8>();
        f64               number = 0;
        Assert(parse_float(str, number, ',') == ParseFloatFromStringError::None);
        Assert(number == cast(f64)1.0);
    }
    {
        MeasureTimeMicro("parsing of negative float without fractional part");
        Slice<const char> cstr   = "-1";
        Slice<u8>         str    = cstr.chop_zero_termination().reinterpret_elements_as<u8>();
        f64               number = 0;
        Assert(parse_float(str, number, ',') == ParseFloatFromStringError::None);
        Assert(number == cast(f64)(-1.0));
    }
    // String str = String::from_utf8_lossy(
    //     allocator, "In the quiet twilight, dreams unfold, soft whispers of a story untold.\n"
    //                "ćeść panśtwu\n"
    //                "月明かりが静かに照らし出し、夢を見る心の奥で詩が静かに囁かれる\n"
    //                "Stars collide in the early light of hope, echoing the silent call of the night.\n"
    //                "夜の静寂、希望と孤独が混ざり合うその中で詩が永遠に続く\n");
    // for (u32 codepoint : str)
    // {
    //     std::cout << std::hex << codepoint << (codepoint == cast(u32)0xA ? '\n' : ' ');
    // }
    // std::cout << '\n' << str << '\n';
}