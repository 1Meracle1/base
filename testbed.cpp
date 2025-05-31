#include "include/array.h"
#include "include/defines.h"
#include "include/memory.h"
#include "include/defer.h"
#include "include/assert.h"
#include "include/slice.h"
#include "include/string.h"
#include "include/parse_num.h"
#include "include/time.h"
#include "include/filesystem.h"
#include "include/types.h"
#include "include/vector.h"
#include <cstdio>
#include <ios>
#include <iostream>

int main()
{
    MeasureTimeMicro("entire program duration");

    auto start = time_now();

    VirtualArena arena{};
    Allocator*   allocator = &arena;

    auto diff = time_diff_micro(start, time_now());
    std::cout << "arena allocation took " << diff << " micros\n";

    // Array<int> arr{allocator, 10};
    // for (u64 i = 0; i < 10; ++i)
    // {
    //     arr.append(i);
    // }
    // {

    // {
    //     MeasureTimeMicro("string split");
    //     auto str = String::from_raw(allocator, "hellope someone out there");
    //     auto parts = str.split_owning(allocator, ' ');
    //     for(auto it = parts.begin(), itEnd = parts.end(); it != itEnd; ++it)
    //     {
    //         std::cout << *it << '\n';
    //     }
    // }

    // {
    //     MeasureTimeMicro("string slice split");
    //     Slice<const char> str{"hellope\nsomeone\r\nout\nthere\r\n"};
    //     auto parts = str.split_lines(allocator);
    //     for(auto it = parts.begin(), itEnd = parts.end(); it != itEnd; ++it)
    //     {
    //         std::cout << *it << '\n';
    //     }
    // }
    {
        // MeasureTimeMicro("read from file");
        Slice<u8> data{};
        bool      ok = fs::read_entire_file(allocator, ByteSliceFromCstr("include/string.h"), data);
        std::cout << "read successfully? - " << std::boolalpha << ok << ", bytes: " << data.len() << '\n';
        // {
        //     auto      start  = time_now();
        //     Slice<u8> needle = ByteSliceFromCstr("operator<<(std::ostream& os, const String& str)");
        //     i64       idx    = data.linear_search(needle);
        //     auto      diff   = time_diff_micro(start, time_now());
        //     std::cout << "index " << idx << " of substring '" << needle << "' took " << diff << " micros\n";

        MeasureTimeStats2 stats_vectorized_search;
        MeasureTimeStats2 stats_seq_search;
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
        // for (std::size_t iteration = 0; iteration < 10000; ++iteration)
        // {
        //     stats_vectorized_search.start();
        //     for (i32 ch = 33; ch < 127; ++ch)
        //     {
        //         u8   c               = cast(u8) ch;
        //         auto maybe_index_vec = firstIndexOfVectorized(data, c);
        //     }
        //     stats_vectorized_search.end();
        // }
        // MeasureTimeStats2 stats_seq_search;
        // for (std::size_t iteration = 0; iteration < 10000; ++iteration)
        // {
        //     stats_seq_search.start();
        //     for (i32 ch = 33; ch < 127; ++ch)
        //     {
        //         u8   c           = cast(u8) ch;
        //         auto maybe_index = firstIndexOf(data, c);
        //     }
        //     stats_seq_search.end();
        // }
        stats_vectorized_search.print_summary_with_reference_ms(
            ByteSliceFromCstr("vectorized search of single character vs sequential search"), stats_seq_search);

        // MeasureTimeStats stats_utf8_lossy{allocator};
        // for (u64 i = 0; i < 10000; ++i)
        // {
        //     auto start = time_now();
        //     auto str   = String::from_utf8_lossy(allocator, data);
        //     auto diff  = time_diff_nano(start, time_now());
        //     stats_utf8_lossy.append(diff);
        // }

        // MeasureTimeStats stats_raw{allocator};
        // for (u64 i = 0; i < 10000; ++i)
        // {
        //     auto start = time_now();
        //     auto str   = String::from_raw(allocator, data);
        //     auto diff  = time_diff_nano(start, time_now());
        //     stats_raw.append(diff);
        // }

        // stats_raw.print_summary_with_reference(
        //     ByteSliceFromCstrZeroTerm("Construction of string from raw bytes vs utf8 lossy: "),
        //     stats_utf8_lossy
        // );

        // {
        //     MeasureTimeMicro("utf8 string creation");
        //     auto str = String::from_utf8_lossy(allocator, data);
        //     // std::cout << str << '\n';
        // }
        // {
        //     MeasureTimeMicro("raw string creation");
        //     auto str = String::from_raw(allocator, data);
        //     // std::cout << str << '\n';
        // }
    }
    // {
    //     MeasureTimeMicro("parsing of negative float with fractional part");
    //     Slice<const char> cstr   = "-1.234";
    //     Slice<u8>         str    = cstr.chop_zero_termination().reinterpret_elements_as<u8>();
    //     f64               number = 0;
    //     Assert(parse_float(str, number) == ParseFloatFromStringError::None);
    // }
    // {
    //     MeasureTimeMicro("parsing of positive float with fractional part");
    //     Slice<const char> cstr   = "1,234";
    //     Slice<u8>         str    = cstr.chop_zero_termination().reinterpret_elements_as<u8>();
    //     f64               number = 0;
    //     Assert(parse_float(str, number, ',') == ParseFloatFromStringError::None);
    // }
    // {
    //     MeasureTimeMicro("parsing of positive float without fractional part");
    //     Slice<const char> cstr   = "1";
    //     Slice<u8>         str    = cstr.chop_zero_termination().reinterpret_elements_as<u8>();
    //     f64               number = 0;
    //     Assert(parse_float(str, number, ',') == ParseFloatFromStringError::None);
    //     Assert(number == cast(f64)1.0);
    // }
    // {
    //     MeasureTimeMicro("parsing of negative float without fractional part");
    //     Slice<const char> cstr   = "-1";
    //     Slice<u8>         str    = cstr.chop_zero_termination().reinterpret_elements_as<u8>();
    //     f64               number = 0;
    //     Assert(parse_float(str, number, ',') == ParseFloatFromStringError::None);
    //     Assert(number == cast(f64)(-1.0));
    // }
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