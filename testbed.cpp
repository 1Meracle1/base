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
#include "include/simd/vector.h"
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

    std::cout << simd::supported_features() << '\n';

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

        auto first_index_of_seq = []<typename T>(const T* ptr, u64 length, T needle) -> i64
        {
            if (length == 0)
                return -1;
            for (u64 i = 0; i < length; ++i)
            {
                if (ptr[i] == needle)
                    return i;
            }
            return -1;
        };

        MeasureTimeStats2 stats_seq_search;
        MeasureTimeStats2 stats_vectorized_lambda_arguments;
        for (std::size_t iteration = 0; iteration < 10000; ++iteration)
        {
            for (i32 ch = 33; ch < 127; ++ch)
            {
                u8 c = cast(u8) ch;

                auto maybe_index = first_index_of_seq(data.data(), data.len(), c);
                stats_seq_search.start();
                maybe_index = first_index_of_seq(data.data(), data.len(), c);
                stats_seq_search.end();

                auto maybe_index1 = simd::first_index_of(data.data(), data.len(), c);
                stats_vectorized_lambda_arguments.start();
                maybe_index1 = simd::first_index_of(data.data(), data.len(), c);
                stats_vectorized_lambda_arguments.end();

                Assert(maybe_index == maybe_index1);
            }
        }
        stats_vectorized_lambda_arguments.print_summary_with_reference_ms(
            ByteSliceFromCstr("vectorized search with lambda and arguments"), stats_seq_search);
    }
}