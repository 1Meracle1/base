#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include "memory.h"
#include "slice.h"
#include "string.h"
#include "defer.h"
#include <fstream>

namespace fs {

[[maybe_unused]] [[nodiscard]] inline bool write_entire_file(Slice<u8> file_name, Slice<u8> data, bool truncate)
{
    auto open_mode = std::ios::out;
    if (truncate)
    {
        open_mode |= std::ios::trunc;
    }
    std::ofstream output_file{file_name.reinterpret_elements_as<const char>().data(), open_mode};
    if (!output_file.is_open())
    {
        return false;
    }
    output_file << data;
    return !output_file.fail();
}

[[maybe_unused]] [[nodiscard]] inline bool
read_entire_file(Allocator* allocator, Slice<u8> file_name, Slice<u8>& res_data)
{
    std::ifstream input_file{file_name.reinterpret_elements_as<const char>().data(), std::ios::binary | std::ios::ate};
    if (!input_file.is_open())
    {
        return false;
    }
    defer([&input_file] { input_file.close(); });
    auto file_size = input_file.tellg();
    if (file_size < 0)
    {
        return false;
    }
    if (file_size > 0)
    {
        input_file.seekg(0, std::ios::beg);
        if (!input_file.good())
        {
            return false;
        }
        res_data.m_ptr = allocator->alloc<u8>(file_size);
        res_data.m_len = file_size;
        if (!input_file.read(cast(char*) res_data.raw(), file_size))
        {
            return input_file.eof() && !input_file.fail();
        }
    }
    return true;
}

}

#endif