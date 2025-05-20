#ifndef STRING_H
#define STRING_H

#include "slice.h"

struct String
{
    Slice<u8> m_data;
};

#endif