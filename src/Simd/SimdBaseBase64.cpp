/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#include "Simd/SimdMemory.h"
#include "Simd/SimdBase64.h"

namespace Simd
{
    namespace Base
    {
        void Base64Decode(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t* dstSize)
        {
            assert(srcSize % 4 == 0 && srcSize >= 4);
            for (const uint8_t * body = src + srcSize - 4; src < body; src += 4, dst += 3)
                Base64Decode3(src, dst);
            *dstSize = srcSize / 4 * 3 + Base64DecodeTail(src, dst) - 3;
        }

        //---------------------------------------------------------------------------------------------

        void Base64Encode(const uint8_t* src, size_t size, uint8_t* dst)
        {
            const uint8_t* body = src + AlignLoAny(size, 3);
            const uint8_t* tail = src + size;
            for (; src < body; src += 3, dst += 4)
                Base64Encode3(src, dst);
            if(body < tail)
                Base64EncodeTail(src, tail - body, dst);
        }
    }
}

