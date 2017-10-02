/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint8_t) * 4 * width);
                    bgra = (uint8_t*)_p;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint8_t * bgra;
            private:
                void *_p;
            };
        }

        void BgrToGray(const uint8_t *bgr, size_t width, size_t height, size_t bgrStride, uint8_t *gray, size_t grayStride)
        {
            assert(width >= A);

            Buffer buffer(width);

            for (size_t row = 1; row < height; ++row)
            {
                Base::BgrToBgra(bgr, width, buffer.bgra, false, false, 0xFF);
                Sse2::BgraToGray(buffer.bgra, width, 1, 4 * width, gray, width);
                bgr += bgrStride;
                gray += grayStride;
            }
            Base::BgrToBgra(bgr, width, buffer.bgra, false, true, 0xFF);
            Sse2::BgraToGray(buffer.bgra, width, 1, 4 * width, gray, width);
        }
    }
#endif//SIMD_SSE2_ENABLE
}
