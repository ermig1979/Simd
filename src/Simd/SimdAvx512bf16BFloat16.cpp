/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#ifdef SIMD_AVX512BF16_ENABLE    
    namespace Avx512bf16
    {
        void Float32ToBFloat16(const float* src, size_t size, uint16_t* dst)
        {
            size_t size32 = AlignLo(size, 32);
            __mmask16 srcMask[2];
            __mmask32 dstMask[1];
            size_t i = 0;
            for (; i < size32; i += 32)
                Float32ToBFloat16<false, false>(src + i, dst + i, srcMask, dstMask);
            if (size32 < size)
            {
                srcMask[0] = TailMask16(size - size32 - F * 0);
                srcMask[1] = TailMask16(size - size32 - F * 1);
                dstMask[0] = TailMask32(size - size32);
                Float32ToBFloat16<false, true>(src + i, dst + i, srcMask, dstMask);
            }
        }
    }
#endif
}
