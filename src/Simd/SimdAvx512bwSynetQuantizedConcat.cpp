/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdSynetQuantizeLinear.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        void SynetQuantizedConcatLayerForward(size_t count, const uint8_t** src, size_t num, const size_t* size, const int32_t* bias, const float* norm, const float* scale, int32_t zero, uint8_t* dst)
        {
            __m512i _bias, _zero = _mm512_set1_epi32(zero);
            __m512 _norm, _scale = _mm512_set1_ps(scale[0]);
            for (size_t o = 0; o < num; ++o)
            {
                for (size_t s = 0; s < count; ++s)
                {
                    size_t size1 = size[s], size16 = size1 & (~15), size64 = size1 & (~63), i = 0;
                    __mmask16 tail = __mmask16(-1) >> (size16 + 16 - size1);
                    const uint8_t* ps = src[s] + o * size1;
                    _bias = _mm512_set1_epi32(bias[s]);
                    _norm = _mm512_set1_ps(norm[s]);
                    for (; i < size64; i += 64)
                        DequantizeQuantizeLinear64(ps + i, _bias, _norm, _scale, _zero, dst + i);
                    for (; i < size16; i += 16)
                        DequantizeQuantizeLinear16(ps + i, _bias, _norm, _scale, _zero, dst + i);
                    if (tail)
                        DequantizeQuantizeLinear16(ps + i, _bias, _norm, _scale, _zero, dst + i, tail);
                    dst += size1;
                }
            }
        }
    }
#endif
}
