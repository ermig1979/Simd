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
#include "Simd/SimdExtract.h"
#include "Simd/SimdArray.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        void SvmSumLinear(const float * x, const float * svs, const float * weights, size_t length, size_t count, float * sum)
        {
            Array32f buffer(count, true);
            size_t alignedCount = AlignLo(count, F);

            for (size_t j = 0; j < length; ++j)
            {
                size_t i = 0;
                float v = x[j];
                __m128 _v = _mm_set1_ps(v);
                for (; i < alignedCount; i += F)
                {
                    __m128 sums = _mm_loadu_ps(buffer.data + i);
                    __m128 _svs = _mm_loadu_ps(svs + i);
                    _mm_storeu_ps(buffer.data + i, _mm_add_ps(sums, _mm_mul_ps(_v, _svs)));
                }
                for (; i < count; ++i)
                    buffer.data[i] += v*svs[i];
                svs += count;
            }

            size_t i = 0;
            __m128 _sum = _mm_setzero_ps();
            for (; i < alignedCount; i += F)
            {
                __m128 sums = _mm_loadu_ps(buffer.data + i);
                __m128 _weights = _mm_loadu_ps(weights + i);
                _sum = _mm_add_ps(_sum, _mm_mul_ps(sums, _weights));
            }
            *sum = ExtractSum(_sum);
            for (; i < count; ++i)
                *sum += buffer.data[i] * weights[i];
        }
    }
#endif
}
