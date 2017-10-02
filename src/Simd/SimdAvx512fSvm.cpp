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
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"

namespace Simd
{
#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t count)
                {
                    size_t size = sizeof(float)*count;
                    _p = Allocate(size);
                    memset(_p, 0, size);
                    sums = (float*)_p;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                float * sums;
            private:
                void *_p;
            };
        }

        void SvmSumLinear(const float * x, const float * svs, const float * weights, size_t length, size_t count, float * sum)
        {
            Buffer buffer(count);
            size_t alignedCount = AlignLo(count, F);
            __mmask16 tailMask = TailMask16(count - alignedCount);

            for (size_t j = 0; j < length; ++j)
            {
                size_t i = 0;
                float v = x[j];
                __m512 _v = _mm512_set1_ps(v);
                for (; i < alignedCount; i += F)
                    Store<true>(buffer.sums + i, _mm512_fmadd_ps(_v, Load<false>(svs + i), Load<true>(buffer.sums + i)));
                if (i < count)
                    Store<true, true>(buffer.sums + i, _mm512_fmadd_ps(_v, (Load<false, true>(svs + i, tailMask)), (Load<true, true>(buffer.sums + i, tailMask))), tailMask);
                svs += count;
            }

            size_t i = 0;
            __m512 _sum = _mm512_setzero_ps();
            for (; i < alignedCount; i += F)
                _sum = _mm512_fmadd_ps(Load<true>(buffer.sums + i), Load<false>(weights + i), _sum);
            if (i < count)
                _sum = _mm512_fmadd_ps((Load<true, true>(buffer.sums + i, tailMask)), (Load<false, true>(weights + i, tailMask)), _sum);
            *sum = ExtractSum(_sum);
        }
    }
#endif// SIMD_AVX512F_ENABLE
}
