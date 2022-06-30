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

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        void SvmSumLinear(const float * x, const float * svs, const float * weights, size_t length, size_t count, float * sum)
        {
            Array32f sums(count, true);
            size_t alignedCount = AlignLo(count, F);
            __mmask16 tailMask = TailMask16(count - alignedCount);

            for (size_t j = 0; j < length; ++j)
            {
                size_t i = 0;
                float v = x[j];
                __m512 _v = _mm512_set1_ps(v);
                for (; i < alignedCount; i += F)
                    _mm512_storeu_ps(sums.data + i, _mm512_fmadd_ps(_v, _mm512_loadu_ps(svs + i), _mm512_loadu_ps(sums.data + i)));
                if (i < count)
                {
                    __m512 _svs = _mm512_maskz_loadu_ps(tailMask, svs + i);
                    __m512 _sums = _mm512_maskz_loadu_ps(tailMask, sums.data + i);
                    _mm512_mask_storeu_ps(sums.data + i, tailMask, _mm512_fmadd_ps(_v, _svs, _sums));
                }
                svs += count;
            }

            size_t i = 0;
            __m512 _sum = _mm512_setzero_ps();
            for (; i < alignedCount; i += F)
                _sum = _mm512_fmadd_ps(Load<true>(sums.data + i), Load<false>(weights + i), _sum);
            if (i < count)
            {
                __m512 _sums = _mm512_maskz_loadu_ps(tailMask, sums.data + i);
                __m512 _weight = _mm512_maskz_loadu_ps(tailMask, weights + i);
                _sum = _mm512_fmadd_ps(_sums, _weight, _sum);
            }
            *sum = ExtractSum(_sum);
        }
    }
#endif
}
