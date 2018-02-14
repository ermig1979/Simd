/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        template<bool align> void CosineDistance32f(const float * a, const float * b, size_t size, float * distance)
        {
            if (align)
                assert(Aligned(a) && Aligned(b));

            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, DF);
            size_t i = 0;
            __m256 _aa[2] = { _mm256_setzero_ps(), _mm256_setzero_ps() };
            __m256 _ab[2] = { _mm256_setzero_ps(), _mm256_setzero_ps() };
            __m256 _bb[2] = { _mm256_setzero_ps(), _mm256_setzero_ps() };
            if (fullAlignedSize)
            {
                for (; i < fullAlignedSize; i += DF)
                {
                    __m256 a0 = Load<align>(a + i + 0 * F);
                    __m256 b0 = Load<align>(b + i + 0 * F);
                    _aa[0] = _mm256_add_ps(_aa[0], _mm256_mul_ps(a0, a0));
                    _ab[0] = _mm256_add_ps(_ab[0], _mm256_mul_ps(a0, b0));
                    _bb[0] = _mm256_add_ps(_bb[0], _mm256_mul_ps(b0, b0));
                    __m256 a1 = Load<align>(a + i + 1 * F);
                    __m256 b1 = Load<align>(b + i + 1 * F);
                    _aa[1] = _mm256_add_ps(_aa[1], _mm256_mul_ps(a1, a1));
                    _ab[1] = _mm256_add_ps(_ab[1], _mm256_mul_ps(a1, b1));
                    _bb[1] = _mm256_add_ps(_bb[1], _mm256_mul_ps(b1, b1));
                }
                _aa[0] = _mm256_add_ps(_aa[0], _aa[1]);
                _ab[0] = _mm256_add_ps(_ab[0], _ab[1]);
                _bb[0] = _mm256_add_ps(_bb[0], _bb[1]);
            }
            for (; i < partialAlignedSize; i += F)
            {
                __m256 a0 = Load<align>(a + i);
                __m256 b0 = Load<align>(b + i);
                _aa[0] = _mm256_add_ps(_aa[0], _mm256_mul_ps(a0, a0));
                _ab[0] = _mm256_add_ps(_ab[0], _mm256_mul_ps(a0, b0));
                _bb[0] = _mm256_add_ps(_bb[0], _mm256_mul_ps(b0, b0));
            }
            float aa = ExtractSum(_aa[0]), ab = ExtractSum(_ab[0]), bb = ExtractSum(_bb[0]);
            for (; i < size; ++i)
            {
                float _a = a[i];
                float _b = b[i];
                aa += _a * _a;
                ab += _a * _b;
                bb += _b * _b;
            }
            *distance = 1.0f - ab / ::sqrt(aa*bb);
        }

        void CosineDistance32f(const float * a, const float * b, size_t size, float * distance)
        {
            if (Aligned(a) && Aligned(b))
                CosineDistance32f<true>(a, b, size, distance);
            else
                CosineDistance32f<false>(a, b, size, distance);
        }
    }
#endif// SIMD_AVX_ENABLE
}
