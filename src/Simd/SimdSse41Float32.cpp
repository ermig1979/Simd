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
#include "Simd/SimdUnpack.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        void CosineDistance32f(const float* a, const float* b, size_t size, float* distance)
        {
            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, DF);
            size_t i = 0;
            __m128 _aa[2] = { _mm_setzero_ps(), _mm_setzero_ps() };
            __m128 _ab[2] = { _mm_setzero_ps(), _mm_setzero_ps() };
            __m128 _bb[2] = { _mm_setzero_ps(), _mm_setzero_ps() };
            if (fullAlignedSize)
            {
                for (; i < fullAlignedSize; i += DF)
                {
                    __m128 a0 = _mm_loadu_ps(a + i + 0 * F);
                    __m128 b0 = _mm_loadu_ps(b + i + 0 * F);
                    _aa[0] = _mm_add_ps(_aa[0], _mm_mul_ps(a0, a0));
                    _ab[0] = _mm_add_ps(_ab[0], _mm_mul_ps(a0, b0));
                    _bb[0] = _mm_add_ps(_bb[0], _mm_mul_ps(b0, b0));
                    __m128 a1 = _mm_loadu_ps(a + i + 1 * F);
                    __m128 b1 = _mm_loadu_ps(b + i + 1 * F);
                    _aa[1] = _mm_add_ps(_aa[1], _mm_mul_ps(a1, a1));
                    _ab[1] = _mm_add_ps(_ab[1], _mm_mul_ps(a1, b1));
                    _bb[1] = _mm_add_ps(_bb[1], _mm_mul_ps(b1, b1));
                }
                _aa[0] = _mm_add_ps(_aa[0], _aa[1]);
                _ab[0] = _mm_add_ps(_ab[0], _ab[1]);
                _bb[0] = _mm_add_ps(_bb[0], _bb[1]);
            }
            for (; i < partialAlignedSize; i += F)
            {
                __m128 a0 = _mm_loadu_ps(a + i);
                __m128 b0 = _mm_loadu_ps(b + i);
                _aa[0] = _mm_add_ps(_aa[0], _mm_mul_ps(a0, a0));
                _ab[0] = _mm_add_ps(_ab[0], _mm_mul_ps(a0, b0));
                _bb[0] = _mm_add_ps(_bb[0], _mm_mul_ps(b0, b0));
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
            *distance = 1.0f - ab / ::sqrt(aa * bb);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m128i Float32ToUint8(const float * src, const __m128 & lower, const __m128 & upper, const __m128 & boost)
        {
            return _mm_cvtps_epi32(_mm_mul_ps(_mm_sub_ps(_mm_min_ps(_mm_max_ps(_mm_loadu_ps(src), lower), upper), lower), boost));
        }

        SIMD_INLINE void Float32ToUint8(const float * src, const __m128 & lower, const __m128 & upper, const __m128 & boost, uint8_t * dst)
        {
            __m128i d0 = Float32ToUint8(src + F * 0, lower, upper, boost);
            __m128i d1 = Float32ToUint8(src + F * 1, lower, upper, boost);
            __m128i d2 = Float32ToUint8(src + F * 2, lower, upper, boost);
            __m128i d3 = Float32ToUint8(src + F * 3, lower, upper, boost);
            _mm_storeu_si128((__m128i*)dst, _mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)));
        }

        void Float32ToUint8(const float * src, size_t size, const float * lower, const float * upper, uint8_t * dst)
        {
            assert(size >= A);

            __m128 _lower = _mm_set1_ps(lower[0]);
            __m128 _upper = _mm_set1_ps(upper[0]);
            __m128 boost = _mm_set1_ps(255.0f / (upper[0] - lower[0]));

            size_t alignedSize = AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                Float32ToUint8(src + i, _lower, _upper, boost, dst + i);
            if (alignedSize != size)
                Float32ToUint8(src + size - A, _lower, _upper, boost, dst + size - A);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m128 Uint8ToFloat32(const __m128i & value, const __m128 & lower, const __m128 & boost)
        {
            return _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(value), boost), lower);
        }

        SIMD_INLINE void Uint8ToFloat32(const uint8_t * src, const __m128 & lower, const __m128 & boost, float * dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128i lo = UnpackU8<0>(_src);
            __m128i hi = UnpackU8<1>(_src);
            _mm_storeu_ps(dst + F * 0, Uint8ToFloat32(UnpackU16<0>(lo), lower, boost));
            _mm_storeu_ps(dst + F * 1, Uint8ToFloat32(UnpackU16<1>(lo), lower, boost));
            _mm_storeu_ps(dst + F * 2, Uint8ToFloat32(UnpackU16<0>(hi), lower, boost));
            _mm_storeu_ps(dst + F * 3, Uint8ToFloat32(UnpackU16<1>(hi), lower, boost));
        }

        void Uint8ToFloat32(const uint8_t * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            assert(size >= A);

            __m128 _lower = _mm_set1_ps(lower[0]);
            __m128 boost = _mm_set1_ps((upper[0] - lower[0]) / 255.0f);

            size_t alignedSize = AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                Uint8ToFloat32(src + i, _lower, boost, dst + i);
            if (alignedSize != size)
                Uint8ToFloat32(src + size - A, _lower, boost, dst + size - A);
        }
    }
#endif
}
