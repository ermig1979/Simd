/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdUnpack.h"
#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLoad.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        static void MinMax32f(const float* src, size_t size, float& min, float& max)
        {
            assert(size % 8 == 0);
            __m256 _min = _mm256_set1_ps(FLT_MAX);
            __m256 _max = _mm256_set1_ps(-FLT_MAX);
            size_t i = 0;
            for (; i < size; i += 8)
            {
                __m256 _src = _mm256_loadu_ps(src + i);
                _min = _mm256_min_ps(_src, _min);
                _max = _mm256_max_ps(_src, _max);
            }
            MinVal32f(_min, min);
            MaxVal32f(_max, max);
        }

        //-------------------------------------------------------------------------------------------------

        static void MinMax16f(const uint16_t* src, size_t size, float& min, float& max)
        {
            assert(size % 8 == 0);
            __m256 _min = _mm256_set1_ps(FLT_MAX);
            __m256 _max = _mm256_set1_ps(-FLT_MAX);
            size_t i = 0;
            for (; i < size; i += 8)
            {
                __m256 _src = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src + i)));
                _min = _mm256_min_ps(_src, _min);
                _max = _mm256_max_ps(_src, _max);
            }
            MinVal32f(_min, min);
            MaxVal32f(_max, max);
        }

        //-------------------------------------------------------------------------------------------------

        static void UnpackNormA(size_t count, const uint8_t* const* src, float* dst, size_t stride)
        {
            size_t count2 = AlignLo(count, 2), i = 0;
            for (; i < count2; i += 2, src += 2, dst += 8)
                _mm256_storeu_ps(dst, Load<false>((float*)src[0], (float*)src[1]));
            for (; i < count; ++i, src += 1, dst += 4)
                _mm_storeu_ps(dst, _mm_loadu_ps((float*)src[0]));
        }

        //-------------------------------------------------------------------------------------------------

        static void UnpackNormB(size_t count, const uint8_t* const* src, float* dst, size_t stride)
        {
            size_t count8 = AlignLo(count, 8), count4 = AlignLo(count, 4), i = 0;
            for (; i < count8; i += 8, src += 8, dst += 8)
            {
                __m256 s0 = Load<false>((float*)src[0], (float*)src[4]);
                __m256 s1 = Load<false>((float*)src[1], (float*)src[5]);
                __m256 s2 = Load<false>((float*)src[2], (float*)src[6]);
                __m256 s3 = Load<false>((float*)src[3], (float*)src[7]);
                __m256 s00 = _mm256_unpacklo_ps(s0, s2);
                __m256 s01 = _mm256_unpacklo_ps(s1, s3);
                __m256 s10 = _mm256_unpackhi_ps(s0, s2);
                __m256 s11 = _mm256_unpackhi_ps(s1, s3);
                _mm256_storeu_ps(dst + 0 * stride, _mm256_unpacklo_ps(s00, s01));
                _mm256_storeu_ps(dst + 1 * stride, _mm256_unpackhi_ps(s00, s01));
                _mm256_storeu_ps(dst + 2 * stride, _mm256_unpacklo_ps(s10, s11));
                _mm256_storeu_ps(dst + 3 * stride, _mm256_unpackhi_ps(s10, s11));
            }
            for (; i < count4; i += 4, src += 4, dst += 4)
            {
                __m128 s0 = _mm_loadu_ps((float*)src[0]);
                __m128 s1 = _mm_loadu_ps((float*)src[1]);
                __m128 s2 = _mm_loadu_ps((float*)src[2]);
                __m128 s3 = _mm_loadu_ps((float*)src[3]);
                __m128 s00 = _mm_unpacklo_ps(s0, s2);
                __m128 s01 = _mm_unpacklo_ps(s1, s3);
                __m128 s10 = _mm_unpackhi_ps(s0, s2);
                __m128 s11 = _mm_unpackhi_ps(s1, s3);
                _mm_storeu_ps(dst + 0 * stride, _mm_unpacklo_ps(s00, s01));
                _mm_storeu_ps(dst + 1 * stride, _mm_unpackhi_ps(s00, s01));
                _mm_storeu_ps(dst + 2 * stride, _mm_unpacklo_ps(s10, s11));
                _mm_storeu_ps(dst + 3 * stride, _mm_unpackhi_ps(s10, s11));
            }
            for (; i < count; i++, src++, dst++)
            {
                dst[0 * stride] = ((float*)src[0])[0];
                dst[1 * stride] = ((float*)src[0])[1];
                dst[2 * stride] = ((float*)src[0])[2];
                dst[3 * stride] = ((float*)src[0])[3];
            }
        }

        //-------------------------------------------------------------------------------------------------

        DescrInt::DescrInt(size_t size, size_t depth)
            : Sse41::DescrInt(size, depth)
        {
            _minMax32f = MinMax32f;
            _minMax16f = MinMax16f;
            _encode32f = GetEncode32f(_depth);
            _encode16f = GetEncode16f(_depth);

            _decode32f = GetDecode32f(_depth);
            _decode16f = GetDecode16f(_depth);

            _cosineDistance = GetCosineDistance(_depth);
            _macroCosineDistancesDirect = GetMacroCosineDistancesDirect(_depth);

            _unpackNormA = UnpackNormA;
            _unpackNormB = UnpackNormB;
            if (_depth != 8)
            {
                _unpackDataA = GetUnpackData(_depth, false);
                _unpackDataB = GetUnpackData(_depth, true);
                _macroCosineDistancesUnpack = GetMacroCosineDistancesUnpack(_depth);
                _microMu = 5;
                _microNu = 16;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void* DescrIntInit(size_t size, size_t depth)
        {
            if (!Base::DescrInt::Valid(size, depth))
                return NULL;
            return new Avx2::DescrInt(size, depth);
        }
    }
#endif
}
