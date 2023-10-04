/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdFloat16.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        static void MinMax32f(const float* src, size_t size, float& min, float& max)
        {
            assert(size % 8 == 0);
            __m128 _min = _mm_set1_ps(FLT_MAX);
            __m128 _max = _mm_set1_ps(-FLT_MAX);
            size_t i = 0;
            for (; i < size; i += 4)
            {
                __m128 _src = _mm_loadu_ps(src + i);
                _min = _mm_min_ps(_src, _min);
                _max = _mm_max_ps(_src, _max);
            }
            MinVal32f(_min, min);
            MaxVal32f(_max, max);
        }

        //-------------------------------------------------------------------------------------------------

        static void MinMax16f(const uint16_t* src, size_t size, float& min, float& max)
        {
            assert(size % 8 == 0);
            __m128 _min = _mm_set1_ps(FLT_MAX);
            __m128 _max = _mm_set1_ps(-FLT_MAX);
            size_t i = 0;
            for (; i < size; i += 4)
            {
                __m128i f16 = _mm_loadl_epi64((__m128i*)(src + i));
                __m128 _src = Float16ToFloat32(UnpackU16<0>(f16));
                _min = _mm_min_ps(_src, _min);
                _max = _mm_max_ps(_src, _max);
            }
            MinVal32f(_min, min);
            MaxVal32f(_max, max);
        }

        //-------------------------------------------------------------------------------------------------

        static void UnpackNormA(size_t count, const uint8_t* const* src, float* dst, size_t stride)
        {
            for (size_t i = 0; i < count; ++i)
                _mm_storeu_si128((__m128i*)dst + i, _mm_loadu_si128((__m128i*)src[i]));
        }

        //-------------------------------------------------------------------------------------------------

        static void UnpackNormB(size_t count, const uint8_t* const* src, float* dst, size_t stride)
        {
            size_t count4 = AlignLo(count, 4), i = 0;
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
            : Base::DescrInt(size, depth)
        {
            _minMax32f = MinMax32f;
            _minMax16f = MinMax16f;
            _encode32f = GetEncode32f(_depth);
            _encode16f = GetEncode16f(_depth);

            _decode32f = GetDecode32f(_depth);
            _decode16f = GetDecode16f(_depth);

            _cosineDistance = GetCosineDistance(_depth);
            _macroCosineDistancesDirect = GetMacroCosineDistancesDirect(_depth);
            _microMd = 2;
            _microNd = 4;

            _unpackNormA = UnpackNormA;
            _unpackNormB = UnpackNormB;
            _unpackDataA = GetUnpackData(_depth, false);
            _unpackDataB = GetUnpackData(_depth, true);
            _macroCosineDistancesUnpack = GetMacroCosineDistancesUnpack(_depth);
            _unpSize = _size * (_depth == 8 ? 2 : 1);
            _microMu = _depth == 8 ? 6 : 5;
            _microNu = 8;
        }

        //-------------------------------------------------------------------------------------------------

        void* DescrIntInit(size_t size, size_t depth)
        {
            if (!Base::DescrInt::Valid(size, depth))
                return NULL;
            return new Sse41::DescrInt(size, depth);
        }
    }
#endif
}
