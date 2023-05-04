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
#include "Simd/SimdDescrInt.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        static void MinMax(const float* src, size_t size, float& min, float& max)
        {
            size_t sizeF = AlignLo(size, F);
            __m128 _min = _mm_set1_ps(FLT_MAX);
            __m128 _max = _mm_set1_ps(-FLT_MAX);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                __m128 _src = _mm_loadu_ps(src + i);
                _min = _mm_min_ps(_src, _min);
                _max = _mm_max_ps(_src, _max);
            }
            for (; i < size; i += 1)
            {
                __m128 _src = _mm_load_ss(src + i);
                _min = _mm_min_ss(_src, _min);
                _max = _mm_max_ss(_src, _max);
            }
            _min = _mm_min_ps(_min, Shuffle32f<0x22>(_min));
            _max = _mm_max_ps(_max, Shuffle32f<0x22>(_max));
            _min = _mm_min_ss(_min, Shuffle32f<0x11>(_min));
            _max = _mm_max_ss(_max, Shuffle32f<0x11>(_max));
            _mm_store_ss(&min, _min);
            _mm_store_ss(&max, _max);
        }

        //-------------------------------------------------------------------------------------------------

        static void Encode8(const float* src, float scale, float min, size_t size, uint8_t* dst)
        {
            size_t sizeA = AlignLo(size, A), sizeF = AlignLo(size, F), i = 0;
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _min = _mm_set1_ps(min);
            for (; i < sizeA; i += A)
            {
                __m128i d0 = _mm_cvtps_epi32(_mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(src + i + F * 0), _min), _scale));
                __m128i d1 = _mm_cvtps_epi32(_mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(src + i + F * 1), _min), _scale));
                __m128i d2 = _mm_cvtps_epi32(_mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(src + i + F * 2), _min), _scale));
                __m128i d3 = _mm_cvtps_epi32(_mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(src + i + F * 3), _min), _scale));
                _mm_storeu_si128((__m128i*)(dst + i), _mm_packus_epi16(_mm_packus_epi32(d0, d1), _mm_packus_epi32(d2, d3)));
            }
            for (; i < sizeF; i += F)
            {
                __m128 _src = _mm_loadu_ps(src + i);
                __m128i _dst = _mm_cvtps_epi32(_mm_mul_ps(_mm_sub_ps(_src, _min), _scale));
                _mm_storeu_si32((uint32_t*)(dst + i), _mm_packus_epi16(_mm_packus_epi32(_dst, _mm_setzero_si128()), _mm_setzero_si128()));
            }
            for (; i < size; i += 1)
                dst[i] = _mm_cvtss_si32(_mm_mul_ss(_mm_sub_ss(_mm_load_ss(src + i), _min), _scale));
        }

        //-------------------------------------------------------------------------------------------------

        DescrInt::DescrInt(size_t size, size_t depth)
            : Base::DescrInt(size, depth)
        {
            _minMax = MinMax;
            switch (depth)
            {
            case 6:
            {
            //    _encode = Encode6;
            //    _decode = Decode6;
            //    _cosineDistance = CosineDistance6;
            //    _vectorNorm = VectorNorm6;
                break;
            }
            case 7:
            {
            //    _encode = Encode7;
            //    _decode = Decode7;
            //    _cosineDistance = CosineDistance7;
            //    _vectorNorm = VectorNorm7;
                break;
            }
            case 8:
            {
                _encode = Encode8;
            //    _decode = Decode8;
            //    _cosineDistance = CosineDistance8;
            //    _vectorNorm = VectorNorm8;
                break;
            }
            default:
                assert(0);
            }
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
