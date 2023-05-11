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
#include "Simd/SimdUnpack.h"
#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        static void MinMax(const float* src, size_t size, float& min, float& max)
        {
            size_t sizeF = AlignLo(size, F), sizeHF = AlignLo(size, HF);
            __m256 _min256 = _mm256_set1_ps(FLT_MAX);
            __m256 _max256 = _mm256_set1_ps(-FLT_MAX);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                __m256 _src = _mm256_loadu_ps(src + i);
                _min256 = _mm256_min_ps(_src, _min256);
                _max256 = _mm256_max_ps(_src, _max256);
            }
            __m128 _min = _mm_min_ps(_mm256_castps256_ps128(_min256), _mm256_extractf128_ps(_min256, 1));
            __m128 _max = _mm_max_ps(_mm256_castps256_ps128(_max256), _mm256_extractf128_ps(_max256, 1));
            for (; i < sizeHF; i += HF)
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
            _min = _mm_min_ps(_min, Sse41::Shuffle32f<0x22>(_min));
            _max = _mm_max_ps(_max, Sse41::Shuffle32f<0x22>(_max));
            _min = _mm_min_ss(_min, Sse41::Shuffle32f<0x11>(_min));
            _max = _mm_max_ss(_max, Sse41::Shuffle32f<0x11>(_max));
            _mm_store_ss(&min, _min);
            _mm_store_ss(&max, _max);
        }

        //-------------------------------------------------------------------------------------------------

        DescrInt::DescrInt(size_t size, size_t depth)
            : Sse41::DescrInt(size, depth)
        {
            _minMax = MinMax;
            //_microM = 2;
            //_microN = 4;
            //switch (depth)
            //{
            //case 6:
            //{
            //    _encode = Encode6;
            //    _decode = Decode6;
            //    _cosineDistance = Sse41::CosineDistance<6>;
            //    _macroCosineDistances = Sse41::MacroCosineDistances<6>;
            //    break;
            //}
            //case 7:
            //{
            //    _encode = Encode7;
            //    _decode = Decode7;
            //    _cosineDistance = Sse41::CosineDistance<7>;
            //    _macroCosineDistances = Sse41::MacroCosineDistances<7>;
            //    break;
            //}
            //case 8:
            //{
            //    _encode = Encode8;
            //    _decode = Decode8;
            //    _cosineDistance = Sse41::CosineDistance<8>;
            //    _macroCosineDistances = Sse41::MacroCosineDistances<8>;
            //    break;
            //}
            //default:
            //    assert(0);
            //}
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
