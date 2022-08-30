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
#include "Simd/SimdMath.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdUnpack.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        void Float32ToBFloat16(const float* src, size_t size, uint16_t* dst)
        {
            size_t size16 = Simd::AlignLo(size, 16);
            size_t size8 = Simd::AlignLo(size, 8);
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size16; i += 16)
            {
                __m256i d0 = Float32ToBFloat16(_mm256_loadu_ps(src + i + 0));
                __m256i d1 = Float32ToBFloat16(_mm256_loadu_ps(src + i + 8));
                _mm256_storeu_si256((__m256i*)(dst + i), _mm256_permute4x64_epi64(_mm256_packus_epi32(d0, d1), 0xD8));
            }
            for (; i < size8; i += 8)
            {
                __m128i d0 = Sse41::Float32ToBFloat16(_mm_loadu_ps(src + i + 0));
                __m128i d1 = Sse41::Float32ToBFloat16(_mm_loadu_ps(src + i + 4));
                _mm_storeu_si128((__m128i*)(dst + i), _mm_packus_epi32(d0, d1));
            }
            for (; i < size4; i += 4)
            {
                __m128i d0 = Sse41::Float32ToBFloat16(_mm_loadu_ps(src + i + 0));
                _mm_storel_epi64((__m128i*)(dst + i), _mm_packus_epi32(d0, Sse41::K_ZERO));
            }
            for (; i < size; ++i)
                dst[i] = Base::Float32ToBFloat16(src[i]);
        }

        //---------------------------------------------------------------------------------------------

        void BFloat16ToFloat32(const uint16_t* src, size_t size, float* dst)
        {
            size_t size16 = Simd::AlignLo(size, 16);
            size_t size8 = Simd::AlignLo(size, 8);
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size16; i += 16)
            {
                __m256 d0 = BFloat16ToFloat32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(src + i) + 0)));
                __m256 d1 = BFloat16ToFloat32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(src + i) + 1)));
                _mm256_storeu_ps(dst + i + 0, d0);
                _mm256_storeu_ps(dst + i + F, d1);
            }
            for (; i < size8; i += 8)
            {
                __m128i s = _mm_loadu_si128((__m128i*)(src + i));
                _mm_storeu_ps(dst + i + 0, Sse41::BFloat16ToFloat32(Sse41::UnpackU16<0>(s)));
                _mm_storeu_ps(dst + i + 4, Sse41::BFloat16ToFloat32(Sse41::UnpackU16<1>(s)));
            }
            for (; i < size4; i += 4)
            {
                __m128i s = _mm_loadl_epi64((__m128i*)(src + i));
                _mm_storeu_ps(dst + i + 0, Sse41::BFloat16ToFloat32(Sse41::UnpackU16<0>(s)));
            }
            for (; i < size; ++i)
                dst[i] = Base::BFloat16ToFloat32(src[i]);
        }
    }
#endif
}
