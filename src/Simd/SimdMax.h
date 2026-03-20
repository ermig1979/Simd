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
#ifndef __SimdMax_h__
#define __SimdMax_h__

#include "Simd/SimdDefs.h"
#include "Simd/SimdMath.h"

namespace Simd
{
    namespace Base
    {

    }

#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        SIMD_INLINE void MaxVal32f(__m128 src, float& dst)
        {
            src = _mm_max_ps(src, Shuffle32f<0x0E>(src));
            src = _mm_max_ss(src, Shuffle32f<0x01>(src));
            _mm_store_ss(&dst, src);
        }

        SIMD_INLINE __m128 BroadcastMax32f(__m128 val)
        {
            val = _mm_max_ps(val, Shuffle32f<0x2E>(val));
            return _mm_max_ps(val, Shuffle32f<0xB1>(val));
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        SIMD_INLINE void MaxVal32f(__m256 src, float& dst)
        {
            Sse41::MaxVal32f(_mm_max_ps(_mm256_castps256_ps128(src), _mm256_extractf128_ps(src, 1)), dst);
        }
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        SIMD_INLINE void MaxVal32f(__m512 src, float& dst)
        {
            Avx2::MaxVal32f(_mm256_max_ps(_mm512_extractf32x8_ps(src, 0), _mm512_extractf32x8_ps(src, 1)), dst);
        }
    }
#endif

#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        SIMD_INLINE void MaxVal32f(float32x4_t src, float& dst)
        {
            float32x2_t half = vpmax_f32(vget_low_f32(src), vget_high_f32(src));
            dst = vget_lane_f32(vpmax_f32(half, half), 0);
        }
    }
#endif

}
#endif
