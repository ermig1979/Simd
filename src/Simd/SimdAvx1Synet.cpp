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
#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        template <bool align> SIMD_INLINE void SynetAddBias(const __m256 & bias, float * dst)
        {
            Store<align>(dst, _mm256_add_ps(Load<align>(dst), bias));
        }

        template <bool align> SIMD_INLINE void SynetAddBias(const float * bias, size_t count, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);     
            for (size_t i = 0; i < count; ++i)
            {
                size_t j = 0;
                if (partial)
                {
                    __m256 _bias = _mm256_set1_ps(bias[i]);
                    for (; j < aligned; j += QF)
                    {
                        SynetAddBias<align>(_bias, dst + j + F * 0);
                        SynetAddBias<align>(_bias, dst + j + F * 1);
                        SynetAddBias<align>(_bias, dst + j + F * 2);
                        SynetAddBias<align>(_bias, dst + j + F * 3);
                    }
                    for (; j < partial; j += F)
                        SynetAddBias<align>(_bias, dst + j);
                }
                for (; j < size; ++j)
                    dst[j] += bias[i];
                dst += size;
            }
        }

        void SynetAddBias(const float * bias, size_t count, size_t size, float * dst)
        {
            if (Aligned(dst) && Aligned(size))
                SynetAddBias<true>(bias, count, size, dst);
            else
                SynetAddBias<false>(bias, count, size, dst);
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m256 & scale, const __m256 & bias, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_add_ps(_mm256_mul_ps(Load<align>(src + offset), scale), bias));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m256 & scale, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_mul_ps(Load<align>(src + offset), scale));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            if (bias)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    if (partial)
                    {
                        __m256 _scale = _mm256_set1_ps(scale[i]);
                        __m256 _bias = _mm256_set1_ps(bias[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetScaleLayerForward<align>(src, _scale, _bias, dst, j + F * 0);
                            SynetScaleLayerForward<align>(src, _scale, _bias, dst, j + F * 1);
                            SynetScaleLayerForward<align>(src, _scale, _bias, dst, j + F * 2);
                            SynetScaleLayerForward<align>(src, _scale, _bias, dst, j + F * 3);
                        }
                        for (; j < partial; j += F)
                            SynetScaleLayerForward<align>(src, _scale, _bias, dst, j);
                    }
                    for (; j < size; ++j)
                        dst[j] = src[j] * scale[i] + bias[i];
                    src += size;
                    dst += size;
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    if (partial)
                    {
                        __m256 _scale = _mm256_set1_ps(scale[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetScaleLayerForward<align>(src, _scale, dst, j + F * 0);
                            SynetScaleLayerForward<align>(src, _scale, dst, j + F * 1);
                            SynetScaleLayerForward<align>(src, _scale, dst, j + F * 2);
                            SynetScaleLayerForward<align>(src, _scale, dst, j + F * 3);
                        }
                        for (; j < partial; j += F)
                            SynetScaleLayerForward<align>(src, _scale, dst, j);
                    }
                    for (; j < size; ++j)
                        dst[j] = src[j] * scale[i];
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst)
        {
            if (Aligned(dst) && Aligned(size))
                SynetScaleLayerForward<true>(src, scale, bias, count, size, dst);
            else
                SynetScaleLayerForward<false>(src, scale, bias, count, size, dst);
        }
    }
#endif// SIMD_AVX_ENABLE
}
