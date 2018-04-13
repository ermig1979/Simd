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

        template <bool align> void SynetEltwiseLayerForwardProduct(const float * src0, const float * src1, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_mul_ps(Load<align>(src0 + offset), Load<align>(src1 + offset)));
        }

        template <bool align> void SynetEltwiseLayerForwardProduct(float const * const * src, size_t count, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            const float * src0 = src[0];
            const float * src1 = src[1];
            size_t j = 0;
            if (partial)
            {
                for (; j < aligned; j += QF)
                {
                    SynetEltwiseLayerForwardProduct<align>(src0, src1, dst, j + F * 0);
                    SynetEltwiseLayerForwardProduct<align>(src0, src1, dst, j + F * 1);
                    SynetEltwiseLayerForwardProduct<align>(src0, src1, dst, j + F * 2);
                    SynetEltwiseLayerForwardProduct<align>(src0, src1, dst, j + F * 3);
                }
                for (; j < partial; j += F)
                    SynetEltwiseLayerForwardProduct<align>(src0, src1, dst, j);
            }
            for (; j < size; ++j)
                dst[j] = src0[j] * src1[j];
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                size_t j = 0;
                if (partial)
                {
                    for (; j < aligned; j += QF)
                    {
                        SynetEltwiseLayerForwardProduct<align>(dst, srci, dst, j + F * 0);
                        SynetEltwiseLayerForwardProduct<align>(dst, srci, dst, j + F * 1);
                        SynetEltwiseLayerForwardProduct<align>(dst, srci, dst, j + F * 2);
                        SynetEltwiseLayerForwardProduct<align>(dst, srci, dst, j + F * 3);
                    }
                    for (; j < partial; j += F)
                        SynetEltwiseLayerForwardProduct<align>(dst, srci, dst, j);
                }
                for (; j < size; ++j)
                    dst[j] *= srci[j];
            }
        }

        template <bool align> void SynetEltwiseLayerForwardSum(const float * src0, const __m256 & weight0, const float * src1, const __m256 & weight1, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_add_ps(_mm256_mul_ps(Load<align>(src0 + offset), weight0), _mm256_mul_ps(Load<align>(src1 + offset), weight1)));
        }

        template <bool align> void SynetEltwiseLayerForwardSum(const float * src, const __m256 & weight, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_add_ps(_mm256_mul_ps(Load<align>(src + offset), weight), Load<align>(dst + offset)));
        }

        template <bool align> void SynetEltwiseLayerForwardSum(float const * const * src, const float * weight, size_t count, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            const float * src0 = src[0];
            const float * src1 = src[1];
            __m256 weight0 = _mm256_set1_ps(weight[0]);
            __m256 weight1 = _mm256_set1_ps(weight[1]);
            size_t j = 0;
            if (partial)
            {
                for (; j < aligned; j += QF)
                {
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j + F * 0);
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j + F * 1);
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j + F * 2);
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j + F * 3);
                }
                for (; j < partial; j += F)
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j);
            }
            for (; j < size; ++j)
                dst[j] = src0[j] * weight[0] + src1[j] * weight[1];
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                __m256 weighti = _mm256_set1_ps(weight[i]);
                size_t j = 0;
                if (partial)
                {
                    for (; j < aligned; j += QF)
                    {
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j + F * 0);
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j + F * 1);
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j + F * 2);
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j + F * 3);
                    }
                    for (; j < partial; j += F)
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j);
                }
                for (; j < size; ++j)
                    dst[j] += srci[j] * weight[i];
            }
        }

        template <bool align> void SynetEltwiseLayerForwardMax(const float * src0, const float * src1, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_max_ps(Load<align>(src0 + offset), Load<align>(src1 + offset)));
        }

        template <bool align> void SynetEltwiseLayerForwardMax(float const * const * src, size_t count, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            const float * src0 = src[0];
            const float * src1 = src[1];
            size_t j = 0;
            if (partial)
            {
                for (; j < aligned; j += QF)
                {
                    SynetEltwiseLayerForwardMax<align>(src0, src1, dst, j + F * 0);
                    SynetEltwiseLayerForwardMax<align>(src0, src1, dst, j + F * 1);
                    SynetEltwiseLayerForwardMax<align>(src0, src1, dst, j + F * 2);
                    SynetEltwiseLayerForwardMax<align>(src0, src1, dst, j + F * 3);
                }
                for (; j < partial; j += F)
                    SynetEltwiseLayerForwardMax<align>(src0, src1, dst, j);
            }
            for (; j < size; ++j)
                dst[j] = Simd::Max(src0[j], src1[j]);
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                size_t j = 0;
                if (partial)
                {
                    for (; j < aligned; j += QF)
                    {
                        SynetEltwiseLayerForwardMax<align>(dst, srci, dst, j + F * 0);
                        SynetEltwiseLayerForwardMax<align>(dst, srci, dst, j + F * 1);
                        SynetEltwiseLayerForwardMax<align>(dst, srci, dst, j + F * 2);
                        SynetEltwiseLayerForwardMax<align>(dst, srci, dst, j + F * 3);
                    }
                    for (; j < partial; j += F)
                        SynetEltwiseLayerForwardMax<align>(dst, srci, dst, j);
                }
                for (; j < size; ++j)
                    dst[j] = Simd::Max(dst[j], srci[j]);
            }
        }

        template <bool align> void SynetEltwiseLayerForward(float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst)
        {
            switch (type)
            {
            case SimdSynetEltwiseOperationProduct:
                SynetEltwiseLayerForwardProduct<align>(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationSum:
                SynetEltwiseLayerForwardSum<align>(src, weight, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMax:
                SynetEltwiseLayerForwardMax<align>(src, count, size, dst);
                break;
            default:
                assert(0);
            }
        }

        void SynetEltwiseLayerForward(float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst)
        {
            assert(count >= 2);
            bool aligned = Aligned(dst) && Aligned(src[0]) && Aligned(src[1]);
            for (size_t i = 2; i < count; ++i)
                aligned = aligned && Aligned(src[i]);
            if (aligned)
                SynetEltwiseLayerForward<true>(src, weight, count, size, type, dst);
            else
                SynetEltwiseLayerForward<false>(src, weight, count, size, type, dst);
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
