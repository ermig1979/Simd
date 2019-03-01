/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdSynet.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE  
    namespace Neon
    {
        template <bool align> SIMD_INLINE void SynetAddBias(const float * bias, float * dst)
        {
            Store<align>(dst, vaddq_f32(Load<align>(dst), Load<align>(bias)));
        }

        template <bool align> SIMD_INLINE void SynetAddBias(float32x4_t bias, float * dst)
        {
            Store<align>(dst, vaddq_f32(Load<align>(dst), bias));
        }

        template <bool align> SIMD_INLINE void SynetAddBias(const float * bias, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert(((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(bias) : Aligned(size)) && Aligned(dst));
            if ((trans || size == 1) && count != 1)
            {
                size_t aligned = AlignLo(count, QF);
                size_t partial = AlignLo(count, F);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    if (partial)
                    {
                        for (; i < aligned; i += QF)
                        {
                            SynetAddBias<align>(bias + i + F * 0, dst + i + F * 0);
                            SynetAddBias<align>(bias + i + F * 1, dst + i + F * 1);
                            SynetAddBias<align>(bias + i + F * 2, dst + i + F * 2);
                            SynetAddBias<align>(bias + i + F * 3, dst + i + F * 3);
                        }
                        for (; i < partial; i += F)
                            SynetAddBias<align>(bias + i, dst + i);
                    }
                    for (; i < count; ++i)
                        dst[i] += bias[i];
                    dst += count;
                }
            }
            else
            {
                size_t aligned = AlignLo(size, QF);
                size_t partial = AlignLo(size, F);
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    if (partial)
                    {
                        float32x4_t _bias = vdupq_n_f32(bias[i]);
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
        }

        void SynetAddBias(const float * bias, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(bias) : Aligned(size)) && Aligned(dst))
                SynetAddBias<true>(bias, count, size, dst, trans);
            else
                SynetAddBias<false>(bias, count, size, dst, trans);
        }

        template <SimdSynetEltwiseOperationType type> float32x4_t SynetEltwiseLayerForward(float32x4_t src0, float32x4_t src1);

        template <> SIMD_INLINE float32x4_t SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct>(float32x4_t src0, float32x4_t src1)
        {
            return vmulq_f32(src0, src1);
        }

        template <> SIMD_INLINE float32x4_t SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax>(float32x4_t src0, float32x4_t src1)
        {
            return vmaxq_f32(src0, src1);
        }

        template <> SIMD_INLINE float32x4_t SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin>(float32x4_t src0, float32x4_t src1)
        {
            return vminq_f32(src0, src1);
        }

        template <SimdSynetEltwiseOperationType type, bool align> SIMD_INLINE void SynetEltwiseLayerForward(const float * src0, const float * src1, float * dst, size_t offset)
        {
            Store<align>(dst + offset, SynetEltwiseLayerForward<type>(Load<align>(src0 + offset), Load<align>(src1 + offset)));
        }

        template <SimdSynetEltwiseOperationType type, bool align> void SynetEltwiseLayerForward(float const * const * src, size_t count, size_t size, float * dst)
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
                    SynetEltwiseLayerForward<type, align>(src0, src1, dst, j + F * 0);
                    SynetEltwiseLayerForward<type, align>(src0, src1, dst, j + F * 1);
                    SynetEltwiseLayerForward<type, align>(src0, src1, dst, j + F * 2);
                    SynetEltwiseLayerForward<type, align>(src0, src1, dst, j + F * 3);
                }
                for (; j < partial; j += F)
                    SynetEltwiseLayerForward<type, align>(src0, src1, dst, j);
            }
            for (; j < size; ++j)
                dst[j] = Base::SynetEltwiseLayerForward<type>(src0[j], src1[j]);
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                size_t j = 0;
                if (partial)
                {
                    for (; j < aligned; j += QF)
                    {
                        SynetEltwiseLayerForward<type, align>(dst, srci, dst, j + F * 0);
                        SynetEltwiseLayerForward<type, align>(dst, srci, dst, j + F * 1);
                        SynetEltwiseLayerForward<type, align>(dst, srci, dst, j + F * 2);
                        SynetEltwiseLayerForward<type, align>(dst, srci, dst, j + F * 3);
                    }
                    for (; j < partial; j += F)
                        SynetEltwiseLayerForward<type, align>(dst, srci, dst, j);
                }
                for (; j < size; ++j)
                    dst[j] = Base::SynetEltwiseLayerForward<type>(dst[j], srci[j]);
            }
        }

        template <bool align> SIMD_INLINE void SynetEltwiseLayerForwardSum(const float * src0, const float32x4_t & weight0, const float * src1, const float32x4_t & weight1, float * dst, size_t offset)
        {
            Store<align>(dst + offset, vmlaq_f32(vmulq_f32(Load<align>(src0 + offset), weight0), Load<align>(src1 + offset), weight1));
        }

        template <bool align> SIMD_INLINE void SynetEltwiseLayerForwardSum(const float * src, const float32x4_t & weight, float * dst, size_t offset)
        {
            Store<align>(dst + offset, vmlaq_f32(Load<align>(dst + offset), Load<align>(src + offset), weight));
        }

        template <bool align> void SynetEltwiseLayerForwardSum(float const * const * src, const float * weight, size_t count, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            const float * src0 = src[0];
            const float * src1 = src[1];
            float32x4_t weight0 = vdupq_n_f32(weight[0]);
            float32x4_t weight1 = vdupq_n_f32(weight[1]);
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
                float32x4_t weighti = vdupq_n_f32(weight[i]);
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

        template <bool align> void SynetEltwiseLayerForward(float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst)
        {
            switch (type)
            {
            case SimdSynetEltwiseOperationProduct:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct, align>(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationSum:
                SynetEltwiseLayerForwardSum<align>(src, weight, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMax:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax, align>(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMin:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin, align>(src, count, size, dst);
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
    }
#endif// SIMD_NEON_ENABLE
}
