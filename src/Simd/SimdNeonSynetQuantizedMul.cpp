/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdSynetQuantizedMul.h"
#include "Simd/SimdSynetQuantizeLinear.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Neon
    {
        SIMD_INLINE int32x4_t QuantizedMul8u8u8u(int32x4_t a, int32x4_t aBias, float32x4_t aNorm,
            int32x4_t b, int32x4_t bBias, float32x4_t bNorm, float32x4_t dNorm, int32x4_t dZero)
        {
            float32x4_t _a = DequantizeLinear(a, aBias, aNorm);
            float32x4_t _b = DequantizeLinear(b, bBias, bNorm);
            return QuantizeLinear(vmulq_f32(_a, _b), dNorm, dZero);
        }

        SIMD_INLINE void Store1(int32x4_t src, uint8_t* dst)
        {
            uint8x8_t u8 = vqmovun_s16(vcombine_s16(vqmovn_s32(src), vdup_n_s16(0)));
            dst[0] = vget_lane_u8(u8, 0);
        }

        SIMD_INLINE void Store4(int32x4_t src, uint8_t* dst)
        {
            uint8x8_t u8 = vqmovun_s16(vcombine_s16(vqmovn_s32(src), vdup_n_s16(0)));
            vst1_lane_u32((uint32_t*)dst, vreinterpret_u32_u8(u8), 0);
        }

        SIMD_INLINE void QuantizedMul8u8u8u1(const uint8_t* a, int32x4_t aBias, float32x4_t aNorm,
            const uint8_t* b, int32x4_t bBias, float32x4_t bNorm, uint8_t* dst, float32x4_t dNorm, int32x4_t dZero)
        {
            int32x4_t a0 = vreinterpretq_s32_u32(vdupq_n_u32((uint32_t)a[0]));
            int32x4_t b0 = vreinterpretq_s32_u32(vdupq_n_u32((uint32_t)b[0]));
            Store1(QuantizedMul8u8u8u(a0, aBias, aNorm, b0, bBias, bNorm, dNorm, dZero), dst);
        }

        SIMD_INLINE void QuantizedMul8u8u8u4(const uint8_t* a, int32x4_t aBias, float32x4_t aNorm,
            const uint8_t* b, int32x4_t bBias, float32x4_t bNorm, uint8_t* dst, float32x4_t dNorm, int32x4_t dZero)
        {
            uint8x8_t a8 = vreinterpret_u8_u32(vdup_n_u32(*(const uint32_t*)a));
            uint8x8_t b8 = vreinterpret_u8_u32(vdup_n_u32(*(const uint32_t*)b));
            int32x4_t a0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(a8))));
            int32x4_t b0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b8))));
            Store4(QuantizedMul8u8u8u(a0, aBias, aNorm, b0, bBias, bNorm, dNorm, dZero), dst);
        }

        SIMD_INLINE void QuantizedMul8u8u8u16(const uint8_t* a, int32x4_t aBias, float32x4_t aNorm,
            const uint8_t* b, int32x4_t bBias, float32x4_t bNorm, uint8_t* dst, float32x4_t dNorm, int32x4_t dZero)
        {
            uint8x16_t a8 = vld1q_u8(a), b8 = vld1q_u8(b);
            uint16x8_t a16lo = vmovl_u8(vget_low_u8(a8)), a16hi = vmovl_u8(vget_high_u8(a8));
            uint16x8_t b16lo = vmovl_u8(vget_low_u8(b8)), b16hi = vmovl_u8(vget_high_u8(b8));
            int32x4_t d0 = QuantizedMul8u8u8u(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(a16lo))), aBias, aNorm,
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(b16lo))), bBias, bNorm, dNorm, dZero);
            int32x4_t d1 = QuantizedMul8u8u8u(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(a16lo))), aBias, aNorm,
                vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(b16lo))), bBias, bNorm, dNorm, dZero);
            int32x4_t d2 = QuantizedMul8u8u8u(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(a16hi))), aBias, aNorm,
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(b16hi))), bBias, bNorm, dNorm, dZero);
            int32x4_t d3 = QuantizedMul8u8u8u(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(a16hi))), aBias, aNorm,
                vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(b16hi))), bBias, bNorm, dNorm, dZero);
            vst1q_u8(dst, vcombine_u8(
                vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vqmovn_s32(d1))),
                vqmovun_s16(vcombine_s16(vqmovn_s32(d2), vqmovn_s32(d3)))));
        }

        SIMD_INLINE void QuantizedMul8u8u8uNN(const uint8_t* a, int32x4_t aBias, float32x4_t aNorm, const uint8_t* b, int32x4_t bBias, float32x4_t bNorm,
            size_t n16, size_t n4, size_t n1, uint8_t* dst, float32x4_t dNorm, int32x4_t dZero)
        {
            size_t i = 0;
            for (; i < n16; i += 16)
                QuantizedMul8u8u8u16(a + i, aBias, aNorm, b + i, bBias, bNorm, dst + i, dNorm, dZero);
            for (; i < n4; i += 4)
                QuantizedMul8u8u8u4(a + i, aBias, aNorm, b + i, bBias, bNorm, dst + i, dNorm, dZero);
            for (; i < n1; i += 1)
                QuantizedMul8u8u8u1(a + i, aBias, aNorm, b + i, bBias, bNorm, dst + i, dNorm, dZero);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE int32x4_t QuantizedMul32f8u8u(float32x4_t a, int32x4_t b, int32x4_t bBias, float32x4_t bNorm, float32x4_t dNorm, int32x4_t dZero)
        {
            float32x4_t _b = DequantizeLinear(b, bBias, bNorm);
            return QuantizeLinear(vmulq_f32(a, _b), dNorm, dZero);
        }

        SIMD_INLINE void QuantizedMul32f8u8u1(float32x4_t a, const uint8_t* b, int32x4_t bBias, float32x4_t bNorm, uint8_t* dst, float32x4_t dNorm, int32x4_t dZero)
        {
            int32x4_t b0 = vreinterpretq_s32_u32(vdupq_n_u32((uint32_t)b[0]));
            Store1(QuantizedMul32f8u8u(a, b0, bBias, bNorm, dNorm, dZero), dst);
        }

        SIMD_INLINE void QuantizedMul32f8u8u4(float32x4_t a, const uint8_t* b, int32x4_t bBias, float32x4_t bNorm, uint8_t* dst, float32x4_t dNorm, int32x4_t dZero)
        {
            uint8x8_t b8 = vreinterpret_u8_u32(vdup_n_u32(*(const uint32_t*)b));
            int32x4_t b0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b8))));
            Store4(QuantizedMul32f8u8u(a, b0, bBias, bNorm, dNorm, dZero), dst);
        }

        SIMD_INLINE void QuantizedMul32f8u8u16(float32x4_t a, const uint8_t* b, int32x4_t bBias, float32x4_t bNorm, uint8_t* dst, float32x4_t dNorm, int32x4_t dZero)
        {
            uint8x16_t b8 = vld1q_u8(b);
            uint16x8_t b16lo = vmovl_u8(vget_low_u8(b8)), b16hi = vmovl_u8(vget_high_u8(b8));
            int32x4_t d0 = QuantizedMul32f8u8u(a, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(b16lo))), bBias, bNorm, dNorm, dZero);
            int32x4_t d1 = QuantizedMul32f8u8u(a, vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(b16lo))), bBias, bNorm, dNorm, dZero);
            int32x4_t d2 = QuantizedMul32f8u8u(a, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(b16hi))), bBias, bNorm, dNorm, dZero);
            int32x4_t d3 = QuantizedMul32f8u8u(a, vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(b16hi))), bBias, bNorm, dNorm, dZero);
            vst1q_u8(dst, vcombine_u8(
                vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vqmovn_s32(d1))),
                vqmovun_s16(vcombine_s16(vqmovn_s32(d2), vqmovn_s32(d3)))));
        }

        SIMD_INLINE void QuantizedMul8u8u8u1N(const uint8_t* a, int32x4_t aBias, float32x4_t aNorm, const uint8_t* b, int32x4_t bBias, float32x4_t bNorm,
            size_t n16, size_t n4, size_t n1, uint8_t* dst, float32x4_t dNorm, int32x4_t dZero)
        {
            float32x4_t _a = DequantizeLinear(vreinterpretq_s32_u32(vdupq_n_u32((uint32_t)a[0])), aBias, aNorm);
            size_t i = 0;
            for (; i < n16; i += 16)
                QuantizedMul32f8u8u16(_a, b + i, bBias, bNorm, dst + i, dNorm, dZero);
            for (; i < n4; i += 4)
                QuantizedMul32f8u8u4(_a, b + i, bBias, bNorm, dst + i, dNorm, dZero);
            for (; i < n1; i += 1)
                QuantizedMul32f8u8u1(_a, b + i, bBias, bNorm, dst + i, dNorm, dZero);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE int32x4_t QuantizedMul8u32f8u(int32x4_t a, int32x4_t aBias, float32x4_t aNorm, float32x4_t b, float32x4_t dNorm, int32x4_t dZero)
        {
            float32x4_t _a = DequantizeLinear(a, aBias, aNorm);
            return QuantizeLinear(vmulq_f32(_a, b), dNorm, dZero);
        }

        SIMD_INLINE void QuantizedMul8u32f8u1(const uint8_t* a, int32x4_t aBias, float32x4_t aNorm, float32x4_t b, uint8_t* dst, float32x4_t dNorm, int32x4_t dZero)
        {
            int32x4_t a0 = vreinterpretq_s32_u32(vdupq_n_u32((uint32_t)a[0]));
            Store1(QuantizedMul8u32f8u(a0, aBias, aNorm, b, dNorm, dZero), dst);
        }

        SIMD_INLINE void QuantizedMul8u32f8u4(const uint8_t* a, int32x4_t aBias, float32x4_t aNorm, float32x4_t b, uint8_t* dst, float32x4_t dNorm, int32x4_t dZero)
        {
            uint8x8_t a8 = vreinterpret_u8_u32(vdup_n_u32(*(const uint32_t*)a));
            int32x4_t a0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(a8))));
            Store4(QuantizedMul8u32f8u(a0, aBias, aNorm, b, dNorm, dZero), dst);
        }

        SIMD_INLINE void QuantizedMul8u32f8u16(const uint8_t* a, int32x4_t aBias, float32x4_t aNorm, float32x4_t b, uint8_t* dst, float32x4_t dNorm, int32x4_t dZero)
        {
            uint8x16_t a8 = vld1q_u8(a);
            uint16x8_t a16lo = vmovl_u8(vget_low_u8(a8)), a16hi = vmovl_u8(vget_high_u8(a8));
            int32x4_t d0 = QuantizedMul8u32f8u(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(a16lo))), aBias, aNorm, b, dNorm, dZero);
            int32x4_t d1 = QuantizedMul8u32f8u(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(a16lo))), aBias, aNorm, b, dNorm, dZero);
            int32x4_t d2 = QuantizedMul8u32f8u(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(a16hi))), aBias, aNorm, b, dNorm, dZero);
            int32x4_t d3 = QuantizedMul8u32f8u(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(a16hi))), aBias, aNorm, b, dNorm, dZero);
            vst1q_u8(dst, vcombine_u8(
                vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vqmovn_s32(d1))),
                vqmovun_s16(vcombine_s16(vqmovn_s32(d2), vqmovn_s32(d3)))));
        }

        SIMD_INLINE void QuantizedMul8u8u8uN1(const uint8_t* a, int32x4_t aBias, float32x4_t aNorm, const uint8_t* b, int32x4_t bBias, float32x4_t bNorm,
            size_t n16, size_t n4, size_t n1, uint8_t* dst, float32x4_t dNorm, int32x4_t dZero)
        {
            float32x4_t _b = DequantizeLinear(vreinterpretq_s32_u32(vdupq_n_u32((uint32_t)b[0])), bBias, bNorm);
            size_t i = 0;
            for (; i < n16; i += 16)
                QuantizedMul8u32f8u16(a + i, aBias, aNorm, _b, dst + i, dNorm, dZero);
            for (; i < n4; i += 4)
                QuantizedMul8u32f8u4(a + i, aBias, aNorm, _b, dst + i, dNorm, dZero);
            for (; i < n1; i += 1)
                QuantizedMul8u32f8u1(a + i, aBias, aNorm, _b, dst + i, dNorm, dZero);
        }

        //-------------------------------------------------------------------------------------------------

        template <size_t N> void QuantizedMulUniversal8u8u8u(const uint8_t* a8, const size_t* aSteps, float aScale, int aZero,
            const uint8_t* b8, const size_t* bSteps, float bScale, int bZero, uint8_t* dst8, const size_t* dstShape, float dScale, int dZero)
        {
            int32x4_t _aBias = vdupq_n_s32(-aZero), _bBias = vdupq_n_s32(-bZero), _dZero = vdupq_n_s32(dZero);
            float32x4_t _aScale = vdupq_n_f32(aScale), _bScale = vdupq_n_f32(bScale), _scale = vdupq_n_f32(1.0f / dScale);
            const uint8_t* a0 = (uint8_t*)a8;
            const uint8_t* b0 = (uint8_t*)b8;
            uint8_t* dst = (uint8_t*)dst8;
            size_t n1 = dstShape[N - 1], n4 = AlignLo(n1, 4), n16 = AlignLo(n1, 16);
            bool aN = aSteps[N - 1] == 1, bN = bSteps[N - 1] == 1;
            if (N == 1)
            {
                if (aN && bN)
                    QuantizedMul8u8u8uNN(a0, _aBias, _aScale, b0, _bBias, _bScale, n16, n4, n1, dst, _scale, _dZero);
                else if (bN)
                    QuantizedMul8u8u8u1N(a0, _aBias, _aScale, b0, _bBias, _bScale, n16, n4, n1, dst, _scale, _dZero);
                else if (aN)
                    QuantizedMul8u8u8uN1(a0, _aBias, _aScale, b0, _bBias, _bScale, n16, n4, n1, dst, _scale, _dZero);
            }
            else if (N == 2)
            {
                for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
                {
                    if (aN && bN)
                        QuantizedMul8u8u8uNN(a0, _aBias, _aScale, b0, _bBias, _bScale, n16, n4, n1, dst, _scale, _dZero);
                    else if (bN)
                        QuantizedMul8u8u8u1N(a0, _aBias, _aScale, b0, _bBias, _bScale, n16, n4, n1, dst, _scale, _dZero);
                    else if (aN)
                        QuantizedMul8u8u8uN1(a0, _aBias, _aScale, b0, _bBias, _bScale, n16, n4, n1, dst, _scale, _dZero);
                    dst += n1;
                    a0 += aSteps[0];
                    b0 += bSteps[0];
                }
            }
            else if (N == 3)
            {
                for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
                {
                    const uint8_t* a1 = a0;
                    const uint8_t* b1 = b0;
                    for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                    {
                        if (aN && bN)
                            QuantizedMul8u8u8uNN(a1, _aBias, _aScale, b1, _bBias, _bScale, n16, n4, n1, dst, _scale, _dZero);
                        else if (bN)
                            QuantizedMul8u8u8u1N(a1, _aBias, _aScale, b1, _bBias, _bScale, n16, n4, n1, dst, _scale, _dZero);
                        else if (aN)
                            QuantizedMul8u8u8uN1(a1, _aBias, _aScale, b1, _bBias, _bScale, n16, n4, n1, dst, _scale, _dZero);
                        dst += n1;
                        a1 += aSteps[1];
                        b1 += bSteps[1];
                    }
                    a0 += aSteps[0];
                    b0 += bSteps[0];
                }
            }
            else if (N == 4)
            {
                for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
                {
                    const uint8_t* a1 = a0;
                    const uint8_t* b1 = b0;
                    for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                    {
                        const uint8_t* a2 = a1;
                        const uint8_t* b2 = b1;
                        for (size_t i2 = 0; i2 < dstShape[2]; ++i2)
                        {
                            if (aN && bN)
                                QuantizedMul8u8u8uNN(a2, _aBias, _aScale, b2, _bBias, _bScale, n16, n4, n1, dst, _scale, _dZero);
                            else if (bN)
                                QuantizedMul8u8u8u1N(a2, _aBias, _aScale, b2, _bBias, _bScale, n16, n4, n1, dst, _scale, _dZero);
                            else if (aN)
                                QuantizedMul8u8u8uN1(a2, _aBias, _aScale, b2, _bBias, _bScale, n16, n4, n1, dst, _scale, _dZero);
                            dst += n1;
                            a2 += aSteps[2];
                            b2 += bSteps[2];
                        }
                        a1 += aSteps[1];
                        b1 += bSteps[1];
                    }
                    a0 += aSteps[0];
                    b0 += bSteps[0];
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        static SynetQuantizedMulUniversal::UniversalPtr GetQuantizedMulUniversal8u8u8u(size_t dim)
        {
            switch (dim)
            {
            case 1: return QuantizedMulUniversal8u8u8u<1>;
            case 2: return QuantizedMulUniversal8u8u8u<2>;
            case 3: return QuantizedMulUniversal8u8u8u<3>;
            case 4: return QuantizedMulUniversal8u8u8u<4>;
            default:
                return NULL;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetQuantizedMulUniversal::SynetQuantizedMulUniversal(const QuantizedMulParam& p)
            : Base::SynetQuantizedMulUniversal(p)
        {
            if (p.aType == SimdTensorData8u && p.bType == SimdTensorData8u && p.dType == SimdTensorData8u)
                _universal = GetQuantizedMulUniversal8u8u8u(_dShape.size());
        }

        ////-------------------------------------------------------------------------------------------------

        void* SynetQuantizedMulInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const float* aScale, int32_t aZero,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, const float* bScale, int32_t bZero, SimdTensorDataType dstType, const float* dstScale, int32_t dstZero)
        {
            QuantizedMulParam param(aShape, aCount, aType, aScale, aZero, bShape, bCount, bType, bScale, bZero, dstType, dstScale, dstZero);
            if (!param.Valid())
                return NULL;
            if (SynetQuantizedMulUniversal::Preferable(param))
                return new SynetQuantizedMulUniversal(param);
            return NULL;
        }
    }
#endif
}
