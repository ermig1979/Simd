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
#include "Simd/SimdSve2.h"
#include "Simd/SimdSynetQuantizeLinear.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE void Store8u(const svint32_t& value, uint8_t* dst, const svbool_t& mask)
        {
            svint32_t lo = svmax_n_s32_x(mask, value, 0);
            svuint32_t u32 = svreinterpret_u32_s32(svmin_n_s32_x(mask, lo, 255));
            svst1b_u32(mask, dst, u32);
        }

        SIMD_INLINE svint32_t QuantizedMul8u8u8u(const svint32_t& a, const svint32_t& aBias, const svfloat32_t& aNorm,
            const svint32_t& b, const svint32_t& bBias, const svfloat32_t& bNorm, const svfloat32_t& dNorm, const svint32_t& dZero, const svbool_t& mask)
        {
            svfloat32_t _a = DequantizeLinear(a, aBias, aNorm, mask);
            svfloat32_t _b = DequantizeLinear(b, bBias, bNorm, mask);
            return QuantizeLinear(svmul_f32_x(mask, _a, _b), dNorm, dZero, mask);
        }

        SIMD_INLINE svint32_t Load8u(const uint8_t* src, const svbool_t& mask)
        {
            return svreinterpret_s32_u32(svld1ub_u32(mask, src));
        }

        SIMD_INLINE void QuantizedMul8u8u8uNN(const uint8_t* a, const svint32_t& aBias, const svfloat32_t& aNorm, const uint8_t* b, const svint32_t& bBias, const svfloat32_t& bNorm,
            size_t size, uint8_t* dst, const svfloat32_t& dNorm, const svint32_t& dZero)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t full = svptrue_b32();
            size_t i = 0;
            for (; i + QF <= size; i += QF)
            {
                Store8u(QuantizedMul8u8u8u(Load8u(a + i + 0 * F, full), aBias, aNorm, Load8u(b + i + 0 * F, full), bBias, bNorm, dNorm, dZero, full), dst + i + 0 * F, full);
                Store8u(QuantizedMul8u8u8u(Load8u(a + i + 1 * F, full), aBias, aNorm, Load8u(b + i + 1 * F, full), bBias, bNorm, dNorm, dZero, full), dst + i + 1 * F, full);
                Store8u(QuantizedMul8u8u8u(Load8u(a + i + 2 * F, full), aBias, aNorm, Load8u(b + i + 2 * F, full), bBias, bNorm, dNorm, dZero, full), dst + i + 2 * F, full);
                Store8u(QuantizedMul8u8u8u(Load8u(a + i + 3 * F, full), aBias, aNorm, Load8u(b + i + 3 * F, full), bBias, bNorm, dNorm, dZero, full), dst + i + 3 * F, full);
            }
            for (; i < size; i += F)
            {
                svbool_t mask = svwhilelt_b32(i, size);
                Store8u(QuantizedMul8u8u8u(Load8u(a + i, mask), aBias, aNorm, Load8u(b + i, mask), bBias, bNorm, dNorm, dZero, mask), dst + i, mask);
            }
        }

        SIMD_INLINE void QuantizedMul8u8u8u1N(const uint8_t* a, const svint32_t& aBias, const svfloat32_t& aNorm, const uint8_t* b, const svint32_t& bBias, const svfloat32_t& bNorm,
            size_t size, uint8_t* dst, const svfloat32_t& dNorm, const svint32_t& dZero)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t full = svptrue_b32();
            svint32_t _a = svdup_n_s32((int32_t)a[0]);
            size_t i = 0;
            for (; i + QF <= size; i += QF)
            {
                Store8u(QuantizedMul8u8u8u(_a, aBias, aNorm, Load8u(b + i + 0 * F, full), bBias, bNorm, dNorm, dZero, full), dst + i + 0 * F, full);
                Store8u(QuantizedMul8u8u8u(_a, aBias, aNorm, Load8u(b + i + 1 * F, full), bBias, bNorm, dNorm, dZero, full), dst + i + 1 * F, full);
                Store8u(QuantizedMul8u8u8u(_a, aBias, aNorm, Load8u(b + i + 2 * F, full), bBias, bNorm, dNorm, dZero, full), dst + i + 2 * F, full);
                Store8u(QuantizedMul8u8u8u(_a, aBias, aNorm, Load8u(b + i + 3 * F, full), bBias, bNorm, dNorm, dZero, full), dst + i + 3 * F, full);
            }
            for (; i < size; i += F)
            {
                svbool_t mask = svwhilelt_b32(i, size);
                Store8u(QuantizedMul8u8u8u(_a, aBias, aNorm, Load8u(b + i, mask), bBias, bNorm, dNorm, dZero, mask), dst + i, mask);
            }
        }

        SIMD_INLINE void QuantizedMul8u8u8uN1(const uint8_t* a, const svint32_t& aBias, const svfloat32_t& aNorm, const uint8_t* b, const svint32_t& bBias, const svfloat32_t& bNorm,
            size_t size, uint8_t* dst, const svfloat32_t& dNorm, const svint32_t& dZero)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t full = svptrue_b32();
            svint32_t _b = svdup_n_s32((int32_t)b[0]);
            size_t i = 0;
            for (; i + QF <= size; i += QF)
            {
                Store8u(QuantizedMul8u8u8u(Load8u(a + i + 0 * F, full), aBias, aNorm, _b, bBias, bNorm, dNorm, dZero, full), dst + i + 0 * F, full);
                Store8u(QuantizedMul8u8u8u(Load8u(a + i + 1 * F, full), aBias, aNorm, _b, bBias, bNorm, dNorm, dZero, full), dst + i + 1 * F, full);
                Store8u(QuantizedMul8u8u8u(Load8u(a + i + 2 * F, full), aBias, aNorm, _b, bBias, bNorm, dNorm, dZero, full), dst + i + 2 * F, full);
                Store8u(QuantizedMul8u8u8u(Load8u(a + i + 3 * F, full), aBias, aNorm, _b, bBias, bNorm, dNorm, dZero, full), dst + i + 3 * F, full);
            }
            for (; i < size; i += F)
            {
                svbool_t mask = svwhilelt_b32(i, size);
                Store8u(QuantizedMul8u8u8u(Load8u(a + i, mask), aBias, aNorm, _b, bBias, bNorm, dNorm, dZero, mask), dst + i, mask);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <size_t N> void QuantizedMulUniversal8u8u8u(const uint8_t* a8, const size_t* aSteps, float aScale, int aZero,
            const uint8_t* b8, const size_t* bSteps, float bScale, int bZero, uint8_t* dst8, const size_t* dstShape, float dScale, int dZero)
        {
            svint32_t _aBias = svdup_n_s32(-aZero), _bBias = svdup_n_s32(-bZero), _dZero = svdup_n_s32(dZero);
            svfloat32_t _aScale = svdup_n_f32(aScale), _bScale = svdup_n_f32(bScale), _scale = svdup_n_f32(1.0f / dScale);
            const uint8_t* a0 = (uint8_t*)a8;
            const uint8_t* b0 = (uint8_t*)b8;
            uint8_t* dst = (uint8_t*)dst8;
            size_t n1 = dstShape[N - 1];
            bool aN = aSteps[N - 1] == 1, bN = bSteps[N - 1] == 1;
            if (N == 1)
            {
                if (aN && bN)
                    QuantizedMul8u8u8uNN(a0, _aBias, _aScale, b0, _bBias, _bScale, n1, dst, _scale, _dZero);
                else if (bN)
                    QuantizedMul8u8u8u1N(a0, _aBias, _aScale, b0, _bBias, _bScale, n1, dst, _scale, _dZero);
                else if (aN)
                    QuantizedMul8u8u8uN1(a0, _aBias, _aScale, b0, _bBias, _bScale, n1, dst, _scale, _dZero);
            }
            else if (N == 2)
            {
                for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
                {
                    if (aN && bN)
                        QuantizedMul8u8u8uNN(a0, _aBias, _aScale, b0, _bBias, _bScale, n1, dst, _scale, _dZero);
                    else if (bN)
                        QuantizedMul8u8u8u1N(a0, _aBias, _aScale, b0, _bBias, _bScale, n1, dst, _scale, _dZero);
                    else if (aN)
                        QuantizedMul8u8u8uN1(a0, _aBias, _aScale, b0, _bBias, _bScale, n1, dst, _scale, _dZero);
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
                            QuantizedMul8u8u8uNN(a1, _aBias, _aScale, b1, _bBias, _bScale, n1, dst, _scale, _dZero);
                        else if (bN)
                            QuantizedMul8u8u8u1N(a1, _aBias, _aScale, b1, _bBias, _bScale, n1, dst, _scale, _dZero);
                        else if (aN)
                            QuantizedMul8u8u8uN1(a1, _aBias, _aScale, b1, _bBias, _bScale, n1, dst, _scale, _dZero);
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
                                QuantizedMul8u8u8uNN(a2, _aBias, _aScale, b2, _bBias, _bScale, n1, dst, _scale, _dZero);
                            else if (bN)
                                QuantizedMul8u8u8u1N(a2, _aBias, _aScale, b2, _bBias, _bScale, n1, dst, _scale, _dZero);
                            else if (aN)
                                QuantizedMul8u8u8uN1(a2, _aBias, _aScale, b2, _bBias, _bScale, n1, dst, _scale, _dZero);
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
