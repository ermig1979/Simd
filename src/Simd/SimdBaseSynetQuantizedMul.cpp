/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdSynetActivation.h"
#include "Simd/SimdFmadd.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)

    SynetQuantizedMul::SynetQuantizedMul(const QuantizedMulParam& p)
        : _param(p)
    {

    }

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        template <class S, class D> SIMD_INLINE D Convert8u(const S& src, float norm, int bias)
        {
            return (D)src;
        }

        template <> SIMD_INLINE float Convert8u(const uint8_t& src, float norm, int bias)
        {
            return DequantizeLinear(src, bias, norm);
        }

        template <> SIMD_INLINE uint8_t Convert8u(const float& src, float norm, int bias)
        {
            return QuantizeLinear(src, norm, bias, 0, 255);
        }

        template <typename A, typename B, typename D> SIMD_INLINE void QuantizedMul(const A& a, int aBias, float aNorm, const B& b, int bBias, float bNorm, D& dst, float dNorm, int dZero)
        {
            float _a = Convert8u<A, float>(a, aNorm, aBias);
            float _b = Convert8u<B, float>(b, bNorm, bBias);
            dst = Convert8u<float, D>(_a * _b, dNorm, dZero);
        }

        //-------------------------------------------------------------------------------------------------

        template <typename A, typename B, typename D, size_t N> void QuantizedMulUniversal(const uint8_t* a8, const size_t* aSteps, float aScale, int aZero,
            const uint8_t* b8, const size_t* bSteps, float bScale, int bZero, uint8_t* dst8, const size_t* dstShape, float dScale, int dZero)
        {
            int aBias = -aZero, bBias = -bZero;
            float scale = 1.0f / dScale;
            const A* a0 = (A*)a8;
            const B* b0 = (B*)b8;
            D* dst = (D*)dst8;
            if (N == 1)
            {
                if (aSteps[0] == 1 && bSteps[0] == 1)
                {
                    for (size_t i0 = 0, n0 = dstShape[0]; i0 < n0; ++i0)
                        QuantizedMul<A, B, D>(a0[i0], aBias, aScale, b0[i0], bBias, bScale, dst[i0], scale, dZero);
                }
                else
                {
                    for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
                    {
                        QuantizedMul<A, B, D>(*a0, aBias, aScale, *b0, bBias, bScale, *dst, scale, dZero);
                        a0 += aSteps[0];
                        b0 += bSteps[0];
                        dst += 1;
                    }
                }
            }
            else if (N == 2)
            {
                for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
                {
                    const A* a1 = a0;
                    const B* b1 = b0;
                    for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                    {
                        QuantizedMul<A, B, D>(*a1, aBias, aScale, *b1, bBias, bScale, *dst, scale, dZero);
                        a1 += aSteps[1];
                        b1 += bSteps[1];
                        dst += 1;
                    }
                    a0 += aSteps[0];
                    b0 += bSteps[0];
                }
            }
            else if (N == 3)
            {
                for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
                {
                    const A* a1 = a0;
                    const B* b1 = b0;
                    for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                    {
                        const A* a2 = a1;
                        const B* b2 = b1;
                        for (size_t i2 = 0; i2 < dstShape[2]; ++i2)
                        {
                            QuantizedMul<A, B, D>(*a2, aBias, aScale, *b2, bBias, bScale, *dst, scale, dZero);
                            a2 += aSteps[2];
                            b2 += bSteps[2];
                            dst += 1;
                        }
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
                    const A* a1 = a0;
                    const B* b1 = b0;
                    for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                    {
                        const A* a2 = a1;
                        const B* b2 = b1;
                        for (size_t i2 = 0; i2 < dstShape[2]; ++i2)
                        {
                            const A* a3 = a2;
                            const B* b3 = b2;
                            for (size_t i3 = 0; i3 < dstShape[3]; ++i3)
                            {
                                QuantizedMul<A, B, D>(*a3, aBias, aScale, *b3, bBias, bScale, *dst, scale, dZero);
                                a3 += aSteps[3];
                                b3 += bSteps[3];
                                dst += 1;
                            }
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

        template<class A, class B, class D> static SynetQuantizedMulUniversal::UniversalPtr GetQuantizedMulUniversal(size_t dim)
        {
            switch (dim)
            {
            case 1: return QuantizedMulUniversal<A, B, D, 1>;
            case 2: return QuantizedMulUniversal<A, B, D, 2>;
            case 3: return QuantizedMulUniversal<A, B, D, 3>;
            case 4: return QuantizedMulUniversal<A, B, D, 4>;
            default:
                return NULL;
            }
        }

        template<class A, class B> static SynetQuantizedMulUniversal::UniversalPtr GetQuantizedMulUniversal(SimdTensorDataType dType, size_t dim)
        {
            switch (dType)
            {
            case SimdTensorData32f: return GetQuantizedMulUniversal<A, B, float>(dim);
            case SimdTensorData8u: return GetQuantizedMulUniversal<A, B, uint8_t>(dim);
            default:
                return NULL;
            }
        }

        template<class A> static SynetQuantizedMulUniversal::UniversalPtr GetQuantizedMulUniversal(SimdTensorDataType bType, SimdTensorDataType dType, size_t dim)
        {
            switch (bType)
            {
            case SimdTensorData32f: return GetQuantizedMulUniversal<A, float>(dType, dim);
            case SimdTensorData8u: return GetQuantizedMulUniversal<A, uint8_t>(dType, dim);
            default:
                return NULL;
            }
        }

        static SynetQuantizedMulUniversal::UniversalPtr GetQuantizedMulUniversal(SimdTensorDataType aType, SimdTensorDataType bType, SimdTensorDataType dType, size_t dim)
        {
            switch (aType)
            {
            case SimdTensorData32f: return GetQuantizedMulUniversal<float>(bType, dType, dim);
            case SimdTensorData8u: return GetQuantizedMulUniversal<uint8_t>(bType, dType, dim);
            default:
                return NULL;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetQuantizedMulUniversal::SynetQuantizedMulUniversal(const QuantizedMulParam& p)
            : SynetQuantizedMul(p)
            , _universal(0)
        {
            Shape aShape = p.aShape, bShape = p.bShape;
            _dShape = OutputShape(aShape, bShape);

            aShape = FullSrcShape(aShape, _dShape);
            bShape = FullSrcShape(bShape, _dShape);

            CompactShapes(aShape, bShape, _dShape);

            _aSteps = SourceSteps(aShape, _dShape);
            _bSteps = SourceSteps(bShape, _dShape);
            _universal = GetQuantizedMulUniversal(p.aType, p.bType, p.dType, _dShape.size());
        }

        bool SynetQuantizedMulUniversal::Preferable(const QuantizedMulParam& p)
        {
            return true;
        }

        void SynetQuantizedMulUniversal::Forward(const uint8_t* a, const uint8_t* b, uint8_t* dst)
        {
            const QuantizedMulParam& p = _param;
            _universal(a, _aSteps.data(), p.aScale, (int)p.aZero, b, _bSteps.data(), p.bScale, (int)p.bZero, dst, _dShape.data(), p.dScale, (int)p.dZero);
        }

        //-------------------------------------------------------------------------------------------------

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
