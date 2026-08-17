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
#include "Simd/SimdSynetQuantizedConvolution.h"
#include "Simd/SimdSynetQuantizedDepthwise.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        using AlgParamV0 = SynetQuantizedConvolutionNhwcDepthwiseV0::AlgParam;

        //------------------------------------------------------------------------------------------------

        SIMD_INLINE svint32_t LoadAs32i(const uint8_t* src, const svbool_t& mask)
        {
            return svreinterpret_s32_u32(svld1ub_u32(mask, src));
        }

        SIMD_INLINE svint32_t LoadAs32i(const int8_t* src, const svbool_t& mask)
        {
            return svld1sb_s32(mask, src);
        }

        SIMD_INLINE void Madd1(svint32_t& i32, const svint32_t& u8, const svint32_t& i8, const svbool_t& mask)
        {
            i32 = svmla_s32_x(mask, i32, u8, i8);
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term> void QuantizedConvolutionNhwcDepthwiseV0_Default(const uint8_t* src, uint32_t srcZero,
            const ConvParam& p, const AlgParamV0& a, const int8_t* weight, const int32_t* bias, const float* norm, uint32_t dstZero, uint8_t* dst)
        {
            const size_t F = svcntw(), DF = F * 2, QF = F * 4;
            const svbool_t body = svptrue_b32();
            svint32_t _srcZero = svdup_n_s32(srcZero);
            svint32_t _dstZero = svdup_n_s32(dstZero);
            svint32_t d00, d01, d02, d03, w0, w1, w2, w3;
            size_t size = p.group, sizeF = AlignLo(size, F), sizeDF = AlignLo(size, DF), sizeQF = AlignLo(size, QF);
            svbool_t tail = svwhilelt_b32((size_t)0, size - sizeF);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t i = 0;
                    for (; i < sizeQF; i += QF)
                    {
                        d00 = svdup_n_s32(0);
                        d01 = svdup_n_s32(0);
                        d02 = svdup_n_s32(0);
                        d03 = svdup_n_s32(0);
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                size_t ow = (ky * p.kernelX + kx) * size + i;
                                w0 = LoadAs32i(weight + ow + 0 * F, body);
                                w1 = LoadAs32i(weight + ow + 1 * F, body);
                                w2 = LoadAs32i(weight + ow + 2 * F, body);
                                w3 = LoadAs32i(weight + ow + 3 * F, body);
                                if (sy < p.srcH && sx < p.srcW)
                                {
                                    size_t os = (sy * p.srcW + sx) * size + i;
                                    Madd1(d00, LoadAs32i(src + os + 0 * F, body), w0, body);
                                    Madd1(d01, LoadAs32i(src + os + 1 * F, body), w1, body);
                                    Madd1(d02, LoadAs32i(src + os + 2 * F, body), w2, body);
                                    Madd1(d03, LoadAs32i(src + os + 3 * F, body), w3, body);
                                }
                                else
                                {
                                    Madd1(d00, _srcZero, w0, body);
                                    Madd1(d01, _srcZero, w1, body);
                                    Madd1(d02, _srcZero, w2, body);
                                    Madd1(d03, _srcZero, w3, body);
                                }
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _dstZero, i + F * 0);
                        Save1<term>(dst, d01, bias, norm, _dstZero, i + F * 1);
                        Save1<term>(dst, d02, bias, norm, _dstZero, i + F * 2);
                        Save1<term>(dst, d03, bias, norm, _dstZero, i + F * 3);
                    }
                    for (; i < sizeDF; i += DF)
                    {
                        d00 = svdup_n_s32(0);
                        d01 = svdup_n_s32(0);
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                size_t ow = (ky * p.kernelX + kx) * size + i;
                                w0 = LoadAs32i(weight + ow + 0 * F, body);
                                w1 = LoadAs32i(weight + ow + 1 * F, body);
                                if (sy < p.srcH && sx < p.srcW)
                                {
                                    size_t os = (sy * p.srcW + sx) * size + i;
                                    Madd1(d00, LoadAs32i(src + os + 0 * F, body), w0, body);
                                    Madd1(d01, LoadAs32i(src + os + 1 * F, body), w1, body);
                                }
                                else
                                {
                                    Madd1(d00, _srcZero, w0, body);
                                    Madd1(d01, _srcZero, w1, body);
                                }
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _dstZero, i + F * 0);
                        Save1<term>(dst, d01, bias, norm, _dstZero, i + F * 1);
                    }
                    for (; i < sizeF; i += F)
                    {
                        d00 = svdup_n_s32(0);
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                w0 = LoadAs32i(weight + (ky * p.kernelX + kx) * size + i, body);
                                if (sy < p.srcH && sx < p.srcW)
                                    Madd1(d00, LoadAs32i(src + (sy * p.srcW + sx) * size + i, body), w0, body);
                                else
                                    Madd1(d00, _srcZero, w0, body);
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _dstZero, i);
                    }
                    for (; i < size; i += F)
                    {
                        d00 = svdup_n_s32(0);
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                w0 = LoadAs32i(weight + (ky * p.kernelX + kx) * size + i, tail);
                                if (sy < p.srcH && sx < p.srcW)
                                    Madd1(d00, LoadAs32i(src + (sy * p.srcW + sx) * size + i, tail), w0, tail);
                                else
                                    Madd1(d00, _srcZero, w0, tail);
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _dstZero, i, tail);
                    }
                    dst += p.dstC * a.srcE;
                }
            }
        }

        //------------------------------------------------------------------------------------------------

        template<Term8iType term> SIMD_INLINE void QuantizedConvolutionNhwcDepthwiseV0_3x3Edge(
            const uint8_t* src, const svint32_t& srcZero, const ConvParam& p, const AlgParamV0& a, size_t dy, size_t dx,
            const int8_t* weight, const int32_t* bias, const float* norm, const svint32_t& dstZero, uint8_t* dst)
        {
            const size_t F = svcntw(), DF = F * 2, QF = F * 4;
            const svbool_t body = svptrue_b32();
            svint32_t d00, d01, d02, d03, w0, w1, w2, w3;
            size_t size = p.group;
            size_t sizeF = AlignLo(size, F), sizeDF = AlignLo(size, DF), sizeQF = AlignLo(size, QF);
            svbool_t tail = svwhilelt_b32((size_t)0, size - sizeF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                d00 = svdup_n_s32(0);
                d01 = svdup_n_s32(0);
                d02 = svdup_n_s32(0);
                d03 = svdup_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        size_t ow = (ky * p.kernelX + kx) * size + i;
                        w0 = LoadAs32i(weight + ow + 0 * F, body);
                        w1 = LoadAs32i(weight + ow + 1 * F, body);
                        w2 = LoadAs32i(weight + ow + 2 * F, body);
                        w3 = LoadAs32i(weight + ow + 3 * F, body);
                        if (sy < p.srcH && sx < p.srcW)
                        {
                            size_t os = (sy * p.srcW + sx) * size + i;
                            Madd1(d00, LoadAs32i(src + os + 0 * F, body), w0, body);
                            Madd1(d01, LoadAs32i(src + os + 1 * F, body), w1, body);
                            Madd1(d02, LoadAs32i(src + os + 2 * F, body), w2, body);
                            Madd1(d03, LoadAs32i(src + os + 3 * F, body), w3, body);
                        }
                        else
                        {
                            Madd1(d00, srcZero, w0, body);
                            Madd1(d01, srcZero, w1, body);
                            Madd1(d02, srcZero, w2, body);
                            Madd1(d03, srcZero, w3, body);
                        }
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, i + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, i + F * 1);
                Save1<term>(dst, d02, bias, norm, dstZero, i + F * 2);
                Save1<term>(dst, d03, bias, norm, dstZero, i + F * 3);
            }
            for (; i < sizeDF; i += DF)
            {
                d00 = svdup_n_s32(0);
                d01 = svdup_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        size_t ow = (ky * p.kernelX + kx) * size + i;
                        w0 = LoadAs32i(weight + ow + 0 * F, body);
                        w1 = LoadAs32i(weight + ow + 1 * F, body);
                        if (sy < p.srcH && sx < p.srcW)
                        {
                            size_t os = (sy * p.srcW + sx) * size + i;
                            Madd1(d00, LoadAs32i(src + os + 0 * F, body), w0, body);
                            Madd1(d01, LoadAs32i(src + os + 1 * F, body), w1, body);
                        }
                        else
                        {
                            Madd1(d00, srcZero, w0, body);
                            Madd1(d01, srcZero, w1, body);
                        }
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, i + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, i + F * 1);
            }
            for (; i < sizeF; i += F)
            {
                d00 = svdup_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        w0 = LoadAs32i(weight + (ky * p.kernelX + kx) * size + i, body);
                        if (sy < p.srcH && sx < p.srcW)
                            Madd1(d00, LoadAs32i(src + (sy * p.srcW + sx) * size + i, body), w0, body);
                        else
                            Madd1(d00, srcZero, w0, body);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, i);
            }
            for (; i < size; i += F)
            {
                d00 = svdup_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        w0 = LoadAs32i(weight + (ky * p.kernelX + kx) * size + i, tail);
                        if (sy < p.srcH && sx < p.srcW)
                            Madd1(d00, LoadAs32i(src + (sy * p.srcW + sx) * size + i, tail), w0, tail);
                        else
                            Madd1(d00, srcZero, w0, tail);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, i, tail);
            }
        }

        template<Term8iType term> SIMD_INLINE void QuantizedConvolutionNhwcDepthwiseV0_3x3Main1(
            const uint8_t* src, const ConvParam& p, const AlgParamV0& a,
            const int8_t* weight, const int32_t* bias, const float* norm, const svint32_t& dstZero, uint8_t* dst)
        {
            const size_t F = svcntw(), DF = F * 2, QF = F * 4;
            const svbool_t body = svptrue_b32();
            svint32_t d00, d01, d02, d03;
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F), srcCDF = AlignLo(srcC, DF), srcCQF = AlignLo(srcC, QF);
            svbool_t tail = svwhilelt_b32((size_t)0, srcC - srcCF);
            size_t srcS = srcC * p.srcW;
            size_t c = 0;
            for (; c < srcCQF; c += QF)
            {
                d00 = svdup_n_s32(0);
                d01 = svdup_n_s32(0);
                d02 = svdup_n_s32(0);
                d03 = svdup_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
                    {
                        Madd1(d00, LoadAs32i(ps + 0 * F, body), LoadAs32i(pw + 0 * F, body), body);
                        Madd1(d01, LoadAs32i(ps + 1 * F, body), LoadAs32i(pw + 1 * F, body), body);
                        Madd1(d02, LoadAs32i(ps + 2 * F, body), LoadAs32i(pw + 2 * F, body), body);
                        Madd1(d03, LoadAs32i(ps + 3 * F, body), LoadAs32i(pw + 3 * F, body), body);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, c + F * 1);
                Save1<term>(dst, d02, bias, norm, dstZero, c + F * 2);
                Save1<term>(dst, d03, bias, norm, dstZero, c + F * 3);
            }
            for (; c < srcCDF; c += DF)
            {
                d00 = svdup_n_s32(0);
                d01 = svdup_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
                    {
                        Madd1(d00, LoadAs32i(ps + 0 * F, body), LoadAs32i(pw + 0 * F, body), body);
                        Madd1(d01, LoadAs32i(ps + 1 * F, body), LoadAs32i(pw + 1 * F, body), body);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, c + F * 1);
            }
            for (; c < srcCF; c += F)
            {
                d00 = svdup_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
                        Madd1(d00, LoadAs32i(ps, body), LoadAs32i(pw, body), body);
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c);
            }
            for (; c < srcC; c += F)
            {
                d00 = svdup_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
                        Madd1(d00, LoadAs32i(ps, tail), LoadAs32i(pw, tail), tail);
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c, tail);
            }
        }

        template<Term8iType term> SIMD_INLINE void QuantizedConvolutionNhwcDepthwiseV0_3x3Main2(
            const uint8_t* src, const ConvParam& p, const AlgParamV0& a,
            const int8_t* weight, const int32_t* bias, const float* norm, const svint32_t& dstZero, uint8_t* dst)
        {
            const size_t F = svcntw(), DF = F * 2, QF = F * 4;
            const svbool_t body = svptrue_b32();
            svint32_t d00, d01, d02, d03, d10, d11, d12, d13, w0;
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F), srcCDF = AlignLo(srcC, DF), srcCQF = AlignLo(srcC, QF);
            svbool_t tail = svwhilelt_b32((size_t)0, srcC - srcCF);
            size_t srcS = srcC * p.srcW;
            size_t srcX = srcC * p.strideX;
            size_t c = 0;
            for (; c < srcCQF; c += QF)
            {
                d00 = svdup_n_s32(0);
                d01 = svdup_n_s32(0);
                d02 = svdup_n_s32(0);
                d03 = svdup_n_s32(0);
                d10 = svdup_n_s32(0);
                d11 = svdup_n_s32(0);
                d12 = svdup_n_s32(0);
                d13 = svdup_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
                    {
                        w0 = LoadAs32i(pw + 0 * F, body);
                        Madd1(d00, LoadAs32i(ps + 0 * F + 0 * srcX, body), w0, body);
                        Madd1(d10, LoadAs32i(ps + 0 * F + 1 * srcX, body), w0, body);
                        w0 = LoadAs32i(pw + 1 * F, body);
                        Madd1(d01, LoadAs32i(ps + 1 * F + 0 * srcX, body), w0, body);
                        Madd1(d11, LoadAs32i(ps + 1 * F + 1 * srcX, body), w0, body);
                        w0 = LoadAs32i(pw + 2 * F, body);
                        Madd1(d02, LoadAs32i(ps + 2 * F + 0 * srcX, body), w0, body);
                        Madd1(d12, LoadAs32i(ps + 2 * F + 1 * srcX, body), w0, body);
                        w0 = LoadAs32i(pw + 3 * F, body);
                        Madd1(d03, LoadAs32i(ps + 3 * F + 0 * srcX, body), w0, body);
                        Madd1(d13, LoadAs32i(ps + 3 * F + 1 * srcX, body), w0, body);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, c + F * 1);
                Save1<term>(dst, d02, bias, norm, dstZero, c + F * 2);
                Save1<term>(dst, d03, bias, norm, dstZero, c + F * 3);
                Save1<term>(dst + srcC, d10, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst + srcC, d11, bias, norm, dstZero, c + F * 1);
                Save1<term>(dst + srcC, d12, bias, norm, dstZero, c + F * 2);
                Save1<term>(dst + srcC, d13, bias, norm, dstZero, c + F * 3);
            }
            for (; c < srcCDF; c += DF)
            {
                d00 = svdup_n_s32(0);
                d01 = svdup_n_s32(0);
                d10 = svdup_n_s32(0);
                d11 = svdup_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
                    {
                        w0 = LoadAs32i(pw + 0 * F, body);
                        Madd1(d00, LoadAs32i(ps + 0 * F + 0 * srcX, body), w0, body);
                        Madd1(d10, LoadAs32i(ps + 0 * F + 1 * srcX, body), w0, body);
                        w0 = LoadAs32i(pw + 1 * F, body);
                        Madd1(d01, LoadAs32i(ps + 1 * F + 0 * srcX, body), w0, body);
                        Madd1(d11, LoadAs32i(ps + 1 * F + 1 * srcX, body), w0, body);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, c + F * 1);
                Save1<term>(dst + srcC, d10, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst + srcC, d11, bias, norm, dstZero, c + F * 1);
            }
            for (; c < srcCF; c += F)
            {
                d00 = svdup_n_s32(0);
                d10 = svdup_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
                    {
                        w0 = LoadAs32i(pw, body);
                        Madd1(d00, LoadAs32i(ps + 0 * srcX, body), w0, body);
                        Madd1(d10, LoadAs32i(ps + 1 * srcX, body), w0, body);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c);
                Save1<term>(dst + srcC, d10, bias, norm, dstZero, c);
            }
            for (; c < srcC; c += F)
            {
                d00 = svdup_n_s32(0);
                d10 = svdup_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
                    {
                        w0 = LoadAs32i(pw, tail);
                        Madd1(d00, LoadAs32i(ps + 0 * srcX, tail), w0, tail);
                        Madd1(d10, LoadAs32i(ps + 1 * srcX, tail), w0, tail);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c, tail);
                Save1<term>(dst + srcC, d10, bias, norm, dstZero, c, tail);
            }
        }

        template<Term8iType term> void QuantizedConvolutionNhwcDepthwiseV0_3x3(const uint8_t* src, uint32_t srcZero,
            const ConvParam& p, const AlgParamV0& a, const int8_t* weight, const int32_t* bias, const float* norm, uint32_t dstZero, uint8_t* dst)
        {
            svint32_t _srcZero = svdup_n_s32(srcZero);
            svint32_t _dstZero = svdup_n_s32(dstZero);
            size_t srcX = p.srcC * p.strideX;
            size_t dstH = p.dstH - p.padH;
            size_t dstW = p.dstW - p.padW;
            size_t dstC = p.dstC * a.dstE;
            size_t dstW2 = AlignLo(dstW - p.padX, 2) + p.padX;
            size_t dy = 0;
            for (; dy < p.padY; ++dy)
                for (size_t dx = 0; dx < p.dstW; ++dx)
                    QuantizedConvolutionNhwcDepthwiseV0_3x3Edge<term>(src, _srcZero, p, a, dy, dx, weight, bias, norm, _dstZero, dst), dst += dstC;
            for (; dy < dstH; ++dy)
            {
                size_t dx = 0;
                for (; dx < p.padX; ++dx)
                    QuantizedConvolutionNhwcDepthwiseV0_3x3Edge<term>(src, _srcZero, p, a, dy, dx, weight, bias, norm, _dstZero, dst), dst += dstC;
                size_t offset = ((dy * p.strideY - p.padY) * p.srcW + dx * p.strideX - p.padX) * p.srcC;
                for (; dx < dstW2; dx += 2)
                    QuantizedConvolutionNhwcDepthwiseV0_3x3Main2<term>(src + offset, p, a, weight, bias, norm, _dstZero, dst), dst += dstC * 2, offset += srcX * 2;
                for (; dx < dstW; dx += 1)
                    QuantizedConvolutionNhwcDepthwiseV0_3x3Main1<term>(src + offset, p, a, weight, bias, norm, _dstZero, dst), dst += dstC, offset += srcX;
                for (; dx < p.dstW; ++dx)
                    QuantizedConvolutionNhwcDepthwiseV0_3x3Edge<term>(src, _srcZero, p, a, dy, dx, weight, bias, norm, _dstZero, dst), dst += dstC;
            }
            for (; dy < p.dstH; ++dy)
                for (size_t dx = 0; dx < p.dstW; ++dx)
                    QuantizedConvolutionNhwcDepthwiseV0_3x3Edge<term>(src, _srcZero, p, a, dy, dx, weight, bias, norm, _dstZero, dst), dst += dstC;
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term> void SetV0(const ConvParam& p, SynetQuantizedConvolutionNhwcDepthwiseV0::ConvolutionPtr& convolution)
        {
            if (p.IsKernel(3) && p.IsDilation(1))
                convolution = QuantizedConvolutionNhwcDepthwiseV0_3x3<term>;
            else
                convolution = QuantizedConvolutionNhwcDepthwiseV0_Default<term>;
        }

        //------------------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNhwcDepthwiseV0::SynetQuantizedConvolutionNhwcDepthwiseV0(const ConvParam& p)
            : Base::SynetQuantizedConvolutionNhwcDepthwiseV0(p)
        {
            if (p.dstT == SimdTensorData8u)
                SetV0<Term8iLast8u>(p, _convolution);
        }
    }
#endif
}
