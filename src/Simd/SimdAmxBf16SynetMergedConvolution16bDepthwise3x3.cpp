/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdSynetMergedConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace AmxBf16
    {
        using AlgParam = Base::SynetMergedConvolution16b::AlgParam;
        using DepthwisePtr = Base::SynetMergedConvolution16b::DepthwiseConvolutionPtr;

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolution3x3Edge2x2(const T* src0,
            const T* src1, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, __mmask16 tail)
        {
            __m512 sum = _mm512_setzero_ps();
            sum = _mm512_fmadd_ps(LoadSrc(src0 + 0 * sX), weight[0], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src0 + 1 * sX), weight[1], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src1 + 0 * sX), weight[3], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src1 + 1 * sX), weight[4], sum);
            Save1<term, type>(dst, NULL, sum, bias, params, tail);
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolution3x3Edge2x3(const T* src0,
            const T* src1, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, __mmask16 tail)
        {
            __m512 sum = _mm512_setzero_ps();
            sum = _mm512_fmadd_ps(LoadSrc(src0 + 0 * sX), weight[0], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src0 + 1 * sX), weight[1], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src0 + 2 * sX), weight[2], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src1 + 0 * sX), weight[3], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src1 + 1 * sX), weight[4], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src1 + 2 * sX), weight[5], sum);
            Save1<term, type>(dst, NULL, sum, bias, params, tail);
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolution3x3Edge3x2(const T* src0,
            const T* src1, const T* src2, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, __mmask16 tail)
        {
            __m512 sum = _mm512_setzero_ps();
            sum = _mm512_fmadd_ps(LoadSrc(src0 + 0 * sX), weight[0], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src0 + 1 * sX), weight[1], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src1 + 0 * sX), weight[3], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src1 + 1 * sX), weight[4], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src2 + 0 * sX), weight[6], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src2 + 1 * sX), weight[7], sum);
            Save1<term, type>(dst, NULL, sum, bias, params, tail);
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolution3x3Main1x1(const T* src0,
            const T* src1, const T* src2, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, __mmask16 tail)
        {
            __m512 sum = _mm512_setzero_ps();
            sum = _mm512_fmadd_ps(LoadSrc(src0 + 0 * sX), weight[0], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src0 + 1 * sX), weight[1], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src0 + 2 * sX), weight[2], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src1 + 0 * sX), weight[3], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src1 + 1 * sX), weight[4], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src1 + 2 * sX), weight[5], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src2 + 0 * sX), weight[6], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src2 + 1 * sX), weight[7], sum);
            sum = _mm512_fmadd_ps(LoadSrc(src2 + 2 * sX), weight[8], sum);
            Save1<term, type>(dst, NULL, sum, bias, params, tail);
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolution3x3Main1x2(const T* src0,
            const T* src1, const T* src2, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, size_t dX, __mmask16 tail)
        {
            __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps(), s0;

            s0 = LoadSrc(src0 + 0 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[0], sum0);
            s0 = LoadSrc(src0 + 1 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[1], sum0);
            sum1 = _mm512_fmadd_ps(s0, weight[0], sum1);
            s0 = LoadSrc(src0 + 2 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[2], sum0);
            sum1 = _mm512_fmadd_ps(s0, weight[1], sum1);
            s0 = LoadSrc(src0 + 3 * sX);
            sum1 = _mm512_fmadd_ps(s0, weight[2], sum1);

            s0 = LoadSrc(src1 + 0 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[3], sum0);
            s0 = LoadSrc(src1 + 1 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[4], sum0);
            sum1 = _mm512_fmadd_ps(s0, weight[3], sum1);
            s0 = LoadSrc(src1 + 2 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[5], sum0);
            sum1 = _mm512_fmadd_ps(s0, weight[4], sum1);
            s0 = LoadSrc(src1 + 3 * sX);
            sum1 = _mm512_fmadd_ps(s0, weight[5], sum1);

            s0 = LoadSrc(src2 + 0 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[6], sum0);
            s0 = LoadSrc(src2 + 1 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[7], sum0);
            sum1 = _mm512_fmadd_ps(s0, weight[6], sum1);
            s0 = LoadSrc(src2 + 2 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[8], sum0);
            sum1 = _mm512_fmadd_ps(s0, weight[7], sum1);
            s0 = LoadSrc(src2 + 3 * sX);
            sum1 = _mm512_fmadd_ps(s0, weight[8], sum1);

            Save1<term, type>(dst + 0 * dX, NULL, sum0, bias, params, tail);
            Save1<term, type>(dst + 1 * dX, NULL, sum1, bias, params, tail);
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolution3x3Main1x4(const T* src0,
            const T* src1, const T* src2, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, size_t dX, __mmask16 tail)
        {
            __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps(), sum3 = _mm512_setzero_ps(), s0;

            s0 = LoadSrc(src0 + 0 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[0], sum0);
            s0 = LoadSrc(src0 + 1 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[1], sum0);
            sum1 = _mm512_fmadd_ps(s0, weight[0], sum1);
            s0 = LoadSrc(src0 + 2 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[2], sum0);
            sum1 = _mm512_fmadd_ps(s0, weight[1], sum1);
            sum2 = _mm512_fmadd_ps(s0, weight[0], sum2);
            s0 = LoadSrc(src0 + 3 * sX);
            sum1 = _mm512_fmadd_ps(s0, weight[2], sum1);
            sum2 = _mm512_fmadd_ps(s0, weight[1], sum2);
            sum3 = _mm512_fmadd_ps(s0, weight[0], sum3);
            s0 = LoadSrc(src0 + 4 * sX);
            sum2 = _mm512_fmadd_ps(s0, weight[2], sum2);
            sum3 = _mm512_fmadd_ps(s0, weight[1], sum3);
            s0 = LoadSrc(src0 + 5 * sX);
            sum3 = _mm512_fmadd_ps(s0, weight[2], sum3);

            s0 = LoadSrc(src1 + 0 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[3], sum0);
            s0 = LoadSrc(src1 + 1 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[4], sum0);
            sum1 = _mm512_fmadd_ps(s0, weight[3], sum1);
            s0 = LoadSrc(src1 + 2 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[5], sum0);
            sum1 = _mm512_fmadd_ps(s0, weight[4], sum1);
            sum2 = _mm512_fmadd_ps(s0, weight[3], sum2);
            s0 = LoadSrc(src1 + 3 * sX);
            sum1 = _mm512_fmadd_ps(s0, weight[5], sum1);
            sum2 = _mm512_fmadd_ps(s0, weight[4], sum2);
            sum3 = _mm512_fmadd_ps(s0, weight[3], sum3);
            s0 = LoadSrc(src1 + 4 * sX);
            sum2 = _mm512_fmadd_ps(s0, weight[5], sum2);
            sum3 = _mm512_fmadd_ps(s0, weight[4], sum3);
            s0 = LoadSrc(src1 + 5 * sX);
            sum3 = _mm512_fmadd_ps(s0, weight[5], sum3);

            s0 = LoadSrc(src2 + 0 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[6], sum0);
            s0 = LoadSrc(src2 + 1 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[7], sum0);
            sum1 = _mm512_fmadd_ps(s0, weight[6], sum1);
            s0 = LoadSrc(src2 + 2 * sX);
            sum0 = _mm512_fmadd_ps(s0, weight[8], sum0);
            sum1 = _mm512_fmadd_ps(s0, weight[7], sum1);
            sum2 = _mm512_fmadd_ps(s0, weight[6], sum2);
            s0 = LoadSrc(src2 + 3 * sX);
            sum1 = _mm512_fmadd_ps(s0, weight[8], sum1);
            sum2 = _mm512_fmadd_ps(s0, weight[7], sum2);
            sum3 = _mm512_fmadd_ps(s0, weight[6], sum3);
            s0 = LoadSrc(src2 + 4 * sX);
            sum2 = _mm512_fmadd_ps(s0, weight[8], sum2);
            sum3 = _mm512_fmadd_ps(s0, weight[7], sum3);
            s0 = LoadSrc(src2 + 5 * sX);
            sum3 = _mm512_fmadd_ps(s0, weight[8], sum3);

            Save1<term, type>(dst + 0 * dX, NULL, sum0, bias, params, tail);
            Save1<term, type>(dst + 1 * dX, NULL, sum1, bias, params, tail);
            Save1<term, type>(dst + 2 * dX, NULL, sum2, bias, params, tail);
            Save1<term, type>(dst + 3 * dX, NULL, sum3, bias, params, tail);
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> void DepthwiseConvolution3x3(const uint8_t* src8, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const T* src = (T*)src8;
            size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW, dstC = maC;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = p.kernelY * p.kernelX * F, ssX = p.strideX * sX, ssX0 = (p.strideX - p.padX) * sX;
            size_t xMainEnd = p.dstW - p.padW, yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;
            size_t xMainEnd2 = AlignLo(xMainEnd - padX, 2) * (p.strideX == 1 ? 1 : 0) + padX;
            size_t xMainEnd4 = AlignLo(xMainEnd - padX, 4) * (p.strideX == 1 ? 1 : 0) + padX;
            size_t dstCe = a.bufH[2] ? AlignHi(dstC, DF) : dstC;

            __m512 _params[2], _bias[1];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            for (size_t c = 0; c < dstCe; c += F)
            {
                __mmask16 tail = TailMask16(dstC - c);
                __m512 _weight[9];
                for (size_t i = 0; i < 9; ++i)
                    _weight[i] = _mm512_loadu_ps(weight + i * F);
                _bias[0] = _mm512_loadu_ps(bias + c);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_loadu_ps(params + c);

                size_t dy = yBeg;
                if (c == dstC)
                {
                    __mmask32 gapMask = a.bufH[2] ? TailMask32(dstCe - dstC) : 0;
                    for (; dy < yEnd; ++dy)
                    {
                        uint8_t* pDst = dst + (dy - dy0) * dY;
                        for (size_t dx = 0; dx < p.dstW; dx++, pDst += dX)
                            SetZero((uint16_t*)pDst, gapMask);
                    }
                    return;
                }
                if (yBeg == 0 && padY)
                {
                    size_t sy = 0, dx = 0;
                    const T* src0 = src + ((sy + 0) & sM) * sY;
                    const T* src1 = src + ((sy + 1) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge2x2<T, term, type>(src0, src1, sX, _weight + 4, _bias, _params, pDst, tail),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
                        DepthwiseConvolution3x3Edge2x3<T, term, type>(src0, src1, sX, _weight + 3, _bias, _params, pDst, tail);
                    if (padW)
                        DepthwiseConvolution3x3Edge2x2<T, term, type>(src0, src1, sX, _weight + 3, _bias, _params, pDst, tail);
                    dy++;
                }
                for (; dy < yMainEnd; ++dy)
                {
                    size_t sy = dy * strideY - padY, dx = 0;
                    const T* src0 = src + ((sy + 0) & sM) * sY;
                    const T* src1 = src + ((sy + 1) & sM) * sY;
                    const T* src2 = src + ((sy + 2) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge3x2<T, term, type>(src0, src1, src2, sX, _weight + 1, _bias, _params, pDst, tail),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0, src2 += ssX0;
                    for (; dx < xMainEnd4; dx += 4, pDst += dX * 4, src0 += ssX * 4, src1 += ssX * 4, src2 += ssX * 4)
                        DepthwiseConvolution3x3Main1x4<T, term, type>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst, dX, tail);
                    for (; dx < xMainEnd2; dx += 2, pDst += dX * 2, src0 += ssX * 2, src1 += ssX * 2, src2 += ssX * 2)
                        DepthwiseConvolution3x3Main1x2<T, term, type>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst, dX, tail);
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX, src2 += ssX)
                        DepthwiseConvolution3x3Main1x1<T, term, type>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst, tail);
                    if (padW)
                        DepthwiseConvolution3x3Edge3x2<T, term, type>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst, tail);
                }
                if (dy < yEnd)
                {
                    size_t sy = dy * strideY - padY, dx = 0;
                    const T* src0 = src + ((sy + 0) & sM) * sY;
                    const T* src1 = src + ((sy + 1) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge2x2<T, term, type>(src0, src1, sX, _weight + 1, _bias, _params, pDst, tail),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
                        DepthwiseConvolution3x3Edge2x3<T, term, type>(src0, src1, sX, _weight + 0, _bias, _params, pDst, tail);
                    if (padW)
                        DepthwiseConvolution3x3Edge2x2<T, term, type>(src0, src1, sX, _weight + 0, _bias, _params, pDst, tail);
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<int H, int I> static void ZeroSrc(__m512 buf[3][H + 2])
        {
            buf[I][0] = _mm512_setzero_ps();
            buf[I][1] = _mm512_setzero_ps();
            if (H > 0) buf[I][2] = _mm512_setzero_ps();
            if (H > 1) buf[I][3] = _mm512_setzero_ps();
            if (H > 2) buf[I][4] = _mm512_setzero_ps();
            if (H > 3) buf[I][5] = _mm512_setzero_ps();
        }

        template<typename T, int H, int I> static void LoadSrc(const T* s0, const T* s1, const T* s2, const T* s3, const T* s4, const T* s5, 
            size_t offs, __mmask16 mask0, __mmask16 mask1, __mmask16 mask2, __m512 buf[3][H + 2])
        {
            buf[I][0] = LoadSrc(s0 + offs, mask0);
            buf[I][1] = LoadSrc(s1 + offs, mask1);
            if (H > 0) buf[I][2] = LoadSrc(s2 + offs, H == 1 ? mask2 : mask1);
            if (H > 1) buf[I][3] = LoadSrc(s3 + offs, H == 2 ? mask2 : mask1);
            if (H > 2) buf[I][4] = LoadSrc(s4 + offs, H == 3 ? mask2 : mask1);
            if (H > 3) buf[I][5] = LoadSrc(s5 + offs, mask2);
        }

        template<int H, int I0, int I1, int I2> static void Convolution3x3(__m512 src[3][H + 2], const __m512* weight, __m512 dst[H])
        {
            if (H > 0) dst[0] = _mm512_fmadd_ps(src[I0][0], weight[0], dst[0]);
            if (H > 1) dst[1] = _mm512_fmadd_ps(src[I0][1], weight[0], dst[1]);
            if (H > 2) dst[2] = _mm512_fmadd_ps(src[I0][2], weight[0], dst[2]);
            if (H > 3) dst[3] = _mm512_fmadd_ps(src[I0][3], weight[0], dst[3]);

            if (H > 0) dst[0] = _mm512_fmadd_ps(src[I1][0], weight[1], dst[0]);
            if (H > 1) dst[1] = _mm512_fmadd_ps(src[I1][1], weight[1], dst[1]);
            if (H > 2) dst[2] = _mm512_fmadd_ps(src[I1][2], weight[1], dst[2]);
            if (H > 3) dst[3] = _mm512_fmadd_ps(src[I1][3], weight[1], dst[3]);

            if (H > 0) dst[0] = _mm512_fmadd_ps(src[I2][0], weight[2], dst[0]);
            if (H > 1) dst[1] = _mm512_fmadd_ps(src[I2][1], weight[2], dst[1]);
            if (H > 2) dst[2] = _mm512_fmadd_ps(src[I2][2], weight[2], dst[2]);
            if (H > 3) dst[3] = _mm512_fmadd_ps(src[I2][3], weight[2], dst[3]);


            if (H > 0) dst[0] = _mm512_fmadd_ps(src[I0][1], weight[3], dst[0]);
            if (H > 1) dst[1] = _mm512_fmadd_ps(src[I0][2], weight[3], dst[1]);
            if (H > 2) dst[2] = _mm512_fmadd_ps(src[I0][3], weight[3], dst[2]);
            if (H > 3) dst[3] = _mm512_fmadd_ps(src[I0][4], weight[3], dst[3]);

            if (H > 0) dst[0] = _mm512_fmadd_ps(src[I1][1], weight[4], dst[0]);
            if (H > 1) dst[1] = _mm512_fmadd_ps(src[I1][2], weight[4], dst[1]);
            if (H > 2) dst[2] = _mm512_fmadd_ps(src[I1][3], weight[4], dst[2]);
            if (H > 3) dst[3] = _mm512_fmadd_ps(src[I1][4], weight[4], dst[3]);

            if (H > 0) dst[0] = _mm512_fmadd_ps(src[I2][1], weight[5], dst[0]);
            if (H > 1) dst[1] = _mm512_fmadd_ps(src[I2][2], weight[5], dst[1]);
            if (H > 2) dst[2] = _mm512_fmadd_ps(src[I2][3], weight[5], dst[2]);
            if (H > 3) dst[3] = _mm512_fmadd_ps(src[I2][4], weight[5], dst[3]);


            if (H > 0) dst[0] = _mm512_fmadd_ps(src[I0][2], weight[6], dst[0]);
            if (H > 1) dst[1] = _mm512_fmadd_ps(src[I0][3], weight[6], dst[1]);
            if (H > 2) dst[2] = _mm512_fmadd_ps(src[I0][4], weight[6], dst[2]);
            if (H > 3) dst[3] = _mm512_fmadd_ps(src[I0][5], weight[6], dst[3]);

            if (H > 0) dst[0] = _mm512_fmadd_ps(src[I1][2], weight[7], dst[0]);
            if (H > 1) dst[1] = _mm512_fmadd_ps(src[I1][3], weight[7], dst[1]);
            if (H > 2) dst[2] = _mm512_fmadd_ps(src[I1][4], weight[7], dst[2]);
            if (H > 3) dst[3] = _mm512_fmadd_ps(src[I1][5], weight[7], dst[3]);

            if (H > 0) dst[0] = _mm512_fmadd_ps(src[I2][2], weight[8], dst[0]);
            if (H > 1) dst[1] = _mm512_fmadd_ps(src[I2][3], weight[8], dst[1]);
            if (H > 2) dst[2] = _mm512_fmadd_ps(src[I2][4], weight[8], dst[2]);
            if (H > 3) dst[3] = _mm512_fmadd_ps(src[I2][5], weight[8], dst[3]);
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, int H> static void DepthwiseConvolution3x3xH(const T* src, size_t dstH, size_t dstW,
            size_t yB, size_t sM, size_t sY, size_t sX, __mmask16 tailS, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, size_t dY, size_t dX, __mmask32 tailD)
        {
            const T* src0 = src + ((yB - 1) & sM) * sY;
            const T* src1 = src + ((yB + 0) & sM) * sY;
            const T* src2 = src + ((yB + 1) & sM) * sY;
            const T* src3 = src + ((yB + 2) & sM) * sY;
            const T* src4 = src + ((yB + 3) & sM) * sY;
            const T* src5 = src + ((yB + 4) & sM) * sY;
            size_t endW = dstW - 1;
            __m512 s[3][H + 2], d[H];
            __mmask16 mask0 = yB == 0 ? 0 : tailS;
            __mmask16 mask2 = yB + H == dstH ? 0 : tailS;
            ZeroSrc<H, 0>(s);
            LoadSrc<T, H, 1>(src0, src1, src2, src3, src4, src5, 0, mask0, tailS, mask2, s);
            for (size_t dx = 0, offs = sX; dx < dstW; dx += 1, offs += sX)
            {
                if (H > 0) d[0] = _mm512_setzero_ps();
                if (H > 1) d[1] = _mm512_setzero_ps();
                if (H > 2) d[2] = _mm512_setzero_ps();
                if (H > 3) d[3] = _mm512_setzero_ps();
                switch (dx % 3)
                {
                case 0:
                {
                    if (dx == endW)
                        ZeroSrc<H, 2>(s);
                    else
                        LoadSrc<T, H, 2>(src0, src1, src2, src3, src4, src5, offs, mask0, tailS, mask2, s);
                    Convolution3x3<H, 0, 1, 2>(s, weight, d);
                    break;
                }
                case 1:
                {
                    if (dx == endW)
                        ZeroSrc<H, 0>(s);
                    else
                        LoadSrc<T, H, 0>(src0, src1, src2, src3, src4, src5, offs, mask0, tailS, mask2, s);
                    Convolution3x3<H, 1, 2, 0>(s, weight, d);
                    break;
                }
                case 2:
                {
                    if (dx == endW)
                        ZeroSrc<H, 1>(s);
                    else
                        LoadSrc<T, H, 1>(src0, src1, src2, src3, src4, src5, offs, mask0, tailS, mask2, s);
                    Convolution3x3<H, 2, 0, 1>(s, weight, d);
                }
                    break;
                }
                if (H > 0) Save1<term, type>(dst + 0 * dY, 0, d[0], bias, params, tailD);
                if (H > 1) Save1<term, type>(dst + 1 * dY, 0, d[1], bias, params, tailD);
                if (H > 2) Save1<term, type>(dst + 2 * dY, 0, d[2], bias, params, tailD);
                if (H > 3) Save1<term, type>(dst + 3 * dY, 0, d[3], bias, params, tailD);
                dst += dX;
            }
        }

        template<typename T> using DepthwiseConvolution3x3xH_Ptr = void (*)(const T * src, size_t dstH, size_t dstW, size_t yB, size_t sM, size_t sY, size_t sX, __mmask16 tailS, 
            const __m512 * weight, const __m512 * bias, const __m512 * params, uint8_t * dst, size_t dY, size_t dX, __mmask32 tailD);

        template<typename T, Term16bType term, SimdConvolutionActivationType type> DepthwiseConvolution3x3xH_Ptr<T> GetDepthwiseConvolution3x3xH(size_t H)
        {
            switch (H)
            {
            case 1: return DepthwiseConvolution3x3xH<T, term, type, 1>;
            case 2: return DepthwiseConvolution3x3xH<T, term, type, 2>;
            case 3: return DepthwiseConvolution3x3xH<T, term, type, 3>;
            case 4: return DepthwiseConvolution3x3xH<T, term, type, 4>;
            default:
                return NULL;
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> static void DepthwiseConvolution3x3_V2(const uint8_t* src8,
            const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            assert(p.IsKernel(3) && p.IsPad(1) && p.IsStride(1) && p.IsDilation(1));
            const T* src = (T*)src8;
            size_t N = 3, M = (yEnd - yBeg) % N, yBody = AlignLoAny(yEnd - yBeg, N) + yBeg;
            DepthwiseConvolution3x3xH_Ptr<T> body = GetDepthwiseConvolution3x3xH<T, term, type>(N);
            DepthwiseConvolution3x3xH_Ptr<T> tail = GetDepthwiseConvolution3x3xH<T, term, type>(M);
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW, dstC = maC;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = 9 * F, dstCF = AlignLo(dstC, F), dstCe = (a.bufH[2] ? AlignHi(dstC, DF) : dstC);

            __m512 _params[2], _bias[1], _weight[9];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            for (size_t dc = 0; dc < dstCe; dc += F)
            {
                for (size_t i = 0; i < 9; ++i)
                    _weight[i] = _mm512_loadu_ps(weight + i * F);
                _bias[0] = _mm512_loadu_ps(bias + dc);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_loadu_ps(params + dc);
                __mmask16 tailS = TailMask16(dstC - dc);
                __mmask32 tailD = (dc == dstCF && a.bufH[2]) ? TailMask32(dstCe - dstCF) : tailS;
                size_t dy = yBeg;
                for (; dy < yBody; dy += N)
                    body(src, p.dstH, p.dstW, dy, sM, sY, sX, tailS, _weight, _bias, _params, dst + (dy - dy0) * dY, dY, dX, tailD);
                if(M)
                    tail(src, p.dstH, p.dstW, dy, sM, sY, sX, tailS, _weight, _bias, _params, dst + (dy - dy0) * dY, dY, dX, tailD);
                src += sD;
                dst += dD;
                weight += wD;
            }
        }


        //-------------------------------------------------------------------------------------------------

        static SIMD_INLINE bool Preferable_k3p1d1s1w8(const ConvParam& p)
        {
            return p.IsKernel(3) && p.IsPad(1) && p.IsStride(1) && p.IsDilation(1) &&
                (p.srcW >= 8 && (p.srcW % 8 == 0 || p.srcW % 8 >= 6)/*&& AlignHiAny(p.srcW, 8) < AlignHiAny(p.srcW, 6) * 1.2*/);
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> static void DepthwiseConvolution_k3p1d1s1w8(const uint8_t* src8,
            const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            assert(p.IsKernel(3) && p.IsPad(1) && p.IsStride(1) && p.IsDilation(1) && p.srcW >= 8);
            const T* src = (T*)src8;
            size_t srcH = p.srcH, srcW = p.srcW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW, dstC = maC;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = 9 * F, dstCF = AlignLo(dstC, F), dstW = p.dstW, endW = dstW - 8;
            size_t dstCe = a.bufH[2] ? AlignHi(dstC, DF) : dstC;

            __m512 s0, s1, w0, w1, w2, d0, d1, d2, d3, d4, d5, d6, d7;

            __m512 _params[2], _bias[1];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            for (size_t dc = 0; dc < dstCe; dc += F)
            {
                _bias[0] = _mm512_loadu_ps(bias + dc);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_loadu_ps(params + dc);
                __mmask16 tailS = TailMask16(dstC - dc);
                __mmask32 tailC = (dc == dstCF && a.bufH[2]) ? TailMask32(dstCe - dstCF) : tailS;
                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    for (size_t dx = 0;; dx += Min<size_t>(8, endW - dx))
                    {
                        d0 = _mm512_setzero_ps();
                        d1 = _mm512_setzero_ps();
                        d2 = _mm512_setzero_ps();
                        d3 = _mm512_setzero_ps();
                        d4 = _mm512_setzero_ps();
                        d5 = _mm512_setzero_ps();
                        d6 = _mm512_setzero_ps();
                        d7 = _mm512_setzero_ps();
                        __mmask16 tailS0 = dx == 0 ? 0 : tailS;
                        __mmask16 tailS1 = dx == endW ? 0 : tailS;
                        for (size_t ky = 0; ky < 3; ++ky)
                        {
                            size_t sy = dy + ky - 1;
                            const T* ps = src + (sy & sM) * sY + (dx - 1) * sX;
                            const float* pw = weight + ky * 3 * F;
                            if (sy < srcH)
                            {
                                w0 = _mm512_maskz_loadu_ps(tailS, pw + 0 * F);
                                s0 = LoadSrc(ps + 0 * sX, tailS0);
                                d0 = _mm512_fmadd_ps(s0, w0, d0);

                                w1 = _mm512_maskz_loadu_ps(tailS, pw + 1 * F);
                                s1 = LoadSrc(ps + 1 * sX, tailS);
                                d0 = _mm512_fmadd_ps(s1, w1, d0);
                                d1 = _mm512_fmadd_ps(s1, w0, d1);

                                s0 = LoadSrc(ps + 2 * sX, tailS);
                                w2 = _mm512_maskz_loadu_ps(tailS, pw + 2 * F);
                                d0 = _mm512_fmadd_ps(s0, w2, d0);
                                d1 = _mm512_fmadd_ps(s0, w1, d1);
                                d2 = _mm512_fmadd_ps(s0, w0, d2);

                                s1 = LoadSrc(ps + 3 * sX, tailS);
                                d1 = _mm512_fmadd_ps(s1, w2, d1);
                                d2 = _mm512_fmadd_ps(s1, w1, d2);
                                d3 = _mm512_fmadd_ps(s1, w0, d3);

                                s0 = LoadSrc(ps + 4 * sX, tailS);
                                d2 = _mm512_fmadd_ps(s0, w2, d2);
                                d3 = _mm512_fmadd_ps(s0, w1, d3);
                                d4 = _mm512_fmadd_ps(s0, w0, d4);

                                s1 = LoadSrc(ps + 5 * sX, tailS);
                                d3 = _mm512_fmadd_ps(s1, w2, d3);
                                d4 = _mm512_fmadd_ps(s1, w1, d4);
                                d5 = _mm512_fmadd_ps(s1, w0, d5);

                                s0 = LoadSrc(ps + 6 * sX, tailS);
                                d4 = _mm512_fmadd_ps(s0, w2, d4);
                                d5 = _mm512_fmadd_ps(s0, w1, d5);
                                d6 = _mm512_fmadd_ps(s0, w0, d6);

                                s1 = LoadSrc(ps + 7 * sX, tailS);
                                d5 = _mm512_fmadd_ps(s1, w2, d5);
                                d6 = _mm512_fmadd_ps(s1, w1, d6);
                                d7 = _mm512_fmadd_ps(s1, w0, d7);

                                s0 = LoadSrc(ps + 8 * sX, tailS);
                                d6 = _mm512_fmadd_ps(s0, w2, d6);
                                d7 = _mm512_fmadd_ps(s0, w1, d7);

                                s1 = LoadSrc(ps + 9 * sX, tailS1);
                                d7 = _mm512_fmadd_ps(s1, w2, d7);
                            }
                        }
                        uint8_t* pd = dst + (dy - dy0) * dY + dx * dX;
                        Save1<term, type>(pd + 0 * dX, dD, d0, _bias, _params, tailC);
                        Save1<term, type>(pd + 1 * dX, dD, d1, _bias, _params, tailC);
                        Save1<term, type>(pd + 2 * dX, dD, d2, _bias, _params, tailC);
                        Save1<term, type>(pd + 3 * dX, dD, d3, _bias, _params, tailC);
                        Save1<term, type>(pd + 4 * dX, dD, d4, _bias, _params, tailC);
                        Save1<term, type>(pd + 5 * dX, dD, d5, _bias, _params, tailC);
                        Save1<term, type>(pd + 6 * dX, dD, d6, _bias, _params, tailC);
                        Save1<term, type>(pd + 7 * dX, dD, d7, _bias, _params, tailC);
                        if (dx == endW)
                            break;
                    }
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type> static bool SetDepthwise3x3(const ConvParam& p, DepthwisePtr& depthwise)
        {
            if (Preferable_k3p1d1s1w8(p))
            {
                depthwise = DepthwiseConvolution_k3p1d1s1w8<T, term, type>;
                return true;
            }
            else if (IsKernel(p, 3) && IsDilation(p, 1) && IsStride(p, 1))
            {
                depthwise = DepthwiseConvolution3x3_V2<T, term, type>;
                return true;
            }            
            else if (IsKernel(p, 3) && IsDilation(p, 1) && Aligned(p.dstC, F))
            {
                depthwise = DepthwiseConvolution3x3<T, term, type>;
                return true;
            }
            else
                return false;
        }

        template<typename T, SimdConvolutionActivationType type> static bool SetDepthwise3x3(const ConvParam& p, DepthwisePtr& depthwise)
        {
            return (p.dstT == SimdTensorData32f ? SetDepthwise3x3<T, Term16bLast32f, type>(p, depthwise) : SetDepthwise3x3<T, Term16bLast16b, type>(p, depthwise));
        }

        template<SimdConvolutionActivationType type> static bool SetDepthwise3x3(const ConvParam& p, DepthwisePtr& depthwise)
        {
            return (p.srcT == SimdTensorData16b ? SetDepthwise3x3<uint16_t, type>(p, depthwise) : SetDepthwise3x3<float, type>(p, depthwise));
        }

        bool SetDepthwise3x3(const ConvParam& p, DepthwisePtr& depthwise)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: return SetDepthwise3x3<SimdConvolutionActivationRestrictRange>(p, depthwise);
            case SimdConvolutionActivationRelu: return SetDepthwise3x3<SimdConvolutionActivationRestrictRange>(p, depthwise);
            case SimdConvolutionActivationLeakyRelu: return SetDepthwise3x3<SimdConvolutionActivationPrelu>(p, depthwise);
            case SimdConvolutionActivationRestrictRange: return SetDepthwise3x3<SimdConvolutionActivationRestrictRange>(p, depthwise);
            case SimdConvolutionActivationPrelu: return SetDepthwise3x3<SimdConvolutionActivationPrelu>(p, depthwise);
            case SimdConvolutionActivationElu: return SetDepthwise3x3<SimdConvolutionActivationElu>(p, depthwise);
            case SimdConvolutionActivationHswish: return SetDepthwise3x3<SimdConvolutionActivationHswish>(p, depthwise);
            case SimdConvolutionActivationMish: return SetDepthwise3x3<SimdConvolutionActivationMish>(p, depthwise);
            case SimdConvolutionActivationHardSigmoid: return SetDepthwise3x3<SimdConvolutionActivationHardSigmoid>(p, depthwise);
            case SimdConvolutionActivationSwish: return SetDepthwise3x3<SimdConvolutionActivationSwish>(p, depthwise);
            case SimdConvolutionActivationGelu: return SetDepthwise3x3<SimdConvolutionActivationGelu>(p, depthwise);
            default:
                return false;
            }
        }
    }
#endif
}
