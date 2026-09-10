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
#include "Simd/SimdSynetQuantizedActivation.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE) 
    namespace Base
    {
        typedef Base::SynetQuantizedConvolutionNchwGemm::AlgParam AlgParam;

        //-----------------------------------------------------------------------------------------

        static void QuantizedConvolutionNchwGemm_ImgToCol(const uint8_t* src, uint8_t zero, const ConvParam& p, uint8_t* dst)
        {
            size_t dS = p.srcW * p.srcH;
            if (p.IsDilation(1) && p.IsStride(2) && p.IsPad(0) && p.IsKernel(1))
            {
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t dy = 0; dy < p.dstH; ++dy)
                    {
                        const uint8_t* psrc = src + 2 * dy * p.srcW;
                        for (size_t dx = 0, sx = 0; dx < p.dstW; ++dx, sx += 2)
                            *(dst++) = psrc[sx];
                    }
                    src += dS;
                }
            }
            else if (p.IsDilation(1) && p.IsStride(1))
            {
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t ky = 0; ky < p.kernelY; ++ky)
                    {
                        for (size_t kx = 0; kx < p.kernelX; ++kx)
                        {
                            size_t sy = ky - p.padY;
                            for (size_t dy = 0; dy < p.dstH; ++dy, ++sy)
                            {
                                if (sy < p.srcH)
                                {
                                    size_t sx = kx - p.padX, dx = 0;
                                    for (size_t dx = 0; dx < p.dstW; ++dx, ++sx)
                                    {
                                        if (sx < p.srcW)
                                            *(dst++) = src[sy * p.srcW + sx];
                                        else
                                            *(dst++) = zero;
                                    }
                                }
                                else
                                {
                                    for (size_t dx = 0; dx < p.dstW; ++dx)
                                        *(dst++) = zero;
                                }
                            }
                        }
                    }
                    src += dS;
                }
            }
            else
            {
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t ky = 0; ky < p.kernelY; ky++)
                    {
                        for (size_t kx = 0; kx < p.kernelX; kx++)
                        {
                            size_t sy = ky * p.dilationY - p.padY;
                            for (size_t dy = 0; dy < p.dstH; ++dy)
                            {
                                if (sy < p.srcH)
                                {
                                    size_t sx = kx * p.dilationX - p.padX;
                                    for (size_t dx = 0; dx < p.dstW; ++dx)
                                    {
                                        if (sx < p.srcW)
                                            *(dst++) = src[sy * p.srcW + sx];
                                        else
                                            *(dst++) = zero;
                                        sx += p.strideX;
                                    }
                                }
                                else
                                {
                                    for (size_t dx = 0; dx < p.dstW; ++dx)
                                        *(dst++) = zero;
                                }
                                sy += p.strideY;
                            }
                        }
                    }
                    src += dS;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void ReorderMain(const uint8_t* src, size_t dS, size_t K, size_t KH, size_t F, uint8_t* dst)
        {
            size_t k = 0, K4 = K & (~3), KT = K - K4;
            for (; k < K4; k += 4)
            {
                const uint8_t* src0 = src + k * dS, * src1 = src0 + dS, * src2 = src1 + dS, * src3 = src2 + dS;
                for (size_t f = 0; f < F; ++f)
                {
                    *dst++ = src0[f];
                    *dst++ = src1[f];
                    *dst++ = src2[f];
                    *dst++ = src3[f];
                }
            }
            if (KT)
            {
                const uint8_t* src0 = src + k * dS;
                size_t kt = 0;
                for (; kt < KT; kt += 1)
                {
                    for (size_t f = 0; f < F; ++f)
                        dst[f * 4 + kt] = src0[f];
                    src0 += dS;
                }
                for (; kt < 4; kt += 1)
                {
                    for (size_t f = 0; f < F; ++f)
                        dst[f * 4 + kt] = 0;
                    src0 += dS;
                }
                k += 4;
                dst += 4 * F;
            }
            for (; k < KH; k += 4)
            {
                for (size_t f = 0; f < F; ++f)
                {
                    *dst++ = 0;
                    *dst++ = 0;
                    *dst++ = 0;
                    *dst++ = 0;
                }
            }
        }

        SIMD_INLINE void ReorderTail(const uint8_t* src, size_t dS, size_t K, size_t KH, size_t F, uint8_t* dst, size_t tail)
        {
            size_t k = 0, K4 = K & (~3), KT = K - K4, f;
            for (; k < K4; k += 4)
            {
                const uint8_t* src0 = src + k * dS, * src1 = src0 + dS, * src2 = src1 + dS, * src3 = src2 + dS;
                for (f = 0; f < tail; ++f)
                {
                    *dst++ = src0[f];
                    *dst++ = src1[f];
                    *dst++ = src2[f];
                    *dst++ = src3[f];
                }
                for (; f < F; ++f)
                {
                    *dst++ = 0;
                    *dst++ = 0;
                    *dst++ = 0;
                    *dst++ = 0;
                }
            }
            if (KT)
            {
                const uint8_t* src0 = src + k * dS;
                size_t kt = 0, f;
                for (; kt < KT; kt += 1)
                {
                    for (f = 0; f < tail; ++f)
                        dst[f * 4 + kt] = src0[f];
                    for (; f < F; ++f)
                        dst[f * 4 + kt] = 0;
                    src0 += dS;
                }
                for (; kt < 4; kt += 1)
                {
                    for (f = 0; f < F; ++f)
                        dst[f * 4 + kt] = 0;
                    src0 += dS;
                }
                k += 4;
                dst += 4 * F;
            }
            for (; k < KH; k += 4)
            {
                for (size_t f = 0; f < F; ++f)
                {
                    *dst++ = 0;
                    *dst++ = 0;
                    *dst++ = 0;
                    *dst++ = 0;
                }
            }
        }

        static void QuantizedConvolutionNchwGemm_Reorder(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, size_t cBeg, size_t cEnd, uint8_t* dst)
        {
            src += (cBeg * p.srcH + yBeg) * p.srcW;
            size_t F = a.F, N = (yEnd - yBeg) * p.srcW, NF = AlignLo(N, a.F), tail = N - NF, dS = p.srcH * p.srcW;
            size_t K = Simd::Min(cEnd, a.K) - cBeg, KH = AlignHi(K, a.microK);
            for (size_t j = 0; j < NF; j += F, src += F, dst += KH * F)
                ReorderMain(src, dS, K, KH, F, dst);
            if (tail)
                ReorderTail(src, dS, K, KH, F, dst, tail);
        }

        //-----------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNchwGemm::SynetQuantizedConvolutionNchwGemm(const ConvParam& p)
            : SynetQuantizedConvolution(p)
        {
            if (!_is1x1)
                _imgToCol = QuantizedConvolutionNchwGemm_ImgToCol;
            _reorder = QuantizedConvolutionNchwGemm_Reorder;
            _gemm[0] = 0;
            _gemm[1] = 0;
        }

        String SynetQuantizedConvolutionNchwGemm::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NchwGemm";
            return desc.str();
        }

        size_t SynetQuantizedConvolutionNchwGemm::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 0;
            if (!_is1x1)
                size += a.N * a.K * sizeof(uint8_t);
            size += a.bufN * a.bufK * sizeof(uint8_t);
            size += a.bufD * a.bufN * sizeof(int32_t);
            if (a.microK == 64)
                size += 4096;
            return size;
        }

        void SynetQuantizedConvolutionNchwGemm::SetWeight(const int8_t* weight)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            _weight.Resize(a.bufK * a.bufD, true);
            int8_t* dst = _weight.data;
            for (size_t mak = 0; mak < a.bufK; mak += a.macroK)
            {
                size_t macroK = Simd::Min(a.bufK, mak + a.macroK) - mak;
                for (size_t d = 0; d < a.bufD; d += 1)
                {
                    const int8_t* src = weight + d * a.K + mak;
                    for (size_t k = 0; k < macroK; k += 1)
                    {
                        if (d < p.dstC && mak + k < a.K)
                            *(dst++) = src[k];
                        else
                            *(dst++) = 0;
                    }
                }
            }
        }

        bool SynetQuantizedConvolutionNchwGemm::Preferable(const ConvParam& p)
        {
            return p.trans == 0 && p.group == 1 && 1;
        }

        void SynetQuantizedConvolutionNchwGemm::SetAlgParam(size_t F, size_t microD, size_t microN, size_t microK)
        {
            const ConvParam& p = _param;
            AlgParam& a = _alg;
            const int L1 = (int)Base::AlgCacheL1(), L2 = int(Base::AlgCacheL2() * 0.5), L3 = (int)Base::AlgCacheL3();

            a.N = p.dstW * p.dstH;
            a.K = p.srcC * p.kernelY * p.kernelX;
            a.F = F;
            a.microD = microD;
            a.microN = microN;
            a.microK = microK;
            a.bufD = AlignHiAny(p.dstC, a.microD);
            a.bufK = AlignHi(a.K, a.microK);
            a.macroK = Simd::RestrictRange(AlignLo(L1 / a.microD, a.microK), a.microK, a.bufK);
            a.macroH = Simd::RestrictRange(L3 / a.macroK / p.dstW, size_t(1), p.dstH);
            a.macroD = Simd::RestrictRange(AlignLoAny(L2 / a.macroK, a.microD), a.microD, a.bufD);
            a.bufN = AlignHi(a.macroH * p.dstW, a.F);
            a.elem = _elemD;
        }

		void SynetQuantizedConvolutionNchwGemm::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
		{
			const ConvParam& p = _param;
			const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            uint8_t* bufC = _is1x1 ? NULL : Allocate<uint8_t>(buf8, a.N * a.K);
            uint8_t* bufT = Allocate<uint8_t>(buf8, a.bufN * a.macroK);
            int32_t* bufS = Allocate<int32_t>(buf8, a.bufD * a.bufN);
            int32_t* bufB = a.microK == 64 ? Allocate<int32_t>(buf8, 1024) : NULL;
            for (size_t b = 0; b < p.batch; b += 1)
            {
                if (_is1x1)
                    Forward(src, bufT, bufS, bufB, dst);
                else
                {
                    _imgToCol(src, _srcZero[0], p, bufC);
                    Forward(bufC, bufT, bufS, bufB, dst);
                }
                src += _sizeS;
                dst += _sizeD;
            }
		}

        void SynetQuantizedConvolutionNchwGemm::Forward(const uint8_t* src, uint8_t* tmp, int32_t* sum, int32_t* buf, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            float dNorm = 1.0f / _dstScale;
            for (size_t yBeg = 0; yBeg < p.dstH;)
            {
                size_t yEnd = Simd::Min(yBeg + a.macroH, p.dstH);
                for (size_t mak = 0; mak < a.K; mak += a.macroK)
                {
                    size_t macroK = Simd::Min(a.bufK, mak + a.macroK) - mak;
                    _reorder(src, p, a, yBeg, yEnd, mak, mak + macroK, tmp);
                    const int32_t* sBias = _bias.data;
                    const float* sNorm = _norm.data;
                    const float* params = _params.data;
                    int update = mak == 0 ? 0 : 1;
                    for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
                    {
                        size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                        size_t sumOffs = a.macroK < a.bufK ? (dc * p.dstH + yBeg) * AlignHi(p.dstW, a.F) : 0;
                        size_t dstOffs = (dc * p.dstH + yBeg) * p.dstW * _elemD;
                        const int8_t* weight = _weight.data + a.bufD * mak + dc * macroK;
                        if (mak + macroK == a.bufK)
                            _gemm[1](weight, p, a, macroD, yEnd - yBeg, macroK, update, tmp, sBias, sNorm, 
                                _intZero, _intScale, params, dNorm, _dstZero, sum + sumOffs, buf, dst + dstOffs);
                        else
                            _gemm[0](weight, p, a, macroD, yEnd - yBeg, macroK, update, tmp, sBias, sNorm, 
                                _intZero, _intScale, params, dNorm, _dstZero, sum + sumOffs, buf, dst + dstOffs);
                        sBias += macroD;
                        sNorm += macroD;
                        if (p.activation == ::SimdConvolutionActivationPrelu)
                            params += macroD;
                    }
                }
                yBeg = yEnd;
            }
        }
    }
#endif
}
