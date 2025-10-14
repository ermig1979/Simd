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
#include "Simd/SimdSynetQuantizedConvolution.h"
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
        SynetQuantizedConvolutionNhwcDepthwiseV0::SynetQuantizedConvolutionNhwcDepthwiseV0(const ConvParam& p)
            : SynetQuantizedConvolution(p)
        {
            _convolution = 0;
            AlgParam& a = _alg;
            a.srcE = (int32_t)_elemS;
            a.dstE = (int32_t)_elemD;
        }

        String SynetQuantizedConvolutionNhwcDepthwiseV0::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NhwcDepthwiseV0";
            return desc.str();
        }

        void SynetQuantizedConvolutionNhwcDepthwiseV0::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            for (size_t b = 0; b < p.batch; b += _merge)
            {
                _convolution(src, _srcZero[0], p, a, _weight.data, _bias.data, _norm.data, _dstZero, dst);
                src += _sizeS * _elemS;
                dst += _sizeD * _elemD;
            }
        }

        void SynetQuantizedConvolutionNhwcDepthwiseV0::SetWeight(const int8_t* weight)
        {
            const ConvParam& p = _param;
            _weight.Resize(p.kernelY * p.kernelX * p.srcC / p.group * p.dstC);
            _weight.Assign(weight, _weight.size);
        }

        bool SynetQuantizedConvolutionNhwcDepthwiseV0::Preferable(const ConvParam& p, size_t F)
        {
            return p.trans != 0 && p.IsDepthwise() && p.group >= F && p.activation == SimdConvolutionActivationIdentity;
        }

        //------------------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNhwcDepthwiseV1::SynetQuantizedConvolutionNhwcDepthwiseV1(const ConvParam& p)
            : SynetQuantizedConvolution(p)
        {
            _preprocess = 0;
            _convolution = 0;
        }

        String SynetQuantizedConvolutionNhwcDepthwiseV1::Desc() const
        {
            const AlgParam& a = _alg;
            std::stringstream desc;
            desc << Ext() << "::NhwcDepthwiseV1-" << a.reorderType;
            return desc.str();
        }

        size_t SynetQuantizedConvolutionNhwcDepthwiseV1::InternalBufferSize() const
        {
            return SynetQuantizedConvolution::InternalBufferSize() + _weight32i.RawSize();
        }

        size_t SynetQuantizedConvolutionNhwcDepthwiseV1::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 0;
            size += a.bufC * a.bufH * a.bufW * sizeof(int32_t);
            return size;
        }

        void SynetQuantizedConvolutionNhwcDepthwiseV1::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            int32_t* buf = Allocate<int32_t>(buf8, a.bufC * a.bufH * a.bufW);
            for (size_t b = 0; b < p.batch; b += 1)
            {
                for (size_t yBeg = 0; yBeg < p.dstH;)
                {
                    size_t yEnd = Simd::Min(yBeg + a.stepH, p.dstH);
                    _preprocess(src, _srcZero[0], p, a, yBeg, yEnd, buf);
                    _convolution(buf, p, a, _weight32i.data, _bias.data, _norm.data, yBeg, yEnd, _dstZero, dst);
                    yBeg = yEnd;
                }
                src += _sizeS * _elemS;
                dst += _sizeD * _elemD;
            }
        }

        void SynetQuantizedConvolutionNhwcDepthwiseV1::SetAlgParam(size_t F)
        {
            const ConvParam& p = _param;
            AlgParam& a = _alg;
            a.srcE = (int32_t)_elemS;
            a.dstE = (int32_t)_elemD;
            a.F = F;
            a.bufC = AlignHi(p.srcC, F);
            a.bufW = p.srcW + p.padX + p.padW;
            a.bufH = Pow2Hi(p.kernelY);
            a.reorderType = p.IsKernel(3) ? 1 : 0;
            a.stepH = (F == 16) ? Simd::Max<size_t>(1, (a.bufH - p.kernelY - 1) / p.strideY) : 1;
        }

        void SynetQuantizedConvolutionNhwcDepthwiseV1::SetWeight(const int8_t* src)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            size_t K = p.kernelY * p.kernelX, C = p.srcC, F = a.F;
            _weight32i.Resize(K * a.bufC);
            int32_t* dst = _weight32i.data;
            if (a.reorderType == 0)
            {
                for (size_t k = 0; k < K; ++k)
                {
                    size_t c = 0;
                    for (; c < C; ++c)
                        dst[c] = src[c];
                    for (; c < a.bufC; ++c)
                        dst[c] = 0;
                    src += p.srcC;
                    dst += a.bufC;
                }
            }
            else if (a.reorderType == 1)
            {
                for (size_t c = 0; c < C; c += F)
                {
                    for (size_t k = 0; k < K; ++k)
                    {
                        for (size_t i = 0; i < F; ++i)
                        {
                            if (c + i < C)
                                *dst++ = src[k * C + c + i];
                            else
                                *dst++ = 0;
                        }
                    }
                }
            }
            else
                assert(0);
        }

        bool SynetQuantizedConvolutionNhwcDepthwiseV1::Preferable(const ConvParam& p, size_t F)
        {
            return p.trans != 0 && p.IsDepthwise() && p.IsDilation(1) && p.group >= F && p.activation == SimdConvolutionActivationIdentity;
        }

        //------------------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNhwcDepthwiseV2::SynetQuantizedConvolutionNhwcDepthwiseV2(const ConvParam& p)
            : SynetQuantizedConvolution(p)
        {
            _preprocess = 0;
            _convolution = 0;
        }

        String SynetQuantizedConvolutionNhwcDepthwiseV2::Desc() const
        {
            const AlgParam& a = _alg;
            std::stringstream desc;
            desc << Ext() << "::NhwcDepthwiseV2-" << a.reorderType;
            return desc.str();
        }

        size_t SynetQuantizedConvolutionNhwcDepthwiseV2::InternalBufferSize() const
        {
            return SynetQuantizedConvolution::InternalBufferSize() + _weight16i.RawSize();
        }

        size_t SynetQuantizedConvolutionNhwcDepthwiseV2::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 0;
            size += a.bufC * a.bufH * a.bufW * sizeof(int16_t);
            return size;
        }

        void SynetQuantizedConvolutionNhwcDepthwiseV2::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            int16_t* buf = Allocate<int16_t>(buf8, a.bufR * a.bufH);
            float dNorm = 1.0f / _dstScale;
            if (_srcZero.size != a.bufR)
            {
                uint8_t zero = _srcZero[0];
                _srcZero.Resize(a.bufR);
                memset(_srcZero.data, zero, _srcZero.size);
            }
            for (size_t b = 0; b < p.batch; b += 1)
            {
                for (size_t yBeg = 0; yBeg < p.dstH;)
                {
                    size_t yEnd = Simd::Min(yBeg + a.stepH, p.dstH);
                    _preprocess(src, _srcZero.data, p, a, yBeg, yEnd, buf);
                    _convolution(buf, p, a, yBeg, yEnd, _weight16i.data, _bias.data, _norm.data, _intZero, _intScale, _params.data, dNorm, _dstZero, dst);
                    yBeg = yEnd;
                }
                src += _sizeS * _elemS;
                dst += _sizeD * _elemD;
            }
        }

        void SynetQuantizedConvolutionNhwcDepthwiseV2::SetAlgParam(size_t F)
        {
            const ConvParam& p = _param;
            AlgParam& a = _alg;
            a.srcE = (int32_t)_elemS;
            a.dstE = (int32_t)_elemD;
            a.F = F;
            a.bufC = AlignHi(p.srcC, F);
            a.bufW = p.srcW + p.padX + p.padW;
            a.bufR = a.bufW * a.bufC;
            a.bufH = Pow2Hi(AlignHi(p.kernelY, 2));
            a.stepW = p.kernelX * AlignHi(p.kernelY + 1, 2);
            a.sizeW = a.stepW * a.bufC;
            a.stepH = 2 / p.strideY;
            a.reorderType = 1;// p.IsKernel(3) ? 1 : 0;
        }

        void SynetQuantizedConvolutionNhwcDepthwiseV2::SetWeight(const int8_t* src)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            size_t Y = p.kernelY, X = p.kernelX, C = p.srcC, F = a.F, B = a.bufC;
            _weight16i.Resize(a.sizeW * 2);
            int16_t* dstE = _weight16i.data, *dstO = dstE + a.sizeW;
            if (a.reorderType == 0)
            {
                for (size_t y = 0; y < Y + 1; y += 2)
                {
                    const int8_t* src0 = src - 1 * X * C;
                    const int8_t* src1 = src + 0 * X * C;
                    const int8_t* src2 = src + 1 * X * C;
                    for (size_t x = 0; x < X; ++x)
                    {
                        for (size_t c = 0; c < B; ++c)
                        {
                            *dstE++ = (c < C) ? src1[c] : 0;
                            *dstE++ = (c < C && y + 1 < Y) ? src2[c] : 0;
                            *dstO++ = (c < C && y > 0) ? src0[c] : 0;
                            *dstO++ = (c < C && y < Y) ? src1[c] : 0;
                        }
                        src0 += C;
                        src1 += C;
                        src2 += C;
                    }
                    src += 2 * X * C;
                }
            }
            else if (a.reorderType == 1)
            {
                for (size_t c = 0; c < C; c += F)
                {
                    for (size_t y = 0; y < Y + 1; y += 2)
                    {
                        const int8_t* src0 = src + (y - 1) * X * C + c;
                        const int8_t* src1 = src + (y + 0) * X * C + c;
                        const int8_t* src2 = src + (y + 1) * X * C + c;
                        for (size_t x = 0; x < X; ++x)
                        {
                            for (size_t i = 0; i < F; ++i)
                            {
                                *dstE++ = (i + c < C) ? src1[i] : 0;
                                *dstE++ = (i + c < C && y + 1 < Y) ? src2[i] : 0;
                                *dstO++ = (i + c < C && y > 0) ? src0[i] : 0;
                                *dstO++ = (i + c < C && y < Y) ? src1[i] : 0;
                            }
                            src0 += C;
                            src1 += C;
                            src2 += C;
                        }
                    }
                }
            }
            else
                assert(0);
        }

        bool SynetQuantizedConvolutionNhwcDepthwiseV2::Preferable(const ConvParam& p, size_t F)
        {
            return p.trans != 0 && p.IsDepthwise() && p.IsDilation(1) && p.group >= F 
                && (p.IsStride(1) || p.IsStride(2))
                && (p.IsKernel(3) || p.IsKernel(5) || p.IsKernel(7));
        }

        //------------------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNhwcDepthwiseV3::SynetQuantizedConvolutionNhwcDepthwiseV3(const ConvParam& p)
            : SynetQuantizedConvolution(p)
        {
            _preprocess = 0;
            _convolution = 0;
        }

        String SynetQuantizedConvolutionNhwcDepthwiseV3::Desc() const
        {
            const AlgParam& a = _alg;
            std::stringstream desc;
            desc << Ext() << "::NhwcDepthwiseV3-" << a.reorderType;

            return desc.str();
        }

        size_t SynetQuantizedConvolutionNhwcDepthwiseV3::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 0;
            size += a.bufC * a.bufH * a.bufW * 2;
            return size;
        }

        void SynetQuantizedConvolutionNhwcDepthwiseV3::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            uint8_t* buf = Allocate<uint8_t>(buf8, a.bufR * a.bufH * 2);
            float dNorm = 1.0f / _dstScale;
            if (_srcZero.size != a.bufR)
            {
                uint8_t zero = _srcZero[0];
                _srcZero.Resize(a.bufR);
                memset(_srcZero.data, zero, _srcZero.size);
            }
            for (size_t b = 0; b < p.batch; b += 1)
            {
                for (size_t yBeg = 0; yBeg < p.dstH;)
                {
                    size_t yEnd = Simd::Min(yBeg + a.stepH, p.dstH);
                    _preprocess(src, _srcZero.data, p, a, yBeg, yEnd, buf);
                    _convolution(buf, p, a, yBeg, yEnd, _weight.data, _bias.data, _norm.data, _intZero, _intScale, _params.data, dNorm, _dstZero, dst);
                    yBeg = yEnd;
                }
                src += _sizeS * _elemS;
                dst += _sizeD * _elemD;
            }
        }

        void SynetQuantizedConvolutionNhwcDepthwiseV3::SetAlgParam(size_t F)
        {
            const ConvParam& p = _param;
            AlgParam& a = _alg;
            a.srcE = (int32_t)_elemS;
            a.dstE = (int32_t)_elemD;
            a.F = F;
            a.bufC = AlignHi(p.srcC, F);
            a.bufW = p.srcW + p.padX + p.padW;
            a.bufR = a.bufW * a.bufC;
            a.bufH = Pow2Hi(AlignHi(p.kernelY, 4));
            a.stepW = p.kernelX * AlignHi(p.kernelY + 1, 4);
            a.sizeW = a.stepW * a.bufC;
            a.stepH = 2 / p.strideY;
            a.reorderType = 1;// p.IsKernel(3) ? 1 : 0;
        }

        void SynetQuantizedConvolutionNhwcDepthwiseV3::SetWeight(const int8_t* src)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            size_t Y = p.kernelY, X = p.kernelX, C = p.srcC, F = a.F, B = a.bufC;
            _weight.Resize(a.sizeW * 2);
            int8_t* dstE = _weight.data, * dstO = dstE + a.sizeW;
            if (a.reorderType == 0)
            {
                for (size_t y = 0; y < Y + 1; y += 4)
                {
                    const int8_t* src0 = src - 1 * X * C;
                    const int8_t* src1 = src + 0 * X * C;
                    const int8_t* src2 = src + 1 * X * C;
                    const int8_t* src3 = src + 2 * X * C;
                    const int8_t* src4 = src + 3 * X * C;
                    for (size_t x = 0; x < X; ++x)
                    {
                        for (size_t c = 0; c < B; ++c)
                        {
                            *dstE++ = (c < C) ? src1[c] : 0;
                            *dstE++ = (c < C && y + 1 < Y) ? src2[c] : 0;
                            *dstE++ = (c < C && y + 2 < Y) ? src3[c] : 0;
                            *dstE++ = (c < C && y + 3 < Y) ? src4[c] : 0;
                            *dstO++ = (c < C && y > 0) ? src0[c] : 0;
                            *dstO++ = (c < C && y + 0 < Y) ? src1[c] : 0;
                            *dstO++ = (c < C && y + 1 < Y) ? src2[c] : 0;
                            *dstO++ = (c < C && y + 2 < Y) ? src3[c] : 0;
                        }
                        src0 += C;
                        src1 += C;
                        src2 += C;
                        src3 += C;
                        src4 += C;
                    }
                    src += 4 * X * C;
                }
            }
            else if (a.reorderType == 1)
            {
                for (size_t c = 0; c < C; c += F)
                {
                    for (size_t y = 0; y < Y + 1; y += 4)
                    {
                        const int8_t* src0 = src + (y - 1) * X * C + c;
                        const int8_t* src1 = src + (y + 0) * X * C + c;
                        const int8_t* src2 = src + (y + 1) * X * C + c;
                        const int8_t* src3 = src + (y + 2) * X * C + c;
                        const int8_t* src4 = src + (y + 3) * X * C + c;
                        for (size_t x = 0; x < X; ++x)
                        {
                            for (size_t i = 0; i < F; ++i)
                            {
                                *dstE++ = (i + c < C) ? src1[i] : 0;
                                *dstE++ = (i + c < C && y + 1 < Y) ? src2[i] : 0;
                                *dstE++ = (i + c < C && y + 2 < Y) ? src3[i] : 0;
                                *dstE++ = (i + c < C && y + 3 < Y) ? src4[i] : 0;
                                *dstO++ = (i + c < C && y > 0) ? src0[i] : 0;
                                *dstO++ = (i + c < C && y + 0 < Y) ? src1[i] : 0;
                                *dstO++ = (i + c < C && y + 1 < Y) ? src2[i] : 0;
                                *dstO++ = (i + c < C && y + 2 < Y) ? src3[i] : 0;
                            }
                            src0 += C;
                            src1 += C;
                            src2 += C;
                            src3 += C;
                            src4 += C;
                        }
                    }
                }
            }
            else
                assert(0);
        }

        bool SynetQuantizedConvolutionNhwcDepthwiseV3::Preferable(const ConvParam& p, size_t F)
        {
            return p.trans != 0 && p.IsDepthwise() && p.IsDilation(1) && p.group >= F
                && (p.IsStride(1) || p.IsStride(2))
                && (p.IsKernel(3) || p.IsKernel(5) || p.IsKernel(7));
        }
     }
#endif
}
