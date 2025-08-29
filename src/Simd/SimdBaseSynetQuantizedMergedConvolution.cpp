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
#include "Simd/SimdSynetQuantizedMergedConvolution.h"
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
    SynetQuantizedMergedConvolution::SynetQuantizedMergedConvolution(const MergConvParam& p)
        : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        , _perf(NULL)
#endif
    {
        _count = p.count;
        _batch = p.conv[0].batch;
        _merge = 1;
        const ConvParam& beg = p.conv[0], &end = p.conv[p.count - 1];
        _sizeS = beg.srcC * beg.srcH * beg.srcW;
        _sizeD = end.dstC * end.dstH * end.dstW;
    }

    uint8_t* SynetQuantizedMergedConvolution::Buffer(uint8_t* buffer)
    {
        if (buffer)
            return buffer;
        else
        {
            _buffer.Resize(ExternalBufferSize());
            return _buffer.data;
        }
    }

    const char* SynetQuantizedMergedConvolution::Info() const
    {
        _info = Desc();
        return _info.c_str();
    }

    size_t SynetQuantizedMergedConvolution::ExternalBufferSize() const
    {
        size_t size = SIMD_ALIGN;
        return size;
    }

    size_t SynetQuantizedMergedConvolution::InternalBufferSize() const
    {
        size_t size = _buffer.RawSize() + _dwSrcZero.RawSize();
        for(size_t c = 0; c < _count; ++c)
            size += _weight[c].RawSize() + _bias[c].RawSize() + _norm[c].RawSize();
        return size;
    }

    void SynetQuantizedMergedConvolution::SetParams(const float* ioScale, const uint8_t* ioZero, const int8_t* const* weight, const float* const* weightScale, const int32_t* const* bias)
    {
        const MergConvParam& p = _param;
        for (size_t i = 0, n = p.count + (p.add ? 1 : 0); i <= n; ++i)
        {
            _ioScale[i] = ioScale[i];
            _ioZero[i] = ioZero[i];
        }
        for (size_t c = 0; c < p.count; ++c)
        {
            if (p.conv[c].IsDepthwise())
            {
                _dwSrcZero.Resize(p.conv[c].group);
                memset(_dwSrcZero.data, ioZero[c], p.conv[c].group);
                SetDepthwise(weight[c], p.conv[c], _weight[c]);
            }
            else
            {
                if(c == 0)
                    SetInput(weight[c], p.conv[c], _weight[c]);
                else
                    SetOutput(weight[c], p.conv[c], _weight[c]);
            }

            SetBias(weight[c], bias[c], ioZero[c], p.conv[c], _bias[c]);

            SetNorm(weightScale[c], ioScale[c], ioScale[c + 1], p.conv[c], _norm[c]);
        }
        if (p.add)
        {
            assert(p.count == 3 && p.conv[0].srcC == p.conv[2].dstC && p.conv[0].srcH == p.conv[2].dstH && p.conv[0].srcW == p.conv[2].dstW);
            _srcBias = -ioZero[0];
            _srcNorm = ioScale[0];
            _dstBias = -ioZero[3];
            _dstNorm = ioScale[3];
            _addZero = ioZero[4];
            _addScale = 1.0f / ioScale[4];
        }
    }

    void SynetQuantizedMergedConvolution::SetBias(const int8_t* weight, const int32_t* bias, int32_t zero, const ConvParam& p, Array32i& dst)
    {
        if (bias)
            dst.Assign(bias, p.dstC);
        else
            dst.Resize(p.dstC, true);
        size_t K = p.kernelY * p.kernelX * p.srcC / p.group, D = p.dstC;
        for (size_t d = 0; d < D; ++d)
            for (size_t k = 0; k < K; ++k)
                dst[d] -= weight[k * D + d] * zero;
    }

    void SynetQuantizedMergedConvolution::SetNorm(const float* weightScale, float srcScale, float dstScale, const ConvParam& p, Array32f& dst)
    {
        size_t D = p.dstC;
        dst.Resize(D);
        for (size_t d = 0; d < D; ++d)
            dst[d] = srcScale * weightScale[d] / dstScale;
    }

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
    Base::PerformanceMeasurer * SynetQuantizedMergedConvolution::Perf(const char* func)
    {
        if (_perf == NULL)
            _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info(false) + " " + Desc(), Param().Flop());
        return _perf;
    }
#endif

    //------------------------------------------------------------------------------------------------

    namespace Base
    {
        SynetQuantizedMergedConvolutionRef::SynetQuantizedMergedConvolutionRef(const MergConvParam& p)
            : Simd::SynetQuantizedMergedConvolution(p)
        {
            _sizeB = 0;
            for (size_t c = 0; c < _param.count; ++c)
            {
                const ConvParam& p = _param.conv[c];
                _sizeB = Simd::Max(_sizeB, p.dstC * p.dstH * p.dstW);
            }
        }

        size_t SynetQuantizedMergedConvolutionRef::ExternalBufferSize() const
        {
            return _sizeB * 5 + SIMD_ALIGN;
        }

        void SynetQuantizedMergedConvolutionRef::SetInput(const int8_t* weight, const ConvParam& p, Array8i& dst)
        {
            dst.Resize(p.kernelY * p.kernelX * p.srcC / p.group * p.dstC);
            dst.Assign(weight, dst.size);
        }

        void SynetQuantizedMergedConvolutionRef::SetDepthwise(const int8_t* weight, const ConvParam& p, Array8i& dst)
        {
            dst.Resize(p.kernelY * p.kernelX * p.srcC / p.group * p.dstC);
            dst.Assign(weight, dst.size);
        }

        void SynetQuantizedMergedConvolutionRef::SetOutput(const int8_t* weight, const ConvParam& p, Array8i& dst)
        {
            dst.Resize(p.kernelY * p.kernelX * p.srcC / p.group * p.dstC);
            dst.Assign(weight, dst.size);
        }

        void SynetQuantizedMergedConvolutionRef::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            buf8 = Buffer(buf8);
            int32_t* sum = Allocate<int32_t>(buf8, _sizeB);
            uint8_t* buf = Allocate<uint8_t>(buf8, _sizeB);
#if defined(__MINGW32__) || defined(__MINGW64__)
            bool overflow = true;
#else
            bool overflow = SimdCpuInfo(SimdCpuInfoAvx512vnni) == 0;
#endif
            for (size_t b = 0; b < _batch; b += 1)
            {
                for (size_t c = 0; c < _count; ++c)
                {
                    const ConvParam& p = _param.conv[c];
                    const uint8_t* ps = c ? buf : src;
                    const int8_t* pw = _weight[c].data;
                    uint8_t* pd = c + 1 < _count ? buf : dst;
                    if (p.IsDepthwise())
                        Depthwise(ps, _dwSrcZero.data, p, pw, sum);
                    else
                        GemmNhwc(p.dstH * p.dstW, p.dstC, 1, p.srcC, ps, p.srcC, pw, p.dstC, sum, p.dstC, overflow);
                    QuantizeSumLinear(sum, 1, p.dstC, p.dstH, p.dstW, p.dstF, _bias[c].data, _norm[c].data, _ioZero[c + 1], pd);
                }
                if (_param.add)
                    AddSrc(src, dst);
                src += _sizeS;
                dst += _sizeD;
            }
        }

        void SynetQuantizedMergedConvolutionRef::Depthwise(const uint8_t* src, const uint8_t* zero, const ConvParam& p, const int8_t* weight, int32_t* dst)
        {
            size_t C = p.srcC;
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    for (size_t c = 0; c < C; ++c)
                        dst[c] = 0;
                    for (size_t ky = 0; ky < p.kernelY; ++ky)
                    {
                        size_t sy = dy * p.strideY + ky - p.padY;
                        for (size_t kx = 0; kx < p.kernelX; ++kx)
                        {                        
                            size_t sx = dx * p.strideX + kx - p.padX;
                            const int8_t* pw = weight + (ky * p.kernelX + kx) * C;
                            if (sy < p.srcH && sx < p.srcW)
                            {
                                const uint8_t* ps = src + (sy * p.srcW + sx) * C;
                                for (size_t c = 0; c < C; ++c)
                                    dst[c] += ps[c] * pw[c];
                            }
                            else
                            {
                                for (size_t c = 0; c < C; ++c)
                                    dst[c] += zero[c] * pw[c];
                            }
                        }
                    }
                    dst += C;
                }
            }
        }

        void SynetQuantizedMergedConvolutionRef::AddSrc(const uint8_t* src, uint8_t* dst)
        {
            for (size_t i = 0; i < _sizeS; ++i)
            {
                float _src = DequantizeLinear(src[i], _srcBias, _srcNorm);
                float _dst = DequantizeLinear(dst[i], _dstBias, _dstNorm);
                dst[i] = QuantizeLinear(_src + _dst, _addScale, _addZero, 0, 255);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetQuantizedMergedConvolution::SynetQuantizedMergedConvolution(const MergConvParam& p)
            : Simd::SynetQuantizedMergedConvolution(p)
            , _inputPreprocess(NULL)
            , _inputConvolution(NULL)
            , _depthwisePreprocess(NULL)
            , _depthwiseConvolution(NULL)
            , _outputConvolution{NULL, NULL}
            , _addInputToOutput(NULL)
        {
            memset(&_alg, 0, sizeof(_alg));
        }

        size_t SynetQuantizedMergedConvolution::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            return a.isB + a.idB * 4 + a.dsB + a.dbB * a.dbE + a.ddB + a.odB * 4 + SIMD_ALIGN * 3;
        }

        void SynetQuantizedMergedConvolution::SetInput(const int8_t* src, const ConvParam& p, Array8i& dst)
        {
            assert(Is1x1(p));
            size_t F = _alg.miC, C = AlignHi(p.srcC, _alg.miK), D = DivHi(p.dstC, F);
            dst.Resize(C * D * F, true);
            int8_t* pd = dst.data;
            for (size_t d = 0; d < D; d++)
            {
                for (size_t c = 0; c < C; c += 4)
                {
                    const int8_t* ps = src + c * p.dstC + d * F;
                    for (size_t f = 0; f < F; ++f)
                    {
                        for (size_t i = 0; i < 4; ++i)
                        {
                            if (d * F + f < p.dstC && c + i < p.srcC)
                                *(pd++) = ps[i * p.dstC];
                            else
                                *(pd++) = 0;
                        }
                        if (c < p.srcC)
                            ps++;
                    }
                }
            }
        }

        void SynetQuantizedMergedConvolution::SetDepthwise(const int8_t* src, const ConvParam& p, Array8i& dst)
        {
            assert(IsDepthwise(p));
            const AlgParam& a = _alg;
            size_t Y = p.kernelY, X = p.kernelX, C = p.srcC, F = a.miC, B = AlignHi(p.srcC, a.miC);
            dst.Resize(a.dwSize * 2 * a.dbE);
            if (a.dbE == 2)
            {
                int16_t* dstE = (int16_t*)dst.data, * dstO = dstE + a.dwSize;
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
            {
                int8_t* dstE = dst.data, * dstO = dstE + a.dwSize;
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
        }

        void SynetQuantizedMergedConvolution::SetOutput(const int8_t* src, const ConvParam& p, Array8i& dst)
        {
            size_t F = _alg.miC, C = DivHi(AlignHi(p.srcC, _alg.miK), 4), D = DivHi(p.dstC, F), M = DivHi(_alg.maC, 4);
            dst.Resize(C * D * F * 4, true);
            int8_t* pd = dst.data;
            for (size_t cB = 0; cB < C; cB += M)
            {
                size_t cE = Simd::Min(C, cB + M);
                for (size_t d = 0; d < D; d++)
                {
                    for (size_t c = cB; c < cE; ++c)
                    {
                        const int8_t* ps = src + c * 4 * p.dstC + d * F;
                        for (size_t f = 0; f < F; ++f)
                        {
                            for (size_t i = 0; i < 4; ++i)
                            {
                                if (d * F + f < p.dstC && c * 4 + i < p.srcC)
                                    *(pd++) = ps[i * p.dstC];
                                else
                                    *(pd++) = 0;
                            }
                            if (c * 4 < p.srcC)
                                ps++;
                        }
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetQuantizedMergedConvolutionCdc::SynetQuantizedMergedConvolutionCdc(const MergConvParam& p)
            : SynetQuantizedMergedConvolution(p)
        {
        }

        bool SynetQuantizedMergedConvolutionCdc::Preferable(const MergConvParam& p)
        {
            return p.count == 3 && Is1x1(p.conv[0]);
        }

        void SynetQuantizedMergedConvolutionCdc::SetSize(size_t miC, size_t miK, size_t dbE)
        {
            const size_t L1 = Base::AlgCacheL1(), L2 = Base::AlgCacheL2(), L3 = Base::AlgCacheL3();
            const MergConvParam& p = _param;
            const ConvParam& c0 = p.conv[0];
            const ConvParam& c1 = p.conv[1];
            const ConvParam& c2 = p.conv[2];
            AlgParam& a = _alg;

            a.miC = miC;
            a.miK = miK;
            a.dbE = dbE;

            a.iwStep = AlignHi(c0.srcC, a.miK);
            a.dwC = AlignHi(c1.srcC, a.miC);
            a.dbStep = 2 / c1.strideY;
            a.dbH = Pow2Hi(AlignHi(c1.kernelY, 4 / a.dbE));
            a.dbW = c1.srcW + c1.padX + c1.padW;
            a.dwStep = c1.kernelX * AlignHi(c1.kernelY + 1, 4 / a.dbE);
            a.dwSize = a.dwStep * a.dwC;
            a.owStep = AlignHi(c2.dstC, a.miC);

            size_t size = 0;
            for (size_t i = 0; i < 3; ++i)
            {
                const ConvParam& c = p.conv[i];
                if (c.group == 1)
                    size += AlignHi(c.srcC, a.miK) * AlignHi(c.dstC, a.miC * 2);
                else
                    size += a.dwSize * a.dbE * 2;
            }
            size_t count = size / (L3 / 2) + 1;
            a.maC = AlignHi(AlignHi(c0.dstC / count, 2 * a.miC), a.miK);
            for (size_t yStep = c1.dstH; yStep >= 1; yStep--)
            {
                a.ddStep = AlignHi(Simd::Max<size_t>(1, yStep), a.dbStep);
                a.ddH = Pow2Hi(a.ddStep);

                a.dsStep = a.ddStep * c1.strideY;
                a.dsStart = Simd::Min((a.ddStep - 1) * c1.strideY + c1.kernelY - c1.padY, c1.srcH);
                a.dsH = Pow2Hi(Simd::Max((a.ddStep - 1) * c1.strideY + c1.kernelY, a.dsStart));

                a.isH = Aligned(c0.srcC, a.miK) ? 0 : a.dsH;

                a.isB = Aligned(c0.srcC, a.miK) ? 0 : a.isH * c0.srcW * AlignHi(c0.srcC, a.miK);
                a.dsB = a.dsH * c1.srcW * a.maC;
                a.dbB = a.dbH * a.dbW * a.maC;
                a.ddB = a.ddH * c1.dstW * a.maC;
                if (a.isB + a.dsB + a.dbB * a.dbE + a.ddB <= L2)
                    break;
            }
            if (a.miK == 32)
            {
                bool aligned = Aligned(c2.dstC, a.miC) && Aligned(c2.dstH * c2.dstW, a.miC) && Aligned((c2.dstH % a.ddStep) * c2.dstW, a.miC);
                a.odB = (count > 1 || !aligned) ? AlignHi(c2.dstC, a.miC) * AlignHi(c2.dstH * c2.dstW + a.miC, a.miC) : 0;
                a.idB = 4 * 16 * 16;
            }
            else
                a.odB = count > 1 ? _sizeD : 0;

            size_t zeroB = a.dbW * AlignHi(c1.dstC, 16);
            if (_dwSrcZero.size != zeroB)
            {
                _dwSrcZero.Resize(zeroB);
                memset(_dwSrcZero.data, _ioZero[1], _dwSrcZero.size);
            }
        }

        void SynetQuantizedMergedConvolutionCdc::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const MergConvParam& p = _param;
            const ConvParam& c0 = p.conv[0];
            const ConvParam& c1 = p.conv[1];
            const ConvParam& c2 = p.conv[2];
            const AlgParam& a = _alg;

            buf = Buffer(buf);
            uint8_t* isBuf = Allocate<uint8_t>(buf, a.isB);
            SetGap(buf);
            int32_t* idBuf = Allocate<int32_t>(buf, a.idB);
            uint8_t* dsBuf = Allocate<uint8_t>(buf, a.dsB);
            SetGap(buf);
            uint8_t* dbBuf = Allocate<uint8_t>(buf, a.dbB * a.dbE);
            uint8_t* ddBuf = Allocate<uint8_t>(buf, a.ddB);
            SetGap(buf);
            int32_t* odBuf = Allocate<int32_t>(buf, a.odB);

            for (size_t b = 0; b < c0.batch; ++b)
            {
                for (size_t c = 0, C = c1.dstC; c < C; c += a.maC)
                {
                    size_t maC = Simd::Min(C, c + a.maC) - c;
                    for (size_t dyBeg = 0, syBeg = 0; dyBeg < c1.dstH;)
                    {
                        size_t dyEnd = Simd::RestrictRange(dyBeg + a.ddStep, a.ddStep, c1.dstH);
                        size_t syEnd = Simd::RestrictRange(syBeg + a.dsStep, a.dsStart, c1.srcH);

                        if (_inputPreprocess)
                            _inputPreprocess(src, c0, a, syBeg, syEnd, isBuf);

                        const uint8_t* isPtr = _inputPreprocess ? isBuf : src;
                        _inputConvolution(isPtr, c0, a, maC, syBeg, syEnd, _weight[0].data + c * a.iwStep,
                            _bias[0].data + c, _norm[0].data + c, _ioZero[1], idBuf, dsBuf);

                        for (size_t byBeg = dyBeg; byBeg < dyEnd;)
                        {
                            size_t byEnd = Simd::Min(byBeg + a.dbStep, dyEnd);

                            _depthwisePreprocess(dsBuf, _dwSrcZero.data, c1, a, maC, byBeg, byEnd, dbBuf);

                            _depthwiseConvolution(dbBuf, c1, a, maC, byBeg, byEnd, _weight[1].data + c * a.dwStep * a.dbE,
                                _bias[1].data + c, _norm[1].data + c, _ioZero[2], ddBuf);

                            byBeg = byEnd;
                        }

                        if (c + maC == C)
                            _outputConvolution[0](ddBuf, c2, a, maC, dyBeg, dyEnd, maC != C ? 1 : 0,
                                _weight[2].data + c * a.owStep, _bias[2].data, _norm[2].data, _ioZero[3], odBuf, dst);
                        else
                            _outputConvolution[1](ddBuf, c2, a, maC, dyBeg, dyEnd, c != 0 ? 1 : 0,
                                _weight[2].data + c * a.owStep, _bias[2].data, _norm[2].data, _ioZero[3], odBuf, dst);
                         
                        if (p.add && c + maC == C)
                            _addInputToOutput(src, _srcBias, _srcNorm, dst, _dstBias, _dstNorm, c2, dyBeg, dyEnd, _addScale, _addZero, dst);
                        dyBeg = dyEnd;
                        syBeg = syEnd;
                    }
                }
                src += _sizeS;
                dst += _sizeD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetQuantizedMergedConvolutionInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add)
        {
            MergConvParam param(batch, convs, count, add);
            if (!param.Valid(SimdTensorData8u, SimdTensorData8u))
                return NULL;
            return new SynetQuantizedMergedConvolutionRef(param);
        }
    }
#endif
}
