/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdSynetDeconvolution16b.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdAlignment.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SynetDeconvolution16bNhwcGemm::SynetDeconvolution16bNhwcGemm(const DeconvParam& p)
            : SynetDeconvolution16b(p)
            , _convert(0)
        {
            assert(p.trans && p.group == 1);
        }

        size_t SynetDeconvolution16bNhwcGemm::ExternalBufferSize() const
        {
            const DeconvParam& p = _param;
            const AlgParam &a = _alg;
            size_t size = 0;
            if (!_src16b || a.bufK != a.K)
                size += a.bufK * a.bufM * sizeof(uint16_t);
            if (!_is1x1)
                size += a.bufN * a.bufM * sizeof(float);
            if (_dst16b)
                size += p.dstH * p.dstW * p.dstC * sizeof(float);
            return size;
        }

        void SynetDeconvolution16bNhwcGemm::SetAlgParam(size_t F, size_t microM, size_t microN, size_t microK, size_t L1, size_t L2, size_t L3)
        {
            const DeconvParam& p = _param;
            AlgParam& a = _alg;
            a.M = p.srcH * p.srcW;
            a.N = p.kernelY * p.kernelX * p.dstC;
            a.K = p.srcC;
            a.F = F;
            a.microM = microM;
            a.microN = microN;
            a.microK = microK;
            a.bufK = AlignHi(a.K, a.microK);
            a.bufN = AlignHi(a.N, a.microN);
            a.bufM = p.dstH * AlignHi(p.dstW, a.F);
            a.macroK = Simd::RestrictRange(AlignLo(L1 / a.microN / 2, a.microK), a.microK, a.bufK);
            a.macroH = Simd::RestrictRange(L2 / a.macroK / p.dstW / 2, size_t(1), p.dstH);
            a.macroN = Simd::RestrictRange(AlignLoAny(L3 / a.macroK / 2, a.microN), a.microN, a.bufN);
            _stepS = p.srcH * p.srcW * p.srcC * _elemS;
            _stepD = p.dstH * p.dstW * p.dstC * _elemD;
        }

        void SynetDeconvolution16bNhwcGemm::SetParams(const float* weight, const float* bias, const float* params)
        {
            const AlgParam& a = _alg;
            size_t N = DivHi(a.N, a.F);
            _weight.Resize(a.bufK * a.bufN, true);
            uint16_t* dst = _weight.data;
            for (size_t n = 0; n < N; n++)
            {
                for (size_t k = 0; k < a.bufK; k += 2)
                {
                    const float* src = weight + k * a.N + n * a.F;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (n * a.F + f < a.N && k + i < a.K)
                                *(dst++) = Float32ToBFloat16(src[i * a.N]);
                            else
                                *(dst++) = 0;
                        }
                        src++;
                    }
                }
            }
            Float32ToBFloat16(weight, _weight.size, _weight.data);
            SynetDeconvolution16b::SetBias(bias, Alignment());
            SynetDeconvolution16b::SetParams(params, Alignment());
        }

        void SynetDeconvolution16bNhwcGemm::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const DeconvParam& p = _param;
            const AlgParam& a = _alg;
            buf = Buffer(buf);
            uint16_t* bufS = _src16b && a.bufK == a.K ? NULL : Allocate<uint16_t>(buf, a.bufK * a.bufM); 
            float* bufB = _is1x1 ? NULL : Allocate<float>(buf, a.bufN * a.bufM);
            float* bufD = _dst16b ? Allocate<float>(buf, p.dstH * p.dstW * p.dstC) : NULL;
            const uint16_t* wgt = _weight.data;
            for (size_t b = 0; b < p.batch; ++b)
            {
                const uint16_t* src16b = _src16b ? (uint16_t*)src : bufS;
                float* dst32f = _dst16b ? bufD : (float*)dst;
                float* buf32f = _is1x1 ? dst32f : bufB;
                if (!_src16b || a.bufK != a.K)
                    _convert(src, p, a, 0, p.srcH, bufS);
                //GemmNN(_M, _N, _K, src16b, _ldS, wgt, _ldW, buf32f, _ldD);
                //if (!_is1x1)
                //    ImgToRow(buf32f, dst32f);
                _biasAct(dst32f, p, a, p.dstC, p.dstH, _bias.data, _params.data, dst);
                src += _stepS;
                dst += _stepD;
            }
        }

        bool SynetDeconvolution16bNhwcGemm::Preferable(const ConvParam& p)
        {
            return false;
        }
    }
#endif
}
