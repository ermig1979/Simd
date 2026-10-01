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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SynetConvolution32fNhwcDirect::SynetConvolution32fNhwcDirect(const ConvParam& p)
            : Base::SynetConvolution32fNhwcDirect(p)
        {
            const size_t F = svcntw();
            //_old.enable = true;
            if (_old.enable)
            {
                if (Set2f(p, _old.convolution))
                    OldSetAlgParam(F);
            }
            else
            {
                RunFuncs funcs;
                for (size_t n = 2; n <= 4; ++n)
                {
                    funcs.push_back(RunFunc(Ext() + "-" + ToStr(n)));
                    SetAlgParam(F, n, funcs.back().alg);
                    if (!SetRt(p, funcs.back().alg))
                        return;
                }
                _run.Init(funcs);
            }
        }

        bool SynetConvolution32fNhwcDirect::SetRt(const ConvParam& p, AlgParam& a)
        {
            if (a.microD == 2 * a.F)
                return Set2r(p, a);
            if (a.microD == 3 * a.F)
                return Set3r(p, a);
            if (a.microD == 4 * a.F)
                return Set4r(p, a);
            return false;
        }

        void SynetConvolution32fNhwcDirect::ReorderWeight(const float* src, float* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _run.At(0).alg;
            const size_t F = a.F;
            const svbool_t ptrue = svptrue_b32();
            size_t K = p.kernelY * p.kernelX * p.srcC, N = p.dstC, NF = AlignLo(N, F);
            for (size_t j = 0; j < NF; j += F)
            {
                const float* ps = src;
                for (size_t k = 0; k < K; ++k, dst += F, ps += N)
                    svst1_f32(ptrue, dst, svld1_f32(ptrue, ps));
                src += F;
            }
            if (NF < N)
            {
                const svbool_t mask = svwhilelt_b32((uint64_t)0, (uint64_t)(N - NF));
                const float* ps = src;
                for (size_t k = 0; k < K; ++k, dst += F, ps += N)
                    svst1_f32(ptrue, dst, svld1_f32(mask, ps));
            }
        }

        bool SynetConvolution32fNhwcDirect::Preferable(const ConvParam& p)
        {
            if (p.trans != SimdTrue || p.group != 1 || !p.IsDilation(1))
                return false;
            if (!p.Is1x1() && p.dstW < 6 + p.padX + p.padY)
                return false;
            if (p.Is1x1() && (p.srcC >= 2 * p.dstC || (p.activation == SimdConvolutionActivationIdentity && p.srcC > 128) || p.srcC > 256))
                return false;
            if (p.kernelY > p.srcH || p.kernelX > p.srcW)
                return false;
            return true;
        }
    }
#endif
}
