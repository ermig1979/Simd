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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdErf.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Sse41
    {
        SynetConvolution32fNhwcDirect::SynetConvolution32fNhwcDirect(const ConvParam& p)
            : Base::SynetConvolution32fNhwcDirect(p)
        {
            //_old.enable = true;
            if (_old.enable)
            {
                if (Set2f(p, _old.convolution))
                    OldSetAlgParam(F);
            }
            else
            {
                RunFuncs funcs;
                for (size_t n = 2; n <= 3; ++n)
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
            switch (a.microD)
            {
            case 2 * F: return Set2r(p, a);
            case 3 * F: return Set3r(p, a);
            default:
                return false;
            }
        }

        bool SynetConvolution32fNhwcDirect::Preferable(const ConvParam& p)
        {
            if (p.trans != SimdTrue || p.group != 1)
                return false;
            if (!p.Is1x1() && p.dstW < 6 + p.padX + p.padW)
                return false;
            if (p.Is1x1() && (p.srcC >= 2 * p.dstC || (p.activation == SimdConvolutionActivationIdentity && p.srcC > 512) || p.srcC > 512) &&
                p.dstH * p.dstW < p.srcC * p.dstC * 10)
                return false;
            if (p.kernelY > p.srcH || p.kernelX > p.srcW)
                return false;
            if ((p.strideY > 1 && p.strideX > 1) && p.srcC > 32 && float(p.kernelY * p.kernelX) / float(p.strideY * p.strideX) < 3.0f)
                return false;
            if ((p.padX + p.padW) * 3.0f > float(p.srcW))
                return false;
            return true;
        }
    }
#endif
}
