/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx512f.h"
#include "Simd/SimdGemm.h"

#if defined(SIMD_X86_ENABLE) && defined(_MSC_VER) && _MSC_VER < 1924
#define SIMD_MSVS2017_WIN32_RELEASE_COMPILER_ERROR
#endif

namespace Simd
{
#if defined(SIMD_AVX512F_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512f
    {
        SynetConvolution32fNhwcDirect::SynetConvolution32fNhwcDirect(const ConvParam32f& p)
            : Avx2::SynetConvolution32fNhwcDirect(p)
        {
            if (p.dstC <= Avx::F)
                return;
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

        bool SynetConvolution32fNhwcDirect::SetRt(const ConvParam32f& p, AlgParam& a)
        {
            switch (a.microD)
            {
            case 2 * F: return Set2r(p, a);
            case 3 * F: return Set3r(p, a);
            default:
                return false;
            }
        }
    }
#endif//SIMD_AVX512F_ENABLE
}
