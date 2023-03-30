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
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAlignment.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdErf.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        template<SimdSynetUnaryOperation32fType type> void SynetUnaryOperation32f(const float* src, size_t size, float* dst)
        {
            size_t size4 = AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4)
            {
                dst[i + 0] = SynetUnaryOperation32f<type>(src[i + 0]);
                dst[i + 1] = SynetUnaryOperation32f<type>(src[i + 1]);
                dst[i + 2] = SynetUnaryOperation32f<type>(src[i + 2]);
                dst[i + 3] = SynetUnaryOperation32f<type>(src[i + 3]);
            }
            for (; i < size; ++i)
                dst[i] = SynetUnaryOperation32f<type>(src[i]);
        }

        void SynetUnaryOperation32f(const float * src, size_t size, SimdSynetUnaryOperation32fType type, float * dst)
        {
            switch (type)
            {
            case SimdSynetUnaryOperation32fAbs: SynetUnaryOperation32f<SimdSynetUnaryOperation32fAbs>(src, size, dst); break;
            case SimdSynetUnaryOperation32fErf: SynetUnaryOperation32f<SimdSynetUnaryOperation32fErf>(src, size, dst); break;
            case SimdSynetUnaryOperation32fExp: SynetUnaryOperation32f<SimdSynetUnaryOperation32fExp>(src, size, dst); break;
            case SimdSynetUnaryOperation32fLog: SynetUnaryOperation32f<SimdSynetUnaryOperation32fLog>(src, size, dst); break;
            case SimdSynetUnaryOperation32fNeg: SynetUnaryOperation32f<SimdSynetUnaryOperation32fNeg>(src, size, dst); break;
            case SimdSynetUnaryOperation32fRsqrt: SynetUnaryOperation32f<SimdSynetUnaryOperation32fRsqrt>(src, size, dst); break;
            case SimdSynetUnaryOperation32fSqrt: SynetUnaryOperation32f<SimdSynetUnaryOperation32fSqrt>(src, size, dst); break;
            case SimdSynetUnaryOperation32fTanh: SynetUnaryOperation32f<SimdSynetUnaryOperation32fTanh>(src, size, dst); break;
            case SimdSynetUnaryOperation32fZero: SynetUnaryOperation32f<SimdSynetUnaryOperation32fZero>(src, size, dst); break;
            default:
                assert(0);
            }
        }
    }
#endif
}
