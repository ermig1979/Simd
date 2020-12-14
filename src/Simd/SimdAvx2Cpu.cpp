/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdEnable.h"
#include "Simd/SimdCpu.h"

#if defined(_MSC_VER)
#include <windows.h>
#endif

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        SIMD_INLINE bool SupportedByCPU()
        {
            return
                Base::CheckBit(Cpuid::Ordinary, Cpuid::Ecx, Cpuid::OSXSAVE) &&
                Base::CheckBit(Cpuid::Extended, Cpuid::Ebx, Cpuid::AVX2) &&
                Base::CheckBit(Cpuid::Ordinary, Cpuid::Ecx, Cpuid::FMA) &&
                Base::CheckBit(Cpuid::Ordinary, Cpuid::Ecx, Cpuid::F16C);
        }

        SIMD_INLINE bool SupportedByOS()
        {
#if defined(_MSC_VER)
            __try
            {
                __m256i value = _mm256_abs_epi8(_mm256_set1_epi8(1));// try to execute of AVX2 instructions;
                return true;
            }
            __except (EXCEPTION_EXECUTE_HANDLER)
            {
                return false;
            }
#else
            return true;
#endif
        }

        bool GetEnable()
        {
            return SupportedByCPU() && SupportedByOS();
        }
    }
#endif
}
