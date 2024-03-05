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
#include "Simd/SimdEnable.h"
#include "Simd/SimdCpu.h"

#if defined(_MSC_VER)
#include <windows.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18
#endif

#include <iostream>

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AMX_EMULATE) && defined(SIMD_AVX512BW_ENABLE))
    namespace AmxBf16
    {
        SIMD_INLINE bool SupportedByCPU()
        {
#if defined(SIMD_AMX_EMULATE)
            return 
                Base::CheckBit(7, 0, Cpuid::Ebx, Cpuid::AVX512_F) &&
                Base::CheckBit(7, 0, Cpuid::Ebx, Cpuid::AVX512_CD) &&
                Base::CheckBit(7, 0, Cpuid::Ebx, Cpuid::AVX512_DQ) &&
                Base::CheckBit(7, 0, Cpuid::Ebx, Cpuid::AVX512_BW) &&
                Base::CheckBit(7, 0, Cpuid::Ebx, Cpuid::AVX512_VL);
#else
            return
                Base::CheckBit(7, 0, Cpuid::Ecx, Cpuid::AVX512_VBMI) &&
                Base::CheckBit(7, 1, Cpuid::Eax, Cpuid::AVX512_BF16) &&
                Base::CheckBit(7, 0, Cpuid::Edx, Cpuid::AMX_TILE) &&
                Base::CheckBit(7, 0, Cpuid::Edx, Cpuid::AMX_INT8) &&
                Base::CheckBit(7, 0, Cpuid::Edx, Cpuid::AMX_BF16);
#endif
        }

        SIMD_INLINE bool SupportedByOS()
        {
#if defined(_MSC_VER)
            __try
            {
#if defined(SIMD_AMX_EMULATE)
                __m512i value = _mm512_abs_epi8(_mm512_set1_epi8(1));// try to execute of AVX-512BW instructions;
#else
                _tile_zero(0);// try to execute of AMX-BF16 instructions;
#endif
                return true;
            }
            __except (EXCEPTION_EXECUTE_HANDLER)
            {
                return false;
            }
#elif defined(__linux__)
            return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
#else
            return false;
#endif
        }

        bool GetEnable()
        {
            return SupportedByCPU() && SupportedByOS();
        }
    }
#endif
}
