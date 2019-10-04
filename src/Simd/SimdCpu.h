/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#ifndef __SimdCpu_h__
#define __SimdCpu_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
    namespace Base
    {
        size_t CpuSocketNumber();

        size_t CpuCoreNumber();

        size_t CpuThreadNumber();

        size_t CpuCacheSize(size_t level);
    }

#ifdef SIMD_SSE_ENABLE
    namespace Sse
    {
        const unsigned int SCR_FTZ = 1 << 15;
        const unsigned int SCR_DAZ = 1 << 6;

        SIMD_INLINE SimdBool GetFastMode()
        {
            return _mm_getcsr() & (SCR_FTZ | SCR_DAZ) ? SimdTrue : SimdFalse;
        }

        SIMD_INLINE void SetFastMode(SimdBool value)
        {
            if (value)
                _mm_setcsr(_mm_getcsr() | (SCR_FTZ | SCR_DAZ));
            else
                _mm_setcsr(_mm_getcsr() & ~(SCR_FTZ | SCR_DAZ));
        }
    }
#endif

#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        SIMD_INLINE unsigned int GetStatusWord()
        {
            unsigned int dst;
#if defined(__GNUC__)
#if defined(SIMD_ARM64_ENABLE)
            __asm__ volatile("mrs %[dst], FPCR" : [dst] "=r" (dst));
#else
            __asm__ volatile("vmrs %[dst], FPSCR" : [dst] "=r" (dst));
#endif
#endif
            return dst;
        }

        SIMD_INLINE void SetStatusWord(unsigned int src)
        {
#if defined(__GNUC__)
#if defined(SIMD_ARM64_ENABLE)
            __asm__ volatile("msr FPCR, %[src]" : : [src] "r" (src));
#else
            __asm__ volatile("vmsr FPSCR, %[src]" : : [src] "r" (src));
#endif
#endif
        }

        const unsigned int FPSCR_FTZ = 1 << 24;

        SIMD_INLINE SimdBool GetFastMode()
        {
            return GetStatusWord() & FPSCR_FTZ ? SimdTrue : SimdFalse;
        }

        SIMD_INLINE void SetFastMode(SimdBool value)
        {
            if (value)
                SetStatusWord(GetStatusWord() | FPSCR_FTZ);
            else
                SetStatusWord(GetStatusWord() & ~FPSCR_FTZ);
        }
    }
#endif

    namespace Cpu
    {
        const size_t SOCKET_NUMBER = Base::CpuSocketNumber();
        const size_t CORE_NUMBER = Base::CpuCoreNumber();
        const size_t THREAD_NUMBER = Base::CpuThreadNumber();
        const size_t L1_CACHE_SIZE = Base::CpuCacheSize(1);
        const size_t L2_CACHE_SIZE = Base::CpuCacheSize(2);
        const size_t L3_CACHE_SIZE = Base::CpuCacheSize(3);
    }

    namespace Base
    {
        SIMD_INLINE size_t AlgCacheL1()
        {
            return Cpu::L1_CACHE_SIZE;
        }

        SIMD_INLINE size_t AlgCacheL2()
        {
            return Cpu::L2_CACHE_SIZE;
        }

        SIMD_INLINE size_t AlgCacheL3()
        {
            return Cpu::L3_CACHE_SIZE * Cpu::SOCKET_NUMBER / Cpu::CORE_NUMBER;
        }
    }
}

#endif//__SimdCpu_h__
