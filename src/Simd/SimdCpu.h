/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#if defined(SIMD_X86_ENABLE) || defined(SIMD_X64_ENABLE)
    namespace Cpuid
    {
        // See http://www.sandpile.org/x86/cpuid.htm for additional information.
        enum Level
        {
            Ordinary = 1,
            Extended = 7,
        };

        enum Register
        {
            Eax = 0,
            Ebx = 1,
            Ecx = 2,
            Edx = 3,
        };

        enum Bit
        {
            // Ordinary:
            // Edx:
            SSE = 1 << 25,
            SSE2 = 1 << 26,

            // Ecx:
            SSE3 = 1 << 0,
            SSSE3 = 1 << 9,
            FMA = 1 << 12,
            SSE41 = 1 << 19,
            SSE42 = 1 << 20,
            OSXSAVE = 1 << 27,
            AVX = 1 << 28,
            F16C = 1 << 29,

            // Extended:
            // Ebx:
            AVX2 = 1 << 5,
            AVX512F = 1 << 16,
            AVX512DQ = 1 << 17,
            AVX512CD = 1 << 28,
            AVX512BW = 1 << 30,
            AVX512VL = 1 << 31,

            // Ecx:
            AVX512VBMI = 1 << 1,
            AVX512VNNI = 1 << 11,
        };
    }
#endif//defined(SIMD_X86_ENABLE) || defined(SIMD_X64_ENABLE)

    namespace Cpu
    {
        extern const size_t SOCKET_NUMBER;
        extern const size_t CORE_NUMBER;
        extern const size_t THREAD_NUMBER;
        extern const size_t L1_CACHE_SIZE;
        extern const size_t L2_CACHE_SIZE;
        extern const size_t L3_CACHE_SIZE;
    }

    namespace Base
    {
#if defined(SIMD_X86_ENABLE) || defined(SIMD_X64_ENABLE)
        bool CheckBit(Cpuid::Level level, Cpuid::Register index, Cpuid::Bit bit);
#endif

#if defined(__GNUC__) && (defined(SIMD_PPC_ENABLE) || defined(SIMD_PPC64_ENABLE) || defined(SIMD_ARM_ENABLE) || defined(SIMD_ARM64_ENABLE))
        bool CheckBit(int at, int bit);
#endif

        size_t CpuSocketNumber();

        size_t CpuCoreNumber();

        size_t CpuThreadNumber();

        size_t CpuCacheSize(size_t level);

        SIMD_INLINE size_t AlgCacheL1()
        {
            return Cpu::L1_CACHE_SIZE;
        }

        SIMD_INLINE size_t AlgCacheL2()
        {
            return Cpu::L3_CACHE_SIZE ? Cpu::L2_CACHE_SIZE : Cpu::L2_CACHE_SIZE * Cpu::SOCKET_NUMBER / Cpu::CORE_NUMBER;
        }

        SIMD_INLINE size_t AlgCacheL3()
        {
            return Cpu::L3_CACHE_SIZE ? Cpu::L3_CACHE_SIZE * Cpu::SOCKET_NUMBER / Cpu::CORE_NUMBER : Cpu::L2_CACHE_SIZE;
        }
    }

#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
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
}

#endif//__SimdCpu_h__
