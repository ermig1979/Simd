/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
#ifndef __SimdEnable_h__
#define __SimdEnable_h__

#include "Simd/SimdDefs.h"

#if defined(_MSC_VER)

#define NOMINMAX
#include <windows.h>
#include <intrin.h>

#elif defined(__GNUC__)

#if defined(SIMD_X86_ENABLE) || defined(SIMD_X64_ENABLE)
#include <cpuid.h>
#endif

#if defined(SIMD_PPC_ENABLE) || defined(SIMD_PPC64_ENABLE)
#include <unistd.h>
#include <fcntl.h>
#include <linux/auxvec.h>
#include <asm/cputable.h>
#endif

#else
# error Do not know how to detect CPU info
#endif

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
            SSSE3 =	1 << 9,
            SSE41 = 1 << 19,
            SSE42 = 1 << 20,
            OSXSAVE = 1 << 27,
            AVX = 1 << 28,

            // Extended:
            // Ebx:
            AVX2 = 1 << 5,
            AVX512F = 1 << 16,
            AVX512BW = 1 << 30,

            // Ecx:
            AVX512VBMI = 1 << 1,
        };

        SIMD_INLINE bool CheckBit(Level level, Register index, Bit bit)
        {
            unsigned int registers[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
            __cpuid((int*)registers, level);
#elif (defined __GNUC__)
            if (__get_cpuid_max(0, NULL) < level)
                return false;
            __cpuid_count(level, 0, registers[Eax], registers[Ebx], registers[Ecx], registers[Edx]);
#else
#error Do not know how to detect CPU info!
#endif
            return (registers[index] & bit) == bit;
        }
    }
#endif//defined(SIMD_X86_ENABLE) || defined(SIMD_X64_ENABLE)

#if defined(SIMD_PPC_ENABLE) || defined(SIMD_PPC64_ENABLE)
    namespace CpuInfo
    {
        SIMD_INLINE bool CheckBit(int at, int bit)
        {
            bool result = false;
            int file = ::open("/proc/self/auxv", O_RDONLY);
            if (file < 0) 
                return false;
            const ssize_t size = 64;
            unsigned long buffer[size];
            for(ssize_t count = size; count == size;)
            {
                count = ::read(file, buffer, sizeof(buffer))/sizeof(unsigned long);
                for (int i = 0; i < count; i += 2) 
                {                                                                                                           
                    if (buffer[i] == at) 
                    {                                                                                                                                      
                        result = !!(buffer[i + 1] & bit);                                                                                                           
                        count = 0;
                    } 
                    if (buffer[i] == AT_NULL)
                        count = 0;
                }
            }
            ::close(file);
            return result;
        }
    }
#endif//defined(SIMD_PPC_ENABLE) || defined(SIMD_PPC64_ENABLE) 

#ifdef SIMD_SSE_ENABLE
    namespace Sse
    {
        SIMD_INLINE bool SupportedByCPU()
        {
            return Cpuid::CheckBit(Cpuid::Ordinary, Cpuid::Edx, Cpuid::SSE);
        }

        SIMD_INLINE bool SupportedByOS()
        {
#if defined(_MSC_VER)
            __try
            {
                __m128 value = _mm_set1_ps(1.0f);// try to execute of SSE instructions;
                return true;
            }
            __except(EXCEPTION_EXECUTE_HANDLER)
            {
                return false;
            }
#else
            return true;
#endif
        }

        const bool Enable = SupportedByCPU() && SupportedByOS();
    }
#endif

#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        SIMD_INLINE bool SupportedByCPU()
        {
            return Cpuid::CheckBit(Cpuid::Ordinary, Cpuid::Edx, Cpuid::SSE2);
        }

        SIMD_INLINE bool SupportedByOS()
        {
#if defined(_MSC_VER)
            __try
            {
                __m128d value = _mm_set1_pd(1.0);// try to execute of SSE2 instructions;
                return true;
            }
            __except(EXCEPTION_EXECUTE_HANDLER)
            {
                return false;
            }
#else
            return true;
#endif
        }

        const bool Enable = SupportedByCPU() && SupportedByOS();
    }
#endif

#ifdef SIMD_SSSE3_ENABLE
    namespace Ssse3
    {
        SIMD_INLINE bool SupportedByCPU()
        {
            return Cpuid::CheckBit(Cpuid::Ordinary, Cpuid::Ecx, Cpuid::SSSE3);
        }

        SIMD_INLINE bool SupportedByOS()
        {
#if defined(_MSC_VER)
            __try
            {
                __m128i value = _mm_abs_epi8(_mm_set1_epi8(-1)); //try to execute of SSSE3 instructions;
                return true;
            }
            __except(EXCEPTION_EXECUTE_HANDLER)
            {
                return false;
            }
#else
            return true;
#endif
        }

        const bool Enable = SupportedByCPU() && SupportedByOS();
    }
#endif

#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        SIMD_INLINE bool SupportedByCPU()
        {
            return Cpuid::CheckBit(Cpuid::Ordinary, Cpuid::Ecx, Cpuid::SSE41);
        }

        SIMD_INLINE bool SupportedByOS()
        {
#if defined(_MSC_VER)
            __try
            {
                int value = _mm_testz_si128(_mm_set1_epi8(0), _mm_set1_epi8(-1)); // try to execute of SSE41 instructions;
                return true;
            }
            __except(EXCEPTION_EXECUTE_HANDLER)
            {
                return false;
            }
#else
            return true;
#endif
        }

        const bool Enable = SupportedByCPU() && SupportedByOS();
    }
#endif

#ifdef SIMD_SSE42_ENABLE
    namespace Sse42
    {
        SIMD_INLINE bool SupportedByCPU()
        {
            return Cpuid::CheckBit(Cpuid::Ordinary, Cpuid::Ecx, Cpuid::SSE42);
        }

        SIMD_INLINE bool SupportedByOS()
        {
#if defined(_MSC_VER)
            __try
            {
                uint32_t value = _mm_crc32_u8(0, 1); // try to execute of SSE42 instructions;
                return true;
            }
            __except(EXCEPTION_EXECUTE_HANDLER)
            {
                return false;
            }
#else
            return true;
#endif
        }

        const bool Enable = SupportedByCPU() && SupportedByOS();
    }
#endif

#ifdef SIMD_AVX_ENABLE
	namespace Avx
	{
        SIMD_INLINE bool SupportedByCPU()
        {
            return
                Cpuid::CheckBit(Cpuid::Ordinary, Cpuid::Ecx, Cpuid::OSXSAVE) &&
                Cpuid::CheckBit(Cpuid::Ordinary, Cpuid::Ecx, Cpuid::AVX);
        }

        SIMD_INLINE bool SupportedByOS()
        {
#if defined(_MSC_VER)
            __try
            {
                __m256d value = _mm256_set1_pd(1.0);// try to execute of AVX instructions;
                return true;
            }
            __except(EXCEPTION_EXECUTE_HANDLER)
            {
                return false;
            }
#else
            return true;
#endif
        }

		const bool Enable = SupportedByCPU() && SupportedByOS();
	}
#endif

#ifdef SIMD_AVX2_ENABLE
	namespace Avx2
	{
        SIMD_INLINE bool SupportedByCPU()
        {
            return
                Cpuid::CheckBit(Cpuid::Ordinary, Cpuid::Ecx, Cpuid::OSXSAVE) &&
                Cpuid::CheckBit(Cpuid::Extended, Cpuid::Ebx, Cpuid::AVX2);
        }

        SIMD_INLINE bool SupportedByOS()
        {
#if defined(_MSC_VER)
            __try
            {
                __m256i value = _mm256_abs_epi8(_mm256_set1_epi8(1));// try to execute of AVX2 instructions;
                return true;
            }
            __except(EXCEPTION_EXECUTE_HANDLER)
            {
                return false;
            }
#else
            return true;
#endif
        }

		const bool Enable = SupportedByCPU() && SupportedByOS();
	}
#endif

#ifdef SIMD_VMX_ENABLE
    namespace Vmx
    {
        SIMD_INLINE bool SupportedByCPU()
        {
            return CpuInfo::CheckBit(AT_HWCAP, PPC_FEATURE_HAS_ALTIVEC);
        }

        SIMD_INLINE bool SupportedByOS()
        {
            return true;
        }

        const bool Enable = SupportedByCPU() && SupportedByOS();
    }
#endif

#ifdef SIMD_VSX_ENABLE
    namespace Vsx
    {
        SIMD_INLINE bool SupportedByCPU()
        {
            return CpuInfo::CheckBit(AT_HWCAP, PPC_FEATURE_HAS_VSX);
        }

        SIMD_INLINE bool SupportedByOS()
        {
            return true;
        }

        const bool Enable = SupportedByCPU() && SupportedByOS();
    }
#endif
}

#define SIMD_BASE_FUNC(func) Simd::Base::func

#ifdef SIMD_SSE_ENABLE
#define SIMD_SSE_FUNC(func) Simd::Sse::Enable ? Simd::Sse::func : 
#else
#define SIMD_SSE_FUNC(func) 
#endif

#ifdef SIMD_SSE2_ENABLE
#define SIMD_SSE2_FUNC(func) Simd::Sse2::Enable ? Simd::Sse2::func : 
#else
#define SIMD_SSE2_FUNC(func) 
#endif

#ifdef SIMD_SSSE3_ENABLE
#define SIMD_SSSE3_FUNC(func) Simd::Ssse3::Enable ? Simd::Ssse3::func : 
#else
#define SIMD_SSSE3_FUNC(func) 
#endif

#ifdef SIMD_SSE41_ENABLE
#define SIMD_SSE41_FUNC(func) Simd::Sse41::Enable ? Simd::Sse41::func : 
#else
#define SIMD_SSE41_FUNC(func) 
#endif

#ifdef SIMD_SSE42_ENABLE
#define SIMD_SSE42_FUNC(func) Simd::Sse42::Enable ? Simd::Sse42::func : 
#else
#define SIMD_SSE42_FUNC(func) 
#endif

#ifdef SIMD_AVX_ENABLE
#define SIMD_AVX_FUNC(func) Simd::Avx::Enable ? Simd::Avx::func : 
#else
#define SIMD_AVX_FUNC(func)
#endif

#ifdef SIMD_AVX2_ENABLE
#define SIMD_AVX2_FUNC(func) Simd::Avx2::Enable ? Simd::Avx2::func : 
#else
#define SIMD_AVX2_FUNC(func)
#endif

#ifdef SIMD_VMX_ENABLE
#define SIMD_VMX_FUNC(func) Simd::Vmx::Enable ? Simd::Vmx::func : 
#else
#define SIMD_VMX_FUNC(func)
#endif

#ifdef SIMD_VSX_ENABLE
#define SIMD_VSX_FUNC(func) Simd::Vsx::Enable ? Simd::Vsx::func : 
#else
#define SIMD_VSX_FUNC(func)
#endif

#define SIMD_FUNC0(func) SIMD_BASE_FUNC(func)
#define SIMD_FUNC1(func, EXT1) EXT1(func) SIMD_BASE_FUNC(func)
#define SIMD_FUNC2(func, EXT1, EXT2) EXT1(func) EXT2(func) SIMD_BASE_FUNC(func)
#define SIMD_FUNC3(func, EXT1, EXT2, EXT3) EXT1(func) EXT2(func) EXT3(func) SIMD_BASE_FUNC(func)
#define SIMD_FUNC4(func, EXT1, EXT2, EXT3, EXT4) EXT1(func) EXT2(func) EXT3(func) EXT4(func) SIMD_BASE_FUNC(func)

#endif//__SimdEnable_h__
