/*
* Simd Library.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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

#if defined(_MSC_VER)
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#elif defined(__GNUC__)
#include <cpuid.h>
#else
#error Do not know how to detect CPU info
#endif

#include "Simd/SimdTypes.h"
#include "Simd/SimdDefs.h"

namespace Simd
{
    namespace Cpuid
    {
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
            //	Ordinary:
            //Edx:
            SSE2 = 1 << 26,

            //Ecx:
            SSSE3 =	1 << 9,
            SSE41 = 1 << 19,
            SSE42 = 1 << 20,
            OSXSAVE = 1 << 27,
            AVX = 1 << 28,

            //	Extended:
            //Ebx:
            AVX2 = 1 << 5,
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
#endif// SIMD_SSE2_ENABLE

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
#endif// SIMD_SSSE3_ENABLE

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
#endif// SIMD_SSE41_ENABLE

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
#endif// SIMD_SSE42_ENABLE

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
#endif// SIMD_AVX_ENABLE

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
#endif// SIMD_AVX2_ENABLE
}
#endif//__SimdEnable_h__
