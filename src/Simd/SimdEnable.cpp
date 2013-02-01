/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
#if defined(_WIN64) || defined(_WIN32)
	#include <windows.h>
    #include <intrin.h>
#elif defined __GNUC__
    #include <cpuid.h>
#else
    #error Do not know how to detect CPU info
#endif

#include "Simd/SimdEnable.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        bool SupportedByCPU()
        {
#if defined WIN32
            //read CPU info and verify SSE2 support:
            int cpuInfo[4];
            __cpuid(cpuInfo, 1);
            return (cpuInfo[3] & 0x04000000) != 0;
#elif defined __GNUC__
            unsigned int eax = 0;
            unsigned int ebx = 0;
            unsigned int edx = 0;
            unsigned int ecx = 0;
            __get_cpuid(1, &eax, &ebx, &ecx, &edx);
            return edx & bit_SSE2;
#else
    #error Do not know how to detect CPU info
#endif
        }

#if defined(_WIN64) || defined(_WIN32)
        bool SupportedByOS()
        {
            // executing SSE2 instruction:
            __try 
            {
                return _mm_setzero_pd().m128d_f64[0] == 0.0;
            }
            __except(EXCEPTION_EXECUTE_HANDLER) 
            {
                return false;
            }
        }
#else
        bool SupportedByOS()
        {
            return true;
        }
#endif
    }
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_SSE42_ENABLE
    namespace Sse42
    {
        bool SupportedByCPU()
        {
#if defined WIN32
            //read CPU info and verify SSE42 support:
            int cpuInfo[4];
            __cpuid(cpuInfo, 1);
            return (cpuInfo[2] & 0x00100000) != 0;
#elif defined __GNUC__
            unsigned int eax = 0;
            unsigned int ebx = 0;
            unsigned int edx = 0;
            unsigned int ecx = 0;
            __get_cpuid(1, &eax, &ebx, &ecx, &edx);
            return ecx & bit_SSE4_2;
#else
    #error Do not know how to detect CPU info
#endif
        }

#if defined(_WIN64) || defined(_WIN32)
        bool SupportedByOS()
        {
            // executing SSE42 instruction:
            __try 
            {
                _mm_crc32_u8(0, 0);
                return true;
            }
            __except(EXCEPTION_EXECUTE_HANDLER) 
            {
                return true;
            }
        }
#else
        bool SupportedByOS()
        {
            return true;
        }
#endif
    }
#endif// SIMD_SSE42_ENABLE
}
