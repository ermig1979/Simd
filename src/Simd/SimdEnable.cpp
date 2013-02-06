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
#include "Simd/SimdTypes.h"

namespace Simd
{
	const uint EDX_SSE2_BIT = 1 << 26;
	const uint ECX_SSE42_BIT = 1 << 20;
	const uint ECX_OSXSAVE_BIT = 1 << 27;
	const uint ECX_AVX_BIT = 1 << 28;

	enum Register
	{
		Eax = 0, 
		Ebx = 1,
		Ecx = 2, 
		Edx = 3,
	};

	inline bool CheckCpuidBits(uint level, Register index, uint bits)
	{
		unsigned int registers[4];
#if defined(_WIN64) || defined(_WIN32)
		__cpuid((int*)registers, level);
#elif defined __GNUC__
		__get_cpuid(level, registers + Eax, registers + Ebx, registers + Ecx, registers + Edx);
#else
#error Do not know how to detect CPU info!
#endif
		return (registers[index] & bits) == bits;
	}


#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        bool SupportedByCPU()
        {
			return CheckCpuidBits(1, Edx, EDX_SSE2_BIT);
        }

        bool SupportedByOS()
        {
#if defined(_WIN64) || defined(_WIN32)
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
    }
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_SSE42_ENABLE
    namespace Sse42
    {
        bool SupportedByCPU()
        {
			return CheckCpuidBits(1, Ecx, ECX_SSE42_BIT);
        }

        bool SupportedByOS()
        {
#if defined(_WIN64) || defined(_WIN32)
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
    }
#endif// SIMD_SSE42_ENABLE

#ifdef SIMD_AVX_ENABLE
	namespace Avx
	{
		bool SupportedByCPU()
		{
			return CheckCpuidBits(1, Ecx, ECX_OSXSAVE_BIT | ECX_AVX_BIT);
		}

		bool SupportedByOS()
		{
#if defined(_WIN64) || defined(_WIN32)
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
	}
#endif// SIMD_AVX_ENABLE
}
