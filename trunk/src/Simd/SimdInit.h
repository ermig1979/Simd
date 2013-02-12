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
#ifndef __SimdInit_h__
#define __SimdInit_h__

#include "Simd/SimdTypes.h"

namespace Simd
{
    template <class T> SIMD_INLINE char GetChar(T value, size_t index)
    {
        return ((char*)&value)[index];
    }

#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        SIMD_INLINE __m128i _mm_set2_epi8(char a0, char a1)
        {
            return _mm_unpacklo_epi8(_mm_set1_epi8(a0), _mm_set1_epi8(a1));
        }
        
        SIMD_INLINE __m128i _mm_set2_epi16(short a0, short a1)
        {
            return _mm_unpacklo_epi16(_mm_set1_epi16(a0), _mm_set1_epi16(a1));
        }

        SIMD_INLINE __m128i _mm_set2_epi32(int a0, int a1)
        {
            return _mm_unpacklo_epi32(_mm_set1_epi32(a0), _mm_set1_epi32(a1));
        }

		SIMD_INLINE __m128 _mm_set2_ps(float a0, float a1)
		{
			return _mm_unpacklo_ps(_mm_set_ps1(a0), _mm_set_ps1(a1));
		}

#if defined(_M_X64) || defined(__GNUC__)

#define SIMD_MM_SET1_EPI8(a) \
    _mm_set1_epi8((char)(a))

#define SIMD_MM_SET2_EPI8(a0, a1) \
    Simd::Sse2::_mm_set2_epi8((char)(a0), (char)(a1))
    
#define SIMD_MM_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af) \
    _mm_setr_epi8((char)(a0), (char)(a1), (char)(a2), (char)(a3), (char)(a4), (char)(a5), (char)(a6), (char)(a7), \
    (char)(a8), (char)(a9), (char)(aa), (char)(ab), (char)(ac), (char)(ad), (char)(ae), (char)(af))

#define SIMD_MM_SET1_EPI16(a) \
    _mm_set1_epi16((short)(a))
    
#define SIMD_MM_SET2_EPI16(a0, a1) \
    Simd::Sse2::_mm_set2_epi16((short)(a0), (short)(a1))

#define SIMD_MM_SETR_EPI16(a0, a1, a2, a3, a4, a5, a6, a7) \
    _mm_setr_epi16((short)(a0), (short)(a1), (short)(a2), (short)(a3), (short)(a4), (short)(a5), (short)(a6), (short)(a7))

#define SIMD_MM_SET1_EPI32(a) \
    _mm_set1_epi32((int)(a))
    
#define SIMD_MM_SET2_EPI32(a0, a1) \
    Simd::Sse2::_mm_set2_epi32((int)(a0), (int)(a1))

#define SIMD_MM_SETR_EPI32(a0, a1, a2, a3) \
    _mm_setr_epi32((int)(a0), (int)(a1), (int)(a2), (int)(a3))

#define SIMD_MM_SET1_PS(a) \
	_mm_set_ps1((float)(a))

#define SIMD_MM_SET2_PS(a0, a1) \
	Simd::Sse2::_mm_set2_ps((float)(a0), (float)(a1))

#define SIMD_MM_SETR_PS(a0, a1, a2, a3) \
	_mm_setr_ps((float)(a0), (float)(a1), (float)(a2), (float)(a3))

#define SIMD_MM_SET1_PD(a) \
	_mm_set1_pd((double)(a))

#define SIMD_MM_SETR_PD(a0, a1) \
	_mm_setr_pd((double)(a0), (double)(a1))

#else //defined(_M_X64) || defined(__GNUC__)

#define SIMD_AS_CHAR(a) char(a)

#define SIMD_AS_2CHARS(a) \
    Simd::GetChar(int16_t(a), 0), Simd::GetChar(int16_t(a), 1)

#define SIMD_AS_4CHARS(a) \
    Simd::GetChar(int32_t(a), 0), Simd::GetChar(int32_t(a), 1), \
    Simd::GetChar(int32_t(a), 2), Simd::GetChar(int32_t(a), 3)

#define SIMD_MM_SET1_EPI8(a) \
    {SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
    SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
    SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
    SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a)}
    
#define SIMD_MM_SET2_EPI8(a0, a1) \
    {SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
    SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
    SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
    SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1)}

#define SIMD_MM_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af) \
    {SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a2), SIMD_AS_CHAR(a3), \
    SIMD_AS_CHAR(a4), SIMD_AS_CHAR(a5), SIMD_AS_CHAR(a6), SIMD_AS_CHAR(a7), \
    SIMD_AS_CHAR(a8), SIMD_AS_CHAR(a9), SIMD_AS_CHAR(aa), SIMD_AS_CHAR(ab), \
    SIMD_AS_CHAR(ac), SIMD_AS_CHAR(ad), SIMD_AS_CHAR(ae), SIMD_AS_CHAR(af)}

#define SIMD_MM_SET1_EPI16(a) \
    {SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), \
    SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a)}

#define SIMD_MM_SET2_EPI16(a0, a1) \
    {SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), \
    SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1)}

#define SIMD_MM_SETR_EPI16(a0, a1, a2, a3, a4, a5, a6, a7, a8) \
    {SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a2), SIMD_AS_2CHARS(a3), \
    SIMD_AS_2CHARS(a4), SIMD_AS_2CHARS(a5), SIMD_AS_2CHARS(a6), SIMD_AS_2CHARS(a7)}

#define SIMD_MM_SET1_EPI32(a) \
    {SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a)}

#define SIMD_MM_SET2_EPI32(a0, a1) \
    {SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1)}

#define SIMD_MM_SETR_EPI32(a0, a1, a2, a3) \
    {SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a2), SIMD_AS_4CHARS(a3)}

#define SIMD_MM_SET1_PS(a) \
	{float(a), float(a), float(a), float(a)}

#define SIMD_MM_SET2_PS(a0, a1) \
	{float(a0), float(a1), float(a0), float(a1)}

#define SIMD_MM_SETR_PS(a0, a1, a2, a3) \
	{float(a0), float(a1), float(a2), float(a3)}

#define SIMD_MM_SET1_PD(a) \
	{double(a), double(a)}

#define SIMD_MM_SETR_PD(a0, a1) \
	{double(a0), double(a1)}

#endif //defined(_M_X64) || defined(__GNUC__)

    }
#endif// SIMD_SSE2_ENABLE
}

#endif//__SimdInit_h__