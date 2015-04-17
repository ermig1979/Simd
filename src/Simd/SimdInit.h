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
#ifndef __SimdInit_h__
#define __SimdInit_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
#if defined(_MSC_VER)

	template <class T> SIMD_INLINE char GetChar(T value, size_t index)
    {
        return ((char*)&value)[index];
    }

#define SIMD_AS_CHAR(a) char(a)

#define SIMD_AS_2CHARS(a) \
	Simd::GetChar(int16_t(a), 0), Simd::GetChar(int16_t(a), 1)

#define SIMD_AS_4CHARS(a) \
	Simd::GetChar(int32_t(a), 0), Simd::GetChar(int32_t(a), 1), \
	Simd::GetChar(int32_t(a), 2), Simd::GetChar(int32_t(a), 3)

#elif defined(__GNUC__)

#define SIMD_CHAR_AS_LONGLONG(a) (((long long)a) & 0xFF)

#define SIMD_SHORT_AS_LONGLONG(a) (((long long)a) & 0xFFFF)

#define SIMD_INT_AS_LONGLONG(a) (((long long)a) & 0xFFFFFFFF)

#define SIMD_LL_SET1_EPI8(a) \
    SIMD_CHAR_AS_LONGLONG(a) | (SIMD_CHAR_AS_LONGLONG(a) << 8) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 16) | (SIMD_CHAR_AS_LONGLONG(a) << 24) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 32) | (SIMD_CHAR_AS_LONGLONG(a) << 40) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 48) | (SIMD_CHAR_AS_LONGLONG(a) << 56)

#define SIMD_LL_SET2_EPI8(a, b) \
    SIMD_CHAR_AS_LONGLONG(a) | (SIMD_CHAR_AS_LONGLONG(b) << 8) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 16) | (SIMD_CHAR_AS_LONGLONG(b) << 24) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 32) | (SIMD_CHAR_AS_LONGLONG(b) << 40) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 48) | (SIMD_CHAR_AS_LONGLONG(b) << 56)

#define SIMD_LL_SETR_EPI8(a, b, c, d, e, f, g, h) \
    SIMD_CHAR_AS_LONGLONG(a) | (SIMD_CHAR_AS_LONGLONG(b) << 8) | \
    (SIMD_CHAR_AS_LONGLONG(c) << 16) | (SIMD_CHAR_AS_LONGLONG(d) << 24) | \
    (SIMD_CHAR_AS_LONGLONG(e) << 32) | (SIMD_CHAR_AS_LONGLONG(f) << 40) | \
    (SIMD_CHAR_AS_LONGLONG(g) << 48) | (SIMD_CHAR_AS_LONGLONG(h) << 56)

#define SIMD_LL_SET1_EPI16(a) \
    SIMD_SHORT_AS_LONGLONG(a) | (SIMD_SHORT_AS_LONGLONG(a) << 16) | \
    (SIMD_SHORT_AS_LONGLONG(a) << 32) | (SIMD_SHORT_AS_LONGLONG(a) << 48)

#define SIMD_LL_SET2_EPI16(a, b) \
    SIMD_SHORT_AS_LONGLONG(a) | (SIMD_SHORT_AS_LONGLONG(b) << 16) | \
    (SIMD_SHORT_AS_LONGLONG(a) << 32) | (SIMD_SHORT_AS_LONGLONG(b) << 48)

#define SIMD_LL_SETR_EPI16(a, b, c, d) \
    SIMD_SHORT_AS_LONGLONG(a) | (SIMD_SHORT_AS_LONGLONG(b) << 16) | \
    (SIMD_SHORT_AS_LONGLONG(c) << 32) | (SIMD_SHORT_AS_LONGLONG(d) << 48)

#define SIMD_LL_SET1_EPI32(a) \
    SIMD_INT_AS_LONGLONG(a) | (SIMD_INT_AS_LONGLONG(a) << 32)

#define SIMD_LL_SET2_EPI32(a, b) \
    SIMD_INT_AS_LONGLONG(a) | (SIMD_INT_AS_LONGLONG(b) << 32)

#endif//__GNUC__

#if defined(SIMD_SSE2_ENABLE)

#if defined(_MSC_VER)

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

#define SIMD_MM_SETR_EPI16(a0, a1, a2, a3, a4, a5, a6, a7) \
    {SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a2), SIMD_AS_2CHARS(a3), \
    SIMD_AS_2CHARS(a4), SIMD_AS_2CHARS(a5), SIMD_AS_2CHARS(a6), SIMD_AS_2CHARS(a7)}

#define SIMD_MM_SET1_EPI32(a) \
    {SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a)}

#define SIMD_MM_SET2_EPI32(a0, a1) \
    {SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1)}

#define SIMD_MM_SETR_EPI32(a0, a1, a2, a3) \
    {SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a2), SIMD_AS_4CHARS(a3)}

#elif defined(__GNUC__)

#define SIMD_MM_SET1_EPI8(a) \
    {SIMD_LL_SET1_EPI8(a), SIMD_LL_SET1_EPI8(a)}

#define SIMD_MM_SET2_EPI8(a0, a1) \
    {SIMD_LL_SET2_EPI8(a0, a1), SIMD_LL_SET2_EPI8(a0, a1)}

#define SIMD_MM_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af) \
    {SIMD_LL_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7), SIMD_LL_SETR_EPI8(a8, a9, aa, ab, ac, ad, ae, af)}

#define SIMD_MM_SET1_EPI16(a) \
    {SIMD_LL_SET1_EPI16(a), SIMD_LL_SET1_EPI16(a)}

#define SIMD_MM_SET2_EPI16(a0, a1) \
    {SIMD_LL_SET2_EPI16(a0, a1), SIMD_LL_SET2_EPI16(a0, a1)}

#define SIMD_MM_SETR_EPI16(a0, a1, a2, a3, a4, a5, a6, a7) \
    {SIMD_LL_SETR_EPI16(a0, a1, a2, a3), SIMD_LL_SETR_EPI16(a4, a5, a6, a7)}

#define SIMD_MM_SET1_EPI32(a) \
    {SIMD_LL_SET1_EPI32(a), SIMD_LL_SET1_EPI32(a)}

#define SIMD_MM_SET2_EPI32(a0, a1) \
    {SIMD_LL_SET2_EPI32(a0, a1), SIMD_LL_SET2_EPI32(a0, a1)}

#define SIMD_MM_SETR_EPI32(a0, a1, a2, a3) \
    {SIMD_LL_SET2_EPI32(a0, a1), SIMD_LL_SET2_EPI32(a2, a3)}

#endif// defined(_MSC_VER) || defined(__GNUC__)

#endif// SIMD_SSE2_ENABLE

#if defined(SIMD_AVX2_ENABLE)

#if defined(_MSC_VER)

#define SIMD_MM256_SET1_EPI8(a) \
	{SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a)}

#define SIMD_MM256_SET2_EPI8(a0, a1) \
	{SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1)}

#define SIMD_MM256_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf) \
    {SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a2), SIMD_AS_CHAR(a3), \
    SIMD_AS_CHAR(a4), SIMD_AS_CHAR(a5), SIMD_AS_CHAR(a6), SIMD_AS_CHAR(a7), \
    SIMD_AS_CHAR(a8), SIMD_AS_CHAR(a9), SIMD_AS_CHAR(aa), SIMD_AS_CHAR(ab), \
    SIMD_AS_CHAR(ac), SIMD_AS_CHAR(ad), SIMD_AS_CHAR(ae), SIMD_AS_CHAR(af), \
    SIMD_AS_CHAR(b0), SIMD_AS_CHAR(b1), SIMD_AS_CHAR(b2), SIMD_AS_CHAR(b3), \
    SIMD_AS_CHAR(b4), SIMD_AS_CHAR(b5), SIMD_AS_CHAR(b6), SIMD_AS_CHAR(b7), \
    SIMD_AS_CHAR(b8), SIMD_AS_CHAR(b9), SIMD_AS_CHAR(ba), SIMD_AS_CHAR(bb), \
    SIMD_AS_CHAR(bc), SIMD_AS_CHAR(bd), SIMD_AS_CHAR(be), SIMD_AS_CHAR(bf)}

#define SIMD_MM256_SET1_EPI16(a) \
	{SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), \
	SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), \
	SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), \
	SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a)}

#define SIMD_MM256_SET2_EPI16(a0, a1) \
	{SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), \
	SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), \
	SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), \
	SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1)}

#define SIMD_MM256_SETR_EPI16(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af) \
    {SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a2), SIMD_AS_2CHARS(a3), \
    SIMD_AS_2CHARS(a4), SIMD_AS_2CHARS(a5), SIMD_AS_2CHARS(a6), SIMD_AS_2CHARS(a7), \
    SIMD_AS_2CHARS(a8), SIMD_AS_2CHARS(a9), SIMD_AS_2CHARS(aa), SIMD_AS_2CHARS(ab), \
    SIMD_AS_2CHARS(ac), SIMD_AS_2CHARS(ad), SIMD_AS_2CHARS(ae), SIMD_AS_2CHARS(af)}

#define SIMD_MM256_SET1_EPI32(a) \
	{SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), \
	SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a)}

#define SIMD_MM256_SET2_EPI32(a0, a1) \
	{SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), \
	SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1)}

#define SIMD_MM256_SETR_EPI32(a0, a1, a2, a3, a4, a5, a6, a7) \
    {SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a2), SIMD_AS_4CHARS(a3), \
    SIMD_AS_4CHARS(a4), SIMD_AS_4CHARS(a5), SIMD_AS_4CHARS(a6), SIMD_AS_4CHARS(a7)}

#elif defined(__GNUC__)

#define SIMD_MM256_SET1_EPI8(a) \
    {SIMD_LL_SET1_EPI8(a), SIMD_LL_SET1_EPI8(a), \
    SIMD_LL_SET1_EPI8(a), SIMD_LL_SET1_EPI8(a)}

#define SIMD_MM256_SET2_EPI8(a0, a1) \
    {SIMD_LL_SET2_EPI8(a0, a1), SIMD_LL_SET2_EPI8(a0, a1), \
    SIMD_LL_SET2_EPI8(a0, a1), SIMD_LL_SET2_EPI8(a0, a1)}

#define SIMD_MM256_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf) \
    {SIMD_LL_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7), SIMD_LL_SETR_EPI8(a8, a9, aa, ab, ac, ad, ae, af), \
    SIMD_LL_SETR_EPI8(b0, b1, b2, b3, b4, b5, b6, b7), SIMD_LL_SETR_EPI8(b8, b9, ba, bb, bc, bd, be, bf)}

#define SIMD_MM256_SET1_EPI16(a) \
    {SIMD_LL_SET1_EPI16(a), SIMD_LL_SET1_EPI16(a), \
    SIMD_LL_SET1_EPI16(a), SIMD_LL_SET1_EPI16(a)}

#define SIMD_MM256_SET2_EPI16(a0, a1) \
    {SIMD_LL_SET2_EPI16(a0, a1), SIMD_LL_SET2_EPI16(a0, a1), \
    SIMD_LL_SET2_EPI16(a0, a1), SIMD_LL_SET2_EPI16(a0, a1)}

#define SIMD_MM256_SETR_EPI16(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af) \
    {SIMD_LL_SETR_EPI16(a0, a1, a2, a3), SIMD_LL_SETR_EPI16(a4, a5, a6, a7), \
    SIMD_LL_SETR_EPI16(a8, a9, aa, ab), SIMD_LL_SETR_EPI16(ac, ad, ae, af)}

#define SIMD_MM256_SET1_EPI32(a) \
    {SIMD_LL_SET1_EPI32(a), SIMD_LL_SET1_EPI32(a), \
    SIMD_LL_SET1_EPI32(a), SIMD_LL_SET1_EPI32(a)}

#define SIMD_MM256_SET2_EPI32(a0, a1) \
    {SIMD_LL_SET2_EPI32(a0, a1), SIMD_LL_SET2_EPI32(a0, a1), \
    SIMD_LL_SET2_EPI32(a0, a1), SIMD_LL_SET2_EPI32(a0, a1)}

#define SIMD_MM256_SETR_EPI32(a0, a1, a2, a3, a4, a5, a6, a7) \
    {SIMD_LL_SET2_EPI32(a0, a1), SIMD_LL_SET2_EPI32(a2, a3), \
    SIMD_LL_SET2_EPI32(a4, a5), SIMD_LL_SET2_EPI32(a6, a7)}

#endif// defined(_MSC_VER) || defined(__GNUC__)

#endif// SIMD_AVX2_ENABLE

#if defined(SIMD_VMX_ENABLE)

#define SIMD_VEC_SET1_EPI8(a) \
    {a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a}

#define SIMD_VEC_SET2_EPI8(a0, a1) \
    {a0, a1, a0, a1, a0, a1, a0, a1, a0, a1, a0, a1, a0, a1, a0, a1}

#define SIMD_VEC_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af) \
    {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af}

#define SIMD_VEC_SET1_EPI16(a) \
    {a, a, a, a, a, a, a, a}

#define SIMD_VEC_SET2_EPI16(a0, a1) \
    {a0, a1, a0, a1, a0, a1, a0, a1}

#define SIMD_VEC_SETR_EPI16(a0, a1, a2, a3, a4, a5, a6, a7) \
    {a0, a1, a2, a3, a4, a5, a6, a7}

#define SIMD_VEC_SET1_EPI32(a) \
    {a, a, a, a}

#define SIMD_VEC_SET2_EPI32(a0, a1) \
    {a0, a1, a0, a1}

#define SIMD_VEC_SETR_EPI32(a0, a1, a2, a3) \
    {a0, a1, a2, a3}

#define SIMD_VEC_SET1_PS(a) \
    {a, a, a, a}

#define SIMD_VEC_SET2_PS(a0, a1) \
    {a0, a1, a0, a1}

#define SIMD_VEC_SETR_PS(a0, a1, a2, a3) \
    {a0, a1, a2, a3}

#endif//SIMD_VMX_ENABLE
}

#endif//__SimdInit_h__
