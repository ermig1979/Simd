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

#include "Simd/SimdLib.h"
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

#define SIMD_LL_SET1_EPI16(a) \
    SIMD_SHORT_AS_LONGLONG(a) | (SIMD_SHORT_AS_LONGLONG(a) << 16) | \
    (SIMD_SHORT_AS_LONGLONG(a) << 32) | (SIMD_SHORT_AS_LONGLONG(a) << 48)

#define SIMD_LL_SET2_EPI16(a, b) \
    SIMD_SHORT_AS_LONGLONG(a) | (SIMD_SHORT_AS_LONGLONG(b) << 16) | \
    (SIMD_SHORT_AS_LONGLONG(a) << 32) | (SIMD_SHORT_AS_LONGLONG(b) << 48)

#define SIMD_LL_SET1_EPI32(a) \
    SIMD_INT_AS_LONGLONG(a) | (SIMD_INT_AS_LONGLONG(a) << 32)

#define SIMD_LL_SET2_EPI32(a, b) \
    SIMD_INT_AS_LONGLONG(a) | (SIMD_INT_AS_LONGLONG(b) << 32)

#endif//__GNUC__

#ifdef SIMD_SSE2_ENABLE

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

#define SIMD_MM_SET1_EPI16(a) \
    {SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), \
    SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a)}

#define SIMD_MM_SET2_EPI16(a0, a1) \
    {SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), \
    SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1)}

#define SIMD_MM_SET1_EPI32(a) \
    {SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a)}

#define SIMD_MM_SET2_EPI32(a0, a1) \
    {SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1)}

#elif defined(__GNUC__)

#define SIMD_MM_SET1_EPI8(a) \
    {SIMD_LL_SET1_EPI8(a), SIMD_LL_SET1_EPI8(a)}

#define SIMD_MM_SET2_EPI8(a0, a1) \
    {SIMD_LL_SET2_EPI8(a0, a1), SIMD_LL_SET2_EPI8(a0, a1)}

#define SIMD_MM_SET1_EPI16(a) \
    {SIMD_LL_SET1_EPI16(a), SIMD_LL_SET1_EPI16(a)}

#define SIMD_MM_SET2_EPI16(a0, a1) \
    {SIMD_LL_SET2_EPI16(a0, a1), SIMD_LL_SET2_EPI16(a0, a1)}

#define SIMD_MM_SET1_EPI32(a) \
    {SIMD_LL_SET1_EPI32(a), SIMD_LL_SET1_EPI32(a)}

#define SIMD_MM_SET2_EPI32(a0, a1) \
    {SIMD_LL_SET2_EPI32(a0, a1), SIMD_LL_SET2_EPI32(a0, a1)}

#endif// defined(_MSC_VER) || defined(__GNUC__)

#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE

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

#define SIMD_MM256_SET1_EPI32(a) \
	{SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), \
	SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a)}

#define SIMD_MM256_SET2_EPI32(a0, a1) \
	{SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), \
	SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1)}

#elif defined(__GNUC__)

#define SIMD_MM256_SET1_EPI8(a) \
    {SIMD_LL_SET1_EPI8(a), SIMD_LL_SET1_EPI8(a), \
    SIMD_LL_SET1_EPI8(a), SIMD_LL_SET1_EPI8(a)}

#define SIMD_MM256_SET2_EPI8(a0, a1) \
    {SIMD_LL_SET2_EPI8(a0, a1), SIMD_LL_SET2_EPI8(a0, a1), \
    SIMD_LL_SET2_EPI8(a0, a1), SIMD_LL_SET2_EPI8(a0, a1)}

#define SIMD_MM256_SET1_EPI16(a) \
    {SIMD_LL_SET1_EPI16(a), SIMD_LL_SET1_EPI16(a), \
    SIMD_LL_SET1_EPI16(a), SIMD_LL_SET1_EPI16(a)}

#define SIMD_MM256_SET2_EPI16(a0, a1) \
    {SIMD_LL_SET2_EPI16(a0, a1), SIMD_LL_SET2_EPI16(a0, a1), \
    SIMD_LL_SET2_EPI16(a0, a1), SIMD_LL_SET2_EPI16(a0, a1)}

#define SIMD_MM256_SET1_EPI32(a) \
    {SIMD_LL_SET1_EPI32(a), SIMD_LL_SET1_EPI32(a), \
    SIMD_LL_SET1_EPI32(a), SIMD_LL_SET1_EPI32(a)}

#define SIMD_MM256_SET2_EPI32(a0, a1) \
    {SIMD_LL_SET2_EPI32(a0, a1), SIMD_LL_SET2_EPI32(a0, a1), \
    SIMD_LL_SET2_EPI32(a0, a1), SIMD_LL_SET2_EPI32(a0, a1)}

#endif// defined(_MSC_VER) || defined(__GNUC__)

#endif// SIMD_AVX2_ENABLE
}

#endif//__SimdInit_h__
