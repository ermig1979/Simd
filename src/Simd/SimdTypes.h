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
#ifndef __SimdTypes_h__
#define __SimdTypes_h__

#include "Simd/SimdDefs.h"

#ifdef SIMD_OWN_STDINT
#if (_MSC_VER < 1300)
	typedef signed char       int8_t;
	typedef signed short      int16_t;
	typedef signed int        int32_t;
	typedef unsigned char     uint8_t;
	typedef unsigned short    uint16_t;
	typedef unsigned int      uint32_t;
#else
	typedef signed __int8     int8_t;
	typedef signed __int16    int16_t;
	typedef signed __int32    int32_t;
	typedef unsigned __int8   uint8_t;
	typedef unsigned __int16  uint16_t;
	typedef unsigned __int32  uint32_t;
#endif
	typedef signed __int64    int64_t;
	typedef unsigned __int64  uint64_t;
#endif//SIMD_OWN_STDINT
	
namespace Simd
{
	typedef unsigned char uchar;
	typedef unsigned short ushort;
	typedef unsigned int uint;
}
#endif//__SimdTypes_h__