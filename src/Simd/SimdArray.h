/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#ifndef __SimdArray_h__
#define __SimdArray_h__

#include "Simd/SimdMemory.h"

namespace Simd
{
	template <class T> struct Array
	{
		T * const data;
		size_t const size;

		Array(size_t s = 0)
			: data(0)
			, size(0)
		{
			Resize(s);
		}

		~Array()
		{
			if (data)
				Simd::Free(data);
		}

		void Resize(size_t s)
		{
			*(size_t*)&size = s;
			*(T**)&data = (T*)Simd::Allocate(size * sizeof(T));
		}

		void Clear()
		{
			::memset(data, 0, size * sizeof(T));
		}

		T & operator[] (size_t i)
		{
			return data[i];
		}

		const T & operator[] (size_t i) const
		{
			return data[i];
		}
	};
}

#endif//__SimdArray_h__
