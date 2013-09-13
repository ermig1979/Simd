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
#ifndef __SimdAddFeatureDifference_h__
#define __SimdAddFeatureDifference_h__

#include "Simd/SimdTypes.h"

namespace Simd
{
	namespace Base
	{
		void AddFeatureDifference(const uchar * value, size_t valueStride, size_t width, size_t height, 
			const uchar * lo, size_t loStride, const uchar * hi, size_t hiStride,
            ushort weight, uchar * difference, size_t differenceStride);
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
        void AddFeatureDifference(const uchar * value, size_t valueStride, size_t width, size_t height, 
            const uchar * lo, size_t loStride, const uchar * hi, size_t hiStride,
            ushort weight, uchar * difference, size_t differenceStride);
	}
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        void AddFeatureDifference(const uchar * value, size_t valueStride, size_t width, size_t height, 
            const uchar * lo, size_t loStride, const uchar * hi, size_t hiStride,
            ushort weight, uchar * difference, size_t differenceStride);
    }
#endif// SIMD_AVX2_ENABLE

    void AddFeatureDifference(const uchar * value, size_t valueStride, size_t width, size_t height, 
        const uchar * lo, size_t loStride, const uchar * hi, size_t hiStride,
        ushort weight, uchar * difference, size_t differenceStride);
}
#endif//__SimdAddFeatureDifference_h__
