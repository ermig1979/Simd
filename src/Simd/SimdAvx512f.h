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
#ifndef __SimdAvx512f_h__
#define __SimdAvx512f_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
		void NeuralProductSum(const float * a, const float * b, size_t size, float * sum);

		void NeuralAddVectorMultipliedByValue(const float * src, size_t size, const float * value, float * dst);

		void NeuralAddVector(const float * src, size_t size, float * dst);

		void NeuralAddValue(const float * value, float * dst, size_t size);

		void NeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst);

		void NeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst);
	}
#endif// SIMD_AVX512F_ENABLE
}
#endif//__SimdAvx512f_h__
