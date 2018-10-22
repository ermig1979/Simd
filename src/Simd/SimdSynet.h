/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#ifndef __SimdSynet_h__
#define __SimdSynet_h__

#include "Simd/SimdMath.h"

namespace Simd
{
    namespace Base
    {
        template <SimdSynetEltwiseOperationType type> float SynetEltwiseLayerForward(float a, float b);

        template <> SIMD_INLINE float SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct>(float a, float b)
        {
            return a * b;
        }

        template <> SIMD_INLINE float SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax>(float a, float b)
        {
            return Simd::Max(a, b);
        }

        template <> SIMD_INLINE float SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin>(float a, float b)
        {
            return Simd::Min(a, b);
        }

        SIMD_INLINE float SynetFusedLayerForward0(float x, float s)
        {
            return (x - ::abs(x))*s + Simd::Max(0.0f, x);
        }

        SIMD_INLINE float SynetFusedLayerForward1(float x, float s, float b)
        {
            return Simd::Max(0.0f, -x)*s + b + Simd::Max(0.0f, x);
        }
    }
}

#endif//__SimdSynet_h__
