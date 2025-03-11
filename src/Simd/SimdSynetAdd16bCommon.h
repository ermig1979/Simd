/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#ifndef __SimdSynetAdd16bCommon_h__
#define __SimdSynetAdd16bCommon_h__

#include "Simd/SimdBFloat16.h"

namespace Simd
{
    namespace Base
    {
        template <class S, class D> SIMD_INLINE D Convert16b(const S& src)
        {
            return (D)src;
        }

        template <> SIMD_INLINE float Convert16b(const uint16_t& src)
        {
            return BFloat16ToFloat32(src);
        }

        template <> SIMD_INLINE uint16_t Convert16b(const float& src)
        {
            return Float32ToBFloat16(src);
        }

        //-------------------------------------------------------------------------------------------------

        template <typename A, typename B, typename D> void Add16b(const A& a, const B& b, D& dst)
        {
            float _a = Convert16b<A, float>(a);
            float _b = Convert16b<B, float>(b);
            dst = Convert16b<float, D>(_a + _b);
        }

        template <typename S, typename D> void NormBias16b(const S& src, float norm, float bias, D& dst)
        {
            float _src = Convert16b<S, float>(src);
            dst = Convert16b<float, D>(_src * norm + bias);
        }

        template <typename S, typename D> void Norm16b(const S& src, float norm, D& dst)
        {
            float _src = Convert16b<S, float>(src);
            dst = Convert16b<float, D>(_src * norm);
        }

        template <typename S, typename D> void Bias16b(const S& src, float bias, D& dst)
        {
            float _src = Convert16b<S, float>(src);
            dst = Convert16b<float, D>(_src + bias);
        }
    }
}

#endif
