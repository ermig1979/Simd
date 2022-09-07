/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdRecursiveBilateralFilter.h"

namespace Simd
{
    RbfParam::RbfParam(size_t w, size_t h, size_t c, const float* s, const float* r, size_t a)
        : width(w)
        , height(h)
        , channels(c)
        , spatial(*s)
        , range(*r)
        , align(a)
    {
    }

    bool RbfParam::Valid() const
    {
        return
            height > 0 &&
            width > 0 &&
            channels > 0 && channels <= 4 &&
            align >= sizeof(float);
    }

    //---------------------------------------------------------------------------------------------

    RecursiveBilateralFilter::RecursiveBilateralFilter(const RbfParam& param)
        : _param(param)
    {
    }

    //---------------------------------------------------------------------------------------------

    namespace Base
    {
        RecursiveBilateralFilterDefault::RecursiveBilateralFilterDefault(const RbfParam& param)
            :Simd::RecursiveBilateralFilter(param)
        {
        }

        void RecursiveBilateralFilterDefault::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {

        }

        //-----------------------------------------------------------------------------------------

        void* RecursiveBilateralFilterInit(size_t width, size_t height, size_t channels, const float* sigmaSpatial, const float* sigmaRange)
        {
            RbfParam param(width, height, channels, sigmaSpatial, sigmaRange, sizeof(void*));
            if (!param.Valid())
                return NULL;
            return new RecursiveBilateralFilterDefault(param);
        }
    }
}

