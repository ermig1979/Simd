/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#ifndef __SimdSynetPermute_h__
#define __SimdSynetPermute_h__

#include "Simd/SimdMath.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdArray.h"

#include <vector>

namespace Simd
{
    namespace Base
    {
        typedef std::vector<size_t> Shape;

        struct PermuteParam
        {
            static const size_t CountMax = 5;
            Shape shape, order;
            size_t count;
            SimdTensorDataType type;

            PermuteParam(const size_t* s, const size_t* o, size_t c, SimdTensorDataType t)
                : shape(s, s + Simd::Min(c, CountMax))
                , order(o, o + Simd::Min(c, CountMax))
                , count(c)
                , type(t) 
            {
            }

            bool Valid() const
            {
                if (count < 2 || count > CountMax)
                    return false;
                int permute = 0;
                for (size_t i = 0; i < count; ++i)
                {
                    if (order[i] >= count || shape[i] == 0)
                        return false;
                    if (order[i] != i && shape[order[i]] != 1)
                        permute++;
                }
                return permute > 1;
            }
        };

        //-------------------------------------------------------------------------------------------------

        class SynetPermute : public Deletable
        {
        public:
            SynetPermute(const PermuteParam & param);

            size_t InternalBufferSize() const
            {
                return _index.RawSize();
            }

            void Forward(const uint8_t* src, uint8_t* dst);

            typedef void (*PermutePtr)(const uint8_t* src, const Shape& shape, const Shape& stride, uint8_t* dst);

        protected:

            PermuteParam _param;
            Array32i _index;
            size_t _count, _outer, _inner;
            Shape _srcShape, _srcOrder, _srcStride;
            Shape _dstShape, _dstOrder, _dstStride;
            PermutePtr _permute;
        };

        //-------------------------------------------------------------------------------------------------

        void * SynetPermuteInit(const size_t* shape, const size_t* order, size_t count, SimdTensorDataType type);
    }
}

#endif
