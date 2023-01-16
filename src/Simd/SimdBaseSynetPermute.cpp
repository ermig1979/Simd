/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdSynetPermute.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SIMD_INLINE Shape Stride(const Shape& shape, const Shape& order)
        {
            Shape buf(shape.size(), 1), out(shape.size(), 1);
            for (ptrdiff_t i = shape.size() - 2; i >= 0; i--)
                buf[i] = buf[i + 1] * shape[i + 1];
            for (size_t i = 0; i < shape.size(); ++i)
                out[order[i]] = buf[i];
            return out;
        }

        SIMD_INLINE void EraseBatch(Shape &order)
        {
            order.erase(order.begin());
            for (size_t j = 0; j < order.size(); ++j)
                order[j]--;
        }

        SIMD_INLINE Shape CompactOrder(Shape order)
        {
            for (size_t i = 1; i < order.size();)
            {
                if (order[i] == order[i - 1] + 1)
                {
                    order.erase(order.begin() + i);
                    for (size_t j = 0; j < order.size(); ++j)
                        if (order[j] > order[i - 1])
                            order[j]--;
                }
                else
                    ++i;
            }
            return order;
        }

        SIMD_INLINE Shape CompactShape(const Shape& shape, const Shape& order)
        {
            Shape compact;
            for (size_t i = 0; i < shape.size(); ++i)
            {
                size_t dim = shape[order[i]];
                if (i && order[i] == order[i - 1] + 1)
                    compact.back() *= dim;
                else
                    compact.push_back(dim);
            }
            return compact;
        }

        //-------------------------------------------------------------------------------------------------

        template<class T> void Permute2(const uint8_t* src_, const Shape& shape, const Shape& stride, uint8_t* dst_)
        {
            const T* src = (const T*)src_;
            T* dst = (T*)dst_;
            for (size_t i = 0; i < shape[0]; ++i)
            {
                for (size_t j = 0; j < shape[1]; ++j)
                {
                    *dst++ = src[i * stride[0] + j * stride[1]];
                }
            }
        }

        template<class T> void Permute3(const uint8_t* src_, const Shape& shape, const Shape& stride, uint8_t* dst_)
        {
            const T* src = (const T*)src_;
            T* dst = (T*)dst_;
            for (size_t i = 0; i < shape[0]; ++i)
            {
                for (size_t j = 0; j < shape[1]; ++j)
                {
                    for (size_t k = 0; k < shape[2]; ++k)
                    {
                        *dst++ = src[i * stride[0] + j * stride[1] + k * stride[2]];
                    }
                }
            }
        }

        template<class T> void Permute4(const uint8_t* src_, const Shape& shape, const Shape& stride, uint8_t* dst_)
        {
            const T* src = (const T*)src_;
            T* dst = (T*)dst_;
            for (size_t i = 0; i < shape[0]; ++i)
            {
                for (size_t j = 0; j < shape[1]; ++j)
                {
                    for (size_t k = 0; k < shape[2]; ++k)
                    {
                        for (size_t l = 0; l < shape[3]; ++l)
                        {
                            *dst++ = src[i * stride[0] + j * stride[1] + k * stride[2] + l * stride[3]];
                        }
                    }
                }
            }
        }

        template<class T> void Permute5(const uint8_t* src_, const Shape& shape, const Shape& stride, uint8_t* dst_)
        {
            const T* src = (const T*)src_;
            T* dst = (T*)dst_;
            for (size_t i = 0; i < shape[0]; ++i)
            {
                for (size_t j = 0; j < shape[1]; ++j)
                {
                    for (size_t k = 0; k < shape[2]; ++k)
                    {
                        for (size_t l = 0; l < shape[3]; ++l)
                        {
                            for (size_t m = 0; m < shape[4]; ++m)
                            {
                                *dst++ = src[i * stride[0] + j * stride[1] + k * stride[2] + l * stride[3] + m * stride[4]];
                            }
                        }
                    }
                }
            }
        }

        template<class T> SynetPermute::PermutePtr GetPermute(size_t count)
        {
            switch (count)
            {
            case 2: return Permute2<T>;
            case 3: return Permute3<T>;
            case 4: return Permute4<T>;
            case 5: return Permute5<T>;
            default:
                return NULL;
            }
        }

        static SynetPermute::PermutePtr GetPermute(SimdTensorDataType type, size_t count)
        {
            switch (type)
            {
            case SimdTensorData32f:
            case SimdTensorData32i:
                return GetPermute<uint32_t>(count);
            case SimdTensorData8i:
            case SimdTensorData8u:
                return GetPermute<uint8_t>(count);
            case SimdTensorData16b:
            case SimdTensorData16f:
                return GetPermute<uint16_t>(count);
            default:
                return NULL;
            }
        }


        //-------------------------------------------------------------------------------------------------

        SynetPermute::SynetPermute(const PermuteParam& p)
            : _param(p)
        {
            _count = p.count;
            _srcShape = p.shape;
            _dstOrder = p.order;
            _srcOrder = p.order;
            for (size_t i = 0; i < _count; ++i)
            {
                _srcOrder[_dstOrder[i]] = i;
                _dstShape.push_back(_srcShape[_dstOrder[i]]);
            }
            size_t count = 0;
            for (size_t i = 0; i < _count; ++i)
                if (i == 0 || _dstOrder[i] != _dstOrder[i - 1] + 1)
                    count++;
            if (count != _count/* && p.align > 1*/)
            {
                _count = count;
                Shape dstShape = _dstShape;
                _dstShape = CompactShape(_srcShape, _dstOrder);
                _srcShape = CompactShape(dstShape, _srcOrder);
                _dstOrder = CompactOrder(_dstOrder);
                _srcOrder = CompactOrder(_srcOrder);
            }
            if (_dstOrder[0] == 0/* && p.align > 1*/)
            {
                _batch = _dstShape[0];
                _stride = Stride(_dstShape, _dstOrder)[0] * p.PixelSize();
                _srcShape.erase(_srcShape.begin());
                _dstShape.erase(_dstShape.begin());
                EraseBatch(_srcOrder);
                EraseBatch(_dstOrder);
                _count--;
            }
            else
            {
                _batch = 1;
                _stride = 0;
            }
            _srcStride = Stride(_srcShape, _srcOrder);
            _dstStride = Stride(_dstShape, _dstOrder);
            _permute = GetPermute(p.type, _count);
        }


        //-------------------------------------------------------------------------------------------------

        void SynetPermute::Forward(const uint8_t* src, uint8_t* dst)
        {
            for (size_t b = 0; b < _batch; ++b)
            {
                _permute(src, _dstShape, _srcStride, dst);
                src += _stride;
                dst += _stride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetPermuteInit(const size_t* shape, const size_t* order, size_t count, SimdTensorDataType type)
        {
            PermuteParam param(shape, order, count, type, 1);
            if (!param.Valid())
                return NULL;
            return new SynetPermute(param);
        }
    }
#endif
}
