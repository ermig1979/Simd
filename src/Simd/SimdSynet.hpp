/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#ifndef __SimdSynet_hpp__
#define __SimdSynet_hpp__

#include "Simd/SimdLib.hpp"

#include <vector>

namespace Simd
{
    /*! @ingroup cpp_synet
        \short Tensor shape type definition.
     */
    typedef std::vector<size_t> Shape;

    //-------------------------------------------------------------------------------------------------

    /*! @ingroup cpp_synet

        \short The SynetAdd16b class is a C++ wrapper of 16-bit (BF16/FP32) element-wise addition.

        The class wraps C API functions ::SimdSynetAdd16bInit and ::SimdSynetAdd16bForward.
        It adds two tensors with equal shapes. BF16 values are converted to FP32 before addition
        and converted back after addition when the corresponding tensor type is ::SimdTensorData16b:
        \verbatim
        for(i = 0; i < shapeSize; ++i)
        {
            A = aType == SimdTensorData16b ? BFloat16ToFloat32(a[i]) : a[i];
            B = bType == SimdTensorData16b ? BFloat16ToFloat32(b[i]) : b[i];
            D = A + B;
            dst[i] = dstType == SimdTensorData16b ? Float32ToBFloat16(D) : D;
        }
        \endverbatim

        The current implementation creates a context only for equal input shapes, FP32/BF16 input
        and output tensor types, and ::SimdTensorFormatUnknown, ::SimdTensorFormatNchw or
        ::SimdTensorFormatNhwc tensor format. Call Init() before Forward(). Use Enable() to check
        that a context was created. The context is released by Clear() or by the destructor.

        Using example:
        \verbatim
        #include "Simd/SimdSynet.hpp"

        int main()
        {
            const size_t n = 64;
            std::vector<float> a(n, 1.0f), b(n, 2.0f), dst(n, 0.0f);
            Simd::Shape dims = Simd::Shape({ n });

            Simd::SynetAdd16b add;
            add.Init(dims, SimdTensorData32f, dims, SimdTensorData32f, SimdTensorData32f, SimdTensorFormatNhwc);
            if (add.Enable())
                add.Forward((const uint8_t*)a.data(), (const uint8_t*)b.data(), (uint8_t*)dst.data());

            return 0;
        }
        \endverbatim
    */
    class SynetAdd16b
    {
    public:
        /*!
            Creates a new empty SynetAdd16b class.
        */
        SynetAdd16b()
            : _context(NULL)
        {
        }

        /*!
            SynetAdd16b class destructor. Releases internal context.
        */
        virtual ~SynetAdd16b()
        {
            Clear();
        }

        /*!
            Initializes (or re-initializes) element-wise addition of two FP32/BF16 tensors.

            Creates an internal context with using of function ::SimdSynetAdd16bInit.
            The context is recreated only if input tensor shapes were changed.

            \note This function is a C++ wrapper for function ::SimdSynetAdd16bInit.

            \param [in] aShape - a shape of input A tensor.
            \param [in] aType - a type of input A tensor. Can be ::SimdTensorData32f or ::SimdTensorData16b.
            \param [in] bShape - a shape of input B tensor.
            \param [in] bType - a type of input B tensor. Can be ::SimdTensorData32f or ::SimdTensorData16b.
            \param [in] dstType - a type of output tensor. Can be ::SimdTensorData32f or ::SimdTensorData16b.
            \param [in] format - a format of input / output tensors.
        */
        SIMD_INLINE void Init(const Shape & aShape, SimdTensorDataType aType, const Shape & bShape, SimdTensorDataType bType, SimdTensorDataType dstType, SimdTensorFormatType format)
        {
            if (_aShape != aShape || _bShape != bShape)
            {
                Clear();
                _aShape = aShape;
                _bShape = bShape;
                _context = SimdSynetAdd16bInit(_aShape.data(), _aShape.size(), aType,
                    _bShape.data(), _bShape.size(), bType, dstType, format);
            }
        }

        /*!
            Checks that the internal addition context was created.

            \return true if the context exists and Forward() can be called.
        */
        SIMD_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        /*!
            Performs element-wise addition of two FP32/BF16 tensors.

            The function adds corresponding elements of input tensors A and B.
            The actual data types, tensor shape and output type are stored in the context created by Init().

            \note This function is a C++ wrapper for function ::SimdSynetAdd16bForward.

            \param [in] a - a pointer to input A tensor.
            \param [in] b - a pointer to input B tensor.
            \param [out] dst - a pointer to output tensor.
        */
        SIMD_INLINE void Forward(const uint8_t * a, const uint8_t * b, uint8_t * dst)
        {
            if (_context)
                SimdSynetAdd16bForward(_context, a, b, dst);
        }

        /*!
            Releases internal context and clears stored tensor shapes.
        */
        SIMD_INLINE void Clear()
        {
            if (_context)
                SimdRelease(_context), _context = NULL;
            _aShape.clear();
            _bShape.clear();
        }

    private:
        void * _context;
        Shape _aShape, _bShape;
    };
}

#endif
