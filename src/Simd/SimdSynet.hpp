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

    //-------------------------------------------------------------------------------------------------

    /*! @ingroup cpp_synet

        \short The SynetQuantizedAdd class is a C++ wrapper of quantized (UINT8/FP32) element-wise addition.

        The class wraps C API functions ::SimdSynetQuantizedAddInit and ::SimdSynetQuantizedAddForward.
        It dequantizes UINT8 inputs as (value - zero)*scale, adds the two values, applies activation
        if it is specified and converts the result to FP32 or UINT8 output. FP32 inputs and outputs
        ignore the corresponding quantization zero. Algorithm's details for UINT8 output:
        \verbatim
        for(i = 0; i < size; ++i)
        {
            value = Activate((a[i] - aZero)*aScale + (b[i] - bZero)*bScale, actType, actParams);
            dst[i] = RestrictRange(Round(value/dstScale) + dstZero, 0, 255);
        }
        \endverbatim

        The current implementation creates a context only for equal input shapes and FP32/UINT8 input
        and output tensor types. Supported optimized activation types are ::SimdConvolutionActivationIdentity
        and ::SimdConvolutionActivationRelu. Call Init() before Forward(). Use Enable() to check
        that a context was created. The context is released by Clear() or by the destructor.

        Using example:
        \verbatim
        #include "Simd/SimdSynet.hpp"

        int main()
        {
            const size_t n = 64;
            std::vector<uint8_t> a(n, 50), b(n, 80), dst(n, 0);
            Simd::Shape dims = Simd::Shape({ n });
            float aScale = 0.010f, bScale = 0.020f, dstScale = 0.015f;
            int32_t aZero = 47, bZero = 30, dstZero = 38;

            Simd::SynetQuantizedAdd add;
            add.Init(dims, SimdTensorData8u, aScale, aZero, dims, SimdTensorData8u, bScale, bZero,
                SimdConvolutionActivationIdentity, NULL, SimdTensorData8u, dstScale, dstZero);
            if (add.Enable())
                add.Forward(a.data(), b.data(), dst.data());

            return 0;
        }
        \endverbatim
    */
    class SynetQuantizedAdd
    {
    public:
        /*!
            Creates a new empty SynetQuantizedAdd class.
        */
        SynetQuantizedAdd()
            : _context(NULL)
        {
        }

        /*!
            SynetQuantizedAdd class destructor. Releases internal context.
        */
        virtual ~SynetQuantizedAdd()
        {
            Clear();
        }

        /*!
            Initializes (or re-initializes) element-wise quantized addition of two UINT8/FP32 tensors.

            Creates an internal context with using of function ::SimdSynetQuantizedAddInit.
            The context is recreated only if input tensor shapes were changed.

            \note This function is a C++ wrapper for function ::SimdSynetQuantizedAddInit.

            \param [in] aShape - a shape of input A tensor.
            \param [in] aType - a type of input A tensor. Can be ::SimdTensorData32f or ::SimdTensorData8u.
            \param [in] aScale - a quantization scale of input A tensor.
            \param [in] aZero - a quantization zero of input A tensor.
            \param [in] bShape - a shape of input B tensor.
            \param [in] bType - a type of input B tensor. Can be ::SimdTensorData32f or ::SimdTensorData8u.
            \param [in] bScale - a quantization scale of input B tensor.
            \param [in] bZero - a quantization zero of input B tensor.
            \param [in] actType - an activation function type applied after addition.
            \param [in] actParams - a pointer to activation function parameters. Can be NULL.
            \param [in] dstType - a type of output tensor. Can be ::SimdTensorData32f or ::SimdTensorData8u.
            \param [in] dstScale - an output quantization scale.
            \param [in] dstZero - an output quantization zero.
        */
        SIMD_INLINE void Init(const Shape & aShape, SimdTensorDataType aType, float aScale, int32_t aZero,
            const Shape & bShape, SimdTensorDataType bType, float bScale, int32_t bZero,
            SimdConvolutionActivationType actType, const float * actParams,
            SimdTensorDataType dstType, float dstScale, int32_t dstZero)
        {
            if (_aShape != aShape || _bShape != bShape)
            {
                Clear();
                _aShape = aShape;
                _bShape = bShape;
                _context = SimdSynetQuantizedAddInit(_aShape.data(), _aShape.size(), aType, &aScale, aZero,
                    _bShape.data(), _bShape.size(), bType, &bScale, bZero,
                    actType, actParams, dstType, &dstScale, dstZero);
            }
        }

        /*!
            Checks that the internal quantized addition context was created.

            \return true if the context exists and Forward() can be called.
        */
        SIMD_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        /*!
            Performs element-wise quantized addition of two UINT8/FP32 tensors.

            The function adds corresponding elements of input tensors A and B with dequantization,
            optional activation and output quantization. The actual data types, tensor shape,
            quantization parameters and activation type are stored in the context created by Init().

            \note This function is a C++ wrapper for function ::SimdSynetQuantizedAddForward.

            \param [in] a - a pointer to input A tensor.
            \param [in] b - a pointer to input B tensor.
            \param [out] dst - a pointer to output tensor.
        */
        SIMD_INLINE void Forward(const uint8_t * a, const uint8_t * b, uint8_t * dst)
        {
            if (_context)
                SimdSynetQuantizedAddForward(_context, a, b, dst);
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

    //-------------------------------------------------------------------------------------------------

    /*! @ingroup cpp_synet

        \short The SynetQuantizedMul class is a C++ wrapper of quantized (UINT8/FP32) element-wise multiplication.

        The class wraps C API functions ::SimdSynetQuantizedMulInit and ::SimdSynetQuantizedMulForward.
        It dequantizes UINT8 inputs as (value - zero)*scale, multiplies the two values and converts
        the result to FP32 or UINT8 output. FP32 inputs and outputs ignore the corresponding
        quantization zero. Algorithm's details for UINT8 output:
        \verbatim
        for(i = 0; i < size; ++i)
        {
            _a = (a[i] - aZero)*aScale;
            _b = (b[i] - bZero)*bScale;
            dst[i] = RestrictRange(Round((_a * _b)/dstScale) + dstZero, 0, 255);
        }
        \endverbatim

        The current implementation creates a context for compatible input shapes (equal or broadcast)
        and FP32/UINT8 input and output tensor types. Call Init() before Forward(). Use Enable() to check
        that a context was created. The context is released by Clear() or by the destructor.

        Using example:
        \verbatim
        #include "Simd/SimdSynet.hpp"

        int main()
        {
            const size_t n = 64;
            std::vector<uint8_t> a(n, 50), b(n, 80), dst(n, 0);
            Simd::Shape dims = Simd::Shape({ n });
            float aScale = 0.010f, bScale = 0.020f, dstScale = 0.015f;
            int32_t aZero = 47, bZero = 30, dstZero = 38;

            Simd::SynetQuantizedMul mul;
            mul.Init(dims, SimdTensorData8u, aScale, aZero, dims, SimdTensorData8u, bScale, bZero,
                SimdTensorData8u, dstScale, dstZero);
            if (mul.Enable())
                mul.Forward(a.data(), b.data(), dst.data());

            return 0;
        }
        \endverbatim
    */
    class SynetQuantizedMul
    {
    public:
        /*!
            Creates a new empty SynetQuantizedMul class.
        */
        SynetQuantizedMul()
            : _context(NULL)
        {
        }

        /*!
            SynetQuantizedMul class destructor. Releases internal context.
        */
        virtual ~SynetQuantizedMul()
        {
            Clear();
        }

        /*!
            Initializes (or re-initializes) element-wise quantized multiplication of two UINT8/FP32 tensors.

            Creates an internal context with using of function ::SimdSynetQuantizedMulInit.
            The context is recreated only if input tensor shapes were changed.

            \note This function is a C++ wrapper for function ::SimdSynetQuantizedMulInit.

            \param [in] aShape - a shape of input A tensor.
            \param [in] aType - a type of input A tensor. Can be ::SimdTensorData32f or ::SimdTensorData8u.
            \param [in] aScale - a quantization scale of input A tensor.
            \param [in] aZero - a quantization zero of input A tensor.
            \param [in] bShape - a shape of input B tensor.
            \param [in] bType - a type of input B tensor. Can be ::SimdTensorData32f or ::SimdTensorData8u.
            \param [in] bScale - a quantization scale of input B tensor.
            \param [in] bZero - a quantization zero of input B tensor.
            \param [in] dstType - a type of output tensor. Can be ::SimdTensorData32f or ::SimdTensorData8u.
            \param [in] dstScale - an output quantization scale.
            \param [in] dstZero - an output quantization zero.
        */
        SIMD_INLINE void Init(const Shape & aShape, SimdTensorDataType aType, float aScale, int32_t aZero,
            const Shape & bShape, SimdTensorDataType bType, float bScale, int32_t bZero,
            SimdTensorDataType dstType, float dstScale, int32_t dstZero)
        {
            if (_aShape != aShape || _bShape != bShape)
            {
                Clear();
                _aShape = aShape;
                _bShape = bShape;
                _context = SimdSynetQuantizedMulInit(_aShape.data(), _aShape.size(), aType, &aScale, aZero,
                    _bShape.data(), _bShape.size(), bType, &bScale, bZero, dstType, &dstScale, dstZero);
            }
        }

        /*!
            Checks that the internal quantized multiplication context was created.

            \return true if the context exists and Forward() can be called.
        */
        SIMD_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        /*!
            Performs element-wise quantized multiplication of two UINT8/FP32 tensors.

            The function multiplies corresponding elements of input tensors A and B with dequantization
            and output quantization. The actual data types, tensor shape and quantization parameters
            are stored in the context created by Init().

            \note This function is a C++ wrapper for function ::SimdSynetQuantizedMulForward.

            \param [in] a - a pointer to input A tensor.
            \param [in] b - a pointer to input B tensor.
            \param [out] dst - a pointer to output tensor.
        */
        SIMD_INLINE void Forward(const uint8_t * a, const uint8_t * b, uint8_t * dst)
        {
            if (_context)
                SimdSynetQuantizedMulForward(_context, a, b, dst);
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

    //-------------------------------------------------------------------------------------------------

    /*! @ingroup cpp_synet

        \short The SynetGatherElements class is a C++ wrapper of ONNX-style GatherElements.

        The class wraps C API functions ::SimdSynetGatherElementsInit, ::SimdSynetGatherElementsSetIndex,
        ::SimdSynetGatherElementsInternalBufferSize and ::SimdSynetGatherElementsForward.
        It gathers elements from an input tensor along one dimension according to an index tensor.
        It supports FP32, BF16 and UINT8 data tensors and INT32 or INT64 index tensors. The input tensor shape is:
        \verbatim
        outer[0] * ... * outer[outer.size() - 1] * srcCount * inner
        \endverbatim
        The index and output tensor shape is:
        \verbatim
        outer[0] * ... * outer[outer.size() - 1] * idxCount * inner
        \endverbatim

        Algorithm's details:
        \verbatim
        for(b = 0; b < outer[0]*...*outer[outer.size() - 1]; ++b)
            for(c = 0; c < idxCount; ++c)
                for(i = 0; i < inner; ++i)
                {
                    ic = idx[b, c, i];
                    if (ic < 0)
                        ic += srcCount;
                    dst[b, c, i] = src[b, ic, i];
                }
        \endverbatim

        If \a indexConst is ::SimdTrue, constant indexes can be analyzed by SetIndex() to avoid
        repeated negative-index checks and to reduce repeated outer index processing when possible.
        Call Init() before Forward(). Use Enable() to check that a context was created.
        The context is released by Clear() or by the destructor.

        Using example:
        \verbatim
        #include "Simd/SimdSynet.hpp"

        int main()
        {
            const size_t srcCount = 4, inner = 1, idxCount = 3;
            std::vector<float> src(8), dst(6);
            std::vector<int32_t> idx(6);
            for (size_t i = 0; i < src.size(); ++i)
                src[i] = float(i);
            idx[0] = 0; idx[1] = 2; idx[2] = 1;
            idx[3] = 3; idx[4] = 1; idx[5] = 0;
            Simd::Shape outer = Simd::Shape({ 2 });

            Simd::SynetGatherElements gather;
            gather.Init(SimdTensorData32f, SimdTensorData32i, SimdFalse, 1, outer, srcCount, inner, idxCount);
            if (gather.Enable())
                gather.Forward((const uint8_t*)src.data(), (const uint8_t*)idx.data(), (uint8_t*)dst.data());

            return 0;
        }
        \endverbatim
    */
    class SynetGatherElements
    {
    public:
        /*!
            Creates a new empty SynetGatherElements class.
        */
        SynetGatherElements()
            : _context(NULL)
            , _srcCount(0)
            , _inner(0)
            , _idxCount(0)
        {
        }

        /*!
            SynetGatherElements class destructor. Releases internal context.
        */
        virtual ~SynetGatherElements()
        {
            Clear();
        }

        /*!
            Initializes (or re-initializes) a gather-elements context.

            Creates an internal context with using of function ::SimdSynetGatherElementsInit.
            The context is recreated only if outer shape, srcCount, inner or idxCount were changed.

            \note This function is a C++ wrapper for function ::SimdSynetGatherElementsInit.

            \param [in] dataType - a type of input and output tensor. It can be ::SimdTensorData32f, ::SimdTensorData16b or ::SimdTensorData8u.
            \param [in] indexType - a type of index tensor. It can be ::SimdTensorData32i or ::SimdTensorData64i.
            \param [in] indexConst - a flag indicating that index tensor is constant and can be set once.
            \param [in] indexUsers - a number of consumers sharing the same constant index tensor.
            \param [in] outer - outer shape dimensions before the gathered dimension.
            \param [in] srcCount - a length of the gathered dimension in the input tensor.
            \param [in] inner - a product of dimensions after the gathered dimension.
            \param [in] idxCount - a length of the gathered dimension in the index and output tensors.
        */
        SIMD_INLINE void Init(SimdTensorDataType dataType, SimdTensorDataType indexType, SimdBool indexConst, size_t indexUsers,
            const Shape & outer, size_t srcCount, size_t inner, size_t idxCount)
        {
            if (_outer != outer || _srcCount != srcCount || _inner != inner || _idxCount != idxCount)
            {
                Clear();
                _outer = outer;
                _srcCount = srcCount;
                _inner = inner;
                _idxCount = idxCount;
                _context = SimdSynetGatherElementsInit(dataType, indexType, indexConst, indexUsers,
                    _outer.data(), _outer.size(), _srcCount, _inner, _idxCount);
            }
        }

        /*!
            Checks that the internal gather-elements context was created.

            \return true if the context exists and Forward() can be called.
        */
        SIMD_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        /*!
            Gets the size in bytes of internal storage used by the gather-elements context.

            \note This function is a C++ wrapper for function ::SimdSynetGatherElementsInternalBufferSize.

            \return size of internal buffer in bytes used inside gather elements algorithm.
        */
        SIMD_INLINE size_t InternalBufferSize() const
        {
            return _context ? SimdSynetGatherElementsInternalBufferSize(_context) : 0;
        }

        /*!
            Sets and analyzes constant gather-elements indexes.

            The function has an effect only when the context was created with \a indexConst equal to ::SimdTrue.

            \note This function is a C++ wrapper for function ::SimdSynetGatherElementsSetIndex.

            \param [in] idx - a pointer to INT32 or INT64 index tensor.
        */
        SIMD_INLINE void SetIndex(const uint8_t * idx)
        {
            if (_context)
                SimdSynetGatherElementsSetIndex(_context, idx);
        }

        /*!
            Performs gather-elements forward propagation.

            The function gathers elements from \a src according to \a idx. If SetIndex() was called,
            the context can use the analysis results, but \a idx must still point to the index tensor
            in the current implementation. Negative indexes are interpreted relative to srcCount.

            \note This function is a C++ wrapper for function ::SimdSynetGatherElementsForward.

            \param [in] src - a pointer to input tensor.
            \param [in] idx - a pointer to INT32 or INT64 index tensor.
            \param [out] dst - a pointer to output tensor.
        */
        SIMD_INLINE void Forward(const uint8_t * src, const uint8_t * idx, uint8_t * dst)
        {
            if (_context)
                SimdSynetGatherElementsForward(_context, src, idx, dst);
        }

        /*!
            Releases internal context and clears stored tensor parameters.
        */
        SIMD_INLINE void Clear()
        {
            if (_context)
                SimdRelease(_context), _context = NULL;
            _outer.clear();
            _srcCount = 0;
            _inner = 0;
            _idxCount = 0;
        }

    private:
        void * _context;
        Shape _outer;
        size_t _srcCount, _inner, _idxCount;
    };

    //-------------------------------------------------------------------------------------------------

    /*! @ingroup cpp_synet

        \short The SynetPermute class is a C++ wrapper of tensor dimension permutation.

        The class wraps C API functions ::SimdSynetPermuteInit, ::SimdSynetPermuteInternalBufferSize
        and ::SimdSynetPermuteForward. It reorders tensor dimensions. If input shape is
        shape[0..count-1], then output dimension i has size shape[order[i]]:
        \verbatim
        dstShape[i] = srcShape[order[i]].
        \endverbatim

        Supported dimension count is from 2 to 5. Dimensions with size 1 can be skipped by the
        implementation, but the requested permutation must change at least two non-unit dimensions.
        Supported tensor types are FP32, INT32, INT8, UINT8, BF16 and FP16.
        Call Init() before Forward(). Use Enable() to check that a context was created.
        The context is released by Clear() or by the destructor.

        Using example:
        \verbatim
        #include "Simd/SimdSynet.hpp"

        int main()
        {
            const size_t n = 4, m = 8;
            std::vector<float> src(n * m), dst(n * m);
            for (size_t i = 0; i < src.size(); ++i)
                src[i] = float(i);
            Simd::Shape shape = Simd::Shape({ n, m });
            Simd::Shape order = Simd::Shape({ 1, 0 });

            Simd::SynetPermute permute;
            permute.Init(shape, order, SimdTensorData32f);
            if (permute.Enable())
                permute.Forward((const uint8_t*)src.data(), (uint8_t*)dst.data());

            return 0;
        }
        \endverbatim
    */
    class SynetPermute
    {
    public:
        /*!
            Creates a new empty SynetPermute class.
        */
        SynetPermute()
            : _context(NULL)
        {
        }

        /*!
            SynetPermute class destructor. Releases internal context.
        */
        virtual ~SynetPermute()
        {
            Clear();
        }

        /*!
            Initializes (or re-initializes) a tensor permutation context.

            Creates an internal context with using of function ::SimdSynetPermuteInit.
            The context is recreated only if input tensor shape or output dimension order were changed.

            \note This function is a C++ wrapper for function ::SimdSynetPermuteInit.

            \param [in] shape - a shape of input tensor. Dimension count must be from 2 to 5.
            \param [in] order - an output dimension order. The size must be equal to shape size and contain a permutation of dimension indices.
            \param [in] type - an input and output tensor data type.
        */
        SIMD_INLINE void Init(const Shape & shape, const Shape & order, SimdTensorDataType type)
        {
            if (_shape != shape || _order != order)
            {
                Clear();
                _shape = shape;
                _order = order;
                _context = SimdSynetPermuteInit(_shape.data(), _order.data(), _shape.size(), type);
            }
        }

        /*!
            Checks that the internal permutation context was created.

            \return true if the context exists and Forward() can be called.
        */
        SIMD_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        /*!
            Gets the size in bytes of internal storage used by the permutation context.

            \note This function is a C++ wrapper for function ::SimdSynetPermuteInternalBufferSize.

            \return size of internal buffer in bytes used inside permutation algorithm.
        */
        SIMD_INLINE size_t InternalBufferSize() const
        {
            return _context ? SimdSynetPermuteInternalBufferSize(_context) : 0;
        }

        /*!
            Performs tensor dimension permutation.

            The function reorders dimensions of \a src according to the order stored in the context
            created by Init() and writes the result to \a dst.

            \note This function is a C++ wrapper for function ::SimdSynetPermuteForward.

            \param [in] src - a pointer to the input tensor bytes.
            \param [out] dst - a pointer to the output tensor bytes.
        */
        SIMD_INLINE void Forward(const uint8_t * src, uint8_t * dst)
        {
            if (_context)
                SimdSynetPermuteForward(_context, src, dst);
        }

        /*!
            Releases internal context and clears stored tensor parameters.
        */
        SIMD_INLINE void Clear()
        {
            if (_context)
                SimdRelease(_context), _context = NULL;
            _shape.clear();
            _order.clear();
        }

    private:
        void * _context;
        Shape _shape, _order;
    };

    //-------------------------------------------------------------------------------------------------

    /*! @ingroup cpp_synet

        \short The SynetInnerProduct32f class is a C++ wrapper of FP32 inner product (matrix multiplication).

        The class wraps C API functions ::SimdSynetInnerProduct32fInit, ::SimdSynetInnerProduct32fInternalBufferSize,
        ::SimdSynetInnerProduct32fExternalBufferSize, ::SimdSynetInnerProduct32fSetParams and
        ::SimdSynetInnerProduct32fForward. It computes C = A*B, optionally adds bias and applies activation:
        \verbatim
        for(i = 0; i < M; ++i)
            for(j = 0; j < N; ++j)
            {
                sum = bias ? bias[j] : 0;
                for(k = 0; k < K; ++k)
                    sum += A[i, k] * (transB ? B[j, k] : B[k, j]);
                C[i, j] = Activate(sum, activation, params);
            }
        \endverbatim

        When \a constB is ::SimdTrue, matrix B must be supplied to SetParams() and can be reordered
        or cached inside the context. Call Init() and SetParams() before Forward(). Use Enable() to check
        that a context was created. The context is released by Clear() or by the destructor.

        Using example:
        \verbatim
        #include "Simd/SimdSynet.hpp"

        int main()
        {
            const size_t M = 4, N = 8, K = 16;
            std::vector<float> A(M * K), B(K * N), C(M * N), bias(N, 0.0f);
            for (size_t i = 0; i < A.size(); ++i)
                A[i] = float(i) * 0.01f;
            for (size_t i = 0; i < B.size(); ++i)
                B[i] = float(i) * 0.02f;

            Simd::SynetInnerProduct32f innerProduct;
            innerProduct.Init(M, N, K, SimdFalse, SimdTrue, SimdTrue, SimdConvolutionActivationIdentity);
            if (innerProduct.Enable())
            {
                innerProduct.SetParams(B.data(), NULL, bias.data(), NULL);
                innerProduct.Forward(A.data(), NULL, NULL, C.data());
            }

            return 0;
        }
        \endverbatim
    */
    class SynetInnerProduct32f
    {
    public:
        /*!
            Creates a new empty SynetInnerProduct32f class.
        */
        SynetInnerProduct32f()
            : _context(NULL)
            , _M(0)
            , _N(0)
            , _K(0)
            , _transB(SimdFalse)
            , _constB(SimdFalse)
            , _bias(SimdFalse)
            , _activation(SimdConvolutionActivationIdentity)
        {
        }

        /*!
            SynetInnerProduct32f class destructor. Releases internal context.
        */
        virtual ~SynetInnerProduct32f()
        {
            Clear();
        }

        /*!
            Initializes (or re-initializes) an FP32 inner-product context.

            Creates an internal context with using of function ::SimdSynetInnerProduct32fInit.
            The context is recreated only if matrix sizes or inner-product flags were changed.

            \note This function is a C++ wrapper for function ::SimdSynetInnerProduct32fInit.

            \param [in] M - a height of A and C matrices.
            \param [in] N - a width of B and C matrices.
            \param [in] K - a width of A and height of B matrices.
            \param [in] transB - a flag indicating that B is stored as N*K instead of K*N.
            \param [in] constB - a flag indicating that matrix B is constant and can be set once.
            \param [in] bias - a flag to add bias to output matrix C.
            \param [in] activation - an activation function type used after inner product.
        */
        SIMD_INLINE void Init(size_t M, size_t N, size_t K, SimdBool transB, SimdBool constB, SimdBool bias, SimdConvolutionActivationType activation)
        {
            if (_M != M || _N != N || _K != K || _transB != transB || _constB != constB || _bias != bias || _activation != activation)
            {
                Clear();
                _M = M;
                _N = N;
                _K = K;
                _transB = transB;
                _constB = constB;
                _bias = bias;
                _activation = activation;
                _context = SimdSynetInnerProduct32fInit(_M, _N, _K, _transB, _constB, _bias, _activation);
            }
        }

        /*!
            Checks that the internal inner-product context was created.

            \return true if the context exists and Forward() can be called.
        */
        SIMD_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        /*!
            Gets the size of internal storage used by the inner-product context.

            The returned value is a number of FP32 elements.

            \note This function is a C++ wrapper for function ::SimdSynetInnerProduct32fInternalBufferSize.

            \return a number of FP32 elements used by internal buffers.
        */
        SIMD_INLINE size_t InternalBufferSize() const
        {
            return _context ? SimdSynetInnerProduct32fInternalBufferSize(_context) : 0;
        }

        /*!
            Gets the size of caller-provided temporary buffer for FP32 inner product.

            The returned value is a number of FP32 elements. The current FP32 implementations do not
            require an external buffer and return 0, but callers can use this value when allocating
            the \a buf argument of Forward().

            \note This function is a C++ wrapper for function ::SimdSynetInnerProduct32fExternalBufferSize.

            \return a number of FP32 elements required for external temporary buffer.
        */
        SIMD_INLINE size_t ExternalBufferSize() const
        {
            return _context ? SimdSynetInnerProduct32fExternalBufferSize(_context) : 0;
        }

        /*!
            Sets weights, bias and activation parameters for FP32 inner product.

            This function must be called before Forward(). If \a constB was ::SimdTrue during
            initialization, \a weight provides matrix B and the implementation may reorder and store it internally.
            If \a internal is not NULL, ::SimdTrue means the weights were copied/reordered into the context;
            ::SimdFalse means the original \a weight pointer can be used by later forward calls and must remain valid.

            \note This function is a C++ wrapper for function ::SimdSynetInnerProduct32fSetParams.

            \param [in] weight - a pointer to FP32 matrix B weights.
            \param [out] internal - a pointer to a flag receiving weight storage mode. Can be NULL.
            \param [in] bias - a pointer to FP32 bias array with N elements. Can be NULL.
            \param [in] params - a pointer to FP32 parameters of activation function (see ::SimdConvolutionActivationType). Can be NULL when activation does not require parameters.
        */
        SIMD_INLINE void SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params)
        {
            if (_context)
                SimdSynetInnerProduct32fSetParams(_context, weight, internal, bias, params);
        }

        /*!
            Performs FP32 inner-product forward propagation.

            The function computes C = A*B, optionally adds bias and applies activation stored in the
            context created by Init() and SetParams(). If B is constant, it can be NULL when it was
            set by SetParams(). The \a buf argument can be NULL (it causes usage of internal buffer).

            \note This function is a C++ wrapper for function ::SimdSynetInnerProduct32fForward.

            \param [in] A - a pointer to FP32 A matrix with M*K elements.
            \param [in] B - a pointer to FP32 B matrix. Can be NULL if B is constant.
            \param [out] buf - a pointer to external temporary FP32 buffer. Can be NULL.
            \param [out] C - a pointer to FP32 output matrix with M*N elements.
        */
        SIMD_INLINE void Forward(const float * A, const float * B, float * buf, float * C)
        {
            if (_context)
                SimdSynetInnerProduct32fForward(_context, A, B, buf, C);
        }

        /*!
            Releases internal context and clears stored inner-product parameters.
        */
        SIMD_INLINE void Clear()
        {
            if (_context)
                SimdRelease(_context), _context = NULL;
            _M = 0;
            _N = 0;
            _K = 0;
            _transB = SimdFalse;
            _constB = SimdFalse;
            _bias = SimdFalse;
            _activation = SimdConvolutionActivationIdentity;
        }

    private:
        void * _context;
        size_t _M, _N, _K;
        SimdBool _transB, _constB, _bias;
        SimdConvolutionActivationType _activation;
    };

    //-------------------------------------------------------------------------------------------------

    /*! @ingroup cpp_synet

        \short The SynetInnerProduct16b class is a C++ wrapper of BF16/FP32 inner product (matrix multiplication).

        The class wraps C API functions ::SimdSynetInnerProduct16bInit, ::SimdSynetInnerProduct16bInternalBufferSize,
        ::SimdSynetInnerProduct16bExternalBufferSize, ::SimdSynetInnerProduct16bInfo, ::SimdSynetInnerProduct16bSetParams and
        ::SimdSynetInnerProduct16bForward. It computes C = A*B with FP32 accumulation, optionally adds bias and applies
        activation. A, B and C can be FP32 or BF16 according to \a typeA, \a typeB and \a typeC:
        \verbatim
        for(i = 0; i < M; ++i)
            for(j = 0; j < N; ++j)
            {
                sum = bias ? bias[j] : 0;
                for(k = 0; k < K; ++k)
                    sum += A[i, k] * (transB ? B[j, k] : B[k, j]);
                C[i, j] = ConvertToTypeC(Activate(sum, activation, params));
            }
        \endverbatim

        When \a constB is ::SimdTrue, matrix B must be supplied to SetParams() in FP32 form and is converted
        or reordered into internal storage. Call Init() and SetParams() before Forward(). Use Enable() to check
        that a context was created. The context is released by Clear() or by the destructor.

        Using example:
        \verbatim
        #include "Simd/SimdSynet.hpp"

        int main()
        {
            const size_t M = 4, N = 8, K = 16;
            std::vector<float> A(M * K), B(K * N), C(M * N), bias(N, 0.0f);
            for (size_t i = 0; i < A.size(); ++i)
                A[i] = float(i) * 0.01f;
            for (size_t i = 0; i < B.size(); ++i)
                B[i] = float(i) * 0.02f;

            Simd::SynetInnerProduct16b innerProduct;
            innerProduct.Init(M, N, K, SimdTensorData32f, SimdTensorData32f, SimdTensorData32f,
                SimdFalse, SimdTrue, SimdTrue, SimdConvolutionActivationIdentity);
            if (innerProduct.Enable())
            {
                innerProduct.SetParams(B.data(), bias.data(), NULL);
                innerProduct.Forward((const uint8_t*)A.data(), NULL, NULL, (uint8_t*)C.data());
            }

            return 0;
        }
        \endverbatim
    */
    class SynetInnerProduct16b
    {
    public:
        /*!
            Creates a new empty SynetInnerProduct16b class.
        */
        SynetInnerProduct16b()
            : _context(NULL)
            , _M(0)
            , _N(0)
            , _K(0)
            , _typeA(SimdTensorData32f)
            , _typeB(SimdTensorData32f)
            , _typeC(SimdTensorData32f)
            , _transB(SimdFalse)
            , _constB(SimdFalse)
            , _bias(SimdFalse)
            , _activation(SimdConvolutionActivationIdentity)
        {
        }

        /*!
            SynetInnerProduct16b class destructor. Releases internal context.
        */
        virtual ~SynetInnerProduct16b()
        {
            Clear();
        }

        /*!
            Initializes (or re-initializes) a BF16/FP32 inner-product context.

            Creates an internal context with using of function ::SimdSynetInnerProduct16bInit.
            The context is recreated only if matrix sizes, tensor types or inner-product flags were changed.

            \note This function is a C++ wrapper for function ::SimdSynetInnerProduct16bInit.

            \param [in] M - a height of A and C matrices.
            \param [in] N - a width of B and C matrices.
            \param [in] K - a width of A and height of B matrices.
            \param [in] typeA - a type of A matrix. It can be ::SimdTensorData32f or ::SimdTensorData16b.
            \param [in] typeB - a type of B matrix. It can be ::SimdTensorData32f or ::SimdTensorData16b.
            \param [in] typeC - a type of C matrix. It can be ::SimdTensorData32f or ::SimdTensorData16b.
            \param [in] transB - a flag indicating that B is stored as N*K instead of K*N.
            \param [in] constB - a flag indicating that matrix B is constant and can be set once.
            \param [in] bias - a flag to add bias to output matrix C.
            \param [in] activation - an activation function type used after inner product.
        */
        SIMD_INLINE void Init(size_t M, size_t N, size_t K, SimdTensorDataType typeA, SimdTensorDataType typeB, SimdTensorDataType typeC,
            SimdBool transB, SimdBool constB, SimdBool bias, SimdConvolutionActivationType activation)
        {
            if (_M != M || _N != N || _K != K || _typeA != typeA || _typeB != typeB || _typeC != typeC ||
                _transB != transB || _constB != constB || _bias != bias || _activation != activation)
            {
                Clear();
                _M = M;
                _N = N;
                _K = K;
                _typeA = typeA;
                _typeB = typeB;
                _typeC = typeC;
                _transB = transB;
                _constB = constB;
                _bias = bias;
                _activation = activation;
                _context = SimdSynetInnerProduct16bInit(_M, _N, _K, _typeA, _typeB, _typeC, _transB, _constB, _bias, _activation);
            }
        }

        /*!
            Checks that the internal inner-product context was created.

            \return true if the context exists and Forward() can be called.
        */
        SIMD_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        /*!
            Gets the size in bytes of internal storage used by the inner-product context.

            The returned value reports internal temporary storage, reordered constant weights, copied bias and copied
            activation parameters.

            \note This function is a C++ wrapper for function ::SimdSynetInnerProduct16bInternalBufferSize.

            \return a number of bytes used by internal buffers.
        */
        SIMD_INLINE size_t InternalBufferSize() const
        {
            return _context ? SimdSynetInnerProduct16bInternalBufferSize(_context) : 0;
        }

        /*!
            Gets the size in bytes of caller-provided temporary buffer for BF16/FP32 inner product.

            The returned value depends on matrix types and implementation. It covers temporary BF16 copies of FP32 inputs,
            packed non-constant B matrices, FP32 accumulation buffers and optional post-processing buffers. It can be used
            when allocating the \a buf argument of Forward().

            \note This function is a C++ wrapper for function ::SimdSynetInnerProduct16bExternalBufferSize.

            \return a number of bytes required for external temporary buffer.
        */
        SIMD_INLINE size_t ExternalBufferSize() const
        {
            return _context ? SimdSynetInnerProduct16bExternalBufferSize(_context) : 0;
        }

        /*!
            Gets a short description of the selected BF16/FP32 inner-product implementation.

            The returned string contains the implementation extension, algorithm name and parameter summary. The returned
            pointer is owned by the context and remains valid until the next call of this function or until the context
            is released.

            \note This function is a C++ wrapper for function ::SimdSynetInnerProduct16bInfo.

            \return a string with description of internal implementation. NULL if the context was not created.
        */
        SIMD_INLINE const char * Info() const
        {
            return _context ? SimdSynetInnerProduct16bInfo(_context) : NULL;
        }

        /*!
            Sets weights, bias and activation parameters for BF16/FP32 inner product.

            This function must be called before Forward(). If \a constB was ::SimdTrue during
            initialization, \a weight provides matrix B in FP32 form and the implementation converts it to BF16 and may
            reorder it into internal storage. Bias is copied to an internal FP32 array; when \a bias is NULL, zeros are
            used. Activation parameters are copied or expanded to the internal FP32 array according to
            ::SimdConvolutionActivationType.

            \note This function is a C++ wrapper for function ::SimdSynetInnerProduct16bSetParams.

            \param [in] weight - a pointer to FP32 matrix B weights. Can be NULL only when B is not constant.
            \param [in] bias - a pointer to FP32 bias array with N elements. Can be NULL.
            \param [in] params - a pointer to FP32 parameters of activation function (see ::SimdConvolutionActivationType). Can be NULL when activation does not require parameters.
        */
        SIMD_INLINE void SetParams(const float * weight, const float * bias, const float * params)
        {
            if (_context)
                SimdSynetInnerProduct16bSetParams(_context, weight, bias, params);
        }

        /*!
            Performs BF16/FP32 inner-product forward propagation.

            The function converts FP32 A or B inputs to BF16 when requested by the context, uses BF16 inputs directly
            otherwise, accumulates the matrix product in FP32, adds bias, applies activation and writes FP32 or BF16
            output according to \a typeC. If B is constant, it can be NULL when it was set by SetParams().
            The \a buf argument can be NULL (it causes usage of internal buffer).

            \note This function is a C++ wrapper for function ::SimdSynetInnerProduct16bForward.

            \param [in] A - a pointer to A matrix. Actual element type is defined by \a typeA in initialization.
            \param [in] B - a pointer to B matrix. Can be NULL if B is constant.
            \param [out] buf - a pointer to external temporary byte buffer. Can be NULL.
            \param [out] C - a pointer to output matrix. Actual element type is defined by \a typeC in initialization.
        */
        SIMD_INLINE void Forward(const uint8_t * A, const uint8_t * B, uint8_t * buf, uint8_t * C)
        {
            if (_context)
                SimdSynetInnerProduct16bForward(_context, A, B, buf, C);
        }

        /*!
            Releases internal context and clears stored inner-product parameters.
        */
        SIMD_INLINE void Clear()
        {
            if (_context)
                SimdRelease(_context), _context = NULL;
            _M = 0;
            _N = 0;
            _K = 0;
            _typeA = SimdTensorData32f;
            _typeB = SimdTensorData32f;
            _typeC = SimdTensorData32f;
            _transB = SimdFalse;
            _constB = SimdFalse;
            _bias = SimdFalse;
            _activation = SimdConvolutionActivationIdentity;
        }

    private:
        void * _context;
        size_t _M, _N, _K;
        SimdTensorDataType _typeA, _typeB, _typeC;
        SimdBool _transB, _constB, _bias;
        SimdConvolutionActivationType _activation;
    };

    //-------------------------------------------------------------------------------------------------

    /*! @ingroup cpp_synet

        \short The SynetQuantizedInnerProduct class is a C++ wrapper of quantized UINT8/INT8 inner product (matrix multiplication).

        The class wraps C API functions ::SimdSynetQuantizedInnerProductInit, ::SimdSynetQuantizedInnerProductInternalBufferSize,
        ::SimdSynetQuantizedInnerProductExternalBufferSize, ::SimdSynetQuantizedInnerProductInfo,
        ::SimdSynetQuantizedInnerProductSetParams and ::SimdSynetQuantizedInnerProductForward.
        It computes C = A*B for UINT8 A, INT8 B and UINT8 C with per-channel B scales, optional INT32 bias
        and output requantization. Algorithm's details (transB = false, bias = true):
        \verbatim
        for(i = 0; i < M; ++i)
            for(j = 0; j < N; ++j)
            {
                sum = bias[j] - aZero*Sum(B[:,j]);
                for(k = 0; k < K; ++k)
                    sum += A[i,k] * B[k,j];
                C[i,j] = RestrictRange(Round(sum*aScale*bScale[j]/cScale) + cZero, 0, 255);
            }
        \endverbatim

        The current implementation requires \a typeA = ::SimdTensorData8u, \a typeB = ::SimdTensorData8i,
        \a typeC = ::SimdTensorData8u and \a constB = ::SimdTrue. Matrix B must be supplied to SetParams()
        and may be stored transposed according to \a transB. Call Init() and SetParams() before Forward().
        Use Enable() to check that a context was created. The context is released by Clear() or by the destructor.

        Using example:
        \verbatim
        #include "Simd/SimdSynet.hpp"

        int main()
        {
            const size_t M = 4, N = 8, K = 16;
            std::vector<uint8_t> A(M * K, 50), C(M * N, 0);
            std::vector<int8_t> B(K * N, 1);
            std::vector<float> bScale(N, 0.02f);
            std::vector<int32_t> bias(N, 10);
            float aScale = 0.01f, cScale = 0.015f;
            uint8_t aZero = 47, cZero = 38;

            Simd::SynetQuantizedInnerProduct innerProduct;
            innerProduct.Init(M, N, K, SimdTensorData8u, SimdTensorData8i, SimdTensorData8u,
                SimdFalse, SimdTrue, SimdTrue);
            if (innerProduct.Enable())
            {
                innerProduct.SetParams(&aScale, &aZero, B.data(), bScale.data(), bias.data(), &cScale, &cZero);
                innerProduct.Forward(A.data(), NULL, NULL, C.data());
            }

            return 0;
        }
        \endverbatim
    */
    class SynetQuantizedInnerProduct
    {
    public:
        /*!
            Creates a new empty SynetQuantizedInnerProduct class.
        */
        SynetQuantizedInnerProduct()
            : _context(NULL)
            , _M(0)
            , _N(0)
            , _K(0)
            , _typeA(SimdTensorData8u)
            , _typeB(SimdTensorData8i)
            , _typeC(SimdTensorData8u)
            , _transB(SimdFalse)
            , _constB(SimdFalse)
            , _bias(SimdFalse)
        {
        }

        /*!
            SynetQuantizedInnerProduct class destructor. Releases internal context.
        */
        virtual ~SynetQuantizedInnerProduct()
        {
            Clear();
        }

        /*!
            Initializes (or re-initializes) a quantized inner-product context.

            Creates an internal context with using of function ::SimdSynetQuantizedInnerProductInit.
            The context is recreated only if matrix sizes, tensor types or inner-product flags were changed.

            \note This function is a C++ wrapper for function ::SimdSynetQuantizedInnerProductInit.

            \param [in] M - a height of A and C matrices.
            \param [in] N - a width of B and C matrices.
            \param [in] K - a width of A and height of B matrices.
            \param [in] typeA - a type of A matrix. Currently it must be ::SimdTensorData8u.
            \param [in] typeB - a type of B matrix. Currently it must be ::SimdTensorData8i.
            \param [in] typeC - a type of C matrix. Currently it must be ::SimdTensorData8u.
            \param [in] transB - a flag indicating that B is stored as N*K instead of K*N.
            \param [in] constB - a flag indicating that matrix B is constant. Currently it must be ::SimdTrue.
            \param [in] bias - a flag to add bias to output matrix C.
        */
        SIMD_INLINE void Init(size_t M, size_t N, size_t K, SimdTensorDataType typeA, SimdTensorDataType typeB, SimdTensorDataType typeC,
            SimdBool transB, SimdBool constB, SimdBool bias)
        {
            if (_M != M || _N != N || _K != K || _typeA != typeA || _typeB != typeB || _typeC != typeC ||
                _transB != transB || _constB != constB || _bias != bias)
            {
                Clear();
                _M = M;
                _N = N;
                _K = K;
                _typeA = typeA;
                _typeB = typeB;
                _typeC = typeC;
                _transB = transB;
                _constB = constB;
                _bias = bias;
                _context = SimdSynetQuantizedInnerProductInit(_M, _N, _K, _typeA, _typeB, _typeC, _transB, _constB, _bias);
            }
        }

        /*!
            Checks that the internal quantized inner-product context was created.

            \return true if the context exists and Forward() can be called.
        */
        SIMD_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        /*!
            Gets the size in bytes of internal storage used by the quantized inner-product context.

            The returned value reports internal storage of constant B, bias, zero points, scales and an optional
            fallback temporary buffer.

            \note This function is a C++ wrapper for function ::SimdSynetQuantizedInnerProductInternalBufferSize.

            \return a number of bytes used by internal buffers.
        */
        SIMD_INLINE size_t InternalBufferSize() const
        {
            return _context ? SimdSynetQuantizedInnerProductInternalBufferSize(_context) : 0;
        }

        /*!
            Gets the size in bytes of caller-provided temporary buffer for quantized inner product.

            The returned value can be used when allocating the \a buf argument of Forward().

            \note This function is a C++ wrapper for function ::SimdSynetQuantizedInnerProductExternalBufferSize.

            \return a number of bytes required for external temporary buffer.
        */
        SIMD_INLINE size_t ExternalBufferSize() const
        {
            return _context ? SimdSynetQuantizedInnerProductExternalBufferSize(_context) : 0;
        }

        /*!
            Gets a short description of the selected quantized inner-product implementation.

            The returned string contains the implementation extension and algorithm name. The returned
            pointer is owned by the context and remains valid until the next call of this function or until the context
            is released.

            \note This function is a C++ wrapper for function ::SimdSynetQuantizedInnerProductInfo.

            \return a string with description of internal implementation. NULL if the context was not created.
        */
        SIMD_INLINE const char * Info() const
        {
            return _context ? SimdSynetQuantizedInnerProductInfo(_context) : NULL;
        }

        /*!
            Sets constant matrix B, bias and quantization parameters for quantized inner product.

            This function must be called before Forward(). If \a constB was ::SimdTrue during
            initialization, \a b provides matrix B and the implementation may reorder and store it internally.

            \note This function is a C++ wrapper for function ::SimdSynetQuantizedInnerProductSetParams.

            \param [in] aScale - a pointer to FP32 quantization scale of A matrix.
            \param [in] aZero - a pointer to UINT8 quantization zero of A matrix.
            \param [in] b - a pointer to constant INT8 B matrix. It must be valid when constB is ::SimdTrue.
            \param [in] bScale - a pointer to per-output-channel FP32 scales of B matrix. The size of the array must be equal to N.
            \param [in] bias - a pointer to INT32 bias values. The size of the array must be equal to N. Can be NULL.
            \param [in] cScale - a pointer to FP32 quantization scale of C matrix.
            \param [in] cZero - a pointer to UINT8 quantization zero of C matrix.
        */
        SIMD_INLINE void SetParams(const float * aScale, const uint8_t * aZero, const int8_t * b, const float * bScale, const int32_t * bias, const float * cScale, const uint8_t * cZero)
        {
            if (_context)
                SimdSynetQuantizedInnerProductSetParams(_context, aScale, aZero, b, bScale, bias, cScale, cZero);
        }

        /*!
            Performs quantized inner-product forward propagation.

            The function computes C = A*B with dequantization of A, INT8 B, optional INT32 bias and UINT8
            requantization of C according to parameters stored by Init() and SetParams(). If B is constant,
            it can be NULL when it was set by SetParams(). The \a buf argument can be NULL (it causes usage of internal buffer).

            \note This function is a C++ wrapper for function ::SimdSynetQuantizedInnerProductForward.

            \param [in] A - a pointer to UINT8 A matrix with size M*K.
            \param [in] B - a pointer to INT8 B matrix. Can be NULL if B is constant.
            \param [out] buf - a pointer to external temporary byte buffer. Can be NULL.
            \param [out] C - a pointer to UINT8 C matrix with size M*N.
        */
        SIMD_INLINE void Forward(const uint8_t * A, const uint8_t * B, uint8_t * buf, uint8_t * C)
        {
            if (_context)
                SimdSynetQuantizedInnerProductForward(_context, A, B, buf, C);
        }

        /*!
            Releases internal context and clears stored inner-product parameters.
        */
        SIMD_INLINE void Clear()
        {
            if (_context)
                SimdRelease(_context), _context = NULL;
            _M = 0;
            _N = 0;
            _K = 0;
            _typeA = SimdTensorData8u;
            _typeB = SimdTensorData8i;
            _typeC = SimdTensorData8u;
            _transB = SimdFalse;
            _constB = SimdFalse;
            _bias = SimdFalse;
        }

    private:
        void * _context;
        size_t _M, _N, _K;
        SimdTensorDataType _typeA, _typeB, _typeC;
        SimdBool _transB, _constB, _bias;
    };

    //-------------------------------------------------------------------------------------------------

    /*! @ingroup cpp_synet

        \short The SynetDeconvolution32f class is a C++ wrapper of FP32 deconvolution (transposed convolution).

        The class wraps C API functions ::SimdSynetDeconvolution32fInit, ::SimdSynetDeconvolution32fExternalBufferSize,
        ::SimdSynetDeconvolution32fInternalBufferSize, ::SimdSynetDeconvolution32fInfo, ::SimdSynetDeconvolution32fSetParams and
        ::SimdSynetDeconvolution32fForward. It applies transposed convolution to each image in the batch, optionally adds bias
        and applies activation:
        \verbatim
        dst[:] = 0;
        for(sc = 0; sc < srcC/group; ++sc)
            for(sy = 0; sy < srcH; ++sy)
                for(sx = 0; sx < srcW; ++sx)
                    for(ky = 0; ky < kernelY; ++ky)
                        for(kx = 0; kx < kernelX; ++kx)
                            dst[outputOffset] += src[inputOffset] * weight[weightOffset];
        dst[outputOffset] = Activate(dst[outputOffset] + bias[dc], activation, params);
        \endverbatim

        The exact offsets depend on tensor format, padding, dilation, stride and group. The current implementation
        supports FP32 source and destination tensors with matching NCHW format, or matching NHWC format when group is 1.
        The destination spatial size must match deconvolution parameters:
        \verbatim
        dstH = strideY*(srcH - 1) + dilationY*(kernelY - 1) + 1 - padY - padH
        dstW = strideX*(srcW - 1) + dilationX*(kernelX - 1) + 1 - padX - padW
        \endverbatim

        Call Init() and SetParams() before Forward(). Use Enable() to check that a context was created.
        The context is released by Clear() or by the destructor.

        Using example:
        \verbatim
        #include "Simd/SimdSynet.hpp"

        int main()
        {
            const size_t batch = 1, srcC = 4, srcH = 3, srcW = 3, dstC = 4;
            SimdConvolutionParameters conv = {};
            conv.srcC = srcC;
            conv.srcH = srcH;
            conv.srcW = srcW;
            conv.srcT = SimdTensorData32f;
            conv.srcF = SimdTensorFormatNhwc;
            conv.dstC = dstC;
            conv.kernelY = 2;
            conv.kernelX = 2;
            conv.dilationY = 1;
            conv.dilationX = 1;
            conv.strideY = 2;
            conv.strideX = 2;
            conv.padY = 0;
            conv.padX = 0;
            conv.padH = 0;
            conv.padW = 0;
            conv.group = 1;
            conv.activation = SimdConvolutionActivationIdentity;
            conv.dstH = conv.strideY * (conv.srcH - 1) + conv.dilationY * (conv.kernelY - 1) + 1 - conv.padY - conv.padH;
            conv.dstW = conv.strideX * (conv.srcW - 1) + conv.dilationX * (conv.kernelX - 1) + 1 - conv.padX - conv.padW;
            conv.dstT = SimdTensorData32f;
            conv.dstF = SimdTensorFormatNhwc;

            std::vector<float> src(batch * srcH * srcW * srcC);
            std::vector<float> weight(conv.kernelY * conv.kernelX * srcC * dstC / conv.group);
            std::vector<float> bias(dstC, 0.0f);
            std::vector<float> dst(batch * conv.dstH * conv.dstW * dstC, 0.0f);
            for (size_t i = 0; i < src.size(); ++i)
                src[i] = float(i) * 0.01f;
            for (size_t i = 0; i < weight.size(); ++i)
                weight[i] = float(i) * 0.02f;

            Simd::SynetDeconvolution32f deconvolution;
            deconvolution.Init(batch, &conv);
            if (deconvolution.Enable())
            {
                deconvolution.SetParams(weight.data(), NULL, bias.data(), NULL);
                deconvolution.Forward(src.data(), NULL, dst.data());
            }

            return 0;
        }
        \endverbatim
    */
    class SynetDeconvolution32f
    {
    public:
        /*!
            Creates a new empty SynetDeconvolution32f class.
        */
        SynetDeconvolution32f()
            : _context(NULL)
            , _batch(0)
            , _compatibility(SimdSynetCompatibilityDefault)
        {
            SimdConvolutionParameters conv = {};
            _conv = conv;
        }

        /*!
            SynetDeconvolution32f class destructor. Releases internal context.
        */
        virtual ~SynetDeconvolution32f()
        {
            Clear();
        }

        /*!
            Initializes (or re-initializes) an FP32 deconvolution context.

            Creates an internal context with using of function ::SimdSynetDeconvolution32fInit.
            The context is recreated only if batch size, deconvolution parameters or compatibility flags were changed.

            \note This function is a C++ wrapper for function ::SimdSynetDeconvolution32fInit.

            \param [in] batch - a batch size.
            \param [in] conv - a pointer to deconvolution parameters. Source and destination tensor types must be FP32.
            \param [in] compatibility - calculation compatibility flags.
        */
        SIMD_INLINE void Init(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility = SimdSynetCompatibilityDefault)
        {
            if (conv == NULL)
                return;
            if (_batch != batch || Changed(*conv) || _compatibility != compatibility)
            {
                Clear();
                _batch = batch;
                _conv = *conv;
                _compatibility = compatibility;
                _context = SimdSynetDeconvolution32fInit(_batch, &_conv, _compatibility);
            }
        }

        /*!
            Checks that the internal deconvolution context was created.

            \return true if the context exists and Forward() can be called.
        */
        SIMD_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        /*!
            Gets the size of caller-provided temporary buffer for FP32 deconvolution.

            The returned value is a number of FP32 elements. It depends on the implementation selected
            during initialization and can be used when allocating the \a buf argument of Forward().
            Some implementations return 1 when they do not need external temporary storage.

            \note This function is a C++ wrapper for function ::SimdSynetDeconvolution32fExternalBufferSize.

            \return a number of FP32 elements required for external temporary buffer.
        */
        SIMD_INLINE size_t ExternalBufferSize() const
        {
            return _context ? SimdSynetDeconvolution32fExternalBufferSize(_context) : 0;
        }

        /*!
            Gets the size of internal storage used by the deconvolution context.

            The returned value is a number of FP32 elements. It reports internal temporary buffers and
            implementation-specific reordered weights, bias or activation parameters already allocated by the context.

            \note This function is a C++ wrapper for function ::SimdSynetDeconvolution32fInternalBufferSize.

            \return a number of FP32 elements used by internal buffers.
        */
        SIMD_INLINE size_t InternalBufferSize() const
        {
            return _context ? SimdSynetDeconvolution32fInternalBufferSize(_context) : 0;
        }

        /*!
            Gets a short description of the selected FP32 deconvolution implementation.

            The returned string contains the implementation extension and algorithm name, for example a GEMM-based or
            NHWC direct 2x2 variant. The returned pointer is owned by the context and remains valid until the next call
            of this function or until the context is released.

            \note This function is a C++ wrapper for function ::SimdSynetDeconvolution32fInfo.

            \return a string with description of internal implementation. NULL if the context was not created.
        */
        SIMD_INLINE const char * Info() const
        {
            return _context ? SimdSynetDeconvolution32fInfo(_context) : NULL;
        }

        /*!
            Sets weights, bias and activation parameters for FP32 deconvolution.

            This function must be called before Forward(). The \a weight array contains FP32 deconvolution weights
            with kernelY*kernelX*srcC*dstC/group elements. Depending on the selected implementation, weights can be
            used directly or transformed and stored inside the context. If \a internal is not NULL, ::SimdTrue means
            the weights were copied/reordered into the context; ::SimdFalse means the original \a weight pointer can
            be used by later forward calls and must remain valid.

            \note This function is a C++ wrapper for function ::SimdSynetDeconvolution32fSetParams.

            \param [in] weight - a pointer to FP32 deconvolution weights.
            \param [out] internal - a pointer to a flag receiving weight storage mode. Can be NULL.
            \param [in] bias - a pointer to FP32 bias array with dstC elements. Can be NULL.
            \param [in] params - a pointer to FP32 parameters of activation function (see ::SimdConvolutionActivationType). Can be NULL when activation does not require parameters.
        */
        SIMD_INLINE void SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params)
        {
            if (_context)
                SimdSynetDeconvolution32fSetParams(_context, weight, internal, bias, params);
        }

        /*!
            Performs FP32 deconvolution forward propagation.

            The function applies transposed convolution to each image in the batch, adds bias when it was set,
            and applies the activation stored in the context created by Init() and SetParams().
            The \a buf argument can be NULL (it causes usage of internal buffer).

            \note This function is a C++ wrapper for function ::SimdSynetDeconvolution32fForward.

            \param [in] src - a pointer to FP32 input tensor.
            \param [out] buf - a pointer to external temporary FP32 buffer. Can be NULL.
            \param [out] dst - a pointer to FP32 output tensor.
        */
        SIMD_INLINE void Forward(const float * src, float * buf, float * dst)
        {
            if (_context)
                SimdSynetDeconvolution32fForward(_context, src, buf, dst);
        }

        /*!
            Releases internal context and clears stored deconvolution parameters.
        */
        SIMD_INLINE void Clear()
        {
            if (_context)
                SimdRelease(_context), _context = NULL;
            _batch = 0;
            SimdConvolutionParameters conv = {};
            _conv = conv;
            _compatibility = SimdSynetCompatibilityDefault;
        }

    private:
        SIMD_INLINE bool Changed(const SimdConvolutionParameters & conv) const
        {
            return _conv.srcC != conv.srcC || _conv.srcH != conv.srcH || _conv.srcW != conv.srcW ||
                _conv.srcT != conv.srcT || _conv.srcF != conv.srcF ||
                _conv.dstC != conv.dstC || _conv.dstH != conv.dstH || _conv.dstW != conv.dstW ||
                _conv.dstT != conv.dstT || _conv.dstF != conv.dstF ||
                _conv.kernelY != conv.kernelY || _conv.kernelX != conv.kernelX ||
                _conv.dilationY != conv.dilationY || _conv.dilationX != conv.dilationX ||
                _conv.strideY != conv.strideY || _conv.strideX != conv.strideX ||
                _conv.padY != conv.padY || _conv.padX != conv.padX ||
                _conv.padH != conv.padH || _conv.padW != conv.padW ||
                _conv.group != conv.group || _conv.activation != conv.activation;
        }

        void * _context;
        size_t _batch;
        SimdConvolutionParameters _conv;
        SimdSynetCompatibilityType _compatibility;
    };

    //-------------------------------------------------------------------------------------------------

    /*! @ingroup cpp_synet

        \short The SynetDeconvolution16b class is a C++ wrapper of BF16/FP32 deconvolution (transposed convolution).

        The class wraps C API functions ::SimdSynetDeconvolution16bInit, ::SimdSynetDeconvolution16bExternalBufferSize,
        ::SimdSynetDeconvolution16bInternalBufferSize, ::SimdSynetDeconvolution16bInfo, ::SimdSynetDeconvolution16bSetParams and
        ::SimdSynetDeconvolution16bForward. It applies transposed convolution to each image in the batch, optionally adds bias
        and applies activation. Source and destination tensors can be FP32 or BF16:
        \verbatim
        dst[:] = 0;
        for(sc = 0; sc < srcC/group; ++sc)
            for(sy = 0; sy < srcH; ++sy)
                for(sx = 0; sx < srcW; ++sx)
                    for(ky = 0; ky < kernelY; ++ky)
                        for(kx = 0; kx < kernelX; ++kx)
                            dst[outputOffset] += inputValue * weightValue;
        value = Activate(dst[outputOffset] + bias[dc], activation, params);
        dst[outputOffset] = dstT == SimdTensorData16b ? Float32ToBFloat16(value) : value;
        \endverbatim
        The input value is read as BF16 or converted from FP32 to BF16 according to srcT. The weight value comes from
        the internal representation prepared by SetParams().

        The exact offsets depend on tensor format, padding, dilation, stride and group. The current implementation
        supports FP32 or BF16 source and destination tensors with matching NCHW format, or matching NHWC format when
        group is 1. The destination spatial size must match deconvolution parameters:
        \verbatim
        dstH = strideY*(srcH - 1) + dilationY*(kernelY - 1) + 1 - padY - padH
        dstW = strideX*(srcW - 1) + dilationX*(kernelX - 1) + 1 - padX - padW
        \endverbatim

        Call Init() and SetParams() before Forward(). Use Enable() to check that a context was created.
        The context is released by Clear() or by the destructor.

        Using example:
        \verbatim
        #include "Simd/SimdSynet.hpp"

        int main()
        {
            const size_t batch = 1, srcC = 4, srcH = 3, srcW = 3, dstC = 4;
            SimdConvolutionParameters conv = {};
            conv.srcC = srcC;
            conv.srcH = srcH;
            conv.srcW = srcW;
            conv.srcT = SimdTensorData32f;
            conv.srcF = SimdTensorFormatNhwc;
            conv.dstC = dstC;
            conv.kernelY = 2;
            conv.kernelX = 2;
            conv.dilationY = 1;
            conv.dilationX = 1;
            conv.strideY = 2;
            conv.strideX = 2;
            conv.padY = 0;
            conv.padX = 0;
            conv.padH = 0;
            conv.padW = 0;
            conv.group = 1;
            conv.activation = SimdConvolutionActivationIdentity;
            conv.dstH = conv.strideY * (conv.srcH - 1) + conv.dilationY * (conv.kernelY - 1) + 1 - conv.padY - conv.padH;
            conv.dstW = conv.strideX * (conv.srcW - 1) + conv.dilationX * (conv.kernelX - 1) + 1 - conv.padX - conv.padW;
            conv.dstT = SimdTensorData32f;
            conv.dstF = SimdTensorFormatNhwc;

            std::vector<float> src(batch * srcH * srcW * srcC);
            std::vector<float> weight(conv.kernelY * conv.kernelX * srcC * dstC / conv.group);
            std::vector<float> bias(dstC, 0.0f);
            std::vector<float> dst(batch * conv.dstH * conv.dstW * dstC, 0.0f);
            for (size_t i = 0; i < src.size(); ++i)
                src[i] = float(i) * 0.01f;
            for (size_t i = 0; i < weight.size(); ++i)
                weight[i] = float(i) * 0.02f;

            Simd::SynetDeconvolution16b deconvolution;
            deconvolution.Init(batch, &conv);
            if (deconvolution.Enable())
            {
                deconvolution.SetParams(weight.data(), bias.data(), NULL);
                deconvolution.Forward((const uint8_t*)src.data(), NULL, (uint8_t*)dst.data());
            }

            return 0;
        }
        \endverbatim
    */
    class SynetDeconvolution16b
    {
    public:
        /*!
            Creates a new empty SynetDeconvolution16b class.
        */
        SynetDeconvolution16b()
            : _context(NULL)
            , _batch(0)
            , _compatibility(SimdSynetCompatibilityDefault)
        {
            SimdConvolutionParameters conv = {};
            _conv = conv;
        }

        /*!
            SynetDeconvolution16b class destructor. Releases internal context.
        */
        virtual ~SynetDeconvolution16b()
        {
            Clear();
        }

        /*!
            Initializes (or re-initializes) a BF16/FP32 deconvolution context.

            Creates an internal context with using of function ::SimdSynetDeconvolution16bInit.
            The context is recreated only if batch size, deconvolution parameters or compatibility flags were changed.

            \note This function is a C++ wrapper for function ::SimdSynetDeconvolution16bInit.

            \param [in] batch - a batch size.
            \param [in] conv - a pointer to deconvolution parameters. Source and destination tensor types must be FP32 or BF16.
            \param [in] compatibility - calculation compatibility flags.
        */
        SIMD_INLINE void Init(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility = SimdSynetCompatibilityDefault)
        {
            if (conv == NULL)
                return;
            if (_batch != batch || Changed(*conv) || _compatibility != compatibility)
            {
                Clear();
                _batch = batch;
                _conv = *conv;
                _compatibility = compatibility;
                _context = SimdSynetDeconvolution16bInit(_batch, &_conv, _compatibility);
            }
        }

        /*!
            Checks that the internal deconvolution context was created.

            \return true if the context exists and Forward() can be called.
        */
        SIMD_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        /*!
            Gets the size in bytes of caller-provided temporary buffer for BF16 deconvolution.

            The returned value is a number of bytes. It depends on the implementation selected
            during initialization and can be used when allocating the \a buf argument of Forward().
            Some implementations return 1 or 0 when they do not need external temporary storage.

            \note This function is a C++ wrapper for function ::SimdSynetDeconvolution16bExternalBufferSize.

            \return a number of bytes required for external temporary buffer.
        */
        SIMD_INLINE size_t ExternalBufferSize() const
        {
            return _context ? SimdSynetDeconvolution16bExternalBufferSize(_context) : 0;
        }

        /*!
            Gets the size in bytes of internal storage used by the deconvolution context.

            The returned value reports internal storage tracked by the selected implementation, including internal
            temporary buffers, transformed weights, copied bias and copied activation parameters.

            \note This function is a C++ wrapper for function ::SimdSynetDeconvolution16bInternalBufferSize.

            \return a number of bytes used by internal buffers.
        */
        SIMD_INLINE size_t InternalBufferSize() const
        {
            return _context ? SimdSynetDeconvolution16bInternalBufferSize(_context) : 0;
        }

        /*!
            Gets a short description of the selected BF16 deconvolution implementation.

            The returned string contains the implementation extension and algorithm name, for example a GEMM or NHWC GEMM
            variant. The returned pointer is owned by the context and remains valid until the next call
            of this function or until the context is released.

            \note This function is a C++ wrapper for function ::SimdSynetDeconvolution16bInfo.

            \return a string with description of internal implementation. NULL if the context was not created.
        */
        SIMD_INLINE const char * Info() const
        {
            return _context ? SimdSynetDeconvolution16bInfo(_context) : NULL;
        }

        /*!
            Sets weights, bias and activation parameters for BF16 deconvolution.

            This function must be called before Forward(). The \a weight array contains FP32 deconvolution weights
            with kernelY*kernelX*srcC*dstC/group elements. The selected implementation transforms weights to its
            internal BF16/reordered representation. Bias is copied to an internal FP32 array; when \a bias is NULL,
            zeros are used. Activation parameters are copied or expanded to the internal FP32 array according to
            ::SimdConvolutionActivationType.

            \note This function is a C++ wrapper for function ::SimdSynetDeconvolution16bSetParams.

            \param [in] weight - a pointer to FP32 deconvolution weights.
            \param [in] bias - a pointer to FP32 bias array with dstC elements. Can be NULL.
            \param [in] params - a pointer to FP32 parameters of activation function (see ::SimdConvolutionActivationType). Can be NULL when activation does not require parameters.
        */
        SIMD_INLINE void SetParams(const float * weight, const float * bias, const float * params)
        {
            if (_context)
                SimdSynetDeconvolution16bSetParams(_context, weight, bias, params);
        }

        /*!
            Performs BF16/FP32 deconvolution forward propagation.

            The function converts FP32 input to BF16 when the context source type is FP32, uses BF16 input directly
            when the source type is BF16, accumulates transposed convolution sums in FP32, adds bias, applies
            activation and writes FP32 or BF16 output according to the context destination type.
            The \a buf argument can be NULL (it causes usage of internal buffer).

            \note This function is a C++ wrapper for function ::SimdSynetDeconvolution16bForward.

            \param [in] src - a pointer to input tensor. Actual element type is defined by srcT in deconvolution parameters.
            \param [out] buf - a pointer to external temporary byte buffer. Can be NULL.
            \param [out] dst - a pointer to output tensor. Actual element type is defined by dstT in deconvolution parameters.
        */
        SIMD_INLINE void Forward(const uint8_t * src, uint8_t * buf, uint8_t * dst)
        {
            if (_context)
                SimdSynetDeconvolution16bForward(_context, src, buf, dst);
        }

        /*!
            Releases internal context and clears stored deconvolution parameters.
        */
        SIMD_INLINE void Clear()
        {
            if (_context)
                SimdRelease(_context), _context = NULL;
            _batch = 0;
            SimdConvolutionParameters conv = {};
            _conv = conv;
            _compatibility = SimdSynetCompatibilityDefault;
        }

    private:
        SIMD_INLINE bool Changed(const SimdConvolutionParameters & conv) const
        {
            return _conv.srcC != conv.srcC || _conv.srcH != conv.srcH || _conv.srcW != conv.srcW ||
                _conv.srcT != conv.srcT || _conv.srcF != conv.srcF ||
                _conv.dstC != conv.dstC || _conv.dstH != conv.dstH || _conv.dstW != conv.dstW ||
                _conv.dstT != conv.dstT || _conv.dstF != conv.dstF ||
                _conv.kernelY != conv.kernelY || _conv.kernelX != conv.kernelX ||
                _conv.dilationY != conv.dilationY || _conv.dilationX != conv.dilationX ||
                _conv.strideY != conv.strideY || _conv.strideX != conv.strideX ||
                _conv.padY != conv.padY || _conv.padX != conv.padX ||
                _conv.padH != conv.padH || _conv.padW != conv.padW ||
                _conv.group != conv.group || _conv.activation != conv.activation;
        }

        void * _context;
        size_t _batch;
        SimdConvolutionParameters _conv;
        SimdSynetCompatibilityType _compatibility;
    };

    //-------------------------------------------------------------------------------------------------

    /*! @ingroup cpp_synet

        \short The SynetMergedConvolution32f class is a C++ wrapper of FP32 merged convolution.

        The class wraps C API functions ::SimdSynetMergedConvolution32fInit, ::SimdSynetMergedConvolution32fExternalBufferSize,
        ::SimdSynetMergedConvolution32fInternalBufferSize, ::SimdSynetMergedConvolution32fInfo, ::SimdSynetMergedConvolution32fSetParams and
        ::SimdSynetMergedConvolution32fForward. It fuses a sequence of two or three NHWC convolutions into one forward call:
        convolution + depthwise convolution, depthwise convolution + convolution, or
        convolution + depthwise convolution + convolution. The first and last tensors must be FP32.
        Supported kernels are 1x1 or 3x3 for ordinary convolutions, 3x3, 5x5 or 7x7 for depthwise
        convolutions; dilation must be 1 and stride must be 1, 2 or 3. If add is ::SimdTrue for a
        three-convolution sequence, the source tensor is added to the final output and therefore must
        have the same shape as the final destination tensor.

        Call Init() and SetParams() before Forward(). Use Enable() to check that a context was created.
        The context is released by Clear() or by the destructor.

        Using example:
        \verbatim
        #include "Simd/SimdSynet.hpp"

        int main()
        {
            const size_t batch = 1, srcC = 4, srcH = 8, srcW = 8, midC = 8, count = 2;
            SimdConvolutionParameters convs[2] = {};

            convs[0].srcC = srcC;
            convs[0].srcH = srcH;
            convs[0].srcW = srcW;
            convs[0].srcT = SimdTensorData32f;
            convs[0].srcF = SimdTensorFormatNhwc;
            convs[0].dstC = midC;
            convs[0].kernelY = 1;
            convs[0].kernelX = 1;
            convs[0].dilationY = 1;
            convs[0].dilationX = 1;
            convs[0].strideY = 1;
            convs[0].strideX = 1;
            convs[0].padY = 0;
            convs[0].padX = 0;
            convs[0].padH = 0;
            convs[0].padW = 0;
            convs[0].group = 1;
            convs[0].activation = SimdConvolutionActivationIdentity;
            convs[0].dstH = srcH;
            convs[0].dstW = srcW;
            convs[0].dstT = SimdTensorData32f;
            convs[0].dstF = SimdTensorFormatNhwc;

            convs[1].srcC = midC;
            convs[1].srcH = convs[0].dstH;
            convs[1].srcW = convs[0].dstW;
            convs[1].srcT = SimdTensorData32f;
            convs[1].srcF = SimdTensorFormatNhwc;
            convs[1].dstC = midC;
            convs[1].kernelY = 3;
            convs[1].kernelX = 3;
            convs[1].dilationY = 1;
            convs[1].dilationX = 1;
            convs[1].strideY = 1;
            convs[1].strideX = 1;
            convs[1].padY = 1;
            convs[1].padX = 1;
            convs[1].padH = 1;
            convs[1].padW = 1;
            convs[1].group = midC;
            convs[1].activation = SimdConvolutionActivationIdentity;
            convs[1].dstH = convs[1].srcH;
            convs[1].dstW = convs[1].srcW;
            convs[1].dstT = SimdTensorData32f;
            convs[1].dstF = SimdTensorFormatNhwc;

            std::vector<float> src(batch * srcH * srcW * srcC);
            std::vector<float> weight0(convs[0].kernelY * convs[0].kernelX * convs[0].srcC * convs[0].dstC);
            std::vector<float> weight1(convs[1].kernelY * convs[1].kernelX * convs[1].dstC);
            std::vector<float> bias0(convs[0].dstC, 0.0f), bias1(convs[1].dstC, 0.0f);
            std::vector<float> dst(batch * convs[1].dstH * convs[1].dstW * convs[1].dstC, 0.0f);
            const float * weight[2] = { weight0.data(), weight1.data() };
            const float * bias[2] = { bias0.data(), bias1.data() };
            for (size_t i = 0; i < src.size(); ++i)
                src[i] = float(i) * 0.01f;
            for (size_t i = 0; i < weight0.size(); ++i)
                weight0[i] = float(i) * 0.02f;
            for (size_t i = 0; i < weight1.size(); ++i)
                weight1[i] = float(i) * 0.03f;

            Simd::SynetMergedConvolution32f mergedConvolution;
            mergedConvolution.Init(batch, convs, count, SimdFalse);
            if (mergedConvolution.Enable())
            {
                mergedConvolution.SetParams(weight, NULL, bias, NULL);
                mergedConvolution.Forward(src.data(), NULL, dst.data());
            }

            return 0;
        }
        \endverbatim
    */
    class SynetMergedConvolution32f
    {
    public:
        /*!
            Creates a new empty SynetMergedConvolution32f class.
        */
        SynetMergedConvolution32f()
            : _context(NULL)
            , _batch(0)
            , _count(0)
            , _add(SimdFalse)
        {
            SimdConvolutionParameters conv = {};
            _convs[0] = conv;
            _convs[1] = conv;
            _convs[2] = conv;
        }

        /*!
            SynetMergedConvolution32f class destructor. Releases internal context.
        */
        virtual ~SynetMergedConvolution32f()
        {
            Clear();
        }

        /*!
            Initializes (or re-initializes) an FP32 merged convolution context.

            Creates an internal context with using of function ::SimdSynetMergedConvolution32fInit.
            The context is recreated only if batch size, convolution count, residual-add flag or
            any convolution parameters were changed.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution32fInit.

            \param [in] batch - a batch size.
            \param [in] convs - an array with convolution parameters in execution order.
            \param [in] count - a number of merged convolutions. It must be 2 or 3.
            \param [in] add - a flag that enables adding the source tensor to the final output tensor.
        */
        SIMD_INLINE void Init(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add)
        {
            if (convs == NULL || count < 2 || count > 3)
                return;
            if (_batch != batch || _count != count || _add != add || Changed(convs, count))
            {
                Clear();
                _batch = batch;
                _count = count;
                _add = add;
                for (size_t i = 0; i < count; ++i)
                    _convs[i] = convs[i];
                _context = SimdSynetMergedConvolution32fInit(_batch, _convs, _count, _add);
            }
        }

        /*!
            Checks that the internal merged convolution context was created.

            \return true if the context exists and Forward() can be called.
        */
        SIMD_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        /*!
            Gets the size of caller-provided temporary buffer for FP32 merged convolution.

            The returned value is a number of FP32 elements. It depends on the implementation selected
            during initialization and can be used when allocating the \a buf argument of Forward().
            Some implementations return 1 when they do not need external temporary storage.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution32fExternalBufferSize.

            \return a number of FP32 elements required for external temporary buffer.
        */
        SIMD_INLINE size_t ExternalBufferSize() const
        {
            return _context ? SimdSynetMergedConvolution32fExternalBufferSize(_context) : 0;
        }

        /*!
            Gets the size of internal storage used by the merged convolution context.

            The returned value is a number of FP32 elements. It reports internal temporary buffers and
            implementation-specific reordered weights, bias or activation parameters already allocated by the context.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution32fInternalBufferSize.

            \return a number of FP32 elements used by internal buffers.
        */
        SIMD_INLINE size_t InternalBufferSize() const
        {
            return _context ? SimdSynetMergedConvolution32fInternalBufferSize(_context) : 0;
        }

        /*!
            Gets a short description of the selected FP32 merged convolution implementation.

            The returned string contains the implementation extension and algorithm name.
            The returned pointer is owned by the context and remains valid until the next call
            of this function or until the context is released.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution32fInfo.

            \return a string with description of internal implementation. NULL if the context was not created.
        */
        SIMD_INLINE const char * Info() const
        {
            return _context ? SimdSynetMergedConvolution32fInfo(_context) : NULL;
        }

        /*!
            Sets weights, biases and activation parameters for FP32 merged convolution.

            This function must be called before Forward(). The \a weight array contains pointers to FP32
            convolution weights, one per merged convolution. Depending on the selected implementation,
            weights can be used directly or transformed and stored inside the context. If \a internal is
            not NULL, ::SimdTrue means the corresponding weights were copied/reordered into the context;
            ::SimdFalse means the original weight pointer can be used by later forward calls and must remain valid.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution32fSetParams.

            \param [in] weight - an array of pointers to FP32 convolution weights. The array size must be equal to the number of merged convolutions.
            \param [out] internal - an array of flags receiving weight storage mode. The array size must be equal to the number of merged convolutions. Can be NULL.
            \param [in] bias - an array of pointers to FP32 bias arrays, one per convolution. Each pointer can be NULL.
            \param [in] params - an array of pointers to activation parameters (see ::SimdConvolutionActivationType), one per convolution. Each pointer can be NULL for activations that do not use parameters.
        */
        SIMD_INLINE void SetParams(const float * const * weight, SimdBool * internal, const float * const * bias, const float * const * params)
        {
            if (_context)
                SimdSynetMergedConvolution32fSetParams(_context, weight, internal, bias, params);
        }

        /*!
            Performs FP32 merged convolution forward propagation.

            The function applies the fused convolution sequence stored in the context created by Init()
            and SetParams(). The \a buf argument can be NULL (it causes usage of internal buffer).
            If Init() was called with add equal to ::SimdTrue, the source tensor is added to the final output.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution32fForward.

            \param [in] src - a pointer to the FP32 input tensor with batch*convs[0].srcC*convs[0].srcH*convs[0].srcW elements.
            \param [out] buf - a pointer to an external temporary FP32 buffer. Can be NULL.
            \param [out] dst - a pointer to the FP32 output tensor with batch*convs[count - 1].dstC*convs[count - 1].dstH*convs[count - 1].dstW elements.
        */
        SIMD_INLINE void Forward(const float * src, float * buf, float * dst)
        {
            if (_context)
                SimdSynetMergedConvolution32fForward(_context, src, buf, dst);
        }

        /*!
            Releases internal context and clears stored merged convolution parameters.
        */
        SIMD_INLINE void Clear()
        {
            if (_context)
                SimdRelease(_context), _context = NULL;
            _batch = 0;
            _count = 0;
            _add = SimdFalse;
            SimdConvolutionParameters conv = {};
            _convs[0] = conv;
            _convs[1] = conv;
            _convs[2] = conv;
        }

    private:
        SIMD_INLINE bool Changed(const SimdConvolutionParameters * convs, size_t count) const
        {
            for (size_t i = 0; i < count; ++i)
            {
                const SimdConvolutionParameters & conv = convs[i];
                const SimdConvolutionParameters & prev = _convs[i];
                if (prev.srcC != conv.srcC || prev.srcH != conv.srcH || prev.srcW != conv.srcW ||
                    prev.srcT != conv.srcT || prev.srcF != conv.srcF ||
                    prev.dstC != conv.dstC || prev.dstH != conv.dstH || prev.dstW != conv.dstW ||
                    prev.dstT != conv.dstT || prev.dstF != conv.dstF ||
                    prev.kernelY != conv.kernelY || prev.kernelX != conv.kernelX ||
                    prev.dilationY != conv.dilationY || prev.dilationX != conv.dilationX ||
                    prev.strideY != conv.strideY || prev.strideX != conv.strideX ||
                    prev.padY != conv.padY || prev.padX != conv.padX ||
                    prev.padH != conv.padH || prev.padW != conv.padW ||
                    prev.group != conv.group || prev.activation != conv.activation)
                    return true;
            }
            return false;
        }

        void * _context;
        size_t _batch, _count;
        SimdBool _add;
        SimdConvolutionParameters _convs[3];
    };

    //-------------------------------------------------------------------------------------------------

    /*! @ingroup cpp_synet

        \short The SynetMergedConvolution16b class is a C++ wrapper of BF16/FP32 merged convolution.

        The class wraps C API functions ::SimdSynetMergedConvolution16bInit, ::SimdSynetMergedConvolution16bExternalBufferSize,
        ::SimdSynetMergedConvolution16bInternalBufferSize, ::SimdSynetMergedConvolution16bInfo, ::SimdSynetMergedConvolution16bSetParams and
        ::SimdSynetMergedConvolution16bForward. It fuses a sequence of two or three NHWC convolutions into one forward call:
        convolution + depthwise convolution, depthwise convolution + convolution, or
        convolution + depthwise convolution + convolution. Internal convolution data is processed in BF16.
        Source and destination tensors can be FP32 or BF16 according to the corresponding ::SimdConvolutionParameters fields.
        Supported kernels are 1x1 or 3x3 for ordinary convolutions, 3x3, 5x5 or 7x7 for depthwise
        convolutions; dilation must be 1 and stride must be 1, 2 or 3. If add is ::SimdTrue for a
        three-convolution sequence, the source tensor is added to the final output and therefore must
        have the same shape as the final destination tensor.

        Call Init() and SetParams() before Forward(). Use Enable() to check that a context was created.
        The context is released by Clear() or by the destructor.

        Using example:
        \verbatim
        #include "Simd/SimdSynet.hpp"

        int main()
        {
            const size_t batch = 1, srcC = 4, srcH = 8, srcW = 8, midC = 8, count = 2;
            SimdConvolutionParameters convs[2] = {};

            convs[0].srcC = srcC;
            convs[0].srcH = srcH;
            convs[0].srcW = srcW;
            convs[0].srcT = SimdTensorData32f;
            convs[0].srcF = SimdTensorFormatNhwc;
            convs[0].dstC = midC;
            convs[0].kernelY = 1;
            convs[0].kernelX = 1;
            convs[0].dilationY = 1;
            convs[0].dilationX = 1;
            convs[0].strideY = 1;
            convs[0].strideX = 1;
            convs[0].padY = 0;
            convs[0].padX = 0;
            convs[0].padH = 0;
            convs[0].padW = 0;
            convs[0].group = 1;
            convs[0].activation = SimdConvolutionActivationIdentity;
            convs[0].dstH = srcH;
            convs[0].dstW = srcW;
            convs[0].dstT = SimdTensorData32f;
            convs[0].dstF = SimdTensorFormatNhwc;

            convs[1].srcC = midC;
            convs[1].srcH = convs[0].dstH;
            convs[1].srcW = convs[0].dstW;
            convs[1].srcT = SimdTensorData32f;
            convs[1].srcF = SimdTensorFormatNhwc;
            convs[1].dstC = midC;
            convs[1].kernelY = 3;
            convs[1].kernelX = 3;
            convs[1].dilationY = 1;
            convs[1].dilationX = 1;
            convs[1].strideY = 1;
            convs[1].strideX = 1;
            convs[1].padY = 1;
            convs[1].padX = 1;
            convs[1].padH = 1;
            convs[1].padW = 1;
            convs[1].group = midC;
            convs[1].activation = SimdConvolutionActivationIdentity;
            convs[1].dstH = convs[1].srcH;
            convs[1].dstW = convs[1].srcW;
            convs[1].dstT = SimdTensorData32f;
            convs[1].dstF = SimdTensorFormatNhwc;

            std::vector<float> src(batch * srcH * srcW * srcC);
            std::vector<float> weight0(convs[0].kernelY * convs[0].kernelX * convs[0].srcC * convs[0].dstC);
            std::vector<float> weight1(convs[1].kernelY * convs[1].kernelX * convs[1].dstC);
            std::vector<float> bias0(convs[0].dstC, 0.0f), bias1(convs[1].dstC, 0.0f);
            std::vector<float> dst(batch * convs[1].dstH * convs[1].dstW * convs[1].dstC, 0.0f);
            const float * weight[2] = { weight0.data(), weight1.data() };
            const float * bias[2] = { bias0.data(), bias1.data() };
            const float * params[2] = { NULL, NULL };
            for (size_t i = 0; i < src.size(); ++i)
                src[i] = float(i) * 0.01f;
            for (size_t i = 0; i < weight0.size(); ++i)
                weight0[i] = float(i) * 0.02f;
            for (size_t i = 0; i < weight1.size(); ++i)
                weight1[i] = float(i) * 0.03f;

            Simd::SynetMergedConvolution16b mergedConvolution;
            mergedConvolution.Init(batch, convs, count, SimdFalse);
            if (mergedConvolution.Enable())
            {
                mergedConvolution.SetParams(weight, bias, params);
                mergedConvolution.Forward((const uint8_t*)src.data(), NULL, (uint8_t*)dst.data());
            }

            return 0;
        }
        \endverbatim
    */
    class SynetMergedConvolution16b
    {
    public:
        /*!
            Creates a new empty SynetMergedConvolution16b class.
        */
        SynetMergedConvolution16b()
            : _context(NULL)
            , _batch(0)
            , _count(0)
            , _add(SimdFalse)
        {
            SimdConvolutionParameters conv = {};
            _convs[0] = conv;
            _convs[1] = conv;
            _convs[2] = conv;
        }

        /*!
            SynetMergedConvolution16b class destructor. Releases internal context.
        */
        virtual ~SynetMergedConvolution16b()
        {
            Clear();
        }

        /*!
            Initializes (or re-initializes) a BF16/FP32 merged convolution context.

            Creates an internal context with using of function ::SimdSynetMergedConvolution16bInit.
            The context is recreated only if batch size, convolution count, residual-add flag or
            any convolution parameters were changed.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution16bInit.

            \param [in] batch - a batch size.
            \param [in] convs - an array with convolution parameters in execution order.
            \param [in] count - a number of merged convolutions. It must be 2 or 3.
            \param [in] add - a flag that enables adding the source tensor to the final output tensor.
        */
        SIMD_INLINE void Init(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add)
        {
            if (convs == NULL || count < 2 || count > 3)
                return;
            if (_batch != batch || _count != count || _add != add || Changed(convs, count))
            {
                Clear();
                _batch = batch;
                _count = count;
                _add = add;
                for (size_t i = 0; i < count; ++i)
                    _convs[i] = convs[i];
                _context = SimdSynetMergedConvolution16bInit(_batch, _convs, _count, _add);
            }
        }

        /*!
            Checks that the internal merged convolution context was created.

            \return true if the context exists and Forward() can be called.
        */
        SIMD_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        /*!
            Gets the size in bytes of caller-provided temporary buffer for BF16 merged convolution.

            The returned value is a number of bytes. It depends on the implementation selected
            during initialization and can be used when allocating the \a buf argument of Forward().
            Some implementations return 1 when they do not need external temporary storage.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution16bExternalBufferSize.

            \return a number of bytes required for external temporary buffer.
        */
        SIMD_INLINE size_t ExternalBufferSize() const
        {
            return _context ? SimdSynetMergedConvolution16bExternalBufferSize(_context) : 0;
        }

        /*!
            Gets the size in bytes of internal storage used by the merged convolution context.

            The returned value reports internal temporary buffers and implementation-specific
            reordered weights, bias or activation parameters already allocated by the context.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution16bInternalBufferSize.

            \return a number of bytes used by internal buffers.
        */
        SIMD_INLINE size_t InternalBufferSize() const
        {
            return _context ? SimdSynetMergedConvolution16bInternalBufferSize(_context) : 0;
        }

        /*!
            Gets a short description of the selected BF16 merged convolution implementation.

            The returned string contains the implementation extension and algorithm name.
            The returned pointer is owned by the context and remains valid until the next call
            of this function or until the context is released.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution16bInfo.

            \return a string with description of internal implementation. NULL if the context was not created.
        */
        SIMD_INLINE const char * Info() const
        {
            return _context ? SimdSynetMergedConvolution16bInfo(_context) : NULL;
        }

        /*!
            Sets FP32 weights, biases and activation parameters for BF16 merged convolution.

            This function must be called before Forward(). The \a weight array contains pointers to FP32
            convolution weights, one per merged convolution. The selected implementation transforms
            weights to its internal BF16/reordered representation. Bias is copied to an internal FP32
            array; when a bias pointer is NULL, zeros are used. Activation parameters are copied or
            expanded to the internal FP32 array according to ::SimdConvolutionActivationType.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution16bSetParams.

            \param [in] weight - an array of pointers to FP32 convolution weights. The array size must be equal to the number of merged convolutions.
            \param [in] bias - an array of pointers to FP32 bias arrays, one per convolution. Each pointer can be NULL.
            \param [in] params - an array of pointers to activation parameters (see ::SimdConvolutionActivationType), one per convolution. The array itself must be valid; each element can be NULL for activations that do not use parameters.
        */
        SIMD_INLINE void SetParams(const float * const * weight, const float * const * bias, const float * const * params)
        {
            if (_context)
                SimdSynetMergedConvolution16bSetParams(_context, weight, bias, params);
        }

        /*!
            Performs BF16/FP32 merged convolution forward propagation.

            The function converts FP32 input to BF16 when the context source type is FP32, uses BF16 input
            directly when the source type is BF16, applies the fused convolution sequence stored in the
            context created by Init() and SetParams(), and writes FP32 or BF16 output according to the
            last convolution destination type. The \a buf argument can be NULL (it causes usage of internal buffer).
            If Init() was called with add equal to ::SimdTrue, the source tensor is added to the final output.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution16bForward.

            \param [in] src - a pointer to the input tensor bytes. The tensor type is determined by convs[0].srcT (FP32 or BF16).
            \param [out] buf - a pointer to an external temporary byte buffer. Can be NULL.
            \param [out] dst - a pointer to the output tensor bytes. The tensor type is determined by convs[count - 1].dstT (FP32 or BF16).
        */
        SIMD_INLINE void Forward(const uint8_t * src, uint8_t * buf, uint8_t * dst)
        {
            if (_context)
                SimdSynetMergedConvolution16bForward(_context, src, buf, dst);
        }

        /*!
            Releases internal context and clears stored merged convolution parameters.
        */
        SIMD_INLINE void Clear()
        {
            if (_context)
                SimdRelease(_context), _context = NULL;
            _batch = 0;
            _count = 0;
            _add = SimdFalse;
            SimdConvolutionParameters conv = {};
            _convs[0] = conv;
            _convs[1] = conv;
            _convs[2] = conv;
        }

    private:
        SIMD_INLINE bool Changed(const SimdConvolutionParameters * convs, size_t count) const
        {
            for (size_t i = 0; i < count; ++i)
            {
                const SimdConvolutionParameters & conv = convs[i];
                const SimdConvolutionParameters & prev = _convs[i];
                if (prev.srcC != conv.srcC || prev.srcH != conv.srcH || prev.srcW != conv.srcW ||
                    prev.srcT != conv.srcT || prev.srcF != conv.srcF ||
                    prev.dstC != conv.dstC || prev.dstH != conv.dstH || prev.dstW != conv.dstW ||
                    prev.dstT != conv.dstT || prev.dstF != conv.dstF ||
                    prev.kernelY != conv.kernelY || prev.kernelX != conv.kernelX ||
                    prev.dilationY != conv.dilationY || prev.dilationX != conv.dilationX ||
                    prev.strideY != conv.strideY || prev.strideX != conv.strideX ||
                    prev.padY != conv.padY || prev.padX != conv.padX ||
                    prev.padH != conv.padH || prev.padW != conv.padW ||
                    prev.group != conv.group || prev.activation != conv.activation)
                    return true;
            }
            return false;
        }

        void * _context;
        size_t _batch, _count;
        SimdBool _add;
        SimdConvolutionParameters _convs[3];
    };

    //-------------------------------------------------------------------------------------------------

    /*! @ingroup cpp_synet

        \short The SynetMergedConvolution8i class is a C++ wrapper of INT8 merged convolution.

        The class wraps C API functions ::SimdSynetMergedConvolution8iInit, ::SimdSynetMergedConvolution8iExternalBufferSize,
        ::SimdSynetMergedConvolution8iInternalBufferSize, ::SimdSynetMergedConvolution8iInfo, ::SimdSynetMergedConvolution8iSetParams and
        ::SimdSynetMergedConvolution8iForward. It fuses a sequence of two or three NHWC convolutions into one forward call:
        convolution + depthwise convolution, depthwise convolution + convolution, or
        convolution + depthwise convolution + convolution. Source and destination tensors can be
        FP32 or UINT8 according to the corresponding ::SimdConvolutionParameters fields. Ordinary
        convolutions use 1x1 or 3x3 kernels, depthwise convolutions use 3x3, 5x5 or 7x7 kernels;
        kernels and strides must be square, dilation must be 1 and stride must be 1, 2 or 3.
        Ordinary convolution weights are quantized to INT8 by SetParams().

        Call Init() and SetParams() before Forward(). Use Enable() to check that a context was created.
        The context is released by Clear() or by the destructor.

        Using example:
        \verbatim
        #include "Simd/SimdSynet.hpp"

        int main()
        {
            const size_t batch = 1, srcC = 4, srcH = 8, srcW = 8, midC = 8, count = 2;
            SimdConvolutionParameters convs[2] = {};

            convs[0].srcC = srcC;
            convs[0].srcH = srcH;
            convs[0].srcW = srcW;
            convs[0].srcT = SimdTensorData32f;
            convs[0].srcF = SimdTensorFormatNhwc;
            convs[0].dstC = midC;
            convs[0].kernelY = 1;
            convs[0].kernelX = 1;
            convs[0].dilationY = 1;
            convs[0].dilationX = 1;
            convs[0].strideY = 1;
            convs[0].strideX = 1;
            convs[0].padY = 0;
            convs[0].padX = 0;
            convs[0].padH = 0;
            convs[0].padW = 0;
            convs[0].group = 1;
            convs[0].activation = SimdConvolutionActivationIdentity;
            convs[0].dstH = srcH;
            convs[0].dstW = srcW;
            convs[0].dstT = SimdTensorData32f;
            convs[0].dstF = SimdTensorFormatNhwc;

            convs[1].srcC = midC;
            convs[1].srcH = convs[0].dstH;
            convs[1].srcW = convs[0].dstW;
            convs[1].srcT = SimdTensorData32f;
            convs[1].srcF = SimdTensorFormatNhwc;
            convs[1].dstC = midC;
            convs[1].kernelY = 3;
            convs[1].kernelX = 3;
            convs[1].dilationY = 1;
            convs[1].dilationX = 1;
            convs[1].strideY = 1;
            convs[1].strideX = 1;
            convs[1].padY = 1;
            convs[1].padX = 1;
            convs[1].padH = 1;
            convs[1].padW = 1;
            convs[1].group = midC;
            convs[1].activation = SimdConvolutionActivationIdentity;
            convs[1].dstH = convs[1].srcH;
            convs[1].dstW = convs[1].srcW;
            convs[1].dstT = SimdTensorData32f;
            convs[1].dstF = SimdTensorFormatNhwc;

            std::vector<float> src(batch * srcH * srcW * srcC);
            std::vector<float> weight0(convs[0].kernelY * convs[0].kernelX * convs[0].srcC * convs[0].dstC);
            std::vector<float> weight1(convs[1].kernelY * convs[1].kernelX * convs[1].dstC);
            std::vector<float> bias0(convs[0].dstC, 0.0f), bias1(convs[1].dstC, 0.0f);
            std::vector<float> srcMin(srcC, -1.0f), srcMax(srcC, 1.0f);
            std::vector<float> midMin(midC, -1.0f), midMax(midC, 1.0f);
            std::vector<float> dstMin(midC, -1.0f), dstMax(midC, 1.0f);
            std::vector<float> dst(batch * convs[1].dstH * convs[1].dstW * convs[1].dstC, 0.0f);
            const float * weight[2] = { weight0.data(), weight1.data() };
            const float * bias[2] = { bias0.data(), bias1.data() };
            const float * params[2] = { NULL, NULL };
            const float * stats[6] = { srcMin.data(), srcMax.data(), midMin.data(), midMax.data(), dstMin.data(), dstMax.data() };
            for (size_t i = 0; i < src.size(); ++i)
                src[i] = float(i) * 0.01f;
            for (size_t i = 0; i < weight0.size(); ++i)
                weight0[i] = float(i) * 0.02f;
            for (size_t i = 0; i < weight1.size(); ++i)
                weight1[i] = float(i) * 0.03f;

            Simd::SynetMergedConvolution8i mergedConvolution;
            mergedConvolution.Init(batch, convs, count);
            if (mergedConvolution.Enable())
            {
                mergedConvolution.SetParams(weight, NULL, bias, params, stats);
                mergedConvolution.Forward((const uint8_t*)src.data(), NULL, (uint8_t*)dst.data());
            }

            return 0;
        }
        \endverbatim
    */
    class SynetMergedConvolution8i
    {
    public:
        /*!
            Creates a new empty SynetMergedConvolution8i class.
        */
        SynetMergedConvolution8i()
            : _context(NULL)
            , _batch(0)
            , _count(0)
            , _compatibility(SimdSynetCompatibilityDefault)
        {
            SimdConvolutionParameters conv = {};
            _convs[0] = conv;
            _convs[1] = conv;
            _convs[2] = conv;
        }

        /*!
            SynetMergedConvolution8i class destructor. Releases internal context.
        */
        virtual ~SynetMergedConvolution8i()
        {
            Clear();
        }

        /*!
            Initializes (or re-initializes) an INT8 merged convolution context.

            Creates an internal context with using of function ::SimdSynetMergedConvolution8iInit.
            The context is recreated only if batch size, convolution count, compatibility flags or
            any convolution parameters were changed.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution8iInit.

            \param [in] batch - a batch size.
            \param [in] convs - an array with convolution parameters in execution order.
            \param [in] count - a number of merged convolutions. It must be 2 or 3.
            \param [in] compatibility - calculation compatibility flags (see ::SimdSynetCompatibilityType).
        */
        SIMD_INLINE void Init(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdSynetCompatibilityType compatibility = SimdSynetCompatibilityDefault)
        {
            if (convs == NULL || count < 2 || count > 3)
                return;
            if (_batch != batch || _count != count || _compatibility != compatibility || Changed(convs, count))
            {
                Clear();
                _batch = batch;
                _count = count;
                _compatibility = compatibility;
                for (size_t i = 0; i < count; ++i)
                    _convs[i] = convs[i];
                _context = SimdSynetMergedConvolution8iInit(_batch, _convs, _count, _compatibility);
            }
        }

        /*!
            Checks that the internal merged convolution context was created.

            \return true if the context exists and Forward() can be called.
        */
        SIMD_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        /*!
            Gets the size in bytes of caller-provided temporary buffer for INT8 merged convolution.

            The returned value is a number of bytes. It depends on the implementation selected
            during initialization and can be used when allocating the \a buf argument of Forward().
            Some implementations return 1 when they do not need external temporary storage.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution8iExternalBufferSize.

            \return a number of bytes required for external temporary buffer.
        */
        SIMD_INLINE size_t ExternalBufferSize() const
        {
            return _context ? SimdSynetMergedConvolution8iExternalBufferSize(_context) : 0;
        }

        /*!
            Gets the size in bytes of internal storage used by the merged convolution context.

            The returned value reports internal temporary buffers, quantized/reordered weights,
            conversion parameters, biases and activation parameters already allocated by the context.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution8iInternalBufferSize.

            \return a number of bytes used by internal buffers.
        */
        SIMD_INLINE size_t InternalBufferSize() const
        {
            return _context ? SimdSynetMergedConvolution8iInternalBufferSize(_context) : 0;
        }

        /*!
            Gets a short description of the selected INT8 merged convolution implementation.

            The returned string contains the implementation extension and algorithm name.
            The returned pointer is owned by the context and remains valid until the next call
            of this function or until the context is released.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution8iInfo.

            \return a string with description of internal implementation. NULL if the context was not created.
        */
        SIMD_INLINE const char * Info() const
        {
            return _context ? SimdSynetMergedConvolution8iInfo(_context) : NULL;
        }

        /*!
            Sets FP32 weights, biases, activation parameters and quantization statistics for INT8 merged convolution.

            This function must be called before Forward(). The \a weight array contains pointers to FP32
            convolution weights, one per merged convolution. The selected implementation quantizes and
            reorders weights into the context. If \a internal is not NULL, ::SimdTrue means the corresponding
            weights were copied/reordered into the context. Bias is copied to an internal FP32 array; when a
            bias pointer is NULL, zeros are used. Activation parameters are copied or expanded to the
            internal FP32 array according to ::SimdConvolutionActivationType. The \a stats array provides
            per-channel min/max ranges used to convert tensors between FP32 and UINT8.

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution8iSetParams.

            \param [in] weight - an array of pointers to FP32 convolution weights. The array size must be equal to the number of merged convolutions.
            \param [out] internal - an array of flags receiving weight storage mode. The array size must be equal to the number of merged convolutions. Can be NULL.
            \param [in] bias - an array of pointers to FP32 bias arrays, one per convolution. Each pointer can be NULL.
            \param [in] params - an array of pointers to activation parameters (see ::SimdConvolutionActivationType), one per convolution. Each pointer can be NULL for activations that do not use parameters.
            \param [in] stats - an array of six pointers to FP32 per-channel statistics: input min/max (stats[0], stats[1]), intermediate min/max before the last convolution (stats[2], stats[3]) and output min/max (stats[4], stats[5]).
        */
        SIMD_INLINE void SetParams(const float * const * weight, SimdBool * internal, const float * const * bias, const float * const * params, const float * const * stats)
        {
            if (_context)
                SimdSynetMergedConvolution8iSetParams(_context, weight, internal, bias, params, stats);
        }

        /*!
            Performs INT8 merged convolution forward propagation.

            The function converts FP32 input to UINT8 when the context source type is FP32, uses UINT8 input
            directly when the source type is UINT8, applies the fused convolution sequence stored in the
            context created by Init() and SetParams(), and writes FP32 or UINT8 output according to the
            last convolution destination type. The \a buf argument can be NULL (it causes usage of internal buffer).

            \note This function is a C++ wrapper for function ::SimdSynetMergedConvolution8iForward.

            \param [in] src - a pointer to the input tensor bytes. The tensor type is determined by convs[0].srcT (FP32 or UINT8).
            \param [out] buf - a pointer to an external temporary byte buffer. Can be NULL.
            \param [out] dst - a pointer to the output tensor bytes. The tensor type is determined by convs[count - 1].dstT (FP32 or UINT8).
        */
        SIMD_INLINE void Forward(const uint8_t * src, uint8_t * buf, uint8_t * dst)
        {
            if (_context)
                SimdSynetMergedConvolution8iForward(_context, src, buf, dst);
        }

        /*!
            Releases internal context and clears stored merged convolution parameters.
        */
        SIMD_INLINE void Clear()
        {
            if (_context)
                SimdRelease(_context), _context = NULL;
            _batch = 0;
            _count = 0;
            _compatibility = SimdSynetCompatibilityDefault;
            SimdConvolutionParameters conv = {};
            _convs[0] = conv;
            _convs[1] = conv;
            _convs[2] = conv;
        }

    private:
        SIMD_INLINE bool Changed(const SimdConvolutionParameters * convs, size_t count) const
        {
            for (size_t i = 0; i < count; ++i)
            {
                const SimdConvolutionParameters & conv = convs[i];
                const SimdConvolutionParameters & prev = _convs[i];
                if (prev.srcC != conv.srcC || prev.srcH != conv.srcH || prev.srcW != conv.srcW ||
                    prev.srcT != conv.srcT || prev.srcF != conv.srcF ||
                    prev.dstC != conv.dstC || prev.dstH != conv.dstH || prev.dstW != conv.dstW ||
                    prev.dstT != conv.dstT || prev.dstF != conv.dstF ||
                    prev.kernelY != conv.kernelY || prev.kernelX != conv.kernelX ||
                    prev.dilationY != conv.dilationY || prev.dilationX != conv.dilationX ||
                    prev.strideY != conv.strideY || prev.strideX != conv.strideX ||
                    prev.padY != conv.padY || prev.padX != conv.padX ||
                    prev.padH != conv.padH || prev.padW != conv.padW ||
                    prev.group != conv.group || prev.activation != conv.activation)
                    return true;
            }
            return false;
        }

        void * _context;
        size_t _batch, _count;
        SimdSynetCompatibilityType _compatibility;
        SimdConvolutionParameters _convs[3];
    };

    //-------------------------------------------------------------------------------------------------

    /*! @ingroup cpp_synet

        \short The SynetQuantizedMergedConvolution class is a C++ wrapper of UINT8 quantized merged convolution.

        The class wraps C API functions ::SimdSynetQuantizedMergedConvolutionInit, ::SimdSynetQuantizedMergedConvolutionExternalBufferSize,
        ::SimdSynetQuantizedMergedConvolutionInternalBufferSize, ::SimdSynetQuantizedMergedConvolutionInfo,
        ::SimdSynetQuantizedMergedConvolutionSetParams and ::SimdSynetQuantizedMergedConvolutionForward.
        It fuses a sequence of two or three NHWC UINT8-to-UINT8 quantized convolutions into one forward call:
        pointwise + depthwise, depthwise + pointwise, or pointwise + depthwise + pointwise. Source and
        destination tensors are UINT8, weights are INT8, and each tensor edge has its own scale and zero point.
        Ordinary convolutions use 1x1 or 3x3 kernels, depthwise convolutions use 3x3, 5x5 or 7x7 kernels;
        kernels and strides must be square, dilation must be 1 and stride must be 1, 2 or 3.
        If add is non-zero for a three-convolution chain, the final output is a requantized residual sum
        of the convolution output and the original input (add = 1 adds output to source, add = 2 adds source to output).

        Call Init() and SetParams() before Forward(). Use Enable() to check that a context was created.
        The context is released by Clear() or by the destructor.

        Using example:
        \verbatim
        #include "Simd/SimdSynet.hpp"

        int main()
        {
            const size_t batch = 1, srcC = 4, srcH = 8, srcW = 8, midC = 8, count = 2, add = 0;
            SimdConvolutionParameters convs[2] = {};

            convs[0].srcC = srcC;
            convs[0].srcH = srcH;
            convs[0].srcW = srcW;
            convs[0].srcT = SimdTensorData8u;
            convs[0].srcF = SimdTensorFormatNhwc;
            convs[0].dstC = midC;
            convs[0].kernelY = 1;
            convs[0].kernelX = 1;
            convs[0].dilationY = 1;
            convs[0].dilationX = 1;
            convs[0].strideY = 1;
            convs[0].strideX = 1;
            convs[0].padY = 0;
            convs[0].padX = 0;
            convs[0].padH = 0;
            convs[0].padW = 0;
            convs[0].group = 1;
            convs[0].activation = SimdConvolutionActivationIdentity;
            convs[0].dstH = srcH;
            convs[0].dstW = srcW;
            convs[0].dstT = SimdTensorData8u;
            convs[0].dstF = SimdTensorFormatNhwc;

            convs[1].srcC = midC;
            convs[1].srcH = convs[0].dstH;
            convs[1].srcW = convs[0].dstW;
            convs[1].srcT = SimdTensorData8u;
            convs[1].srcF = SimdTensorFormatNhwc;
            convs[1].dstC = midC;
            convs[1].kernelY = 3;
            convs[1].kernelX = 3;
            convs[1].dilationY = 1;
            convs[1].dilationX = 1;
            convs[1].strideY = 1;
            convs[1].strideX = 1;
            convs[1].padY = 1;
            convs[1].padX = 1;
            convs[1].padH = 1;
            convs[1].padW = 1;
            convs[1].group = midC;
            convs[1].activation = SimdConvolutionActivationIdentity;
            convs[1].dstH = convs[1].srcH;
            convs[1].dstW = convs[1].srcW;
            convs[1].dstT = SimdTensorData8u;
            convs[1].dstF = SimdTensorFormatNhwc;

            std::vector<uint8_t> src(batch * srcH * srcW * srcC);
            std::vector<int8_t> weight0(convs[0].kernelY * convs[0].kernelX * convs[0].srcC * convs[0].dstC);
            std::vector<int8_t> weight1(convs[1].kernelY * convs[1].kernelX * convs[1].dstC);
            std::vector<float> weightScale0(convs[0].dstC, 0.02f), weightScale1(convs[1].dstC, 0.03f);
            std::vector<int32_t> bias0(convs[0].dstC, 1), bias1(convs[1].dstC, 2);
            float ioScale[3] = { 0.01f, 0.015f, 0.02f };
            uint8_t ioZero[3] = { 128, 127, 126 };
            std::vector<uint8_t> dst(batch * convs[1].dstH * convs[1].dstW * convs[1].dstC, 0);
            const int8_t * weight[2] = { weight0.data(), weight1.data() };
            const float * weightScale[2] = { weightScale0.data(), weightScale1.data() };
            const int32_t * bias[2] = { bias0.data(), bias1.data() };
            for (size_t i = 0; i < src.size(); ++i)
                src[i] = uint8_t(i);
            for (size_t i = 0; i < weight0.size(); ++i)
                weight0[i] = int8_t(i);
            for (size_t i = 0; i < weight1.size(); ++i)
                weight1[i] = int8_t(i);

            Simd::SynetQuantizedMergedConvolution mergedConvolution;
            mergedConvolution.Init(batch, convs, count, add);
            if (mergedConvolution.Enable())
            {
                mergedConvolution.SetParams(ioScale, ioZero, weight, weightScale, bias);
                mergedConvolution.Forward(src.data(), NULL, dst.data());
            }

            return 0;
        }
        \endverbatim
    */
    class SynetQuantizedMergedConvolution
    {
    public:
        /*!
            Creates a new empty SynetQuantizedMergedConvolution class.
        */
        SynetQuantizedMergedConvolution()
            : _context(NULL)
            , _batch(0)
            , _count(0)
            , _add(0)
        {
            SimdConvolutionParameters conv = {};
            _convs[0] = conv;
            _convs[1] = conv;
            _convs[2] = conv;
        }

        /*!
            SynetQuantizedMergedConvolution class destructor. Releases internal context.
        */
        virtual ~SynetQuantizedMergedConvolution()
        {
            Clear();
        }

        /*!
            Initializes (or re-initializes) a quantized merged convolution context.

            Creates an internal context with using of function ::SimdSynetQuantizedMergedConvolutionInit.
            The context is recreated only if batch size, convolution count, residual-add mode or
            any convolution parameters were changed.

            \note This function is a C++ wrapper for function ::SimdSynetQuantizedMergedConvolutionInit.

            \param [in] batch - a batch size.
            \param [in] convs - an array with convolution parameters in execution order.
            \param [in] count - a number of merged convolutions. It must be 2 or 3.
            \param [in] add - a residual addition mode: 0 disables addition, 1 adds output to source, 2 adds source to output.
        */
        SIMD_INLINE void Init(size_t batch, const SimdConvolutionParameters * convs, size_t count, int add)
        {
            if (convs == NULL || count < 2 || count > 3)
                return;
            if (_batch != batch || _count != count || _add != add || Changed(convs, count))
            {
                Clear();
                _batch = batch;
                _count = count;
                _add = add;
                for (size_t i = 0; i < count; ++i)
                    _convs[i] = convs[i];
                _context = SimdSynetQuantizedMergedConvolutionInit(_batch, _convs, _count, _add);
            }
        }

        /*!
            Checks that the internal merged convolution context was created.

            \return true if the context exists and Forward() can be called.
        */
        SIMD_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        /*!
            Gets the size in bytes of caller-provided temporary buffer for quantized merged convolution.

            The returned value is a number of bytes. It depends on the implementation selected
            during initialization and can be used when allocating the \a buf argument of Forward().
            Some implementations return 1 when they do not need external temporary storage.

            \note This function is a C++ wrapper for function ::SimdSynetQuantizedMergedConvolutionExternalBufferSize.

            \return a number of bytes required for external temporary buffer.
        */
        SIMD_INLINE size_t ExternalBufferSize() const
        {
            return _context ? SimdSynetQuantizedMergedConvolutionExternalBufferSize(_context) : 0;
        }

        /*!
            Gets the size in bytes of internal storage used by the quantized merged convolution context.

            The returned value reports internal temporary buffers, reordered weights, biases, norms,
            zero points and an optional fallback temporary buffer already allocated by the context.

            \note This function is a C++ wrapper for function ::SimdSynetQuantizedMergedConvolutionInternalBufferSize.

            \return a number of bytes used by internal buffers.
        */
        SIMD_INLINE size_t InternalBufferSize() const
        {
            return _context ? SimdSynetQuantizedMergedConvolutionInternalBufferSize(_context) : 0;
        }

        /*!
            Gets a short description of the selected quantized merged convolution implementation.

            The returned string contains the implementation extension and algorithm name.
            The returned pointer is owned by the context and remains valid until the next call
            of this function or until the context is released.

            \note This function is a C++ wrapper for function ::SimdSynetQuantizedMergedConvolutionInfo.

            \return a string with description of internal implementation. NULL if the context was not created.
        */
        SIMD_INLINE const char * Info() const
        {
            return _context ? SimdSynetQuantizedMergedConvolutionInfo(_context) : NULL;
        }

        /*!
            Sets INT8 weights, INT32 biases and quantization parameters for quantized merged convolution.

            This function must be called before Forward(). Arrays \a weight, \a weightScale and \a bias
            contain one pointer per merged convolution. The \a ioScale and \a ioZero arrays contain
            quantization parameters for every edge between convolutions: input, intermediate outputs
            and final output. When residual addition is enabled, one additional scale and zero point
            are used for the residual-sum output. Individual bias pointers can be NULL.

            \note This function is a C++ wrapper for function ::SimdSynetQuantizedMergedConvolutionSetParams.

            \param [in] ioScale - a pointer to FP32 input/intermediate/output tensor scales.
            \param [in] ioZero - a pointer to UINT8 input/intermediate/output tensor zero points.
            \param [in] weight - an array of pointers to INT8 convolution weights. The array size must be equal to the number of merged convolutions.
            \param [in] weightScale - an array of pointers to per-output-channel FP32 weight scales. The array size must be equal to the number of merged convolutions.
            \param [in] bias - an array of pointers to per-output-channel INT32 biases. The array size must be equal to the number of merged convolutions. Individual pointers can be NULL.
        */
        SIMD_INLINE void SetParams(const float * ioScale, const uint8_t * ioZero, const int8_t * const * weight, const float * const * weightScale, const int32_t * const * bias)
        {
            if (_context)
                SimdSynetQuantizedMergedConvolutionSetParams(_context, ioScale, ioZero, weight, weightScale, bias);
        }

        /*!
            Performs quantized merged convolution forward propagation.

            The function applies the fused UINT8 convolution sequence stored in the context created
            by Init() and SetParams(). The \a buf argument can be NULL (it causes usage of internal buffer).
            If Init() was called with a non-zero add mode, the source tensor is combined with the
            convolution output and the result is requantized to UINT8.

            \note This function is a C++ wrapper for function ::SimdSynetQuantizedMergedConvolutionForward.

            \param [in] src - a pointer to UINT8 input tensor of the first convolution.
            \param [out] buf - a pointer to an external temporary byte buffer. Can be NULL.
            \param [out] dst - a pointer to UINT8 output tensor of the last convolution or residual sum.
        */
        SIMD_INLINE void Forward(const uint8_t * src, uint8_t * buf, uint8_t * dst)
        {
            if (_context)
                SimdSynetQuantizedMergedConvolutionForward(_context, src, buf, dst);
        }

        /*!
            Releases internal context and clears stored merged convolution parameters.
        */
        SIMD_INLINE void Clear()
        {
            if (_context)
                SimdRelease(_context), _context = NULL;
            _batch = 0;
            _count = 0;
            _add = 0;
            SimdConvolutionParameters conv = {};
            _convs[0] = conv;
            _convs[1] = conv;
            _convs[2] = conv;
        }

    private:
        SIMD_INLINE bool Changed(const SimdConvolutionParameters * convs, size_t count) const
        {
            for (size_t i = 0; i < count; ++i)
            {
                const SimdConvolutionParameters & conv = convs[i];
                const SimdConvolutionParameters & prev = _convs[i];
                if (prev.srcC != conv.srcC || prev.srcH != conv.srcH || prev.srcW != conv.srcW ||
                    prev.srcT != conv.srcT || prev.srcF != conv.srcF ||
                    prev.dstC != conv.dstC || prev.dstH != conv.dstH || prev.dstW != conv.dstW ||
                    prev.dstT != conv.dstT || prev.dstF != conv.dstF ||
                    prev.kernelY != conv.kernelY || prev.kernelX != conv.kernelX ||
                    prev.dilationY != conv.dilationY || prev.dilationX != conv.dilationX ||
                    prev.strideY != conv.strideY || prev.strideX != conv.strideX ||
                    prev.padY != conv.padY || prev.padX != conv.padX ||
                    prev.padH != conv.padH || prev.padW != conv.padW ||
                    prev.group != conv.group || prev.activation != conv.activation)
                    return true;
            }
            return false;
        }

        void * _context;
        size_t _batch, _count;
        int _add;
        SimdConvolutionParameters _convs[3];
    };
}

#endif
