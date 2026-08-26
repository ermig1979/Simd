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
}

#endif
