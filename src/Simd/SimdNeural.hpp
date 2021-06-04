/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#ifndef __SimdNeural_hpp__
#define __SimdNeural_hpp__

#include "Simd/SimdLib.hpp"
#include "Simd/SimdParallel.hpp"

#include <float.h>

#include <numeric>
#include <random>
#include <iterator>

#ifndef SIMD_CHECK_PERFORMANCE
#define SIMD_CHECK_PERFORMANCE()
#endif

//#define SIMD_CHECK_OVERFLOW

#if defined(SIMD_CHECK_OVERFLOW) && !defined(NDEBUG)
#define SIMD_CHECK_OVERFLOW_1(vector) Simd::Neural::Detail::CheckOverflow(vector.data(), vector.size());
#define SIMD_CHECK_OVERFLOW_2(data, size) Simd::Neural::Detail::CheckOverflow(data, size);
#else
#define SIMD_CHECK_OVERFLOW_1(vector)
#define SIMD_CHECK_OVERFLOW_2(data, size)
#endif

namespace Simd
{
    /*! @ingroup cpp_neural

        \short Contains Framework for learning of Convolutional Neural Network.
    */
    namespace Neural
    {
        typedef Point<ptrdiff_t> Size; /*!< \brief 2D-size (width and height). */
        typedef std::vector<uint8_t, Allocator<uint8_t>> Buffer; /*!< \brief Vector with 8-bit unsigned integer values. */
        typedef std::vector<float, Allocator<float>> Vector; /*!< \brief Vector with 32-bit float point values. */
        typedef std::vector<ptrdiff_t, Allocator<ptrdiff_t>> VectorI; /*!< \brief Vector with integer values. */
        typedef std::vector<Vector> Vectors; /*!< \brief Vector of vectors with 32-bit float point values. */
        typedef size_t Label; /*!< \brief Integer name (label) of object class. */
        typedef std::vector<Label> Labels; /*!< \brief Vector of labels. */
        typedef Simd::View<Allocator> View; /*!< \brief Image. */

        namespace Detail
        {
            template <class T, class A> SIMD_INLINE void SetZero(std::vector<T, A> & vector)
            {
                memset(vector.data(), 0, vector.size() * sizeof(T));
            }

            SIMD_INLINE int RandomUniform(int min, int max)
            {
                static std::mt19937 gen(1);
                std::uniform_int_distribution<int> dst(min, max);
                return dst(gen);
            }

            SIMD_INLINE float RandomUniform(float min, float max)
            {
                static std::mt19937 gen(1);
                std::uniform_real_distribution<float> dst(min, max);
                return dst(gen);
            }

            SIMD_INLINE void CheckOverflow(const float * data, size_t size)
            {
                for (size_t i = 0; i < size; ++i)
                {
                    const float & value = data[i];
                    bool isNaN = (value != value);
                    bool isInfinity = (value == std::numeric_limits<double>::infinity() || value == -std::numeric_limits<double>::infinity());
                    if (isNaN || isInfinity)
                        throw std::runtime_error("Float overflow!");
                }
            }
        }

        /*! @ingroup cpp_neural

            \short Activation Function structure.

            Provides activation functions and their derivatives.
        */
        struct Function
        {
            /*!
                \enum Type

                Describes types of activation function. It is used in order to create a Layer in Network.
            */
            enum Type
            {
                /*!
                    Identity:
                    \verbatim
                    f(x) = x;
                    \endverbatim
                    \verbatim
                    df(y) = 1;
                    \endverbatim
                */
                Identity,
                /*! Hyperbolic Tangent:
                    \verbatim
                    f(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x));
                    \endverbatim
                    \verbatim
                    df(y) = 1 - y*y;
                    \endverbatim
                    See implementation details: ::SimdNeuralRoughTanh and ::SimdNeuralDerivativeTanh.
                */
                Tanh,
                /*! Sigmoid:
                    \verbatim
                    f(x) = 1/(1 + exp(-x));
                    \endverbatim
                    \verbatim
                    df(y) = (1 - y)*y;
                    \endverbatim
                    See implementation details: ::SimdNeuralRoughSigmoid2 and ::SimdNeuralDerivativeSigmoid.
                */
                Sigmoid,
                /*! ReLU (Rectified Linear Unit):
                    \verbatim
                    f(x) = max(0, x);
                    \endverbatim
                    \verbatim
                    df(y) = y > 0 ? 1 : 0;
                    \endverbatim
                    See implementation details: ::SimdSynetRelu32f and ::SimdNeuralDerivativeRelu.
                */
                Relu,
                /*! Leaky ReLU(Rectified Linear Unit):
                    \verbatim
                    f(x) = x > 0 ? x : 0.01*x;
                    \endverbatim
                    \verbatim
                    df(y) = y > 0 ? 1 : 0.01;
                    \endverbatim
                    See implementation details: ::SimdSynetRelu32f and ::SimdNeuralDerivativeRelu.
                */
                LeakyRelu,
                /*! Softmax (normalized exponential function):
                    \verbatim
                    f(x[i]) = exp(x[i])/sum(exp(x[i]));
                    \endverbatim
                    \verbatim
                    df(y) = y*(1 - y);
                    \endverbatim
                */
                Softmax,
            } const type;

            typedef void(*FuncPtr) (const float * src, size_t size, float * dst);
            FuncPtr function, derivative;

            float min, max;

            Function(Type t)
                : type(t)
                , function(IdentityFunction)
                , derivative(IdentityDerivative)
                , min(0.1f)
                , max(0.9f)
            {
                switch (type)
                {
                case Identity:
                    break;
                case Tanh:
                    function = TanhFunction;
                    derivative = TanhDerivative;
                    min = -0.8f;
                    max = 0.8f;
                    break;
                case Sigmoid:
                    function = SigmoidFunction;
                    derivative = SigmoidDerivative;
                    break;
                case Relu:
                    function = ReluFunction;
                    derivative = ReluDerivative;
                    break;
                case LeakyRelu:
                    function = LeakyReluFunction;
                    derivative = LeakyReluDerivative;
                    break;
                case Softmax:
                    function = SoftmaxFunction;
                    derivative = SoftmaxDerivative;
                    min = 0.0f;
                    max = 1.0f;
                    break;
                }
            }

        private:

            static SIMD_INLINE void IdentityFunction(const float * src, size_t size, float * dst)
            {
                if (src != dst)
                    memcpy(dst, src, sizeof(float)*size);
            }

            static SIMD_INLINE void IdentityDerivative(const float * src, size_t size, float * dst)
            {
            }

            static SIMD_INLINE void TanhFunction(const float * src, size_t size, float * dst)
            {
                const float slope = 1.0f;
                ::SimdNeuralRoughTanh(src, size, &slope, dst);
            }

            static SIMD_INLINE void TanhDerivative(const float * src, size_t size, float * dst)
            {
                const float slope = 1.0f;
                ::SimdNeuralDerivativeTanh(src, size, &slope, dst);
            }

            static SIMD_INLINE void SigmoidFunction(const float * src, size_t size, float * dst)
            {
                const float slope = 1.0f;
                ::SimdNeuralRoughSigmoid2(src, size, &slope, dst);
            }

            static SIMD_INLINE void SigmoidDerivative(const float * src, size_t size, float * dst)
            {
                const float slope = 1.0f;
                ::SimdNeuralDerivativeSigmoid(src, size, &slope, dst);
            }

            static SIMD_INLINE void ReluFunction(const float * src, size_t size, float * dst)
            {
                const float slope = 0.0f;
                ::SimdSynetRelu32f(src, size, &slope, dst);
            }

            static SIMD_INLINE void ReluDerivative(const float * src, size_t size, float * dst)
            {
                const float slope = 0.0f;
                ::SimdNeuralDerivativeRelu(src, size, &slope, dst);
            }

            static SIMD_INLINE void LeakyReluFunction(const float * src, size_t size, float * dst)
            {
                const float slope = 0.01f;
                ::SimdSynetRelu32f(src, size, &slope, dst);
            }

            static SIMD_INLINE void LeakyReluDerivative(const float * src, size_t size, float * dst)
            {
                const float slope = 0.01f;
                ::SimdNeuralDerivativeRelu(src, size, &slope, dst);
            }

            static SIMD_INLINE void SoftmaxFunction(const float * src, size_t size, float * dst)
            {
                float max = -FLT_MAX;
                for (size_t i = 0; i < size; ++i)
                    max = std::max(max, src[i]);
                float sum = 0;
                for (size_t i = 0; i < size; ++i)
                    sum += ::exp(src[i] - max);
                for (size_t i = 0; i < size; ++i)
                    dst[i] = ::exp(src[i] - max) / sum;
            }

            static SIMD_INLINE void SoftmaxDerivative(const float * src, size_t size, float * dst)
            {
                for (size_t i = 0; i < size; ++i)
                    dst[i] = src[i] * (1.0f - src[i]);
            }
        };

        /*! @ingroup cpp_neural

            \short Index structure.

            Provides access to complex 3D-array inside simple 1D-array.
        */
        struct Index
        {
            ptrdiff_t width; /*!< \brief A width of 3D-array. */
            ptrdiff_t height; /*!< \brief A height of 3D-array. */
            ptrdiff_t depth; /*!< \brief A depth of 3D-array. */

            /*!
                Creates a new Index structure that contains the default (0, 0, 0) parameters.
            */
            SIMD_INLINE Index()
                : width(0)
                , height(0)
                , depth(0)
            {
            }

            /*!
                Creates a new Index structure that contains the specified parameters.

                \param [in] w - initial value for width.
                \param [in] h - initial value for height.
                \param [in] d - initial value for depth (default value is equal to 1).
            */
            SIMD_INLINE Index(ptrdiff_t w, ptrdiff_t h, ptrdiff_t d = 1)
                : width(w)
                , height(h)
                , depth(d)
            {
            }

            /*!
                Creates a new Index structure that contains the specified parameters.

                \param [in] s - initial size (width and height).
                \param [in] d - initial value for depth (default value is equal to 1).
            */
            SIMD_INLINE Index(const Size & s, ptrdiff_t d = 1)
                : width(s.x)
                , height(s.y)
                , depth(d)
            {
            }

            /*!
                Creates a new Index structure on the base of another Index structure.

                \param [in] i - another Index structure.
            */
            SIMD_INLINE Index(const Index & i)
                : width(i.width)
                , height(i.height)
                , depth(i.depth)
            {
            }

            /*!
                Recreates the Index structure with specified width, height and depth.

                \param [in] w - new value for width.
                \param [in] h - new value for height.
                \param [in] d - new value for depth.
            */
            SIMD_INLINE void Resize(ptrdiff_t w, ptrdiff_t h, ptrdiff_t d)
            {
                width = w;
                height = h;
                depth = d;
            }

            /*!
                Recreates the Index structure with specified width, height and depth.

                \param [in] s - new size (width and height).
                \param [in] d - new value for depth.
            */
            SIMD_INLINE void Resize(const Size & s, ptrdiff_t d)
            {
                width = s.x;
                height = s.y;
                depth = d;
            }

            /*!
                Gets offset for 3D-point with specified coordinates.

                \param [in] x - x-coordinate of 3D-point.
                \param [in] y - y-coordinate of 3D-point.
                \param [in] c - z-coordinate(channel) of 3D-point.
                \return - an offset of the point in 1D-array.
            */
            SIMD_INLINE ptrdiff_t Offset(ptrdiff_t x, ptrdiff_t y, ptrdiff_t c) const
            {
                return (height * c + y) * width + x;
            }

            /*!
                Gets constant address of 3D-point in 1D-array (std::vector).

                \param [in] v - std::vector.
                \param [in] x - x-coordinate of 3D-point.
                \param [in] y - y-coordinate of 3D-point.
                \param [in] c - z-coordinate(channel) of 3D-point.
                \return - a constant address of the point in std::vector.
            */
            template<class T, class A> SIMD_INLINE const T * Get(const std::vector<T, A> & v, ptrdiff_t x, ptrdiff_t y, ptrdiff_t c) const
            {
                size_t offset = Offset(x, y, c);
                assert(offset < v.size());
                return v.data() + offset;
            }

            /*!
                Gets address of 3D-point in 1D-array (std::vector).

                \param [in] v - std::vector.
                \param [in] x - x-coordinate of 3D-point.
                \param [in] y - y-coordinate of 3D-point.
                \param [in] c - z-coordinate(channel) of 3D-point.
                \return - an address of the point in std::vector.
            */
            template<class T, class A> SIMD_INLINE T * Get(std::vector<T, A> & v, ptrdiff_t x, ptrdiff_t y, ptrdiff_t c) const
            {
                size_t offset = Offset(x, y, c);
                assert(offset < v.size());
                return v.data() + offset;
            }

            /*!
                Gets 2D-size (width and height) of the channel in the Index.

                \return - a new Point structure with width and height.
            */
            SIMD_INLINE Neural::Size Size() const
            {
                return Neural::Size(width, height);
            }

            /*!
                Gets area of the channel plane in the Index.

                \return - an area of the channel.
            */
            SIMD_INLINE ptrdiff_t Area() const
            {
                return width * height;
            }

            /*!
                Gets total size (volume) of the Index.

                \return - total size (volume) of 3D-array.
            */
            SIMD_INLINE ptrdiff_t Volume() const
            {
                return width * height * depth;
            }
        };

        /*! @ingroup cpp_neural

            \short Layer class.

            Abstract base class for all possible layers.
        */
        class Layer
        {
        public:
            /*!
                \enum Type

                Describes types of network layers.
            */
            enum Type
            {
                Input, /*!< \brief Layer type corresponding to Simd::Neural::InputLayer. */
                Convolutional, /*!< \brief Layer type corresponding to Simd::Neural::ConvolutionalLayer. */
                MaxPooling, /*!< \brief Layer type corresponding to Simd::Neural::MaxPoolingLayer. */
                AveragePooling, /*!< \brief Layer type corresponding to Simd::Neural::AveragePooling. */
                FullyConnected, /*!< \brief Layer type corresponding to Simd::Neural::FullyConnectedLayer. */
                Dropout, /*!< \brief Layer type corresponding to Simd::Neural::DropoutLayer. */
            };

            /*!
                \enum Method

                Describes method of forward propagation in the network layer.
            */
            enum Method
            {
                Fast, /*!< \brief The fastest method. It is incompatible with train process.*/
                Check, /*!< \brief Control checking during train process.*/
                Train, /*!< \brief Forward propagation in train process.*/
            };

            /*!
                Virtual destructor.
            */
            virtual ~Layer()
            {
            }

            virtual void Forward(const Vector & src, size_t thread, Method method) = 0;

            virtual void Backward(const Vector & src, size_t thread) = 0;

            virtual size_t FanSrc() const = 0;

            virtual size_t FanDst() const = 0;

            virtual void SetThreadNumber(size_t number, bool train)
            {
                _common.resize(number);
                for (size_t i = 0; i < _common.size(); ++i)
                {
                    _common[i].sum.resize(_dst.Volume());
                    _common[i].dst.resize(_dst.Volume());
                    if (train)
                    {
                        _common[i].dWeight.resize(_weight.size());
                        _common[i].dBias.resize(_bias.size());
                        _common[i].prevDelta.resize(_src.Volume());
                    }
                }
                if (train)
                {
                    _gWeight.resize(_weight.size());
                    _gBias.resize(_bias.size());
                }
            }

        protected:
            Layer(Layer::Type l, Function::Type f)
                : _type(l)
                , _function(f)
                , _prev(0)
                , _next(0)
            {
            }

            SIMD_INLINE bool Link(Layer * prev)
            {
                if (prev->_dst.Volume() == _src.Volume())
                {
                    _prev = prev;
                    prev->_next = this;
                    return true;
                }
                else
                    return false;
            }

            SIMD_INLINE const Vector & Dst(size_t thread) const
            {
                return _common[thread].dst;
            }

            SIMD_INLINE const Vector & Delta(size_t thread) const
            {
                return _common[thread].prevDelta;
            }

            const Type _type;
            const Function _function;

            Layer * _prev, *_next;

            Index _src, _dst;
            Vector _weight, _bias, _gWeight, _gBias;

            struct Common
            {
                Vector sum, dst;

                Vector dWeight, dBias, prevDelta;
            };
            std::vector<Common> _common;

            friend class InputLayer;
            friend class ConvolutionalLayer;
            friend class PoolingLayer;
            friend class MaxPoolingLayer;
            friend class AveragePoolingLayer;
            friend class FullyConnectedLayer;
            friend class DropoutLayer;
            friend class Network;
        };
        typedef std::shared_ptr<Layer> LayerPtr;
        typedef std::vector<LayerPtr> LayerPtrs;

        /*! @ingroup cpp_neural

            \short InputLayer class.

            First input layer in neural network. This layer can't be created, it is added automatically.
        */
        class InputLayer : public Layer
        {
        public:
            void Forward(const Vector & src, size_t thread, Method method) override
            {
                _common[thread].dst = src;
            }

            void Backward(const Vector & src, size_t thread) override
            {
            }

            size_t FanSrc() const override
            {
                return 1;
            }

            size_t FanDst() const override
            {
                return 1;
            }

        private:
            InputLayer(const Layer & next)
                : Layer(Input, Function::Identity)
            {
                _dst = next._src;
                SetThreadNumber(1, false);
            }

            friend class Network;
        };

        /*! @ingroup cpp_neural

            \short ConfolutionLayer class.

            Convolutional layer in neural network.
        */
        class ConvolutionalLayer : public Layer
        {
        public:
            /*!
                \short Creates new ConfolutionLayer class.

                \param [in] f - a type of activation function used in this layer.
                \param [in] srcSize - a size (width and height) of input image.
                \param [in] srcDepth - a number of input channels (images).
                \param [in] dstDepth - a number of output channels (images).
                \param [in] coreSize - a size of convolution core.
                \param [in] valid - a boolean flag (True - only original image points are used in convolution, so output image is decreased;
                                    False - input image is padded by zeros and output image has the same size). By default its true.
                \param [in] bias - a boolean flag (enabling of bias). By default its True.
                \param [in] connection - a table of connections between input and output channels. By default all channels are connected.
            */
            ConvolutionalLayer(Function::Type f, const Size & srcSize, size_t srcDepth, size_t dstDepth, const Size & coreSize,
                bool valid = true, bool bias = true, const View & connection = View())
                : Layer(Convolutional, f)
                , _functionForward(0)
                , _functionBackward(0)
                , _functionSum(0)
            {
                _valid = valid;
                _indent = coreSize / 2;
                Size pad = coreSize - Size(1, 1);
                _src.Resize(srcSize, srcDepth);
                _dst.Resize(srcSize - (_valid ? pad : Size()), dstDepth);
                _padded.Resize(srcSize + (_valid ? Size() : pad), srcDepth);
                _core.Resize(coreSize, srcDepth*dstDepth);
                _weight.resize(_core.Volume());
                if (bias)
                    _bias.resize(dstDepth);
                SetThreadNumber(1, false);

                _connection.Recreate(dstDepth, srcDepth, View::Gray8);
                _partial = Simd::Compatible(connection, _connection);
                if (_partial)
                    Simd::Copy(connection, _connection);
                else
                    Simd::Fill(_connection, 1);

                if (_core.width == 2 && _core.height == 2)
                {
                    _functionForward = ::SimdNeuralAddConvolution2x2Forward;
                    _functionBackward = ::SimdNeuralAddConvolution2x2Backward;
                    _functionSum = ::SimdNeuralAddConvolution2x2Sum;
                }
                if (_core.width == 3 && _core.height == 3)
                {
                    _functionForward = ::SimdNeuralAddConvolution3x3Forward;
                    _functionBackward = ::SimdNeuralAddConvolution3x3Backward;
                    _functionSum = ::SimdNeuralAddConvolution3x3Sum;
                }
                if (_core.width == 4 && _core.height == 4)
                {
                    _functionForward = ::SimdNeuralAddConvolution4x4Forward;
                    _functionBackward = ::SimdNeuralAddConvolution4x4Backward;
                    _functionSum = ::SimdNeuralAddConvolution4x4Sum;
                }
                if (_core.width == 5 && _core.height == 5)
                {
                    _functionForward = ::SimdNeuralAddConvolution5x5Forward;
                    _functionBackward = ::SimdNeuralAddConvolution5x5Backward;
                    _functionSum = ::SimdNeuralAddConvolution5x5Sum;
                }
            }

            void Forward(const Vector & src, size_t thread, Method method) override
            {
                const Vector & padded = PaddedSrc(src, thread);
                Vector & sum = _common[thread].sum;
                Vector & dst = _common[thread].dst;
                if (_partial)
                {
                    Detail::SetZero(sum);
                    for (ptrdiff_t dc = 0; dc < _dst.depth; ++dc)
                    {
                        for (ptrdiff_t sc = 0; sc < _src.depth; ++sc)
                        {
                            if (!_connection.At<bool>(dc, sc))
                                return;

                            const float * pweight = _core.Get(_weight, 0, 0, _src.depth*dc + sc);
                            const float * psrc = _padded.Get(padded, 0, 0, sc);
                            float * psum = _dst.Get(sum, 0, 0, dc);

                            if (_functionForward)
                                _functionForward(psrc, _padded.width, _dst.width, _dst.height, pweight, psum, _dst.width);
                            else if (_core.width == 1 && _core.height == 1)
                                ::SimdNeuralAddVectorMultipliedByValue(psrc, _dst.width*_dst.height, pweight, psum);
                            else
                            {
                                for (ptrdiff_t y = 0; y < _dst.height; y++)
                                {
                                    for (ptrdiff_t x = 0; x < _dst.width; x++)
                                    {
                                        const float * pw = pweight;
                                        const float * ps = psrc + y * _padded.width + x;
                                        float s = 0;
                                        for (ptrdiff_t wy = 0; wy < _core.height; wy++)
                                            for (ptrdiff_t wx = 0; wx < _core.width; wx++)
                                                s += *pw++ * ps[wy * _padded.width + wx];
                                        psum[y * _dst.width + x] += s;
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    Buffer & buffer = _specific[thread].buffer;
                    size_t size = buffer.size();
                    ::SimdNeuralConvolutionForward(padded.data(), _padded.width, _padded.height, _padded.depth, _weight.data(),
                        _core.width, _core.height, 0, 0, 1, 1, 1, 1, buffer.data(), &size, sum.data(), _dst.width, _dst.height, _dst.depth, 0);
                    if (size > buffer.size())
                        buffer.resize(size);
                }
                for (ptrdiff_t dc = 0; dc < _dst.depth; ++dc)
                {
                    if (_bias.size())
                        ::SimdNeuralAddValue(_bias.data() + dc, _dst.Get(sum, 0, 0, dc), _dst.Area());
                }
                _function.function(sum.data(), sum.size(), dst.data());
            }

            void Backward(const Vector & currDelta, size_t thread) override
            {
                const Vector & prevDst = _valid ? _prev->Dst(thread) : _specific[thread].paddedSrc;
                Vector & prevDelta = _valid ? _common[thread].prevDelta : _specific[thread].paddedDelta;
                Vector & dWeight = _common[thread].dWeight;
                Vector & dBias = _common[thread].dBias;

                Detail::SetZero(prevDelta);

                for (ptrdiff_t sc = 0; sc < _src.depth; ++sc)
                {
                    for (ptrdiff_t dc = 0; dc < _dst.depth; ++dc)
                    {
                        if (!_connection.At<bool>(dc, sc))
                            return;

                        const float * pweight = _core.Get(_weight, 0, 0, _src.depth*dc + sc);
                        const float * psrc = _dst.Get(currDelta, 0, 0, dc);
                        float * pdst = _padded.Get(prevDelta, 0, 0, sc);

                        if (_functionBackward)
                            _functionBackward(psrc, _dst.width, _dst.width, _dst.height, pweight, pdst, _padded.width);
                        else if (_core.width == 1 && _core.height == 1)
                            ::SimdNeuralAddVectorMultipliedByValue(psrc, _dst.width*_dst.height, pweight, pdst);
                        else
                        {
                            for (ptrdiff_t y = 0; y < _dst.height; y++)
                            {
                                for (ptrdiff_t x = 0; x < _dst.width; x++)
                                {
                                    const float * ppweight = pweight;
                                    const float ppsrc = psrc[y*_dst.width + x];
                                    float * ppdst = pdst + y*_padded.width + x;
                                    for (ptrdiff_t wy = 0; wy < _core.height; wy++)
                                        for (ptrdiff_t wx = 0; wx < _core.width; wx++)
                                            ppdst[wy * _padded.width + wx] += *ppweight++ * ppsrc;
                                }
                            }
                        }
                    }
                }

                _prev->_function.derivative(&prevDst[0], prevDst.size(), &prevDelta[0]);

                for (ptrdiff_t sc = 0; sc < _src.depth; ++sc)
                {
                    for (ptrdiff_t dc = 0; dc < _dst.depth; ++dc)
                    {
                        const float * delta = _dst.Get(currDelta, 0, 0, dc);
                        const float * prevo = _padded.Get(prevDst, 0, 0, sc);
                        float * sums = _core.Get(dWeight, 0, 0, _src.depth*dc + sc);

                        if (_functionSum)
                            _functionSum(prevo, _padded.width, delta, _dst.width, _dst.width, _dst.height, sums);
                        else if (_core.width == 1 && _core.height == 1)
                        {
                            float sum;
                            ::SimdNeuralProductSum(prevo, delta, _dst.width*_dst.height, &sum);
                            sums[0] += sum;
                        }
                        else
                        {
                            for (ptrdiff_t wy = 0; wy < _core.height; wy++)
                            {
                                for (ptrdiff_t wx = 0; wx < _core.width; wx++)
                                {
                                    float dst = 0;
                                    const float * prevo = _padded.Get(prevDst, wx, wy, sc);
                                    for (ptrdiff_t y = 0; y < _dst.height; y++)
                                    {
                                        float sum;
                                        ::SimdNeuralProductSum(prevo + y*_padded.width, delta + y*_dst.width, _dst.width, &sum);
                                        dst += sum;
                                    }
                                    dWeight[_core.Offset(wx, wy, _src.depth *dc + sc)] += dst;
                                }
                            }
                        }
                    }
                }

                if (dBias.size())
                {
                    for (ptrdiff_t dc = 0; dc < _dst.depth; ++dc)
                    {
                        const float * delta = _dst.Get(currDelta, 0, 0, dc);
                        dBias[dc] += std::accumulate(delta, delta + _dst.width*_dst.height, float(0));
                    }
                }

                UnpadDelta(prevDelta, thread);
            }

            size_t FanSrc() const override
            {
                return _core.width*_core.height*_src.depth;
            }

            size_t FanDst() const override
            {
                return _core.width*_core.height*_dst.depth;
            }

            virtual void SetThreadNumber(size_t number, bool train) override
            {
                Layer::SetThreadNumber(number, train);
                _specific.resize(number);
                for (size_t i = 0; i < _specific.size(); ++i)
                {
                    if (!_valid)
                    {
                        _specific[i].paddedSrc.resize(_padded.Volume(), 0);
                        if (train)
                            _specific[i].paddedDelta.resize(_padded.Volume(), 0);
                    }
                }
            }

        private:

            const Vector & PaddedSrc(const Vector & src, size_t thread)
            {
                if (_valid)
                    return src;
                else
                {
                    Vector & padded = _specific[thread].paddedSrc;
                    size_t size = _src.width * sizeof(float);
                    for (ptrdiff_t c = 0; c < _src.depth; ++c)
                    {
                        for (ptrdiff_t y = 0; y < _src.height; ++y)
                            memcpy(_padded.Get(padded, _indent.x, _indent.y + y, c), _src.Get(src, 0, y, c), size);
                    }
                    return padded;
                }
            }

            void UnpadDelta(const Vector & src, size_t thread)
            {
                if (!_valid)
                {
                    Vector & dst = _common[thread].prevDelta;
                    size_t size = _src.width * sizeof(float);
                    for (ptrdiff_t c = 0; c < _src.depth; c++)
                    {
                        for (ptrdiff_t y = 0; y < _src.height; ++y)
                            memcpy(_src.Get(dst, 0, y, c), _padded.Get(src, _indent.x, _indent.y + y, c), size);
                    }
                }
            }

            struct Specific
            {
                Vector paddedSrc, paddedDelta;
                Buffer buffer;
            };
            std::vector<Specific> _specific;

            Index _core;
            Index _padded;
            Size _indent;
            bool _valid;
            bool _partial;
            View _connection;

            typedef void(*FunctionForwardPtr)(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);
            FunctionForwardPtr _functionForward;

            typedef void(*FunctionBackwardPtr)(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);
            FunctionBackwardPtr _functionBackward;

            typedef void(*FunctionSumPtr)(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums);
            FunctionSumPtr _functionSum;
        };

        /*! @ingroup cpp_neural

            \short PoolingLayer class.

            Abstract class pooling layer (base for MaxPoolingLayer and AveragePoolingLayer) in neural network.
        */
        class PoolingLayer : public Layer
        {
        public:
            size_t FanSrc() const override
            {
                return _poolingSize.x*_poolingSize.y;
            }

            size_t FanDst() const override
            {
                return 1;
            }

        protected:
            PoolingLayer(Layer::Type t, Function::Type f, const Size & srcSize, size_t srcDepth, const Size & poolingSize, const Size & poolingStride, const Size & poolingPad)
                : Layer(t, f)
                , _poolingSize(poolingSize)
                , _poolingStride(poolingStride)
                , _poolingPad(poolingPad)
            {
                assert(t == Layer::MaxPooling || t == Layer::AveragePooling);

                _src.Resize(srcSize, srcDepth);
                Size dstSize = (srcSize - _poolingSize + 2 * _poolingStride + 2 * _poolingPad - Size(1, 1)) / _poolingStride;
                _dst.Resize(dstSize, srcDepth);
                SetThreadNumber(1, false);
            }

            Size _poolingSize;
            Size _poolingStride;
            Size _poolingPad;
        };

        /*! @ingroup cpp_neural

            \short MaxPoolingLayer class.

            Max pooling layer in neural network.
        */
        class MaxPoolingLayer : public PoolingLayer
        {
        public:
            /*!
                \short Creates new MaxPoolingLayer class.

                \param [in] f - a type of activation function used in this layer.
                \param [in] srcSize - a size (width and height) of input image.
                \param [in] srcDepth - a number of input channels (images).
                \param [in] poolingSize - a pooling size.
                \param [in] poolingStride - a pooling stride.
                \param [in] poolingPad - a pooling pad. By default it is equal to (0, 0).
            */
            MaxPoolingLayer(Function::Type f, const Size & srcSize, size_t srcDepth, const Size & poolingSize, const Size & poolingStride, const Size & poolingPad = Size())
                : PoolingLayer(Layer::MaxPooling, f, srcSize, srcDepth, poolingSize, poolingStride, poolingPad)
                , _functionForward(0)
            {
                if (_poolingSize == Size(2, 2) && _poolingStride == Size(2, 2) && _poolingPad == Size(0, 0))
                    _functionForward = ::SimdNeuralPooling2x2Max2x2;
                if (_poolingSize == Size(3, 3) && _poolingStride == Size(2, 2) && _poolingPad == Size(0, 0))
                    _functionForward = ::SimdNeuralPooling2x2Max3x3;
            }

            void Forward(const Vector & src, size_t thread, Method method) override
            {
                Vector & sum = _common[thread].sum;
                Vector & dst = _common[thread].dst;
                if (method != Layer::Train && _functionForward)
                {
                    for (ptrdiff_t c = 0; c < _dst.depth; ++c)
                        _functionForward(_src.Get(src, 0, 0, c), _src.width, _src.width, _src.height, _dst.Get(sum, 0, 0, c), _dst.width);
                }
                else
                {
                    ptrdiff_t * idx = _specific[thread].index.data();
                    for (ptrdiff_t c = 0; c < _dst.depth; ++c)
                    {
                        for (ptrdiff_t y = 0; y < _dst.height; y++)
                        {
                            ptrdiff_t dyStart = y*_poolingStride.y - _poolingPad.y;
                            ptrdiff_t dyEnd = std::min(dyStart + _poolingSize.y, _src.height);
                            dyStart = std::max(ptrdiff_t(0), dyStart);
                            for (ptrdiff_t x = 0; x < _dst.width; x++)
                            {
                                ptrdiff_t dxStart = x*_poolingStride.x - _poolingPad.x;
                                ptrdiff_t dxEnd = std::min(dxStart + _poolingSize.x, _src.width);
                                dxStart = std::max(ptrdiff_t(0), dxStart);
                                ptrdiff_t maxIndex = _src.Offset(dxStart, dyStart, c);
                                float maxValue = std::numeric_limits<float>::lowest();
                                for (ptrdiff_t dy = dyStart; dy < dyEnd; dy++)
                                {
                                    for (ptrdiff_t dx = dxStart; dx < dxEnd; dx++)
                                    {
                                        ptrdiff_t index = _src.Offset(dx, dy, c);
                                        float value = src[index];
                                        if (value > maxValue)
                                        {
                                            maxValue = value;
                                            maxIndex = index;
                                        }
                                    }
                                }
                                ptrdiff_t dstOffset = _dst.Offset(x, y, c);
                                sum[dstOffset] = maxValue;
                                idx[dstOffset] = maxIndex;
                                assert(idx[dstOffset] < (int)_common[thread].prevDelta.size());
                            }
                        }
                    }
                }
                _function.function(sum.data(), sum.size(), dst.data());
            }

            void Backward(const Vector & currDelta, size_t thread) override
            {
                const Vector & prevDst = _prev->Dst(thread);
                Vector & prevDelta = _common[thread].prevDelta;
                const VectorI & index = _specific[thread].index;

                Detail::SetZero(prevDelta);

                for (size_t i = 0; i < currDelta.size(); ++i)
                    prevDelta[index[i]] = currDelta[i];

                _prev->_function.derivative(&prevDst[0], prevDst.size(), &prevDelta[0]);
            }

            virtual void SetThreadNumber(size_t number, bool train) override
            {
                Layer::SetThreadNumber(number, train);
                if (train || _functionForward == 0)
                {
                    _specific.resize(number);
                    for (size_t i = 0; i < _specific.size(); ++i)
                    {
                        _specific[i].index.resize(_dst.Volume());
                    }
                }
            }

        protected:

            struct Specific
            {
                std::vector<ptrdiff_t, Allocator<ptrdiff_t>> index;
            };
            std::vector<Specific> _specific;

            typedef void(*FunctionForwardPtr)(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride);
            FunctionForwardPtr _functionForward;
        };

        /*! @ingroup cpp_neural

            \short AveragePoolingLayer class.

            Average pooling layer in neural network.
        */
        class AveragePoolingLayer : public PoolingLayer
        {
        public:
            /*!
                \short Creates new AveragePoolingLayer class.

                \param [in] f - a type of activation function used in this layer.
                \param [in] srcSize - a size (width and height) of input image.
                \param [in] srcDepth - a number of input channels (images).
                \param [in] poolingSize - a pooling size.
                \param [in] poolingStride - a pooling stride.
                \param [in] poolingPad - a pooling pad. By default it is equal to (0, 0).
            */
            AveragePoolingLayer(Function::Type f, const Size & srcSize, size_t srcDepth, const Size & poolingSize, const Size & poolingStride, const Size & poolingPad = Size())
                : PoolingLayer(Layer::AveragePooling, f, srcSize, srcDepth, poolingSize, poolingStride, poolingPad)
            {
                _scaleFactor = 1.0f / float(_poolingSize.x*poolingSize.y);
            }

            void Forward(const Vector & src, size_t thread, Method method) override
            {
                Vector & sum = _common[thread].sum;
                Vector & dst = _common[thread].dst;
                for (ptrdiff_t c = 0; c < _dst.depth; ++c)
                {
                    for (ptrdiff_t y = 0; y < _dst.height; y++)
                    {
                        ptrdiff_t dyStart = std::max(ptrdiff_t(0), y - _poolingPad.y);
                        ptrdiff_t dyEnd = std::min(dyStart + _poolingSize.y, _src.height);
                        for (ptrdiff_t x = 0; x < _dst.width; x++)
                        {
                            ptrdiff_t dxStart = std::max(ptrdiff_t(0), y - _poolingPad.x);
                            ptrdiff_t dxEnd = std::min(dxStart + _poolingSize.x, _src.width);
                            const float * psrc = _src.Get(src, x*_poolingStride.x, y*_poolingStride.y, c);
                            float average = 0;
                            for (ptrdiff_t dy = dyStart; dy < dyEnd; dy++)
                                for (ptrdiff_t dx = dxStart; dx < dxEnd; dx++)
                                    average += psrc[dy*_src.width + dx];
                            _dst.Get(sum, x, y, c)[0] = average*_scaleFactor;
                        }
                    }
                }
                _function.function(sum.data(), sum.size(), dst.data());
            }

            void Backward(const Vector & currDelta, size_t thread) override
            {
                const Vector & prevDst = _prev->Dst(thread);
                Vector & prevDelta = _common[thread].prevDelta;
                for (ptrdiff_t c = 0; c < _dst.depth; ++c)
                {
                    for (ptrdiff_t y = 0; y < _dst.height; y++)
                    {
                        ptrdiff_t dyStart = std::max(ptrdiff_t(0), y - _poolingPad.y);
                        ptrdiff_t dyEnd = std::min(dyStart + _poolingSize.y, _src.height);
                        for (ptrdiff_t x = 0; x < _dst.width; x++)
                        {
                            ptrdiff_t dxStart = std::max(ptrdiff_t(0), y - _poolingPad.x);
                            ptrdiff_t dxEnd = std::min(dxStart + _poolingSize.x, _src.width);
                            float delta = _dst.Get(currDelta, x, y, c)[0] * _scaleFactor;
                            float * prev = _src.Get(prevDelta, x*_poolingStride.x, y*_poolingStride.y, c);
                            for (ptrdiff_t dy = dyStart; dy < dyEnd; dy++)
                                for (ptrdiff_t dx = dxStart; dx < dxEnd; dx++)
                                    prev[dy*_src.width + dx] = delta;
                        }
                    }
                }
                _prev->_function.derivative(&prevDst[0], prevDst.size(), &prevDelta[0]);
            }

        protected:
            float _scaleFactor;
        };

        /*! @ingroup cpp_neural

            \short FullyConnectedLayer class.

            Fully connected layer in neural network.
        */
        class FullyConnectedLayer : public Layer
        {
        public:
            /*!
                \short Creates new FullyConnectedLayer class.

                \param [in] f - a type of activation function used in this layer.
                \param [in] srcSize - a size of input vector.
                \param [in] dstSize - a size of output vector.
                \param [in] bias - a boolean flag (enabling of bias). By default it is True.
            */
            FullyConnectedLayer(Function::Type f, size_t srcSize, size_t dstSize, bool bias = true)
                : Layer(FullyConnected, f)
                , _reordered(false)
            {
                _src.Resize(srcSize, 1, 1);
                _dst.Resize(dstSize, 1, 1);
                _weight.resize(dstSize*srcSize);
                if (bias)
                    _bias.resize(dstSize);
                SetThreadNumber(1, false);
            }

            virtual ~FullyConnectedLayer()
            {
            }

            void Forward(const Vector & src, size_t thread, Method method) override
            {
                Vector & sum = _common[thread].sum;
                Vector & dst = _common[thread].dst;

                if (method == Layer::Fast)
                {
                    if (!_reordered)
                    {
                        std::lock_guard<std::mutex> lock(_mutex);
                        if (!_reordered)
                        {
                            Vector buffer(_weight.size());
                            for (ptrdiff_t i = 0; i < _dst.width; ++i)
                                for (ptrdiff_t j = 0; j < _src.width; ++j)
                                    buffer[i*_src.width + j] = _weight[j*_dst.width + i];
                            _weight.swap(buffer);
                            _reordered = true;  
                        }
                    }
#if !defined(SIMD_SYNET_DISABLE)
                    SimdSynetInnerProductLayerForward(src.data(), _weight.data(), _bias.size() ? _bias.data() : NULL, _dst.width, _src.width, sum.data());
#else
                    for (size_t i = 0; i < sum.size(); ++i)
                        ::SimdNeuralProductSum(src.data(), &_weight[i * _src.width], src.size(), &sum[i]);
                    if (_bias.size())
                        ::SimdNeuralAddVector(_bias.data(), sum.size(), sum.data());
#endif                
                }
                else
                {
                    assert(!_reordered);
                    Detail::SetZero(sum);
                    for (size_t i = 0; i < src.size(); i++)
                        ::SimdNeuralAddVectorMultipliedByValue(&_weight[i*_dst.width], sum.size(), &src[i], sum.data());
                    if (_bias.size())
                        ::SimdNeuralAddVector(_bias.data(), sum.size(), sum.data());
                }

                _function.function(sum.data(), sum.size(), dst.data());
            }

            void Backward(const Vector & currDelta, size_t thread) override
            {
                const Vector & prevDst = _prev->Dst(thread);
                Vector & prevDelta = _common[thread].prevDelta;
                Vector & dWeight = _common[thread].dWeight;
                Vector & dBias = _common[thread].dBias;

                for (ptrdiff_t i = 0; i < _src.width; i++)
                    ::SimdNeuralProductSum(&currDelta[0], &_weight[i*_dst.width], _dst.width, &prevDelta[i]);

                _prev->_function.derivative(&prevDst[0], prevDst.size(), &prevDelta[0]);

                for (ptrdiff_t i = 0; i < _src.width; i++)
                    ::SimdNeuralAddVectorMultipliedByValue(&currDelta[0], _dst.width, &prevDst[i], &dWeight[i*_dst.width]);

                if (_bias.size())
                    ::SimdNeuralAddVector(currDelta.data(), _dst.width, dBias.data());
            }

            size_t FanSrc() const override
            {
                return _src.width;
            }

            size_t FanDst() const override
            {
                return _dst.width;
            }

        protected:
            bool _reordered;
            std::mutex _mutex;
        };

        /*! @ingroup cpp_neural

            \short DroputLayer class.

            Dropout layer in neural network.
        */
        class DropoutLayer : public Layer
        {
            static size_t SIMD_INLINE RandomSize() { return 256; }
        public:
            /*!
            \short Creates new DropoutLayer class.

            \param [in] srcSize - a size of input vector.
            \param [in] rate - a retention probability (dropout rate is 1 - rate).
            */
            DropoutLayer(size_t srcSize, float rate)
                : Layer(Dropout, Function::Identity)
                , _rate(rate)
                , _scale(1.0f / rate)
            {
                _src.Resize(srcSize, 1, 1);
                _dst.Resize(srcSize, 1, 1);
                SetThreadNumber(1, false);
            }

            void Forward(const Vector & src, size_t thread, Method method) override
            {
                Vector & dst = _common[thread].dst;
                if (method == Layer::Train)
                {
                    _specific[thread].mask = Mask();
                    const float * mask = _specific[thread].mask;

                    for (size_t i = 0; i < src.size(); ++i)
                        dst[i] = mask[i] * _scale * src[i];
                }
                else
                    dst = src;
            }

            void Backward(const Vector & currDelta, size_t thread) override
            {
                const Vector & prevDst = _prev->Dst(thread);
                Vector & prevDelta = _common[thread].prevDelta;
                const float * mask = _specific[thread].mask;
                for (size_t i = 0; i < currDelta.size(); i++)
                {
                    prevDelta[i] = mask[i] * currDelta[i];
                }

                _prev->_function.derivative(&prevDst[0], prevDst.size(), &prevDelta[0]);
            }

            size_t FanSrc() const override
            {
                return 1;
            }

            size_t FanDst() const override
            {
                return 1;
            }

            virtual void SetThreadNumber(size_t number, bool train) override
            {
                Layer::SetThreadNumber(number, train);
                _specific.resize(number);
                if (train)
                {
                    _mask.resize(_src.Volume()*(1 + RandomSize()));
                    for (size_t i = 0; i < _mask.size(); ++i)
                        _mask[i] = Detail::RandomUniform(0.0f, 1.0f) <= _rate ? 1.0f : 0.0f;
                }
            }

        protected:
            float _rate, _scale;
            Vector _mask;

            struct Specific
            {
                const float * mask;
            };
            std::vector<Specific> _specific;

            const float * Mask()
            {
                size_t start = Detail::RandomUniform(0, int(RandomSize()*_src.Volume()));
                return _mask.data() + start;
            }
        };

        /*! @ingroup cpp_neural

            \short Contains a set of training options.
        */
        struct TrainOptions
        {
            /*!
                \enum InitType

                Describes method to initialize weights of neural network.
            */
            enum InitType
            {
                /*!
                     Use fan-in and fan-out for scaling
                     Xavier Glorot, Yoshua Bengio.
                     "Understanding the difficulty of training deep feedforward neural networks"
                     Proc. AISTATS 10, May 2010, vol.9, pp249-256
                */
                Xavier,
            };

            /*!
                \enum LossType

                Describes loss function.
            */
            enum LossType
            {
                /*!
                    Mean-Squared-Error loss function for regression.
                */
                Mse,
                /*!
                    Cross-entropy loss function for (multiple independent) binary classifications.
                */
                CrossEntropy,
                /*!
                    Cross-entropy loss function for multi-class classification.
                */
                CrossEntropyMulticlass
            };

            /*!
                \enum UpdateType

                Method of weights' updating.
            */
            enum UpdateType
            {
                /*!
                    Adaptive gradients method.
                    J Duchi, E Hazan and Y Singer,
                    "Adaptive subgradient methods for online learning and stochastic optimization"
                    The Journal of Machine Learning Research, pages 2121-2159, 2011.

                    \note See ::SimdNeuralAdaptiveGradientUpdate.
                */
                AdaptiveGradient,
            };

            InitType initType; /*!< \brief Method to initialize weights. */
            LossType lossType; /*!< \brief Loss function type. */
            UpdateType updateType; /*!< \brief Weights' update type. */
            mutable size_t threadNumber; /*!< \brief Number of threads used to train. Use -1 to auto detect thread number.  */
            size_t epochStart; /*!< \brief Start epoch. It is used to continue training process. */
            size_t epochFinish; /*!< \brief Finish epoch. Describes total epoch number. */
            size_t batchSize; /*!< \brief A batch size. */
            float alpha; /*!< \brief Describes training speed. */
            float epsilon; /*!< \brief Used to prevent division by zero. */
            bool shuffle; /*!< \brief A flag to shuffle training set. */

            /*!
                \short Default constructor.
            */
            TrainOptions()
                : initType(Xavier)
                , lossType(Mse)
                , updateType(AdaptiveGradient)
                , threadNumber(std::thread::hardware_concurrency())
                , epochStart(0)
                , epochFinish(100)
                , batchSize(64)
                , alpha(0.01f)
                , epsilon(0.0001f)
                , shuffle(true)
            {
            }
        };

        namespace Detail
        {
            template<TrainOptions::InitType type> void InitWeight(Vector & dst, const Layer & layer);

            template<> SIMD_INLINE void InitWeight<TrainOptions::Xavier>(Vector & dst, const Layer & layer)
            {
                float halfRange = (float)(std::sqrt(6.0 / (layer.FanSrc() + layer.FanDst())));
                for (size_t i = 0; i < dst.size(); ++i)
                    dst[i] = RandomUniform(-halfRange, halfRange);
            }

            template<TrainOptions::LossType type> void Gradient(const Vector & current, const Vector & control, Vector & delta);

            template<> SIMD_INLINE void Gradient<TrainOptions::Mse>(const Vector & current, const Vector & control, Vector & delta)
            {
                for (size_t i = 0; i < current.size(); ++i)
                    delta[i] = current[i] - control[i];
            }

            template<> SIMD_INLINE void Gradient<TrainOptions::CrossEntropy>(const Vector & current, const Vector & control, Vector & delta)
            {
                for (size_t i = 0; i < current.size(); ++i)
                    delta[i] = (current[i] - control[i]) / (current[i] * (1.0f - current[i]));
            }

            template<> SIMD_INLINE void Gradient<TrainOptions::CrossEntropyMulticlass>(const Vector & current, const Vector & control, Vector & delta)
            {
                for (size_t i = 0; i < current.size(); ++i)
                    delta[i] = -control[i] / current[i];
            }

            template<TrainOptions::UpdateType type> void UpdateWeight(const TrainOptions & o, const Vector & d, Vector & g, Vector & v);

            template<> SIMD_INLINE void UpdateWeight<TrainOptions::AdaptiveGradient>(const TrainOptions & o, const Vector & d, Vector & g, Vector & v)
            {
                ::SimdNeuralAdaptiveGradientUpdate(d.data(), d.size(), o.batchSize, &o.alpha, &o.epsilon, g.data(), v.data());
            }
        }

        /*! @ingroup cpp_neural

            \short Network class.

            Class Network provides functionality for construction, loading, saving, prediction and training of convolutional neural network.
        */
        class Network
        {
        public:
            /*!
                \short Creates a new object of Network class.

                Creates empty network without any layers.
            */
            Network()
            {
            }

            /*!
                \short Clears all layers of the neural network.
            */
            void Clear()
            {
                _layers.clear();
            }

            /*!
                \short Returns true if the neural network is empty.
            */
            bool Empty() const
            {
                return _layers.empty();
            }

            /*!
                \short Adds new Layer to the neural network.

                \param [in] layer - a pointer to the new layer. You can add ConvolutionalLayer, MaxPoolingLayer and FullyConnectedLayer.
                \return a result of addition. If added layer is not compatible with previous layer the result might be negative.
            */
            bool Add(Layer * layer)
            {
                if (_layers.empty())
                    _layers.push_back(LayerPtr(new InputLayer(*layer)));
                if (layer->Link(_layers.back().get()))
                {
                    _layers.push_back(LayerPtr(layer));
                    return true;
                }
                else
                    return false;
            }

            /*!
                \short Gets dimensions of input data.

                \return a dimensions of input data.
            */
            const Index & InputIndex() const
            {
                return _layers.front()->_dst;
            }

            /*!
                \short Gets dimensions of output data.

                \return a dimensions of output data.
            */
            const Index & OutputIndex() const
            {
                return _layers.back()->_dst;
            }

            /*!
                \short Trains the neural network.

                \param [in] src - a set of input training samples.
                \param [in] dst - a set of classification results.
                \param [in] options - an options of training process.
                \param [in] logger - a functor to log training process.
                \return a result of the training.
            */
            template <class Logger> bool Train(const Vectors & src, const Labels & dst, const TrainOptions & options, Logger logger)
            {
                if (src.size() != dst.size())
                    return false;

                Vectors converted;
                Convert(dst, converted);

                return Train(src, converted, options, logger);
            }

            /*!
                \short Trains the neural network.

                \param [in] src - a set of input training samples.
                \param [in] dst - a set of classification results.
                \param [in] options - an options of training process.
                \param [in] logger - a functor to log training process.
                \return a result of the training.
            */
            template <class Logger> bool Train(const Vectors & src, const Vectors & dst, const TrainOptions & options, Logger logger)
            {
                SIMD_CHECK_PERFORMANCE();

                if (src.size() != dst.size())
                    return false;

                options.threadNumber = std::max<size_t>(1, std::min<size_t>(options.threadNumber, std::thread::hardware_concurrency()));

                for (size_t i = 0; i < _layers.size(); ++i)
                    _layers[i]->SetThreadNumber(options.threadNumber, true);

                if (options.epochStart == 0)
                    InitWeight(options);

                Labels index(src.size());
                for (size_t i = 0; i < index.size(); ++i)
                    index[i] = i;
                if (options.shuffle)
                {
#ifdef SIMD_CPP_2017_ENABLE
                    std::random_device device;
                    std::minstd_rand generator(device());
                    std::shuffle(index.begin(), index.end(), generator);
#else
                    std::random_shuffle(index.begin(), index.end());
#endif
                }

                for (size_t epoch = options.epochStart; epoch < options.epochFinish; ++epoch)
                {
                    for (size_t i = 0; i < src.size(); i += options.batchSize)
                    {
                        Propagate(src, dst, index, i, std::min(i + options.batchSize, src.size()), options);
                        UpdateWeight(options);
                    }
                    logger();
                }

                return true;
            }

            /*!
                \short Sets thread number.
                
                \note Call this function if you want to call method Predict from dirrerent thread.

                \param [in] number - a number of threads.
                \param [in] train - a train process boolean flag. By default it is equal to False.
            */
            void SetThreadNumber(size_t number, bool train = false)
            {
                for (size_t i = 0; i < _layers.size(); ++i)
                    _layers[i]->SetThreadNumber(number, train);
            }

            /*!
                \short Classifies given sample.

                \param [in] x - an input sample.
                \param [in] thread - a work thread number. By default it is equal to 0.
                \param [in] method - a method of prediction. By default it is equal to Layer::Fast.
                \return a result of classification (vector with predicted probabilities).
            */
            SIMD_INLINE const Vector & Predict(const Vector & x, size_t thread = 0, Layer::Method method = Layer::Fast)
            {
                return Forward(x, thread, method);
            }

            /*!
                \short Loads the weights of neural network from an external buffer.

                \note The network has to be created previously with using of methods Clear/Add.

                \param [in] data - a pointer to the external buffer.
                \param [in] size - a size of the external buffer.
                \param [in] train - a boolean flag (True - if we need to load temporary training data, False - otherwise). By default it is equal to False.
                \return a result of loading.
            */
            bool Load(const void * data, size_t size, bool train = false)
            {
                if (Requred(train) > size)
                    return false;
                typedef  Vector::value_type Type;
                Type * ptr = (Type*)data;
                if (train)
                {
                    for (size_t i = 0; i < _layers.size(); ++i)
                        _layers[i]->SetThreadNumber(1, true);
                }
                for (size_t i = 0; i < _layers.size(); ++i)
                {
                    Layer & layer = *_layers[i];
                    memcpy(layer._weight.data(), ptr, layer._weight.size() * sizeof(Type));
                    ptr += layer._weight.size();
                    memcpy(layer._bias.data(), ptr, layer._bias.size() * sizeof(Type));
                    ptr += layer._bias.size();
                }
                if (train)
                {
                    for (size_t i = 0; i < _layers.size(); ++i)
                    {
                        Layer & layer = *_layers[i];
                        memcpy(layer._gWeight.data(), ptr, layer._gWeight.size() * sizeof(Type));
                        ptr += layer._gWeight.size();
                        memcpy(layer._gBias.data(), ptr, layer._gBias.size() * sizeof(Type));
                        ptr += layer._gBias.size();
                    }
                }
                return true;
            }

            /*!
                \short Loads the weights of neural network from file stream.

                \note The network has to be created previously with using of methods Clear/Add.

                \param [in] is - a input stream.
                \param [in] train - a boolean flag (True - if we need to load temporary training data, False - otherwise). By default it is equal to False.
                \return a result of loading.
            */
            bool Load(std::istream & is, bool train = false)
            {
                SIMD_CHECK_PERFORMANCE();

                if (train)
                {
                    for (size_t i = 0; i < _layers.size(); ++i)
                        _layers[i]->SetThreadNumber(1, true);
                }
                for (size_t i = 0; i < _layers.size(); ++i)
                {
                    Layer & layer = *_layers[i];
                    for (size_t j = 0; j < layer._weight.size(); ++j)
                        Load(is, layer._weight[j]);
                    for (size_t j = 0; j < layer._bias.size(); ++j)
                        Load(is, layer._bias[j]);
                }
                if (train)
                {
                    for (size_t i = 0; i < _layers.size(); ++i)
                    {
                        Layer & layer = *_layers[i];
                        for (size_t j = 0; j < layer._gWeight.size(); ++j)
                            Load(is, layer._gWeight[j]);
                        for (size_t j = 0; j < layer._gBias.size(); ++j)
                            Load(is, layer._gBias[j]);
                    }
                }
                return true;
            }

            /*!
                \short Loads the weights of neural network from file.

                \note The network has to be created previously with using of methods Clear/Add.

                \param [in] path - a path to input file.
                \param [in] train - a boolean flag (True - if we need to load temporary training data, False - otherwise). By default it is equal to False.
                \return a result of loading.
            */
            bool Load(const std::string & path, bool train = false)
            {
                std::ifstream ifs(path.c_str());
                if (ifs.is_open())
                {
                    bool result = Load(ifs, train);
                    ifs.close();
                    return result;
                }
                return false;
            }

            /*!
                \short Saves the weights of neural network into external buffer.

                \param [out] data - a pointer to the external buffer.
                \param [in, out] size - a pointer to the size of external buffer. Returns requred buffer size. 
                \param [in] train - a boolean flag (True - if we need to save temporary training data, False - otherwise). By default it is equal to False.
                \return a result of saving.
            */
            bool Save(void * data, size_t * size, bool train = false) const
            {
                typedef  Vector::value_type Type;
                Type * ptr = (Type*)data;
                size_t requred = Requred(train);
                if (requred > *size)
                {
                    *size = requred;
                    return false;
                }
                else
                    *size = requred;

                for (size_t i = 0; i < _layers.size(); ++i)
                {
                    const Layer & layer = *_layers[i];
                    memcpy(ptr, layer._weight.data(), layer._weight.size() * sizeof(Type));
                    ptr += layer._weight.size();
                    memcpy(ptr, layer._bias.data(), layer._bias.size() * sizeof(Type));
                    ptr += layer._bias.size();
                }
                if (train)
                {
                    for (size_t i = 0; i < _layers.size(); ++i)
                    {
                        const Layer & layer = *_layers[i];
                        memcpy(ptr, layer._gWeight.data(), layer._gWeight.size() * sizeof(Type));
                        ptr += layer._gWeight.size();
                        memcpy(ptr, layer._gBias.data(), layer._gBias.size() * sizeof(Type));
                        ptr += layer._gBias.size();
                    }
                }
                return true;
            }

            /*!
                \short Saves the weights of neural network to file stream.

                \param [out] os - a output stream.
                \param [in] train - a boolean flag (True - if we need to save temporary training data, False - otherwise). By default it is equal to False.
                \return a result of saving.
            */
            bool Save(std::ostream & os, bool train = false) const
            {
                for (size_t i = 0; i < _layers.size(); ++i)
                {
                    const Layer & layer = *_layers[i];
                    for (size_t j = 0; j < layer._weight.size(); ++j)
                        os << layer._weight[j] << " ";
                    for (size_t j = 0; j < layer._bias.size(); ++j)
                        os << layer._bias[j] << " ";
                }
                if (train)
                {
                    os << std::endl;
                    for (size_t i = 0; i < _layers.size(); ++i)
                    {
                        const Layer & layer = *_layers[i];
                        for (size_t j = 0; j < layer._gWeight.size(); ++j)
                            os << layer._gWeight[j] << " ";
                        for (size_t j = 0; j < layer._gBias.size(); ++j)
                            os << layer._gBias[j] << " ";
                    }
                }
                return true;
            }

            /*!
                \short Saves the weights of neural network to file.

                \param [in] path - a path to output file.
                \param [in] train - a boolean flag (True - if we need to save temporary training data, False - otherwise). By default it is equal to False.
                \return a result of saving.
            */
            bool Save(const std::string & path, bool train = false) const
            {
                std::ofstream ofs(path.c_str());
                if (ofs.is_open())
                {
                    bool result = Save(ofs, train);
                    ofs.close();
                    return result;
                }
                return false;
            }

            /*!
                \short Converts format of classification results.

                \param [in] src - a set of class indexes.
                \param [out] dst - a set of vectors with predicted probabilities.
            */
            void Convert(const Labels & src, Vectors & dst) const
            {
                size_t size = _layers.back()->_dst.Volume();
                float min = _layers.back()->_function.min;
                float max = _layers.back()->_function.max;
                dst.resize(src.size());
                for (size_t i = 0; i < dst.size(); ++i)
                {
                    dst[i].resize(size, min);
                    if (src[i] < size)
                        dst[i][src[i]] = max;
                }
            }

        private:
            LayerPtrs _layers;

            size_t Requred(bool train) const
            {
                typedef Vector::value_type Type;
                size_t requred = 0;
                for (size_t i = 0; i < _layers.size(); ++i)
                {
                    const Layer & layer = *_layers[i];
                    requred += (layer._weight.size() + layer._bias.size()) * sizeof(Type);
                    if (train)
                        requred += (layer._gWeight.size() + layer._gBias.size()) * sizeof(Type);
                }
                return requred;
            }

            static SIMD_INLINE void Load(std::istream & is, float & value)
            {
                char buffer[64];
                is >> buffer;
                value = (float)::atof(buffer);
            }

            const Vector & Forward(const Vector & src, size_t thread, Layer::Method method)
            {
                SIMD_CHECK_PERFORMANCE();

                _layers.front()->Forward(src, thread, method);
                for (size_t i = 1; i < _layers.size(); ++i)
                    _layers[i]->Forward(_layers[i - 1]->Dst(thread), thread, method);
                return _layers.back()->Dst(thread);
            }

            bool Cannonical(const TrainOptions & options) const
            {
                const Function::Type & func = _layers.back()->_function.type;
                const TrainOptions::LossType & loss = options.lossType;
                if (loss == TrainOptions::Mse && func == Function::Identity)
                    return true;
                if (loss == TrainOptions::CrossEntropy && (func == Function::Sigmoid || func == Function::Tanh))
                    return true;
                if (loss == TrainOptions::CrossEntropyMulticlass && func == Function::Softmax)
                    return true;
                return false;
            }

            void Backward(const Vector & current, const Vector & control, size_t thread, const TrainOptions & options)
            {
                SIMD_CHECK_PERFORMANCE();

                Vector delta(current.size());
                if (Cannonical(options))
                {
                    for (size_t i = 0; i < current.size(); ++i)
                        delta[i] = current[i] - control[i];
                }
                else
                {
                    if (_layers.back()->_function.type == Function::Softmax)
                    {
                        Vector grad(current.size());
                        LossGradient(options, current, control, grad);
                        for (size_t i = 0; i < delta.size(); ++i)
                        {
                            float sum = grad[i] * current[i] * (1.0f - current[i]);
                            for (size_t j = 0; j < i; ++j)
                                sum -= grad[j] * current[j] * current[i];
                            for (size_t j = i + 1; j < grad.size(); ++j)
                                sum -= grad[j] * current[j] * current[i];
                            delta[i] = sum;
                        }
                    }
                    else
                    {
                        LossGradient(options, current, control, delta);
                        _layers.back()->_function.derivative(current.data(), current.size(), delta.data());
                    }
                }

                _layers.back()->Backward(delta, thread);
                for (ptrdiff_t i = _layers.size() - 2; i >= 0; --i)
                    _layers[i]->Backward(_layers[i + 1]->Delta(thread), thread);
            }

            void Propagate(const Vectors & src, const Vectors & dst, const Labels & index, size_t start, size_t finish, const TrainOptions & options)
            {
                SIMD_CHECK_PERFORMANCE();

                Parallel(start, finish, [&](size_t thread, size_t begin, size_t end)
                {
                    for (size_t i = begin; i < end; ++i)
                    {
                        Vector current = Forward(src[index[i]], thread, Layer::Train);
                        Backward(current, dst[index[i]], thread, options);
                    }
                }, options.threadNumber);
            }

            template<TrainOptions::InitType type> void InitWeight()
            {
                for (size_t l = 0; l < _layers.size(); ++l)
                {
                    Layer & layer = *_layers[l];
                    Detail::InitWeight<type>(layer._weight, layer);
                    Detail::InitWeight<type>(layer._bias, layer);
                    Detail::SetZero(layer._gWeight);
                    Detail::SetZero(layer._gBias);
                }
            }

            void InitWeight(const TrainOptions & options)
            {
                switch (options.initType)
                {
                case TrainOptions::Xavier: InitWeight<TrainOptions::Xavier>(); break;
                }
            }

            void LossGradient(const TrainOptions & options, const Vector & current, const Vector & control, Vector & delta)
            {
                switch (options.lossType)
                {
                case TrainOptions::Mse: Detail::Gradient<TrainOptions::Mse>(current, control, delta); break;
                case TrainOptions::CrossEntropy: Detail::Gradient<TrainOptions::CrossEntropy>(current, control, delta); break;
                case TrainOptions::CrossEntropyMulticlass: Detail::Gradient<TrainOptions::CrossEntropyMulticlass>(current, control, delta); break;
                }
            }

            template<TrainOptions::UpdateType type> void UpdateWeight(const TrainOptions & options)
            {
                for (size_t l = 0; l < _layers.size(); ++l)
                {
                    Layer & layer = *_layers[l];
                    for (size_t t = 1; t < layer._common.size(); ++t)
                    {
                        ::SimdNeuralAddVector(layer._common[t].dWeight.data(), layer._common[t].dWeight.size(), layer._common[0].dWeight.data());
                        ::SimdNeuralAddVector(layer._common[t].dBias.data(), layer._common[t].dBias.size(), layer._common[0].dBias.data());
                    }
                    Detail::UpdateWeight<type>(options, layer._common[0].dWeight, layer._gWeight, layer._weight);
                    Detail::UpdateWeight<type>(options, layer._common[0].dBias, layer._gBias, layer._bias);
                    for (size_t t = 0; t < layer._common.size(); ++t)
                    {
                        Detail::SetZero(layer._common[t].dWeight);
                        Detail::SetZero(layer._common[t].dBias);
                    }
                }
            }

            void UpdateWeight(const TrainOptions & options)
            {
                SIMD_CHECK_PERFORMANCE();

                switch (options.updateType)
                {
                case TrainOptions::AdaptiveGradient: UpdateWeight<TrainOptions::AdaptiveGradient>(options); break;
                }
            }
        };
    }
}

#endif//__SimdNeural_hpp__
