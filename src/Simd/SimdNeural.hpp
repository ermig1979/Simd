/*
* Simd Library (http://simd.sourceforge.net).
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
#ifndef __SimdNeural_hpp__
#define __SimdNeural_hpp__

#include "Simd/SimdLib.hpp"
#include "Simd/SimdParallel.hpp"

#include <numeric>
#include <random>

#ifndef SIMD_CHECK_PERFORMANCE
#define SIMD_CHECK_PERFORMANCE()
#endif

namespace Simd
{
    /*! @ingroup cpp_neural

        \short Contains Framework for learning of Convolutional Neural Network.
    */
    namespace Neural
    {
        typedef Point<ptrdiff_t> Size; /*!< \brief 2D-size (width and height). */
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
                memset(vector.data(), 0, vector.size()*sizeof(T));
            }

            SIMD_INLINE void AddTo(const Vector & src, Vector & dst)
            {
                const float one = 1;
                ::SimdNeuralAddVectorMultipliedByValue(src.data(), src.size(), &one, dst.data());
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
                    See implementation details: ::SimdNeuralRelu and ::SimdNeuralDerivativeRelu.
                */
                Relu,
                /*! Leaky ReLU(Rectified Linear Unit):
                    \verbatim
                    f(x) = x > 0 ? x : 0.01*x;
                    \endverbatim
                    \verbatim
                    df(y) = y > 0 ? 1 : 0.01;
                    \endverbatim
                    See implementation details: ::SimdNeuralRelu and ::SimdNeuralDerivativeRelu.
                */
                LeakyRelu,
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
                ::SimdNeuralRelu(src, size, &slope, dst);
            }

            static SIMD_INLINE void ReluDerivative(const float * src, size_t size, float * dst)
            {
                const float slope = 0.0f;
                ::SimdNeuralDerivativeRelu(src, size, &slope, dst);
            }

            static SIMD_INLINE void LeakyReluFunction(const float * src, size_t size, float * dst)
            {
                const float slope = 0.01f;
                ::SimdNeuralRelu(src, size, &slope, dst);
            }

            static SIMD_INLINE void LeakyReluDerivative(const float * src, size_t size, float * dst)
            {
                const float slope = 0.01f;
                ::SimdNeuralDerivativeRelu(src, size, &slope, dst);
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
            friend class MaxPoolingLayer;
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
            ConvolutionalLayer(Function::Type f, const Size & srcSize, size_t srcDepth, size_t dstDepth, size_t coreSize, 
                bool valid = true, bool bias = true, const View & connection = View())
                : Layer(Convolutional, f)
            {
                _valid = valid;
                _indent = coreSize/2;
                Size pad(coreSize - 1, coreSize - 1);
                _src.Resize(srcSize, srcDepth);
                _dst.Resize(srcSize - (_valid ? pad : Size()), dstDepth);
                _padded.Resize(srcSize + (_valid ? Size() : pad), srcDepth);
                _core.Resize(coreSize, coreSize, srcDepth*dstDepth);
                _weight.resize(_core.Volume());
                if (bias)
                    _bias.resize(dstDepth);
                SetThreadNumber(1, false);

                _connection.Recreate(dstDepth, srcDepth, View::Gray8);
                if (Simd::Compatible(connection, _connection))
                    Simd::Copy(connection, _connection);
                else
                    Simd::Fill(_connection, 1);
            }

            void Forward(const Vector & src, size_t thread, Method method) override
            {
                const Vector & padded = PaddedSrc(src, thread);
                Vector & sum = _common[thread].sum;
                Vector & dst = _common[thread].dst;
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

                        if (_core.width == 3 && _core.height == 3)
                        {
                            ::SimdNeuralAddConvolution3x3(psrc, _padded.width, _dst.width, _dst.height, pweight, psum, _dst.width);
                        }
                        else if (_core.width == 5 && _core.height == 5)
                        {
                            ::SimdNeuralAddConvolution5x5(psrc, _padded.width, _dst.width, _dst.height, pweight, psum, _dst.width);
                        }
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
                    if (_bias.size())
                    {
                        float bias = _bias[dc];
                        size_t size = _dst.Area();
                        float * psum = _dst.Get(sum, 0, 0, dc);
                        for (size_t i = 0; i < size; ++i)
                            psum[i] += bias;
                    }
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

                        if (_core.width == 3 && _core.height == 3)
                        {
                            ::SimdNeuralAddConvolution3x3Back(psrc, _dst.width, _dst.width, _dst.height, pweight, pdst, _padded.width);
                        }
                        else if (_core.width == 5 && _core.height == 5)
                        {
                            ::SimdNeuralAddConvolution5x5Back(psrc, _dst.width, _dst.width, _dst.height, pweight, pdst, _padded.width);
                        }
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

                        if (_core.width == 3 && _core.height == 3)
                        {
                            ::SimdNeuralAddConvolution3x3Sum(prevo, _padded.width, delta, _dst.width, _dst.width, _dst.height, sums);
                        }
                        else if (_core.width == 5 && _core.height == 5)
                        {
                            ::SimdNeuralAddConvolution5x5Sum(prevo, _padded.width, delta, _dst.width, _dst.width, _dst.height, sums);
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
                    size_t size = _src.width*sizeof(float);
                    for (ptrdiff_t c = 0; c < _src.depth; ++c)
                    {
                        for (ptrdiff_t y = 0; y < _src.height; ++y)
                            memcpy(_padded.Get(padded, _indent, _indent + y, c), _src.Get(src, 0, y, c), size);
                    }
                    return padded;
                }
            }

            void UnpadDelta(const Vector & src, size_t thread)
            {
                if (!_valid)
                {
                    Vector & dst = _common[thread].prevDelta;
                    size_t size = _src.width*sizeof(float);
                    for (ptrdiff_t c = 0; c < _src.depth; c++)
                    {
                        for (ptrdiff_t y = 0; y < _src.height; ++y)
                            memcpy(_src.Get(dst, 0, y, c), _padded.Get(src, _indent, _indent + y, c), size);
                    }
                }
            }

            struct Specific
            {
                Vector paddedSrc, paddedDelta;
            };
            std::vector<Specific> _specific;

            Index _core;
            Index _padded;
            size_t _indent;
            bool _valid;
            View _connection;
        };

        /*! @ingroup cpp_neural

            \short MaxPoolingLayer class.

            Max pooling layer in neural network.
        */
        class MaxPoolingLayer : public Layer
        {
        public:
            /*!
                \short Creates new MaxPoolingLayer class.

                \param [in] f - a type of activation function used in this layer.
                \param [in] srcSize - a size (width and height) of input image.
                \param [in] srcDepth - a number of input channels (images).
                \param [in] poolingSize - a pooling size.
            */
            MaxPoolingLayer(Function::Type f, const Size & srcSize, size_t srcDepth, size_t poolingSize)
                : Layer(MaxPooling, f)
            {
                _poolingSize = poolingSize;
                _src.Resize(srcSize, srcDepth);
                _dst.Resize(srcSize/_poolingSize, srcDepth);
                SetThreadNumber(1, false);
            }

            void Forward(const Vector & src, size_t thread, Method method) override
            {
                Vector & sum = _common[thread].sum;
                Vector & dst = _common[thread].dst;
                if (method != Layer::Train && _poolingSize == 2)
                {
                    ::SimdNeuralMax2x2(src.data(), _src.width, _src.width, _src.height*_src.depth, sum.data(), _dst.width);
                }
                else
                {
                    ptrdiff_t * idx = _specific[thread].index.data();
                    for (ptrdiff_t c = 0; c < _dst.depth; ++c)
                    {
                        for (ptrdiff_t y = 0; y < _dst.height; y++)
                        {
                            for (ptrdiff_t x = 0; x < _dst.width; x++)
                            {
                                ptrdiff_t srcOffset = _src.Offset(x*_poolingSize, y*_poolingSize, c);
                                const float * psrc = src.data() + srcOffset;
                                ptrdiff_t maxIndex = 0;
                                float maxValue = std::numeric_limits<float>::lowest();
                                for (size_t dy = 0; dy < _poolingSize; dy++)
                                {
                                    for (size_t dx = 0; dx < _poolingSize; dx++)
                                    {
                                        ptrdiff_t index = dy*_src.width + dx;
                                        float value = psrc[index];
                                        if (value > maxValue)
                                        {
                                            maxValue = value;
                                            maxIndex = index;
                                        }
                                    }
                                }
                                ptrdiff_t dstOffset = _dst.Offset(x, y, c);
                                sum[dstOffset] = maxValue;
                                idx[dstOffset] = srcOffset + maxIndex;
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

            size_t FanSrc() const override
            {
                return _poolingSize*_poolingSize;
            }

            size_t FanDst() const override
            {
                return 1;
            }

            virtual void SetThreadNumber(size_t number, bool train) override
            {
                Layer::SetThreadNumber(number, train);
                if (train || _poolingSize != 2)
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

            size_t _poolingSize;
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

            void Forward(const Vector & src, size_t thread, Method method) override
            {
                Vector & sum = _common[thread].sum;
                Vector & dst = _common[thread].dst;

                if (method == Layer::Fast)
                {
                    if (!_reordered)
                    {
                        Vector buffer(_weight.size());
                        for (ptrdiff_t i = 0; i < _dst.width; ++i)
                            for (ptrdiff_t j = 0; j < _src.width; ++j)
                                buffer[i*_src.width + j] = _weight[j*_dst.width + i];
                        _weight.swap(buffer);
                        _reordered = true;
                    }
                    for (size_t i = 0; i < sum.size(); ++i)
                        ::SimdNeuralProductSum(src.data(), &_weight[i*_src.width], src.size(), &sum[i]);
                }
                else
                {
                    assert(!_reordered);
                    Detail::SetZero(sum);
                    for (size_t i = 0; i < src.size(); i++)
                        ::SimdNeuralAddVectorMultipliedByValue(&_weight[i*_dst.width], sum.size(), &src[i], sum.data());
                }

                if (_bias.size())
                {
                    for (size_t i = 0; i < sum.size(); ++i)
                        sum[i] += _bias[i];
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
                {
                    for (ptrdiff_t i = 0; i < _dst.width; ++i)
                        dBias[i] += currDelta[i];
                }
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
        };

        /*! @ingroup cpp_neural

            \short DroputLayer class.

            Dropout layer in neural network.
        */
        class DropoutLayer : public Layer
        {
            const size_t RANDOM_SIZE = 256;
        public:
            /*!
            \short Creates new DropoutLayer class.

            \param [in] srcSize - a size of input vector.
            \param [in] rate - a retention probability (dropout rate is 1 - rate).
            */
            DropoutLayer(size_t srcSize, float rate)
                : Layer(Dropout, Function::Identity)
                , _rate(rate)
                , _scale(1.0f/rate)
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
                    _mask.resize(_src.Volume()*(1 + RANDOM_SIZE));
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
                size_t start = Detail::RandomUniform(0, int(RANDOM_SIZE*_src.Volume()));
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

            /*!
                \short Default constructor.
            */
            TrainOptions()
                : initType(Xavier)
                , lossType(Mse)
                , updateType(AdaptiveGradient)
                , threadNumber(std::thread::hardware_concurrency())
                , batchSize(64)
                , epochStart(0)
                , epochFinish(100)
                , alpha(0.01f)
                , epsilon(0.0001f)
            {
            }
        };

        namespace Detail
        {
            template<TrainOptions::InitType type> void InitWeight(Vector & dst, const Layer & layer);

            template<> void InitWeight<TrainOptions::Xavier>(Vector & dst, const Layer & layer)
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

                for (size_t epoch = options.epochStart; epoch < options.epochFinish; ++epoch)
                {
                    for (size_t i = 0; i < src.size(); i += options.batchSize)
                    {
                        Propagate(src, dst, i, std::min(i + options.batchSize, src.size()), options);
                        UpdateWeight(options);
                    }
                    logger();
                }

                return true;
            }

            /*!
                \short Classifies given sample.

                \param [in] x - an input sample.
                \param [in] method - a method of prediction. By default it is equal to Layer::Fast.
                \return a result of classification (vector with predicted probabilities).
            */
            SIMD_INLINE const Vector & Predict(const Vector & x, Layer::Method method = Layer::Fast)
            {
                return Forward(x, 0, method);
            }

            /*!
                \short Loads the neural network from file stream.

                \note The network has to be created previously with using of methods Clear/Add.

                \param [in] ifs - a input file stream.
                \param [in] train - a boolean flag (True - if we need to load temporary training data, False - otherwise). By default it is equal to False.
                \return a result of loading.
            */
            bool Load(std::ifstream & ifs, bool train = false)
            {
                if (train)
                {
                    for (size_t i = 0; i < _layers.size(); ++i)
                        _layers[i]->SetThreadNumber(1, true);
                }
                for (size_t i = 0; i < _layers.size(); ++i)
                {
                    Layer & layer = *_layers[i];
                    for (size_t j = 0; j < layer._weight.size(); ++j)
                        ifs >> layer._weight[j];
                    for (size_t j = 0; j < layer._bias.size(); ++j)
                        ifs >> layer._bias[j];
                }
                if (train)
                {
                    for (size_t i = 0; i < _layers.size(); ++i)
                    {
                        Layer & level = *_layers[i];
                        for (size_t j = 0; j < level._gWeight.size(); ++j)
                            ifs >> level._gWeight[j];
                        for (size_t j = 0; j < level._gBias.size(); ++j)
                            ifs >> level._gBias[j];
                    }
                }
                return true;
            }

            /*!
                \short Loads the neural network from file.

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
                \short Saves the neural network to file stream.

                \param [out] ofs - a output file stream.
                \param [in] train - a boolean flag (True - if we need to save temporary training data, False - otherwise). By default it is equal to False.
                \return a result of saving.
            */
            bool Save(std::ofstream & ofs, bool train = false) const
            {
                for (size_t i = 0; i < _layers.size(); ++i)
                {
                    const Layer & layer = *_layers[i];
                    for (size_t j = 0; j < layer._weight.size(); ++j)
                        ofs << layer._weight[j] << " ";
                    for (size_t j = 0; j < layer._bias.size(); ++j)
                        ofs << layer._bias[j] << " ";
                }
                if (train)
                {
                    ofs << std::endl;
                    for (size_t i = 0; i < _layers.size(); ++i)
                    {
                        const Layer & level = *_layers[i];
                        for (size_t j = 0; j < level._gWeight.size(); ++j)
                            ofs << level._gWeight[j] << " ";
                        for (size_t j = 0; j < level._gBias.size(); ++j)
                            ofs << level._gBias[j] << " ";
                    }
                }
                return true;
            }

            /*!
                \short Saves the neural network to file.

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

            const Vector & Forward(const Vector & src, size_t thread, Layer::Method method)
            {
                SIMD_CHECK_PERFORMANCE();

                _layers.front()->Forward(src, thread, method);
                for (size_t i = 1; i < _layers.size(); ++i)
                    _layers[i]->Forward(_layers[i - 1]->Dst(thread), thread, method);
                return _layers.back()->Dst(thread);
            }

            void Backward(const Vector & current, const Vector & control, size_t thread, const TrainOptions & options)
            {
                SIMD_CHECK_PERFORMANCE();

                Vector delta(current.size());
                LossGradient(options, current, control, delta);
                _layers.back()->_function.derivative(current.data(), current.size(), delta.data());

                _layers.back()->Backward(delta, thread);
                for (ptrdiff_t i = _layers.size() - 2; i >= 0; --i)
                    _layers[i]->Backward(_layers[i + 1]->Delta(thread), thread);
            }

            void Propagate(const Vectors & src, const Vectors & dst, size_t start, size_t finish, const TrainOptions & options)
            {
                SIMD_CHECK_PERFORMANCE();

                Parallel(start, finish, [&](size_t thread, size_t begin, size_t end)
                {
                    for (size_t i = begin; i < end; ++i)
                    {
                        Vector current = Forward(src[i], thread, Layer::Train);
                        Backward(current, dst[i], thread, options);
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
                }
            }

            template<TrainOptions::UpdateType type> void UpdateWeight(const TrainOptions & options)
            {
                for (size_t l = 0; l < _layers.size(); ++l)
                {
                    Layer & layer = *_layers[l];
                    for (size_t t = 1; t < layer._common.size(); ++t)
                    {
                        Detail::AddTo(layer._common[t].dWeight, layer._common[0].dWeight);
                        Detail::AddTo(layer._common[t].dBias, layer._common[0].dBias);
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
