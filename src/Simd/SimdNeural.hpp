/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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

namespace Simd
{
    namespace Neural
    {
        typedef Point<ptrdiff_t> Size;
        typedef std::vector< float, Allocator<float> > Vector;
        typedef std::vector<Vector> Vectors;
        typedef size_t Label;
        typedef std::vector<Label> Labels;

        struct Function
        {
            enum Type
            {
                Identity,
                Tanh,
                Sigmoid,
                Relu,
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
                    break;
                case Sigmoid:
                    function = SigmoidFunction;
                    derivative = SigmoidDerivative;
                    min = -0.8f;
                    max = 0.8f;
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
                    memcpy(dst, src, sizeof(float_t)*size);
            }

            static SIMD_INLINE void IdentityDerivative(const float * src, size_t size, float * dst)
            {
                for (size_t i = 0; i < size; ++i)
                    dst[i] = 1.0f;
            }

            static SIMD_INLINE void TanhFunction(const float * src, size_t size, float * dst)
            {
                const float slope = 1.0f;
                ::SimdAnnRoughTanh(src, size, &slope, dst);
            }

            static SIMD_INLINE void TanhDerivative(const float * src, size_t size, float * dst)
            {
                const float slope = 1.0f;
                ::SimdAnnDerivativeTanh(src, size, &slope, dst);
            }

            static SIMD_INLINE void SigmoidFunction(const float * src, size_t size, float * dst)
            {
                const float slope = 1.0f;
                ::SimdAnnRoughSigmoid(src, size, &slope, dst);
            }

            static SIMD_INLINE void SigmoidDerivative(const float * src, size_t size, float * dst)
            {
                const float slope = 1.0f;
                ::SimdAnnDerivativeSigmoid(src, size, &slope, dst);
            }

            static SIMD_INLINE void ReluFunction(const float * src, size_t size, float * dst)
            {
                const float slope = 0.0f;
                ::SimdAnnRelu(src, size, &slope, dst);
            }

            static SIMD_INLINE void ReluDerivative(const float * src, size_t size, float * dst)
            {
                const float slope = 0.0f;
                ::SimdAnnDerivativeRelu(src, size, &slope, dst);
            }

            static SIMD_INLINE void LeakyReluFunction(const float * src, size_t size, float * dst)
            {
                const float slope = 0.01f;
                ::SimdAnnRelu(src, size, &slope, dst);
            }

            static SIMD_INLINE void LeakyReluDerivative(const float * src, size_t size, float * dst)
            {
                const float slope = 0.01f;
                ::SimdAnnDerivativeRelu(src, size, &slope, dst);
            }
        };

        template <class T> struct Buffer
        {
            Size size;
            size_t count;
            std::vector<T, Allocator<T>> value;

            SIMD_INLINE Buffer(const Size & s = Size(), size_t c = 0)
            {
                Resize(s, c);
            }

            SIMD_INLINE void Resize(const Size & s, size_t c)
            {
                size = s;
                count = c;
                value.resize(size.x*size.y*count, 0);
            }

            SIMD_INLINE void Resize(size_t w, size_t h, size_t c)
            {
                Resize(Size(w, h), c);
            }

            SIMD_INLINE T * Get(size_t x, size_t y, size_t c)
            {
                return value.data() + (c*size.y + y)*size.x + x;
            }

            SIMD_INLINE const T * Get(size_t x, size_t y, size_t c) const
            {
                return value.data() + (c*size.y + y)*size.x + x;
            }
        };

        struct Layer
        {
            enum Type
            {
                Input,
                Convolutional,
                MaxPooling,
                FullyConnected,
            } const type;

            const Function function;

            virtual void Forward(const Layer & src, bool train) = 0;

            virtual void Backward(const Vector & src) = 0;

            Layer(Layer::Type l, Function::Type f)
                : type(l)
                , function(f)
            {
            }

        protected:
            Buffer<float> _src;
            Buffer<float> _weight;
            Buffer<float> _bias;
            Buffer<float> _sum;
            Buffer<float> _dst;

            friend struct InputLayer;
            friend struct ConvolutionalLayer;
            friend struct MaxPoolingLayer;
            friend struct FullyConnectedLayer;
            friend struct Network;
        };
        typedef std::shared_ptr<Layer> LayerPtr;
        typedef std::vector<LayerPtr> LayerPtrs;

        struct InputLayer : public Layer
        {
            InputLayer(const Size & dstSize)
                : Layer(Input, Function::Identity)
            {
                _dst.Resize(dstSize, 1);
            }

            void Forward(const Layer & src, bool train) override {}

            void Backward(const Vector & src) override {}
        };

        struct ConvolutionalLayer : public Layer
        {
            ConvolutionalLayer(Function::Type f, size_t srcCount, const Size & dstSize, size_t dstCount, size_t half, bool valid = true, bool bias = true)
                : Layer(Convolutional, f)
            {
                _half = half;
                _valid = valid;
                _src.Resize(Size(dstSize.x + 2 * _half, dstSize.y + 2 * _half), srcCount);
                _weight.Resize(Size(1 + 2 * _half, 1 + 2 * _half), srcCount*dstCount);
                if (bias)
                    _bias.Resize(Size(1, 1), dstCount);
                _sum.Resize(dstSize, dstCount);
                _dst.Resize(dstSize, dstCount);
            }

            void Forward(const Layer & src, bool train) override
            {
                CopyAndPad(src);
                memset(_sum.value.data(), 0, _sum.value.size()*sizeof(float));
                for (size_t dc = 0; dc < _dst.count; ++dc)
                {
                    for (size_t sc = 0; sc < _src.count; ++sc)
                    {
                        //if (!tbl_.is_connected(o, inc))
                        //    continue;

                        const float_t * weight = _weight.Get(0, 0, _src.count*dc + sc);
                        const float_t * src = _src.Get(0, 0, sc);
                        float_t * dst = _sum.Get(0, 0, dc);

                        if (_weight.size.x == 3 && _weight.size.y == 3)
                        {
                            ::SimdAnnAddConvolution3x3(src, _src.size.x, _dst.size.x, _dst.size.y, weight, dst, _dst.size.x);
                        }
                        else if (_weight.size.y == 5 && _weight.size.y == 5)
                        {
                            ::SimdAnnAddConvolution5x5(src, _src.size.x, _dst.size.x, _dst.size.y, weight, dst, _dst.size.x);
                        }
                        else
                        {
                            for (ptrdiff_t y = 0; y < _sum.size.y; y++)
                            {
                                for (ptrdiff_t x = 0; x < _sum.size.x; x++)
                                {
                                    const float * pw = weight;
                                    const float * ps = src + y * _src.size.x + x;
                                    float sum = 0;
                                    for (ptrdiff_t wy = 0; wy < _weight.size.y; wy++)
                                        for (ptrdiff_t wx = 0; wx < _weight.size.x; wx++)
                                            sum += *pw++ * ps[wy * _src.size.x + wx];
                                    dst[y * _dst.size.x + x] += sum;
                                }
                            }
                        }
                    }

                    if (!_bias.value.empty())
                    {
                        float_t bias = *_bias.Get(0, 0, dc);
                        size_t size = _sum.size.x*_sum.size.y;
                        float_t * sum = _sum.Get(0, 0, dc);
                        for (size_t i = 0; i < size; ++i)
                            sum[i] += bias;
                    }
                }
                function.function(_sum.value.data(), _sum.value.size(), _dst.value.data());
            }

            void Backward(const Vector & src) override
            {

            }

        protected:

            void CopyAndPad(const Layer & src)
            {
                size_t half = _valid ? 0 : _half;
                size_t size = src._dst.size.x*sizeof(float);
                for (size_t c = 0; c < src._dst.count; ++c)
                {
                    for (ptrdiff_t y = 0; y < _dst.size.y; ++y)
                        memcpy(_src.Get(half, half + y, c), src._dst.Get(0, y, c), size);
                }
            }

            size_t _half;
            bool _valid;
        };

        struct MaxPoolingLayer : public Layer
        {
            MaxPoolingLayer(Function::Type f, const Size & srcSize, size_t srcCount, size_t poolingSize)
                : Layer(MaxPooling, f)
            {
                _poolingSize = poolingSize;
                _src.Resize(srcSize, srcCount);
                _sum.Resize(srcSize/_poolingSize, srcCount);
                _dst.Resize(srcSize/_poolingSize, srcCount);

                _index.Resize(srcSize/_poolingSize, srcCount);
            }

            void Forward(const Layer & src, bool train) override
            {
                if (train || _poolingSize != 2)
                {
                    _src.value = src._dst.value;
                    for (size_t c = 0; c < _dst.count; ++c)
                    {
                        for (ptrdiff_t y = 0; y < _sum.size.y; y++)
                        {
                            for (ptrdiff_t x = 0; x < _sum.size.x; x++)
                            {
                                const float * ps = _src.Get(x*_poolingSize, y*_poolingSize, c);
                                ptrdiff_t maxIndex = 0;
                                float maxValue = std::numeric_limits<float_t>::lowest();
                                for (size_t dy = 0; dy < _poolingSize; dy++)
                                {
                                    for (size_t dx = 0; dx < _poolingSize; dx++)
                                    {
                                        ptrdiff_t index = dy*_src.size.x + dx;
                                        float value = ps[index];
                                        if (value > maxValue)
                                        {
                                            maxValue = value;
                                            maxIndex = index;
                                        }
                                    }
                                }
                                _sum.Get(x, y, c)[0] = maxValue;
                                _index.Get(x, y, c)[0] = maxIndex + (ps - _src.value.data());
                            }
                        }
                    }                    
                }
                else
                {
                    const Buffer<float> & s = src._dst;
                    ::SimdAnnMax2x2(s.value.data(), s.size.x, s.size.x, s.size.x*s.count, _sum.value.data(), _sum.size.x);
                }
                function.function(_sum.value.data(), _sum.value.size(), _dst.value.data());
            }

            void Backward(const Vector & src) override
            {

            }
        protected:
            Buffer<ptrdiff_t> _index;
            size_t _poolingSize;
        };

        struct FullyConnectedLayer : public Layer
        {
            FullyConnectedLayer(Function::Type f, size_t srcCount, size_t dstCount, bool bias = true)
                : Layer(FullyConnected, f)
                , _reordered(false)
            {
                Size size(1, 1);
                _src.Resize(size, srcCount);
                _weight.Resize(dstCount, srcCount, 1);
                if (bias)
                    _bias.Resize(Size(1, 1), dstCount);
                _sum.Resize(size, dstCount);
                _dst.Resize(size, dstCount);
            }

            void Forward(const Layer & src, bool train) override
            {
                if (train || true)
                {
                    _src.value = src._dst.value;
                    memset(_sum.value.data(), 0, sizeof(float_t)*_sum.value.size());
                    for (size_t i = 0; i < _src.value.size(); i++)
                        ::SimdAnnAddVectorMultipliedByValue(_weight.Get(0, i, 0), _sum.value.size(), _src.Get(0, 0, i), _sum.Get(0, 0, 0));
                }
                else
                {
                    if (!_reordered)
                    {
                        Vector buffer(_weight.value.size());
                        for (ptrdiff_t i = 0; i < _weight.size.x; ++i)
                            for (ptrdiff_t j = 0; j < _weight.size.y; ++j)
                                buffer[i*_weight.size.y + j] = _weight.value[j*_weight.size.x + i];
                        _weight.value.swap(buffer);
                        std::swap(_weight.size.x, _weight.size.y);
                        _reordered = true;
                    }
                    const Buffer<float> & s = src._dst;
                    for (size_t i = 0; i < _sum.value.size(); ++i)
                        ::SimdAnnProductSum(s.value.data(), _weight.Get(0, i, 0), s.value.size(), _sum.Get(0, 0, i));
                }

                if (!_bias.value.empty())
                {
                    for (size_t i = 0; i < _sum.value.size(); ++i)
                        _sum.value[i] += _bias.value[i];
                }
                function.function(_sum.value.data(), _sum.value.size(), _dst.value.data());
            }

            void Backward(const Vector & src) override
            {

            }
        protected:
            bool _reordered;
        };

        struct TrainOptions
        {

        };
        
        struct Network
        {
            void Clear()
            {
                _layers.clear();
            }

            void Add(Layer * layer)
            {
                if (_layers.empty())
                    _layers.push_back(LayerPtr(new InputLayer(layer->_dst.size)));
                _layers.push_back(LayerPtr(layer));
            }

            bool Train(const Vectors & src, const Labels & dst, TrainOptions & options)
            {
                if (src.size() != dst.size())
                    return false;
                size_t size = _layers.back()->_dst.value.size();
                float min = _layers.back()->function.min;
                float max = _layers.back()->function.max;
                Vectors converted(dst.size());
                for (size_t i = 0; i < dst.size(); ++i)
                {
                    converted[i].resize(size, min);
                    if (dst[i] < size)
                        converted[i][dst[i]] = max;
                }
                return Train(src, converted, options);
            }

            bool Train(const Vectors & src, const Vectors & dst, TrainOptions & options)
            {
                if (src.size() != dst.size())
                    return false;

                return false;
            }

            Vector & Predict(const Vector & x, bool train = false)
            {
                Forward(x, train);
                return _layers.back()->_dst.value;
            }

            bool Load(std::ifstream & ifs, bool train = false)
            {
                for (size_t i = 0; i < _layers.size(); ++i)
                {
                    Layer & layer = *_layers[i];
                    for (size_t j = 0; j < layer._weight.value.size(); ++j)
                        ifs >> layer._weight.value[j];
                    for (size_t j = 0; j < layer._bias.value.size(); ++j)
                        ifs >> layer._bias.value[j];
                }
                return true;
            }

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

            bool Save(std::ofstream & ofs, bool train = false) const
            {
                for (size_t i = 0; i < _layers.size(); ++i)
                {
                    const Layer & layer = *_layers[i];
                    for (size_t j = 0; j < layer._weight.value.size(); ++j)
                        ofs << layer._weight.value[j] << " ";
                    for (size_t j = 0; j < layer._bias.value.size(); ++j)
                        ofs << layer._bias.value[j] << " ";
                }
                return true;
            }

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

        private:
            LayerPtrs _layers;

            void Forward(const Vector & src, bool train)
            {
                _layers[0]->_dst.value = src;
                for (size_t i = 1; i < _layers.size(); ++i)
                    _layers[i]->Forward(*_layers[i - 1], train);
            }

            void Backward(const Vector & src)
            {

            }
        };
    }
}

#endif//__SimdNeural_hpp__
