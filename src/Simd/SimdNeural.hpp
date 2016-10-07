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
        typedef std::vector<float, Allocator<float>> Vector;
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

        struct Index 
        {
            ptrdiff_t width, height, depth;

            SIMD_INLINE Index()
                : width(0)
                , height(0)
                , depth(0)
            {
            }

            SIMD_INLINE Index(ptrdiff_t w, ptrdiff_t h, ptrdiff_t d = 1)
                : width(w)
                , height(h)
                , depth(d)
            {
            }

            SIMD_INLINE Index(const Size & s, ptrdiff_t d = 1)
                : width(s.x)
                , height(s.y)
                , depth(d)
            {
            }

            SIMD_INLINE Index(const Index & i)
                : width(i.width)
                , height(i.height)
                , depth(i.depth)
            {
            }

            SIMD_INLINE void Resize(ptrdiff_t w, ptrdiff_t h, ptrdiff_t d)
            {
                width = w;
                height = h;
                depth = d;
            }

            SIMD_INLINE void Resize(const Size & s, ptrdiff_t d)
            {
                width = s.x;
                height = s.y;
                depth = d;
            }

            SIMD_INLINE ptrdiff_t Offset(ptrdiff_t x, ptrdiff_t y, ptrdiff_t c) const
            { 
                return (height * c + y) * width + x; 
            }

            template<class T, class A> SIMD_INLINE const T * Get(const std::vector<T, A> & v, ptrdiff_t x, ptrdiff_t y, ptrdiff_t c) const
            {
                return v.data() + Offset(x, y, c);
            }

            template<class T, class A> SIMD_INLINE T * Get(std::vector<T, A> & v, ptrdiff_t x, ptrdiff_t y, ptrdiff_t c) const
            {
                return v.data() + Offset(x, y, c);
            }

            SIMD_INLINE Neural::Size Size() const
            {
                return Neural::Size(width, height);
            }

            SIMD_INLINE ptrdiff_t Area() const
            { 
                return width * height; 
            }

            SIMD_INLINE ptrdiff_t Volume() const
            { 
                return width * height * depth; 
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

            Layer(Layer::Type l, Function::Type f)
                : type(l)
                , function(f)
                , _prev(0)
                , _next(0)
            {
            }

            virtual void Forward(const Vector & src, size_t thread, bool train) = 0;

            virtual void Backward(const Vector & src, size_t thread) = 0;

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
            }

            SIMD_INLINE const Vector & Dst(size_t thread) const
            { 
                return _common[thread].dst; 
            }

        protected:
            Layer * _prev, *_next;

            Index _src, _dst;
            Vector _weight, _bias;

            struct Common
            {
                Vector sum, dst;

                Vector dWeight, dBias, prevDelta;

            };
            std::vector<Common> _common;

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
                SetThreadNumber(1, false);
            }

            void Forward(const Vector & src, size_t thread, bool train) override
            {
                _common[thread].dst = src;
            }

            void Backward(const Vector & src, size_t thread) override
            {
            }
        };

        struct ConvolutionalLayer : public Layer
        {
            ConvolutionalLayer(Function::Type f, const Size & srcSize, size_t srcDepth, size_t dstDepth, size_t coreSize, bool valid = true, bool bias = true)
                : Layer(Convolutional, f)
            {
                _valid = valid;
                _indent = (coreSize - 1)/2;
                Size pad = _indent*Size(2, 2);
                _src.Resize(srcSize, srcDepth);
                _dst.Resize(srcSize - (_valid ? pad : Size()), dstDepth);
                _padded.Resize(srcSize + (_valid ? Size() : pad), srcDepth);
                _core.Resize(coreSize, coreSize, srcDepth*dstDepth);
                _weight.resize(_core.Volume());
                if (bias)
                    _bias.resize(dstDepth);
                SetThreadNumber(1, false);
            }

            void Forward(const Vector & src, size_t thread, bool train) override
            {
                const Vector & padded = PaddedSrc(src, thread);
                Vector & sum = _common[thread].sum;
                Vector & dst = _common[thread].dst;
                memset(sum.data(), 0, sum.size()*sizeof(float));
                for (ptrdiff_t dc = 0; dc < _dst.depth; ++dc)
                {
                    for (ptrdiff_t sc = 0; sc < _src.depth; ++sc)
                    {
                        const float_t * pweight = _core.Get(_weight, 0, 0, _src.depth*dc + sc);
                        const float_t * psrc = _padded.Get(padded, 0, 0, sc);
                        float_t * psum = _dst.Get(sum, 0, 0, dc);

                        if (_core.width == 3 && _core.height == 3)
                        {
                            ::SimdAnnAddConvolution3x3(psrc, _padded.width, _dst.width, _dst.height, pweight, psum, _dst.width);
                        }
                        else if (_core.width == 5 && _core.height == 5)
                        {
                            ::SimdAnnAddConvolution5x5(psrc, _padded.width, _dst.width, _dst.height, pweight, psum, _dst.width);
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
                    if (!_bias.empty())
                    {
                        float_t bias = _bias[dc];
                        size_t size = _dst.Area();
                        float_t * psum = _dst.Get(sum, 0, 0, dc);
                        for (size_t i = 0; i < size; ++i)
                            psum[i] += bias;
                    }
                }
                function.function(sum.data(), sum.size(), dst.data());
            }

            void Backward(const Vector & currDelta, size_t thread) override
            {
            }

            virtual void SetThreadNumber(size_t number, bool train) override
            {
                Layer::SetThreadNumber(number, train);
                _specific.resize(number);
                for (size_t i = 0; i < _specific.size(); ++i)
                {
                    if (!_valid)
                    {
                        _specific[i].paddedSrc.resize(_padded.Volume());
                    }
                    if (train)
                    {

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

            struct Specific
            {
                Vector paddedSrc;
            };
            std::vector<Specific> _specific;

            Index _core;
            Index _padded;
            size_t _indent;
            bool _valid;
        };

        struct MaxPoolingLayer : public Layer
        {
            MaxPoolingLayer(Function::Type f, const Size & srcSize, size_t srcDepth, size_t poolingSize)
                : Layer(MaxPooling, f)
            {
                _poolingSize = poolingSize;
                _src.Resize(srcSize, srcDepth);
                _dst.Resize(srcSize/_poolingSize, srcDepth);
                SetThreadNumber(1, false);
            }

            void Forward(const Vector & src, size_t thread, bool train) override
            {
                Vector & sum = _common[thread].sum;
                Vector & dst = _common[thread].dst;
                ptrdiff_t * idx = _specific[thread].index.data();

                if (train || _poolingSize != 2)
                {
                    for (ptrdiff_t c = 0; c < _dst.depth; ++c)
                    {
                        for (ptrdiff_t y = 0; y < _dst.height; y++)
                        {
                            for (ptrdiff_t x = 0; x < _dst.width; x++)
                            {
                                ptrdiff_t srcOffset = _src.Offset(x*_poolingSize, y*_poolingSize, c);
                                const float * psrc = src.data() + srcOffset;
                                ptrdiff_t maxIndex = 0;
                                float maxValue = std::numeric_limits<float_t>::lowest();
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
                else
                {
                    ::SimdAnnMax2x2(src.data(), _src.width, _src.width, _src.height*_src.depth, sum.data(), _dst.width);
                }
                function.function(sum.data(), sum.size(), dst.data());
            }

            void Backward(const Vector & src, size_t thread) override
            {
            }

            virtual void SetThreadNumber(size_t number, bool train) override
            {
                Layer::SetThreadNumber(number, train);
                _specific.resize(number);
                for (size_t i = 0; i < _specific.size(); ++i)
                {
                    _specific[i].index.resize(_dst.Volume());
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

        struct FullyConnectedLayer : public Layer
        {
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

            void Forward(const Vector & src, size_t thread, bool train) override
            {
                Vector & sum = _common[thread].sum;
                Vector & dst = _common[thread].dst;
                if (train)
                {
                    memset(sum.data(), 0, sizeof(float_t)*sum.size());
                    for (size_t i = 0; i < src.size(); i++)
                        ::SimdAnnAddVectorMultipliedByValue(&_weight[i*_dst.width], sum.size(), &src[i], sum.data());
                }
                else
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
                        ::SimdAnnProductSum(src.data(), &_weight[i*_src.width], src.size(), &sum[i]);
                }

                if (!_bias.empty())
                {
                    for (size_t i = 0; i < sum.size(); ++i)
                        sum[i] += _bias[i];
                }
                function.function(sum.data(), sum.size(), dst.data());
            }

            void Backward(const Vector & currDelta, size_t thread) override
            {
                const Vector & prevDst = _prev->Dst(thread);
                Vector & prevDelta = _common[thread].prevDelta;
                Vector & dWeight = _common[thread].dWeight;
                Vector & dBias = _common[thread].dBias;

                for (size_t i = 0; i < _src.width; i++)
                    ::SimdAnnProductSum(&currDelta[0], &_weight[i*_dst.width], _dst.width, &prevDelta[i]);

                _prev->function.derivative(&prevDst[0], prevDst.size(), &prevDelta[0]);

                for (size_t i = 0; i < _src.width; i++)
                    ::SimdAnnAddVectorMultipliedByValue(&currDelta[0], _dst.width, &prevDst[i], &dWeight[i*_dst.width]);

                if (_bias.size())
                {
                    for (size_t i = 0; i < _dst.width; ++i)
                        dBias[i] += currDelta[i];
                }
            }

        protected:
            bool _reordered;
        };

        struct TrainOptions
        {
            size_t threadNumber;
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
                    _layers.push_back(LayerPtr(new InputLayer(layer->_src.Size())));
                layer->_prev = _layers.back().get();
                _layers.back()->_next = layer;
                _layers.push_back(LayerPtr(layer));
            }

            bool Train(const Vectors & src, const Labels & dst, TrainOptions & options)
            {
                if (src.size() != dst.size())
                    return false;
                size_t size = _layers.back()->_dst.Volume();
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

                for (size_t i = 0; i < _layers.size(); ++i)
                    _layers[i]->SetThreadNumber(options.threadNumber, true);

                return false;
            }

            SIMD_INLINE const Vector & Predict(const Vector & x, bool train = false)
            {
                return Forward(x, 0, train);
            }

            bool Load(std::ifstream & ifs, bool train = false)
            {
                for (size_t i = 0; i < _layers.size(); ++i)
                {
                    Layer & layer = *_layers[i];
                    for (size_t j = 0; j < layer._weight.size(); ++j)
                        ifs >> layer._weight[j];
                    for (size_t j = 0; j < layer._bias.size(); ++j)
                        ifs >> layer._bias[j];
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
                    for (size_t j = 0; j < layer._weight.size(); ++j)
                        ofs << layer._weight[j] << " ";
                    for (size_t j = 0; j < layer._bias.size(); ++j)
                        ofs << layer._bias[j] << " ";
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

            const Vector & Forward(const Vector & src, size_t thread, bool train)
            {
                _layers[0]->Forward(src, thread, train);
                for (size_t i = 1; i < _layers.size(); ++i)
                    _layers[i]->Forward(_layers[i - 1]->Dst(thread), thread, train);
                return _layers.back()->Dst(thread);
            }

            void Backward(const Vector & src, size_t thread)
            {

            }
        };
    }
}

#endif//__SimdNeural_hpp__
