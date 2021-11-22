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
#include "Simd/SimdArray.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        void SynetElu32f(const float * src, size_t size, const float * alpha, float * dst)
        {
            float _alpha = alpha[0];
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4)
            {
                dst[i + 0] = SynetElu32f(src[i + 0], _alpha);
                dst[i + 1] = SynetElu32f(src[i + 1], _alpha);
                dst[i + 2] = SynetElu32f(src[i + 2], _alpha);
                dst[i + 3] = SynetElu32f(src[i + 3], _alpha);
            }
            for (; i < size; ++i)
                dst[i] = SynetElu32f(src[i], _alpha);
        }

        //---------------------------------------------------------------------

        void SynetHardSigmoid32f(const float* src, size_t size, const float* scale, const float* shift, float* dst)
        {
            float _scale = scale[0];
            float _shift = shift[0];
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4)
            {
                dst[i + 0] = SynetHardSigmoid32f(src[i + 0], _scale, _shift);
                dst[i + 1] = SynetHardSigmoid32f(src[i + 1], _scale, _shift);
                dst[i + 2] = SynetHardSigmoid32f(src[i + 2], _scale, _shift);
                dst[i + 3] = SynetHardSigmoid32f(src[i + 3], _scale, _shift);
            }
            for (; i < size; ++i)
                dst[i] = SynetHardSigmoid32f(src[i], _scale, _shift);
        }

        //---------------------------------------------------------------------

        void SynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst)
        {
            float _shift = shift[0];
            float _scale = scale[0];
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4)
            {
                dst[i + 0] = SynetHswish32f(src[i + 0], _shift, _scale);
                dst[i + 1] = SynetHswish32f(src[i + 1], _shift, _scale);
                dst[i + 2] = SynetHswish32f(src[i + 2], _shift, _scale);
                dst[i + 3] = SynetHswish32f(src[i + 3], _shift, _scale);
            }
            for (; i < size; ++i)
                dst[i] = SynetHswish32f(src[i], _shift, _scale);
        }

        //---------------------------------------------------------------------

        void SynetMish32f(const float* src, size_t size, const float* threshold, float* dst)
        {
            float _threshold = threshold[0];
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4)
            {
                dst[i + 0] = SynetMish32f(src[i + 0], _threshold);
                dst[i + 1] = SynetMish32f(src[i + 1], _threshold);
                dst[i + 2] = SynetMish32f(src[i + 2], _threshold);
                dst[i + 3] = SynetMish32f(src[i + 3], _threshold);
            }
            for (; i < size; ++i)
                dst[i] = SynetMish32f(src[i], _threshold);
        }

        //---------------------------------------------------------------------

        void SynetPreluLayerForwardNchw(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            size_t aligned = Simd::AlignLo(spatial, 4);
            for (size_t c = 0; c < channels; ++c)
            {
                float _slope = slope[c];
                size_t s = 0;
                for (; s < aligned; s += 4)
                {
                    dst[s + 0] = SynetRelu32f(src[s + 0], _slope);
                    dst[s + 1] = SynetRelu32f(src[s + 1], _slope);
                    dst[s + 2] = SynetRelu32f(src[s + 2], _slope);
                    dst[s + 3] = SynetRelu32f(src[s + 3], _slope);
                }
                for (; s < spatial; ++s)
                    dst[s] = SynetRelu32f(src[s], _slope);
                src += spatial;
                dst += spatial;
            }
        }

        void SynetPreluLayerForwardNhwc(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            size_t aligned = Simd::AlignLo(channels, 4);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < aligned; c += 4)
                {
                    dst[c + 0] = SynetRelu32f(src[c + 0], slope[c + 0]);
                    dst[c + 1] = SynetRelu32f(src[c + 1], slope[c + 1]);
                    dst[c + 2] = SynetRelu32f(src[c + 2], slope[c + 2]);
                    dst[c + 3] = SynetRelu32f(src[c + 3], slope[c + 3]);
                }
                for (; c < channels; ++c)
                    dst[c] = SynetRelu32f(src[c], slope[c]);
                src += channels;
                dst += channels;

            }
        }

        template<int N> void SynetPreluLayerForwardNchwXc(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            for (size_t c = 0; c < channels; c += N)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t i = 0; i < N; ++i)
                        dst[i] = SynetRelu32f(src[i], slope[i]);
                    src += N;
                    dst += N;
                }
                slope += N;
            }
        }

        void SynetPreluLayerForward(const float* src, const float* slope, size_t channels, size_t spatial, float* dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetPreluLayerForwardNchw(src, slope, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetPreluLayerForwardNhwc(src, slope, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetPreluLayerForwardNchwXc<4>(src, slope, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw8c)
                SynetPreluLayerForwardNchwXc<8>(src, slope, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw16c)
                SynetPreluLayerForwardNchwXc<16>(src, slope, channels, spatial, dst);
            else
                assert(0);
        }

        //---------------------------------------------------------------------

        void SynetRelu32f(const float* src, size_t size, const float* slope, float* dst)
        {
            float _slope = slope[0];
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4)
            {
                dst[i + 0] = SynetRelu32f(src[i + 0], _slope);
                dst[i + 1] = SynetRelu32f(src[i + 1], _slope);
                dst[i + 2] = SynetRelu32f(src[i + 2], _slope);
                dst[i + 3] = SynetRelu32f(src[i + 3], _slope);
            }
            for (; i < size; ++i)
                dst[i] = SynetRelu32f(src[i], _slope);
        }

        //---------------------------------------------------------------------

        void SynetRestrictRange32f(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            float min = *lower;
            float max = *upper;
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4)
            {
                dst[i + 0] = Simd::RestrictRange(src[i + 0], min, max);
                dst[i + 1] = Simd::RestrictRange(src[i + 1], min, max);
                dst[i + 2] = Simd::RestrictRange(src[i + 2], min, max);
                dst[i + 3] = Simd::RestrictRange(src[i + 3], min, max);
            }
            for (; i < size; ++i)
                dst[i] = Simd::RestrictRange(src[i], min, max);
        }

        //---------------------------------------------------------------------

        void SynetSigmoid32f(const float* src, size_t size, const float* slope, float* dst)
        {
            float _slope = slope[0];
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4)
            {
                dst[i + 0] = SynetSigmoid32f(src[i + 0], _slope);
                dst[i + 1] = SynetSigmoid32f(src[i + 1], _slope);
                dst[i + 2] = SynetSigmoid32f(src[i + 2], _slope);
                dst[i + 3] = SynetSigmoid32f(src[i + 3], _slope);
            }
            for (; i < size; ++i)
                dst[i] = SynetSigmoid32f(src[i], _slope);
        }

        //---------------------------------------------------------------------

        void SynetSwish32f(const float* src, size_t size, const float* slope, float* dst)
        {
            float _slope = slope[0];
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4)
            {
                dst[i + 0] = SynetSwish32f(src[i + 0], _slope);
                dst[i + 1] = SynetSwish32f(src[i + 1], _slope);
                dst[i + 2] = SynetSwish32f(src[i + 2], _slope);
                dst[i + 3] = SynetSwish32f(src[i + 3], _slope);
            }
            for (; i < size; ++i)
                dst[i] = SynetSwish32f(src[i], _slope);
        }

        //---------------------------------------------------------------------

        void SynetSoftplus32f(const float* src, size_t size, const float * beta, const float * threshold, float* dst)
        {
            float _beta = beta[0];
            float _threshold = threshold[0];
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4)
            {
                dst[i + 0] = SynetSoftplus32f(src[i + 0], _beta, _threshold);
                dst[i + 1] = SynetSoftplus32f(src[i + 1], _beta, _threshold);
                dst[i + 2] = SynetSoftplus32f(src[i + 2], _beta, _threshold);
                dst[i + 3] = SynetSoftplus32f(src[i + 3], _beta, _threshold);
            }
            for (; i < size; ++i)
                dst[i] = SynetSoftplus32f(src[i], _beta, _threshold);
        }

        //---------------------------------------------------------------------

        void SynetTanh32f(const float* src, size_t size, const float* slope, float* dst)
        {
            float _slope = slope[0];
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4)
            {
                dst[i + 0] = SynetTanh32f(src[i + 0], _slope);
                dst[i + 1] = SynetTanh32f(src[i + 1], _slope);
                dst[i + 2] = SynetTanh32f(src[i + 2], _slope);
                dst[i + 3] = SynetTanh32f(src[i + 3], _slope);
            }
            for (; i < size; ++i)
                dst[i] = SynetTanh32f(src[i], _slope);
        }
    }
#endif
}
