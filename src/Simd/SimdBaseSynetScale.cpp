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
#include "Simd/SimdSynet.h"
#include "Simd/SimdSynetScale8i.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        void SynetScaleLayerForwardNchw(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(spatial, 4);
            if (bias)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float _scale = scale[c];
                    float _bias = bias[c];
                    size_t s = 0;
                    for (; s < aligned; s += 4)
                    {
                        dst[s + 0] = src[s + 0] * _scale + _bias;
                        dst[s + 1] = src[s + 1] * _scale + _bias;
                        dst[s + 2] = src[s + 2] * _scale + _bias;
                        dst[s + 3] = src[s + 3] * _scale + _bias;
                    }
                    for (; s < spatial; ++s)
                        dst[s] = src[s] * _scale + _bias;
                    src += spatial;
                    dst += spatial;
                }
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float _scale = scale[c];
                    size_t s = 0;
                    for (; s < aligned; s += 4)
                    {
                        dst[s + 0] = src[s + 0] * _scale;
                        dst[s + 1] = src[s + 1] * _scale;
                        dst[s + 2] = src[s + 2] * _scale;
                        dst[s + 3] = src[s + 3] * _scale;
                    }
                    for (; s < spatial; ++s)
                        dst[s] = src[s] * _scale;
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        void SynetScaleLayerForwardNhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(channels, 4);
            if (bias)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < aligned; c += 4)
                    {
                        dst[c + 0] = src[c + 0] * scale[c + 0] + bias[c + 0];
                        dst[c + 1] = src[c + 1] * scale[c + 1] + bias[c + 1];
                        dst[c + 2] = src[c + 2] * scale[c + 2] + bias[c + 2];
                        dst[c + 3] = src[c + 3] * scale[c + 3] + bias[c + 3];
                    }
                    for (; c < channels; ++c)
                        dst[c] = src[c] * scale[c] + bias[c];
                    src += channels;
                    dst += channels;

                }
            }
            else
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < aligned; c += 4)
                    {
                        dst[c + 0] = src[c + 0] * scale[c + 0];
                        dst[c + 1] = src[c + 1] * scale[c + 1];
                        dst[c + 2] = src[c + 2] * scale[c + 2];
                        dst[c + 3] = src[c + 3] * scale[c + 3];
                    }
                    for (; c < channels; ++c)
                        dst[c] = src[c] * scale[c];
                    src += channels;
                    dst += channels;
                }
            }
        }

        template<int N> void SynetScaleLayerForwardNchwXc(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (bias)
            {
                for (size_t c = 0; c < channels; c += N)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        for (size_t i = 0; i < N; ++i)
                            dst[i] = src[i]*scale[i] + bias[i];
                        src += N;
                        dst += N;
                    }
                    scale += N;
                    bias += N;
                }
            }
            else
            {
                for (size_t c = 0; c < channels; c += N)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        for (size_t i = 0; i < N; ++i)
                            dst[i] = src[i] * scale[i];
                        src += N;
                        dst += N;
                    }
                    scale += N;
                }
            }
        }

        void SynetScaleLayerForward(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            size_t spatial = height * width;
            if (Base::NchwCompatible(channels, spatial, format))
                SynetScaleLayerForwardNchw(src, scale, bias, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetScaleLayerForwardNhwc(src, scale, bias, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetScaleLayerForwardNchwXc<4>(src, scale, bias, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw8c)
                SynetScaleLayerForwardNchwXc<8>(src, scale, bias, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw16c)
                SynetScaleLayerForwardNchwXc<16>(src, scale, bias, channels, spatial, dst);
            else
                assert(0);
        }

        //---------------------------------------------------------------------

        SynetScale8i::SynetScale8i(const Scale8iParam& p)
            : _param(p)
        {
        }

        size_t SynetScale8i::InternalBufferSize() const
        {
            return _srcCvt.Size() + _dstCvt.Size() + (_scale.size + _shift.size) * sizeof(float);
        }

        void SynetScale8i::SetParams(const float* scale, const float* bias, const float* const* stats)
        {
            const Scale8iParam& p = _param;
            if (stats)
            {
                _srcCvt.Init(stats[0], stats[1], p.channels, p.compatibility);
                _dstCvt.Init(stats[2], stats[3], p.channels, p.compatibility);
            }
            else
                assert(_srcCvt.Size() && _dstCvt.Size());
            _scale.Resize(p.channels);
            _shift.Resize(p.channels);
            if (p.srcType == SimdTensorData8u)
            {
                for (size_t c = 0; c < p.channels; ++c)
                {
                    _scale[c] = _srcCvt.iScale[c];
                    _shift[c] = _srcCvt.iShift[c];
                }
            }
            else
            {
                for (size_t c = 0; c < p.channels; ++c)
                {
                    _scale[c] = 1.0f;
                    _shift[c] = 0.0f;
                }
            }
            if (bias)
            {
                for (size_t c = 0; c < p.channels; ++c)
                {
                    _scale[c] = _scale[c] * scale[c];
                    _shift[c] = _shift[c] * scale[c] + bias[c];
                }
            }
            else
            {
                for (size_t c = 0; c < p.channels; ++c)
                {
                    _scale[c] = _scale[c] * scale[c];
                    _shift[c] = _shift[c] * scale[c];
                }
            }
            if (p.dstType == SimdTensorData8u)
            {
                for (size_t c = 0; c < p.channels; ++c)
                {
                    _scale[c] = _scale[c] * _dstCvt.scale[c];
                    _shift[c] = _shift[c] * _dstCvt.scale[c] + _dstCvt.shift[c];
                }
            }
        }

        void SynetScale8i::Forward(const uint8_t* src, uint8_t* dst)
        {
            const Scale8iParam& p = _param;
            if (p.srcType == SimdTensorData8u && p.dstType == SimdTensorData8u)
                Scale((const uint8_t*)src, (uint8_t*)dst);
            else if (p.srcType == SimdTensorData32f && p.dstType == SimdTensorData8u)
                Scale((const float*)src, (uint8_t*)dst);
            else if (p.srcType == SimdTensorData8u && p.dstType == SimdTensorData32f)
                Scale((const uint8_t*)src, (float*)dst);
            else if (p.srcType == SimdTensorData32f && p.dstType == SimdTensorData32f)
                Scale((const float*)src, (float*)dst);
            else
                assert(0);
        }

        template<class S, class D> D Scale(S value, float scale, float shift, int lower, int upper);

        template<> SIMD_INLINE uint8_t Scale<uint8_t, uint8_t>(uint8_t value, float scale, float shift, int lower, int upper)
        {
            return (uint8_t)Simd::RestrictRange(Round(float(value) * scale + shift), lower, upper);
        }

        template<> SIMD_INLINE float Scale<uint8_t, float>(uint8_t value, float scale, float shift, int lower, int upper)
        {
            return float(value) * scale + shift;
        }

        template<> SIMD_INLINE uint8_t Scale<float, uint8_t>(float value, float scale, float shift, int lower, int upper)
        {
            return (uint8_t)Simd::RestrictRange(Round(value * scale + shift), lower, upper);
        }

        template<> SIMD_INLINE float Scale<float, float>(float value, float scale, float shift, int lower, int upper)
        {
            return value * scale + shift;
        }

        template<class S, class D> void Scale(const S* src, size_t batch, size_t channels, size_t spatial,
            SimdTensorFormatType format, const float* scale, const float* shift, int lower, int upper, D* dst)
        {
            for (size_t b = 0; b < batch; ++b)
            {
                if (format == SimdTensorFormatNchw)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        float _scale = scale[c];
                        float _shift = shift[c];
                        for (size_t s = 0; s < spatial; ++s)
                            dst[s] = Scale<S, D>(src[s], _scale, _shift, lower, upper);
                        src += spatial;
                        dst += spatial;
                    }
                }
                else if (format == SimdTensorFormatNhwc)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = Scale<S, D>(src[c], scale[c], shift[c], lower, upper);
                        src += channels;
                        dst += channels;
                    }
                }
                else
                    assert(0);
            }
        }

        void SynetScale8i::Scale(const uint8_t* src, uint8_t* dst)
        {
            const Scale8iParam& p = _param;
            Base::Scale(src, p.batch, p.channels, p.spatial, p.format, _scale.data, _shift.data, _dstCvt.uMin, _dstCvt.uMax, dst);
        }

        void SynetScale8i::Scale(const uint8_t* src, float* dst)
        {
            const Scale8iParam& p = _param;
            Base::Scale(src, p.batch, p.channels, p.spatial, p.format, _scale.data, _shift.data, _dstCvt.uMin, _dstCvt.uMax, dst);
        }

        void SynetScale8i::Scale(const float* src, uint8_t* dst)
        {
            const Scale8iParam& p = _param;
            Base::Scale(src, p.batch, p.channels, p.spatial, p.format, _scale.data, _shift.data, _dstCvt.uMin, _dstCvt.uMax, dst);
        }

        void SynetScale8i::Scale(const float* src, float* dst)
        {
            const Scale8iParam& p = _param;
            Base::Scale(src, p.batch, p.channels, p.spatial, p.format, _scale.data, _shift.data, _dstCvt.uMin, _dstCvt.uMax, dst);
        }

        void* SynetScale8iInit(size_t batch, size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            Base::Scale8iParam param(batch, channels, spatial, srcType, dstType, format, compatibility);
            if (!param.Valid())
                return NULL;
            return new Base::SynetScale8i(param);
        }
    }
#endif
}
