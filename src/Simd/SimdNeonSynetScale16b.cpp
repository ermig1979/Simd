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
#include "Simd/SimdSynetScale16b.h"
#include "Simd/SimdSynetAdd16bCommon.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Neon
    {
        SIMD_INLINE uint16x8_t Float32ToBFloat16(float32x4_t lo, float32x4_t hi)
        {
            return vcombine_u16(vmovn_u32(Float32ToBFloat16(lo)), vmovn_u32(Float32ToBFloat16(hi)));
        }

        SIMD_INLINE float32x4_t BFloat16ToFloat32(uint16x4_t value)
        {
            return BFloat16ToFloat32(vmovl_u16(value));
        }

        SIMD_INLINE float32x4_t SetF32(float a, float b, float c, float d)
        {
            float value[4] = { a, b, c, d };
            return vld1q_f32(value);
        }

        template<class S> SIMD_INLINE float32x4_t Load16b(const S* src);

        template<> SIMD_INLINE float32x4_t Load16b(const float* src)
        {
            return vld1q_f32(src);
        }

        template<> SIMD_INLINE float32x4_t Load16b(const uint16_t* src)
        {
            return BFloat16ToFloat32(vld1_u16(src));
        }

        template<class D> SIMD_INLINE void Store16b(float32x4_t value, D* dst);

        template<> SIMD_INLINE void Store16b(float32x4_t value, float* dst)
        {
            vst1q_f32(dst, value);
        }

        template<> SIMD_INLINE void Store16b(float32x4_t value, uint16_t* dst)
        {
            vst1_u16(dst, vmovn_u32(Float32ToBFloat16(value)));
        }

        //-------------------------------------------------------------------------------------------------

        template<class S, class D> SIMD_INLINE void NormBias16bF(const S* src, const float* norm, const float* bias, D* dst)
        {
            Store16b(vaddq_f32(vmulq_f32(Load16b(src), vld1q_f32(norm)), vld1q_f32(bias)), dst);
        }

        template<class S, class D> SIMD_INLINE void NormBias16bF(const S* src, float32x4_t norm, float32x4_t bias, D* dst)
        {
            Store16b(vaddq_f32(vmulq_f32(Load16b(src), norm), bias), dst);
        }

        template<class S, class D> void SynetNormBias16b(const uint8_t* src8, size_t channels, size_t spatial, SimdTensorFormatType format, const float* norm, const float* bias, uint8_t* dst8)
        {
            const S* src = (const S*)src8;
            D* dst = (D*)dst8;
            if (format == SimdTensorFormatNchw)
            {
                size_t spatialF = AlignLo(spatial, F);
                for (size_t c = 0; c < channels; ++c)
                {
                    float32x4_t _norm = vdupq_n_f32(norm[c]);
                    float32x4_t _bias = vdupq_n_f32(bias[c]);
                    size_t s = 0;
                    for (; s < spatialF; s += F)
                        NormBias16bF<S, D>(src + s, _norm, _bias, dst + s);
                    for (; s < spatial; ++s)
                        Base::NormBias16b<S, D>(src[s], norm[c], bias[c], dst[s]);
                    src += spatial;
                    dst += spatial;
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                if (channels == 3)
                {
                    size_t spatialF = AlignLo(spatial, F) * 3;
                    spatial *= 3;
                    float32x4_t _norm[3];
                    _norm[0] = SetF32(norm[0], norm[1], norm[2], norm[0]);
                    _norm[1] = SetF32(norm[1], norm[2], norm[0], norm[1]);
                    _norm[2] = SetF32(norm[2], norm[0], norm[1], norm[2]);
                    float32x4_t _bias[3];
                    _bias[0] = SetF32(bias[0], bias[1], bias[2], bias[0]);
                    _bias[1] = SetF32(bias[1], bias[2], bias[0], bias[1]);
                    _bias[2] = SetF32(bias[2], bias[0], bias[1], bias[2]);
                    size_t s = 0;
                    for (; s < spatialF; s += 3 * F)
                    {
                        NormBias16bF<S, D>(src + s + 0 * F, _norm[0], _bias[0], dst + s + 0 * F);
                        NormBias16bF<S, D>(src + s + 1 * F, _norm[1], _bias[1], dst + s + 1 * F);
                        NormBias16bF<S, D>(src + s + 2 * F, _norm[2], _bias[2], dst + s + 2 * F);
                    }
                    for (; s < spatial; s += 3)
                    {
                        Base::NormBias16b<S, D>(src[s + 0], norm[0], bias[0], dst[s + 0]);
                        Base::NormBias16b<S, D>(src[s + 1], norm[1], bias[1], dst[s + 1]);
                        Base::NormBias16b<S, D>(src[s + 2], norm[2], bias[2], dst[s + 2]);
                    }
                }
                else
                {
                    size_t channelsF = AlignLo(channels, F);
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        size_t c = 0;
                        for (; c < channelsF; c += F)
                            NormBias16bF<S, D>(src + c, norm + c, bias + c, dst + c);
                        for (; c < channels; ++c)
                            Base::NormBias16b<S, D>(src[c], norm[c], bias[c], dst[c]);
                        src += channels;
                        dst += channels;
                    }
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template<class S, class D> SIMD_INLINE void Norm16bF(const S* src, const float* norm, D* dst)
        {
            Store16b(vmulq_f32(Load16b(src), vld1q_f32(norm)), dst);
        }

        template<class S, class D> SIMD_INLINE void Norm16bF(const S* src, float32x4_t norm, D* dst)
        {
            Store16b(vmulq_f32(Load16b(src), norm), dst);
        }

        template<class S, class D> void SynetNorm16b(const uint8_t* src8, size_t channels, size_t spatial, SimdTensorFormatType format, const float* norm, const float* bias, uint8_t* dst8)
        {
            const S* src = (const S*)src8;
            D* dst = (D*)dst8;
            if (format == SimdTensorFormatNchw)
            {
                size_t spatialF = AlignLo(spatial, F);
                for (size_t c = 0; c < channels; ++c)
                {
                    float32x4_t _norm = vdupq_n_f32(norm[c]);
                    size_t s = 0;
                    for (; s < spatialF; s += F)
                        Norm16bF<S, D>(src + s, _norm, dst + s);
                    for (; s < spatial; ++s)
                        Base::Norm16b<S, D>(src[s], norm[c], dst[s]);
                    src += spatial;
                    dst += spatial;
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                size_t channelsF = AlignLo(channels, F);
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                        Norm16bF<S, D>(src + c, norm + c, dst + c);
                    for (; c < channels; ++c)
                        Base::Norm16b<S, D>(src[c], norm[c], dst[c]);
                    src += channels;
                    dst += channels;
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template<class S, class D> SIMD_INLINE void Bias16bF(const S* src, const float* bias, D* dst)
        {
            Store16b(vaddq_f32(Load16b(src), vld1q_f32(bias)), dst);
        }

        template<class S, class D> SIMD_INLINE void Bias16bF(const S* src, float32x4_t bias, D* dst)
        {
            Store16b(vaddq_f32(Load16b(src), bias), dst);
        }

        template<class S, class D> void SynetBias16b(const uint8_t* src8, size_t channels, size_t spatial, SimdTensorFormatType format, const float* norm, const float* bias, uint8_t* dst8)
        {
            const S* src = (const S*)src8;
            D* dst = (D*)dst8;
            if (format == SimdTensorFormatNchw)
            {
                size_t spatialF = AlignLo(spatial, F);
                for (size_t c = 0; c < channels; ++c)
                {
                    float32x4_t _bias = vdupq_n_f32(bias[c]);
                    size_t s = 0;
                    for (; s < spatialF; s += F)
                        Bias16bF<S, D>(src + s, _bias, dst + s);
                    for (; s < spatial; ++s)
                        Base::Bias16b<S, D>(src[s], bias[c], dst[s]);
                    src += spatial;
                    dst += spatial;
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                size_t channelsF = AlignLo(channels, F);
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                        Bias16bF<S, D>(src + c, bias + c, dst + c);
                    for (; c < channels; ++c)
                        Base::Bias16b<S, D>(src[c], bias[c], dst[c]);
                    src += channels;
                    dst += channels;
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template<class S, class D> static SynetScale16b::WorkerPtr GetScale16bWorker(SimdBool norm, SimdBool bias)
        {
            if (norm)
                return bias ? SynetNormBias16b<S, D> : SynetNorm16b<S, D>;
            else
                return bias ? SynetBias16b<S, D> : NULL;
        }

        template<class S> static SynetScale16b::WorkerPtr GetScale16bWorker(SimdTensorDataType dType, SimdBool norm, SimdBool bias)
        {
            switch (dType)
            {
            case SimdTensorData32f: return GetScale16bWorker<S, float>(norm, bias);
            case SimdTensorData16b: return GetScale16bWorker<S, uint16_t>(norm, bias);
            default:
                return NULL;
            }
        }

        static SynetScale16b::WorkerPtr GetScale16bWorker(SimdTensorDataType sType, SimdTensorDataType dType, SimdBool norm, SimdBool bias)
        {
            switch (sType)
            {
            case SimdTensorData32f: return GetScale16bWorker<float>(dType, norm, bias);
            case SimdTensorData16b: return GetScale16bWorker<uint16_t>(dType, norm, bias);
            default:
                return NULL;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetScale16b::SynetScale16b(const Scale16bParam& p)
            : Base::SynetScale16b(p)
        {
            _worker = GetScale16bWorker(p.sType, p.dType, p.norm, p.bias);
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetScale16bInit(size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdBool norm, SimdBool bias)
        {
            Scale16bParam param(channels, spatial, srcType, dstType, format, norm, bias);
            if (!param.Valid())
                return NULL;
            if (SynetScale16b::Preferable(param))
                return new SynetScale16b(param);
            return NULL;
        }
    }
#endif
}
