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
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE svuint32_t Float32ToBFloat16(svfloat32_t value, const svbool_t& mask)
        {
            svuint32_t bits = svreinterpret_u32_f32(value);
            svuint32_t round = svadd_n_u32_x(mask, svand_n_u32_x(mask, svlsr_n_u32_x(mask, bits, Base::Bf16::SHIFT), 1), Base::Bf16::ROUND);
            return svlsr_n_u32_x(mask, svadd_u32_x(mask, bits, round), Base::Bf16::SHIFT);
        }

        SIMD_INLINE svfloat32_t BFloat16ToFloat32(svuint32_t value, const svbool_t& mask)
        {
            return svreinterpret_f32_u32(svlsl_n_u32_x(mask, value, Base::Bf16::SHIFT));
        }

        template<class S> SIMD_INLINE svfloat32_t LoadScale16b(const S* src, const svbool_t& mask);

        template<> SIMD_INLINE svfloat32_t LoadScale16b(const float* src, const svbool_t& mask)
        {
            return svld1_f32(mask, src);
        }

        template<> SIMD_INLINE svfloat32_t LoadScale16b(const uint16_t* src, const svbool_t& mask)
        {
            return BFloat16ToFloat32(svld1uh_u32(mask, src), mask);
        }

        template<class D> SIMD_INLINE void StoreScale16b(svfloat32_t value, D* dst, const svbool_t& mask);

        template<> SIMD_INLINE void StoreScale16b(svfloat32_t value, float* dst, const svbool_t& mask)
        {
            svst1_f32(mask, dst, value);
        }

        template<> SIMD_INLINE void StoreScale16b(svfloat32_t value, uint16_t* dst, const svbool_t& mask)
        {
            svst1h_u32(mask, dst, Float32ToBFloat16(value, mask));
        }

        //-------------------------------------------------------------------------------------------------

        template<class S, class D> SIMD_INLINE void NormBias16bF(const S* src, const float* norm, const float* bias, D* dst, const svbool_t& mask)
        {
            StoreScale16b(svmla_f32_x(mask, svld1_f32(mask, bias), LoadScale16b(src, mask), svld1_f32(mask, norm)), dst, mask);
        }

        template<class S, class D> SIMD_INLINE void NormBias16bF(const S* src, svfloat32_t norm, svfloat32_t bias, D* dst, const svbool_t& mask)
        {
            StoreScale16b(svmla_f32_x(mask, bias, LoadScale16b(src, mask), norm), dst, mask);
        }

        template<class S, class D> void SynetNormBias16b(const uint8_t* src8, size_t channels, size_t spatial, SimdTensorFormatType format, const float* norm, const float* bias, uint8_t* dst8)
        {
            const S* src = (const S*)src8;
            D* dst = (D*)dst8;
            const size_t F = svcntw();
            if (format == SimdTensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    svfloat32_t _norm = svdup_n_f32(norm[c]);
                    svfloat32_t _bias = svdup_n_f32(bias[c]);
                    for (size_t s = 0; s < spatial; s += F)
                        NormBias16bF<S, D>(src + s, _norm, _bias, dst + s, svwhilelt_b32(s, spatial));
                    src += spatial;
                    dst += spatial;
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; c += F)
                        NormBias16bF<S, D>(src + c, norm + c, bias + c, dst + c, svwhilelt_b32(c, channels));
                    src += channels;
                    dst += channels;
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template<class S, class D> SIMD_INLINE void Norm16bF(const S* src, const float* norm, D* dst, const svbool_t& mask)
        {
            StoreScale16b(svmul_f32_x(mask, LoadScale16b(src, mask), svld1_f32(mask, norm)), dst, mask);
        }

        template<class S, class D> SIMD_INLINE void Norm16bF(const S* src, svfloat32_t norm, D* dst, const svbool_t& mask)
        {
            StoreScale16b(svmul_f32_x(mask, LoadScale16b(src, mask), norm), dst, mask);
        }

        template<class S, class D> void SynetNorm16b(const uint8_t* src8, size_t channels, size_t spatial, SimdTensorFormatType format, const float* norm, const float* bias, uint8_t* dst8)
        {
            const S* src = (const S*)src8;
            D* dst = (D*)dst8;
            const size_t F = svcntw();
            if (format == SimdTensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    svfloat32_t _norm = svdup_n_f32(norm[c]);
                    for (size_t s = 0; s < spatial; s += F)
                        Norm16bF<S, D>(src + s, _norm, dst + s, svwhilelt_b32(s, spatial));
                    src += spatial;
                    dst += spatial;
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; c += F)
                        Norm16bF<S, D>(src + c, norm + c, dst + c, svwhilelt_b32(c, channels));
                    src += channels;
                    dst += channels;
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template<class S, class D> SIMD_INLINE void Bias16bF(const S* src, const float* bias, D* dst, const svbool_t& mask)
        {
            StoreScale16b(svadd_f32_x(mask, LoadScale16b(src, mask), svld1_f32(mask, bias)), dst, mask);
        }

        template<class S, class D> SIMD_INLINE void Bias16bF(const S* src, svfloat32_t bias, D* dst, const svbool_t& mask)
        {
            StoreScale16b(svadd_f32_x(mask, LoadScale16b(src, mask), bias), dst, mask);
        }

        template<class S, class D> void SynetBias16b(const uint8_t* src8, size_t channels, size_t spatial, SimdTensorFormatType format, const float* norm, const float* bias, uint8_t* dst8)
        {
            const S* src = (const S*)src8;
            D* dst = (D*)dst8;
            const size_t F = svcntw();
            if (format == SimdTensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    svfloat32_t _bias = svdup_n_f32(bias[c]);
                    for (size_t s = 0; s < spatial; s += F)
                        Bias16bF<S, D>(src + s, _bias, dst + s, svwhilelt_b32(s, spatial));
                    src += spatial;
                    dst += spatial;
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; c += F)
                        Bias16bF<S, D>(src + c, bias + c, dst + c, svwhilelt_b32(c, channels));
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
