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
#include "Simd/SimdMemory.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE void SynetAddBias(const svfloat32_t& bias, float* dst, const svbool_t& mask)
        {
            svst1_f32(mask, dst, svadd_f32_x(mask, svld1_f32(mask, dst), bias));
        }

        SIMD_INLINE void SynetAddBias(const float* bias, float* dst, const svbool_t& mask)
        {
            svst1_f32(mask, dst, svadd_f32_x(mask, svld1_f32(mask, dst), svld1_f32(mask, bias)));
        }

        void SynetAddBiasNchw(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                svfloat32_t _bias = svdup_n_f32(bias[c]);
                for (; s + QF <= spatial; s += QF)
                {
                    SynetAddBias(_bias, dst + s + 0 * F, body);
                    SynetAddBias(_bias, dst + s + 1 * F, body);
                    SynetAddBias(_bias, dst + s + 2 * F, body);
                    SynetAddBias(_bias, dst + s + 3 * F, body);
                }
                for (; s + F <= spatial; s += F)
                    SynetAddBias(_bias, dst + s, body);
                if (s < spatial)
                    SynetAddBias(_bias, dst + s, svwhilelt_b32(s, spatial));
                dst += spatial;
            }
        }

        void SynetAddBiasNhwc(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c + QF <= channels; c += QF)
                {
                    SynetAddBias(bias + c + 0 * F, dst + c + 0 * F, body);
                    SynetAddBias(bias + c + 1 * F, dst + c + 1 * F, body);
                    SynetAddBias(bias + c + 2 * F, dst + c + 2 * F, body);
                    SynetAddBias(bias + c + 3 * F, dst + c + 3 * F, body);
                }
                for (; c + F <= channels; c += F)
                    SynetAddBias(bias + c, dst + c, body);
                if (c < channels)
                    SynetAddBias(bias + c, dst + c, svwhilelt_b32(c, channels));
                dst += channels;
            }
        }

        void SynetAddBias(const float* bias, size_t channels, size_t spatial, float* dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetAddBiasNchw(bias, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetAddBiasNhwc(bias, channels, spatial, dst);
            else
                assert(0);
        }
    }
#endif
}
