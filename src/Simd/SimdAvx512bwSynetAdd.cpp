/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdInterleave.h"
#include "Simd/SimdDeinterleave.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)     
    namespace Avx512bw
    {
        template <bool mask> SIMD_INLINE void SynetAddBias(const __m512& bias, float* dst, __mmask16 tail = -1)
        {
            if (mask)
                _mm512_mask_storeu_ps(dst, tail, _mm512_add_ps(((mask ? _mm512_maskz_loadu_ps(tail, dst) : _mm512_loadu_ps(dst))), bias));
            else
                _mm512_storeu_ps(dst, _mm512_add_ps(((mask ? _mm512_maskz_loadu_ps(tail, dst) : _mm512_loadu_ps(dst))), bias));
        }

        template <bool mask> SIMD_INLINE void SynetAddBias(const float* bias, float* dst, __mmask16 tail = -1)
        {
            __m512 _bias = (mask ? _mm512_maskz_loadu_ps(tail, bias) : _mm512_loadu_ps(bias));
            __m512 _dst = (mask ? _mm512_maskz_loadu_ps(tail, dst) : _mm512_loadu_ps(dst));
            if (mask)
                _mm512_mask_storeu_ps(dst, tail, _mm512_add_ps(_dst, _bias));
            else
                _mm512_storeu_ps(dst, _mm512_add_ps(_dst, _bias));
        }

        void SynetAddBiasNchw(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            __mmask16 tail = TailMask16(spatial - partial);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                __m512 _bias = _mm512_set1_ps(bias[c]);
                for (; s < aligned; s += QF)
                {
                    SynetAddBias<false>(_bias, dst + s + F * 0);
                    SynetAddBias<false>(_bias, dst + s + F * 1);
                    SynetAddBias<false>(_bias, dst + s + F * 2);
                    SynetAddBias<false>(_bias, dst + s + F * 3);
                }
                for (; s < partial; s += F)
                    SynetAddBias<false>(_bias, dst + s);
                if (s < spatial)
                    SynetAddBias<true>(_bias, dst + s, tail);
                dst += spatial;
            }
        }

        void SynetAddBiasNhwc(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            __mmask16 tail = TailMask16(channels - partial);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < aligned; c += QF)
                {
                    SynetAddBias<false>(bias + c + F * 0, dst + c + F * 0);
                    SynetAddBias<false>(bias + c + F * 1, dst + c + F * 1);
                    SynetAddBias<false>(bias + c + F * 2, dst + c + F * 2);
                    SynetAddBias<false>(bias + c + F * 3, dst + c + F * 3);
                }
                for (; c < partial; c += F)
                    SynetAddBias<false>(bias + c, dst + c);
                if (c < channels)
                    SynetAddBias<true>(bias + c, dst + c, tail);
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
