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
        template <bool align, bool mask> SIMD_INLINE void SynetAddBias(const __m512& bias, float* dst, __mmask16 tail = -1)
        {
            Store<align, mask>(dst, _mm512_add_ps((Load<align, mask>(dst, tail)), bias), tail);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetAddBias(const float* bias, float* dst, __mmask16 tail = -1)
        {
            __m512 _bias = Load<align, mask>(bias, tail);
            __m512 _dst = Load<align, mask>(dst, tail);
            Store<align, mask>(dst, _mm512_add_ps(_dst, _bias), tail);
        }

        template <bool align> void SynetAddBiasNchw(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            if (align)
                assert(Aligned(spatial, F) && Aligned(dst));

            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            __mmask16 tail = TailMask16(spatial - partial);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                __m512 _bias = _mm512_set1_ps(bias[c]);
                for (; s < aligned; s += QF)
                {
                    SynetAddBias<align, false>(_bias, dst + s + F * 0);
                    SynetAddBias<align, false>(_bias, dst + s + F * 1);
                    SynetAddBias<align, false>(_bias, dst + s + F * 2);
                    SynetAddBias<align, false>(_bias, dst + s + F * 3);
                }
                for (; s < partial; s += F)
                    SynetAddBias<align, false>(_bias, dst + s);
                if (s < spatial)
                    SynetAddBias<align, true>(_bias, dst + s, tail);
                dst += spatial;
            }
        }

        SIMD_INLINE void SynetAddBiasNchw(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(spatial, F) && Aligned(dst))
                SynetAddBiasNchw<true>(bias, channels, spatial, dst);
            else
                SynetAddBiasNchw<false>(bias, channels, spatial, dst);
        }

        template <bool align> void SynetAddBiasNhwc(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            if (align)
                assert(Aligned(channels, F) && Aligned(bias) && Aligned(dst));

            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            __mmask16 tail = TailMask16(channels - partial);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < aligned; c += QF)
                {
                    SynetAddBias<align, false>(bias + c + F * 0, dst + c + F * 0);
                    SynetAddBias<align, false>(bias + c + F * 1, dst + c + F * 1);
                    SynetAddBias<align, false>(bias + c + F * 2, dst + c + F * 2);
                    SynetAddBias<align, false>(bias + c + F * 3, dst + c + F * 3);
                }
                for (; c < partial; c += F)
                    SynetAddBias<align, false>(bias + c, dst + c);
                if (c < channels)
                    SynetAddBias<align, true>(bias + c, dst + c, tail);
                dst += channels;
            }
        }

        SIMD_INLINE void SynetAddBiasNhwc(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(bias) && Aligned(channels, F) && Aligned(dst))
                SynetAddBiasNhwc<true>(bias, channels, spatial, dst);
            else
                SynetAddBiasNhwc<false>(bias, channels, spatial, dst);
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
