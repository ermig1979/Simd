/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        const v128_u16 K16_BLUE_RED = SIMD_VEC_SET2_EPI16(Base::BLUE_TO_GRAY_WEIGHT, Base::RED_TO_GRAY_WEIGHT);
        const v128_u16 K16_GREEN_0000 = SIMD_VEC_SET2_EPI16(Base::GREEN_TO_GRAY_WEIGHT, 0x0000);
        const v128_u32 K32_ROUND_TERM = SIMD_VEC_SET1_EPI32(Base::BGR_TO_GRAY_ROUND_TERM);
        const v128_u32 K32_SHIFT = SIMD_VEC_SET1_EPI32(Base::BGR_TO_GRAY_AVERAGING_SHIFT);

        SIMD_INLINE v128_u32 BgraToGray32(v128_u8 bgra)
        {
            const v128_u16 _b_r = vec_mule(bgra, K8_01);
            const v128_u16 _g_a = vec_mulo(bgra, K8_01);
            const v128_u32 weightedSum = vec_add(vec_mule(_g_a, K16_GREEN_0000),
                vec_add(vec_mule(_b_r, K16_BLUE_RED), vec_mulo(_b_r, K16_BLUE_RED)));
            return vec_sr(vec_add(weightedSum, K32_ROUND_TERM), K32_SHIFT);
        }

        template<bool align, bool first>
        SIMD_INLINE void BgraToGray(const Loader<align> & bgra, Storer<align> & gray)
        {
            v128_u8 _bgra[4];
            _bgra[0] = Load<align, first>(bgra);
            _bgra[1] = Load<align, false>(bgra);
            _bgra[2] = Load<align, false>(bgra);
            _bgra[3] = Load<align, false>(bgra);

            const v128_u16 lo = vec_packsu(BgraToGray32(_bgra[0]), BgraToGray32(_bgra[1]));
            const v128_u16 hi = vec_packsu(BgraToGray32(_bgra[2]), BgraToGray32(_bgra[3]));
            Store<align, first>(gray, vec_packsu(lo, hi));
        }

        template <bool align> void BgraToGray(const uint8_t *bgra, size_t width, size_t height, size_t bgraStride, uint8_t *gray, size_t grayStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(gray) && Aligned(grayStride));

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _bgra(bgra);
                Storer<align> _gray(gray);
                BgraToGray<align, true>(_bgra, _gray);
                for (size_t col = A; col < alignedWidth; col += A)
                    BgraToGray<align, false>(_bgra, _gray);
                Flush(_gray);

                if (alignedWidth != width)
                {
                    Loader<false> _bgra(bgra + 4 * (width - A));
                    Storer<false> _gray(gray + width - A);
                    BgraToGray<false, true>(_bgra, _gray);
                    Flush(_gray);
                }

                bgra += bgraStride;
                gray += grayStride;
            }
        }

        void BgraToGray(const uint8_t *bgra, size_t width, size_t height, size_t bgraStride, uint8_t *gray, size_t grayStride)
        {
            if (Aligned(bgra) && Aligned(gray) && Aligned(bgraStride) && Aligned(grayStride))
                BgraToGray<true>(bgra, width, height, bgraStride, gray, grayStride);
            else
                BgraToGray<false>(bgra, width, height, bgraStride, gray, grayStride);
        }
    }
#endif// SIMD_VMX_ENABLE
}
