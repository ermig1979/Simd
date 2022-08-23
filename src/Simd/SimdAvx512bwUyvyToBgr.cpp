/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdYuvToBgr.h"
#include "Simd/SimdInterleave.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <class T> SIMD_INLINE __m512i UnpackY(__m512i uyvy)
        {
            static const __m512i Y_SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x1, -1, 0x3, -1, 0x5, -1, 0x7, -1, 0x9, -1, 0xB, -1, 0xD, -1, 0xF, -1,
                0x1, -1, 0x3, -1, 0x5, -1, 0x7, -1, 0x9, -1, 0xB, -1, 0xD, -1, 0xF, -1,
                0x1, -1, 0x3, -1, 0x5, -1, 0x7, -1, 0x9, -1, 0xB, -1, 0xD, -1, 0xF, -1,
                0x1, -1, 0x3, -1, 0x5, -1, 0x7, -1, 0x9, -1, 0xB, -1, 0xD, -1, 0xF, -1);
            static const __m512i Y_LO = SIMD_MM512_SET1_EPI16(T::Y_LO);
            return _mm512_subs_epi16(_mm512_shuffle_epi8(uyvy, Y_SHUFFLE), Y_LO);
        }

        template <class T> SIMD_INLINE __m512i UnpackU(__m512i uyvy)
        {
            static const __m512i U_SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, -1, 0x0, -1, 0x4, -1, 0x4, -1, 0x8, -1, 0x8, -1, 0xC, -1, 0xC, -1,
                0x0, -1, 0x0, -1, 0x4, -1, 0x4, -1, 0x8, -1, 0x8, -1, 0xC, -1, 0xC, -1,
                0x0, -1, 0x0, -1, 0x4, -1, 0x4, -1, 0x8, -1, 0x8, -1, 0xC, -1, 0xC, -1,
                0x0, -1, 0x0, -1, 0x4, -1, 0x4, -1, 0x8, -1, 0x8, -1, 0xC, -1, 0xC, -1);
            static const __m512i U_Z = SIMD_MM512_SET1_EPI16(T::UV_Z);
            return _mm512_subs_epi16(_mm512_shuffle_epi8(uyvy, U_SHUFFLE), U_Z);
        }

        template <class T> SIMD_INLINE __m512i UnpackV(__m512i uyvy)
        {
            static const __m512i V_SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x2, -1, 0x2, -1, 0x6, -1, 0x6, -1, 0xA, -1, 0xA, -1, 0xE, -1, 0xE, -1,
                0x2, -1, 0x2, -1, 0x6, -1, 0x6, -1, 0xA, -1, 0xA, -1, 0xE, -1, 0xE, -1,
                0x2, -1, 0x2, -1, 0x6, -1, 0x6, -1, 0xA, -1, 0xA, -1, 0xE, -1, 0xE, -1,
                0x2, -1, 0x2, -1, 0x6, -1, 0x6, -1, 0xA, -1, 0xA, -1, 0xE, -1, 0xE, -1);
            static const __m512i V_Z= SIMD_MM512_SET1_EPI16(T::UV_Z);
            return _mm512_subs_epi16(_mm512_shuffle_epi8(uyvy, V_SHUFFLE), V_Z);
        }

        template <bool align, bool mask, class T> SIMD_INLINE void Uyvy422ToBgr(const uint8_t* uyvy, uint8_t* bgr, __mmask64 tails[5])
        {
            __m512i uyvy0 = Load<align, mask>(uyvy + 0 * A, tails[0]);
            __m512i y0 = UnpackY<T>(uyvy0);
            __m512i u0 = UnpackU<T>(uyvy0);
            __m512i v0 = UnpackV<T>(uyvy0);
            __m512i b0 = YuvToBlue16<T>(y0, u0);
            __m512i g0 = YuvToGreen16<T>(y0, u0, v0);
            __m512i r0 = YuvToRed16<T>(y0, v0);

            __m512i uyvy1 = Load<align, mask>(uyvy + 1 * A, tails[1]);
            __m512i y1 = UnpackY<T>(uyvy1);
            __m512i u1 = UnpackU<T>(uyvy1);
            __m512i v1 = UnpackV<T>(uyvy1);
            __m512i b1 = YuvToBlue16<T>(y1, u1);
            __m512i g1 = YuvToGreen16<T>(y1, u1, v1);
            __m512i r1 = YuvToRed16<T>(y1, v1);

            __m512i b = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi16(b0, b1));
            __m512i g = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi16(g0, g1));
            __m512i r = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi16(r0, r1));
            Store<align, mask>(bgr + 0 * A, InterleaveBgr<0>(b, g, r), tails[2]);
            Store<align, mask>(bgr + 1 * A, InterleaveBgr<1>(b, g, r), tails[3]);
            Store<align, mask>(bgr + 2 * A, InterleaveBgr<2>(b, g, r), tails[4]);
        }

        template <bool align, class T> void Uyvy422ToBgr(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (width >= 2 * A));
            if (align)
                assert(Aligned(uyvy) && Aligned(uyvyStride) && Aligned(bgr) && Aligned(bgrStride));

            size_t widthA = AlignLo(width, A);
            size_t sizeS = width * 2, sizeD = width * 3;
            size_t sizeSA = widthA * 2, sizeDA = widthA * 3;
            __mmask64 tails[5];
            if (widthA < width)
            {
                tails[0] = TailMask64(sizeS - sizeSA - A * 0);
                tails[1] = TailMask64(sizeS - sizeSA - A * 1);
                tails[2] = TailMask64(sizeD - sizeDA - A * 0);
                tails[3] = TailMask64(sizeD - sizeDA - A * 1);
                tails[4] = TailMask64(sizeD - sizeDA - A * 2);
            }
            for (size_t row = 0; row < height; ++row)
            {
                size_t colS = 0, colD = 0;
                for (; colS < sizeSA; colS += 2 * A, colD += 3 * A)
                    Uyvy422ToBgr<align, false, T>(uyvy + colS, bgr + colD, tails);
                if(widthA < width)
                    Uyvy422ToBgr<align, true, T>(uyvy + colS, bgr + colD, tails);
                uyvy += uyvyStride;
                bgr += bgrStride;
            }
        }

        template<bool align> void Uyvy422ToBgr(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Uyvy422ToBgr<align, Base::Bt601>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt709: Uyvy422ToBgr<align, Base::Bt709>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt2020: Uyvy422ToBgr<align, Base::Bt2020>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            case SimdYuvTrect871: Uyvy422ToBgr<align, Base::Trect871>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            default:
                assert(0);
            }
        }

        void Uyvy422ToBgr(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            if (Aligned(uyvy) && Aligned(uyvyStride) && Aligned(bgr) && Aligned(bgrStride))
                Uyvy422ToBgr<true>(uyvy, uyvyStride, width, height, bgr, bgrStride, yuvType);
            else
                Uyvy422ToBgr<false>(uyvy, uyvyStride, width, height, bgr, bgrStride, yuvType);
        }
    }
#endif
}
