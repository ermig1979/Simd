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
#include "Simd/SimdUnpack.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {

        template<bool align> SIMD_INLINE void Yuv420pToUyvy422(const uint8_t* y0, size_t yStride, 
            const uint8_t* u, const uint8_t* v, uint8_t* uyvy0, size_t uyvyStride)
        {
            static const __m256i K32_PERMUTE_UV = SIMD_MM256_SETR_EPI32(0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7);
            __m256i u0 = _mm256_permutevar8x32_epi32(Load<align>((__m256i*)u), K32_PERMUTE_UV);
            __m256i v0 = _mm256_permutevar8x32_epi32(Load<align>((__m256i*)v), K32_PERMUTE_UV);
            __m256i uv0 = UnpackU8<0>(u0, v0);
            __m256i uv1 = UnpackU8<1>(u0, v0);

            __m256i y00 = LoadPermuted<align>((__m256i*)y0 + 0);
            __m256i y01 = LoadPermuted<align>((__m256i*)y0 + 1);
            Store<align>((__m256i*)uyvy0 + 0, UnpackU8<0>(uv0, y00));
            Store<align>((__m256i*)uyvy0 + 1, UnpackU8<1>(uv0, y00));
            Store<align>((__m256i*)uyvy0 + 2, UnpackU8<0>(uv1, y01));
            Store<align>((__m256i*)uyvy0 + 3, UnpackU8<1>(uv1, y01));

            const uint8_t* y1 = y0 + yStride;
            __m256i y10 = LoadPermuted<align>((__m256i*)y1 + 0);
            __m256i y11 = LoadPermuted<align>((__m256i*)y1 + 1);
            uint8_t* uyvy1 = uyvy0 + uyvyStride;
            Store<align>((__m256i*)uyvy1 + 0, UnpackU8<0>(uv0, y10));
            Store<align>((__m256i*)uyvy1 + 1, UnpackU8<1>(uv0, y10));
            Store<align>((__m256i*)uyvy1 + 2, UnpackU8<0>(uv1, y11));
            Store<align>((__m256i*)uyvy1 + 3, UnpackU8<1>(uv1, y11));
        }

        template<bool align> void Yuv420pToUyvy422(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, 
            const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* uyvy, size_t uyvyStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && width >= 2 * A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(uyvy) && Aligned(uyvyStride));
            }

            size_t width2A = AlignLo(width, 2 * A);
            size_t tailY = width - 2 * A;
            size_t tailUV = width / 2 - A;
            size_t tailUyvy = width * 2 - 4 * A;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colY = 0, colUV = 0, colUyvy = 0; colY < width2A; colY += 2 * A, colUV += 1 * A, colUyvy += 4 * A)
                    Yuv420pToUyvy422<align>(y + colY, yStride, u + colUV, v + colUV, uyvy + colUyvy, uyvyStride);
                if (width2A != width)
                    Yuv420pToUyvy422<false>(y + tailY, yStride, u + tailUV, v + tailUV, uyvy + tailUyvy, uyvyStride);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                uyvy += 2 * uyvyStride;
            }
        }

        void Yuv420pToUyvy422(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, 
            const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* uyvy, size_t uyvyStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(uyvy) && Aligned(uyvyStride))
                Yuv420pToUyvy422<true>(y, yStride, u, uStride, v, vStride, width, height, uyvy, uyvyStride);
            else
                Yuv420pToUyvy422<false>(y, yStride, u, uStride, v, vStride, width, height, uyvy, uyvyStride);
        }
    }
#endif
}
