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
#include "Simd/SimdAlphaBlending.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svuint16_t DivideBy255(const svuint16_t& value)
        {
            const svbool_t mask = svptrue_b16();
            return svlsr_n_u16_x(mask, svadd_u16_x(mask, svadd_u16_x(mask, value, svdup_n_u16(1)), svlsr_n_u16_x(mask, value, 8)), 8);
        }

        SIMD_INLINE svuint8_t AlphaBlending(const svuint8_t& src, const svuint8_t& dst, const svuint8_t& alpha)
        {
            const svbool_t mask = svptrue_b16();
            svuint16_t alphaLo = svmovlb_u16(alpha);
            svuint16_t alphaHi = svmovlt_u16(alpha);
            svuint16_t invAlphaLo = svsub_u16_x(mask, svdup_n_u16(0xFF), alphaLo);
            svuint16_t invAlphaHi = svsub_u16_x(mask, svdup_n_u16(0xFF), alphaHi);
            svuint16_t lo = svadd_u16_x(mask, svmul_u16_x(mask, svmovlb_u16(src), alphaLo), svmul_u16_x(mask, svmovlb_u16(dst), invAlphaLo));
            svuint16_t hi = svadd_u16_x(mask, svmul_u16_x(mask, svmovlt_u16(src), alphaHi), svmul_u16_x(mask, svmovlt_u16(dst), invAlphaHi));
            return svqxtnt_u16(svqxtnb_u16(DivideBy255(lo)), DivideBy255(hi));
        }

        template<size_t channelCount> struct AlphaBlender
        {
            void operator()(const uint8_t* src, uint8_t* dst, const svuint8_t& alpha, const svbool_t& mask);
        };

        template<> struct AlphaBlender<1>
        {
            SIMD_INLINE void operator()(const uint8_t* src, uint8_t* dst, const svuint8_t& alpha, const svbool_t& mask)
            {
                svst1_u8(mask, dst, AlphaBlending(svld1_u8(mask, src), svld1_u8(mask, dst), alpha));
            }
        };

        template<> struct AlphaBlender<2>
        {
            SIMD_INLINE void operator()(const uint8_t* src, uint8_t* dst, const svuint8_t& alpha, const svbool_t& mask)
            {
                svuint8x2_t _src = svld2_u8(mask, src);
                svuint8x2_t _dst = svld2_u8(mask, dst);
                svst2_u8(mask, dst, svcreate2_u8(
                    AlphaBlending(svget2(_src, 0), svget2(_dst, 0), alpha),
                    AlphaBlending(svget2(_src, 1), svget2(_dst, 1), alpha)));
            }
        };

        template<> struct AlphaBlender<3>
        {
            SIMD_INLINE void operator()(const uint8_t* src, uint8_t* dst, const svuint8_t& alpha, const svbool_t& mask)
            {
                svuint8x3_t _src = svld3_u8(mask, src);
                svuint8x3_t _dst = svld3_u8(mask, dst);
                svst3_u8(mask, dst, svcreate3_u8(
                    AlphaBlending(svget3(_src, 0), svget3(_dst, 0), alpha),
                    AlphaBlending(svget3(_src, 1), svget3(_dst, 1), alpha),
                    AlphaBlending(svget3(_src, 2), svget3(_dst, 2), alpha)));
            }
        };

        template<> struct AlphaBlender<4>
        {
            SIMD_INLINE void operator()(const uint8_t* src, uint8_t* dst, const svuint8_t& alpha, const svbool_t& mask)
            {
                svuint8x4_t _src = svld4_u8(mask, src);
                svuint8x4_t _dst = svld4_u8(mask, dst);
                svst4_u8(mask, dst, svcreate4_u8(
                    AlphaBlending(svget4(_src, 0), svget4(_dst, 0), alpha),
                    AlphaBlending(svget4(_src, 1), svget4(_dst, 1), alpha),
                    AlphaBlending(svget4(_src, 2), svget4(_dst, 2), alpha),
                    AlphaBlending(svget4(_src, 3), svget4(_dst, 3), alpha)));
            }
        };

        template<size_t channelCount> void AlphaBlending(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* alpha, size_t alphaStride, uint8_t* dst, size_t dstStride)
        {
            size_t A = svlen(svuint8_t()), widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A * channelCount)
                    AlphaBlender<channelCount>()(src + offset, dst + offset, svld1_u8(body, alpha + col), body);
                if (widthA < width)
                    AlphaBlender<channelCount>()(src + offset, dst + offset, svld1_u8(tail, alpha + col), tail);
                src += srcStride;
                alpha += alphaStride;
                dst += dstStride;
            }
        }

        void AlphaBlending(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t* alpha, size_t alphaStride, uint8_t* dst, size_t dstStride)
        {
            assert(channelCount >= 1 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: AlphaBlending<1>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 2: AlphaBlending<2>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 3: AlphaBlending<3>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 4: AlphaBlending<4>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            }
        }
    }
#endif
}
