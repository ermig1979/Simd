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

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        template <SimdOperationBinary16iType type> SIMD_INLINE svint16_t OperationBinary16i(const svbool_t& mask, const svint16_t& a, const svint16_t& b);

        template <> SIMD_INLINE svint16_t OperationBinary16i<SimdOperationBinary16iAddition>(const svbool_t& mask, const svint16_t& a, const svint16_t& b)
        {
            return svadd_s16_x(mask, a, b);
        }

        template <> SIMD_INLINE svint16_t OperationBinary16i<SimdOperationBinary16iSubtraction>(const svbool_t& mask, const svint16_t& a, const svint16_t& b)
        {
            return svsub_s16_x(mask, a, b);
        }

        template <SimdOperationBinary16iType type> void OperationBinary16i(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            size_t A = svlen(svint16_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b16();
            const svbool_t tail = svwhilelt_b16(widthA, width);
            for (size_t row = 0; row < height; ++row)
            {
                const int16_t* pa = (const int16_t*)a;
                const int16_t* pb = (const int16_t*)b;
                int16_t* pd = (int16_t*)dst;
                size_t col = 0;
                for (; col < widthA; col += A)
                {
                    svint16_t _a = svld1_s16(body, pa + col);
                    svint16_t _b = svld1_s16(body, pb + col);
                    svst1_s16(body, pd + col, OperationBinary16i<type>(body, _a, _b));
                }
                if (widthA < width)
                {
                    svint16_t _a = svld1_s16(tail, pa + col);
                    svint16_t _b = svld1_s16(tail, pb + col);
                    svst1_s16(tail, pd + col, OperationBinary16i<type>(tail, _a, _b));
                }
                a += aStride;
                b += bStride;
                dst += dstStride;
            }
        }

        void OperationBinary16i(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            size_t width, size_t height, uint8_t* dst, size_t dstStride, SimdOperationBinary16iType type)
        {
            switch (type)
            {
            case SimdOperationBinary16iAddition:
                return OperationBinary16i<SimdOperationBinary16iAddition>(a, aStride, b, bStride, width, height, dst, dstStride);
            case SimdOperationBinary16iSubtraction:
                return OperationBinary16i<SimdOperationBinary16iSubtraction>(a, aStride, b, bStride, width, height, dst, dstStride);
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svuint8_t DivideI16By255(const svuint16_t & value)
        {
            const svbool_t full = svptrue_b16();
            svuint16_t sum = svadd_n_u16_x(full, value, 1);
            sum = svadd_u16_x(full, sum, svlsr_n_u16_x(full, sum, 8));
            return svqxtnb_u16(svlsr_n_u16_x(full, sum, 8));
        }

        SIMD_INLINE svuint8_t VectorProduct(const svuint16_t & vertical, const svuint8_t & horizontal)
        {
            const svbool_t full = svptrue_b16();
            svuint8_t lo = DivideI16By255(svmul_u16_x(full, vertical, svunpklo_u16(horizontal)));
            svuint8_t hi = DivideI16By255(svmul_u16_x(full, vertical, svunpkhi_u16(horizontal)));
            return svuzp1_u8(lo, hi);
        }

        void VectorProduct(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
        {
            size_t A = svcntb();
            size_t widthA = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                const svuint16_t _vertical = svdup_n_u16(vertical[row]);
                size_t col = 0;
                for (; col < widthA; col += A)
                {
                    svuint8_t _horizontal = svld1_u8(svptrue_b8(), horizontal + col);
                    svst1_u8(svptrue_b8(), dst + col, VectorProduct(_vertical, _horizontal));
                }
                if (widthA < width)
                {
                    svbool_t tail = svwhilelt_b8(col, width);
                    svuint8_t _horizontal = svld1_u8(tail, horizontal + col);
                    svst1_u8(tail, dst + col, VectorProduct(_vertical, _horizontal));
                }
                dst += stride;
            }
        }
    }
#endif
}
