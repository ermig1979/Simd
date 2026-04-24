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
#include "Simd/SimdCompare.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdNeon.h"

namespace Simd
{
#ifdef SIMD_SVE_ENABLE  
    namespace Sve
    {
        template <SimdOperationBinary8uType type> SIMD_INLINE svuint8_t OperationBinary8u(const svbool_t & mask, const svuint8_t & a, const svuint8_t & b);

        //template <> SIMD_INLINE svuint8_t OperationBinary8u<SimdOperationBinary8uAverage>(const svbool_t& mask, const svuint8_t & a, const svuint8_t & b)
        //{
        //    return svrhadd_x(mask, a, b);
        //}

        template <> SIMD_INLINE svuint8_t OperationBinary8u<SimdOperationBinary8uAnd>(const svbool_t& mask, const svuint8_t & a, const svuint8_t & b)
        {
            return svand_x(mask, a, b);
        }

        template <> SIMD_INLINE svuint8_t OperationBinary8u<SimdOperationBinary8uOr>(const svbool_t& mask, const svuint8_t & a, const svuint8_t & b)
        {
            return svorr_x(mask, a, b);
        }

        template <> SIMD_INLINE svuint8_t OperationBinary8u<SimdOperationBinary8uMaximum>(const svbool_t& mask, const svuint8_t & a, const svuint8_t & b)
        {
            return svmax_x(mask, a, b);
        }

        template <> SIMD_INLINE svuint8_t OperationBinary8u<SimdOperationBinary8uMinimum>(const svbool_t& mask, const svuint8_t & a, const svuint8_t & b)
        {
            return svmin_x(mask, a, b);
        }

        template <> SIMD_INLINE svuint8_t OperationBinary8u<SimdOperationBinary8uSaturatedSubtraction>(const svbool_t& mask, const svuint8_t & a, const svuint8_t & b)
        {
            return svqsub(a, b);
        }

        template <> SIMD_INLINE svuint8_t OperationBinary8u<SimdOperationBinary8uSaturatedAddition>(const svbool_t& mask, const svuint8_t & a, const svuint8_t & b)
        {
            return svqadd(a, b);
        }

        template <SimdOperationBinary8uType type> void OperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            size_t A = svlen(svuint8_t());
            size_t size = channelCount*width;
            size_t sizeA = Simd::AlignLo(size, A);
            const svbool_t body = svwhilelt_b8(size_t(0), A);
            const svbool_t tail = svwhilelt_b8(sizeA, size);
            for (size_t row = 0; row < height; ++row)
            {
                size_t offset = 0;
                for (; offset < sizeA; offset += A)
                {
                    svuint8_t _a = svld1_u8(body, a + offset);
                    svuint8_t _b = svld1_u8(body, b + offset);
                    svst1_u8(body, dst + offset, OperationBinary8u<type>(body, _a, _b));
                }
                if (sizeA < size)
                {
                    svuint8_t _a = svld1_u8(tail, a + offset);
                    svuint8_t _b = svld1_u8(tail, b + offset);
                    svst1_u8(tail, dst + offset, OperationBinary8u<type>(tail, _a, _b));
                }
                a += aStride;
                b += bStride;
                dst += dstStride;
            }
        }

        void OperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type)
        {
            switch (type)
            {
            case SimdOperationBinary8uAverage:
                if(channelCount * width >= Neon::A)
                    return Neon::OperationBinary8u(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, SimdOperationBinary8uAverage);
                else
                    return Base::OperationBinary8u(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, SimdOperationBinary8uAverage);
            case SimdOperationBinary8uAnd:
                return OperationBinary8u<SimdOperationBinary8uAnd>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uOr:
                return OperationBinary8u<SimdOperationBinary8uOr>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uMaximum:
                return OperationBinary8u<SimdOperationBinary8uMaximum>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uMinimum:
                return OperationBinary8u<SimdOperationBinary8uMinimum>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uSaturatedSubtraction:
                return OperationBinary8u<SimdOperationBinary8uSaturatedSubtraction>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uSaturatedAddition:
                return OperationBinary8u<SimdOperationBinary8uSaturatedAddition>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <SimdOperationBinary16iType type> SIMD_INLINE svint16_t OperationBinary16i(const svbool_t& mask, const svint16_t& a, const svint16_t& b);

        template <> SIMD_INLINE svint16_t OperationBinary16i<SimdOperationBinary16iAddition>(const svbool_t& mask, const svint16_t& a, const svint16_t& b)
        {
            return svadd_x(mask, a, b);
        }

        template <> SIMD_INLINE svint16_t OperationBinary16i<SimdOperationBinary16iSubtraction>(const svbool_t& mask, const svint16_t& a, const svint16_t& b)
        {
            return svsub_x(mask, a, b);
        }

        template <SimdOperationBinary16iType type> void OperationBinary16i(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            size_t A = svlen(svint16_t());
            size_t widthA = Simd::AlignLo(width, A);
            const svbool_t body = svwhilelt_b16(size_t(0), A);
            const svbool_t tail = svwhilelt_b16(widthA, width);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                {
                    const svint16_t _a = svld1_s16(body, (int16_t*)a + col);
                    const svint16_t _b = svld1_s16(body, (int16_t*)b + col);
                    svst1_s16(body, (int16_t*)dst + col, OperationBinary16i<type>(body, _a, _b));
                }
                if (widthA < width)
                {
                    const svint16_t _a = svld1_s16(tail, (int16_t*)a + col);
                    const svint16_t _b = svld1_s16(tail, (int16_t*)b + col);
                    svst1_s16(tail, (int16_t*)dst + col, OperationBinary16i<type>(tail, _a, _b));
                }
                a += aStride;
                b += bStride;
                dst += dstStride;
            }
        }

        void OperationBinary16i(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride, size_t width, size_t height, uint8_t* dst, size_t dstStride, SimdOperationBinary16iType type)
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
    }
#endif
}
