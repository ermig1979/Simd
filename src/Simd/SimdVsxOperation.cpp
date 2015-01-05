/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
#include "Simd/SimdVsx.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE  
    namespace Vsx
    {
        template <SimdOperationBinary8uType type> SIMD_INLINE v128_u8 OperationBinary8u(const v128_u8 & a, const v128_u8 & b);

        template <> SIMD_INLINE v128_u8 OperationBinary8u<SimdOperationBinary8uAverage>(const v128_u8 & a, const v128_u8 & b)
        {
            return vec_avg(a, b);
        }

        template <> SIMD_INLINE v128_u8 OperationBinary8u<SimdOperationBinary8uAnd>(const v128_u8 & a, const v128_u8 & b)
        {
            return vec_and(a, b);
        }

        template <> SIMD_INLINE v128_u8 OperationBinary8u<SimdOperationBinary8uMaximum>(const v128_u8 & a, const v128_u8 & b)
        {
            return vec_max(a, b);
        }

        template <> SIMD_INLINE v128_u8 OperationBinary8u<SimdOperationBinary8uSaturatedSubtraction>(const v128_u8 & a, const v128_u8 & b)
        {
            return vec_subs(a, b);
        }

        template <> SIMD_INLINE v128_u8 OperationBinary8u<SimdOperationBinary8uSaturatedAddition>(const v128_u8 & a, const v128_u8 & b)
        {
            return vec_adds(a, b);
        }

        template <SimdOperationBinary8uType type, bool align, bool first> 
        SIMD_INLINE void OperationBinary8u(const Loader<align> & a, const Loader<align> & b, Storer<align> & dst)
        {
            Store<align, first>(dst, OperationBinary8u<type>(Load<align, first>(a), Load<align, first>(b)));
        }

        template <bool align, SimdOperationBinary8uType type> void OperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, 
            size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            assert(width*channelCount >= A);
            if(align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride));

            size_t size = channelCount*width;
            size_t alignedSize = Simd::AlignLo(size, A);
            for(size_t row = 0; row < height; ++row)
            {
                Loader<align> _a(a), _b(b);
                Storer<align> _dst(dst);
                OperationBinary8u<type, align, true>(_a, _b, _dst);
                for(size_t offset = A; offset < alignedSize; offset += A)
                    OperationBinary8u<type, align, false>(_a, _b, _dst);
                Flush(_dst);

                if(alignedSize != size)
                {
                    Loader<false> _a(a + size - A), _b(b + size - A);
                    Storer<false> _dst(dst + size - A);
                    OperationBinary8u<type, false, true>(_a, _b, _dst);
                    Flush(_dst);
                }

                a += aStride;
                b += bStride;
                dst += dstStride;
            }
        }

        template <bool align> void OperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, 
            size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type)
        {
            switch(type)
            {
            case SimdOperationBinary8uAverage:
                return OperationBinary8u<align, SimdOperationBinary8uAverage>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uAnd:
                return OperationBinary8u<align, SimdOperationBinary8uAnd>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uMaximum:
                return OperationBinary8u<align, SimdOperationBinary8uMaximum>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uSaturatedSubtraction:
                return OperationBinary8u<align, SimdOperationBinary8uSaturatedSubtraction>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uSaturatedAddition:
                return OperationBinary8u<align, SimdOperationBinary8uSaturatedAddition>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            default:
                assert(0);
            }
        }

        void OperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, 
            size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type)
        {
            if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride))
                OperationBinary8u<true>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
            else
                OperationBinary8u<false>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
        }

        template <SimdOperationBinary16iType type> SIMD_INLINE v128_s16 OperationBinary16i(const v128_s16 & a, const v128_s16 & b);

        template <> SIMD_INLINE v128_s16 OperationBinary16i<SimdOperationBinary16iAddition>(const v128_s16 & a, const v128_s16 & b)
        {
            return vec_add(a, b);
        }

        template <SimdOperationBinary16iType type, bool align, bool first> 
        SIMD_INLINE void OperationBinary16i(const Loader<align> & a, const Loader<align> & b, Storer<align> & dst)
        {
            Store<align, first>(dst, OperationBinary16i<type>((v128_s16)Load<align, first>(a), (v128_s16)Load<align, first>(b)));
        }

        template <bool align, SimdOperationBinary16iType type> void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, 
            size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(width >= HA);
            if(align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = Simd::AlignLo(width, HA);
            for(size_t row = 0; row < height; ++row)
            {
                Loader<align> _a(a), _b(b);
                Storer<align> _dst(dst);
                OperationBinary16i<type, align, true>(_a, _b, _dst);
                for(size_t col = HA; col < alignedWidth; col += HA)
                    OperationBinary16i<type, align, false>(_a, _b, _dst);
                Flush(_dst);

                if(alignedWidth != width)
                {
                    size_t offset = 2*width - A;
                    Loader<false> _a(a + offset), _b(b + offset);
                    Storer<false> _dst(dst + offset);
                    OperationBinary16i<type, false, true>(_a, _b, _dst);
                    Flush(_dst);
                }

                a += aStride;
                b += bStride;
                dst += dstStride;
            }
        }

        template <bool align> void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, 
            size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type)
        {
            switch(type)
            {
            case SimdOperationBinary16iAddition:
                return OperationBinary16i<align, SimdOperationBinary16iAddition>(a, aStride, b, bStride, width, height, dst, dstStride);
            default:
                assert(0);
            }
        }

        void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, 
            size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type)
        {
            if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride))
                OperationBinary16i<true>(a, aStride, b, bStride, width, height, dst, dstStride, type);
            else
                OperationBinary16i<false>(a, aStride, b, bStride, width, height, dst, dstStride, type);
        }

        template <bool align, bool first> SIMD_INLINE void VectorProduct(const v128_u16 & vertical, const uint8_t * horizontal, Storer<align> & dst)
        {
            v128_u8 _horizontal = Load<align>(horizontal);
            v128_u16 lo = DivideBy255(vec_mladd(vertical, UnpackLoU8(_horizontal), K16_0000));
            v128_u16 hi = DivideBy255(vec_mladd(vertical, UnpackHiU8(_horizontal), K16_0000));
            Store<align, first>(dst, vec_pack(lo, hi));
        } 

        template <bool align> void VectorProduct(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(horizontal) && Aligned(dst) && Aligned(stride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            for(size_t row = 0; row < height; ++row)
            {
                v128_u16 _vertical = SetU16(vertical[row]);
                Storer<align> _dst(dst);
                VectorProduct<align, true>(_vertical, horizontal, _dst);
                for(size_t col = A; col < alignedWidth; col += A)
                    VectorProduct<align, false>(_vertical, horizontal + col, _dst);
                Flush(_dst);
                if(alignedWidth != width)
                {
                    Storer<false> _dst(dst + width - A);
                    VectorProduct<false, true>(_vertical, horizontal + width - A, _dst);
                    Flush(_dst);
                }
                dst += stride;
            }
        }

        void VectorProduct(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
        {
            if(Aligned(horizontal) && Aligned(dst) && Aligned(stride))
                VectorProduct<true>(vertical, horizontal, dst, stride, width, height);
            else
                VectorProduct<false>(vertical, horizontal, dst, stride, width, height);
        }
    }
#endif// SIMD_VSX_ENABLE
}