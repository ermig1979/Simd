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

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        const v128_u8 K8_PERM_REORDER_16 = SIMD_VEC_SETR_EPI8(0x1, 0x0, 0x3, 0x2, 0x5, 0x4, 0x7, 0x6, 0x9, 0x8, 0xB, 0xA, 0xD, 0xC, 0xF, 0xE);

        template <bool align, bool first> SIMD_INLINE void Reorder16bit(const uint8_t * src, Storer<align> & dst)
        {
            v128_u8 _src = Load<align>(src);
            Store<align, first>(dst, vec_perm(_src, _src, K8_PERM_REORDER_16));
        }

        template <bool align> void Reorder16bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size >= A && size % 2 == 0);

            size_t alignedSize = AlignLo(size, A);
            Storer<align> _dst(dst);
            Reorder16bit<align, true>(src, _dst);
            for (size_t i = A; i < alignedSize; i += A)
                Reorder16bit<align, false>(src + i, _dst);
            Flush(_dst);
            for (size_t i = alignedSize; i < size; i += 2)
                Base::Reorder16bit(src + i, dst + i);
        }

        void Reorder16bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Reorder16bit<true>(src, size, dst);
            else
                Reorder16bit<false>(src, size, dst);
        }

        const v128_u8 K8_PERM_REORDER_32 = SIMD_VEC_SETR_EPI8(0x3, 0x2, 0x1, 0x0, 0x7, 0x6, 0x5, 0x4, 0xB, 0xA, 0x9, 0x8, 0xF, 0xE, 0xD, 0xC);

        template <bool align, bool first> SIMD_INLINE void Reorder32bit(const uint8_t * src, Storer<align> & dst)
        {
            v128_u8 _src = Load<align>(src);
            Store<align, first>(dst, vec_perm(_src, _src, K8_PERM_REORDER_32));
        }

        template <bool align> void Reorder32bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size >= A && size % 4 == 0);

            size_t alignedSize = AlignLo(size, A);
            Storer<align> _dst(dst);
            Reorder32bit<align, true>(src, _dst);
            for (size_t i = A; i < alignedSize; i += A)
                Reorder32bit<align, false>(src + i, _dst);
            Flush(_dst);
            for (size_t i = alignedSize; i < size; i += 4)
                Base::Reorder32bit(src + i, dst + i);
        }

        void Reorder32bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Reorder32bit<true>(src, size, dst);
            else
                Reorder32bit<false>(src, size, dst);
        }

        const v128_u8 K8_PERM_REORDER_64 = SIMD_VEC_SETR_EPI8(0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0, 0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8);

        template <bool align, bool first> SIMD_INLINE void Reorder64bit(const uint8_t * src, Storer<align> & dst)
        {
            v128_u8 _src = Load<align>(src);
            Store<align, first>(dst, vec_perm(_src, _src, K8_PERM_REORDER_64));
        }

        template <bool align> void Reorder64bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size >= A && size % 8 == 0);

            size_t alignedSize = AlignLo(size, A);
            Storer<align> _dst(dst);
            Reorder64bit<align, true>(src, _dst);
            for (size_t i = A; i < alignedSize; i += A)
                Reorder64bit<align, false>(src + i, _dst);
            Flush(_dst);
            for (size_t i = alignedSize; i < size; i += 8)
                Base::Reorder64bit(src + i, dst + i);
        }

        void Reorder64bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Reorder64bit<true>(src, size, dst);
            else
                Reorder64bit<false>(src, size, dst);
        }
    }
#endif// SIMD_VMX_ENABLE
}
