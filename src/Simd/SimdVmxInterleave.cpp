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
        const v128_u8 K8_PERM_UV0 = SIMD_VEC_SETR_EPI8(0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 0x04, 0x14, 0x05, 0x15, 0x06, 0x16, 0x07, 0x17);
        const v128_u8 K8_PERM_UV1 = SIMD_VEC_SETR_EPI8(0x08, 0x18, 0x09, 0x19, 0x0A, 0x1A, 0x0B, 0x1B, 0x0C, 0x1C, 0x0D, 0x1D, 0x0E, 0x1E, 0x0F, 0x1F);

        template<bool align, bool first>
        SIMD_INLINE void InterleavedUv(const Loader<align> & u, const Loader<align> & v, Storer<align> & uv)
        {
            v128_u8 _u = Load<align, first>(u);
            v128_u8 _v = Load<align, first>(v);

            Store<align, first>(uv, vec_perm(_u, _v, K8_PERM_UV0));
            Store<align, false>(uv, vec_perm(_u, _v, K8_PERM_UV1));
        }

        template <bool align> void InterleaveUv(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * uv, size_t uvStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride));

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _u(u), _v(v);
                Storer<align> _uv(uv);
                InterleavedUv<align, true>(_u, _v, _uv);
                for (size_t col = A; col < alignedWidth; col += A)
                    InterleavedUv<align, false>(_u, _v, _uv);
                Flush(_uv);

                if (width != alignedWidth)
                {
                    Loader<false> _u(u + width - A), _v(v + width - A);
                    Storer<false> _uv(uv + 2 * (width - A));
                    InterleavedUv<false, true>(_u, _v, _uv);
                    Flush(_uv);
                }

                u += uStride;
                v += vStride;
                uv += uvStride;
            }
        }

        void InterleaveUv(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * uv, size_t uvStride)
        {
            if (Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride))
                InterleaveUv<true>(u, uStride, v, vStride, width, height, uv, uvStride);
            else
                InterleaveUv<false>(u, uStride, v, vStride, width, height, uv, uvStride);
        }
    }
#endif// SIMD_VMX_ENABLE
}
