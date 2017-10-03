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
        const v128_u8 K8_PERM_U = SIMD_VEC_SETR_EPI8(0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E);
        const v128_u8 K8_PERM_V = SIMD_VEC_SETR_EPI8(0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F, 0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F);

        template<bool align, bool first>
        SIMD_INLINE void DeinterleavedUv(const Loader<align> & uv, Storer<align> & u, Storer<align> & v)
        {
            v128_u8 _uv0 = Load<align, first>(uv);
            v128_u8 _uv1 = Load<align, false>(uv);

            Store<align, first>(u, vec_perm(_uv0, _uv1, K8_PERM_U));
            Store<align, first>(v, vec_perm(_uv0, _uv1, K8_PERM_V));
        }

        template <bool align> void DeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride));

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _uv(uv);
                Storer<align> _u(u), _v(v);
                DeinterleavedUv<align, true>(_uv, _u, _v);
                for (size_t col = A; col < alignedWidth; col += A)
                    DeinterleavedUv<align, false>(_uv, _u, _v);
                Flush(_u, _v);

                if (width != alignedWidth)
                {
                    Loader<false> _uv(uv + 2 * (width - A));
                    Storer<false> _u(u + width - A), _v(v + width - A);
                    DeinterleavedUv<false, true>(_uv, _u, _v);
                    Flush(_u, _v);
                }

                uv += uvStride;
                u += uStride;
                v += vStride;
            }
        }

        void DeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride))
                DeinterleaveUv<true>(uv, uvStride, width, height, u, uStride, v, vStride);
            else
                DeinterleaveUv<false>(uv, uvStride, width, height, u, uStride, v, vStride);
        }
    }
#endif// SIMD_VMX_ENABLE
}
