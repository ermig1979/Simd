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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
	namespace Neon
	{
        template<bool align> SIMD_INLINE void Float32ToFloat16(const float * src, uint16_t * dst)
        {
            Store<align>(dst, (uint16x4_t)vcvt_f16_f32(Load<align>(src)));
        }

        template <bool align> void Float32ToFloat16(const float * src, size_t size, uint16_t * dst)
        {
            assert(size >= F);
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            size_t partialAlignedSize = Simd::AlignLo(size, F);

            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
            {
                Float32ToFloat16<align>(src + i + F * 0, dst + i + F * 0);
                Float32ToFloat16<align>(src + i + F * 1, dst + i + F * 1);
                Float32ToFloat16<align>(src + i + F * 2, dst + i + F * 2);
                Float32ToFloat16<align>(src + i + F * 3, dst + i + F * 3);
            }
            for (; i < partialAlignedSize; i += F)
                Float32ToFloat16<align>(src + i, dst + i);
            if (partialAlignedSize != size)
                Float32ToFloat16<false>(src + size - F, dst + size - F);
        }

        void Float32ToFloat16(const float * src, size_t size, uint16_t * dst)
		{
			if (Aligned(src) && Aligned(dst))
                Float32ToFloat16<true>(src, size, dst);
			else
                Float32ToFloat16<false>(src, size, dst);
		}
	}
#endif // defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
}
