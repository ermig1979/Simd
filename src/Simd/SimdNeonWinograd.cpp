/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdWinograd.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE  
    namespace Neon
    {
        SIMD_INLINE void Load4(const float * src, size_t step, float32x4_t * dst)
        {
            float32x4_t a0 = Load<false>(src + 0 * step);
            float32x4_t a1 = Load<false>(src + 1 * step);
            float32x4_t a2 = Load<false>(src + 2 * step);
            float32x4_t a3 = Load<false>(src + 3 * step);
            float32x4x2_t b0 = vzipq_f32(a0, a2);
            float32x4x2_t b1 = vzipq_f32(a1, a3);
            *(float32x4x2_t*)(dst + 0) = vzipq_f32(b0.val[0], b1.val[0]);
            *(float32x4x2_t*)(dst + 2) = vzipq_f32(b0.val[1], b1.val[1]);
        }

        SIMD_INLINE void Winograd2x3SetFilter4n(const float * src, float * dst, size_t stride)
        {
            const float32x4_t r2 = vdupq_n_f32(1.0f / 2.0f);
            const float32x4_t r4 = vdupq_n_f32(1.0f / 4.0f);

            float32x4_t s[9];
            Load4(src + 0, 9, s + 0);
            Load4(src + 4, 9, s + 4);
            s[8] = SetF32(src[8], src[17], src[26], src[35]);

            Store<false>(dst + 0 * stride, s[0]);
            float32x4_t _0a2 = vaddq_f32(s[0], s[2]);
            Store<false>(dst + 1 * stride, vmulq_f32(vaddq_f32(_0a2, s[1]), r2));
            Store<false>(dst + 2 * stride, vmulq_f32(vsubq_f32(_0a2, s[1]), r2));
            Store<false>(dst + 3 * stride, s[2]);

            float32x4_t _0a6a3 = vaddq_f32(vaddq_f32(s[0], s[6]), s[3]);
            Store<false>(dst + 4 * stride, vmulq_f32(_0a6a3, r2));
            float32x4_t _2a8a5 = vaddq_f32(vaddq_f32(s[2], s[8]), s[5]);
            float32x4_t _1a7a4 = vaddq_f32(vaddq_f32(s[1], s[7]), s[4]);
            Store<false>(dst + 5 * stride, vmulq_f32(vaddq_f32(vaddq_f32(_0a6a3, _2a8a5), _1a7a4), r4));
            Store<false>(dst + 6 * stride, vmulq_f32(vsubq_f32(vaddq_f32(_0a6a3, _2a8a5), _1a7a4), r4));
            Store<false>(dst + 7 * stride, vmulq_f32(_2a8a5, r2));

            float32x4_t _0a6s3 = vsubq_f32(vaddq_f32(s[0], s[6]), s[3]);
            Store<false>(dst + 8 * stride, vmulq_f32(_0a6s3, r2));
            float32x4_t _2a8s5 = vsubq_f32(vaddq_f32(s[2], s[8]), s[5]);
            float32x4_t _1a7s4 = vsubq_f32(vaddq_f32(s[1], s[7]), s[4]);
            Store<false>(dst + 9 * stride, vmulq_f32(vaddq_f32(vaddq_f32(_0a6s3, _2a8s5), _1a7s4), r4));
            Store<false>(dst + 10 * stride, vmulq_f32(vsubq_f32(vaddq_f32(_0a6s3, _2a8s5), _1a7s4), r4));
            Store<false>(dst + 11 * stride, vmulq_f32(_2a8s5, r2));

            Store<false>(dst + 12 * stride, s[6]);
            float32x4_t _6a8 = vaddq_f32(s[6], s[8]);
            Store<false>(dst + 13 * stride, vmulq_f32(vaddq_f32(_6a8, s[7]), r2));
            Store<false>(dst + 14 * stride, vmulq_f32(vsubq_f32(_6a8, s[7]), r2));
            Store<false>(dst + 15 * stride, s[8]);
        }

        SIMD_INLINE void Winograd2x3SetFilter4t(const float * src, float * dst, size_t stride)
        {
            const float32x4_t r2 = vdupq_n_f32(1.0f / 2.0f);
            const float32x4_t r4 = vdupq_n_f32(1.0f / 4.0f);

            float32x4_t s[9];
            s[0] = Load<false>(src + 0 * stride);
            s[1] = Load<false>(src + 1 * stride);
            s[2] = Load<false>(src + 2 * stride);
            s[3] = Load<false>(src + 3 * stride);
            s[4] = Load<false>(src + 4 * stride);
            s[5] = Load<false>(src + 5 * stride);
            s[6] = Load<false>(src + 6 * stride);
            s[7] = Load<false>(src + 7 * stride);
            s[8] = Load<false>(src + 8 * stride);

            Store<false>(dst + 0 * stride, s[0]);
            float32x4_t _0a2 = vaddq_f32(s[0], s[2]);
            Store<false>(dst + 1 * stride, vmulq_f32(vaddq_f32(_0a2, s[1]), r2));
            Store<false>(dst + 2 * stride, vmulq_f32(vsubq_f32(_0a2, s[1]), r2));
            Store<false>(dst + 3 * stride, s[2]);

            float32x4_t _0a6a3 = vaddq_f32(vaddq_f32(s[0], s[6]), s[3]);
            Store<false>(dst + 4 * stride, vmulq_f32(_0a6a3, r2));
            float32x4_t _2a8a5 = vaddq_f32(vaddq_f32(s[2], s[8]), s[5]);
            float32x4_t _1a7a4 = vaddq_f32(vaddq_f32(s[1], s[7]), s[4]);
            Store<false>(dst + 5 * stride, vmulq_f32(vaddq_f32(vaddq_f32(_0a6a3, _2a8a5), _1a7a4), r4));
            Store<false>(dst + 6 * stride, vmulq_f32(vsubq_f32(vaddq_f32(_0a6a3, _2a8a5), _1a7a4), r4));
            Store<false>(dst + 7 * stride, vmulq_f32(_2a8a5, r2));

            float32x4_t _0a6s3 = vsubq_f32(vaddq_f32(s[0], s[6]), s[3]);
            Store<false>(dst + 8 * stride, vmulq_f32(_0a6s3, r2));
            float32x4_t _2a8s5 = vsubq_f32(vaddq_f32(s[2], s[8]), s[5]);
            float32x4_t _1a7s4 = vsubq_f32(vaddq_f32(s[1], s[7]), s[4]);
            Store<false>(dst + 9 * stride, vmulq_f32(vaddq_f32(vaddq_f32(_0a6s3, _2a8s5), _1a7s4), r4));
            Store<false>(dst + 10 * stride, vmulq_f32(vsubq_f32(vaddq_f32(_0a6s3, _2a8s5), _1a7s4), r4));
            Store<false>(dst + 11 * stride, vmulq_f32(_2a8s5, r2));

            Store<false>(dst + 12 * stride, s[6]);
            float32x4_t _6a8 = vaddq_f32(s[6], s[8]);
            Store<false>(dst + 13 * stride, vmulq_f32(vaddq_f32(_6a8, s[7]), r2));
            Store<false>(dst + 14 * stride, vmulq_f32(vsubq_f32(_6a8, s[7]), r2));
            Store<false>(dst + 15 * stride, s[8]);
}

        void Winograd2x3SetFilter(const float * src, size_t size, float * dst, SimdBool trans)
        {
            size_t size4 = AlignLo(size, 4), i = 0;
            if (trans)
            {
                for (; i < size4; i += 4)
                    Winograd2x3SetFilter4t(src + i, dst + i, size);
                for (; i < size; i += 1)
                    Base::Winograd2x3SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                for (; i < size4; i += 4, src += 36, dst += 4)
                    Winograd2x3SetFilter4n(src, dst, size);
                for (; i < size; i += 1, src += 9, dst += 1)
                    Base::Winograd2x3SetFilter1n(src, dst, size);
            }
        }

        SIMD_INLINE void Winograd4x3SetFilter4Row(const float32x4_t * t, float * dst, size_t stride)
        {
            const float32x4_t r4 = vdupq_n_f32(1.0f / 4.0f);
            const float32x4_t r6 = vdupq_n_f32(1.0f / 6.0f);
            const float32x4_t mr6 = vdupq_n_f32(-1.0f / 6.0f);
            const float32x4_t r12 = vdupq_n_f32(1.0f / 12.0f);
            const float32x4_t r24 = vdupq_n_f32(1.0f / 24.0f);
            Store<false>(dst + 0 * stride, vmulq_f32(r4, t[0]));
            float32x4_t t0 = vaddq_f32(t[0], t[2]);
            Store<false>(dst + 1 * stride, vmulq_f32(mr6, vaddq_f32(t0, t[1])));
            Store<false>(dst + 2 * stride, vmulq_f32(mr6, vsubq_f32(t0, t[1])));
            float32x4_t t1 = vaddq_f32(vmulq_f32(r24, t[0]), vmulq_f32(r6, t[2]));
            float32x4_t t2 = vmulq_f32(r12, t[1]);
            Store<false>(dst + 3 * stride, vaddq_f32(t1, t2));
            Store<false>(dst + 4 * stride, vsubq_f32(t1, t2));
            Store<false>(dst + 5 * stride, t[2]);
        }

        SIMD_INLINE void Winograd4x3SetFilter4All(const float32x4_t * s, float * dst, size_t stride)
        {
            const float32x4_t r4 = vdupq_n_f32(1.0f / 4.0f);
            const float32x4_t r6 = vdupq_n_f32(1.0f / 6.0f);
            const float32x4_t mr6 = vdupq_n_f32(-1.0f / 6.0f);
            const float32x4_t r12 = vdupq_n_f32(1.0f / 12.0f);
            const float32x4_t r24 = vdupq_n_f32(1.0f / 24.0f);

            float32x4_t t[3];
            t[0] = vmulq_f32(r4, s[0]);
            t[1] = vmulq_f32(r4, s[1]);
            t[2] = vmulq_f32(r4, s[2]);
            Winograd4x3SetFilter4Row(t, dst + 0 * stride, stride);

            t[0] = vmulq_f32(mr6, vaddq_f32(vaddq_f32(s[0], s[3]), s[6]));
            t[1] = vmulq_f32(mr6, vaddq_f32(vaddq_f32(s[1], s[4]), s[7]));
            t[2] = vmulq_f32(mr6, vaddq_f32(vaddq_f32(s[2], s[5]), s[8]));
            Winograd4x3SetFilter4Row(t, dst + 6 * stride, stride);

            t[0] = vmulq_f32(mr6, vaddq_f32(vsubq_f32(s[0], s[3]), s[6]));
            t[1] = vmulq_f32(mr6, vaddq_f32(vsubq_f32(s[1], s[4]), s[7]));
            t[2] = vmulq_f32(mr6, vaddq_f32(vsubq_f32(s[2], s[5]), s[8]));
            Winograd4x3SetFilter4Row(t, dst + 12 * stride, stride);

            t[0] = vaddq_f32(vaddq_f32(vmulq_f32(r24, s[0]), vmulq_f32(r12, s[3])), vmulq_f32(r6, s[6]));
            t[1] = vaddq_f32(vaddq_f32(vmulq_f32(r24, s[1]), vmulq_f32(r12, s[4])), vmulq_f32(r6, s[7]));
            t[2] = vaddq_f32(vaddq_f32(vmulq_f32(r24, s[2]), vmulq_f32(r12, s[5])), vmulq_f32(r6, s[8]));
            Winograd4x3SetFilter4Row(t, dst + 18 * stride, stride);

            t[0] = vaddq_f32(vsubq_f32(vmulq_f32(r24, s[0]), vmulq_f32(r12, s[3])), vmulq_f32(r6, s[6]));
            t[1] = vaddq_f32(vsubq_f32(vmulq_f32(r24, s[1]), vmulq_f32(r12, s[4])), vmulq_f32(r6, s[7]));
            t[2] = vaddq_f32(vsubq_f32(vmulq_f32(r24, s[2]), vmulq_f32(r12, s[5])), vmulq_f32(r6, s[8]));
            Winograd4x3SetFilter4Row(t, dst + 24 * stride, stride);

            Winograd4x3SetFilter4Row(s + 6, dst + 30 * stride, stride);
        }


        SIMD_INLINE void Winograd4x3SetFilter4n(const float * src, float * dst, size_t stride)
        {
            float32x4_t s[9];
            Load4(src + 0, 9, s + 0);
            Load4(src + 4, 9, s + 4);
            s[8] = SetF32(src[8], src[17], src[26], src[35]);
            Winograd4x3SetFilter4All(s, dst + 0 * stride, stride);
        }

        SIMD_INLINE void Winograd4x3SetFilter4t(const float * src, float * dst, size_t stride)
        {
            float32x4_t s[9];
            s[0] = Load<false>(src + 0 * stride);
            s[1] = Load<false>(src + 1 * stride);
            s[2] = Load<false>(src + 2 * stride);
            s[3] = Load<false>(src + 3 * stride);
            s[4] = Load<false>(src + 4 * stride);
            s[5] = Load<false>(src + 5 * stride);
            s[6] = Load<false>(src + 6 * stride);
            s[7] = Load<false>(src + 7 * stride);
            s[8] = Load<false>(src + 8 * stride);
            Winograd4x3SetFilter4All(s, dst + 0 * stride, stride);
        }

        void Winograd4x3SetFilter(const float * src, size_t size, float * dst, SimdBool trans)
        {
            size_t size4 = AlignLo(size, 4), i = 0;
            if (trans)
            {
                for (; i < size4; i += 4)
                    Winograd4x3SetFilter4t(src + i, dst + i, size);
                for (; i < size; i += 1)
                    Base::Winograd4x3SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                for (; i < size4; i += 4, src += 36, dst += 4)
                    Winograd4x3SetFilter4n(src, dst, size);
                for (; i < size; i += 1, src += 9, dst += 1)
                    Base::Winograd4x3SetFilter1n(src, dst, size);
            }
        }
    }
#endif// SIMD_NEON_ENABLE
}
