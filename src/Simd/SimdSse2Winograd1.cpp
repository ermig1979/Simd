/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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

namespace Simd
{
#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Sse2
    {
        SIMD_INLINE void WinogradKernel1x3Block1x4SetFilter(const __m128 * t, float* dst, size_t stride)
        {
            const __m128 r4 = _mm_set1_ps(1.0f / 4.0f);
            const __m128 r6 = _mm_set1_ps(1.0f / 6.0f);
            const __m128 mr6 = _mm_set1_ps(-1.0f / 6.0f);
            const __m128 r12 = _mm_set1_ps(1.0f / 12.0f);
            const __m128 r24 = _mm_set1_ps(1.0f / 24.0f);
            _mm_storeu_ps(dst + 0 * stride, _mm_mul_ps(r4, t[0]));
            __m128 t0 = _mm_add_ps(t[0], t[2]);
            _mm_storeu_ps(dst + 1 * stride, _mm_mul_ps(mr6, _mm_add_ps(t0, t[1])));
            _mm_storeu_ps(dst + 2 * stride, _mm_mul_ps(mr6, _mm_sub_ps(t0, t[1])));
            __m128 t1 = _mm_add_ps(_mm_mul_ps(r24, t[0]), _mm_mul_ps(r6, t[2]));
            __m128 t2 = _mm_mul_ps(r12, t[1]);
            _mm_storeu_ps(dst + 3 * stride, _mm_add_ps(t1, t2));
            _mm_storeu_ps(dst + 4 * stride, _mm_sub_ps(t1, t2));
            _mm_storeu_ps(dst + 5 * stride, t[2]);
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetFilter4n(const float* src, float* dst, size_t stride)
        {
            __m128 s[3];
            s[0] = _mm_setr_ps(src[0], src[3], src[6], src[9]);
            s[1] = _mm_setr_ps(src[1], src[4], src[7], src[10]);
            s[2] = _mm_setr_ps(src[2], src[5], src[8], src[11]);
            WinogradKernel1x3Block1x4SetFilter(s, dst, stride);
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetFilter4t(const float* src, float* dst, size_t stride)
        {
            __m128 s[3];
            s[0] = _mm_loadu_ps(src + 0 * stride);
            s[1] = _mm_loadu_ps(src + 1 * stride);
            s[2] = _mm_loadu_ps(src + 2 * stride);
            WinogradKernel1x3Block1x4SetFilter(s, dst, stride);
        }

        void WinogradKernel1x3Block1x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
        {
            size_t size4 = AlignLo(size, 4), i = 0;
            if (trans)
            {
                for (; i < size4; i += 4)
                    WinogradKernel1x3Block1x4SetFilter4t(src + i, dst + i, size);
                for (; i < size; i += 1)
                    Base::WinogradKernel1x3Block1x4SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                for (; i < size4; i += 4, src += 12, dst += 4)
                    WinogradKernel1x3Block1x4SetFilter4n(src, dst, size);
                for (; i < size; i += 1, src += 3, dst += 1)
                    Base::WinogradKernel1x3Block1x4SetFilter1n(src, dst, size);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel1x3Block1x4SetInput4Store(const __m128 src[6], float* dst, size_t stride)
        {
            __m128 _2 = _mm_set1_ps(2.0f);
            __m128 _4 = _mm_set1_ps(4.0f);
            __m128 _5 = _mm_set1_ps(5.0f);
            _mm_storeu_ps(dst + 0 * stride, _mm_add_ps(_mm_sub_ps(_mm_mul_ps(_4, src[0]), _mm_mul_ps(_5, src[2])), src[4]));
            _mm_storeu_ps(dst + 1 * stride, _mm_sub_ps(_mm_add_ps(src[3], src[4]), _mm_mul_ps(_4, _mm_add_ps(src[1], src[2]))));
            _mm_storeu_ps(dst + 2 * stride, _mm_add_ps(_mm_mul_ps(_4, _mm_sub_ps(src[1], src[2])), _mm_sub_ps(src[4], src[3])));
            _mm_storeu_ps(dst + 3 * stride, _mm_add_ps(_mm_mul_ps(_2, _mm_sub_ps(src[3], src[1])), _mm_sub_ps(src[4], src[2])));
            _mm_storeu_ps(dst + 4 * stride, _mm_add_ps(_mm_mul_ps(_2, _mm_sub_ps(src[1], src[3])), _mm_sub_ps(src[4], src[2])));
            _mm_storeu_ps(dst + 5 * stride, _mm_add_ps(_mm_sub_ps(_mm_mul_ps(_4, src[1]), _mm_mul_ps(_5, src[3])), src[5]));
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetInput4t(const float* src, size_t srcC, __m128 dst[6])
        {
            dst[0] = _mm_loadu_ps(src + 0 * srcC);
            dst[1] = _mm_loadu_ps(src + 1 * srcC);
            dst[2] = _mm_loadu_ps(src + 2 * srcC);
            dst[3] = _mm_loadu_ps(src + 3 * srcC);
            dst[4] = _mm_loadu_ps(src + 4 * srcC);
            dst[5] = _mm_loadu_ps(src + 5 * srcC);
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetInput4t(const float* src, size_t srcC, float* dst, size_t dstStride)
        {
            size_t srcCF = AlignLo(srcC, F);
            for (size_t c = 0; c < srcCF; c += F)
            {
                __m128 tmp[6];
                WinogradKernel1x3Block1x4SetInput4t(src + c, srcC, tmp);
                WinogradKernel1x3Block1x4SetInput4Store(tmp, dst + c, dstStride);
            }
            if (srcCF < srcC)
            {
                __m128 tmp[6];
                WinogradKernel1x3Block1x4SetInput4t(src + srcC - F, srcC, tmp);
                WinogradKernel1x3Block1x4SetInput4Store(tmp, dst + srcC - F, dstStride);
            }
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetInput4t(const float* src, size_t srcC, size_t colB, size_t colE, __m128 dst[6])
        {
            for (size_t col = 0; col < colB; ++col)
                dst[col] = _mm_setzero_ps();
            for (size_t col = colB; col < colE; ++col)
                dst[col] = _mm_loadu_ps(src + col * srcC);
            for (size_t col = colE; col < 6; ++col)
                dst[col] = _mm_setzero_ps();
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetInput4t(const float* src, size_t srcC, size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            size_t srcCF = AlignLo(srcC, F);
            for (size_t c = 0; c < srcCF; c += F)
            {
                __m128 tmp[6];
                WinogradKernel1x3Block1x4SetInput4t(src + c, srcC, colB, colE, tmp);
                WinogradKernel1x3Block1x4SetInput4Store(tmp, dst + c, dstStride);
            }
            if (srcCF < srcC)
            {
                __m128 tmp[6];
                WinogradKernel1x3Block1x4SetInput4t(src + srcC - F, srcC, colB, colE, tmp);
                WinogradKernel1x3Block1x4SetInput4Store(tmp, dst + srcC - F, dstStride);
            }
        }

        void WinogradKernel1x3Block1x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padX == padW && padY == 0 && padH == 0 && (padX == 0 || padX == 1));
            if (trans ? (srcChannels < 4) : (srcWidth < 12))
            {
                Base::WinogradKernel1x3Block1x4SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
                return;
            }
            size_t dstH = srcHeight;
            size_t dstW = padX ? srcWidth : srcWidth - 2;
            size_t tileW = (dstW + 3) / 4;
            size_t dstW4 = AlignLo(dstW, 4);
            if (trans)
            {
                size_t noseW = Simd::Min<size_t>(6, dstW + 1);
                size_t startX = padX ? 4 : 0;
                if (padX)
                {
                    if (dstW == dstW4)
                        dstW4 -= 4;
                    src -= srcChannels;
                }
                size_t tailW = dstW - dstW4 + (padX ? 1 : 2);
                for (size_t row = 0; row < dstH; row += 1)
                {
                    size_t col = 0;
                    if (padX)
                        WinogradKernel1x3Block1x4SetInput4t(src, srcChannels, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = startX; col < dstW4; col += 4)
                        WinogradKernel1x3Block1x4SetInput4t(src + col * srcChannels, srcChannels, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        WinogradKernel1x3Block1x4SetInput4t(src + col * srcChannels, srcChannels, 0, tailW, dst, dstStride), dst += srcChannels;
                    src += srcWidth * srcChannels;
                }
            }
            else
            {
                Base::WinogradKernel1x3Block1x4SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel1x3Block1x4SetOutputLoad6(const float* src, size_t stride, __m128 dst[4])
        {
            __m128 s[6];
            s[0] = _mm_loadu_ps(src + 0 * stride);
            s[1] = _mm_loadu_ps(src + 1 * stride);
            s[2] = _mm_loadu_ps(src + 2 * stride);
            s[3] = _mm_loadu_ps(src + 3 * stride);
            s[4] = _mm_loadu_ps(src + 4 * stride);
            s[5] = _mm_loadu_ps(src + 5 * stride);
            __m128 _2 = _mm_set1_ps(2.0f);
            __m128 _4 = _mm_set1_ps(4.0f);
            __m128 _8 = _mm_set1_ps(8.0f);
            dst[0] = _mm_add_ps(_mm_add_ps(_mm_add_ps(s[0], s[1]), _mm_add_ps(s[2], s[3])), s[4]);
            dst[1] = _mm_add_ps(_mm_sub_ps(s[1], s[2]), _mm_mul_ps(_2, _mm_sub_ps(s[3], s[4])));
            dst[2] = _mm_add_ps(_mm_add_ps(s[1], s[2]), _mm_mul_ps(_4, _mm_add_ps(s[3], s[4])));
            dst[3] = _mm_add_ps(_mm_add_ps(_mm_sub_ps(s[1], s[2]), _mm_mul_ps(_8, _mm_sub_ps(s[3], s[4]))), s[5]);
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetOutputStore4(const __m128 src[4], float* dst, size_t dstC)
        {
            _mm_storeu_ps(dst + 0 * dstC, src[0]);
            _mm_storeu_ps(dst + 1 * dstC, src[1]);
            _mm_storeu_ps(dst + 2 * dstC, src[2]);
            _mm_storeu_ps(dst + 3 * dstC, src[3]);
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetOutput4t(const float* src, size_t srcStride, float* dst, size_t dstC)
        {
            size_t dstCF = AlignLo(dstC, F);
            for (size_t d = 0; d < dstCF; d += F)
            {
                __m128 tmp[4];
                WinogradKernel1x3Block1x4SetOutputLoad6(src + d, srcStride, tmp);
                WinogradKernel1x3Block1x4SetOutputStore4(tmp, dst + d, dstC);
            }
            if (dstCF < dstC)
            {
                __m128 tmp[4];
                WinogradKernel1x3Block1x4SetOutputLoad6(src + dstC - F, srcStride, tmp);
                WinogradKernel1x3Block1x4SetOutputStore4(tmp, dst + dstC - F, dstC);
            }
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetOutputStore4(const __m128 src[4], float* dst, size_t dstC, size_t colE)
        {
            for (size_t col = 0; col < colE; ++col)
                _mm_storeu_ps(dst + col * dstC, src[col]);
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetOutput4t(const float* src, size_t srcStride, float* dst, size_t dstC, size_t colE)
        {
            size_t dstCF = AlignLo(dstC, F);
            for (size_t d = 0; d < dstCF; d += F)
            {
                __m128 tmp[4];
                WinogradKernel1x3Block1x4SetOutputLoad6(src + d, srcStride, tmp);
                WinogradKernel1x3Block1x4SetOutputStore4(tmp, dst + d, dstC, colE);
            }
            if (dstCF < dstC)
            {
                __m128 tmp[4];
                WinogradKernel1x3Block1x4SetOutputLoad6(src + dstC - F, srcStride, tmp);
                WinogradKernel1x3Block1x4SetOutputStore4(tmp, dst + dstC - F, dstC, colE);
            }
        }

        void WinogradKernel1x3Block1x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            if (trans ? (dstChannels < 4) : (dstWidth < 16))
            {
                Base::WinogradKernel1x3Block1x4SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
                return;
            }
            size_t tileW = (dstWidth + 3) / 4;
            size_t dstW4 = AlignLo(dstWidth, 4);
            if (trans)
            {
                for (size_t row = 0; row < dstHeight; row += 1)
                {
                    size_t col;
                    for (col = 0; col < dstW4; col += 4)
                        WinogradKernel1x3Block1x4SetOutput4t(src, srcStride, dst + col * dstChannels, dstChannels), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel1x3Block1x4SetOutput4t(src, srcStride, dst + col * dstChannels, dstChannels, dstWidth - col), src += dstChannels;
                    dst += dstWidth * dstChannels;
                }
            }
            else
            {
                Base::WinogradKernel1x3Block1x4SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel1x5Block1x4SetFilter(const __m128 * t, float* dst, size_t stride)
        {
            const __m128 r36 = _mm_set1_ps(1.0f / 36.0f);
            const __m128 r48 = _mm_set1_ps(1.0f / 48.0f);
            const __m128 mr120 = _mm_set1_ps(-1.0f / 120.0f);
            const __m128 r720 = _mm_set1_ps(1.0f / 720.0f);
            const __m128 _2 = _mm_set1_ps(2.0f);
            const __m128 _3 = _mm_set1_ps(3.0f);
            const __m128 _4 = _mm_set1_ps(4.0f);
            const __m128 _9 = _mm_set1_ps(9.0f);
            _mm_storeu_ps(dst + 0 * stride, _mm_mul_ps(r36, t[0]));
            __m128 a[2];
            a[0] = _mm_add_ps(_mm_add_ps(t[0], t[2]), t[4]);
            a[1] = _mm_add_ps(t[1], t[3]);
            _mm_storeu_ps(dst + 1 * stride, _mm_mul_ps(r48, _mm_add_ps(a[0], a[1])));
            _mm_storeu_ps(dst + 2 * stride, _mm_mul_ps(r48, _mm_sub_ps(a[0], a[1])));
            a[0] = _mm_add_ps(t[0], _mm_mul_ps(_4, _mm_add_ps(t[2], _mm_mul_ps(_4, t[4]))));
            a[1] = _mm_mul_ps(_2, _mm_add_ps(t[1], _mm_mul_ps(_4, t[3])));
            _mm_storeu_ps(dst + 3 * stride, _mm_mul_ps(mr120, _mm_add_ps(a[0], a[1])));
            _mm_storeu_ps(dst + 4 * stride, _mm_mul_ps(mr120, _mm_sub_ps(a[0], a[1])));
            a[0] = _mm_add_ps(t[0], _mm_mul_ps(_9, _mm_add_ps(t[2], _mm_mul_ps(_9, t[4]))));
            a[1] = _mm_mul_ps(_3, _mm_add_ps(t[1], _mm_mul_ps(_9, t[3])));
            _mm_storeu_ps(dst + 5 * stride, _mm_mul_ps(r720, _mm_add_ps(a[0], a[1])));
            _mm_storeu_ps(dst + 6 * stride, _mm_mul_ps(r720, _mm_sub_ps(a[0], a[1])));
            _mm_storeu_ps(dst + 7 * stride, t[4]);
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetFilter4n(const float* src, float* dst, size_t stride)
        {
            __m128 s[5];
            Load4(src + 0, 5, s + 0);
            s[4] = _mm_setr_ps(src[4], src[9], src[14], src[19]);
            WinogradKernel1x5Block1x4SetFilter(s, dst, stride);
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetFilter4t(const float* src, float* dst, size_t stride)
        {
            __m128 s[5];
            s[0] = _mm_loadu_ps(src + 0 * stride);
            s[1] = _mm_loadu_ps(src + 1 * stride);
            s[2] = _mm_loadu_ps(src + 2 * stride);
            s[3] = _mm_loadu_ps(src + 3 * stride);
            s[4] = _mm_loadu_ps(src + 4 * stride);
            WinogradKernel1x5Block1x4SetFilter(s, dst, stride);
        }

        void WinogradKernel1x5Block1x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
        {
            size_t size4 = AlignLo(size, 4), i = 0;
            if (trans)
            {
                for (; i < size4; i += 4)
                    WinogradKernel1x5Block1x4SetFilter4t(src + i, dst + i, size);
                for (; i < size; i += 1)
                    Base::WinogradKernel1x5Block1x4SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                for (; i < size4; i += 4, src += 20, dst += 4)
                    WinogradKernel1x5Block1x4SetFilter4n(src, dst, size);
                for (; i < size; i += 1, src += 5, dst += 1)
                    Base::WinogradKernel1x5Block1x4SetFilter1n(src, dst, size);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel1x5Block1x4SetInput4Store(const __m128 src[8], float* dst, size_t stride)
        {
            __m128 _2 = _mm_set1_ps(2.0f);
            __m128 _3 = _mm_set1_ps(3.0f);
            __m128 _4 = _mm_set1_ps(4.0f);
            __m128 _5 = _mm_set1_ps(5.0f);
            __m128 _9 = _mm_set1_ps(9.0f);
            __m128 _10 = _mm_set1_ps(10.0f);
            __m128 _13 = _mm_set1_ps(13.0f);
            __m128 _14 = _mm_set1_ps(14.0f);
            __m128 _36 = _mm_set1_ps(36.0f);
            __m128 _49 = _mm_set1_ps(49.0f);
            _mm_storeu_ps(dst + 0 * stride, _mm_add_ps(_mm_sub_ps(_mm_mul_ps(_36, src[0]), _mm_mul_ps(_49, src[2])), _mm_sub_ps(_mm_mul_ps(_14, src[4]), src[6])));
            __m128 a[2];
            a[0] = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(_36, src[2]), _mm_mul_ps(_13, src[4])), src[6]);
            a[1] = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(_36, src[1]), _mm_mul_ps(_13, src[3])), src[5]);
            _mm_storeu_ps(dst + 1 * stride, _mm_add_ps(a[0], a[1]));
            _mm_storeu_ps(dst + 2 * stride, _mm_sub_ps(a[0], a[1]));
            a[0] = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(_9, src[2]), _mm_mul_ps(_10, src[4])), src[6]);
            a[1] = _mm_mul_ps(_2, _mm_add_ps(_mm_sub_ps(_mm_mul_ps(_9, src[1]), _mm_mul_ps(_10, src[3])), src[5]));
            _mm_storeu_ps(dst + 3 * stride, _mm_add_ps(a[0], a[1]));
            _mm_storeu_ps(dst + 4 * stride, _mm_sub_ps(a[0], a[1]));
            a[0] = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(_4, src[2]), _mm_mul_ps(_5, src[4])), src[6]);
            a[1] = _mm_mul_ps(_3, _mm_add_ps(_mm_sub_ps(_mm_mul_ps(_4, src[1]), _mm_mul_ps(_5, src[3])), src[5]));
            _mm_storeu_ps(dst + 5 * stride, _mm_add_ps(a[0], a[1]));
            _mm_storeu_ps(dst + 6 * stride, _mm_sub_ps(a[0], a[1]));
            _mm_storeu_ps(dst + 7 * stride, _mm_add_ps(_mm_sub_ps(_mm_mul_ps(_49, src[3]), _mm_mul_ps(_36, src[1])), _mm_sub_ps(src[7], _mm_mul_ps(_14, src[5]))));
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetInput4t(const float* src, size_t srcC, __m128 dst[8])
        {
            dst[0] = _mm_loadu_ps(src + 0 * srcC);
            dst[1] = _mm_loadu_ps(src + 1 * srcC);
            dst[2] = _mm_loadu_ps(src + 2 * srcC);
            dst[3] = _mm_loadu_ps(src + 3 * srcC);
            dst[4] = _mm_loadu_ps(src + 4 * srcC);
            dst[5] = _mm_loadu_ps(src + 5 * srcC);
            dst[6] = _mm_loadu_ps(src + 6 * srcC);
            dst[7] = _mm_loadu_ps(src + 7 * srcC);
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetInput4t(const float* src, size_t srcC, float* dst, size_t dstStride)
        {
            size_t srcCF = AlignLo(srcC, F);
            for (size_t c = 0; c < srcCF; c += F)
            {
                __m128 tmp[8];
                WinogradKernel1x5Block1x4SetInput4t(src + c, srcC, tmp);
                WinogradKernel1x5Block1x4SetInput4Store(tmp, dst + c, dstStride);
            }
            if (srcCF < srcC)
            {
                __m128 tmp[8];
                WinogradKernel1x5Block1x4SetInput4t(src + srcC - F, srcC, tmp);
                WinogradKernel1x5Block1x4SetInput4Store(tmp, dst + srcC - F, dstStride);
            }
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetInput4t(const float* src, size_t srcC, size_t colB, size_t colE, __m128 dst[8])
        {
            for (size_t col = 0; col < colB; ++col)
                dst[col] = _mm_setzero_ps();
            for (size_t col = colB; col < colE; ++col)
                dst[col] = _mm_loadu_ps(src + col * srcC);
            for (size_t col = colE; col < 8; ++col)
                dst[col] = _mm_setzero_ps();
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetInput4t(const float* src, size_t srcC, size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            size_t srcCF = AlignLo(srcC, F);
            for (size_t c = 0; c < srcCF; c += F)
            {
                __m128 tmp[8];
                WinogradKernel1x5Block1x4SetInput4t(src + c, srcC, colB, colE, tmp);
                WinogradKernel1x5Block1x4SetInput4Store(tmp, dst + c, dstStride);
            }
            if (srcCF < srcC)
            {
                __m128 tmp[8];
                WinogradKernel1x5Block1x4SetInput4t(src + srcC - F, srcC, colB, colE, tmp);
                WinogradKernel1x5Block1x4SetInput4Store(tmp, dst + srcC - F, dstStride);
            }
        }

        void WinogradKernel1x5Block1x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padX == padW && padY == 0 && padH == 0 && (padX == 0 || padX == 2));
            if (trans ? (srcChannels < F) : true)
            {
                Base::WinogradKernel1x5Block1x4SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
                return;
            }
            size_t dstH = srcHeight;
            size_t dstW = padX ? srcWidth : srcWidth - 4;
            size_t tileW = (dstW + 3) / 4;
            size_t dstW4 = AlignLo(dstW, 4);
            size_t noseW = Simd::Min<size_t>(8, dstW + 2);
            size_t startX = padX ? 4 : 0;
            if (padX)
            {
                if (dstW == dstW4 || dstW == dstW4 + 1)
                    dstW4 -= 4;
                src -= 2*srcChannels;
            }
            size_t tailW = dstW - dstW4 + (padX ? 2 : 4);
            for (size_t row = 0; row < dstH; row += 1)
            {
                size_t col = 0;
                if (padX)
                    WinogradKernel1x5Block1x4SetInput4t(src, srcChannels, 2, noseW, dst, dstStride), dst += srcChannels;
                for (col = startX; col < dstW4; col += 4)
                    WinogradKernel1x5Block1x4SetInput4t(src + col * srcChannels, srcChannels, dst, dstStride), dst += srcChannels;
                for (size_t tail = tailW; col < dstW; col += 4, tail -= 4)
                    WinogradKernel1x5Block1x4SetInput4t(src + col * srcChannels, srcChannels, 0, tail, dst, dstStride), dst += srcChannels;
                src += srcWidth * srcChannels;
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel1x5Block1x4SetOutputLoad8(const float* src, size_t stride, __m128 dst[4])
        {
            const __m128 _2 = _mm_set1_ps(2.0f);
            const __m128 _3 = _mm_set1_ps(3.0f);
            const __m128 _4 = _mm_set1_ps(4.0f);
            const __m128 _9 = _mm_set1_ps(9.0f);
            __m128 s[8];
            s[0] = _mm_loadu_ps(src + 1 * stride);
            s[7] = _mm_loadu_ps(src + 2 * stride);
            s[1] = _mm_add_ps(s[0], s[7]);
            s[2] = _mm_sub_ps(s[0], s[7]);
            s[0] = _mm_loadu_ps(src + 3 * stride);
            s[7] = _mm_loadu_ps(src + 4 * stride);
            s[3] = _mm_add_ps(s[0], s[7]);
            s[4] = _mm_mul_ps(_2, _mm_sub_ps(s[0], s[7]));
            s[0] = _mm_loadu_ps(src + 5 * stride);
            s[7] = _mm_loadu_ps(src + 6 * stride);
            s[5] = _mm_add_ps(s[0], s[7]);
            s[6] = _mm_mul_ps(_3, _mm_sub_ps(s[0], s[7]));
            dst[0] = _mm_add_ps(_mm_loadu_ps(src + 0 * stride), _mm_add_ps(_mm_add_ps(s[1], s[3]), s[5]));
            dst[1] = _mm_add_ps(s[2], _mm_add_ps(s[4], s[6]));
            dst[2] = _mm_add_ps(s[1], _mm_add_ps(_mm_mul_ps(_4, s[3]), _mm_mul_ps(_9, s[5])));
            dst[3] = _mm_add_ps(_mm_loadu_ps(src + 7 * stride), _mm_add_ps(_mm_add_ps(s[2], _mm_mul_ps(_4, s[4])), _mm_mul_ps(_9, s[6])));
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetOutputStore4(const __m128 src[4], float* dst, size_t dstC)
        {
            _mm_storeu_ps(dst + 0 * dstC, src[0]);
            _mm_storeu_ps(dst + 1 * dstC, src[1]);
            _mm_storeu_ps(dst + 2 * dstC, src[2]);
            _mm_storeu_ps(dst + 3 * dstC, src[3]);
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetOutput4t(const float* src, size_t srcStride, float* dst, size_t dstC)
        {
            size_t dstCF = AlignLo(dstC, F);
            for (size_t d = 0; d < dstCF; d += F)
            {
                __m128 tmp[4];
                WinogradKernel1x5Block1x4SetOutputLoad8(src + d, srcStride, tmp);
                WinogradKernel1x5Block1x4SetOutputStore4(tmp, dst + d, dstC);
            }
            if (dstCF < dstC)
            {
                __m128 tmp[4];
                WinogradKernel1x5Block1x4SetOutputLoad8(src + dstC - F, srcStride, tmp);
                WinogradKernel1x5Block1x4SetOutputStore4(tmp, dst + dstC - F, dstC);
            }
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetOutputStore4(const __m128 src[4], float* dst, size_t dstC, size_t colE)
        {
            for (size_t col = 0; col < colE; ++col)
                _mm_storeu_ps(dst + col * dstC, src[col]);
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetOutput4t(const float* src, size_t srcStride, float* dst, size_t dstC, size_t colE)
        {
            size_t dstCF = AlignLo(dstC, F);
            for (size_t d = 0; d < dstCF; d += F)
            {
                __m128 tmp[4];
                WinogradKernel1x5Block1x4SetOutputLoad8(src + d, srcStride, tmp);
                WinogradKernel1x5Block1x4SetOutputStore4(tmp, dst + d, dstC, colE);
            }
            if (dstCF < dstC)
            {
                __m128 tmp[4];
                WinogradKernel1x5Block1x4SetOutputLoad8(src + dstC - F, srcStride, tmp);
                WinogradKernel1x5Block1x4SetOutputStore4(tmp, dst + dstC - F, dstC, colE);
            }
        }

        void WinogradKernel1x5Block1x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            if (trans ? (dstChannels < F) : true)
            {
                Base::WinogradKernel1x5Block1x4SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
                return;
            }
            size_t tileW = (dstWidth + 3) / 4;
            size_t dstW4 = AlignLo(dstWidth, 4);
            for (size_t row = 0; row < dstHeight; row += 1)
            {
                size_t col;
                for (col = 0; col < dstW4; col += 4)
                    WinogradKernel1x5Block1x4SetOutput4t(src, srcStride, dst + col * dstChannels, dstChannels), src += dstChannels;
                if (col < dstWidth)
                    WinogradKernel1x5Block1x4SetOutput4t(src, srcStride, dst + col * dstChannels, dstChannels, dstWidth - col), src += dstChannels;
                dst += dstWidth * dstChannels;
            }
        }
    }
#endif// SIMD_SSE2_ENABLE
}
