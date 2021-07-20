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
#ifndef __SimdWinograd_h__
#define __SimdWinograd_h__

#include "Simd/SimdMath.h"
#include "Simd/SimdLoad.h"

namespace Simd
{
    enum PadType
    {
        PadNose1,
        PadNone,
        PadTail1,
        PadTail2,
    };

    namespace Base
    {
        SIMD_INLINE void WinogradKernel1x3Block1x4SetFilter1n(const float* src, float* dst, size_t stride)
        {
            const float r4 = float(1.0f / 4.0f);
            const float r6 = float(1.0f / 6.0f);
            const float r12 = float(1.0f / 12.0f);
            const float r24 = float(1.0f / 24.0f);

            dst[stride * 0] = r4 * src[0];
            dst[stride * 1] = -r6 * (src[0] + src[1] + src[2]);
            dst[stride * 2] = -r6 * (src[0] - src[1] + src[2]);
            dst[stride * 3] = r24 * src[0] + r12 * src[1] + r6 * src[2];
            dst[stride * 4] = r24 * src[0] - r12 * src[1] + r6 * src[2];
            dst[stride * 5] = src[2];
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetFilter1t(const float* src, float* dst, size_t stride)
        {
            float _src[3];
            _src[0] = src[0 * stride];
            _src[1] = src[1 * stride];
            _src[2] = src[2 * stride];
            WinogradKernel1x3Block1x4SetFilter1n(_src, dst, stride);
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel1x5Block1x4SetFilter1n(const float* src, float* dst, size_t stride)
        {
            const float r36 = float(1.0f / 36.0f);
            const float r48 = float(1.0f / 48.0f);
            const float r120 = float(1.0f / 120.0f);
            const float r720 = float(1.0f / 720.0f);

            dst[stride * 0] = r36 * src[0];
            dst[stride * 1] = r48 * (src[0] + src[1] + src[2] + src[3] + src[4]);
            dst[stride * 2] = r48 * (src[0] - src[1] + src[2] - src[3] + src[4]);
            dst[stride * 3] = -r120 * (src[0] + 2 * src[1] + 4 * src[2] + 8 * src[3] + 16 * src[4]);
            dst[stride * 4] = -r120 * (src[0] - 2 * src[1] + 4 * src[2] - 8 * src[3] + 16 * src[4]);
            dst[stride * 5] = r720 * (src[0] + 3 * src[1] + 9 * src[2] + 27 * src[3] + 81 * src[4]);
            dst[stride * 6] = r720 * (src[0] - 3 * src[1] + 9 * src[2] - 27 * src[3] + 81 * src[4]);
            dst[stride * 7] = src[4];
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetFilter1t(const float* src, float* dst, size_t stride)
        {
            float _src[5];
            _src[0] = src[0 * stride];
            _src[1] = src[1 * stride];
            _src[2] = src[2 * stride];
            _src[3] = src[3 * stride];
            _src[4] = src[4 * stride];
            WinogradKernel1x5Block1x4SetFilter1n(_src, dst, stride);
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel2x2Block2x2SetFilter1n(const float* src, float* dst, size_t stride)
        {
            dst[0 * stride] = src[0];
            dst[1 * stride] = src[0] + src[1];
            dst[2 * stride] = src[1];
            dst[3 * stride] = src[0] + src[2];
            dst[4 * stride] = src[0] + src[1] + src[2] + src[3];
            dst[5 * stride] = src[1] + src[3];
            dst[6 * stride] = src[2];
            dst[7 * stride] = src[2] + src[3];
            dst[8 * stride] = src[3];
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetFilter1t(const float* src, float* dst, size_t stride)
        {
            float src0 = src[0 * stride];
            float src1 = src[1 * stride];
            float src2 = src[2 * stride];
            float src3 = src[3 * stride];
            dst[0 * stride] = src0;
            dst[1 * stride] = src0 + src1;
            dst[2 * stride] = src1;
            dst[3 * stride] = src0 + src2;
            dst[4 * stride] = src0 + src1 + src2 + src3;
            dst[5 * stride] = src1 + src3;
            dst[6 * stride] = src2;
            dst[7 * stride] = src2 + src3;
            dst[8 * stride] = src3;
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel2x2Block4x4SetFilter1n(const float* src, float* dst, size_t stride)
        {
            const float r2 = 1.0f / 2.0f;
            const float r3 = 1.0f / 3.0f;
            const float r6 = 1.0f / 6.0f;

            float t[10];
            t[0] = r2 * src[0];
            t[1] = r2 * src[1];
            t[2] = r2 * (- src[0] - src[2]);
            t[3] = r2 * (- src[1] - src[3]);
            t[4] = r6 * (src[2] - src[0]);
            t[5] = r6 * (src[3] - src[1]);
            t[6] = r6 * src[0] + r3 * src[2];
            t[7] = r6 * src[1] + r3 * src[3];
            t[8] = src[2];
            t[9] = src[3];

            dst[0 * stride] = r2 * t[0];
            dst[1 * stride] = r2 * (-t[0] - t[1]);
            dst[2 * stride] = r6 * (t[1] - t[0]);
            dst[3 * stride] = r6 * t[0] + r3* t[1];
            dst[4 * stride] = t[1];

            dst[5 * stride] = r2 * t[2];
            dst[6 * stride] = r2 * (-t[2] - t[3]);
            dst[7 * stride] = r6 * (t[3] - t[2]);
            dst[8 * stride] = r6 * t[2] + r3 * t[3];
            dst[9 * stride] = t[3];

            dst[10 * stride] = r2 * t[4];
            dst[11 * stride] = r2 * (-t[4] - t[5]);
            dst[12 * stride] = r6 * (t[5] - t[4]);
            dst[13 * stride] = r6 * t[4] + r3 * t[5];
            dst[14 * stride] = t[5];

            dst[15 * stride] = r2 * t[6];
            dst[16 * stride] = r2 * (-t[6] - t[7]);
            dst[17 * stride] = r6 * (t[7] - t[6]);
            dst[18 * stride] = r6 * t[6] + r3 * t[7];
            dst[19 * stride] = t[7];

            dst[20 * stride] = r2 * t[8];
            dst[21 * stride] = r2 * (-t[8] - t[9]);
            dst[22 * stride] = r6 * (t[9] - t[8]);
            dst[23 * stride] = r6 * t[8] + r3 * t[9];
            dst[24 * stride] = t[9];
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetFilter1t(const float* src, float* dst, size_t stride)
        {
            const float r2 = 1.0f / 2.0f;
            const float r3 = 1.0f / 3.0f;
            const float r6 = 1.0f / 6.0f;
            float src0 = src[0 * stride];
            float src1 = src[1 * stride];
            float src2 = src[2 * stride];
            float src3 = src[3 * stride];

            float t[10];
            t[0] = r2 * src0;
            t[1] = r2 * src1;
            t[2] = r2 * (-src0 - src2);
            t[3] = r2 * (-src1 - src3);
            t[4] = r6 * (src2 - src0);
            t[5] = r6 * (src3 - src1);
            t[6] = r6 * src0 + r3 * src2;
            t[7] = r6 * src1 + r3 * src3;
            t[8] = src2;
            t[9] = src3;

            dst[0 * stride] = r2 * t[0];
            dst[1 * stride] = r2 * (-t[0] - t[1]);
            dst[2 * stride] = r6 * (t[1] - t[0]);
            dst[3 * stride] = r6 * t[0] + r3 * t[1];
            dst[4 * stride] = t[1];

            dst[5 * stride] = r2 * t[2];
            dst[6 * stride] = r2 * (-t[2] - t[3]);
            dst[7 * stride] = r6 * (t[3] - t[2]);
            dst[8 * stride] = r6 * t[2] + r3 * t[3];
            dst[9 * stride] = t[3];

            dst[10 * stride] = r2 * t[4];
            dst[11 * stride] = r2 * (-t[4] - t[5]);
            dst[12 * stride] = r6 * (t[5] - t[4]);
            dst[13 * stride] = r6 * t[4] + r3 * t[5];
            dst[14 * stride] = t[5];

            dst[15 * stride] = r2 * t[6];
            dst[16 * stride] = r2 * (-t[6] - t[7]);
            dst[17 * stride] = r6 * (t[7] - t[6]);
            dst[18 * stride] = r6 * t[6] + r3 * t[7];
            dst[19 * stride] = t[7];

            dst[20 * stride] = r2 * t[8];
            dst[21 * stride] = r2 * (-t[8] - t[9]);
            dst[22 * stride] = r6 * (t[9] - t[8]);
            dst[23 * stride] = r6 * t[8] + r3 * t[9];
            dst[24 * stride] = t[9];
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block2x2SetFilter1n(const float * src, float * dst, size_t stride)
        {
            const float r2 = 1.0f / 2.0f;
            const float r4 = 1.0f / 4.0f;
            dst[0 * stride] = src[0];
            dst[1 * stride] = (src[0] + src[2] + src[1])*r2;
            dst[2 * stride] = (src[0] + src[2] - src[1])*r2;
            dst[3 * stride] = src[2];
            dst[4 * stride] = (src[0] + src[6] + src[3])*r2;
            dst[5 * stride] = ((src[0] + src[6] + src[3]) + (src[2] + src[8] + src[5]) + (src[1] + src[7] + src[4]))*r4;
            dst[6 * stride] = ((src[0] + src[6] + src[3]) + (src[2] + src[8] + src[5]) - (src[1] + src[7] + src[4]))*r4;
            dst[7 * stride] = (src[2] + src[8] + src[5])*r2;
            dst[8 * stride] = (src[0] + src[6] - src[3])*r2;
            dst[9 * stride] = ((src[0] + src[6] - src[3]) + (src[2] + src[8] - src[5]) + (src[1] + src[7] - src[4]))*r4;
            dst[10 * stride] = ((src[0] + src[6] - src[3]) + (src[2] + src[8] - src[5]) - (src[1] + src[7] - src[4]))*r4;
            dst[11 * stride] = (src[2] + src[8] - src[5])*r2;
            dst[12 * stride] = src[6];
            dst[13 * stride] = (src[6] + src[8] + src[7])*r2;
            dst[14 * stride] = (src[6] + src[8] - src[7])*r2;
            dst[15 * stride] = src[8];
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetFilter1t(const float * src, float * dst, size_t stride)
        {
            const float r2 = 1.0f / 2.0f;
            const float r4 = 1.0f / 4.0f;
            float src0 = src[0 * stride];
            float src1 = src[1 * stride];
            float src2 = src[2 * stride];
            float src3 = src[3 * stride];
            float src4 = src[4 * stride];
            float src5 = src[5 * stride];
            float src6 = src[6 * stride];
            float src7 = src[7 * stride];
            float src8 = src[8 * stride];
            dst[0 * stride] = src0;
            dst[1 * stride] = (src0 + src2 + src1)*r2;
            dst[2 * stride] = (src0 + src2 - src1)*r2;
            dst[3 * stride] = src2;
            dst[4 * stride] = (src0 + src6 + src3)*r2;
            dst[5 * stride] = ((src0 + src6 + src3) + (src2 + src8 + src5) + (src1 + src7 + src4))*r4;
            dst[6 * stride] = ((src0 + src6 + src3) + (src2 + src8 + src5) - (src1 + src7 + src4))*r4;
            dst[7 * stride] = (src2 + src8 + src5)*r2;
            dst[8 * stride] = (src0 + src6 - src3)*r2;
            dst[9 * stride] = ((src0 + src6 - src3) + (src2 + src8 - src5) + (src1 + src7 - src4))*r4;
            dst[10 * stride] = ((src0 + src6 - src3) + (src2 + src8 - src5) - (src1 + src7 - src4))*r4;
            dst[11 * stride] = (src2 + src8 - src5)*r2;
            dst[12 * stride] = src6;
            dst[13 * stride] = (src6 + src8 + src7)*r2;
            dst[14 * stride] = (src6 + src8 - src7)*r2;
            dst[15 * stride] = src8;
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block3x3SetFilter1n(const float * src, float * dst, size_t stride)
        {
            const float r6 = float(1.0f / 6.0f);
            const float r3 = float(1.0f / 3.0f);
            const float r2 = float(1.0f / 2.0f);
            const float f2_3 = float(2.0f / 3.0f);
            float t[15];
            t[0] = r2 * src[0];
            t[1] = r2 * src[1];
            t[2] = r2 * src[2];
            t[3] = -r2 * (src[0] + src[3] + src[6]);
            t[4] = -r2 * (src[1] + src[4] + src[7]);
            t[5] = -r2 * (src[2] + src[5] + src[8]);
            t[6] = -r6 * (src[0] - src[3] + src[6]);
            t[7] = -r6 * (src[1] - src[4] + src[7]);
            t[8] = -r6 * (src[2] - src[5] + src[8]);
            t[9] = r6 * src[0] + r3 * src[3] + f2_3 * src[6];
            t[10] = r6 * src[1] + r3 * src[4] + f2_3 * src[7];
            t[11] = r6 * src[2] + r3 * src[5] + f2_3 * src[8];
            t[12] = src[6];
            t[13] = src[7];
            t[14] = src[8];

            dst[stride * 0] = r2 * t[0];
            dst[stride * 1] = -r2 * (t[0] + t[1] + t[2]);
            dst[stride * 2] = -r6 * (t[0] - t[1] + t[2]);
            dst[stride * 3] = r6 * t[0] + r3 * t[1] + f2_3 * t[2];
            dst[stride * 4] = t[2];

            dst[stride * 5] = r2 * t[3];
            dst[stride * 6] = -r2 * (t[3] + t[4] + t[5]);
            dst[stride * 7] = -r6 * (t[3] - t[4] + t[5]);
            dst[stride * 8] = r6 * t[3] + r3 * t[4] + f2_3 * t[5];
            dst[stride * 9] = t[5];

            dst[stride * 10] = r2 * t[6];
            dst[stride * 11] = -r2 * (t[6] + t[7] + t[8]);
            dst[stride * 12] = -r6 * (t[6] - t[7] + t[8]);
            dst[stride * 13] = r6 * t[6] + r3 * t[7] + f2_3 * t[8];
            dst[stride * 14] = t[8];

            dst[stride * 15] = r2 * t[9];
            dst[stride * 16] = -r2 * (t[9] + t[10] + t[11]);
            dst[stride * 17] = -r6 * (t[9] - t[10] + t[11]);
            dst[stride * 18] = r6 * t[9] + r3 * t[10] + f2_3 * t[11];
            dst[stride * 19] = t[11];

            dst[stride * 20] = r2 * t[12];
            dst[stride * 21] = -r2 * (t[12] + t[13] + t[14]);
            dst[stride * 22] = -r6 * (t[12] - t[13] + t[14]);
            dst[stride * 23] = r6 * t[12] + r3 * t[13] + f2_3 * t[14];
            dst[stride * 24] = t[14];
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetFilter1t(const float * src, float * dst, size_t stride)
        {
            const float r6 = float(1.0f / 6.0f);
            const float r3 = float(1.0f / 3.0f);
            const float r2 = float(1.0f / 2.0f);
            const float f2_3 = float(2.0f / 3.0f);
            float src0 = src[0 * stride];
            float src1 = src[1 * stride];
            float src2 = src[2 * stride];
            float src3 = src[3 * stride];
            float src4 = src[4 * stride];
            float src5 = src[5 * stride];
            float src6 = src[6 * stride];
            float src7 = src[7 * stride];
            float src8 = src[8 * stride];
            float t[15];

            t[0] = r2 * src0;
            t[1] = r2 * src1;
            t[2] = r2 * src2;
            t[3] = -r2 * (src0 + src3 + src6);
            t[4] = -r2 * (src1 + src4 + src7);
            t[5] = -r2 * (src2 + src5 + src8);
            t[6] = -r6 * (src0 - src3 + src6);
            t[7] = -r6 * (src1 - src4 + src7);
            t[8] = -r6 * (src2 - src5 + src8);
            t[9] = r6 * src0 + r3 * src3 + f2_3 * src6;
            t[10] = r6 * src1 + r3 * src4 + f2_3 * src7;
            t[11] = r6 * src2 + r3 * src5 + f2_3 * src8;
            t[12] = src6;
            t[13] = src7;
            t[14] = src8;

            dst[stride * 0] = r2 * t[0];
            dst[stride * 1] = -r2 * (t[0] + t[1] + t[2]);
            dst[stride * 2] = -r6 * (t[0] - t[1] + t[2]);
            dst[stride * 3] = r6 * t[0] + r3 * t[1] + f2_3 * t[2];
            dst[stride * 4] = t[2];

            dst[stride * 5] = r2 * t[3];
            dst[stride * 6] = -r2 * (t[3] + t[4] + t[5]);
            dst[stride * 7] = -r6 * (t[3] - t[4] + t[5]);
            dst[stride * 8] = r6 * t[3] + r3 * t[4] + f2_3 * t[5];
            dst[stride * 9] = t[5];

            dst[stride * 10] = r2 * t[6];
            dst[stride * 11] = -r2 * (t[6] + t[7] + t[8]);
            dst[stride * 12] = -r6 * (t[6] - t[7] + t[8]);
            dst[stride * 13] = r6 * t[6] + r3 * t[7] + f2_3 * t[8];
            dst[stride * 14] = t[8];

            dst[stride * 15] = r2 * t[9];
            dst[stride * 16] = -r2 * (t[9] + t[10] + t[11]);
            dst[stride * 17] = -r6 * (t[9] - t[10] + t[11]);
            dst[stride * 18] = r6 * t[9] + r3 * t[10] + f2_3 * t[11];
            dst[stride * 19] = t[11];

            dst[stride * 20] = r2 * t[12];
            dst[stride * 21] = -r2 * (t[12] + t[13] + t[14]);
            dst[stride * 22] = -r6 * (t[12] - t[13] + t[14]);
            dst[stride * 23] = r6 * t[12] + r3 * t[13] + f2_3 * t[14];
            dst[stride * 24] = t[14];
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block4x4SetFilter1n(const float * src, float * dst, size_t stride)
        {
            const float r4 = float(1.0f / 4.0f);
            const float r6 = float(1.0f / 6.0f);
            const float r12 = float(1.0f / 12.0f);
            const float r24 = float(1.0f / 24.0f);
            float t[18];
            t[0] = r4 * src[0];
            t[1] = r4 * src[1];
            t[2] = r4 * src[2];
            t[3] = -r6 * (src[0] + src[3] + src[6]);
            t[4] = -r6 * (src[1] + src[4] + src[7]);
            t[5] = -r6 * (src[2] + src[5] + src[8]);
            t[6] = -r6 * (src[0] - src[3] + src[6]);
            t[7] = -r6 * (src[1] - src[4] + src[7]);
            t[8] = -r6 * (src[2] - src[5] + src[8]);
            t[9] = r24 * src[0] + r12 * src[3] + r6 * src[6];
            t[10] = r24 * src[1] + r12 * src[4] + r6 * src[7];
            t[11] = r24 * src[2] + r12 * src[5] + r6 * src[8];
            t[12] = r24 * src[0] - r12 * src[3] + r6 * src[6];
            t[13] = r24 * src[1] - r12 * src[4] + r6 * src[7];
            t[14] = r24 * src[2] - r12 * src[5] + r6 * src[8];
            t[15] = src[6];
            t[16] = src[7];
            t[17] = src[8];

            dst[stride*0] = r4 * t[0];
            dst[stride*1] = -r6 * (t[0] + t[1] + t[2]);
            dst[stride*2] = -r6 * (t[0] - t[1] + t[2]);
            dst[stride*3] = r24 * t[0] + r12 * t[1] + r6 * t[2];
            dst[stride*4] = r24 * t[0] - r12 * t[1] + r6 * t[2];
            dst[stride*5] = t[2];

            dst[stride*6] = r4 * t[3];
            dst[stride*7] = -r6 * (t[3] + t[4] + t[5]);
            dst[stride*8] = -r6 * (t[3] - t[4] + t[5]);
            dst[stride*9] = r24 * t[3] + r12 * t[4] + r6 * t[5];
            dst[stride*10] = r24 * t[3] - r12 * t[4] + r6 * t[5];
            dst[stride*11] = t[5];

            dst[stride*12] = r4 * t[6];
            dst[stride*13] = -r6 * (t[6] + t[7] + t[8]);
            dst[stride*14] = -r6 * (t[6] - t[7] + t[8]);
            dst[stride*15] = r24 * t[6] + r12 * t[7] + r6 * t[8];
            dst[stride*16] = r24 * t[6] - r12 * t[7] + r6 * t[8];
            dst[stride*17] = t[8];

            dst[stride*18] = r4 * t[9];
            dst[stride*19] = -r6 * (t[9] + t[10] + t[11]);
            dst[stride*20] = -r6 * (t[9] - t[10] + t[11]);
            dst[stride*21] = r24 * t[9] + r12 * t[10] + r6 * t[11];
            dst[stride*22] = r24 * t[9] - r12 * t[10] + r6 * t[11];
            dst[stride*23] = t[11];

            dst[stride*24] = r4 * t[12];
            dst[stride*25] = -r6 * (t[12] + t[13] + t[14]);
            dst[stride*26] = -r6 * (t[12] - t[13] + t[14]);
            dst[stride*27] = r24 * t[12] + r12 * t[13] + r6 * t[14];
            dst[stride*28] = r24 * t[12] - r12 * t[13] + r6 * t[14];
            dst[stride*29] = t[14];

            dst[stride*30] = r4 * t[15];
            dst[stride*31] = -r6 * (t[15] + t[16] + t[17]);
            dst[stride*32] = -r6 * (t[15] - t[16] + t[17]);
            dst[stride*33] = r24 * t[15] + r12 * t[16] + r6 * t[17];
            dst[stride*34] = r24 * t[15] - r12 * t[16] + r6 * t[17];
            dst[stride*35] = t[17];
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetFilter1t(const float * src, float * dst, size_t stride)
        {
            const float r4 = float(1.0f / 4.0f);
            const float r6 = float(1.0f / 6.0f);
            const float r12 = float(1.0f / 12.0f);
            const float r24 = float(1.0f / 24.0f);
            float src0 = src[0 * stride];
            float src1 = src[1 * stride];
            float src2 = src[2 * stride];
            float src3 = src[3 * stride];
            float src4 = src[4 * stride];
            float src5 = src[5 * stride];
            float src6 = src[6 * stride];
            float src7 = src[7 * stride];
            float src8 = src[8 * stride];
            float t[18];
            t[0] = r4 * src0;
            t[1] = r4 * src1;
            t[2] = r4 * src2;
            t[3] = -r6 * (src0 + src3 + src6);
            t[4] = -r6 * (src1 + src4 + src7);
            t[5] = -r6 * (src2 + src5 + src8);
            t[6] = -r6 * (src0 - src3 + src6);
            t[7] = -r6 * (src1 - src4 + src7);
            t[8] = -r6 * (src2 - src5 + src8);
            t[9] = r24 * src0 + r12 * src3 + r6 * src6;
            t[10] = r24 * src1 + r12 * src4 + r6 * src7;
            t[11] = r24 * src2 + r12 * src5 + r6 * src8;
            t[12] = r24 * src0 - r12 * src3 + r6 * src6;
            t[13] = r24 * src1 - r12 * src4 + r6 * src7;
            t[14] = r24 * src2 - r12 * src5 + r6 * src8;
            t[15] = src6;
            t[16] = src7;
            t[17] = src8;

            dst[stride * 0] = r4 * t[0];
            dst[stride * 1] = -r6 * (t[0] + t[1] + t[2]);
            dst[stride * 2] = -r6 * (t[0] - t[1] + t[2]);
            dst[stride * 3] = r24 * t[0] + r12 * t[1] + r6 * t[2];
            dst[stride * 4] = r24 * t[0] - r12 * t[1] + r6 * t[2];
            dst[stride * 5] = t[2];

            dst[stride * 6] = r4 * t[3];
            dst[stride * 7] = -r6 * (t[3] + t[4] + t[5]);
            dst[stride * 8] = -r6 * (t[3] - t[4] + t[5]);
            dst[stride * 9] = r24 * t[3] + r12 * t[4] + r6 * t[5];
            dst[stride * 10] = r24 * t[3] - r12 * t[4] + r6 * t[5];
            dst[stride * 11] = t[5];

            dst[stride * 12] = r4 * t[6];
            dst[stride * 13] = -r6 * (t[6] + t[7] + t[8]);
            dst[stride * 14] = -r6 * (t[6] - t[7] + t[8]);
            dst[stride * 15] = r24 * t[6] + r12 * t[7] + r6 * t[8];
            dst[stride * 16] = r24 * t[6] - r12 * t[7] + r6 * t[8];
            dst[stride * 17] = t[8];

            dst[stride * 18] = r4 * t[9];
            dst[stride * 19] = -r6 * (t[9] + t[10] + t[11]);
            dst[stride * 20] = -r6 * (t[9] - t[10] + t[11]);
            dst[stride * 21] = r24 * t[9] + r12 * t[10] + r6 * t[11];
            dst[stride * 22] = r24 * t[9] - r12 * t[10] + r6 * t[11];
            dst[stride * 23] = t[11];

            dst[stride * 24] = r4 * t[12];
            dst[stride * 25] = -r6 * (t[12] + t[13] + t[14]);
            dst[stride * 26] = -r6 * (t[12] - t[13] + t[14]);
            dst[stride * 27] = r24 * t[12] + r12 * t[13] + r6 * t[14];
            dst[stride * 28] = r24 * t[12] - r12 * t[13] + r6 * t[14];
            dst[stride * 29] = t[14];

            dst[stride * 30] = r4 * t[15];
            dst[stride * 31] = -r6 * (t[15] + t[16] + t[17]);
            dst[stride * 32] = -r6 * (t[15] - t[16] + t[17]);
            dst[stride * 33] = r24 * t[15] + r12 * t[16] + r6 * t[17];
            dst[stride * 34] = r24 * t[15] - r12 * t[16] + r6 * t[17];
            dst[stride * 35] = t[17];
        }
    }

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Sse2
    {
        SIMD_INLINE void Load4(const float* src, size_t step, __m128* dst)
        {
            __m128 a0 = _mm_loadu_ps(src + 0 * step);
            __m128 a1 = _mm_loadu_ps(src + 1 * step);
            __m128 a2 = _mm_loadu_ps(src + 2 * step);
            __m128 a3 = _mm_loadu_ps(src + 3 * step);
            __m128 b0 = _mm_unpacklo_ps(a0, a2);
            __m128 b1 = _mm_unpackhi_ps(a0, a2);
            __m128 b2 = _mm_unpacklo_ps(a1, a3);
            __m128 b3 = _mm_unpackhi_ps(a1, a3);
            dst[0] = _mm_unpacklo_ps(b0, b2);
            dst[1] = _mm_unpackhi_ps(b0, b2);
            dst[2] = _mm_unpacklo_ps(b1, b3);
            dst[3] = _mm_unpackhi_ps(b1, b3);
        }
    }
#endif

#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Neon
    {
        SIMD_INLINE void Load4(const float* src, size_t step, float32x4_t* dst)
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
    }
#endif
}

#endif//__SimdWinograd_h__
