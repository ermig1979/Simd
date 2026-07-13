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

#include "Simd/SimdSynetGridSample.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Neon
    {
        template <int align> SIMD_INLINE float Denormalize32f(float pos, int dim)
        {
            if (align)
                return float((pos + 1) / 2.0f * (dim - 1));
            else
                return float(((pos + 1) * dim - 1) / 2.0f);
        }

        SIMD_INLINE float32x4_t Floor(float32x4_t value)
        {
#ifdef SIMD_ARM64_ENABLE
            return vrndmq_f32(value);
#else
            float32x4_t integer = vcvtq_f32_s32(vcvtq_s32_f32(value));
            uint32x4_t mask = vcgtq_f32(integer, value);
            return vsubq_f32(integer, vbslq_f32(mask, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
#endif
        }

        SIMD_INLINE int MinVal32i(int32x4_t value)
        {
            SIMD_ALIGNED(16) int32_t tmp[F];
            vst1q_s32(tmp, value);
            return Min(Min(tmp[0], tmp[1]), Min(tmp[2], tmp[3]));
        }

        SIMD_INLINE int MaxVal32i(int32x4_t value)
        {
            SIMD_ALIGNED(16) int32_t tmp[F];
            vst1q_s32(tmp, value);
            return Max(Max(tmp[0], tmp[1]), Max(tmp[2], tmp[3]));
        }

        template<int align, int range>  void IndexCoeffs32fBlZ(const float* grd, size_t dstS, int srcH, int srcW, int padW, uint32_t* idx, float* dy, float* dx, int& yMin, int& yMax)
        {
            size_t dstSF = AlignLo(dstS, F), d = 0;
            const float32x4_t ax = vdupq_n_f32((srcW - align) / 2.0f);
            const float32x4_t ay = vdupq_n_f32((srcH - align) / 2.0f);
            const float32x4_t bx = vdupq_n_f32((srcW - 1) / 2.0f);
            const float32x4_t by = vdupq_n_f32((srcH - 1) / 2.0f);
            const int32x4_t _0 = vdupq_n_s32(0);
            const int32x4_t _2 = vdupq_n_s32(2);
            const int32x4_t _srcH = vdupq_n_s32(srcH + 2);
            const int32x4_t _srcW = vdupq_n_s32(srcW + 2);
            const int32x4_t _padW = vdupq_n_s32(padW);
            int32x4_t _yMin, _yMax;
            if (range)
            {
                _yMin = vdupq_n_s32(yMin);
                _yMax = vdupq_n_s32(yMax);
            }
            for (; d < dstSF; d += F)
            {
                float32x4x2_t xy = vld2q_f32(grd);
                float32x4_t x = vaddq_f32(vmulq_f32(xy.val[0], ax), bx);
                float32x4_t y = vaddq_f32(vmulq_f32(xy.val[1], ay), by);
                float32x4_t xf = Floor(x);
                float32x4_t yf = Floor(y);
                vst1q_f32(dy + d, vsubq_f32(y, yf));
                vst1q_f32(dx + d, vsubq_f32(x, xf));
                int32x4_t xi = vminq_s32(vmaxq_s32(vaddq_s32(vcvtq_s32_f32(xf), _2), _0), _srcW);
                int32x4_t yi = vminq_s32(vmaxq_s32(vaddq_s32(vcvtq_s32_f32(yf), _2), _0), _srcH);
                vst1q_u32(idx + d, vreinterpretq_u32_s32(vaddq_s32(vmulq_s32(_padW, yi), xi)));
                if (range)
                {
                    _yMin = vminq_s32(_yMin, yi);
                    _yMax = vmaxq_s32(_yMax, yi);
                }
                grd += 2 * F;
            }
            if (range)
            {
                yMin = MinVal32i(_yMin);
                yMax = MaxVal32i(_yMax);
            }
            for (; d < dstS; ++d)
            {
                float x = Denormalize32f<align>(grd[0], srcW);
                float y = Denormalize32f<align>(grd[1], srcH);
                int x0 = int(std::floor(x));
                int y0 = int(std::floor(y));
                dy[d] = y - float(y0);
                dx[d] = x - float(x0);
                x0 = Simd::RestrictRange(x0, -2, srcW) + 2;
                y0 = Simd::RestrictRange(y0, -2, srcH) + 2;
                idx[d] = padW * y0 + x0;
                if (range)
                {
                    yMin = Min(yMin, y0);
                    yMax = Max(yMax, y0);
                }
                grd += 2;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE float32x4_t Gather32f(const float* src, int i0, int i1, int i2, int i3)
        {
            float32x4_t dst = vdupq_n_f32(0.0f);
            dst = vsetq_lane_f32(src[i0], dst, 0);
            dst = vsetq_lane_f32(src[i1], dst, 1);
            dst = vsetq_lane_f32(src[i2], dst, 2);
            dst = vsetq_lane_f32(src[i3], dst, 3);
            return dst;
        }

        void BilinearInterp32fBlZ(const float* pad0, size_t dstS, int padW, uint32_t* idx, float* dy, float* dx, float* dst)
        {
            size_t dstSF = AlignLo(dstS, F), d = 0;
            const float* pad1 = pad0 + padW;
            const float32x4_t _1 = vdupq_n_f32(1.0f);
            for (; d < dstSF; d += F)
            {
                int i0 = idx[d + 0], i1 = idx[d + 1], i2 = idx[d + 2], i3 = idx[d + 3];
                float32x4_t p00 = Gather32f(pad0, i0 + 0, i1 + 0, i2 + 0, i3 + 0);
                float32x4_t p01 = Gather32f(pad0, i0 + 1, i1 + 1, i2 + 1, i3 + 1);
                float32x4_t p10 = Gather32f(pad1, i0 + 0, i1 + 0, i2 + 0, i3 + 0);
                float32x4_t p11 = Gather32f(pad1, i0 + 1, i1 + 1, i2 + 1, i3 + 1);
                float32x4_t dy1 = vld1q_f32(dy + d);
                float32x4_t dy0 = vsubq_f32(_1, dy1);
                float32x4_t dx1 = vld1q_f32(dx + d);
                float32x4_t dx0 = vsubq_f32(_1, dx1);
                float32x4_t d0 = vmlaq_f32(vmulq_f32(dx0, p00), dx1, p01);
                float32x4_t d1 = vmlaq_f32(vmulq_f32(dx0, p10), dx1, p11);
                vst1q_f32(dst + d, vmlaq_f32(vmulq_f32(dy0, d0), dy1, d1));
            }
            for (; d < dstS; ++d)
            {
                int offs = idx[d];
                float p00 = pad0[offs + 0];
                float p01 = pad0[offs + 1];
                float p10 = pad1[offs + 0];
                float p11 = pad1[offs + 1];
                float dy1 = dy[d];
                float dy0 = 1.0f - dy1;
                float dx1 = dx[d];
                float dx0 = 1.0f - dx1;
                dst[d] = dy0 * (dx0 * p00 + dx1 * p01) + dy1 * (dx0 * p10 + dx1 * p11);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetGridSample2d32fBlZ::SynetGridSample2d32fBlZ(const GridSample2dParam& param)
            : Base::SynetGridSample2d32fBlZ(param)
        {
            if (_sparse)
                _indexCoeffs = _param.align ? IndexCoeffs32fBlZ<1, 1> : IndexCoeffs32fBlZ<0, 1>;
            else
                _indexCoeffs = _param.align ? IndexCoeffs32fBlZ<1, 0> : IndexCoeffs32fBlZ<0, 0>;
            _bilinearInterp = BilinearInterp32fBlZ;
        }

        void* SynetGridSample2dInit(size_t batch, size_t channels, size_t srcH, size_t srcW, size_t dstH, size_t dstW,
            SimdTensorDataType type, SimdGridSampleInterpType interp, SimdGridSamplePaddingType padding, SimdBool align)
        {
            GridSample2dParam param(batch, channels, srcH, srcW, dstH, dstW, type, interp, padding, align);
            if (!param.Valid())
                return NULL;
            if (param.Is32fBlZ())
                return new Neon::SynetGridSample2d32fBlZ(param);
            else
                return new Base::SynetGridSample2dRef(param);
        }
    }
#endif
}
