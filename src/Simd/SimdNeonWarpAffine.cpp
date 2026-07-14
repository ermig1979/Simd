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
#include "Simd/SimdWarpAffine.h"
#include "Simd/SimdWarpAffineCommon.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdStore.h"

#include "Simd/SimdPoint.hpp"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        template<int N> SIMD_INLINE void FillBorder(uint8_t* dst, int count, const uint8x16_t& bv, const uint8_t* bs)
        {
            int i = 0, size = count * N, size16 = (int)AlignLo(size, A);
            for (; i < size16; i += A)
                vst1q_u8(dst + i, bv);
            for (; i < size; i += N)
                Base::CopyPixel<N>(bs, dst + i);
        }

        template<> SIMD_INLINE void FillBorder<3>(uint8_t* dst, int count, const uint8x16_t& bv, const uint8_t* bs)
        {
            int i = 0, size = count * 3, size3 = size - 3;
            for (; i < size3; i += 3)
                Base::CopyPixel<4>(bs, dst + i);
            for (; i < size; i += 3)
                Base::CopyPixel<3>(bs, dst + i);
        }

        template<int N> SIMD_INLINE uint8x16_t InitBorder(const uint8_t* border)
        {
            switch (N)
            {
            case 1: return vdupq_n_u8(*border);
            case 2: return (uint8x16_t)vdupq_n_u16(*(uint16_t*)border);
            case 3: return K8_00;
            case 4: return (uint8x16_t)vdupq_n_u32(*(uint32_t*)border);
            }
            return K8_00;
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE uint32x4_t NearestOffset(float32x4_t x, float32x4_t y, const float32x4_t* m, int32x4_t w, const int32x4_t& h, const int32x4_t& n, const int32x4_t& s)
        {
            float32x4_t dx = vaddq_f32(vaddq_f32(vmulq_f32(x, m[0]), vmulq_f32(y, m[1])), m[2]);
            float32x4_t dy = vaddq_f32(vaddq_f32(vmulq_f32(x, m[3]), vmulq_f32(y, m[4])), m[5]);
            int32x4_t zero = vdupq_n_s32(0);
            int32x4_t ix = vminq_s32(vmaxq_s32(Round(dx), zero), w);
            int32x4_t iy = vminq_s32(vmaxq_s32(Round(dy), zero), h);
            return vreinterpretq_u32_s32(vaddq_s32(vmulq_s32(ix, n), vmulq_s32(iy, s)));
        }

        //-----------------------------------------------------------------------------------------

        template<int N> void NearestRun(const WarpAffParam& p, int yBeg, int yEnd, const int32_t* beg, const int32_t* end, const uint8_t* src, uint8_t* dst, uint32_t* buf)
        {
            bool fill = p.NeedFill();
            int width = (int)p.dstW, s = (int)p.srcS, w = (int)p.srcW - 1, h = (int)p.srcH - 1;
            const float32x4_t _4 = vdupq_n_f32(4.0f);
            const int32x4_t _0123 = SIMD_VEC_SETR_EPI32(0, 1, 2, 3);
            float32x4_t _m[6];
            for (int i = 0; i < 6; ++i)
                _m[i] = vdupq_n_f32(p.inv[i]);
            int32x4_t _w = vdupq_n_s32(w);
            int32x4_t _h = vdupq_n_s32(h);
            int32x4_t _n = vdupq_n_s32(N);
            int32x4_t _s = vdupq_n_s32(s);
            uint8x16_t _border = InitBorder<N>(p.border);
            dst += yBeg * p.dstS;
            for (int y = yBeg; y < yEnd; ++y)
            {
                int nose = beg[y], tail = end[y];
                {
                    int x = nose;
                    float32x4_t _y = vcvtq_f32_s32(vdupq_n_s32(y));
                    float32x4_t _x = vcvtq_f32_s32(vaddq_s32(vdupq_n_s32(x), _0123));
                    for (; x < tail; x += 4)
                    {
                        vst1q_u32(buf + x, NearestOffset(_x, _y, _m, _w, _h, _n, _s));
                        _x = vaddq_f32(_x, _4);
                    }
                }
                if (fill)
                    FillBorder<N>(dst, nose, _border, p.border);
                Base::NearestGather<N>(src, buf + nose, tail - nose, dst + N * nose);
                if (fill)
                    FillBorder<N>(dst + tail * N, width - tail, _border, p.border);
                dst += p.dstS;
            }
        }

        //-------------------------------------------------------------------------------------------------

        WarpAffineNearest::WarpAffineNearest(const WarpAffParam& param)
            : Base::WarpAffineNearest(param)
        {
            switch (_param.channels)
            {
            case 1: _run = NearestRun<1>; break;
            case 2: _run = NearestRun<2>; break;
            case 3: _run = NearestRun<3>; break;
            case 4: _run = NearestRun<4>; break;
            }
        }

        void WarpAffineNearest::SetRange(const Base::Point* points)
        {
            const WarpAffParam& p = _param;
            int w = (int)p.dstW, h = (int)p.dstH, h4 = (int)AlignLo(h, 4);
            const int32x4_t _0123 = SIMD_VEC_SETR_EPI32(0, 1, 2, 3);
            int32x4_t _w = vdupq_n_s32(w), _1 = vdupq_n_s32(1), _0 = vdupq_n_s32(0);
            int y = 0;
            for (; y < h4; y += 4)
            {
                vst1q_s32(_beg.data + y, _w);
                vst1q_s32(_end.data + y, _0);
            }
            for (; y < h; ++y)
            {
                _beg[y] = w;
                _end[y] = 0;
            }
            for (int v = 0; v < 4; ++v)
            {
                const Base::Point& curr = points[v];
                const Base::Point& next = points[(v + 1) & 3];
                float yMin = Simd::Max(Simd::Min(curr.y, next.y), 0.0f);
                float yMax = Simd::Min(Simd::Max(curr.y, next.y), (float)p.dstH);
                int yBeg = Simd::Round(yMin);
                int yEnd = Simd::Round(yMax);
                int yEnd4 = (int)AlignLo(yEnd - yBeg, 4) + yBeg;
                if (next.y == curr.y)
                    continue;
                float a = (next.x - curr.x) / (next.y - curr.y);
                float b = curr.x - curr.y * a;
                float32x4_t _a = vdupq_n_f32(a);
                float32x4_t _b = vdupq_n_f32(b);
                if (abs(a) <= 1.0f)
                {
                    int y = yBeg;
                    for (; y < yEnd4; y += 4)
                    {
                        float32x4_t _y = vcvtq_f32_s32(vaddq_s32(vdupq_n_s32(y), _0123));
                        int32x4_t _x = Round(vaddq_f32(vmulq_f32(_y, _a), _b));
                        int32x4_t xBeg = vld1q_s32(_beg.data + y);
                        int32x4_t xEnd = vld1q_s32(_end.data + y);
                        xBeg = vminq_s32(xBeg, vmaxq_s32(_x, _0));
                        xEnd = vmaxq_s32(xEnd, vminq_s32(vaddq_s32(_x, _1), _w));
                        vst1q_s32(_beg.data + y, xBeg);
                        vst1q_s32(_end.data + y, xEnd);
                    }
                    for (; y < yEnd; ++y)
                    {
                        int x = Simd::Round(y * a + b);
                        _beg[y] = Simd::Min(_beg[y], Simd::Max(x, 0));
                        _end[y] = Simd::Max(_end[y], Simd::Min(x + 1, w));
                    }
                }
                else
                {
                    int y = yBeg;
                    float32x4_t _05 = vdupq_n_f32(0.5f);
                    float32x4_t _yMin = vdupq_n_f32(yMin);
                    float32x4_t _yMax = vdupq_n_f32(yMax);
                    for (; y < yEnd4; y += 4)
                    {
                        float32x4_t _y = vcvtq_f32_s32(vaddq_s32(vdupq_n_s32(y), _0123));
                        float32x4_t yM = vminq_f32(vmaxq_f32(vsubq_f32(_y, _05), _yMin), _yMax);
                        float32x4_t yP = vminq_f32(vmaxq_f32(vaddq_f32(_y, _05), _yMin), _yMax);
                        float32x4_t xM = vaddq_f32(vmulq_f32(yM, _a), _b);
                        float32x4_t xP = vaddq_f32(vmulq_f32(yP, _a), _b);
                        int32x4_t xBeg = vld1q_s32(_beg.data + y);
                        int32x4_t xEnd = vld1q_s32(_end.data + y);
                        xBeg = vminq_s32(xBeg, vmaxq_s32(Round(vminq_f32(xM, xP)), _0));
                        xEnd = vmaxq_s32(xEnd, vminq_s32(vaddq_s32(Round(vmaxq_f32(xM, xP)), _1), _w));
                        vst1q_s32(_beg.data + y, xBeg);
                        vst1q_s32(_end.data + y, xEnd);
                    }
                    for (; y < yEnd; ++y)
                    {
                        float xM = b + Simd::RestrictRange(float(y) - 0.5f, yMin, yMax) * a;
                        float xP = b + Simd::RestrictRange(float(y) + 0.5f, yMin, yMax) * a;
                        int xBeg = Simd::Round(Simd::Min(xM, xP));
                        int xEnd = Simd::Round(Simd::Max(xM, xP));
                        _beg[y] = Simd::Min(_beg[y], Simd::Max(xBeg, 0));
                        _end[y] = Simd::Max(_end[y], Simd::Min(xEnd + 1, w));
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

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

        SIMD_INLINE float32x4_t Ceil(float32x4_t value)
        {
#ifdef SIMD_ARM64_ENABLE
            return vrndpq_f32(value);
#else
            float32x4_t integer = vcvtq_f32_s32(vcvtq_s32_f32(value));
            uint32x4_t mask = vcgtq_f32(value, integer);
            return vaddq_f32(integer, vbslq_f32(mask, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
#endif
        }

        const uint32x4_t K32_WA_FRACTION_RANGE = SIMD_VEC_SET1_EPI32(Base::WA_FRACTION_RANGE);

        SIMD_INLINE void ByteBilinearPrepMain4(float32x4_t x, float32x4_t y, const float32x4_t* m, uint32x4_t n, const uint32x4_t& s, uint32_t* offs, uint8_t* fx, uint16_t* fy)
        {
            float32x4_t dx = vaddq_f32(vaddq_f32(vmulq_f32(x, m[0]), vmulq_f32(y, m[1])), m[2]);
            float32x4_t dy = vaddq_f32(vaddq_f32(vmulq_f32(x, m[3]), vmulq_f32(y, m[4])), m[5]);
            float32x4_t ix = Floor(dx);
            float32x4_t iy = Floor(dy);
            float32x4_t range = vcvtq_f32_u32(K32_WA_FRACTION_RANGE);
            uint32x4_t _fx = RoundPositive(vmulq_f32(vsubq_f32(dx, ix), range));
            uint32x4_t _fy = RoundPositive(vmulq_f32(vsubq_f32(dy, iy), range));
            vst1q_u32(offs, vaddq_u32(vmulq_u32(vcvtq_u32_f32(ix), n), vmulq_u32(vcvtq_u32_f32(iy), s)));

            uint16x4_t fx0 = vmovn_u32(vsubq_u32(K32_WA_FRACTION_RANGE, _fx));
            uint16x4_t fx1 = vmovn_u32(_fx);
            uint16x4x2_t fxx = vzip_u16(fx0, fx1);
            vst1_u8(fx, vmovn_u16(vcombine_u16(fxx.val[0], fxx.val[1])));

            uint16x4_t fy0 = vmovn_u32(vsubq_u32(K32_WA_FRACTION_RANGE, _fy));
            uint16x4_t fy1 = vmovn_u32(_fy);
            uint16x4x2_t fyy = vzip_u16(fy0, fy1);
            vst1q_u16(fy, vcombine_u16(fyy.val[0], fyy.val[1]));
        }

        SIMD_INLINE uint8x16_t Shuffle(uint8x16_t value, const uint8x16_t& index)
        {
            uint8x8x2_t table = { { vget_low_u8(value), vget_high_u8(value) } };
            return vcombine_u8(vtbl2_u8(table, vget_low_u8(index)), vtbl2_u8(table, vget_high_u8(index)));
        }

        SIMD_INLINE uint8x16_t UnpackU16(uint8x16_t value, int part)
        {
            uint16x8x2_t zip = vzipq_u16(vreinterpretq_u16_u8(value), vreinterpretq_u16_u8(value));
            return vreinterpretq_u8_u16(part ? zip.val[1] : zip.val[0]);
        }

        SIMD_INLINE uint16x8_t UnpackU32(uint16x8_t value, int part)
        {
            uint32x4x2_t zip = vzipq_u32(vreinterpretq_u32_u16(value), vreinterpretq_u32_u16(value));
            return vreinterpretq_u16_u32(part ? zip.val[1] : zip.val[0]);
        }

        SIMD_INLINE uint16x8_t MaddU8(uint8x16_t value, uint8x16_t weight)
        {
            return Hadd16u(vmull_u8(vget_low_u8(value), vget_low_u8(weight)), vmull_u8(vget_high_u8(value), vget_high_u8(weight)));
        }

        SIMD_INLINE uint32x4_t MaddU16(uint16x8_t src0, uint16x8_t src1, uint16x8_t weight, int part)
        {
            uint16x8x2_t src = vzipq_u16(src0, src1);
            uint16x8_t value = part ? src.val[1] : src.val[0];
            return Hadd32u(vmull_u16(vget_low_u16(value), vget_low_u16(weight)), vmull_u16(vget_high_u16(value), vget_high_u16(weight)));
        }

        const uint32x4_t K32_WA_BILINEAR_ROUND_TERM = SIMD_VEC_SET1_EPI32(Base::WA_BILINEAR_ROUND_TERM);

        SIMD_INLINE uint32x4_t Interp4(uint16x8_t src0, uint16x8_t src1, uint16x8_t fy, int part)
        {
            return vshrq_n_u32(vaddq_u32(MaddU16(src0, src1, fy, part), K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);
        }

        template<int N> void ByteBilinearInterpMainN(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst);

        template<> SIMD_INLINE void ByteBilinearInterpMainN<1>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst)
        {
            uint8x16_t fx0 = vld1q_u8(fx + 0);
            uint8x16_t fx1 = vld1q_u8(fx + A);
            uint16x8_t r00 = MaddU8(vld1q_u8(src0 + 0), fx0);
            uint16x8_t r01 = MaddU8(vld1q_u8(src0 + A), fx1);
            uint16x8_t r10 = MaddU8(vld1q_u8(src1 + 0), fx0);
            uint16x8_t r11 = MaddU8(vld1q_u8(src1 + A), fx1);

            uint32x4_t d0 = Interp4(r00, r10, vld1q_u16(fy + 0 * HA), 0);
            uint32x4_t d1 = Interp4(r00, r10, vld1q_u16(fy + 1 * HA), 1);
            uint32x4_t d2 = Interp4(r01, r11, vld1q_u16(fy + 2 * HA), 0);
            uint32x4_t d3 = Interp4(r01, r11, vld1q_u16(fy + 3 * HA), 1);

            vst1q_u8(dst, PackSaturatedU16(PackU32(d0, d1), PackU32(d2, d3)));
        }

        template<> SIMD_INLINE void ByteBilinearInterpMainN<2>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst)
        {
            const uint8x16_t SHUFFLE = SIMD_VEC_SETR_EPI8(0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF);
            uint8x16_t _fx = vld1q_u8(fx);
            uint8x16_t fx0 = UnpackU16(_fx, 0);
            uint8x16_t fx1 = UnpackU16(_fx, 1);
            uint16x8_t r00 = MaddU8(Shuffle(vld1q_u8(src0 + 0), SHUFFLE), fx0);
            uint16x8_t r01 = MaddU8(Shuffle(vld1q_u8(src0 + A), SHUFFLE), fx1);
            uint16x8_t r10 = MaddU8(Shuffle(vld1q_u8(src1 + 0), SHUFFLE), fx0);
            uint16x8_t r11 = MaddU8(Shuffle(vld1q_u8(src1 + A), SHUFFLE), fx1);

            uint16x8_t fy0 = vld1q_u16(fy + 0 * HA);
            uint32x4_t d0 = Interp4(r00, r10, UnpackU32(fy0, 0), 0);
            uint32x4_t d1 = Interp4(r00, r10, UnpackU32(fy0, 1), 1);
            uint16x8_t fy1 = vld1q_u16(fy + 1 * HA);
            uint32x4_t d2 = Interp4(r01, r11, UnpackU32(fy1, 0), 0);
            uint32x4_t d3 = Interp4(r01, r11, UnpackU32(fy1, 1), 1);

            vst1q_u8(dst, PackSaturatedU16(PackU32(d0, d1), PackU32(d2, d3)));
        }

        template<> SIMD_INLINE void ByteBilinearInterpMainN<3>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst)
        {
            const uint8x16_t SRC_SHUFFLE = SIMD_VEC_SETR_EPI8(0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0xFF, 0xFF, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, 0xFF, 0xFF);
            const uint8x16_t DST_SHUFFLE = SIMD_VEC_SETR_EPI8(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, 0xFF, 0xFF, 0xFF, 0xFF);
            uint8x16_t _fx = UnpackU16(vld1q_u8(fx), 0);
            uint8x16_t fx0 = UnpackU16(_fx, 0);
            uint8x16_t fx1 = UnpackU16(_fx, 1);
            uint16x8_t r00 = MaddU8(Shuffle(vld1q_u8(src0 + 0), SRC_SHUFFLE), fx0);
            uint16x8_t r01 = MaddU8(Shuffle(vld1q_u8(src0 + A), SRC_SHUFFLE), fx1);
            uint16x8_t r10 = MaddU8(Shuffle(vld1q_u8(src1 + 0), SRC_SHUFFLE), fx0);
            uint16x8_t r11 = MaddU8(Shuffle(vld1q_u8(src1 + A), SRC_SHUFFLE), fx1);

            uint16x8_t _fy = vld1q_u16(fy);
            uint16x8_t fy0 = UnpackU32(_fy, 0);
            uint32x4_t d0 = Interp4(r00, r10, UnpackU32(fy0, 0), 0);
            uint32x4_t d1 = Interp4(r00, r10, UnpackU32(fy0, 1), 1);
            uint16x8_t fy1 = UnpackU32(_fy, 1);
            uint32x4_t d2 = Interp4(r01, r11, UnpackU32(fy1, 0), 0);
            uint32x4_t d3 = Interp4(r01, r11, UnpackU32(fy1, 1), 1);

            uint8x16_t value = Shuffle(PackSaturatedU16(PackU32(d0, d1), PackU32(d2, d3)), DST_SHUFFLE);
            vst1_u8(dst, vget_low_u8(value));
            ((uint32_t*)dst)[2] = vgetq_lane_u32(vreinterpretq_u32_u8(value), 2);
        }

        template<> SIMD_INLINE void ByteBilinearInterpMainN<4>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst)
        {
            const uint8x16_t SHUFFLE = SIMD_VEC_SETR_EPI8(0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF);
            uint8x16_t _fx = UnpackU16(vld1q_u8(fx), 0);
            uint8x16_t fx0 = UnpackU16(_fx, 0);
            uint8x16_t fx1 = UnpackU16(_fx, 1);
            uint16x8_t r00 = MaddU8(Shuffle(vld1q_u8(src0 + 0), SHUFFLE), fx0);
            uint16x8_t r01 = MaddU8(Shuffle(vld1q_u8(src0 + A), SHUFFLE), fx1);
            uint16x8_t r10 = MaddU8(Shuffle(vld1q_u8(src1 + 0), SHUFFLE), fx0);
            uint16x8_t r11 = MaddU8(Shuffle(vld1q_u8(src1 + A), SHUFFLE), fx1);

            uint16x8_t _fy = vld1q_u16(fy);
            uint16x8_t fy0 = UnpackU32(_fy, 0);
            uint32x4_t d0 = Interp4(r00, r10, UnpackU32(fy0, 0), 0);
            uint32x4_t d1 = Interp4(r00, r10, UnpackU32(fy0, 1), 1);
            uint16x8_t fy1 = UnpackU32(_fy, 1);
            uint32x4_t d2 = Interp4(r01, r11, UnpackU32(fy1, 0), 0);
            uint32x4_t d3 = Interp4(r01, r11, UnpackU32(fy1, 1), 1);

            vst1q_u8(dst, PackSaturatedU16(PackU32(d0, d1), PackU32(d2, d3)));
        }

        template<int N> void ByteBilinearRun(const WarpAffParam& p, int yBeg, int yEnd, const int* ib, const int* ie, const int* ob, const int* oe, const uint8_t* src, uint8_t* dst, uint8_t* buf)
        {
            constexpr int M = (N == 3 ? 4 : N);
            bool fill = p.NeedFill();
            int width = (int)p.dstW, s = (int)p.srcS, w = (int)p.srcW - 2, h = (int)p.srcH - 2, n = A / M;
            size_t wa = AlignHi(p.dstW, p.align) + p.align;
            uint32_t* offs = (uint32_t*)buf;
            uint8_t* fx = (uint8_t*)(offs + wa);
            uint16_t* fy = (uint16_t*)(fx + wa * 2);
            uint8_t* rb0 = (uint8_t*)(fy + wa * 2);
            uint8_t* rb1 = (uint8_t*)(rb0 + wa * M * 2);
            const float32x4_t _4 = vdupq_n_f32(4.0f);
            const int32x4_t _0123 = SIMD_VEC_SETR_EPI32(0, 1, 2, 3);
            float32x4_t _m[6];
            for (int i = 0; i < 6; ++i)
                _m[i] = vdupq_n_f32(p.inv[i]);
            uint32x4_t _n = vdupq_n_u32(N);
            uint32x4_t _s = vdupq_n_u32(s);
            uint8x16_t _border = InitBorder<N>(p.border);
            dst += yBeg * p.dstS;
            for (int y = yBeg; y < yEnd; ++y)
            {
                int iB = ib[y], iE = ie[y], oB = ob[y], oE = oe[y];
                if (fill)
                {
                    FillBorder<N>(dst, oB, _border, p.border);
                    for (int x = oB; x < iB; ++x)
                        Base::ByteBilinearInterpEdge<N>(x, y, p.inv, w, h, s, src, p.border, dst + x * N);
                }
                else
                {
                    for (int x = oB; x < iB; ++x)
                        Base::ByteBilinearInterpEdge<N>(x, y, p.inv, w, h, s, src, dst + x * N, dst + x * N);
                }
                {
                    int x = iB, iEn = (int)AlignLo(iE - iB, n) + iB;
                    float32x4_t _y = vcvtq_f32_s32(vdupq_n_s32(y));
                    float32x4_t _x = vcvtq_f32_s32(vaddq_s32(vdupq_n_s32(x), _0123));
                    for (; x < iE; x += 4)
                    {
                        ByteBilinearPrepMain4(_x, _y, _m, _n, _s, offs + x, fx + 2 * x, fy + 2 * x);
                        _x = vaddq_f32(_x, _4);
                    }
                    Base::ByteBilinearGather<M>(src, src + s, offs + iB, iE - iB, rb0 + 2 * M * iB, rb1 + 2 * M * iB);
                    for (x = iB; x < iEn; x += n)
                        ByteBilinearInterpMainN<N>(rb0 + x * M * 2, rb1 + x * M * 2, fx + 2 * x, fy + 2 * x, dst + x * N);
                    for (; x < iE; ++x)
                        Base::ByteBilinearInterpMain<N>(rb0 + x * M * 2, rb1 + x * M * 2, fx + 2 * x, fy + 2 * x, dst + x * N);
                }
                if (fill)
                {
                    for (int x = iE; x < oE; ++x)
                        Base::ByteBilinearInterpEdge<N>(x, y, p.inv, w, h, s, src, p.border, dst + x * N);
                    FillBorder<N>(dst + oE * N, width - oE, _border, p.border);
                }
                else
                {
                    for (int x = iE; x < oE; ++x)
                        Base::ByteBilinearInterpEdge<N>(x, y, p.inv, w, h, s, src, dst + x * N, dst + x * N);
                }
                dst += p.dstS;
            }
        }

        //-------------------------------------------------------------------------------------------------

        WarpAffineByteBilinear::WarpAffineByteBilinear(const WarpAffParam& param)
            : Base::WarpAffineByteBilinear(param)
        {
            switch (_param.channels)
            {
            case 1: _run = ByteBilinearRun<1>; break;
            case 2: _run = ByteBilinearRun<2>; break;
            case 3: _run = ByteBilinearRun<3>; break;
            case 4: _run = ByteBilinearRun<4>; break;
            }
        }

        void WarpAffineByteBilinear::SetRange(const Base::Point* rect, int* beg, int* end, const int* lo, const int* hi)
        {
            const WarpAffParam& p = _param;
            float* min = (float*)_buf.data;
            float* max = min + p.dstH;
            float w = (float)p.dstW, h = (float)p.dstH, z = 0.0f;
            const int32x4_t _0123 = SIMD_VEC_SETR_EPI32(0, 1, 2, 3);
            float32x4_t _w = vdupq_n_f32(w), _z = vdupq_n_f32(z);
            int y = 0, dH = (int)p.dstH, dH4 = (int)AlignLo(dH, 4);
            for (; y < dH4; y += 4)
            {
                vst1q_f32(min + y, _w);
                vst1q_f32(max + y, _z);
            }
            for (; y < dH; ++y)
            {
                min[y] = w;
                max[y] = 0;
            }
            for (int v = 0; v < 4; ++v)
            {
                const Base::Point& curr = rect[v];
                const Base::Point& next = rect[(v + 1) & 3];
                if (next.y == curr.y)
                    continue;
                float yMin = Simd::Max(Simd::Min(curr.y, next.y), z);
                float yMax = Simd::Min(Simd::Max(curr.y, next.y), h);
                int yBeg = (int)ceil(yMin);
                int yEnd = (int)ceil(yMax);
                int yEnd4 = (int)AlignLo(yEnd - yBeg, 4) + yBeg;
                float a = (next.x - curr.x) / (next.y - curr.y);
                float b = curr.x - curr.y * a;
                float32x4_t _a = vdupq_n_f32(a);
                float32x4_t _b = vdupq_n_f32(b);
                float32x4_t _yMin = vdupq_n_f32(yMin);
                float32x4_t _yMax = vdupq_n_f32(yMax);
                for (y = yBeg; y < yEnd4; y += 4)
                {
                    float32x4_t _y = vcvtq_f32_s32(vaddq_s32(vdupq_n_s32(y), _0123));
                    _y = vminq_f32(_yMax, vmaxq_f32(_y, _yMin));
                    float32x4_t _x = vaddq_f32(vmulq_f32(_y, _a), _b);
                    vst1q_f32(min + y, vminq_f32(vld1q_f32(min + y), vmaxq_f32(_x, _z)));
                    vst1q_f32(max + y, vmaxq_f32(vld1q_f32(max + y), vminq_f32(_x, _w)));
                }
                for (; y < yEnd; ++y)
                {
                    float x = Simd::RestrictRange(float(y), yMin, yMax) * a + b;
                    min[y] = Simd::Min(min[y], Simd::Max(x, z));
                    max[y] = Simd::Max(max[y], Simd::Min(x, w));
                }
            }
            for (y = 0; y < dH4; y += 4)
            {
                int32x4_t _beg = vcvtq_s32_f32(Ceil(vld1q_f32(min + y)));
                int32x4_t _end = vcvtq_s32_f32(Ceil(vld1q_f32(max + y)));
                vst1q_s32(beg + y, _beg);
                vst1q_s32(end + y, vmaxq_s32(_beg, _end));
            }
            for (; y < dH; ++y)
            {
                beg[y] = (int)ceil(min[y]);
                end[y] = (int)ceil(max[y]);
                end[y] = Simd::Max(beg[y], end[y]);
            }
            if (hi)
            {
                for (y = 0; y < dH4; y += 4)
                {
                    int32x4_t _hi = vld1q_s32(hi + y);
                    vst1q_s32(beg + y, vminq_s32(vld1q_s32(beg + y), _hi));
                    vst1q_s32(end + y, vminq_s32(vld1q_s32(end + y), _hi));
                }
                for (; y < dH; ++y)
                {
                    beg[y] = Simd::Min(beg[y], hi[y]);
                    end[y] = Simd::Min(end[y], hi[y]);
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        void* WarpAffineInit(size_t srcW, size_t srcH, size_t srcS, size_t dstW, size_t dstH, size_t dstS, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border)
        {
            WarpAffParam param(srcW, srcH, srcS, dstW, dstH, dstS, channels, mat, flags, border, A);
            if (!param.Valid())
                return NULL;
            if (param.IsNearest())
                return new WarpAffineNearest(param);
            else if (param.IsByteBilinear())
                return new WarpAffineByteBilinear(param);
            else
                return Base::WarpAffineInit(srcW, srcH, srcS, dstW, dstH, dstS, channels, mat, flags, border);
        }
    }
#endif
}
