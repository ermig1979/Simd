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

#include "Simd/SimdPoint.hpp"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svint32_t Round(svbool_t mask, svfloat32_t value)
        {
            svbool_t positive = svcmpge_n_f32(mask, value, 0.0f);
            svfloat32_t round = svsel_f32(positive, svdup_n_f32(0.5f), svdup_n_f32(-0.5f));
            return svcvt_s32_f32_x(mask, svadd_f32_x(mask, value, round));
        }

        template<int N> SIMD_INLINE void FillBorder(uint8_t* dst, int count, const svuint8_t& bv, const uint8_t* bs)
        {
            size_t i = 0, size = count * N, A = svcntb(), sizeA = AlignLo(size, A);
            const svbool_t body = svptrue_b8();
            for (; i < sizeA; i += A)
                svst1_u8(body, dst + i, bv);
            for (; i < size; i += N)
                Base::CopyPixel<N>(bs, dst + i);
        }

        template<> SIMD_INLINE void FillBorder<3>(uint8_t* dst, int count, const svuint8_t& bv, const uint8_t* bs)
        {
            int i = 0, size = count * 3, size3 = size - 3;
            for (; i < size3; i += 3)
                Base::CopyPixel<4>(bs, dst + i);
            for (; i < size; i += 3)
                Base::CopyPixel<3>(bs, dst + i);
        }

        template<int N> SIMD_INLINE svuint8_t InitBorder(const uint8_t* border)
        {
            switch (N)
            {
            case 1: return svdup_n_u8(*border);
            case 2: return svreinterpret_u8_u16(svdup_n_u16(*(uint16_t*)border));
            case 3: return svdup_n_u8(0);
            case 4: return svreinterpret_u8_u32(svdup_n_u32(*(uint32_t*)border));
            }
            return svdup_n_u8(0);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE svuint32_t NearestOffset(svbool_t mask, svfloat32_t x, svfloat32_t y, const svfloat32_t& m0, const svfloat32_t& m1, const svfloat32_t& m2,
            const svfloat32_t& m3, const svfloat32_t& m4, const svfloat32_t& m5, const svint32_t& w, const svint32_t& h, int n, int s)
        {
            svfloat32_t dx = svmla_f32_x(mask, svmla_f32_x(mask, m2, x, m0), y, m1);
            svfloat32_t dy = svmla_f32_x(mask, svmla_f32_x(mask, m5, x, m3), y, m4);
            svint32_t ix = svmin_s32_x(mask, svmax_n_s32_x(mask, Round(mask, dx), 0), w);
            svint32_t iy = svmin_s32_x(mask, svmax_n_s32_x(mask, Round(mask, dy), 0), h);
            return svreinterpret_u32_s32(svmla_n_s32_x(mask, svmul_n_s32_x(mask, iy, s), ix, n));
        }

        //-----------------------------------------------------------------------------------------

        template<int N> void NearestRun(const WarpAffParam& p, int yBeg, int yEnd, const int32_t* beg, const int32_t* end, const uint8_t* src, uint8_t* dst, uint32_t* buf)
        {
            bool fill = p.NeedFill();
            int width = (int)p.dstW, s = (int)p.srcS, w = (int)p.srcW - 1, h = (int)p.srcH - 1;
            size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            svfloat32_t m0 = svdup_n_f32(p.inv[0]);
            svfloat32_t m1 = svdup_n_f32(p.inv[1]);
            svfloat32_t m2 = svdup_n_f32(p.inv[2]);
            svfloat32_t m3 = svdup_n_f32(p.inv[3]);
            svfloat32_t m4 = svdup_n_f32(p.inv[4]);
            svfloat32_t m5 = svdup_n_f32(p.inv[5]);
            svint32_t _w = svdup_n_s32(w);
            svint32_t _h = svdup_n_s32(h);
            svuint8_t _border = InitBorder<N>(p.border);
            dst += yBeg * p.dstS;
            for (int y = yBeg; y < yEnd; ++y)
            {
                int nose = beg[y], tail = end[y];
                {
                    int x = nose;
                    svfloat32_t _y = svdup_n_f32((float)y);
                    for (; x < tail; x += (int)F)
                    {
                        svfloat32_t _x = svcvt_f32_u32_x(body, svindex_u32((uint32_t)x, 1));
                        svst1_u32(body, buf + x, NearestOffset(body, _x, _y, m0, m1, m2, m3, m4, m5, _w, _h, N, s));
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
#ifdef SIMD_NEON_ENABLE
            : Neon::WarpAffineNearest(param)
#else
            : Base::WarpAffineNearest(param)
#endif
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
            int w = (int)p.dstW, h = (int)p.dstH, F = (int)svcntw(), hF = (int)AlignLo(h, F);
            const svbool_t body = svptrue_b32();
            svint32_t _w = svdup_n_s32(w), _1 = svdup_n_s32(1), _0 = svdup_n_s32(0);
            int y = 0;
            for (; y < hF; y += F)
            {
                svst1_s32(body, _beg.data + y, _w);
                svst1_s32(body, _end.data + y, _0);
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
                int yEndF = (int)AlignLo(yEnd - yBeg, F) + yBeg;
                if (next.y == curr.y)
                    continue;
                float a = (next.x - curr.x) / (next.y - curr.y);
                float b = curr.x - curr.y * a;
                svfloat32_t _a = svdup_n_f32(a);
                svfloat32_t _b = svdup_n_f32(b);
                if (abs(a) <= 1.0f)
                {
                    int y = yBeg;
                    for (; y < yEndF; y += F)
                    {
                        svfloat32_t _y = svcvt_f32_u32_x(body, svindex_u32((uint32_t)y, 1));
                        svint32_t _x = Round(body, svmla_f32_x(body, _b, _y, _a));
                        svint32_t xBeg = svld1_s32(body, _beg.data + y);
                        svint32_t xEnd = svld1_s32(body, _end.data + y);
                        xBeg = svmin_s32_x(body, xBeg, svmax_s32_x(body, _x, _0));
                        xEnd = svmax_s32_x(body, xEnd, svmin_s32_x(body, svadd_s32_x(body, _x, _1), _w));
                        svst1_s32(body, _beg.data + y, xBeg);
                        svst1_s32(body, _end.data + y, xEnd);
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
                    svfloat32_t _05 = svdup_n_f32(0.5f);
                    svfloat32_t _yMin = svdup_n_f32(yMin);
                    svfloat32_t _yMax = svdup_n_f32(yMax);
                    for (; y < yEndF; y += F)
                    {
                        svfloat32_t _y = svcvt_f32_u32_x(body, svindex_u32((uint32_t)y, 1));
                        svfloat32_t yM = svmin_f32_x(body, svmax_f32_x(body, svsub_f32_x(body, _y, _05), _yMin), _yMax);
                        svfloat32_t yP = svmin_f32_x(body, svmax_f32_x(body, svadd_f32_x(body, _y, _05), _yMin), _yMax);
                        svfloat32_t xM = svmla_f32_x(body, _b, yM, _a);
                        svfloat32_t xP = svmla_f32_x(body, _b, yP, _a);
                        svint32_t xBeg = svld1_s32(body, _beg.data + y);
                        svint32_t xEnd = svld1_s32(body, _end.data + y);
                        xBeg = svmin_s32_x(body, xBeg, svmax_s32_x(body, Round(body, svmin_f32_x(body, xM, xP)), _0));
                        xEnd = svmax_s32_x(body, xEnd, svmin_s32_x(body, svadd_s32_x(body, Round(body, svmax_f32_x(body, xM, xP)), _1), _w));
                        svst1_s32(body, _beg.data + y, xBeg);
                        svst1_s32(body, _end.data + y, xEnd);
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

        SIMD_INLINE svfloat32_t Floor(svbool_t mask, svfloat32_t value)
        {
            return svrintm_f32_x(mask, value);
        }

        SIMD_INLINE svfloat32_t Ceil(svbool_t mask, svfloat32_t value)
        {
            return svrintp_f32_x(mask, value);
        }

        SIMD_INLINE svuint32_t RoundPositive(svbool_t mask, svfloat32_t value)
        {
            return svcvt_u32_f32_x(mask, svadd_n_f32_x(mask, value, 0.5f));
        }

        SIMD_INLINE void ByteBilinearPrepMain(svbool_t mask, svfloat32_t x, svfloat32_t y, const svfloat32_t& m0, const svfloat32_t& m1, const svfloat32_t& m2,
            const svfloat32_t& m3, const svfloat32_t& m4, const svfloat32_t& m5, int n, int s, uint32_t* offs, uint8_t* fx, uint16_t* fy)
        {
            svfloat32_t dx = svmla_f32_x(mask, svmla_f32_x(mask, m2, x, m0), y, m1);
            svfloat32_t dy = svmla_f32_x(mask, svmla_f32_x(mask, m5, x, m3), y, m4);
            svfloat32_t ix = Floor(mask, dx);
            svfloat32_t iy = Floor(mask, dy);
            svfloat32_t range = svdup_n_f32((float)Base::WA_FRACTION_RANGE);
            svuint32_t _fx = RoundPositive(mask, svmul_f32_x(mask, svsub_f32_x(mask, dx, ix), range));
            svuint32_t _fy = RoundPositive(mask, svmul_f32_x(mask, svsub_f32_x(mask, dy, iy), range));
            svst1_u32(mask, offs, svmla_n_u32_x(mask, svmul_n_u32_x(mask, svcvt_u32_f32_x(mask, iy), s), svcvt_u32_f32_x(mask, ix), n));

            svuint16_t fx0 = svqxtnb_u32(svsub_u32_x(mask, svdup_n_u32(Base::WA_FRACTION_RANGE), _fx));
            svuint16_t fx1 = svqxtnb_u32(_fx);
            svst1_u8(svwhilelt_b8((size_t)0, 2 * svcntw()), fx, svqxtnb_u16(svzip1_u16(fx0, fx1)));

            svuint16_t fy0 = svqxtnb_u32(svsub_u32_x(mask, svdup_n_u32(Base::WA_FRACTION_RANGE), _fy));
            svuint16_t fy1 = svqxtnb_u32(_fy);
            svst1_u16(svptrue_b16(), fy, svzip1_u16(fy0, fy1));
        }

        SIMD_INLINE svuint8_t Shuffle2(svuint8_t value)
        {
            const svbool_t mask = svptrue_b8();
            svuint8_t idx = svindex_u8(0, 1);
            svuint8_t lo = svlsl_n_u8_x(mask, svand_n_u8_x(mask, idx, 0x01), 1);
            svuint8_t hi = svlsr_n_u8_x(mask, svand_n_u8_x(mask, idx, 0x02), 1);
            return svtbl_u8(value, svadd_u8_x(mask, svand_n_u8_x(mask, idx, 0xFC), svadd_u8_x(mask, lo, hi)));
        }

        SIMD_INLINE svuint8_t Shuffle4(svuint8_t value)
        {
            const svbool_t mask = svptrue_b8();
            svuint8_t idx = svindex_u8(0, 1);
            svuint8_t lo = svlsl_n_u8_x(mask, svand_n_u8_x(mask, idx, 0x03), 1);
            svuint8_t hi = svlsr_n_u8_x(mask, svand_n_u8_x(mask, idx, 0x04), 2);
            return svtbl_u8(value, svadd_u8_x(mask, svand_n_u8_x(mask, idx, 0xF8), svadd_u8_x(mask, lo, hi)));
        }

        SIMD_INLINE svuint8_t UnpackU16(svuint8_t value, int part)
        {
            svuint16_t _value = svreinterpret_u16_u8(value);
            return svreinterpret_u8_u16(part ? svzip2_u16(_value, _value) : svzip1_u16(_value, _value));
        }

        SIMD_INLINE svuint16_t UnpackU32(svuint16_t value, int part)
        {
            svuint32_t _value = svreinterpret_u32_u16(value);
            return svreinterpret_u16_u32(part ? svzip2_u32(_value, _value) : svzip1_u32(_value, _value));
        }

        SIMD_INLINE svuint16_t MaddU8(svuint8_t value, svuint8_t weight)
        {
            const svbool_t mask = svptrue_b16();
            svuint16_t lo = svmul_u16_x(mask, svunpklo_u16(svuzp1_u8(value, value)), svunpklo_u16(svuzp1_u8(weight, weight)));
            return svmla_u16_x(mask, lo, svunpklo_u16(svuzp2_u8(value, value)), svunpklo_u16(svuzp2_u8(weight, weight)));
        }

        SIMD_INLINE svuint16_t PackU32ToU16(svuint32_t lo, svuint32_t hi)
        {
            return svqxtnt_u32(svqxtnb_u32(lo), hi);
        }

        SIMD_INLINE svuint8_t PackU16ToU8(svuint16_t lo, svuint16_t hi)
        {
            return svuzp1_u8(svqxtnb_u16(lo), svqxtnb_u16(hi));
        }

        SIMD_INLINE svuint16_t Interp(svuint16_t src0, svuint16_t src1, svuint16_t fy)
        {
            const svbool_t mask = svptrue_b32();
            svuint16_t fy0 = svuzp1_u16(fy, fy);
            svuint16_t fy1 = svuzp2_u16(fy, fy);
            svuint32_t lo = svmul_u32_x(mask, svunpklo_u32(src0), svunpklo_u32(fy0));
            svuint32_t hi = svmul_u32_x(mask, svunpkhi_u32(src0), svunpkhi_u32(fy0));
            lo = svmla_u32_x(mask, lo, svunpklo_u32(src1), svunpklo_u32(fy1));
            hi = svmla_u32_x(mask, hi, svunpkhi_u32(src1), svunpkhi_u32(fy1));
            lo = svlsr_n_u32_x(mask, svadd_n_u32_x(mask, lo, Base::WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);
            hi = svlsr_n_u32_x(mask, svadd_n_u32_x(mask, hi, Base::WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);
            return PackU32ToU16(lo, hi);
        }

        template<int N> void ByteBilinearInterpMainN(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst);

        template<> SIMD_INLINE void ByteBilinearInterpMainN<1>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst)
        {
            const svbool_t mask8 = svptrue_b8(), mask16 = svptrue_b16();
            size_t A = svcntb(), HA = svcnth();
            svuint8_t fx0 = svld1_u8(mask8, fx + 0);
            svuint8_t fx1 = svld1_u8(mask8, fx + A);
            svuint16_t r00 = MaddU8(svld1_u8(mask8, src0 + 0), fx0);
            svuint16_t r01 = MaddU8(svld1_u8(mask8, src0 + A), fx1);
            svuint16_t r10 = MaddU8(svld1_u8(mask8, src1 + 0), fx0);
            svuint16_t r11 = MaddU8(svld1_u8(mask8, src1 + A), fx1);
            svuint16_t d0 = Interp(r00, r10, svld1_u16(mask16, fy + 0 * HA));
            svuint16_t d1 = Interp(r01, r11, svld1_u16(mask16, fy + 1 * HA));
            svst1_u8(mask8, dst, PackU16ToU8(d0, d1));
        }

        template<> SIMD_INLINE void ByteBilinearInterpMainN<2>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst)
        {
            const svbool_t mask8 = svptrue_b8(), mask16 = svptrue_b16();
            size_t A = svcntb(), HA = svcnth();
            svuint8_t _fx = svld1_u8(mask8, fx);
            svuint8_t fx0 = UnpackU16(_fx, 0);
            svuint8_t fx1 = UnpackU16(_fx, 1);
            svuint16_t r00 = MaddU8(Shuffle2(svld1_u8(mask8, src0 + 0)), fx0);
            svuint16_t r01 = MaddU8(Shuffle2(svld1_u8(mask8, src0 + A)), fx1);
            svuint16_t r10 = MaddU8(Shuffle2(svld1_u8(mask8, src1 + 0)), fx0);
            svuint16_t r11 = MaddU8(Shuffle2(svld1_u8(mask8, src1 + A)), fx1);
            svuint16_t fy0 = svld1_u16(mask16, fy + 0 * HA);
            svuint16_t fy1 = svld1_u16(mask16, fy + 1 * HA);
            svuint16_t d0 = Interp(r00, r10, UnpackU32(fy0, 0));
            svuint16_t d1 = Interp(r01, r11, UnpackU32(fy1, 0));
            svst1_u8(mask8, dst, PackU16ToU8(d0, d1));
        }

        template<> SIMD_INLINE void ByteBilinearInterpMainN<3>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst)
        {
        }

        template<> SIMD_INLINE void ByteBilinearInterpMainN<4>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst)
        {
            const svbool_t mask8 = svptrue_b8(), mask16 = svptrue_b16();
            size_t A = svcntb();
            svuint8_t _fx = UnpackU16(svld1_u8(mask8, fx), 0);
            svuint8_t fx0 = UnpackU16(_fx, 0);
            svuint8_t fx1 = UnpackU16(_fx, 1);
            svuint16_t r00 = MaddU8(Shuffle4(svld1_u8(mask8, src0 + 0)), fx0);
            svuint16_t r01 = MaddU8(Shuffle4(svld1_u8(mask8, src0 + A)), fx1);
            svuint16_t r10 = MaddU8(Shuffle4(svld1_u8(mask8, src1 + 0)), fx0);
            svuint16_t r11 = MaddU8(Shuffle4(svld1_u8(mask8, src1 + A)), fx1);
            svuint16_t _fy = svld1_u16(mask16, fy);
            svuint16_t fy0 = UnpackU32(_fy, 0);
            svuint16_t fy1 = UnpackU32(_fy, 1);
            svuint16_t d0 = Interp(r00, r10, fy0);
            svuint16_t d1 = Interp(r01, r11, fy1);
            svst1_u8(mask8, dst, PackU16ToU8(d0, d1));
        }

        template<int N> void ByteBilinearRun(const WarpAffParam& p, int yBeg, int yEnd, const int* ib, const int* ie, const int* ob, const int* oe, const uint8_t* src, uint8_t* dst, uint8_t* buf)
        {
            constexpr int M = (N == 3 ? 4 : N);
            bool fill = p.NeedFill();
            int width = (int)p.dstW, s = (int)p.srcS, w = (int)p.srcW - 2, h = (int)p.srcH - 2;
            size_t wa = AlignHi(p.dstW, p.align) + p.align;
            uint32_t* offs = (uint32_t*)buf;
            uint8_t* fx = (uint8_t*)(offs + wa);
            uint16_t* fy = (uint16_t*)(fx + wa * 2);
            uint8_t* rb0 = (uint8_t*)(fy + wa * 2);
            uint8_t* rb1 = (uint8_t*)(rb0 + wa * M * 2);
            svuint8_t _border = InitBorder<N>(p.border);
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
                    int x = iB;
                    for (; x < iE; ++x)
                        Base::ByteBilinearPrepMain(x, y, p.inv, N, s, offs + x, fx + 2 * x, fy + 2 * x);
                    Base::ByteBilinearGather<M>(src, src + s, offs + iB, iE - iB, rb0 + 2 * M * iB, rb1 + 2 * M * iB);
                    for (x = iB; x < iE; ++x)
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
#ifdef SIMD_NEON_ENABLE
            : Neon::WarpAffineByteBilinear(param)
#else
            : Base::WarpAffineByteBilinear(param)
#endif
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
            const svbool_t mask = svptrue_b32();
            svfloat32_t _w = svdup_n_f32(w), _z = svdup_n_f32(z);
            int y = 0, dH = (int)p.dstH, F = (int)svcntw(), dHF = (int)AlignLo(dH, F);
            for (; y < dHF; y += F)
            {
                svst1_f32(mask, min + y, _w);
                svst1_f32(mask, max + y, _z);
            }
            for (; y < dH; ++y)
            {
                min[y] = w;
                max[y] = z;
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
                int yEndF = (int)AlignLo(yEnd - yBeg, F) + yBeg;
                float a = (next.x - curr.x) / (next.y - curr.y);
                float b = curr.x - curr.y * a;
                svfloat32_t _a = svdup_n_f32(a);
                svfloat32_t _b = svdup_n_f32(b);
                svfloat32_t _yMin = svdup_n_f32(yMin);
                svfloat32_t _yMax = svdup_n_f32(yMax);
                for (y = yBeg; y < yEndF; y += F)
                {
                    svfloat32_t _y = svcvt_f32_u32_x(mask, svindex_u32((uint32_t)y, 1));
                    _y = svmin_f32_x(mask, _yMax, svmax_f32_x(mask, _y, _yMin));
                    svfloat32_t _x = svmla_f32_x(mask, _b, _y, _a);
                    svst1_f32(mask, min + y, svmin_f32_x(mask, svld1_f32(mask, min + y), svmax_f32_x(mask, _x, _z)));
                    svst1_f32(mask, max + y, svmax_f32_x(mask, svld1_f32(mask, max + y), svmin_f32_x(mask, _x, _w)));
                }
                for (; y < yEnd; ++y)
                {
                    float x = Simd::RestrictRange(float(y), yMin, yMax) * a + b;
                    min[y] = Simd::Min(min[y], Simd::Max(x, z));
                    max[y] = Simd::Max(max[y], Simd::Min(x, w));
                }
            }
            for (y = 0; y < dHF; y += F)
            {
                svint32_t _beg = svcvt_s32_f32_x(mask, Ceil(mask, svld1_f32(mask, min + y)));
                svint32_t _end = svcvt_s32_f32_x(mask, Ceil(mask, svld1_f32(mask, max + y)));
                svst1_s32(mask, beg + y, _beg);
                svst1_s32(mask, end + y, svmax_s32_x(mask, _beg, _end));
            }
            for (; y < dH; ++y)
            {
                beg[y] = (int)ceil(min[y]);
                end[y] = (int)ceil(max[y]);
                end[y] = Simd::Max(beg[y], end[y]);
            }
            if (hi)
            {
                for (y = 0; y < dHF; y += F)
                {
                    svint32_t _hi = svld1_s32(mask, hi + y);
                    svst1_s32(mask, beg + y, svmin_s32_x(mask, svld1_s32(mask, beg + y), _hi));
                    svst1_s32(mask, end + y, svmin_s32_x(mask, svld1_s32(mask, end + y), _hi));
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
            WarpAffParam param(srcW, srcH, srcS, dstW, dstH, dstS, channels, mat, flags, border, svcntb());
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
