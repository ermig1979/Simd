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
#include "Simd/SimdMemory.h"
#include "Simd/SimdYuvToBgr.h"
#include "Simd/SimdPack.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        template<class T> SIMD_INLINE svint16_t AdjustYLo(const svuint8_t& y)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlb_u16(y)), T::Y_LO);
        }

        template<class T> SIMD_INLINE svint16_t AdjustYHi(const svuint8_t& y)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlt_u16(y)), T::Y_LO);
        }

        template<class T> SIMD_INLINE svint16_t AdjustUVLo(const svuint8_t& uv)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlb_u16(uv)), T::UV_Z);
        }

        template<class T> SIMD_INLINE svint16_t AdjustUVHi(const svuint8_t& uv)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlt_u16(uv)), T::UV_Z);
        }

        template<class T> SIMD_INLINE svint16_t YuvToBlue16(const svint16_t& y, const svint16_t& u)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t lo = svmlalb_n_s32(svdup_n_s32(T::F_ROUND), y, T::Y_2_A);
            svint32_t hi = svmlalt_n_s32(svdup_n_s32(T::F_ROUND), y, T::Y_2_A);
            lo = svmlalb_n_s32(lo, u, T::U_2_B);
            hi = svmlalt_n_s32(hi, u, T::U_2_B);
            return svqxtnt_s32(svqxtnb_s32(svasr_n_s32_x(mask, lo, T::F_SHIFT)), svasr_n_s32_x(mask, hi, T::F_SHIFT));
        }

        template<class T> SIMD_INLINE svint16_t YuvToGreen16(const svint16_t& y, const svint16_t& u, const svint16_t& v)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t lo = svmlalb_n_s32(svdup_n_s32(T::F_ROUND), y, T::Y_2_A);
            svint32_t hi = svmlalt_n_s32(svdup_n_s32(T::F_ROUND), y, T::Y_2_A);
            lo = svmlalb_n_s32(svmlalb_n_s32(lo, u, T::U_2_G), v, T::V_2_G);
            hi = svmlalt_n_s32(svmlalt_n_s32(hi, u, T::U_2_G), v, T::V_2_G);
            return svqxtnt_s32(svqxtnb_s32(svasr_n_s32_x(mask, lo, T::F_SHIFT)), svasr_n_s32_x(mask, hi, T::F_SHIFT));
        }

        template<class T> SIMD_INLINE svint16_t YuvToRed16(const svint16_t& y, const svint16_t& v)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t lo = svmlalb_n_s32(svdup_n_s32(T::F_ROUND), y, T::Y_2_A);
            svint32_t hi = svmlalt_n_s32(svdup_n_s32(T::F_ROUND), y, T::Y_2_A);
            lo = svmlalb_n_s32(lo, v, T::V_2_R);
            hi = svmlalt_n_s32(hi, v, T::V_2_R);
            return svqxtnt_s32(svqxtnb_s32(svasr_n_s32_x(mask, lo, T::F_SHIFT)), svasr_n_s32_x(mask, hi, T::F_SHIFT));
        }

        template<class T> SIMD_INLINE svuint8_t YuvToBlue(const svuint8_t& y, const svuint8_t& u)
        {
            return PackSatIntI16ToU8(
                YuvToBlue16<T>(AdjustYLo<T>(y), AdjustUVLo<T>(u)),
                YuvToBlue16<T>(AdjustYHi<T>(y), AdjustUVHi<T>(u)));
        }

        template<class T> SIMD_INLINE svuint8_t YuvToGreen(const svuint8_t& y, const svuint8_t& u, const svuint8_t& v)
        {
            return PackSatIntI16ToU8(
                YuvToGreen16<T>(AdjustYLo<T>(y), AdjustUVLo<T>(u), AdjustUVLo<T>(v)),
                YuvToGreen16<T>(AdjustYHi<T>(y), AdjustUVHi<T>(u), AdjustUVHi<T>(v)));
        }

        template<class T> SIMD_INLINE svuint8_t YuvToRed(const svuint8_t& y, const svuint8_t& v)
        {
            return PackSatIntI16ToU8(
                YuvToRed16<T>(AdjustYLo<T>(y), AdjustUVLo<T>(v)),
                YuvToRed16<T>(AdjustYHi<T>(y), AdjustUVHi<T>(v)));
        }

        template <class T> SIMD_INLINE void Yuv422pToBgrV2(const uint8_t* y, const svuint8_t& u, const svuint8_t& v, uint8_t* bgr)
        {
            const svbool_t body = svptrue_b8();
            const size_t A = svcntb(), A3 = A * 3;
            svuint8_t _y = svld1_u8(body, y);
            svuint8_t blue = YuvToBlue<T>(_y, u);
            svuint8_t green = YuvToGreen<T>(_y, u, v);
            svuint8_t red = YuvToRed<T>(_y, v);
            svst3_u8(body, bgr + 0 * A3, svcreate3_u8(blue, green, red));
        }

        template <class T> SIMD_INLINE void Yuv422pToBgr(const uint8_t* y, int u, int v, uint8_t* bgr)
        {
            Base::YuvToBgr<T>(y[0], u, v, bgr + 0);
            Base::YuvToBgr<T>(y[1], u, v, bgr + 3);
        }

        template <class T> SIMD_INLINE void Yuv422pToRgbV2(const uint8_t* y, const svuint8_t& u, const svuint8_t& v, uint8_t* rgb)
        {
            const svbool_t body = svptrue_b8();
            const size_t A = svcntb(), A3 = A * 3;
            svuint8_t _y = svld1_u8(body, y);
            svuint8_t blue = YuvToBlue<T>(_y, u);
            svuint8_t green = YuvToGreen<T>(_y, u, v);
            svuint8_t red = YuvToRed<T>(_y, v);
            svst3_u8(body, rgb + 0 * A3, svcreate3_u8(red, green, blue));
        }

        template <class T> SIMD_INLINE void Yuv422pToRgb(const uint8_t* y, int u, int v, uint8_t* rgb)
        {
            Base::YuvToRgb<T>(y[0], u, v, rgb + 0);
            Base::YuvToRgb<T>(y[1], u, v, rgb + 3);
        }

        template <class T> SIMD_INLINE void Yuv444pToBgrV2(const uint8_t* y, const uint8_t* u, const uint8_t* v, uint8_t* bgr)
        {
            const svbool_t body = svptrue_b8();
            const size_t A = svcntb(), A3 = A * 3;
            svuint8_t _y = svld1_u8(body, y);
            svuint8_t _u = svld1_u8(body, u);
            svuint8_t _v = svld1_u8(body, v);
            svuint8_t blue = YuvToBlue<T>(_y, _u);
            svuint8_t green = YuvToGreen<T>(_y, _u, _v);
            svuint8_t red = YuvToRed<T>(_y, _v);
            svst3_u8(body, bgr + 0 * A3, svcreate3_u8(blue, green, red));
        }

        template <class T> void Yuv444pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            const size_t A = svcntb(), A3 = A * 3;
            const size_t widthA = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, colBgr = 0;
                for (; col < widthA; col += A, colBgr += A3)
                    Yuv444pToBgrV2<T>(y + col, u + col, v + col, bgr + colBgr);
                for (; col < width; ++col, colBgr += 3)
                    Base::YuvToBgr<T>(y[col], u[col], v[col], bgr + colBgr);
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void Yuv444pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv444pToBgrV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt709: Yuv444pToBgrV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt2020: Yuv444pToBgrV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvTrect871: Yuv444pToBgrV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            default:
                assert(0);
            }
        }

        template <class T> void Yuv420pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            const size_t A = svcntb(), A3 = 3 * A, DA = 2 * A, A6 = 6 * A;
            const size_t widthDA = AlignLo(width, DA);
            const svbool_t body = svptrue_b8();
            for (size_t row = 0; row < height; row += 2)
            {
                size_t colUV = 0, colY = 0, colBgr = 0;
                for (; colY < widthDA; colY += DA, colUV += A, colBgr += A6)
                {
                    svuint8_t _u = svld1_u8(body, u + colUV);
                    svuint8_t _v = svld1_u8(body, v + colUV);
                    svuint8_t u0 = svzip1_u8(_u, _u);
                    svuint8_t u1 = svzip2_u8(_u, _u);
                    svuint8_t v0 = svzip1_u8(_v, _v);
                    svuint8_t v1 = svzip2_u8(_v, _v);
                    Yuv422pToBgrV2<T>(y + colY + 0 * A, u0, v0, bgr + colBgr + 0 * A3);
                    Yuv422pToBgrV2<T>(y + colY + 1 * A, u1, v1, bgr + colBgr + 1 * A3);
                    Yuv422pToBgrV2<T>(y + yStride + colY + 0 * A, u0, v0, bgr + bgrStride + colBgr + 0 * A3);
                    Yuv422pToBgrV2<T>(y + yStride + colY + 1 * A, u1, v1, bgr + bgrStride + colBgr + 1 * A3);
                }
                for (; colY < width; colY += 2, colUV += 1, colBgr += 6)
                {
                    int _u = u[colUV], _v = v[colUV];
                    Yuv422pToBgr<T>(y + colY, _u, _v, bgr + colBgr);
                    Yuv422pToBgr<T>(y + yStride + colY, _u, _v, bgr + bgrStride + colBgr);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgr += 2 * bgrStride;
            }
        }

        void Yuv420pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv420pToBgrV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt709: Yuv420pToBgrV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt2020: Yuv420pToBgrV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvTrect871: Yuv420pToBgrV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            default:
                assert(0);
            }
        }

        template <class T> void Yuv420pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            const size_t A = svcntb(), A3 = 3 * A, DA = 2 * A, A6 = 6 * A;
            const size_t widthDA = AlignLo(width, DA);
            const svbool_t body = svptrue_b8();
            for (size_t row = 0; row < height; row += 2)
            {
                size_t colUV = 0, colY = 0, colRgb = 0;
                for (; colY < widthDA; colY += DA, colUV += A, colRgb += A6)
                {
                    svuint8_t _u = svld1_u8(body, u + colUV);
                    svuint8_t _v = svld1_u8(body, v + colUV);
                    svuint8_t u0 = svzip1_u8(_u, _u);
                    svuint8_t u1 = svzip2_u8(_u, _u);
                    svuint8_t v0 = svzip1_u8(_v, _v);
                    svuint8_t v1 = svzip2_u8(_v, _v);
                    Yuv422pToRgbV2<T>(y + colY + 0 * A, u0, v0, rgb + colRgb + 0 * A3);
                    Yuv422pToRgbV2<T>(y + colY + 1 * A, u1, v1, rgb + colRgb + 1 * A3);
                    Yuv422pToRgbV2<T>(y + yStride + colY + 0 * A, u0, v0, rgb + rgbStride + colRgb + 0 * A3);
                    Yuv422pToRgbV2<T>(y + yStride + colY + 1 * A, u1, v1, rgb + rgbStride + colRgb + 1 * A3);
                }
                for (; colY < width; colY += 2, colUV += 1, colRgb += 6)
                {
                    int _u = u[colUV], _v = v[colUV];
                    Yuv422pToRgb<T>(y + colY, _u, _v, rgb + colRgb);
                    Yuv422pToRgb<T>(y + yStride + colY, _u, _v, rgb + rgbStride + colRgb);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                rgb += 2 * rgbStride;
            }
        }

        void Yuv420pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv420pToRgbV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvBt709: Yuv420pToRgbV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvBt2020: Yuv420pToRgbV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvTrect871: Yuv420pToRgbV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            default:
                assert(0);
            }
        }

        template <class T> void Yuv422pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (width >= 2));

            const size_t A = svcntb(), A3 = 3 * A, DA = 2 * A, A6 = 6 * A;
            const size_t widthDA = AlignLo(width, DA);
            const svbool_t body = svptrue_b8();
            for (size_t row = 0; row < height; ++row)
            {
                size_t colUV = 0, colY = 0, colBgr = 0;
                for (; colY < widthDA; colY += DA, colUV += A, colBgr += A6)
                {
                    svuint8_t _u = svld1_u8(body, u + colUV);
                    svuint8_t _v = svld1_u8(body, v + colUV);
                    Yuv422pToBgrV2<T>(y + colY + 0 * A, svzip1_u8(_u, _u), svzip1_u8(_v, _v), bgr + colBgr + 0 * A3);
                    Yuv422pToBgrV2<T>(y + colY + 1 * A, svzip2_u8(_u, _u), svzip2_u8(_v, _v), bgr + colBgr + 1 * A3);
                }
                for (; colY < width; colY += 2, colUV += 1, colBgr += 6)
                    Yuv422pToBgr<T>(y + colY, u[colUV], v[colUV], bgr + colBgr);
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void Yuv422pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv422pToBgrV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt709: Yuv422pToBgrV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt2020: Yuv422pToBgrV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvTrect871: Yuv422pToBgrV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            default:
                assert(0);
            }
        }

        template <class T> void Yuv422pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            assert((width % 2 == 0) && (width >= 2));

            const size_t A = svcntb(), A3 = 3 * A, DA = 2 * A, A6 = 6 * A;
            const size_t widthDA = AlignLo(width, DA);
            const svbool_t body = svptrue_b8();
            for (size_t row = 0; row < height; ++row)
            {
                size_t colUV = 0, colY = 0, colRgb = 0;
                for (; colY < widthDA; colY += DA, colUV += A, colRgb += A6)
                {
                    svuint8_t _u = svld1_u8(body, u + colUV);
                    svuint8_t _v = svld1_u8(body, v + colUV);
                    Yuv422pToRgbV2<T>(y + colY + 0 * A, svzip1_u8(_u, _u), svzip1_u8(_v, _v), rgb + colRgb + 0 * A3);
                    Yuv422pToRgbV2<T>(y + colY + 1 * A, svzip2_u8(_u, _u), svzip2_u8(_v, _v), rgb + colRgb + 1 * A3);
                }
                for (; colY < width; colY += 2, colUV += 1, colRgb += 6)
                    Yuv422pToRgb<T>(y + colY, u[colUV], v[colUV], rgb + colRgb);
                y += yStride;
                u += uStride;
                v += vStride;
                rgb += rgbStride;
            }
        }

        void Yuv422pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv422pToRgbV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvBt709: Yuv422pToRgbV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvBt2020: Yuv422pToRgbV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvTrect871: Yuv422pToRgbV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            default:
                assert(0);
            }
        }
    }
#endif
}
