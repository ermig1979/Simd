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

        template <class T> SIMD_INLINE void Yuv422pToBgraV2(const uint8_t* y, const svuint8_t& u, const svuint8_t& v, const svuint8_t& a, uint8_t* bgra)
        {
            const svbool_t body = svptrue_b8();
            const size_t A = svcntb(), A4 = A * 4;
            svuint8_t _y = svld1_u8(body, y);
            svuint8_t blue = YuvToBlue<T>(_y, u);
            svuint8_t green = YuvToGreen<T>(_y, u, v);
            svuint8_t red = YuvToRed<T>(_y, v);
            svst4_u8(body, bgra + 0 * A4, svcreate4_u8(blue, green, red, a));
        }

        template <class T> SIMD_INLINE void Yuva422pToBgraV2(const uint8_t* y, const svuint8_t& u, const svuint8_t& v, const uint8_t* a, uint8_t* bgra)
        {
            const svbool_t body = svptrue_b8();
            const size_t A = svcntb(), A4 = A * 4;
            svuint8_t _y = svld1_u8(body, y);
            svuint8_t _a = svld1_u8(body, a);
            svuint8_t blue = YuvToBlue<T>(_y, u);
            svuint8_t green = YuvToGreen<T>(_y, u, v);
            svuint8_t red = YuvToRed<T>(_y, v);
            svst4_u8(body, bgra + 0 * A4, svcreate4_u8(blue, green, red, _a));
        }

        template <class T> SIMD_INLINE void Yuv422pToBgra(const uint8_t* y, int u, int v, uint8_t alpha, uint8_t* bgra)
        {
            Base::YuvToBgra<T>(y[0], u, v, alpha, bgra + 0);
            Base::YuvToBgra<T>(y[1], u, v, alpha, bgra + 4);
        }

        template <class T> SIMD_INLINE void Yuva422pToBgra(const uint8_t* y, int u, int v, const uint8_t* a, uint8_t* bgra)
        {
            Base::YuvToBgra<T>(y[0], u, v, a[0], bgra + 0);
            Base::YuvToBgra<T>(y[1], u, v, a[1], bgra + 4);
        }

        template <class T> void Yuva420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            const size_t A = svcntb(), A4 = 4 * A, DA = 2 * A, A8 = 8 * A;
            const size_t widthDA = AlignLo(width, DA);
            const svbool_t body = svptrue_b8();
            for (size_t row = 0; row < height; row += 2)
            {
                size_t colUV = 0, colY = 0, colBgra = 0;
                for (; colY < widthDA; colY += DA, colUV += A, colBgra += A8)
                {
                    svuint8_t _u = svld1_u8(body, u + colUV);
                    svuint8_t _v = svld1_u8(body, v + colUV);
                    svuint8_t u0 = svzip1_u8(_u, _u);
                    svuint8_t u1 = svzip2_u8(_u, _u);
                    svuint8_t v0 = svzip1_u8(_v, _v);
                    svuint8_t v1 = svzip2_u8(_v, _v);
                    Yuva422pToBgraV2<T>(y + colY + 0 * A, u0, v0, a + colY + 0 * A, bgra + colBgra + 0 * A4);
                    Yuva422pToBgraV2<T>(y + colY + 1 * A, u1, v1, a + colY + 1 * A, bgra + colBgra + 1 * A4);
                    Yuva422pToBgraV2<T>(y + yStride + colY + 0 * A, u0, v0, a + aStride + colY + 0 * A, bgra + bgraStride + colBgra + 0 * A4);
                    Yuva422pToBgraV2<T>(y + yStride + colY + 1 * A, u1, v1, a + aStride + colY + 1 * A, bgra + bgraStride + colBgra + 1 * A4);
                }
                for (; colY < width; colY += 2, colUV += 1, colBgra += 8)
                {
                    int _u = u[colUV], _v = v[colUV];
                    Yuva422pToBgra<T>(y + colY, _u, _v, a + colY, bgra + colBgra);
                    Yuva422pToBgra<T>(y + yStride + colY, _u, _v, a + aStride + colY, bgra + bgraStride + colBgra);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                a += 2 * aStride;
                bgra += 2 * bgraStride;
            }
        }

        void Yuva420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuva420pToBgraV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvBt709: Yuva420pToBgraV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvBt2020: Yuva420pToBgraV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvTrect871: Yuva420pToBgraV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            default:
                assert(0);
            }
        }

        template <class T> void Yuva422pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride)
        {
            assert((width % 2 == 0) && (width >= 2));

            const size_t A = svcntb(), A4 = 4 * A, DA = 2 * A, A8 = 8 * A;
            const size_t widthDA = AlignLo(width, DA);
            const svbool_t body = svptrue_b8();
            for (size_t row = 0; row < height; ++row)
            {
                size_t colUV = 0, colY = 0, colBgra = 0;
                for (; colY < widthDA; colY += DA, colUV += A, colBgra += A8)
                {
                    svuint8_t _u = svld1_u8(body, u + colUV);
                    svuint8_t _v = svld1_u8(body, v + colUV);
                    Yuva422pToBgraV2<T>(y + colY + 0 * A, svzip1_u8(_u, _u), svzip1_u8(_v, _v), a + colY + 0 * A, bgra + colBgra + 0 * A4);
                    Yuva422pToBgraV2<T>(y + colY + 1 * A, svzip2_u8(_u, _u), svzip2_u8(_v, _v), a + colY + 1 * A, bgra + colBgra + 1 * A4);
                }
                for (; colY < width; colY += 2, colUV += 1, colBgra += 8)
                    Yuva422pToBgra<T>(y + colY, u[colUV], v[colUV], a + colY, bgra + colBgra);
                y += yStride;
                u += uStride;
                v += vStride;
                a += aStride;
                bgra += bgraStride;
            }
        }

        void Yuva422pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuva422pToBgraV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvBt709: Yuva422pToBgraV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvBt2020: Yuva422pToBgraV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvTrect871: Yuva422pToBgraV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            default:
                assert(0);
            }
        }

        template <class T> void Yuv422pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            assert((width % 2 == 0) && (width >= 2));

            const size_t A = svcntb(), A4 = 4 * A, DA = 2 * A, A8 = 8 * A;
            const size_t widthDA = AlignLo(width, DA);
            const svbool_t body = svptrue_b8();
            const svuint8_t _alpha = svdup_n_u8(alpha);
            for (size_t row = 0; row < height; ++row)
            {
                size_t colUV = 0, colY = 0, colBgra = 0;
                for (; colY < widthDA; colY += DA, colUV += A, colBgra += A8)
                {
                    svuint8_t _u = svld1_u8(body, u + colUV);
                    svuint8_t _v = svld1_u8(body, v + colUV);
                    Yuv422pToBgraV2<T>(y + colY + 0 * A, svzip1_u8(_u, _u), svzip1_u8(_v, _v), _alpha, bgra + colBgra + 0 * A4);
                    Yuv422pToBgraV2<T>(y + colY + 1 * A, svzip2_u8(_u, _u), svzip2_u8(_v, _v), _alpha, bgra + colBgra + 1 * A4);
                }
                for (; colY < width; colY += 2, colUV += 1, colBgra += 8)
                    Yuv422pToBgra<T>(y + colY, u[colUV], v[colUV], alpha, bgra + colBgra);
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void Yuv422pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv422pToBgraV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt709: Yuv422pToBgraV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt2020: Yuv422pToBgraV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvTrect871: Yuv422pToBgraV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            default:
                assert(0);
            }
        }

        template <class T> void Yuv420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            const size_t A = svcntb(), A4 = 4 * A, DA = 2 * A, A8 = 8 * A;
            const size_t widthDA = AlignLo(width, DA);
            const svbool_t body = svptrue_b8();
            const svuint8_t _alpha = svdup_n_u8(alpha);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t colUV = 0, colY = 0, colBgra = 0;
                for (; colY < widthDA; colY += DA, colUV += A, colBgra += A8)
                {
                    svuint8_t _u = svld1_u8(body, u + colUV);
                    svuint8_t _v = svld1_u8(body, v + colUV);
                    svuint8_t u0 = svzip1_u8(_u, _u);
                    svuint8_t u1 = svzip2_u8(_u, _u);
                    svuint8_t v0 = svzip1_u8(_v, _v);
                    svuint8_t v1 = svzip2_u8(_v, _v);
                    Yuv422pToBgraV2<T>(y + colY + 0 * A, u0, v0, _alpha, bgra + colBgra + 0 * A4);
                    Yuv422pToBgraV2<T>(y + colY + 1 * A, u1, v1, _alpha, bgra + colBgra + 1 * A4);
                    Yuv422pToBgraV2<T>(y + yStride + colY + 0 * A, u0, v0, _alpha, bgra + bgraStride + colBgra + 0 * A4);
                    Yuv422pToBgraV2<T>(y + yStride + colY + 1 * A, u1, v1, _alpha, bgra + bgraStride + colBgra + 1 * A4);
                }
                for (; colY < width; colY += 2, colUV += 1, colBgra += 8)
                {
                    int _u = u[colUV], _v = v[colUV];
                    Yuv422pToBgra<T>(y + colY, _u, _v, alpha, bgra + colBgra);
                    Yuv422pToBgra<T>(y + yStride + colY, _u, _v, alpha, bgra + bgraStride + colBgra);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgra += 2 * bgraStride;
            }
        }

        void Yuv420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv420pToBgraV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt709: Yuv420pToBgraV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt2020: Yuv420pToBgraV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvTrect871: Yuv420pToBgraV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            default:
                assert(0);
            }
        }

        template <class T> SIMD_INLINE void Yuv444pToBgraV2(const uint8_t* y, const uint8_t* u, const uint8_t* v, const svuint8_t& a, uint8_t* bgra)
        {
            const svbool_t body = svptrue_b8();
            const size_t A = svcntb(), A4 = A * 4;
            svuint8_t _y = svld1_u8(body, y);
            svuint8_t _u = svld1_u8(body, u);
            svuint8_t _v = svld1_u8(body, v);
            svuint8_t blue = YuvToBlue<T>(_y, _u);
            svuint8_t green = YuvToGreen<T>(_y, _u, _v);
            svuint8_t red = YuvToRed<T>(_y, _v);
            svst4_u8(body, bgra + 0 * A4, svcreate4_u8(blue, green, red, a));
        }

        template <class T> void Yuv444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            const size_t A = svcntb();
            const size_t widthA = AlignLo(width, A);
            const svuint8_t _alpha = svdup_n_u8(alpha);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, colBgra = 0;
                for (; col < widthA; col += A, colBgra += 4 * A)
                    Yuv444pToBgraV2<T>(y + col, u + col, v + col, _alpha, bgra + colBgra);
                for (; col < width; ++col, colBgra += 4)
                    Base::YuvToBgra<T>(y[col], u[col], v[col], alpha, bgra + colBgra);
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void Yuv444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv444pToBgraV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt709: Yuv444pToBgraV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt2020: Yuv444pToBgraV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvTrect871: Yuv444pToBgraV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            default:
                assert(0);
            }
        }

        template <class T> SIMD_INLINE void Yuva444pToBgraV2(const uint8_t* y, const uint8_t* u, const uint8_t* v, const uint8_t* a, uint8_t* bgra)
        {
            const svbool_t body = svptrue_b8();
            const size_t A = svcntb(), A4 = A * 4;
            svuint8_t _y = svld1_u8(body, y);
            svuint8_t _u = svld1_u8(body, u);
            svuint8_t _v = svld1_u8(body, v);
            svuint8_t _a = svld1_u8(body, a);
            svuint8_t blue = YuvToBlue<T>(_y, _u);
            svuint8_t green = YuvToGreen<T>(_y, _u, _v);
            svuint8_t red = YuvToRed<T>(_y, _v);
            svst4_u8(body, bgra + 0 * A4, svcreate4_u8(blue, green, red, _a));
        }

        template <class T> void Yuva444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride)
        {
            const size_t A = svcntb();
            const size_t widthA = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, colBgra = 0;
                for (; col < widthA; col += A, colBgra += 4 * A)
                    Yuva444pToBgraV2<T>(y + col, u + col, v + col, a + col, bgra + colBgra);
                for (; col < width; ++col, colBgra += 4)
                    Base::YuvToBgra<T>(y[col], u[col], v[col], a[col], bgra + colBgra);
                y += yStride;
                u += uStride;
                v += vStride;
                a += aStride;
                bgra += bgraStride;
            }
        }

        void Yuva444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuva444pToBgraV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvBt709: Yuva444pToBgraV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvBt2020: Yuva444pToBgraV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvTrect871: Yuva444pToBgraV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE void Yuv444pToRgbaV2(const uint8_t* y, const uint8_t* u, const uint8_t* v, const svuint8_t& a, uint8_t* rgba)
        {
            const svbool_t body = svptrue_b8();
            const size_t A = svcntb(), A4 = A * 4;
            svuint8_t _y = svld1_u8(body, y);
            svuint8_t _u = svld1_u8(body, u);
            svuint8_t _v = svld1_u8(body, v);
            svuint8_t red = YuvToRed<T>(_y, _v);
            svuint8_t green = YuvToGreen<T>(_y, _u, _v);
            svuint8_t blue = YuvToBlue<T>(_y, _u);
            svst4_u8(body, rgba + 0 * A4, svcreate4_u8(red, green, blue, a));
        }

        template <class T> void Yuv444pToRgbaV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgba, size_t rgbaStride, uint8_t alpha)
        {
            const size_t A = svcntb();
            const size_t widthA = AlignLo(width, A);
            const svuint8_t _alpha = svdup_n_u8(alpha);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, colRgba = 0;
                for (; col < widthA; col += A, colRgba += 4 * A)
                    Yuv444pToRgbaV2<T>(y + col, u + col, v + col, _alpha, rgba + colRgba);
                for (; col < width; ++col, colRgba += 4)
                    Base::YuvToRgba<T>(y[col], u[col], v[col], alpha, rgba + colRgba);
                y += yStride;
                u += uStride;
                v += vStride;
                rgba += rgbaStride;
            }
        }

        void Yuv444pToRgbaV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgba, size_t rgbaStride, uint8_t alpha, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv444pToRgbaV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha); break;
            case SimdYuvBt709: Yuv444pToRgbaV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha); break;
            case SimdYuvBt2020: Yuv444pToRgbaV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha); break;
            case SimdYuvTrect871: Yuv444pToRgbaV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha); break;
            default:
                assert(0);
            }
        }
    }
#endif
}
