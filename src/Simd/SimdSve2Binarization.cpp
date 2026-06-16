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
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE    
    namespace Sve2
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t width, size_t edge)
                {
                    size_t size = sizeof(uint16_t) * (width + 2 * edge) + sizeof(uint32_t) * (2 * width + 2 * edge);
                    _p = Allocate(size);
                    memset(_p, 0, size);
                    sa = (uint16_t*)_p + edge;
                    s0a0 = (uint32_t*)(sa + width + edge) + edge;
                    sum = (uint32_t*)(s0a0 + width + edge);
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t* sa;
                uint32_t* s0a0;
                uint32_t* sum;
            private:
                void* _p;
            };

            struct BufferV2
            {
                BufferV2(size_t width, size_t edge, size_t A)
                {
                    _width = AlignHi(width, A);
                    _edge = AlignHi(edge, A);
                    _size = _width + 2 * _edge;
                    size_t size = sizeof(int32_t) * (2 * _size + 2 * _width);
                    _p = Allocate(size);
                    memset(_p, 0, size);
                    int32_t* data = (int32_t*)_p;
                    rs = data + _edge;
                    ra = data + _size + _edge;
                    sum = data + 2 * _size;
                    area = sum + _width;
                }

                ~BufferV2()
                {
                    Free(_p);
                }

                int32_t* rs;
                int32_t* ra;
                int32_t* sum;
                int32_t* area;
            private:
                size_t _width;
                size_t _edge;
                size_t _size;
                void* _p;
            };
        }

        SIMD_INLINE void SetMasks(size_t col, size_t width, svbool_t& mask0, svbool_t& mask1, svbool_t& mask2, svbool_t& mask3)
        {
            const size_t F = svcntw();
            mask0 = svwhilelt_b32(col + 0 * F, width);
            mask1 = svwhilelt_b32(col + 1 * F, width);
            mask2 = svwhilelt_b32(col + 2 * F, width);
            mask3 = svwhilelt_b32(col + 3 * F, width);
        }

        SIMD_INLINE svuint32_t UnpackSa(const svbool_t& mask, const svuint32_t& sa)
        {
            return svorr_u32_x(mask, svand_n_u32_x(mask, sa, 0x00FF), svlsl_n_u32_x(mask, svand_n_u32_x(mask, sa, 0xFF00), 8));
        }

        SIMD_INLINE void UnpackSa(const uint16_t* sa, uint32_t* s0a0, size_t col, size_t width)
        {
            const size_t F = svcntw();
            svbool_t mask0 = svwhilelt_b32(col + 0 * F, width);
            svbool_t mask1 = svwhilelt_b32(col + 1 * F, width);
            svuint16_t _sa = svld1_u16(svwhilelt_b16(col, width), sa + col);
            svst1_u32(mask0, s0a0 + col + 0 * F, UnpackSa(mask0, svunpklo_u32(_sa)));
            svst1_u32(mask1, s0a0 + col + 1 * F, UnpackSa(mask1, svunpkhi_u32(_sa)));
        }

        template <SimdCompareType compareType> SIMD_INLINE void AddRow(const uint8_t* src, uint16_t* sa, const svuint8_t& value, const svuint8_t& one, const svbool_t& mask)
        {
            svuint8x2_t _sa = svld2_u8(mask, (uint8_t*)sa);
            svuint8_t sums = svget2(_sa, 0);
            svuint8_t areas = svget2(_sa, 1);
            sums = svadd_u8_m(Compare8u<compareType>(mask, svld1_u8(mask, src), value), sums, one);
            areas = svadd_u8_m(mask, areas, one);
            svst2_u8(mask, (uint8_t*)sa, svcreate2_u8(sums, areas));
        }

        template <SimdCompareType compareType> SIMD_INLINE void SubRow(const uint8_t* src, uint16_t* sa, const svuint8_t& value, const svuint8_t& one, const svbool_t& mask)
        {
            svuint8x2_t _sa = svld2_u8(mask, (uint8_t*)sa);
            svuint8_t sums = svget2(_sa, 0);
            svuint8_t areas = svget2(_sa, 1);
            sums = svsub_u8_m(Compare8u<compareType>(mask, svld1_u8(mask, src), value), sums, one);
            areas = svsub_u8_m(mask, areas, one);
            svst2_u8(mask, (uint8_t*)sa, svcreate2_u8(sums, areas));
        }

        SIMD_INLINE void CompareSum(const uint32_t* sum, const svuint32_t& threshold,
            const svuint32_t& positive, const svuint32_t& negative, uint8_t* dst,
            const svbool_t& mask0, const svbool_t& mask1, const svbool_t& mask2, const svbool_t& mask3)
        {
            const size_t F = svcntw();
            svuint32_t sum0 = svld1_u32(mask0, sum + 0 * F);
            svuint32_t sum1 = svld1_u32(mask1, sum + 1 * F);
            svuint32_t sum2 = svld1_u32(mask2, sum + 2 * F);
            svuint32_t sum3 = svld1_u32(mask3, sum + 3 * F);
            const svbool_t compare0 = svcmpgt_u32(mask0, svmul_n_u32_x(mask0, svand_n_u32_x(mask0, sum0, 0xFFFF), 0xFF), svmul_u32_x(mask0, svlsr_n_u32_x(mask0, sum0, 16), threshold));
            const svbool_t compare1 = svcmpgt_u32(mask1, svmul_n_u32_x(mask1, svand_n_u32_x(mask1, sum1, 0xFFFF), 0xFF), svmul_u32_x(mask1, svlsr_n_u32_x(mask1, sum1, 16), threshold));
            const svbool_t compare2 = svcmpgt_u32(mask2, svmul_n_u32_x(mask2, svand_n_u32_x(mask2, sum2, 0xFFFF), 0xFF), svmul_u32_x(mask2, svlsr_n_u32_x(mask2, sum2, 16), threshold));
            const svbool_t compare3 = svcmpgt_u32(mask3, svmul_n_u32_x(mask3, svand_n_u32_x(mask3, sum3, 0xFFFF), 0xFF), svmul_u32_x(mask3, svlsr_n_u32_x(mask3, sum3, 16), threshold));
            svst1b_u32(mask0, dst + 0 * F, svsel_u32(compare0, positive, negative));
            svst1b_u32(mask1, dst + 1 * F, svsel_u32(compare1, positive, negative));
            svst1b_u32(mask2, dst + 2 * F, svsel_u32(compare2, positive, negative));
            svst1b_u32(mask3, dst + 3 * F, svsel_u32(compare3, positive, negative));
        }

        template <SimdCompareType compareType> void AveragingBinarization(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative, uint8_t* dst, size_t dstStride)
        {
            assert(width > neighborhood && height > neighborhood && neighborhood < 0x80);

            const size_t A = svcntb();
            const size_t HA = svcnth();
            const size_t alignedWidth = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svuint8_t _value = svdup_n_u8(value);
            const svuint8_t _one = svdup_n_u8(1);
            const svuint32_t _positive = svdup_n_u32(positive);
            const svuint32_t _negative = svdup_n_u32(negative);
            const svuint32_t _threshold = svdup_n_u32(threshold);

            Buffer buffer(AlignHi(width, A), AlignHi(neighborhood + 1, A));

            for (size_t row = 0; row < neighborhood; ++row)
            {
                const uint8_t* ps = src + row * srcStride;
                for (size_t col = 0; col < alignedWidth; col += A)
                    AddRow<compareType>(ps + col, buffer.sa + col, _value, _one, body);
                if (alignedWidth != width)
                    AddRow<compareType>(ps + alignedWidth, buffer.sa + alignedWidth, _value, _one, svwhilelt_b8(alignedWidth, width));
            }

            for (size_t row = 0; row < height; ++row)
            {
                if (row < height - neighborhood)
                {
                    const uint8_t* ps = src + (row + neighborhood) * srcStride;
                    for (size_t col = 0; col < alignedWidth; col += A)
                        AddRow<compareType>(ps + col, buffer.sa + col, _value, _one, body);
                    if (alignedWidth != width)
                        AddRow<compareType>(ps + alignedWidth, buffer.sa + alignedWidth, _value, _one, svwhilelt_b8(alignedWidth, width));
                }

                if (row > neighborhood)
                {
                    const uint8_t* ps = src + (row - neighborhood - 1) * srcStride;
                    for (size_t col = 0; col < alignedWidth; col += A)
                        SubRow<compareType>(ps + col, buffer.sa + col, _value, _one, body);
                    if (alignedWidth != width)
                        SubRow<compareType>(ps + alignedWidth, buffer.sa + alignedWidth, _value, _one, svwhilelt_b8(alignedWidth, width));
                }

                for (size_t col = 0; col < width; col += HA)
                    UnpackSa(buffer.sa, buffer.s0a0, col, width);

                uint32_t sum = 0;
                for (size_t col = 0; col < neighborhood; ++col)
                    sum += buffer.s0a0[col];
                for (size_t col = 0; col < width; ++col)
                {
                    sum += buffer.s0a0[col + neighborhood];
                    sum -= buffer.s0a0[col - neighborhood - 1];
                    buffer.sum[col] = sum;
                }

                for (size_t col = 0; col < alignedWidth; col += A)
                    CompareSum(buffer.sum + col, _threshold, _positive, _negative, dst + col, svptrue_b32(), svptrue_b32(), svptrue_b32(), svptrue_b32());
                if (alignedWidth != width)
                {
                    svbool_t tail0, tail1, tail2, tail3;
                    SetMasks(alignedWidth, width, tail0, tail1, tail2, tail3);
                    CompareSum(buffer.sum + alignedWidth, _threshold, _positive, _negative, dst + alignedWidth, tail0, tail1, tail2, tail3);
                }
                dst += dstStride;
            }
        }

        void AveragingBinarization(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative,
            uint8_t* dst, size_t dstStride, SimdCompareType compareType)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return AveragingBinarization<SimdCompareEqual>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case SimdCompareNotEqual:
                return AveragingBinarization<SimdCompareNotEqual>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case SimdCompareGreater:
                return AveragingBinarization<SimdCompareGreater>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case SimdCompareGreaterOrEqual:
                return AveragingBinarization<SimdCompareGreaterOrEqual>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case SimdCompareLesser:
                return AveragingBinarization<SimdCompareLesser>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case SimdCompareLesserOrEqual:
                return AveragingBinarization<SimdCompareLesserOrEqual>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            default:
                assert(0);
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void AddRow(const uint8_t* src, int32_t* rs, int32_t* ra, const svbool_t& mask0, const svbool_t& mask1, const svbool_t& mask2, const svbool_t& mask3)
        {
            const size_t F = svcntw();
            svst1_s32(mask0, rs + 0 * F, svadd_s32_x(mask0, svld1_s32(mask0, rs + 0 * F), svld1ub_s32(mask0, src + 0 * F)));
            svst1_s32(mask1, rs + 1 * F, svadd_s32_x(mask1, svld1_s32(mask1, rs + 1 * F), svld1ub_s32(mask1, src + 1 * F)));
            svst1_s32(mask2, rs + 2 * F, svadd_s32_x(mask2, svld1_s32(mask2, rs + 2 * F), svld1ub_s32(mask2, src + 2 * F)));
            svst1_s32(mask3, rs + 3 * F, svadd_s32_x(mask3, svld1_s32(mask3, rs + 3 * F), svld1ub_s32(mask3, src + 3 * F)));
            svst1_s32(mask0, ra + 0 * F, svadd_n_s32_x(mask0, svld1_s32(mask0, ra + 0 * F), 1));
            svst1_s32(mask1, ra + 1 * F, svadd_n_s32_x(mask1, svld1_s32(mask1, ra + 1 * F), 1));
            svst1_s32(mask2, ra + 2 * F, svadd_n_s32_x(mask2, svld1_s32(mask2, ra + 2 * F), 1));
            svst1_s32(mask3, ra + 3 * F, svadd_n_s32_x(mask3, svld1_s32(mask3, ra + 3 * F), 1));
        }

        SIMD_INLINE void SubRow(const uint8_t* src, int32_t* rs, int32_t* ra, const svbool_t& mask0, const svbool_t& mask1, const svbool_t& mask2, const svbool_t& mask3)
        {
            const size_t F = svcntw();
            svst1_s32(mask0, rs + 0 * F, svsub_s32_x(mask0, svld1_s32(mask0, rs + 0 * F), svld1ub_s32(mask0, src + 0 * F)));
            svst1_s32(mask1, rs + 1 * F, svsub_s32_x(mask1, svld1_s32(mask1, rs + 1 * F), svld1ub_s32(mask1, src + 1 * F)));
            svst1_s32(mask2, rs + 2 * F, svsub_s32_x(mask2, svld1_s32(mask2, rs + 2 * F), svld1ub_s32(mask2, src + 2 * F)));
            svst1_s32(mask3, rs + 3 * F, svsub_s32_x(mask3, svld1_s32(mask3, rs + 3 * F), svld1ub_s32(mask3, src + 3 * F)));
            svst1_s32(mask0, ra + 0 * F, svsub_n_s32_x(mask0, svld1_s32(mask0, ra + 0 * F), 1));
            svst1_s32(mask1, ra + 1 * F, svsub_n_s32_x(mask1, svld1_s32(mask1, ra + 1 * F), 1));
            svst1_s32(mask2, ra + 2 * F, svsub_n_s32_x(mask2, svld1_s32(mask2, ra + 2 * F), 1));
            svst1_s32(mask3, ra + 3 * F, svsub_n_s32_x(mask3, svld1_s32(mask3, ra + 3 * F), 1));
        }

        SIMD_INLINE void Compare(const uint8_t* src, const int32_t* sum, const int32_t* area, const svint32_t& shift,
            const svuint32_t& positive, const svuint32_t& negative, uint8_t* dst,
            const svbool_t& mask0, const svbool_t& mask1, const svbool_t& mask2, const svbool_t& mask3)
        {
            const size_t F = svcntw();
            const svint32_t v0 = svadd_s32_x(mask0, svld1ub_s32(mask0, src + 0 * F), shift);
            const svint32_t v1 = svadd_s32_x(mask1, svld1ub_s32(mask1, src + 1 * F), shift);
            const svint32_t v2 = svadd_s32_x(mask2, svld1ub_s32(mask2, src + 2 * F), shift);
            const svint32_t v3 = svadd_s32_x(mask3, svld1ub_s32(mask3, src + 3 * F), shift);
            const svbool_t compare0 = svcmpgt_s32(mask0, svmul_s32_x(mask0, v0, svld1_s32(mask0, area + 0 * F)), svld1_s32(mask0, sum + 0 * F));
            const svbool_t compare1 = svcmpgt_s32(mask1, svmul_s32_x(mask1, v1, svld1_s32(mask1, area + 1 * F)), svld1_s32(mask1, sum + 1 * F));
            const svbool_t compare2 = svcmpgt_s32(mask2, svmul_s32_x(mask2, v2, svld1_s32(mask2, area + 2 * F)), svld1_s32(mask2, sum + 2 * F));
            const svbool_t compare3 = svcmpgt_s32(mask3, svmul_s32_x(mask3, v3, svld1_s32(mask3, area + 3 * F)), svld1_s32(mask3, sum + 3 * F));
            svst1b_u32(mask0, dst + 0 * F, svsel_u32(compare0, positive, negative));
            svst1b_u32(mask1, dst + 1 * F, svsel_u32(compare1, positive, negative));
            svst1b_u32(mask2, dst + 2 * F, svsel_u32(compare2, positive, negative));
            svst1b_u32(mask3, dst + 3 * F, svsel_u32(compare3, positive, negative));
        }

        void AveragingBinarizationV2(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t neighborhood, int32_t shift, uint8_t positive, uint8_t negative, uint8_t* dst, size_t dstStride)
        {
            assert(width > neighborhood && height > neighborhood);

            const size_t A = svcntb();
            const size_t alignedWidth = AlignLo(width, A);
            const svbool_t body0 = svptrue_b32(), body1 = svptrue_b32(), body2 = svptrue_b32(), body3 = svptrue_b32();
            const svuint32_t _positive = svdup_n_u32(positive);
            const svuint32_t _negative = svdup_n_u32(negative);
            const svint32_t _shift = svdup_n_s32(shift);

            BufferV2 buffer(width, neighborhood + 1, A);

            for (size_t row = 0; row < neighborhood; ++row)
            {
                const uint8_t* ps = src + row * srcStride;
                for (size_t col = 0; col < alignedWidth; col += A)
                    AddRow(ps + col, buffer.rs + col, buffer.ra + col, body0, body1, body2, body3);
                if (alignedWidth != width)
                {
                    svbool_t tail0, tail1, tail2, tail3;
                    SetMasks(alignedWidth, width, tail0, tail1, tail2, tail3);
                    AddRow(ps + alignedWidth, buffer.rs + alignedWidth, buffer.ra + alignedWidth, tail0, tail1, tail2, tail3);
                }
            }

            for (size_t row = 0; row < height; ++row)
            {
                if (row < height - neighborhood)
                {
                    const uint8_t* ps = src + (row + neighborhood) * srcStride;
                    for (size_t col = 0; col < alignedWidth; col += A)
                        AddRow(ps + col, buffer.rs + col, buffer.ra + col, body0, body1, body2, body3);
                    if (alignedWidth != width)
                    {
                        svbool_t tail0, tail1, tail2, tail3;
                        SetMasks(alignedWidth, width, tail0, tail1, tail2, tail3);
                        AddRow(ps + alignedWidth, buffer.rs + alignedWidth, buffer.ra + alignedWidth, tail0, tail1, tail2, tail3);
                    }
                }

                if (row > neighborhood)
                {
                    const uint8_t* ps = src + (row - neighborhood - 1) * srcStride;
                    for (size_t col = 0; col < alignedWidth; col += A)
                        SubRow(ps + col, buffer.rs + col, buffer.ra + col, body0, body1, body2, body3);
                    if (alignedWidth != width)
                    {
                        svbool_t tail0, tail1, tail2, tail3;
                        SetMasks(alignedWidth, width, tail0, tail1, tail2, tail3);
                        SubRow(ps + alignedWidth, buffer.rs + alignedWidth, buffer.ra + alignedWidth, tail0, tail1, tail2, tail3);
                    }
                }

                int32_t sum = 0, area = 0;
                for (size_t col = 0; col < neighborhood; ++col)
                {
                    sum += buffer.rs[col];
                    area += buffer.ra[col];
                }
                const uint8_t* ps = src + row * srcStride;
                for (size_t col = 0; col < width; ++col)
                {
                    sum += buffer.rs[col + neighborhood] - buffer.rs[col - neighborhood - 1];
                    area += buffer.ra[col + neighborhood] - buffer.ra[col - neighborhood - 1];
                    buffer.sum[col] = sum;
                    buffer.area[col] = area;
                }

                for (size_t col = 0; col < alignedWidth; col += A)
                    Compare(ps + col, buffer.sum + col, buffer.area + col, _shift, _positive, _negative, dst + col, body0, body1, body2, body3);
                if (alignedWidth != width)
                {
                    svbool_t tail0, tail1, tail2, tail3;
                    SetMasks(alignedWidth, width, tail0, tail1, tail2, tail3);
                    Compare(ps + alignedWidth, buffer.sum + alignedWidth, buffer.area + alignedWidth, _shift, _positive, _negative, dst + alignedWidth, tail0, tail1, tail2, tail3);
                }
                dst += dstStride;
            }
        }
    }
#endif// SIMD_SVE2_ENABLE
}
