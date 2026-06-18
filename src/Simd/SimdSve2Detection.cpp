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

#include "Simd/SimdDetection.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        using namespace Simd::Detection;

        SIMD_INLINE svuint32_t Sum32ip(uint32_t* const ptr[4], size_t offset, const svbool_t& mask)
        {
            svuint32_t s0 = svld1_u32(mask, ptr[0] + offset);
            svuint32_t s1 = svld1_u32(mask, ptr[1] + offset);
            svuint32_t s2 = svld1_u32(mask, ptr[2] + offset);
            svuint32_t s3 = svld1_u32(mask, ptr[3] + offset);
            return svsub_u32_x(mask, svsub_u32_x(mask, s0, s1), svsub_u32_x(mask, s2, s3));
        }

        SIMD_INLINE svuint32_t Sum32ii(uint32_t* const ptr[4], size_t offset, const svbool_t& mask)
        {
            size_t F = svcntw();
            svuint32_t s0 = svuzp1_u32(svld1_u32(mask, ptr[0] + offset), svld1_u32(mask, ptr[0] + offset + F));
            svuint32_t s1 = svuzp1_u32(svld1_u32(mask, ptr[1] + offset), svld1_u32(mask, ptr[1] + offset + F));
            svuint32_t s2 = svuzp1_u32(svld1_u32(mask, ptr[2] + offset), svld1_u32(mask, ptr[2] + offset + F));
            svuint32_t s3 = svuzp1_u32(svld1_u32(mask, ptr[3] + offset), svld1_u32(mask, ptr[3] + offset + F));
            return svsub_u32_x(mask, svsub_u32_x(mask, s0, s1), svsub_u32_x(mask, s2, s3));
        }

        SIMD_INLINE svfloat32_t ValidSqrt(const svfloat32_t& value, const svbool_t& mask)
        {
            svbool_t positive = svcmpgt_n_f32(mask, value, 0.0f);
            return svsqrt_f32_x(mask, svsel_f32(positive, value, svdup_n_f32(1.0f)));
        }

        SIMD_INLINE svfloat32_t Norm32fp(const HidHaarCascade& hid, size_t offset, const svbool_t& mask)
        {
            svfloat32_t area = svdup_n_f32(hid.windowArea);
            svfloat32_t sum = svcvt_f32_u32_x(mask, Sum32ip(hid.p, offset, mask));
            svfloat32_t sqsum = svcvt_f32_u32_x(mask, Sum32ip(hid.pq, offset, mask));
            return ValidSqrt(svsub_f32_x(mask, svmul_f32_x(mask, sqsum, area), svmul_f32_x(mask, sum, sum)), mask);
        }

        SIMD_INLINE svfloat32_t Norm32fi(const HidHaarCascade& hid, size_t offset, const svbool_t& mask)
        {
            svfloat32_t area = svdup_n_f32(hid.windowArea);
            svfloat32_t sum = svcvt_f32_u32_x(mask, Sum32ii(hid.p, offset, mask));
            svfloat32_t sqsum = svcvt_f32_u32_x(mask, Sum32ii(hid.pq, offset, mask));
            return ValidSqrt(svsub_f32_x(mask, svmul_f32_x(mask, sqsum, area), svmul_f32_x(mask, sum, sum)), mask);
        }

        SIMD_INLINE svfloat32_t WeightedSum32f(const WeightedRect& rect, size_t offset, const svbool_t& mask)
        {
            svuint32_t s0 = svld1_u32(mask, rect.p0 + offset);
            svuint32_t s1 = svld1_u32(mask, rect.p1 + offset);
            svuint32_t s2 = svld1_u32(mask, rect.p2 + offset);
            svuint32_t s3 = svld1_u32(mask, rect.p3 + offset);
            svuint32_t sum = svsub_u32_x(mask, svsub_u32_x(mask, s0, s1), svsub_u32_x(mask, s2, s3));
            return svmul_f32_x(mask, svcvt_f32_u32_x(mask, sum), svdup_n_f32(rect.weight));
        }

        SIMD_INLINE void StageSum32f(const float* leaves, float threshold, const svfloat32_t& sum, const svfloat32_t& norm,
            svfloat32_t& stageSum, const svbool_t& mask)
        {
            svbool_t select = svcmpge_f32(mask, sum, svmul_n_f32_x(mask, norm, threshold));
            stageSum = svadd_f32_x(mask, stageSum, svsel_f32(select, svdup_n_f32(leaves[1]), svdup_n_f32(leaves[0])));
        }

        svbool_t Detect32f(const HidHaarCascade& hid, size_t offset, const svfloat32_t& norm, svbool_t result, const svbool_t& mask)
        {
            typedef HidHaarCascade Hid;
            const float* leaves = hid.leaves.data();
            const Hid::Node* node = hid.nodes.data();
            const Hid::Stage* stages = hid.stages.data();
            for (int i = 0, n = (int)hid.stages.size(); i < n; ++i)
            {
                const Hid::Stage& stage = stages[i];
                if (stage.canSkip)
                    continue;
                const Hid::Node* end = node + stage.ntrees;
                svfloat32_t stageSum = svdup_n_f32(0.0f);
                if (stage.hasThree)
                {
                    for (; node < end; ++node, leaves += 2)
                    {
                        const Hid::Feature& feature = hid.features[node->featureIdx];
                        svfloat32_t sum = svadd_f32_x(result, WeightedSum32f(feature.rect[0], offset, result), WeightedSum32f(feature.rect[1], offset, result));
                        if (feature.rect[2].p0)
                            sum = svadd_f32_x(result, sum, WeightedSum32f(feature.rect[2], offset, result));
                        StageSum32f(leaves, node->threshold, sum, norm, stageSum, result);
                    }
                }
                else
                {
                    for (; node < end; ++node, leaves += 2)
                    {
                        const Hid::Feature& feature = hid.features[node->featureIdx];
                        svfloat32_t sum = svadd_f32_x(result, WeightedSum32f(feature.rect[0], offset, result), WeightedSum32f(feature.rect[1], offset, result));
                        StageSum32f(leaves, node->threshold, sum, norm, stageSum, result);
                    }
                }
                result = svcmpge_n_f32(result, stageSum, stage.threshold);
                size_t resultCount = svcntp_b32(mask, result);
                if (resultCount == 0)
                {
                    return result;
                }
                else if (resultCount == 1)
                {
                    const size_t F = svcntw();
                    Buffer<float> buffer(F);
                    svst1_f32(mask, buffer.m, norm);
                    for (size_t j = 0; j < F; ++j)
                    {
                        svbool_t lane = svcmpeq_n_u32(mask, svindex_u32(0, 1), (uint32_t)j);
                        if (svcntp_b32(lane, result))
                            return Base::Detect32f(hid, offset + j, i + 1, buffer.m[j]) > 0 ? lane : svpfalse_b();
                    }
                }
            }
            return result;
        }

        void DetectionHaarDetect32fp(const HidHaarCascade& hid, const Image& mask, const Rect& rect, Image& dst)
        {
            size_t width = rect.Width();
            const svuint32_t zero = svdup_n_u32(0);
            const svuint32_t one = svdup_n_u32(1);
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += 1)
            {
                size_t col = 0;
                size_t p_offset = row * hid.sum.stride / sizeof(uint32_t) + rect.left;
                size_t pq_offset = row * hid.sqsum.stride / sizeof(uint32_t) + rect.left;
                const uint8_t* maskRow = mask.data + row * mask.stride + rect.left;
                uint8_t* dstRow = dst.data + row * dst.stride + rect.left;
                for (; col < width; col += svcntw())
                {
                    svbool_t pg = svwhilelt_b32(col, width);
                    svbool_t result = svcmpne_n_u32(pg, svld1ub_u32(pg, maskRow + col), 0);
                    if (svcntp_b32(pg, result))
                    {
                        svfloat32_t norm = Norm32fp(hid, pq_offset + col, pg);
                        result = Detect32f(hid, p_offset + col, norm, result, pg);
                        svst1b_u32(pg, dstRow + col, svsel_u32(result, one, zero));
                    }
                    else
                        svst1b_u32(pg, dstRow + col, zero);
                }
            }
        }

        void DetectionHaarDetect32fp(const void* _hid, const uint8_t* mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t* dst, size_t dstStride)
        {
            const HidHaarCascade& hid = *(HidHaarCascade*)_hid;
            return DetectionHaarDetect32fp(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }

        SIMD_INLINE svuint32_t LoadMask32fi(const uint8_t* src, const svbool_t& mask, const svuint8_t& index)
        {
            svuint8_t even = svtbl_u8(svld1_u8(mask, src), index);
            return svunpklo_u32(svunpklo_u16(even));
        }

        SIMD_INLINE svuint8_t PackU32ToU8(const svuint32_t& value)
        {
            svuint16_t lo = svuzp1_u16(svreinterpret_u16_u32(value), svreinterpret_u16_u32(value));
            return svuzp1_u8(svreinterpret_u8_u16(lo), svreinterpret_u8_u16(lo));
        }

        void DetectionHaarDetect32fi(const HidHaarCascade& hid, const Image& mask, const Rect& rect, Image& dst)
        {
            const size_t step = 2;
            const size_t F = svcntw();
            size_t width = rect.Width();
            size_t evenWidth = Simd::AlignLo(width, 2);
            svbool_t mask32 = svptrue_b32();
            svbool_t mask8 = svwhilelt_b8(size_t(0), 2 * F);
            svuint8_t even = svindex_u8(0, 2);
            svuint32_t zero32 = svdup_n_u32(0);
            svuint32_t one32 = svdup_n_u32(1);
            svuint8_t zero8 = svdup_n_u8(0);
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += step)
            {
                size_t col = 0;
                size_t p_offset = row * hid.isum.stride / sizeof(uint32_t) + rect.left / 2;
                size_t pq_offset = row * hid.sqsum.stride / sizeof(uint32_t) + rect.left;
                const uint8_t* maskRow = mask.data + row * mask.stride + rect.left;
                uint8_t* dstRow = dst.data + row * dst.stride + rect.left;
                for (; col + 2 * F <= evenWidth; col += 2 * F)
                {
                    svbool_t result = svcmpne_n_u32(mask32, LoadMask32fi(maskRow + col, mask8, even), 0);
                    if (svcntp_b32(mask32, result))
                    {
                        svfloat32_t norm = Norm32fi(hid, pq_offset + col, mask32);
                        result = Detect32f(hid, p_offset + col / 2, norm, result, mask32);
                        svuint8_t value = PackU32ToU8(svsel_u32(result, one32, zero32));
                        svst1_u8(mask8, dstRow + col, svzip1_u8(value, zero8));
                    }
                    else
                        svst1_u8(mask8, dstRow + col, zero8);
                }
                for (; col < width; col += step)
                {
                    if (maskRow[col] == 0)
                        continue;
                    float norm = Base::Norm32f(hid, pq_offset + col);
                    if (Base::Detect32f(hid, p_offset + col / 2, 0, norm) > 0)
                        dstRow[col] = 1;
                }
            }
        }

        void DetectionHaarDetect32fi(const void* _hid, const uint8_t* mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t* dst, size_t dstStride)
        {
            const HidHaarCascade& hid = *(HidHaarCascade*)_hid;
            return DetectionHaarDetect32fi(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }
    }
#endif
}
