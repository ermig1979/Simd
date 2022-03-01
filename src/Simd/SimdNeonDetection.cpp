/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdDetection.h"
#include "Simd/SimdShuffle.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        using namespace Simd::Detection;

        SIMD_INLINE void UnpackMask16i(const uint8_t * src, uint16_t * dst, const uint8x16_t & mask, uint8x16x2_t & buffer)
        {
            buffer.val[0] = vandq_u8(mask, Load<false>(src));
            Store2<false>((uint8_t*)dst, buffer);
        }

        SIMD_INLINE void UnpackMask16i(const uint8_t * src, size_t size, uint16_t * dst, const uint8x16_t & mask)
        {
            uint8x16x2_t buffer;
            buffer.val[1] = K8_00;
            size_t alignedSize = Simd::AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                UnpackMask16i(src + i, dst + i, mask, buffer);
            if (size != alignedSize)
                UnpackMask16i(src + size - A, dst + size - A, mask, buffer);
        }

        SIMD_INLINE void UnpackMask32i(const uint8_t * src, uint32_t * dst, const uint8x16_t & mask, uint8x16x4_t & buffer)
        {
            buffer.val[0] = vandq_u8(mask, Load<false>(src));
            Store4<false>((uint8_t*)dst, buffer);
        }

        SIMD_INLINE void UnpackMask32i(const uint8_t * src, size_t size, uint32_t * dst, const uint8x16_t & mask)
        {
            uint8x16x4_t buffer;
            buffer.val[1] = K8_00;
            buffer.val[2] = K8_00;
            buffer.val[3] = K8_00;
            size_t alignedSize = Simd::AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                UnpackMask32i(src + i, dst + i, mask, buffer);
            if (size != alignedSize)
                UnpackMask32i(src + size - A, dst + size - A, mask, buffer);
        }

        SIMD_INLINE void PackResult16i(const uint16_t * src, uint8_t * dst)
        {
            uint8x16x2_t _src = Load2<false>((const uint8_t *)src);
            Store<false>(dst, _src.val[0]);
        }

        SIMD_INLINE void PackResult16i(const uint16_t * src, size_t size, uint8_t * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                PackResult16i(src + i, dst + i);
            if (size != alignedSize)
                PackResult16i(src + size - A, dst + size - A);
        }

        SIMD_INLINE void PackResult32i(const uint32_t * src, uint8_t * dst)
        {
            uint8x16x4_t _src = Load4<false>((const uint8_t *)src);
            Store<false>(dst, _src.val[0]);
        }

        SIMD_INLINE void PackResult32i(const uint32_t * src, size_t size, uint8_t * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                PackResult32i(src + i, dst + i);
            if (size != alignedSize)
                PackResult32i(src + size - A, dst + size - A);
        }

        SIMD_INLINE int ResultCount(const uint32x4_t & result)
        {
            uint32x4_t a = (uint32x4_t)vpaddlq_u32(result);
            return vgetq_lane_u32(a, 0) + vgetq_lane_u32(a, 2);
        }

        SIMD_INLINE int ResultCount(const uint16x8_t & result)
        {
            uint32x4_t a = (uint32x4_t)vpaddlq_u32(vpaddlq_u16(result));
            return vgetq_lane_u32(a, 0) + vgetq_lane_u32(a, 2);
        }

        SIMD_INLINE float32x4_t ValidSqrt(const float32x4_t & value)
        {
            uint32x4_t mask = vcgtq_f32(value, vdupq_n_f32(0.0f));
            return Sqrt<1>(vbslq_f32(mask, value, vdupq_n_f32(1.0f)));
        }

        SIMD_INLINE uint32x4_t Sum32ip(uint32_t * const ptr[4], size_t offset)
        {
            uint32x4_t s0 = vld1q_u32(ptr[0] + offset);
            uint32x4_t s1 = vld1q_u32(ptr[1] + offset);
            uint32x4_t s2 = vld1q_u32(ptr[2] + offset);
            uint32x4_t s3 = vld1q_u32(ptr[3] + offset);
            return vsubq_u32(vsubq_u32(s0, s1), vsubq_u32(s2, s3));
        }

        SIMD_INLINE uint32x4_t Sum32ii(uint32_t * const ptr[4], size_t offset)
        {
            uint32x4x2_t s0 = vld2q_u32(ptr[0] + offset);
            uint32x4x2_t s1 = vld2q_u32(ptr[1] + offset);
            uint32x4x2_t s2 = vld2q_u32(ptr[2] + offset);
            uint32x4x2_t s3 = vld2q_u32(ptr[3] + offset);
            return vsubq_u32(vsubq_u32(s0.val[0], s1.val[0]), vsubq_u32(s2.val[0], s3.val[0]));
        }

        SIMD_INLINE float32x4_t Norm32fp(const HidHaarCascade & hid, size_t offset)
        {
            float32x4_t area = vdupq_n_f32(hid.windowArea);
            float32x4_t sum = vcvtq_f32_u32(Sum32ip(hid.p, offset));
            float32x4_t sqsum = vcvtq_f32_u32(Sum32ip(hid.pq, offset));
            return ValidSqrt(vmlsq_f32(vmulq_f32(sqsum, area), sum, sum));
        }

        SIMD_INLINE float32x4_t Norm32fi(const HidHaarCascade & hid, size_t offset)
        {
            float32x4_t area = vdupq_n_f32(hid.windowArea);
            float32x4_t sum = vcvtq_f32_u32(Sum32ii(hid.p, offset));
            float32x4_t sqsum = vcvtq_f32_u32(Sum32ii(hid.pq, offset));
            return ValidSqrt(vmlsq_f32(vmulq_f32(sqsum, area), sum, sum));
        }

        SIMD_INLINE float32x4_t WeightedSum32f(const WeightedRect & rect, size_t offset)
        {
            uint32x4_t s0 = vld1q_u32(rect.p0 + offset);
            uint32x4_t s1 = vld1q_u32(rect.p1 + offset);
            uint32x4_t s2 = vld1q_u32(rect.p2 + offset);
            uint32x4_t s3 = vld1q_u32(rect.p3 + offset);
            uint32x4_t sum = vsubq_u32(vsubq_u32(s0, s1), vsubq_u32(s2, s3));
            return vmulq_f32(vcvtq_f32_u32(sum), vdupq_n_f32(rect.weight));
        }

        SIMD_INLINE void StageSum32f(const float * leaves, float threshold, const float32x4_t & sum, const float32x4_t & norm, float32x4_t & stageSum)
        {
            uint32x4_t mask = vcltq_f32(sum, vmulq_f32(vdupq_n_f32(threshold), norm));
            stageSum = vaddq_f32(stageSum, vbslq_f32(mask, vdupq_n_f32(leaves[0]), vdupq_n_f32(leaves[1])));
        }

        void Detect32f(const HidHaarCascade & hid, size_t offset, const float32x4_t & norm, uint32x4_t & result)
        {
            typedef HidHaarCascade Hid;
            const float * leaves = hid.leaves.data();
            const Hid::Node * node = hid.nodes.data();
            const Hid::Stage * stages = hid.stages.data();
            for (int i = 0, n = (int)hid.stages.size(); i < n; ++i)
            {
                const Hid::Stage & stage = stages[i];
                if (stage.canSkip)
                    continue;
                const Hid::Node * end = node + stage.ntrees;
                float32x4_t stageSum = vdupq_n_f32(0.0f);
                if (stage.hasThree)
                {
                    for (; node < end; ++node, leaves += 2)
                    {
                        const Hid::Feature & feature = hid.features[node->featureIdx];
                        float32x4_t sum = vaddq_f32(WeightedSum32f(feature.rect[0], offset), WeightedSum32f(feature.rect[1], offset));
                        if (feature.rect[2].p0)
                            sum = vaddq_f32(sum, WeightedSum32f(feature.rect[2], offset));
                        StageSum32f(leaves, node->threshold, sum, norm, stageSum);
                    }
                }
                else
                {
                    for (; node < end; ++node, leaves += 2)
                    {
                        const Hid::Feature & feature = hid.features[node->featureIdx];
                        float32x4_t sum = vaddq_f32(WeightedSum32f(feature.rect[0], offset), WeightedSum32f(feature.rect[1], offset));
                        StageSum32f(leaves, node->threshold, sum, norm, stageSum);
                    }
                }
                result = vandq_u32(vcleq_f32(vdupq_n_f32(stage.threshold), stageSum), result);
                int resultCount = ResultCount(result);
                if (resultCount == 0)
                {
                    return;
                }
                else if (resultCount == 1)
                {
                    uint32_t _result[4];
                    float _norm[4];
                    vst1q_u32(_result, result);
                    vst1q_f32(_norm, norm);
                    for (int j = 0; j < 4; ++j)
                    {
                        if (_result[j])
                        {
                            _result[j] = Base::Detect32f(hid, offset + j, i + 1, _norm[j]) > 0 ? 1 : 0;
                            break;
                        }
                    }
                    result = vld1q_u32(_result);
                    return;
                }
            }
        }

        void DetectionHaarDetect32fp(const HidHaarCascade & hid, const Image & mask, const Rect & rect, Image & dst)
        {
            size_t width = rect.Width();
            size_t alignedWidth = Simd::AlignLo(width, F);
            size_t evenWidth = Simd::AlignLo(width, 2);

            Buffer<uint32_t> buffer(width);
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += 1)
            {
                size_t col = 0;
                size_t p_offset = row * hid.sum.stride / sizeof(uint32_t) + rect.left;
                size_t pq_offset = row * hid.sqsum.stride / sizeof(uint32_t) + rect.left;

                UnpackMask32i(mask.data + row*mask.stride + rect.left, width, buffer.m, K8_01);
                memset(buffer.d, 0, width * sizeof(uint32_t));
                for (; col < alignedWidth; col += F)
                {
                    uint32x4_t result = vld1q_u32(buffer.m + col);
                    if (ResultCount(result) == 0)
                        continue;
                    float32x4_t norm = Norm32fp(hid, pq_offset + col);
                    Detect32f(hid, p_offset + col, norm, result);
                    vst1q_u32(buffer.d + col, result);
                }
                if (evenWidth > alignedWidth + 2)
                {
                    col = evenWidth - F;
                    uint32x4_t result = vld1q_u32(buffer.m + col);
                    if (ResultCount(result) != 0)
                    {
                        float32x4_t norm = Norm32fp(hid, pq_offset + col);
                        Detect32f(hid, p_offset + col, norm, result);
                        vst1q_u32(buffer.d + col, result);
                    }
                    col += F;
                }
                for (; col < width; col += 1)
                {
                    if (buffer.m[col] == 0)
                        continue;
                    float norm = Base::Norm32f(hid, pq_offset + col);
                    buffer.d[col] = Base::Detect32f(hid, p_offset + col, 0, norm) > 0 ? 1 : 0;
                }
                PackResult32i(buffer.d, width, dst.data + row*dst.stride + rect.left);
            }
        }

        void DetectionHaarDetect32fp(const void * _hid, const uint8_t * mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
        {
            const HidHaarCascade & hid = *(HidHaarCascade*)_hid;
            return DetectionHaarDetect32fp(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }

        void DetectionHaarDetect32fi(const HidHaarCascade & hid, const Image & mask, const Rect & rect, Image & dst)
        {
            const size_t step = 2;
            size_t width = rect.Width();
            size_t alignedWidth = Simd::AlignLo(width, HA);
            size_t evenWidth = Simd::AlignLo(width, 2);

            Buffer<uint16_t> buffer(evenWidth);
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += step)
            {
                size_t col = 0;
                size_t p_offset = row * hid.isum.stride / sizeof(uint32_t) + rect.left / 2;
                size_t pq_offset = row * hid.sqsum.stride / sizeof(uint32_t) + rect.left;

                UnpackMask16i(mask.data + row*mask.stride + rect.left, evenWidth, buffer.m, (uint8x16_t)K16_0001);
                memset(buffer.d, 0, evenWidth * sizeof(uint16_t));
                for (; col < alignedWidth; col += HA)
                {
                    uint32x4_t result = (uint32x4_t)vld1q_u16(buffer.m + col);
                    if (ResultCount(result) == 0)
                        continue;
                    float32x4_t norm = Norm32fi(hid, pq_offset + col);
                    Detect32f(hid, p_offset + col / 2, norm, result);
                    vst1q_u16(buffer.d + col, (uint16x8_t)result);
                }
                if (evenWidth > alignedWidth)
                {
                    col = evenWidth - HA;
                    uint32x4_t result = (uint32x4_t)vld1q_u16(buffer.m + col);
                    if (ResultCount(result) != 0)
                    {
                        float32x4_t norm = Norm32fi(hid, pq_offset + col);
                        Detect32f(hid, p_offset + col / 2, norm, result);
                        vst1q_u16(buffer.d + col, (uint16x8_t)result);
                    }
                    col += HA;
                }
                for (; col < width; col += step)
                {
                    if (mask.At<uint8_t>(col + rect.left, row) == 0)
                        continue;
                    float norm = Base::Norm32f(hid, pq_offset + col);
                    if (Base::Detect32f(hid, p_offset + col / 2, 0, norm) > 0)
                        dst.At<uint8_t>(col + rect.left, row) = 1;
                }
                PackResult16i(buffer.d, evenWidth, dst.data + row*dst.stride + rect.left);
            }
        }

        void DetectionHaarDetect32fi(const void * _hid, const uint8_t * mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
        {
            const HidHaarCascade & hid = *(HidHaarCascade*)_hid;
            return DetectionHaarDetect32fi(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }

        const uint8x16_t K8_TBL_BITS = SIMD_VEC_SETR_EPI8(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);

        SIMD_INLINE uint32x4_t IntegralSum32i(const uint32x4_t & s0, const uint32x4_t & s1, const uint32x4_t & s2, const uint32x4_t & s3)
        {
            return vsubq_u32(vsubq_u32(s0, s1), vsubq_u32(s2, s3));
        }

        template<int i> SIMD_INLINE void Load(uint32x4_t a[16], const HidLbpFeature<int> & feature, ptrdiff_t offset)
        {
            a[i] = vld1q_u32((uint32_t*)feature.p[i] + offset);
        }

        SIMD_INLINE void Calculate(const HidLbpFeature<int> & feature, ptrdiff_t offset, uint32x4_t & shuffle, uint32x4_t & mask)
        {
            uint32x4_t a[16];
            Load<5>(a, feature, offset);
            Load<6>(a, feature, offset);
            Load<9>(a, feature, offset);
            Load<10>(a, feature, offset);
            uint32x4_t central = IntegralSum32i(a[5], a[6], a[9], a[10]);

            Load<0>(a, feature, offset);
            Load<1>(a, feature, offset);
            Load<4>(a, feature, offset);

            shuffle = K32_FFFFFF00;
            shuffle = vorrq_u32(shuffle, vandq_u32(vcgeq_u32(IntegralSum32i(a[0], a[1], a[4], a[5]), central), K32_00000010));
            Load<2>(a, feature, offset);
            shuffle = vorrq_u32(shuffle, vandq_u32(vcgeq_u32(IntegralSum32i(a[1], a[2], a[5], a[6]), central), K32_00000008));
            Load<3>(a, feature, offset);
            Load<7>(a, feature, offset);
            shuffle = vorrq_u32(shuffle, vandq_u32(vcgeq_u32(IntegralSum32i(a[2], a[3], a[6], a[7]), central), K32_00000004));
            Load<11>(a, feature, offset);
            shuffle = vorrq_u32(shuffle, vandq_u32(vcgeq_u32(IntegralSum32i(a[6], a[7], a[10], a[11]), central), K32_00000002));
            Load<14>(a, feature, offset);
            Load<15>(a, feature, offset);
            shuffle = vorrq_u32(shuffle, vandq_u32(vcgeq_u32(IntegralSum32i(a[10], a[11], a[14], a[15]), central), K32_00000001));

            mask = K32_08080800;
            Load<13>(a, feature, offset);
            mask = vorrq_u32(mask, vandq_u32(vcgeq_u32(IntegralSum32i(a[9], a[10], a[13], a[14]), central), K32_00000004));
            Load<12>(a, feature, offset);
            Load<8>(a, feature, offset);
            mask = vorrq_u32(mask, vandq_u32(vcgeq_u32(IntegralSum32i(a[8], a[9], a[12], a[13]), central), K32_00000002));
            mask = vorrq_u32(mask, vandq_u32(vcgeq_u32(IntegralSum32i(a[4], a[5], a[8], a[9]), central), K32_00000001));
            mask = (uint32x4_t)Shuffle(K8_TBL_BITS, (uint8x16_t)mask);
        }

        SIMD_INLINE uint32x4_t LeafMask(const HidLbpFeature<int> & feature, ptrdiff_t offset, const int * subset)
        {
            uint32x4_t shuffle, mask;
            Calculate(feature, offset, shuffle, mask);

            uint8x16x2_t _subset;
            _subset.val[0] = vld1q_u8((uint8_t*)subset + 0);
            _subset.val[1] = vld1q_u8((uint8_t*)subset + A);
            uint32x4_t value = vandq_u32((uint32x4_t)Shuffle(_subset, (uint8x16_t)shuffle), mask);

            return vmvnq_u32(vceqq_u32(value, K32_00000000));
        }

        void Detect(const HidLbpCascade<float, int> & hid, size_t offset, uint32x4_t & result)
        {
            typedef HidLbpCascade<float, int> Hid;

            size_t subsetSize = (hid.ncategories + 31) / 32;
            const int * subsets = hid.subsets.data();
            const Hid::Leave * leaves = hid.leaves.data();
            const Hid::Node * nodes = hid.nodes.data();
            const Hid::Stage * stages = hid.stages.data();
            int nodeOffset = 0, leafOffset = 0;
            for (int i_stage = 0, n_stages = (int)hid.stages.size(); i_stage < n_stages; i_stage++)
            {
                const Hid::Stage & stage = stages[i_stage];
                float32x4_t sum = vdupq_n_f32(0.0f);
                for (int i_tree = 0, n_trees = stage.ntrees; i_tree < n_trees; i_tree++)
                {
                    const Hid::Feature & feature = hid.features[nodes[nodeOffset].featureIdx];
                    const int * subset = subsets + nodeOffset*subsetSize;
                    uint32x4_t mask = LeafMask(feature, offset, subset);
                    sum = vaddq_f32(sum, vbslq_f32(mask, vdupq_n_f32(leaves[leafOffset + 0]), vdupq_n_f32(leaves[leafOffset + 1])));
                    nodeOffset++;
                    leafOffset += 2;
                }
                result = vandq_u32(vcleq_f32(vdupq_n_f32(stage.threshold), sum), result);
                int resultCount = ResultCount(result);
                if (resultCount == 0)
                    return;
                else if (resultCount == 1)
                {
                    uint32_t _result[4];
                    vst1q_u32(_result, result);
                    for (int i = 0; i < 4; ++i)
                    {
                        if (_result[i])
                        {
                            _result[i] = Base::Detect(hid, offset + i, i_stage + 1) > 0 ? 1 : 0;
                            break;
                        }
                    }
                    result = vld1q_u32(_result);
                    return;
                }
            }
        }

        void DetectionLbpDetect32fp(const HidLbpCascade<float, int> & hid, const Image & mask, const Rect & rect, Image & dst)
        {
            size_t width = rect.Width();
            size_t alignedWidth = Simd::AlignLo(width, 4);
            size_t evenWidth = Simd::AlignLo(width, 2);

            Buffer<uint32_t> buffer(width);
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += 1)
            {
                size_t col = 0;
                size_t offset = row * hid.sum.stride / sizeof(uint32_t) + rect.left;

                UnpackMask32i(mask.data + row*mask.stride + rect.left, width, buffer.m, K8_01);
                memset(buffer.d, 0, width * sizeof(uint32_t));
                for (; col < alignedWidth; col += 4)
                {
                    uint32x4_t result = vld1q_u32(buffer.m + col);
                    if (ResultCount(result) == 0)
                        continue;
                    Detect(hid, offset + col, result);
                    vst1q_u32(buffer.d + col, result);
                }
                if (evenWidth > alignedWidth + 2)
                {
                    col = evenWidth - 4;
                    uint32x4_t result = vld1q_u32(buffer.m + col);
                    if (ResultCount(result) != 0)
                    {
                        Detect(hid, offset + col, result);
                        vst1q_u32(buffer.d + col, result);
                    }
                    col += 4;
                }
                for (; col < width; col += 1)
                {
                    if (buffer.m[col] == 0)
                        continue;
                    buffer.d[col] = Base::Detect(hid, offset + col, 0) > 0 ? 1 : 0;
                }
                PackResult32i(buffer.d, width, dst.data + row*dst.stride + rect.left);
            }
        }

        void DetectionLbpDetect32fp(const void * _hid, const uint8_t * mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
        {
            const HidLbpCascade<float, int> & hid = *(HidLbpCascade<float, int>*)_hid;
            return DetectionLbpDetect32fp(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }

        void DetectionLbpDetect32fi(const HidLbpCascade<float, int> & hid, const Image & mask, const Rect & rect, Image & dst)
        {
            const size_t step = 2;
            size_t width = rect.Width();
            size_t alignedWidth = Simd::AlignLo(width, HA);
            size_t evenWidth = Simd::AlignLo(width, 2);

            Buffer<uint16_t> buffer(evenWidth);
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += step)
            {
                size_t col = 0;
                size_t offset = row * hid.isum.stride / sizeof(uint32_t) + rect.left / 2;

                UnpackMask16i(mask.data + row*mask.stride + rect.left, evenWidth, buffer.m, (uint8x16_t)K16_0001);
                memset(buffer.d, 0, evenWidth * sizeof(uint16_t));
                for (; col < alignedWidth; col += HA)
                {
                    uint32x4_t result = (uint32x4_t)vld1q_u16(buffer.m + col);
                    if (ResultCount(result) == 0)
                        continue;
                    Detect(hid, offset + col / 2, result);
                    vst1q_u16(buffer.d + col, (uint16x8_t)result);
                }
                if (evenWidth > alignedWidth)
                {
                    col = evenWidth - HA;
                    uint32x4_t result = (uint32x4_t)vld1q_u16(buffer.m + col);
                    if (ResultCount(result) != 0)
                    {
                        Detect(hid, offset + col / 2, result);
                        vst1q_u16(buffer.d + col, (uint16x8_t)result);
                    }
                    col += HA;
                }
                for (; col < width; col += step)
                {
                    if (mask.At<uint8_t>(col + rect.left, row) == 0)
                        continue;
                    if (Base::Detect(hid, offset + col / 2, 0) > 0)
                        dst.At<uint8_t>(col + rect.left, row) = 1;
                }
                PackResult16i(buffer.d, evenWidth, dst.data + row*dst.stride + rect.left);
            }
        }

        void DetectionLbpDetect32fi(const void * _hid, const uint8_t * mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
        {
            const HidLbpCascade<float, int> & hid = *(HidLbpCascade<float, int>*)_hid;
            return DetectionLbpDetect32fi(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }

        SIMD_INLINE uint16x8_t IntegralSum16i(const uint16x8_t & s0, const uint16x8_t & s1, const uint16x8_t & s2, const uint16x8_t & s3)
        {
            return vsubq_u16(vsubq_u16(s0, s1), vsubq_u16(s2, s3));
        }

        template<int i> SIMD_INLINE void Load(uint16x8_t a[16], const HidLbpFeature<short> & feature, ptrdiff_t offset)
        {
            a[i] = vld1q_u16((uint16_t*)feature.p[i] + offset);
        }

        SIMD_INLINE void Calculate(const HidLbpFeature<short> & feature, ptrdiff_t offset, uint16x8_t & shuffle, uint16x8_t & mask)
        {
            uint16x8_t a[16];
            Load<5>(a, feature, offset);
            Load<6>(a, feature, offset);
            Load<9>(a, feature, offset);
            Load<10>(a, feature, offset);
            uint16x8_t central = IntegralSum16i(a[5], a[6], a[9], a[10]);

            Load<0>(a, feature, offset);
            Load<1>(a, feature, offset);
            Load<4>(a, feature, offset);

            shuffle = K16_FF00;
            shuffle = vorrq_u16(shuffle, vandq_u16(vcgeq_u16(IntegralSum16i(a[0], a[1], a[4], a[5]), central), K16_0010));
            Load<2>(a, feature, offset);
            shuffle = vorrq_u16(shuffle, vandq_u16(vcgeq_u16(IntegralSum16i(a[1], a[2], a[5], a[6]), central), K16_0008));
            Load<3>(a, feature, offset);
            Load<7>(a, feature, offset);
            shuffle = vorrq_u16(shuffle, vandq_u16(vcgeq_u16(IntegralSum16i(a[2], a[3], a[6], a[7]), central), K16_0004));
            Load<11>(a, feature, offset);
            shuffle = vorrq_u16(shuffle, vandq_u16(vcgeq_u16(IntegralSum16i(a[6], a[7], a[10], a[11]), central), K16_0002));
            Load<14>(a, feature, offset);
            Load<15>(a, feature, offset);
            shuffle = vorrq_u16(shuffle, vandq_u16(vcgeq_u16(IntegralSum16i(a[10], a[11], a[14], a[15]), central), K16_0001));

            mask = K16_0800;
            Load<13>(a, feature, offset);
            mask = vorrq_u16(mask, vandq_u16(vcgeq_u16(IntegralSum16i(a[9], a[10], a[13], a[14]), central), K16_0004));
            Load<12>(a, feature, offset);
            Load<8>(a, feature, offset);
            mask = vorrq_u16(mask, vandq_u16(vcgeq_u16(IntegralSum16i(a[8], a[9], a[12], a[13]), central), K16_0002));
            mask = vorrq_u16(mask, vandq_u16(vcgeq_u16(IntegralSum16i(a[4], a[5], a[8], a[9]), central), K16_0001));
            mask = (uint16x8_t)Shuffle(K8_TBL_BITS, (uint8x16_t)mask);
        }

        SIMD_INLINE uint16x8_t LeafMask(const HidLbpFeature<short> & feature, ptrdiff_t offset, const int * subset)
        {
            uint16x8_t shuffle, mask;
            Calculate(feature, offset, shuffle, mask);

            uint8x16x2_t _subset;
            _subset.val[0] = vld1q_u8((uint8_t*)subset + 0);
            _subset.val[1] = vld1q_u8((uint8_t*)subset + A);
            uint16x8_t value = vandq_u16((uint16x8_t)Shuffle(_subset, (uint8x16_t)shuffle), mask);

            return vmvnq_u16(vceqq_u16(value, K16_0000));
        }

        void Detect(const HidLbpCascade<int, short> & hid, size_t offset, uint16x8_t & result)
        {
            typedef HidLbpCascade<int, short> Hid;

            size_t subsetSize = (hid.ncategories + 31) / 32;
            const int * subsets = hid.subsets.data();
            const Hid::Leave * leaves = hid.leaves.data();
            const Hid::Node * nodes = hid.nodes.data();
            const Hid::Stage * stages = hid.stages.data();
            int nodeOffset = 0, leafOffset = 0;
            for (int i_stage = 0, n_stages = (int)hid.stages.size(); i_stage < n_stages; i_stage++)
            {
                const Hid::Stage & stage = stages[i_stage];
                int16x8_t sum = vdupq_n_s16(0);
                for (int i_tree = 0, n_trees = stage.ntrees; i_tree < n_trees; i_tree++)
                {
                    const Hid::Feature & feature = hid.features[nodes[nodeOffset].featureIdx];
                    const int * subset = subsets + nodeOffset*subsetSize;
                    uint16x8_t mask = LeafMask(feature, offset, subset);
                    sum = vaddq_s16(sum, vbslq_s16(mask, vdupq_n_s16(leaves[leafOffset + 0]), vdupq_n_s16(leaves[leafOffset + 1])));
                    nodeOffset++;
                    leafOffset += 2;
                }
                result = vandq_u16(vcleq_s16(vdupq_n_s16(stage.threshold), sum), result);
                int resultCount = ResultCount(result);
                if (resultCount == 0)
                    return;
                else if (resultCount == 1)
                {
                    uint16_t _result[HA];
                    vst1q_u16(_result, result);
                    for (int i = 0; i < HA; ++i)
                    {
                        if (_result[i])
                        {
                            _result[i] = Base::Detect(hid, offset + i, i_stage + 1) > 0 ? 1 : 0;
                            break;
                        }
                    }
                    result = vld1q_u16(_result);
                    return;
                }
            }
        }

        void DetectionLbpDetect16ip(const HidLbpCascade<int, short> & hid, const Image & mask, const Rect & rect, Image & dst)
        {
            size_t width = rect.Width();
            size_t alignedWidth = Simd::AlignLo(width, HA);
            size_t evenWidth = Simd::AlignLo(width, 2);
            Buffer<uint16_t> buffer(width);
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += 1)
            {
                size_t col = 0;
                size_t offset = row * hid.isum.stride / sizeof(uint16_t) + rect.left;
                UnpackMask16i(mask.data + row*mask.stride + rect.left, width, buffer.m, K8_01);
                memset(buffer.d, 0, width * sizeof(uint16_t));
                for (; col < alignedWidth; col += HA)
                {
                    uint16x8_t result = vld1q_u16(buffer.m + col);
                    if (ResultCount(result) == 0)
                        continue;
                    Detect(hid, offset + col, result);
                    vst1q_u16(buffer.d + col, result);
                }
                if (evenWidth > alignedWidth + 2)
                {
                    col = evenWidth - HA;
                    uint16x8_t result = vld1q_u16(buffer.m + col);
                    if (ResultCount(result) != 0)
                    {
                        Detect(hid, offset + col, result);
                        vst1q_u16(buffer.d + col, result);
                    }
                    col += HA;
                }
                for (; col < width; ++col)
                {
                    if (buffer.m[col] == 0)
                        continue;
                    buffer.d[col] = Base::Detect(hid, offset + col, 0) > 0 ? 1 : 0;
                }
                PackResult16i(buffer.d, width, dst.data + row*dst.stride + rect.left);
            }
        }

        void DetectionLbpDetect16ip(const void * _hid, const uint8_t * mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
        {
            const HidLbpCascade<int, short> & hid = *(HidLbpCascade<int, short>*)_hid;
            return DetectionLbpDetect16ip(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }


        void DetectionLbpDetect16ii(const HidLbpCascade<int, short> & hid, const Image & mask, const Rect & rect, Image & dst)
        {
            const size_t step = 2;
            size_t width = rect.Width();
            size_t alignedWidth = Simd::AlignLo(width, A);
            size_t evenWidth = Simd::AlignLo(width, 2);

            for (ptrdiff_t row = rect.top; row < rect.bottom; row += step)
            {
                size_t col = 0;
                size_t offset = row * hid.isum.stride / sizeof(uint16_t) + rect.left / 2;
                const uint8_t * m = mask.data + row*mask.stride + rect.left;
                uint8_t * d = dst.data + row*dst.stride + rect.left;
                for (; col < alignedWidth; col += A)
                {
                    uint16x8_t result = vandq_u16((uint16x8_t)vld1q_u8(m + col), K16_0001);
                    if (ResultCount(result) == 0)
                        continue;
                    Detect(hid, offset + col / 2, result);
                    vst1q_u8(d + col, (uint8x16_t)result);
                }
                if (evenWidth > alignedWidth + 2)
                {
                    col = evenWidth - A;
                    uint16x8_t result = vandq_u16((uint16x8_t)vld1q_u8(m + col), K16_0001);
                    if (ResultCount(result) != 0)
                    {
                        Detect(hid, offset + col / 2, result);
                        vst1q_u8(d + col, (uint8x16_t)result);
                    }
                    col += A;
                }
                for (; col < width; col += step)
                {
                    if (mask.At<uint8_t>(col + rect.left, row) == 0)
                        continue;
                    if (Base::Detect(hid, offset + col / 2, 0) > 0)
                        dst.At<uint8_t>(col + rect.left, row) = 1;
                }
            }
        }

        void DetectionLbpDetect16ii(const void * _hid, const uint8_t * mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
        {
            const HidLbpCascade<int, short> & hid = *(HidLbpCascade<int, short>*)_hid;
            return DetectionLbpDetect16ii(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }
    }
#endif// SIMD_NEON_ENABLE
}
