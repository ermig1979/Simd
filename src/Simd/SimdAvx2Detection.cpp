/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        using namespace Simd::Detection;

        const __m256i K32_PERMUTE = SIMD_MM256_SETR_EPI32(0, 2, 4, 6, 1, 3, 5, 7);

        SIMD_INLINE void UnpackMask16i(const uint8_t * src, uint16_t * dst, const __m256i & mask)
        {
            __m256i s = _mm256_and_si256(mask, LoadPermuted<false>((__m256i*)src));
            _mm256_storeu_si256((__m256i*)dst + 0, _mm256_unpacklo_epi8(s, _mm256_setzero_si256()));
            _mm256_storeu_si256((__m256i*)dst + 1, _mm256_unpackhi_epi8(s, _mm256_setzero_si256()));
        }

        SIMD_INLINE void UnpackMask16i(const uint8_t * src, size_t size, uint16_t * dst, const __m256i & mask)
        {
            size_t alignedSize = Simd::AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                UnpackMask16i(src + i, dst + i, mask);
            if (size != alignedSize)
                UnpackMask16i(src + size - A, dst + size - A, mask);
        }

        SIMD_INLINE void UnpackMask32i(const uint8_t * src, uint32_t * dst, const __m256i & mask)
        {
            __m256i s = _mm256_permutevar8x32_epi32(_mm256_and_si256(mask, _mm256_loadu_si256((__m256i*)src)), K32_PERMUTE);
            __m256i lo = _mm256_unpacklo_epi8(s, _mm256_setzero_si256());
            _mm256_storeu_si256((__m256i*)dst + 0, _mm256_unpacklo_epi16(lo, _mm256_setzero_si256()));
            _mm256_storeu_si256((__m256i*)dst + 1, _mm256_unpackhi_epi16(lo, _mm256_setzero_si256()));
            __m256i hi = _mm256_unpackhi_epi8(s, _mm256_setzero_si256());
            _mm256_storeu_si256((__m256i*)dst + 2, _mm256_unpacklo_epi16(hi, _mm256_setzero_si256()));
            _mm256_storeu_si256((__m256i*)dst + 3, _mm256_unpackhi_epi16(hi, _mm256_setzero_si256()));
        }

        SIMD_INLINE void UnpackMask32i(const uint8_t * src, size_t size, uint32_t * dst, const __m256i & mask)
        {
            size_t alignedSize = Simd::AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                UnpackMask32i(src + i, dst + i, mask);
            if (size != alignedSize)
                UnpackMask32i(src + size - A, dst + size - A, mask);
        }

        SIMD_INLINE void PackResult16i(const uint16_t * src, uint8_t * dst)
        {
            __m256i lo = _mm256_loadu_si256((__m256i*)src + 0);
            __m256i hi = _mm256_loadu_si256((__m256i*)src + 1);
            _mm256_storeu_si256((__m256i*)dst, PackI16ToU8(lo, hi));
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
            const __m256i lo = Simd::Avx2::PackI32ToI16(_mm256_loadu_si256((__m256i*)src + 0), _mm256_loadu_si256((__m256i*)src + 1));
            const __m256i hi = Simd::Avx2::PackI32ToI16(_mm256_loadu_si256((__m256i*)src + 2), _mm256_loadu_si256((__m256i*)src + 3));
            _mm256_storeu_si256((__m256i*)dst, Simd::Avx2::PackI16ToU8(lo, hi));
        }

        SIMD_INLINE void PackResult32i(const uint32_t * src, size_t size, uint8_t * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                PackResult32i(src + i, dst + i);
            if (size != alignedSize)
                PackResult32i(src + size - A, dst + size - A);
        }

        SIMD_INLINE int ResultCount(__m256i result)
        {
            uint32_t SIMD_ALIGNED(32) buffer[8];
            _mm256_store_si256((__m256i*)buffer, _mm256_sad_epu8(result, _mm256_setzero_si256()));
            return buffer[0] + buffer[2] + buffer[4] + buffer[6];
        }

        SIMD_INLINE __m256 ValidSqrt(__m256 value)
        {
            __m256 mask = _mm256_cmp_ps(value, _mm256_set1_ps(0.0f), _CMP_GT_OQ);
            return _mm256_sqrt_ps(_mm256_or_ps(_mm256_and_ps(mask, value), _mm256_andnot_ps(mask, _mm256_set1_ps(1.0f))));
        }

        SIMD_INLINE __m256i Sum32ip(uint32_t * const ptr[4], size_t offset)
        {
            __m256i s0 = _mm256_loadu_si256((__m256i*)(ptr[0] + offset));
            __m256i s1 = _mm256_loadu_si256((__m256i*)(ptr[1] + offset));
            __m256i s2 = _mm256_loadu_si256((__m256i*)(ptr[2] + offset));
            __m256i s3 = _mm256_loadu_si256((__m256i*)(ptr[3] + offset));
            return _mm256_sub_epi32(_mm256_sub_epi32(s0, s1), _mm256_sub_epi32(s2, s3));
        }

        SIMD_INLINE __m256i Sum32ii(uint32_t * const ptr[4], size_t offset)
        {
            __m256i lo = Sum32ip(ptr, offset + 0);
            __m256i hi = Sum32ip(ptr, offset + 8);
            return _mm256_permute2x128_si256(
                _mm256_permutevar8x32_epi32(lo, K32_PERMUTE),
                _mm256_permutevar8x32_epi32(hi, K32_PERMUTE), 0x20);
        }

        SIMD_INLINE __m256 Norm32fp(const HidHaarCascade & hid, size_t offset)
        {
            __m256 area = _mm256_broadcast_ss(&hid.windowArea);
            __m256 sum = _mm256_cvtepi32_ps(Sum32ip(hid.p, offset));
            __m256 sqsum = _mm256_cvtepi32_ps(Sum32ip(hid.pq, offset));
            return ValidSqrt(_mm256_sub_ps(_mm256_mul_ps(sqsum, area), _mm256_mul_ps(sum, sum)));
        }

        SIMD_INLINE __m256 Norm32fi(const HidHaarCascade & hid, size_t offset)
        {
            __m256 area = _mm256_broadcast_ss(&hid.windowArea);
            __m256 sum = _mm256_cvtepi32_ps(Sum32ii(hid.p, offset));
            __m256 sqsum = _mm256_cvtepi32_ps(Sum32ii(hid.pq, offset));
            return ValidSqrt(_mm256_sub_ps(_mm256_mul_ps(sqsum, area), _mm256_mul_ps(sum, sum)));
        }

        SIMD_INLINE __m256 WeightedSum32f(const WeightedRect & rect, size_t offset)
        {
            __m256i s0 = _mm256_loadu_si256((__m256i*)(rect.p0 + offset));
            __m256i s1 = _mm256_loadu_si256((__m256i*)(rect.p1 + offset));
            __m256i s2 = _mm256_loadu_si256((__m256i*)(rect.p2 + offset));
            __m256i s3 = _mm256_loadu_si256((__m256i*)(rect.p3 + offset));
            __m256i sum = _mm256_sub_epi32(_mm256_sub_epi32(s0, s1), _mm256_sub_epi32(s2, s3));
            return _mm256_mul_ps(_mm256_cvtepi32_ps(sum), _mm256_broadcast_ss(&rect.weight));
        }

        SIMD_INLINE void StageSum32f(const float * leaves, float threshold, const __m256 & sum, const __m256 & norm, __m256 & stageSum)
        {
            __m256 mask = _mm256_cmp_ps(_mm256_mul_ps(_mm256_set1_ps(threshold), norm), sum, _CMP_GT_OQ);
            stageSum = _mm256_add_ps(stageSum, _mm256_blendv_ps(_mm256_broadcast_ss(leaves + 1), _mm256_broadcast_ss(leaves + 0), mask));
        }

        void Detect32f(const HidHaarCascade & hid, size_t offset, const __m256 & norm, __m256i & result)
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
                __m256 stageSum = _mm256_setzero_ps();
                if (stage.hasThree)
                {
                    for (; node < end; ++node, leaves += 2)
                    {
                        const Hid::Feature & feature = hid.features[node->featureIdx];
                        __m256 sum = _mm256_add_ps(WeightedSum32f(feature.rect[0], offset), WeightedSum32f(feature.rect[1], offset));
                        if (feature.rect[2].p0)
                            sum = _mm256_add_ps(sum, WeightedSum32f(feature.rect[2], offset));
                        StageSum32f(leaves, node->threshold, sum, norm, stageSum);
                    }
                }
                else
                {
                    for (; node < end; ++node, leaves += 2)
                    {
                        const Hid::Feature & feature = hid.features[node->featureIdx];
                        __m256 sum = _mm256_add_ps(WeightedSum32f(feature.rect[0], offset), WeightedSum32f(feature.rect[1], offset));
                        StageSum32f(leaves, node->threshold, sum, norm, stageSum);
                    }
                }
                result = _mm256_andnot_si256(_mm256_castps_si256(_mm256_cmp_ps(_mm256_broadcast_ss(&stage.threshold), stageSum, _CMP_GT_OQ)), result);
                int resultCount = ResultCount(result);
                if (resultCount == 0)
                    return;
                else if (resultCount == 1)
                {
                    uint32_t SIMD_ALIGNED(32) _result[8];
                    float SIMD_ALIGNED(32) _norm[8];
                    _mm256_store_si256((__m256i*)_result, result);
                    _mm256_store_ps(_norm, norm);
                    for (int j = 0; j < 8; ++j)
                    {
                        if (_result[j])
                        {
                            _result[j] = Base::Detect32f(hid, offset + j, i + 1, _norm[j]) > 0 ? 1 : 0;
                            break;
                        }
                    }
                    result = _mm256_load_si256((__m256i*)_result);
                    return;
                }
            }
        }

        void DetectionHaarDetect32fp(const HidHaarCascade & hid, const Image & mask, const Rect & rect, Image & dst)
        {
            size_t width = rect.Width();
            size_t alignedWidth = Simd::AlignLo(width, 8);
            size_t evenWidth = Simd::AlignLo(width, 2);

            Buffer<uint32_t> buffer(width);
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += 1)
            {
                size_t col = 0;
                size_t p_offset = row * hid.sum.stride / sizeof(uint32_t) + rect.left;
                size_t pq_offset = row * hid.sqsum.stride / sizeof(uint32_t) + rect.left;

                UnpackMask32i(mask.data + row*mask.stride + rect.left, width, buffer.m, K8_01);
                memset(buffer.d, 0, width * sizeof(uint32_t));
                for (; col < alignedWidth; col += 8)
                {
                    __m256i result = _mm256_loadu_si256((__m256i*)(buffer.m + col));
                    if (_mm256_testz_si256(result, K32_00000001))
                        continue;
                    __m256 norm = Norm32fp(hid, pq_offset + col);
                    Detect32f(hid, p_offset + col, norm, result);
                    _mm256_storeu_si256((__m256i*)(buffer.d + col), result);
                }
                if (evenWidth > alignedWidth + 2)
                {
                    col = evenWidth - 8;
                    __m256i result = _mm256_loadu_si256((__m256i*)(buffer.m + col));
                    if (!_mm256_testz_si256(result, K32_00000001))
                    {
                        __m256 norm = Norm32fp(hid, pq_offset + col);
                        Detect32f(hid, p_offset + col, norm, result);
                        _mm256_storeu_si256((__m256i*)(buffer.d + col), result);
                    }
                    col += 8;
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

                UnpackMask16i(mask.data + row*mask.stride + rect.left, evenWidth, buffer.m, K16_0001);
                memset(buffer.d, 0, evenWidth * sizeof(uint16_t));
                for (; col < alignedWidth; col += HA)
                {
                    __m256i result = _mm256_loadu_si256((__m256i*)(buffer.m + col));
                    if (_mm256_testz_si256(result, K32_00000001))
                        continue;
                    __m256 norm = Norm32fi(hid, pq_offset + col);
                    Detect32f(hid, p_offset + col / 2, norm, result);
                    _mm256_storeu_si256((__m256i*)(buffer.d + col), result);
                }
                if (evenWidth > alignedWidth)
                {
                    col = evenWidth - HA;
                    __m256i result = _mm256_loadu_si256((__m256i*)(buffer.m + col));
                    if (!_mm256_testz_si256(result, K32_00000001))
                    {
                        __m256 norm = Norm32fi(hid, pq_offset + col);
                        Detect32f(hid, p_offset + col / 2, norm, result);
                        _mm256_storeu_si256((__m256i*)(buffer.d + col), result);
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

        const __m256i K8_SHUFFLE_BITS = SIMD_MM256_SETR_EPI8(
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);

        SIMD_INLINE __m256i IntegralSum32i(const __m256i & s0, const __m256i & s1, const __m256i & s2, const __m256i & s3)
        {
            return _mm256_sub_epi32(_mm256_sub_epi32(s0, s1), _mm256_sub_epi32(s2, s3));
        }

        SIMD_INLINE __m256i GreaterOrEqual32i(__m256i a, __m256i b)
        {
            return _mm256_cmpeq_epi32(_mm256_max_epu32(a, b), a);
        }

        template<int i> SIMD_INLINE void Load(__m256i a[16], const HidLbpFeature<uint32_t> & feature, ptrdiff_t offset)
        {
            a[i] = _mm256_loadu_si256((__m256i*)(feature.p[i] + offset));
        }

        SIMD_INLINE void Calculate(const HidLbpFeature<uint32_t> & feature, ptrdiff_t offset, __m256i & index, __m256i & shuffle, __m256i & mask)
        {
            __m256i a[16];
            Load<5>(a, feature, offset);
            Load<6>(a, feature, offset);
            Load<9>(a, feature, offset);
            Load<10>(a, feature, offset);
            __m256i central = IntegralSum32i(a[5], a[6], a[9], a[10]);

            Load<0>(a, feature, offset);
            Load<1>(a, feature, offset);
            Load<4>(a, feature, offset);
            index = GreaterOrEqual32i(IntegralSum32i(a[0], a[1], a[4], a[5]), central);

            shuffle = K32_FFFFFF00;
            Load<2>(a, feature, offset);
            shuffle = _mm256_or_si256(shuffle, _mm256_and_si256(GreaterOrEqual32i(IntegralSum32i(a[1], a[2], a[5], a[6]), central), K32_00000008));
            Load<3>(a, feature, offset);
            Load<7>(a, feature, offset);
            shuffle = _mm256_or_si256(shuffle, _mm256_and_si256(GreaterOrEqual32i(IntegralSum32i(a[2], a[3], a[6], a[7]), central), K32_00000004));
            Load<11>(a, feature, offset);
            shuffle = _mm256_or_si256(shuffle, _mm256_and_si256(GreaterOrEqual32i(IntegralSum32i(a[6], a[7], a[10], a[11]), central), K32_00000002));
            Load<14>(a, feature, offset);
            Load<15>(a, feature, offset);
            shuffle = _mm256_or_si256(shuffle, _mm256_and_si256(GreaterOrEqual32i(IntegralSum32i(a[10], a[11], a[14], a[15]), central), K32_00000001));

            mask = K32_FFFFFF00;
            Load<13>(a, feature, offset);
            mask = _mm256_or_si256(mask, _mm256_and_si256(GreaterOrEqual32i(IntegralSum32i(a[9], a[10], a[13], a[14]), central), K32_00000004));
            Load<12>(a, feature, offset);
            Load<8>(a, feature, offset);
            mask = _mm256_or_si256(mask, _mm256_and_si256(GreaterOrEqual32i(IntegralSum32i(a[8], a[9], a[12], a[13]), central), K32_00000002));
            mask = _mm256_or_si256(mask, _mm256_and_si256(GreaterOrEqual32i(IntegralSum32i(a[4], a[5], a[8], a[9]), central), K32_00000001));
            mask = _mm256_shuffle_epi8(K8_SHUFFLE_BITS, mask);
        }

        SIMD_INLINE __m256i LeafMask(const HidLbpFeature<uint32_t> & feature, ptrdiff_t offset, const int * subset)
        {
            __m256i index, shuffle, mask;
            Calculate(feature, offset, index, shuffle, mask);

            __m256i _subset = _mm256_loadu_si256((__m256i*)subset);
            __m256i subset0 = _mm256_permute4x64_epi64(_subset, 0x44);
            __m256i subset1 = _mm256_permute4x64_epi64(_subset, 0xEE);

            __m256i value0 = _mm256_and_si256(_mm256_shuffle_epi8(subset0, shuffle), mask);
            __m256i value1 = _mm256_and_si256(_mm256_shuffle_epi8(subset1, shuffle), mask);
            __m256i value = _mm256_blendv_epi8(value0, value1, index);

            return _mm256_andnot_si256(_mm256_cmpeq_epi32(value, _mm256_setzero_si256()), K_INV_ZERO);
        }

        void Detect(const HidLbpCascade<float, uint32_t> & hid, size_t offset, int startStage, __m256i & result)
        {
            typedef HidLbpCascade<float, uint32_t> Hid;

            size_t subsetSize = (hid.ncategories + 31) / 32;
            const int * subsets = hid.subsets.data();
            const Hid::Leave * leaves = hid.leaves.data();
            const Hid::Node * nodes = hid.nodes.data();
            const Hid::Stage * stages = hid.stages.data();
            int nodeOffset = stages[startStage].first;
            int leafOffset = 2 * nodeOffset;
            for (int i_stage = startStage, n_stages = (int)hid.stages.size(); i_stage < n_stages; i_stage++)
            {
                const Hid::Stage & stage = stages[i_stage];
                __m256 sum = _mm256_setzero_ps();
                for (int i_tree = 0, n_trees = stage.ntrees; i_tree < n_trees; i_tree++)
                {
                    const Hid::Feature & feature = hid.features[nodes[nodeOffset].featureIdx];
                    const int * subset = subsets + nodeOffset*subsetSize;
                    __m256i mask = LeafMask(feature, offset, subset);
                    sum = _mm256_add_ps(sum, _mm256_blendv_ps(_mm256_broadcast_ss(leaves + leafOffset + 1), _mm256_broadcast_ss(leaves + leafOffset + 0), _mm256_castsi256_ps(mask)));
                    nodeOffset++;
                    leafOffset += 2;
                }
                result = _mm256_andnot_si256(_mm256_castps_si256(_mm256_cmp_ps(_mm256_broadcast_ss(&stage.threshold), sum, _CMP_GT_OQ)), result);
                int resultCount = ResultCount(result);
                if (resultCount == 0)
                    return;
                else if (resultCount == 1)
                {
                    uint32_t SIMD_ALIGNED(32) _result[8];
                    _mm256_store_si256((__m256i*)_result, result);
                    for (int i = 0; i < 8; ++i)
                    {
                        if (_result[i])
                        {
                            _result[i] = Base::Detect(hid, offset + i, i_stage + 1) > 0 ? 1 : 0;
                            break;
                        }
                    }
                    result = _mm256_load_si256((__m256i*)_result);
                    return;
                }
            }
        }

        void DetectionLbpDetect32fp(const HidLbpCascade<float, uint32_t> & hid, const Image & mask, const Rect & rect, Image & dst)
        {
            size_t width = rect.Width();
            size_t alignedWidth = Simd::AlignLo(width, 8);
            size_t evenWidth = Simd::AlignLo(width, 2);

            Buffer<uint32_t> buffer(width);
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += 1)
            {
                size_t col = 0;
                size_t offset = row * hid.sum.stride / sizeof(uint32_t) + rect.left;

                UnpackMask32i(mask.data + row*mask.stride + rect.left, width, buffer.m, K8_01);
                memset(buffer.d, 0, width * sizeof(uint32_t));
                for (; col < alignedWidth; col += 8)
                {
                    __m256i result = _mm256_loadu_si256((__m256i*)(buffer.m + col));
                    if (_mm256_testz_si256(result, K32_00000001))
                        continue;
                    Detect(hid, offset + col, 0, result);
                    _mm256_storeu_si256((__m256i*)(buffer.d + col), result);
                }
                if (evenWidth > alignedWidth + 2)
                {
                    col = evenWidth - 8;
                    __m256i result = _mm256_loadu_si256((__m256i*)(buffer.m + col));
                    if (!_mm256_testz_si256(result, K32_00000001))
                    {
                        Detect(hid, offset + col, 0, result);
                        _mm256_storeu_si256((__m256i*)(buffer.d + col), result);
                    }
                    col += 8;
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
            const HidLbpCascade<float, uint32_t> & hid = *(HidLbpCascade<float, uint32_t>*)_hid;
            return DetectionLbpDetect32fp(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }

        void DetectionLbpDetect32fi(const HidLbpCascade<float, uint32_t> & hid, const Image & mask, const Rect & rect, Image & dst)
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

                UnpackMask16i(mask.data + row*mask.stride + rect.left, evenWidth, buffer.m, K16_0001);
                memset(buffer.d, 0, evenWidth * sizeof(uint16_t));
                for (; col < alignedWidth; col += HA)
                {
                    __m256i result = _mm256_loadu_si256((__m256i*)(buffer.m + col));
                    if (_mm256_testz_si256(result, K32_00000001))
                        continue;
                    Detect(hid, offset + col / 2, 0, result);
                    _mm256_storeu_si256((__m256i*)(buffer.d + col), result);
                }
                if (evenWidth > alignedWidth)
                {
                    col = evenWidth - HA;
                    __m256i result = _mm256_loadu_si256((__m256i*)(buffer.m + col));
                    if (!_mm256_testz_si256(result, K32_00000001))
                    {
                        Detect(hid, offset + col / 2, 0, result);
                        _mm256_storeu_si256((__m256i*)(buffer.d + col), result);
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
            const HidLbpCascade<float, uint32_t> & hid = *(HidLbpCascade<float, uint32_t>*)_hid;
            return DetectionLbpDetect32fi(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }

        SIMD_INLINE __m256i IntegralSum16i(const __m256i & s0, const __m256i & s1, const __m256i & s2, const __m256i & s3)
        {
            return _mm256_sub_epi16(_mm256_sub_epi16(s0, s1), _mm256_sub_epi16(s2, s3));
        }

        SIMD_INLINE __m256i GreaterOrEqual16i(__m256i a, __m256i b)
        {
            return _mm256_cmpeq_epi16(_mm256_max_epu16(a, b), a);
        }

        template<int i> SIMD_INLINE void Load(__m256i a[16], const HidLbpFeature<uint16_t> & feature, ptrdiff_t offset)
        {
            a[i] = _mm256_loadu_si256((__m256i*)(feature.p[i] + offset));
        }

        SIMD_INLINE void Calculate(const HidLbpFeature<uint16_t> & feature, ptrdiff_t offset, __m256i & index, __m256i & shuffle, __m256i & mask)
        {
            __m256i a[16];
            Load<5>(a, feature, offset);
            Load<6>(a, feature, offset);
            Load<9>(a, feature, offset);
            Load<10>(a, feature, offset);
            __m256i central = IntegralSum16i(a[5], a[6], a[9], a[10]);

            Load<0>(a, feature, offset);
            Load<1>(a, feature, offset);
            Load<4>(a, feature, offset);
            index = GreaterOrEqual16i(IntegralSum16i(a[0], a[1], a[4], a[5]), central);

            shuffle = K16_FF00;
            Load<2>(a, feature, offset);
            shuffle = _mm256_or_si256(shuffle, _mm256_and_si256(GreaterOrEqual16i(IntegralSum16i(a[1], a[2], a[5], a[6]), central), K16_0008));
            Load<3>(a, feature, offset);
            Load<7>(a, feature, offset);
            shuffle = _mm256_or_si256(shuffle, _mm256_and_si256(GreaterOrEqual16i(IntegralSum16i(a[2], a[3], a[6], a[7]), central), K16_0004));
            Load<11>(a, feature, offset);
            shuffle = _mm256_or_si256(shuffle, _mm256_and_si256(GreaterOrEqual16i(IntegralSum16i(a[6], a[7], a[10], a[11]), central), K16_0002));
            Load<14>(a, feature, offset);
            Load<15>(a, feature, offset);
            shuffle = _mm256_or_si256(shuffle, _mm256_and_si256(GreaterOrEqual16i(IntegralSum16i(a[10], a[11], a[14], a[15]), central), K16_0001));

            mask = K16_FF00;
            Load<13>(a, feature, offset);
            mask = _mm256_or_si256(mask, _mm256_and_si256(GreaterOrEqual16i(IntegralSum16i(a[9], a[10], a[13], a[14]), central), K16_0004));
            Load<12>(a, feature, offset);
            Load<8>(a, feature, offset);
            mask = _mm256_or_si256(mask, _mm256_and_si256(GreaterOrEqual16i(IntegralSum16i(a[8], a[9], a[12], a[13]), central), K16_0002));
            mask = _mm256_or_si256(mask, _mm256_and_si256(GreaterOrEqual16i(IntegralSum16i(a[4], a[5], a[8], a[9]), central), K16_0001));
            mask = _mm256_shuffle_epi8(K8_SHUFFLE_BITS, mask);
        }

        SIMD_INLINE __m256i LeafMask(const HidLbpFeature<uint16_t> & feature, ptrdiff_t offset, const int * subset)
        {
            __m256i index, shuffle, mask;
            Calculate(feature, offset, index, shuffle, mask);

            __m256i _subset = _mm256_loadu_si256((__m256i*)subset);
            __m256i subset0 = _mm256_permute4x64_epi64(_subset, 0x44);
            __m256i subset1 = _mm256_permute4x64_epi64(_subset, 0xEE);

            __m256i value0 = _mm256_and_si256(_mm256_shuffle_epi8(subset0, shuffle), mask);
            __m256i value1 = _mm256_and_si256(_mm256_shuffle_epi8(subset1, shuffle), mask);
            __m256i value = _mm256_blendv_epi8(value0, value1, index);

            return _mm256_andnot_si256(_mm256_cmpeq_epi16(value, _mm256_setzero_si256()), Simd::Avx2::K_INV_ZERO);
        }

        void Detect(const HidLbpCascade<int, uint16_t> & hid, size_t offset, __m256i & result)
        {
            typedef HidLbpCascade<int, uint16_t> Hid;

            size_t subsetSize = (hid.ncategories + 31) / 32;
            const int * subsets = hid.subsets.data();
            const Hid::Leave * leaves = hid.leaves.data();
            const Hid::Node * nodes = hid.nodes.data();
            const Hid::Stage * stages = hid.stages.data();
            int nodeOffset = 0, leafOffset = 0;
            for (int i_stage = 0, n_stages = (int)hid.stages.size(); i_stage < n_stages; i_stage++)
            {
                const Hid::Stage & stage = stages[i_stage];
                __m256i sum = _mm256_setzero_si256();
                for (int i_tree = 0, n_trees = stage.ntrees; i_tree < n_trees; i_tree++)
                {
                    const Hid::Feature & feature = hid.features[nodes[nodeOffset].featureIdx];
                    const int * subset = subsets + nodeOffset*subsetSize;
                    __m256i mask = LeafMask(feature, offset, subset);
                    sum = _mm256_add_epi16(sum, _mm256_blendv_epi8(_mm256_set1_epi16(leaves[leafOffset + 1]), _mm256_set1_epi16(leaves[leafOffset + 0]), mask));
                    nodeOffset++;
                    leafOffset += 2;
                }
                result = _mm256_andnot_si256(_mm256_cmpgt_epi16(_mm256_set1_epi16(stage.threshold), sum), result);
                int resultCount = ResultCount(result);
                if (resultCount == 0)
                    return;
                else if (resultCount == 1)
                {
                    uint16_t SIMD_ALIGNED(32) _result[HA];
                    _mm256_store_si256((__m256i*)_result, result);
                    for (int i = 0; i < HA; ++i)
                    {
                        if (_result[i])
                        {
                            _result[i] = Base::Detect(hid, offset + i, i_stage + 1) > 0 ? 1 : 0;
                            break;
                        }
                    }
                    result = _mm256_load_si256((__m256i*)_result);
                    return;
                }
            }
        }

        void DetectionLbpDetect16ip(const HidLbpCascade<int, uint16_t> & hid, const Image & mask, const Rect & rect, Image & dst)
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
                    __m256i result = _mm256_loadu_si256((__m256i*)(buffer.m + col));
                    if (_mm256_testz_si256(result, K16_0001))
                        continue;
                    Detect(hid, offset + col, result);
                    _mm256_storeu_si256((__m256i*)(buffer.d + col), result);
                }
                if (evenWidth > alignedWidth + 2)
                {
                    col = evenWidth - HA;
                    __m256i result = _mm256_loadu_si256((__m256i*)(buffer.m + col));
                    if (!_mm256_testz_si256(result, K16_0001))
                    {
                        Detect(hid, offset + col, result);
                        _mm256_storeu_si256((__m256i*)(buffer.d + col), result);
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
            const HidLbpCascade<int, uint16_t> & hid = *(HidLbpCascade<int, uint16_t>*)_hid;
            return DetectionLbpDetect16ip(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }

        void DetectionLbpDetect16ii(const HidLbpCascade<int, uint16_t> & hid, const Image & mask, const Rect & rect, Image & dst)
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
                    __m256i result = _mm256_and_si256(_mm256_loadu_si256((__m256i*)(m + col)), K16_0001);
                    if (_mm256_testz_si256(result, K16_0001))
                        continue;
                    Detect(hid, offset + col / 2, result);
                    _mm256_storeu_si256((__m256i*)(d + col), result);
                }
                if (evenWidth > alignedWidth + 2)
                {
                    col = evenWidth - A;
                    __m256i result = _mm256_and_si256(_mm256_loadu_si256((__m256i*)(m + col)), K16_0001);
                    if (!_mm256_testz_si256(result, K16_0001))
                    {
                        Detect(hid, offset + col / 2, result);
                        _mm256_storeu_si256((__m256i*)(d + col), result);
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
            const HidLbpCascade<int, uint16_t> & hid = *(HidLbpCascade<int, uint16_t>*)_hid;
            return DetectionLbpDetect16ii(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }
    }
#endif// SIMD_AVX2_ENABLE
}
