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
#include "Simd/SimdDetection.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        using namespace Simd::Detection;

        SIMD_INLINE void UnpackMask16i(const uint8_t * src, uint16_t * dst, const __m128i & mask)
        {
            __m128i s = _mm_and_si128(mask, _mm_loadu_si128((__m128i*)src));
            _mm_storeu_si128((__m128i*)dst + 0, _mm_unpacklo_epi8(s, _mm_setzero_si128()));
            _mm_storeu_si128((__m128i*)dst + 1, _mm_unpackhi_epi8(s, _mm_setzero_si128()));
        }

        SIMD_INLINE void UnpackMask16i(const uint8_t * src, size_t size, uint16_t * dst, const __m128i & mask)
        {
            size_t alignedSize = Simd::AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                UnpackMask16i(src + i, dst + i, mask);
            if (size != alignedSize)
                UnpackMask16i(src + size - A, dst + size - A, mask);
        }

        SIMD_INLINE void UnpackMask32i(const uint8_t * src, uint32_t * dst, const __m128i & mask)
        {
            __m128i s = _mm_and_si128(mask, _mm_loadu_si128((__m128i*)src));
            __m128i lo = _mm_unpacklo_epi8(s, _mm_setzero_si128());
            _mm_storeu_si128((__m128i*)dst + 0, _mm_unpacklo_epi16(lo, _mm_setzero_si128()));
            _mm_storeu_si128((__m128i*)dst + 1, _mm_unpackhi_epi16(lo, _mm_setzero_si128()));
            __m128i hi = _mm_unpackhi_epi8(s, _mm_setzero_si128());
            _mm_storeu_si128((__m128i*)dst + 2, _mm_unpacklo_epi16(hi, _mm_setzero_si128()));
            _mm_storeu_si128((__m128i*)dst + 3, _mm_unpackhi_epi16(hi, _mm_setzero_si128()));
        }

        SIMD_INLINE void UnpackMask32i(const uint8_t * src, size_t size, uint32_t * dst, const __m128i & mask)
        {
            size_t alignedSize = Simd::AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                UnpackMask32i(src + i, dst + i, mask);
            if (size != alignedSize)
                UnpackMask32i(src + size - A, dst + size - A, mask);
        }

        SIMD_INLINE void PackResult16i(const uint16_t * src, uint8_t * dst)
        {
            __m128i lo = _mm_loadu_si128((__m128i*)src + 0);
            __m128i hi = _mm_loadu_si128((__m128i*)src + 1);
            _mm_storeu_si128((__m128i*)dst, _mm_packus_epi16(lo, hi));
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
            __m128i lo = _mm_packus_epi32(_mm_loadu_si128((__m128i*)src + 0), _mm_loadu_si128((__m128i*)src + 1));
            __m128i hi = _mm_packus_epi32(_mm_loadu_si128((__m128i*)src + 2), _mm_loadu_si128((__m128i*)src + 3));
            _mm_storeu_si128((__m128i*)dst, _mm_packus_epi16(lo, hi));
        }

        SIMD_INLINE void PackResult32i(const uint32_t * src, size_t size, uint8_t * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                PackResult32i(src + i, dst + i);
            if (size != alignedSize)
                PackResult32i(src + size - A, dst + size - A);
        }

        SIMD_INLINE int ResultCount(__m128i result)
        {
            uint32_t SIMD_ALIGNED(16) buffer[4];
            _mm_store_si128((__m128i*)buffer, _mm_sad_epu8(result, _mm_setzero_si128()));
            return buffer[0] + buffer[2];
        }

        SIMD_INLINE __m128 ValidSqrt(__m128 value)
        {
            return _mm_blendv_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(value), _mm_cmpgt_ps(value, _mm_set1_ps(0.0f)));
        }

        SIMD_INLINE __m128i Sum32ip(uint32_t * const ptr[4], size_t offset)
        {
            __m128i s0 = _mm_loadu_si128((__m128i*)(ptr[0] + offset));
            __m128i s1 = _mm_loadu_si128((__m128i*)(ptr[1] + offset));
            __m128i s2 = _mm_loadu_si128((__m128i*)(ptr[2] + offset));
            __m128i s3 = _mm_loadu_si128((__m128i*)(ptr[3] + offset));
            return _mm_sub_epi32(_mm_sub_epi32(s0, s1), _mm_sub_epi32(s2, s3));
        }

        SIMD_INLINE __m128i Sum32ii(uint32_t * const ptr[4], size_t offset)
        {
            __m128i lo = Sum32ip(ptr, offset + 0);
            __m128i hi = Sum32ip(ptr, offset + 4);
            return _mm_or_si128(_mm_srli_si128(_mm_shuffle_epi32(lo, 0x80), 8), _mm_slli_si128(_mm_shuffle_epi32(hi, 0x08), 8));
        }

        SIMD_INLINE __m128 Norm32fp(const HidHaarCascade & hid, size_t offset)
        {
            __m128 area = _mm_set1_ps(hid.windowArea);
            __m128 sum = _mm_cvtepi32_ps(Sum32ip(hid.p, offset));
            __m128 sqsum = _mm_cvtepi32_ps(Sum32ip(hid.pq, offset));
            return ValidSqrt(_mm_sub_ps(_mm_mul_ps(sqsum, area), _mm_mul_ps(sum, sum)));
        }

        SIMD_INLINE __m128 Norm32fi(const HidHaarCascade & hid, size_t offset)
        {
            __m128 area = _mm_set1_ps(hid.windowArea);
            __m128 sum = _mm_cvtepi32_ps(Sum32ii(hid.p, offset));
            __m128 sqsum = _mm_cvtepi32_ps(Sum32ii(hid.pq, offset));
            return ValidSqrt(_mm_sub_ps(_mm_mul_ps(sqsum, area), _mm_mul_ps(sum, sum)));
        }

        SIMD_INLINE __m128 WeightedSum32f(const WeightedRect & rect, size_t offset)
        {
            __m128i s0 = _mm_loadu_si128((__m128i*)(rect.p0 + offset));
            __m128i s1 = _mm_loadu_si128((__m128i*)(rect.p1 + offset));
            __m128i s2 = _mm_loadu_si128((__m128i*)(rect.p2 + offset));
            __m128i s3 = _mm_loadu_si128((__m128i*)(rect.p3 + offset));
            __m128i sum = _mm_sub_epi32(_mm_sub_epi32(s0, s1), _mm_sub_epi32(s2, s3));
            return _mm_mul_ps(_mm_cvtepi32_ps(sum), _mm_set1_ps(rect.weight));
        }

        SIMD_INLINE void StageSum32f(const float * leaves, float threshold, const __m128 & sum, const __m128 & norm, __m128 & stageSum)
        {
            __m128 mask = _mm_cmplt_ps(sum, _mm_mul_ps(_mm_set1_ps(threshold), norm));
            stageSum = _mm_add_ps(stageSum, _mm_blendv_ps(_mm_set1_ps(leaves[1]), _mm_set1_ps(leaves[0]), mask));
        }

        void Detect32f(const HidHaarCascade & hid, size_t offset, const __m128 & norm, __m128i & result)
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
                __m128 stageSum = _mm_setzero_ps();
                if (stage.hasThree)
                {
                    for (; node < end; ++node, leaves += 2)
                    {
                        const Hid::Feature & feature = hid.features[node->featureIdx];
                        __m128 sum = _mm_add_ps(WeightedSum32f(feature.rect[0], offset), WeightedSum32f(feature.rect[1], offset));
                        if (feature.rect[2].p0)
                            sum = _mm_add_ps(sum, WeightedSum32f(feature.rect[2], offset));
                        StageSum32f(leaves, node->threshold, sum, norm, stageSum);
                    }
                }
                else
                {
                    for (; node < end; ++node, leaves += 2)
                    {
                        const Hid::Feature & feature = hid.features[node->featureIdx];
                        __m128 sum = _mm_add_ps(WeightedSum32f(feature.rect[0], offset), WeightedSum32f(feature.rect[1], offset));
                        StageSum32f(leaves, node->threshold, sum, norm, stageSum);
                    }
                }
                result = _mm_andnot_si128(_mm_castps_si128(_mm_cmpgt_ps(_mm_set1_ps(stage.threshold), stageSum)), result);
                int resultCount = ResultCount(result);
                if (resultCount == 0)
                {
                    return;
                }
                else if (resultCount == 1)
                {
                    uint32_t SIMD_ALIGNED(16) _result[4];
                    float SIMD_ALIGNED(16) _norm[4];
                    _mm_store_si128((__m128i*)_result, result);
                    _mm_store_ps(_norm, norm);
                    for (int j = 0; j < 4; ++j)
                    {
                        if (_result[j])
                        {
                            _result[j] = Base::Detect32f(hid, offset + j, i + 1, _norm[j]) > 0 ? 1 : 0;
                            break;
                        }
                    }
                    result = _mm_load_si128((__m128i*)_result);
                    return;
                }
            }
        }

        void DetectionHaarDetect32fp(const HidHaarCascade & hid, const Image & mask, const Rect & rect, Image & dst)
        {
            size_t width = rect.Width();
            size_t alignedWidth = Simd::AlignLo(width, 4);
            size_t evenWidth = Simd::AlignLo(width, 2);

            Buffer<uint32_t> buffer(width);
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += 1)
            {
                size_t col = 0;
                size_t p_offset = row * hid.sum.stride / sizeof(uint32_t) + rect.left;
                size_t pq_offset = row * hid.sqsum.stride / sizeof(uint32_t) + rect.left;

                UnpackMask32i(mask.data + row*mask.stride + rect.left, width, buffer.m, K8_01);
                memset(buffer.d, 0, width * sizeof(uint32_t));
                for (; col < alignedWidth; col += 4)
                {
                    __m128i result = _mm_loadu_si128((__m128i*)(buffer.m + col));
                    if (_mm_testz_si128(result, K32_00000001))
                        continue;
                    __m128 norm = Norm32fp(hid, pq_offset + col);
                    Detect32f(hid, p_offset + col, norm, result);
                    _mm_storeu_si128((__m128i*)(buffer.d + col), result);
                }
                if (evenWidth > alignedWidth + 2)
                {
                    col = evenWidth - 4;
                    __m128i result = _mm_loadu_si128((__m128i*)(buffer.m + col));
                    if (!_mm_testz_si128(result, K32_00000001))
                    {
                        __m128 norm = Norm32fp(hid, pq_offset + col);
                        Detect32f(hid, p_offset + col, norm, result);
                        _mm_storeu_si128((__m128i*)(buffer.d + col), result);
                    }
                    col += 4;
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
                    __m128i result = _mm_loadu_si128((__m128i*)(buffer.m + col));
                    if (_mm_testz_si128(result, K32_00000001))
                        continue;
                    __m128 norm = Norm32fi(hid, pq_offset + col);
                    Detect32f(hid, p_offset + col / 2, norm, result);
                    _mm_storeu_si128((__m128i*)(buffer.d + col), result);
                }
                if (evenWidth > alignedWidth)
                {
                    col = evenWidth - HA;
                    __m128i result = _mm_loadu_si128((__m128i*)(buffer.m + col));
                    if (!_mm_testz_si128(result, K32_00000001))
                    {
                        __m128 norm = Norm32fi(hid, pq_offset + col);
                        Detect32f(hid, p_offset + col / 2, norm, result);
                        _mm_storeu_si128((__m128i*)(buffer.d + col), result);
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

        const __m128i K8_SHUFFLE_BITS = SIMD_MM_SETR_EPI8(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);

        SIMD_INLINE __m128i IntegralSum32i(const __m128i & s0, const __m128i & s1, const __m128i & s2, const __m128i & s3)
        {
            return _mm_sub_epi32(_mm_sub_epi32(s0, s1), _mm_sub_epi32(s2, s3));
        }

        SIMD_INLINE __m128i GreaterOrEqual32i(__m128i a, __m128i b)
        {
            return _mm_cmpeq_epi32(_mm_max_epu32(a, b), a);
        }

        template<int i> SIMD_INLINE void Load(__m128i a[16], const HidLbpFeature<uint32_t> & feature, ptrdiff_t offset)
        {
            a[i] = _mm_loadu_si128((__m128i*)(feature.p[i] + offset));
        }

        SIMD_INLINE void Calculate(const HidLbpFeature<uint32_t> & feature, ptrdiff_t offset, __m128i & index, __m128i & shuffle, __m128i & mask)
        {
            __m128i a[16];
            Load<5>(a, feature, offset);
            Load<6>(a, feature, offset);
            Load<9>(a, feature, offset);
            Load<10>(a, feature, offset);
            __m128i central = IntegralSum32i(a[5], a[6], a[9], a[10]);

            Load<0>(a, feature, offset);
            Load<1>(a, feature, offset);
            Load<4>(a, feature, offset);
            index = GreaterOrEqual32i(IntegralSum32i(a[0], a[1], a[4], a[5]), central);

            shuffle = K32_FFFFFF00;
            Load<2>(a, feature, offset);
            shuffle = _mm_or_si128(shuffle, _mm_and_si128(GreaterOrEqual32i(IntegralSum32i(a[1], a[2], a[5], a[6]), central), K32_00000008));
            Load<3>(a, feature, offset);
            Load<7>(a, feature, offset);
            shuffle = _mm_or_si128(shuffle, _mm_and_si128(GreaterOrEqual32i(IntegralSum32i(a[2], a[3], a[6], a[7]), central), K32_00000004));
            Load<11>(a, feature, offset);
            shuffle = _mm_or_si128(shuffle, _mm_and_si128(GreaterOrEqual32i(IntegralSum32i(a[6], a[7], a[10], a[11]), central), K32_00000002));
            Load<14>(a, feature, offset);
            Load<15>(a, feature, offset);
            shuffle = _mm_or_si128(shuffle, _mm_and_si128(GreaterOrEqual32i(IntegralSum32i(a[10], a[11], a[14], a[15]), central), K32_00000001));

            mask = K32_FFFFFF00;
            Load<13>(a, feature, offset);
            mask = _mm_or_si128(mask, _mm_and_si128(GreaterOrEqual32i(IntegralSum32i(a[9], a[10], a[13], a[14]), central), K32_00000004));
            Load<12>(a, feature, offset);
            Load<8>(a, feature, offset);
            mask = _mm_or_si128(mask, _mm_and_si128(GreaterOrEqual32i(IntegralSum32i(a[8], a[9], a[12], a[13]), central), K32_00000002));
            mask = _mm_or_si128(mask, _mm_and_si128(GreaterOrEqual32i(IntegralSum32i(a[4], a[5], a[8], a[9]), central), K32_00000001));
            mask = _mm_shuffle_epi8(K8_SHUFFLE_BITS, mask);
        }

        SIMD_INLINE __m128i LeafMask(const HidLbpFeature<uint32_t> & feature, ptrdiff_t offset, const int * subset)
        {
            __m128i index, shuffle, mask;
            Calculate(feature, offset, index, shuffle, mask);

            __m128i subset0 = _mm_loadu_si128((__m128i*)subset + 0);
            __m128i subset1 = _mm_loadu_si128((__m128i*)subset + 1);

            __m128i value0 = _mm_and_si128(_mm_shuffle_epi8(subset0, shuffle), mask);
            __m128i value1 = _mm_and_si128(_mm_shuffle_epi8(subset1, shuffle), mask);
            __m128i value = _mm_blendv_epi8(value0, value1, index);

            return _mm_andnot_si128(_mm_cmpeq_epi32(value, _mm_setzero_si128()), Simd::Sse2::K_INV_ZERO);
        }

        void Detect(const HidLbpCascade<float, uint32_t> & hid, size_t offset, __m128i & result)
        {
            typedef HidLbpCascade<float, uint32_t> Hid;

            size_t subsetSize = (hid.ncategories + 31) / 32;
            const int * subsets = hid.subsets.data();
            const Hid::Leave * leaves = hid.leaves.data();
            const Hid::Node * nodes = hid.nodes.data();
            const Hid::Stage * stages = hid.stages.data();
            int nodeOffset = 0, leafOffset = 0;
            for (int i_stage = 0, n_stages = (int)hid.stages.size(); i_stage < n_stages; i_stage++)
            {
                const Hid::Stage & stage = stages[i_stage];
                __m128 sum = _mm_setzero_ps();
                for (int i_tree = 0, n_trees = stage.ntrees; i_tree < n_trees; i_tree++)
                {
                    const Hid::Feature & feature = hid.features[nodes[nodeOffset].featureIdx];
                    const int * subset = subsets + nodeOffset*subsetSize;
                    __m128i mask = LeafMask(feature, offset, subset);
                    sum = _mm_add_ps(sum, _mm_blendv_ps(_mm_set1_ps(leaves[leafOffset + 1]),
                        _mm_set1_ps(leaves[leafOffset + 0]), _mm_castsi128_ps(mask)));
                    nodeOffset++;
                    leafOffset += 2;
                }
                result = _mm_andnot_si128(_mm_castps_si128(_mm_cmpgt_ps(_mm_set1_ps(stage.threshold), sum)), result);
                int resultCount = ResultCount(result);
                if (resultCount == 0)
                    return;
                else if (resultCount == 1)
                {
                    uint32_t SIMD_ALIGNED(16) _result[4];
                    _mm_store_si128((__m128i*)_result, result);
                    for (int i = 0; i < 4; ++i)
                    {
                        if (_result[i])
                        {
                            _result[i] = Base::Detect(hid, offset + i, i_stage + 1) > 0 ? 1 : 0;
                            break;
                        }
                    }
                    result = _mm_load_si128((__m128i*)_result);
                    return;
                }
            }
        }

        void DetectionLbpDetect32fp(const HidLbpCascade<float, uint32_t> & hid, const Image & mask, const Rect & rect, Image & dst)
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
                    __m128i result = _mm_loadu_si128((__m128i*)(buffer.m + col));
                    if (_mm_testz_si128(result, K32_00000001))
                        continue;
                    Detect(hid, offset + col, result);
                    _mm_storeu_si128((__m128i*)(buffer.d + col), result);
                }
                if (evenWidth > alignedWidth + 2)
                {
                    col = evenWidth - 4;
                    __m128i result = _mm_loadu_si128((__m128i*)(buffer.m + col));
                    if (!_mm_testz_si128(result, K32_00000001))
                    {
                        Detect(hid, offset + col, result);
                        _mm_storeu_si128((__m128i*)(buffer.d + col), result);
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
                    __m128i result = _mm_loadu_si128((__m128i*)(buffer.m + col));
                    if (_mm_testz_si128(result, K32_00000001))
                        continue;
                    Detect(hid, offset + col / 2, result);
                    _mm_storeu_si128((__m128i*)(buffer.d + col), result);
                }
                if (evenWidth > alignedWidth)
                {
                    col = evenWidth - HA;
                    __m128i result = _mm_loadu_si128((__m128i*)(buffer.m + col));
                    if (!_mm_testz_si128(result, K32_00000001))
                    {
                        Detect(hid, offset + col / 2, result);
                        _mm_storeu_si128((__m128i*)(buffer.d + col), result);
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

        SIMD_INLINE __m128i IntegralSum16i(const __m128i & s0, const __m128i & s1, const __m128i & s2, const __m128i & s3)
        {
            return _mm_sub_epi16(_mm_sub_epi16(s0, s1), _mm_sub_epi16(s2, s3));
        }

        SIMD_INLINE __m128i GreaterOrEqual16i(__m128i a, __m128i b)
        {
            return _mm_cmpeq_epi16(_mm_max_epu16(a, b), a);
        }

        template<int i> SIMD_INLINE void Load(__m128i a[16], const HidLbpFeature<uint16_t> & feature, ptrdiff_t offset)
        {
            a[i] = _mm_loadu_si128((__m128i*)(feature.p[i] + offset));
        }

        SIMD_INLINE void Calculate(const HidLbpFeature<uint16_t> & feature, ptrdiff_t offset, __m128i & index, __m128i & shuffle, __m128i & mask)
        {
            __m128i a[16];
            Load<5>(a, feature, offset);
            Load<6>(a, feature, offset);
            Load<9>(a, feature, offset);
            Load<10>(a, feature, offset);
            __m128i central = IntegralSum16i(a[5], a[6], a[9], a[10]);

            Load<0>(a, feature, offset);
            Load<1>(a, feature, offset);
            Load<4>(a, feature, offset);
            index = GreaterOrEqual16i(IntegralSum16i(a[0], a[1], a[4], a[5]), central);

            shuffle = K16_FF00;
            Load<2>(a, feature, offset);
            shuffle = _mm_or_si128(shuffle, _mm_and_si128(GreaterOrEqual16i(IntegralSum16i(a[1], a[2], a[5], a[6]), central), K16_0008));
            Load<3>(a, feature, offset);
            Load<7>(a, feature, offset);
            shuffle = _mm_or_si128(shuffle, _mm_and_si128(GreaterOrEqual16i(IntegralSum16i(a[2], a[3], a[6], a[7]), central), K16_0004));
            Load<11>(a, feature, offset);
            shuffle = _mm_or_si128(shuffle, _mm_and_si128(GreaterOrEqual16i(IntegralSum16i(a[6], a[7], a[10], a[11]), central), K16_0002));
            Load<14>(a, feature, offset);
            Load<15>(a, feature, offset);
            shuffle = _mm_or_si128(shuffle, _mm_and_si128(GreaterOrEqual16i(IntegralSum16i(a[10], a[11], a[14], a[15]), central), K16_0001));

            mask = K16_FF00;
            Load<13>(a, feature, offset);
            mask = _mm_or_si128(mask, _mm_and_si128(GreaterOrEqual16i(IntegralSum16i(a[9], a[10], a[13], a[14]), central), K16_0004));
            Load<12>(a, feature, offset);
            Load<8>(a, feature, offset);
            mask = _mm_or_si128(mask, _mm_and_si128(GreaterOrEqual16i(IntegralSum16i(a[8], a[9], a[12], a[13]), central), K16_0002));
            mask = _mm_or_si128(mask, _mm_and_si128(GreaterOrEqual16i(IntegralSum16i(a[4], a[5], a[8], a[9]), central), K16_0001));
            mask = _mm_shuffle_epi8(K8_SHUFFLE_BITS, mask);
        }

        SIMD_INLINE __m128i LeafMask(const HidLbpFeature<uint16_t> & feature, ptrdiff_t offset, const int * subset)
        {
            __m128i index, shuffle, mask;
            Calculate(feature, offset, index, shuffle, mask);

            __m128i subset0 = _mm_loadu_si128((__m128i*)subset + 0);
            __m128i subset1 = _mm_loadu_si128((__m128i*)subset + 1);

            __m128i value0 = _mm_and_si128(_mm_shuffle_epi8(subset0, shuffle), mask);
            __m128i value1 = _mm_and_si128(_mm_shuffle_epi8(subset1, shuffle), mask);
            __m128i value = _mm_blendv_epi8(value0, value1, index);

            return _mm_andnot_si128(_mm_cmpeq_epi16(value, _mm_setzero_si128()), Simd::Sse2::K_INV_ZERO);
        }

        void Detect(const HidLbpCascade<int, uint16_t> & hid, size_t offset, __m128i & result)
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
                __m128i sum = _mm_setzero_si128();
                for (int i_tree = 0, n_trees = stage.ntrees; i_tree < n_trees; i_tree++)
                {
                    const Hid::Feature & feature = hid.features[nodes[nodeOffset].featureIdx];
                    const int * subset = subsets + nodeOffset*subsetSize;
                    __m128i mask = LeafMask(feature, offset, subset);
                    sum = _mm_add_epi16(sum, _mm_blendv_epi8(_mm_set1_epi16(leaves[leafOffset + 1]),
                        _mm_set1_epi16(leaves[leafOffset + 0]), mask));
                    nodeOffset++;
                    leafOffset += 2;
                }
                result = _mm_andnot_si128(_mm_cmpgt_epi16(_mm_set1_epi16(stage.threshold), sum), result);
                int resultCount = ResultCount(result);
                if (resultCount == 0)
                    return;
                else if (resultCount == 1)
                {
                    uint16_t SIMD_ALIGNED(16) _result[HA];
                    _mm_store_si128((__m128i*)_result, result);
                    for (int i = 0; i < HA; ++i)
                    {
                        if (_result[i])
                        {
                            _result[i] = Base::Detect(hid, offset + i, i_stage + 1) > 0 ? 1 : 0;
                            break;
                        }
                    }
                    result = _mm_load_si128((__m128i*)_result);
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
                    __m128i result = _mm_loadu_si128((__m128i*)(buffer.m + col));
                    if (_mm_testz_si128(result, K16_0001))
                        continue;
                    Detect(hid, offset + col, result);
                    _mm_storeu_si128((__m128i*)(buffer.d + col), result);
                }
                if (evenWidth > alignedWidth + 2)
                {
                    col = evenWidth - HA;
                    __m128i result = _mm_loadu_si128((__m128i*)(buffer.m + col));
                    if (!_mm_testz_si128(result, K16_0001))
                    {
                        Detect(hid, offset + col, result);
                        _mm_storeu_si128((__m128i*)(buffer.d + col), result);
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
                    __m128i result = _mm_and_si128(_mm_loadu_si128((__m128i*)(m + col)), K16_0001);
                    if (_mm_testz_si128(result, K16_0001))
                        continue;
                    Detect(hid, offset + col / 2, result);
                    _mm_storeu_si128((__m128i*)(d + col), result);
                }
                if (evenWidth > alignedWidth + 2)
                {
                    col = evenWidth - A;
                    __m128i result = _mm_and_si128(_mm_loadu_si128((__m128i*)(m + col)), K16_0001);
                    if (!_mm_testz_si128(result, K16_0001))
                    {
                        Detect(hid, offset + col / 2, result);
                        _mm_storeu_si128((__m128i*)(d + col), result);
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
#endif//SIMD_SSE41_ENABLE
}
