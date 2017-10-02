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
#include "Simd/SimdExtract.h"
#include "Simd/SimdDetection.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        using namespace Simd::Detection;

        template <bool masked> SIMD_INLINE void UnpackMask16i(const uint8_t * src, uint16_t * dst, const __m512i & mask, __mmask64 tail = -1)
        {
            __m512i src0 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, _mm512_and_si512(mask, (Load<false, masked>(src, tail))));
            Store<false, masked>(dst + 0 * HA, UnpackU8<0>(src0), __mmask32(tail >> 00));
            Store<false, masked>(dst + 1 * HA, UnpackU8<1>(src0), __mmask32(tail >> 32));
        }

        SIMD_INLINE void UnpackMask16i(const uint8_t * src, size_t size, uint16_t * dst, const __m512i & mask)
        {
            size_t alignedSize = Simd::AlignLo(size, A);
            __mmask64 tailMask = TailMask64(size - alignedSize);
            size_t i = 0;
            for (; i < alignedSize; i += A)
                UnpackMask16i<false>(src + i, dst + i, mask);
            if (i < size)
                UnpackMask16i<true>(src + i, dst + i, mask, tailMask);
        }

        template <bool masked> SIMD_INLINE void UnpackMask32i(const uint8_t * src, uint32_t * dst, const __m512i & mask, __mmask64 tail = -1)
        {
            __m512i _src = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_and_si512(mask, (Load<false, masked>(src, tail))));
            __m512i src0 = UnpackU8<0>(_src);
            Store<false, masked>(dst + 0 * F, UnpackU8<0>(src0), __mmask16(tail >> 00));
            Store<false, masked>(dst + 1 * F, UnpackU8<1>(src0), __mmask16(tail >> 16));
            __m512i src1 = UnpackU8<1>(_src);
            Store<false, masked>(dst + 2 * F, UnpackU8<0>(src1), __mmask16(tail >> 32));
            Store<false, masked>(dst + 3 * F, UnpackU8<1>(src1), __mmask16(tail >> 48));
        }

        SIMD_INLINE void UnpackMask32i(const uint8_t * src, size_t size, uint32_t * dst, const __m512i & mask)
        {
            size_t alignedSize = Simd::AlignLo(size, A);
            __mmask64 tailMask = TailMask64(size - alignedSize);
            size_t i = 0;
            for (; i < alignedSize; i += A)
                UnpackMask32i<false>(src + i, dst + i, mask);
            if (i < size)
                UnpackMask32i<true>(src + i, dst + i, mask, tailMask);
        }

        template <bool masked> SIMD_INLINE void PackResult16i(const uint16_t * src, uint8_t * dst, __mmask64 tail = -1)
        {
            __m512i src0 = Load<false, masked>(src + 00, __mmask32(tail >> 00));
            __m512i src1 = Load<false, masked>(src + HA, __mmask32(tail >> 32));
            Store<false, masked>(dst, _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi16(src0, src1)), tail);
        }

        SIMD_INLINE void PackResult16i(const uint16_t * src, size_t size, uint8_t * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, A);
            __mmask64 tailMask = TailMask64(size - alignedSize);
            size_t i = 0;
            for (; i < alignedSize; i += A)
                PackResult16i<false>(src + i, dst + i);
            if (i < size)
                PackResult16i<true>(src + i, dst + i, tailMask);
        }

        template <bool masked> SIMD_INLINE void PackResult32i(const uint32_t * src, uint8_t * dst, __mmask64 tail = -1)
        {
            __m512i src0 = Load<false, masked>(src + 0 * F, __mmask16(tail >> 00));
            __m512i src1 = Load<false, masked>(src + 1 * F, __mmask16(tail >> 16));
            __m512i src2 = Load<false, masked>(src + 2 * F, __mmask16(tail >> 32));
            __m512i src3 = Load<false, masked>(src + 3 * F, __mmask16(tail >> 48));
            Store<false, masked>(dst, _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(src0, src1), _mm512_packs_epi32(src2, src3))), tail);
        }

        SIMD_INLINE void PackResult32i(const uint32_t * src, size_t size, uint8_t * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, A);
            __mmask64 tailMask = TailMask64(size - alignedSize);
            size_t i = 0;
            for (; i < alignedSize; i += A)
                PackResult32i<false>(src + i, dst + i);
            if (i < size)
                PackResult32i<true>(src + i, dst + i, tailMask);
        }

        SIMD_INLINE int ResultCount(__m512i result)
        {
            return _mm_popcnt_u32(_mm512_test_epi16_mask(result, result));
        }

        SIMD_INLINE __m512 ValidSqrt(__m512 value)
        {
            __mmask16 mask = _mm512_cmp_ps_mask(value, _mm512_set1_ps(0.0f), _CMP_GT_OQ);
            __m512 valid = _mm512_mask_blend_ps(mask, _mm512_set1_ps(1.0f), value);
#if 0
            __m512 rsqrt = _mm512_rsqrt14_ps(valid);
            return _mm512_mul_ps(rsqrt, value);
#else
            return _mm512_sqrt_ps(valid);
#endif
        }

        template <bool masked> SIMD_INLINE __m512i Sum32ip(uint32_t * const ptr[4], size_t offset, __mmask16 tail = -1)
        {
            __m512i s0 = Load<false, masked>(ptr[0] + offset, tail);
            __m512i s1 = Load<false, masked>(ptr[1] + offset, tail);
            __m512i s2 = Load<false, masked>(ptr[2] + offset, tail);
            __m512i s3 = Load<false, masked>(ptr[3] + offset, tail);
            return _mm512_sub_epi32(_mm512_sub_epi32(s0, s1), _mm512_sub_epi32(s2, s3));
        }

        const __m512i K32_PERMUTE_EVEN = SIMD_MM512_SETR_EPI32(0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E);

        template <bool masked> SIMD_INLINE __m512i Sum32ii(uint32_t * const ptr[4], size_t offset, const __mmask16 * tails)
        {
            __m512i lo = Sum32ip<masked>(ptr, offset + 0, tails[0]);
            __m512i hi = Sum32ip<masked>(ptr, offset + F, tails[1]);
            return _mm512_permutex2var_epi32(lo, K32_PERMUTE_EVEN, hi);
        }

        template <bool masked> SIMD_INLINE __m512 Norm32fp(const HidHaarCascade & hid, size_t offset, __mmask16 tail = -1)
        {
            __m512 area = _mm512_set1_ps(hid.windowArea);
            __m512 sum = _mm512_cvtepi32_ps(Sum32ip<masked>(hid.p, offset, tail));
            __m512 sqsum = _mm512_cvtepi32_ps(Sum32ip<masked>(hid.pq, offset, tail));
            return ValidSqrt(_mm512_sub_ps(_mm512_mul_ps(sqsum, area), _mm512_mul_ps(sum, sum)));
        }

        template <bool masked> SIMD_INLINE __m512 Norm32fi(const HidHaarCascade & hid, size_t offset, const __mmask16 * tails)
        {
            __m512 area = _mm512_set1_ps(hid.windowArea);
            __m512 sum = _mm512_cvtepi32_ps(Sum32ii<masked>(hid.p, offset, tails));
            __m512 sqsum = _mm512_cvtepi32_ps(Sum32ii<masked>(hid.pq, offset, tails));
            return ValidSqrt(_mm512_sub_ps(_mm512_mul_ps(sqsum, area), _mm512_mul_ps(sum, sum)));
        }

        template <bool masked> SIMD_INLINE __m512 WeightedSum32f(const WeightedRect & rect, size_t offset, __mmask16 tail = -1)
        {
            __m512i s0 = Load<false, masked>(rect.p0 + offset, tail);
            __m512i s1 = Load<false, masked>(rect.p1 + offset, tail);
            __m512i s2 = Load<false, masked>(rect.p2 + offset, tail);
            __m512i s3 = Load<false, masked>(rect.p3 + offset, tail);
            __m512i sum = _mm512_sub_epi32(_mm512_sub_epi32(s0, s1), _mm512_sub_epi32(s2, s3));
            return _mm512_mul_ps(_mm512_cvtepi32_ps(sum), _mm512_set1_ps(rect.weight));
        }

        SIMD_INLINE void StageSum32f(const float * leaves, float threshold, const __m512 & sum, const __m512 & norm, __m512 & stageSum)
        {
            __mmask16 mask = _mm512_cmp_ps_mask(sum, _mm512_mul_ps(_mm512_set1_ps(threshold), norm), _CMP_GE_OQ);
            stageSum = _mm512_add_ps(stageSum, _mm512_mask_blend_ps(mask, _mm512_set1_ps(leaves[0]), _mm512_set1_ps(leaves[1])));
        }

        template <bool masked> __mmask16 Detect32f(const HidHaarCascade & hid, size_t offset, const __m512 & norm, __mmask16 result)
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
                __m512 stageSum = _mm512_setzero_ps();
                if (stage.hasThree)
                {
                    for (; node < end; ++node, leaves += 2)
                    {
                        const Hid::Feature & feature = hid.features[node->featureIdx];
                        __m512 sum = _mm512_add_ps(
                            WeightedSum32f<masked>(feature.rect[0], offset, result),
                            WeightedSum32f<masked>(feature.rect[1], offset, result));
                        if (feature.rect[2].p0)
                            sum = _mm512_add_ps(sum, WeightedSum32f<masked>(feature.rect[2], offset, result));
                        StageSum32f(leaves, node->threshold, sum, norm, stageSum);
                    }
                }
                else
                {
                    for (; node < end; ++node, leaves += 2)
                    {
                        const Hid::Feature & feature = hid.features[node->featureIdx];
                        __m512 sum = _mm512_add_ps(WeightedSum32f<masked>(feature.rect[0], offset, result),
                            WeightedSum32f<masked>(feature.rect[1], offset, result));
                        StageSum32f(leaves, node->threshold, sum, norm, stageSum);
                    }
                }
                result = result & _mm512_cmp_ps_mask(stageSum, _mm512_set1_ps(stage.threshold), _CMP_GE_OQ);
                if (!result)
                    return result;
                int resultCount = _mm_popcnt_u32(result);
                if (resultCount == 1)
                {
                    int j = _tzcnt_u32(result);
                    return Base::Detect32f(hid, offset + j, i + 1, Avx512f::Extract(norm, j)) > 0 ? result : __mmask16(0);
                }
            }
            return result;
        }

        void DetectionHaarDetect32fp(const HidHaarCascade & hid, const Image & mask, const Rect & rect, Image & dst)
        {
            size_t width = rect.Width();
            size_t alignedWidth = Simd::AlignLo(width, F);
            __mmask16 tailMask = TailMask16(width - alignedWidth);
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
                    __mmask16 result = _mm512_cmpneq_epi32_mask(Load<false>(buffer.m + col), K_ZERO);
                    if (result)
                    {
                        __m512 norm = Norm32fp<false>(hid, pq_offset + col);
                        result = Detect32f<false>(hid, p_offset + col, norm, result);
                        Store<false>(buffer.d + col, _mm512_maskz_set1_epi32(result, 1));
                    }
                }
                if (col < width)
                {
                    __mmask16 result = _mm512_cmpneq_epi32_mask((Load<false, true>(buffer.m + col, tailMask)), K_ZERO);
                    if (result)
                    {
                        __m512 norm = Norm32fp<true>(hid, pq_offset + col, tailMask);
                        result = Detect32f<true>(hid, p_offset + col, norm, result);
                        Store<false, true>(buffer.d + col, _mm512_maskz_set1_epi32(result, 1), tailMask);
                    }
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
            __mmask16 tailMasks[3];
            for (size_t c = 0; c < 2; ++c)
                tailMasks[c] = TailMask16(width - alignedWidth - F*c);
            tailMasks[2] = TailMask16((width - alignedWidth) / 2);
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
                    __mmask16 result = _mm512_cmpneq_epi32_mask(_mm512_and_si512(Load<false>(buffer.m + col), K32_0000FFFF), K_ZERO);
                    if (result)
                    {
                        __m512 norm = Norm32fi<false>(hid, pq_offset + col, tailMasks);
                        result = Detect32f<false>(hid, p_offset + col / 2, norm, result);
                        Store<false>(buffer.d + col, _mm512_maskz_set1_epi32(result, 1));
                    }
                }
                if (col < evenWidth)
                {
                    __mmask16 result = _mm512_cmpneq_epi32_mask(_mm512_and_si512((Load<false, true>((uint32_t*)buffer.m + col / 2, tailMasks[2])), K32_0000FFFF), K_ZERO);
                    if (result)
                    {
                        __m512 norm = Norm32fi<true>(hid, pq_offset + col, tailMasks);
                        result = Detect32f<true>(hid, p_offset + col / 2, norm, result);
                        Store<false, true>((uint32_t*)buffer.d + col / 2, _mm512_maskz_set1_epi32(result, 1), tailMasks[2]);
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

        const __m512i K8_SHUFFLE_BITS = SIMD_MM512_SETR_EPI8(
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);

        SIMD_INLINE __m512i IntegralSum32i(const __m512i & s0, const __m512i & s1, const __m512i & s2, const __m512i & s3)
        {
            return _mm512_sub_epi32(_mm512_sub_epi32(s0, s1), _mm512_sub_epi32(s2, s3));
        }

        template<int i, bool masked> SIMD_INLINE void Load(__m512i a[16], const HidLbpFeature<uint32_t> & feature, ptrdiff_t offset, __mmask16 tail = -1)
        {
            a[i] = Load<false, masked>(feature.p[i] + offset, tail);
        }

        template <bool masked> SIMD_INLINE void Calculate(const HidLbpFeature<uint32_t> & feature, ptrdiff_t offset, __mmask16 & index, __m512i & shuffle, __m512i & mask, __mmask16 tail = -1)
        {
            __m512i a[16];
            Load<5, masked>(a, feature, offset, tail);
            Load<6, masked>(a, feature, offset, tail);
            Load<9, masked>(a, feature, offset, tail);
            Load<10, masked>(a, feature, offset, tail);
            __m512i central = IntegralSum32i(a[5], a[6], a[9], a[10]);

            Load<0, masked>(a, feature, offset, tail);
            Load<1, masked>(a, feature, offset, tail);
            Load<4, masked>(a, feature, offset, tail);
            index = _mm512_cmpge_epu32_mask(IntegralSum32i(a[0], a[1], a[4], a[5]), central);

            shuffle = K32_FFFFFF00;
            Load<2, masked>(a, feature, offset, tail);
            shuffle = _mm512_or_si512(shuffle, _mm512_maskz_set1_epi32(_mm512_cmpge_epu32_mask(IntegralSum32i(a[1], a[2], a[5], a[6]), central), 8));
            Load<3, masked>(a, feature, offset, tail);
            Load<7, masked>(a, feature, offset, tail);
            shuffle = _mm512_or_si512(shuffle, _mm512_maskz_set1_epi32(_mm512_cmpge_epu32_mask(IntegralSum32i(a[2], a[3], a[6], a[7]), central), 4));
            Load<11, masked>(a, feature, offset, tail);
            shuffle = _mm512_or_si512(shuffle, _mm512_maskz_set1_epi32(_mm512_cmpge_epu32_mask(IntegralSum32i(a[6], a[7], a[10], a[11]), central), 2));
            Load<14, masked>(a, feature, offset, tail);
            Load<15, masked>(a, feature, offset, tail);
            shuffle = _mm512_or_si512(shuffle, _mm512_maskz_set1_epi32(_mm512_cmpge_epu32_mask(IntegralSum32i(a[10], a[11], a[14], a[15]), central), 1));

            mask = K32_FFFFFF00;
            Load<13, masked>(a, feature, offset, tail);
            mask = _mm512_or_si512(mask, _mm512_maskz_set1_epi32(_mm512_cmpge_epu32_mask(IntegralSum32i(a[9], a[10], a[13], a[14]), central), 4));
            Load<12, masked>(a, feature, offset, tail);
            Load<8, masked>(a, feature, offset, tail);
            mask = _mm512_or_si512(mask, _mm512_maskz_set1_epi32(_mm512_cmpge_epu32_mask(IntegralSum32i(a[8], a[9], a[12], a[13]), central), 2));
            mask = _mm512_or_si512(mask, _mm512_maskz_set1_epi32(_mm512_cmpge_epu32_mask(IntegralSum32i(a[4], a[5], a[8], a[9]), central), 1));
            mask = _mm512_shuffle_epi8(K8_SHUFFLE_BITS, mask);
        }

        template <bool masked> SIMD_INLINE __mmask16 LeafMask(const HidLbpFeature<uint32_t> & feature, ptrdiff_t offset, const int * subset, __mmask16 tail = -1)
        {
            __mmask16 index;
            __m512i shuffle, mask;
            Calculate<masked>(feature, offset, index, shuffle, mask, tail);

            __m256i _subset = _mm256_loadu_si256((__m256i*)subset);
            __m512i subset0 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_subset, 0));
            __m512i subset1 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_subset, 1));

            __m512i value0 = _mm512_and_si512(_mm512_shuffle_epi8(subset0, shuffle), mask);
            __m512i value1 = _mm512_and_si512(_mm512_shuffle_epi8(subset1, shuffle), mask);
            __m512i value = _mm512_mask_blend_epi32(index, value0, value1);

            return _mm512_cmpneq_epi32_mask(value, K_ZERO);
        }

        template<bool masked> __mmask16 Detect(const HidLbpCascade<float, uint32_t> & hid, size_t offset, int startStage, __mmask16 result)
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
                __m512 sum = _mm512_setzero_ps();
                for (int i_tree = 0, n_trees = stage.ntrees; i_tree < n_trees; i_tree++)
                {
                    const Hid::Feature & feature = hid.features[nodes[nodeOffset].featureIdx];
                    const int * subset = subsets + nodeOffset*subsetSize;
                    __mmask16 mask = LeafMask<masked>(feature, offset, subset, result);
                    sum = _mm512_add_ps(sum, _mm512_mask_blend_ps(mask, _mm512_set1_ps(leaves[leafOffset + 1]), _mm512_set1_ps(leaves[leafOffset + 0])));
                    nodeOffset++;
                    leafOffset += 2;
                }
                result = result & _mm512_cmp_ps_mask(sum, _mm512_set1_ps(stage.threshold), _CMP_GE_OQ);
                if (!result)
                    return result;
                int resultCount = _mm_popcnt_u32(result);
                if (resultCount == 1)
                {
                    int j = _tzcnt_u32(result);
                    return Base::Detect(hid, offset + j, i_stage + 1) > 0 ? result : __mmask16(0);
                }
            }
            return result;
        }

        void DetectionLbpDetect32fp(const HidLbpCascade<float, uint32_t> & hid, const Image & mask, const Rect & rect, Image & dst)
        {
            size_t width = rect.Width();
            size_t alignedWidth = Simd::AlignLo(width, F);
            __mmask16 tailMask = TailMask16(width - alignedWidth);
            Buffer<uint32_t> buffer(width);
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += 1)
            {
                size_t col = 0;
                size_t offset = row * hid.sum.stride / sizeof(uint32_t) + rect.left;

                UnpackMask32i(mask.data + row*mask.stride + rect.left, width, buffer.m, K8_01);
                memset(buffer.d, 0, width * sizeof(uint32_t));
                for (; col < alignedWidth; col += F)
                {
                    __mmask16 result = _mm512_cmpneq_epi32_mask(Load<false>(buffer.m + col), K_ZERO);
                    if (result)
                    {
                        result = Detect<false>(hid, offset + col, 0, result);
                        Store<false>(buffer.d + col, _mm512_maskz_set1_epi32(result, 1));
                    }
                }
                if (col < width)
                {
                    __mmask16 result = _mm512_cmpneq_epi32_mask((Load<false, true>(buffer.m + col, tailMask)), K_ZERO);
                    if (result)
                    {
                        result = Detect<true>(hid, offset + col, 0, result);
                        Store<false, true>(buffer.d + col, _mm512_maskz_set1_epi32(result, 1), tailMask);
                    }
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
            __mmask16 tailMask = TailMask16((width - alignedWidth) / 2);
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
                    __mmask16 result = _mm512_cmpneq_epi32_mask(_mm512_and_si512(Load<false>(buffer.m + col), K32_0000FFFF), K_ZERO);
                    if (result)
                    {
                        result = Detect<false>(hid, offset + col / 2, 0, result);
                        Store<false>(buffer.d + col, _mm512_maskz_set1_epi32(result, 1));
                    }
                }
                if (col < evenWidth)
                {
                    __mmask16 result = _mm512_cmpneq_epi32_mask(_mm512_and_si512((Load<false, true>((uint32_t*)buffer.m + col / 2, tailMask)), K32_0000FFFF), K_ZERO);
                    if (result)
                    {
                        result = Detect<true>(hid, offset + col / 2, 0, result);
                        Store<false, true>((uint32_t*)buffer.d + col / 2, _mm512_maskz_set1_epi32(result, 1), tailMask);
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

        SIMD_INLINE __m512i IntegralSum16i(const __m512i & s0, const __m512i & s1, const __m512i & s2, const __m512i & s3)
        {
            return _mm512_sub_epi16(_mm512_sub_epi16(s0, s1), _mm512_sub_epi16(s2, s3));
        }

        template<int i, bool masked> SIMD_INLINE void Load(__m512i a[16], const HidLbpFeature<uint16_t> & feature, ptrdiff_t offset, __mmask32 tail = -1)
        {
            a[i] = Load<false, masked>(feature.p[i] + offset, tail);
        }

        template <bool masked> SIMD_INLINE void Calculate(const HidLbpFeature<uint16_t> & feature, ptrdiff_t offset, __mmask32 & index, __m512i & shuffle, __m512i & mask, __mmask32 tail = -1)
        {
            __m512i a[16];
            Load<5, masked>(a, feature, offset, tail);
            Load<6, masked>(a, feature, offset, tail);
            Load<9, masked>(a, feature, offset, tail);
            Load<10, masked>(a, feature, offset, tail);
            __m512i central = IntegralSum16i(a[5], a[6], a[9], a[10]);

            Load<0, masked>(a, feature, offset, tail);
            Load<1, masked>(a, feature, offset, tail);
            Load<4, masked>(a, feature, offset, tail);
            index = _mm512_cmpge_epu16_mask(IntegralSum16i(a[0], a[1], a[4], a[5]), central);

            shuffle = K16_FF00;
            Load<2, masked>(a, feature, offset, tail);
            shuffle = _mm512_or_si512(shuffle, _mm512_maskz_set1_epi16(_mm512_cmpge_epu16_mask(IntegralSum16i(a[1], a[2], a[5], a[6]), central), 8));
            Load<3, masked>(a, feature, offset, tail);
            Load<7, masked>(a, feature, offset, tail);
            shuffle = _mm512_or_si512(shuffle, _mm512_maskz_set1_epi16(_mm512_cmpge_epu16_mask(IntegralSum16i(a[2], a[3], a[6], a[7]), central), 4));
            Load<11, masked>(a, feature, offset, tail);
            shuffle = _mm512_or_si512(shuffle, _mm512_maskz_set1_epi16(_mm512_cmpge_epu16_mask(IntegralSum16i(a[6], a[7], a[10], a[11]), central), 2));
            Load<14, masked>(a, feature, offset, tail);
            Load<15, masked>(a, feature, offset, tail);
            shuffle = _mm512_or_si512(shuffle, _mm512_maskz_set1_epi16(_mm512_cmpge_epu16_mask(IntegralSum16i(a[10], a[11], a[14], a[15]), central), 1));

            mask = K16_FF00;
            Load<13, masked>(a, feature, offset, tail);
            mask = _mm512_or_si512(mask, _mm512_maskz_set1_epi16(_mm512_cmpge_epu16_mask(IntegralSum16i(a[9], a[10], a[13], a[14]), central), 4));
            Load<12, masked>(a, feature, offset, tail);
            Load<8, masked>(a, feature, offset, tail);
            mask = _mm512_or_si512(mask, _mm512_maskz_set1_epi16(_mm512_cmpge_epu16_mask(IntegralSum16i(a[8], a[9], a[12], a[13]), central), 2));
            mask = _mm512_or_si512(mask, _mm512_maskz_set1_epi16(_mm512_cmpge_epu16_mask(IntegralSum16i(a[4], a[5], a[8], a[9]), central), 1));
            mask = _mm512_shuffle_epi8(K8_SHUFFLE_BITS, mask);
        }

        template <bool masked> SIMD_INLINE __mmask32 LeafMask(const HidLbpFeature<uint16_t> & feature, ptrdiff_t offset, const int * subset, __mmask32 tail = -1)
        {
            __mmask32 index;
            __m512i shuffle, mask;
            Calculate<masked>(feature, offset, index, shuffle, mask, tail);

            __m256i _subset = _mm256_loadu_si256((__m256i*)subset);
            __m512i subset0 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_subset, 0));
            __m512i subset1 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_subset, 1));

            __m512i value0 = _mm512_and_si512(_mm512_shuffle_epi8(subset0, shuffle), mask);
            __m512i value1 = _mm512_and_si512(_mm512_shuffle_epi8(subset1, shuffle), mask);
            __m512i value = _mm512_mask_blend_epi16(index, value0, value1);

            return _mm512_cmpneq_epi16_mask(value, K_ZERO);
        }

        template<bool masked> __mmask32 Detect(const HidLbpCascade<int, uint16_t> & hid, size_t offset, int startStage, __mmask32 result)
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
                __m512i sum = _mm512_setzero_si512();
                for (int i_tree = 0, n_trees = stage.ntrees; i_tree < n_trees; i_tree++)
                {
                    const Hid::Feature & feature = hid.features[nodes[nodeOffset].featureIdx];
                    const int * subset = subsets + nodeOffset*subsetSize;
                    __mmask32 mask = LeafMask<masked>(feature, offset, subset, result);
                    sum = _mm512_add_epi16(sum, _mm512_mask_blend_epi16(mask, _mm512_set1_epi16(leaves[leafOffset + 1]), _mm512_set1_epi16(leaves[leafOffset + 0])));
                    nodeOffset++;
                    leafOffset += 2;
                }
                result = result & _mm512_cmpge_epi16_mask(sum, _mm512_set1_epi16(stage.threshold));
                if (!result)
                    return result;
                int resultCount = _mm_popcnt_u32(result);
                if (resultCount == 1)
                {
                    int j = _tzcnt_u32(result);
                    return Base::Detect(hid, offset + j, i_stage + 1) > 0 ? result : __mmask32(0);
                }
            }
            return result;
        }

        void DetectionLbpDetect16ip(const HidLbpCascade<int, uint16_t> & hid, const Image & mask, const Rect & rect, Image & dst)
        {
            size_t width = rect.Width();
            size_t alignedWidth = Simd::AlignLo(width, HA);
            __mmask32 tailMask = TailMask32(width - alignedWidth);
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
                    __mmask32 result = _mm512_cmpneq_epi16_mask(Load<false>(buffer.m + col), K_ZERO);
                    if (result)
                    {
                        result = Detect<false>(hid, offset + col, 0, result);
                        Store<false>(buffer.d + col, _mm512_maskz_set1_epi16(result, 1));
                    }
                }
                if (col < width)
                {
                    __mmask32 result = _mm512_cmpneq_epi16_mask((Load<false, true>(buffer.m + col, tailMask)), K_ZERO);
                    if (result)
                    {
                        result = Detect<true>(hid, offset + col, 0, result);
                        Store<false, true>(buffer.d + col, _mm512_maskz_set1_epi16(result, 1), tailMask);
                    }
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
            __mmask32 tailMask = TailMask32((width - alignedWidth) / 2);
            size_t evenWidth = Simd::AlignLo(width, 2);

            for (ptrdiff_t row = rect.top; row < rect.bottom; row += step)
            {
                size_t col = 0;
                size_t offset = row * hid.isum.stride / sizeof(uint16_t) + rect.left / 2;
                const uint8_t * m = mask.data + row*mask.stride + rect.left;
                uint8_t * d = dst.data + row*dst.stride + rect.left;
                for (; col < alignedWidth; col += A)
                {
                    __mmask32 result = _mm512_cmpneq_epi16_mask(_mm512_and_si512(Load<false>(m + col), K16_00FF), K_ZERO);
                    if (result)
                    {
                        result = Detect<false>(hid, offset + col / 2, 0, result);
                        Store<false>(d + col, _mm512_maskz_set1_epi16(result, 1));
                    }
                }
                if (col < evenWidth)
                {
                    __mmask32 result = _mm512_cmpneq_epi16_mask(_mm512_and_si512((Load<false, true>((uint16_t*)m + col / 2, tailMask)), K16_00FF), K_ZERO);
                    if (result)
                    {
                        result = Detect<true>(hid, offset + col / 2, 0, result);
                        Store<false, true>((uint16_t*)d + col / 2, _mm512_maskz_set1_epi16(result, 1), tailMask);
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
#endif// SIMD_AVX512BW_ENABLE
}
