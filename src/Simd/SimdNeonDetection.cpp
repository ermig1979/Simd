/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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

        //SIMD_INLINE __m128i Sum32ii(uint32_t * const ptr[4], size_t offset)
        //{
        //    __m128i lo = Sum32ip(ptr, offset + 0);
        //    __m128i hi = Sum32ip(ptr, offset + 4);
        //    return _mm_or_si128(_mm_srli_si128(_mm_shuffle_epi32(lo, 0x80), 8), _mm_slli_si128(_mm_shuffle_epi32(hi, 0x08), 8));
        //}

        SIMD_INLINE float32x4_t Norm32fp(const HidHaarCascade & hid, size_t offset)
        {
            float32x4_t area = vdupq_n_f32(hid.windowArea);
            float32x4_t sum = vcvtq_f32_u32(Sum32ip(hid.p, offset));
            float32x4_t sqsum = vcvtq_f32_u32(Sum32ip(hid.pq, offset));
            return ValidSqrt(vmlsq_f32(vmulq_f32(sqsum, area), sum, sum));
        }

        //SIMD_INLINE __m128 Norm32fi(const HidHaarCascade & hid, size_t offset)
        //{
        //    __m128 area = _mm_set1_ps(hid.windowArea);
        //    __m128 sum = _mm_cvtepi32_ps(Sum32ii(hid.p, offset));
        //    __m128 sqsum = _mm_cvtepi32_ps(Sum32ii(hid.pq, offset));
        //    return ValidSqrt(_mm_sub_ps(_mm_mul_ps(sqsum, area), _mm_mul_ps(sum, sum)));
        //}

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
            typedef HidHaarCascade Hid;

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
                memset(buffer.d, 0, width*sizeof(uint32_t));
                for (; col < alignedWidth; col += 4)
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
                    col = evenWidth - 4;
                    uint32x4_t result = vld1q_u32(buffer.m + col);
                    if (ResultCount(result) != 0)
                    {
                        float32x4_t norm = Norm32fp(hid, pq_offset + col);
                        Detect32f(hid, p_offset + col, norm, result);
                        vst1q_u32(buffer.d + col, result);
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
    }
#endif// SIMD_NEON_ENABLE
}