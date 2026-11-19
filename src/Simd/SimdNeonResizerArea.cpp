/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdShuffle.h"
#include "Simd/SimdResizer.h"
#include "Simd/SimdResizerCommon.h"
#include "Simd/SimdUpdate.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        ResizerByteArea1x1::ResizerByteArea1x1(const ResParam & param)
            : Base::ResizerByteArea1x1(param)
        {
        }

        const uint8x16_t K8_TAIL_SHUFFLE = SIMD_VEC_SETR_EPI8(0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF);

        SIMD_INLINE uint8x16_t ResizerByteArea1x1LoadTail(const uint8_t * src, size_t size)
        {
            size_t sizeA = AlignLo(size, A), tail = size - sizeA;
            if (sizeA)
                return Shuffle(Load<false>(src + size - A), vaddq_u8(K8_TAIL_SHUFFLE, vdupq_n_u8(uint8_t(A - tail))));
            uint8_t buffer[A] = { 0 };
            memcpy(buffer, src, tail);
            return vld1q_u8(buffer);
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1Update(uint8x16_t s0, int16x4_t a0, int32_t * dst)
        {
            int16x8_t u00 = UnpackU8s<0>(s0);
            int16x8_t u01 = UnpackU8s<1>(s0);
            Update<update, true>(dst + 0 * F, vmull_s16(a0, Half<0>(u00)));
            Update<update, true>(dst + 1 * F, vmull_s16(a0, Half<1>(u00)));
            Update<update, true>(dst + 2 * F, vmull_s16(a0, Half<0>(u01)));
            Update<update, true>(dst + 3 * F, vmull_s16(a0, Half<1>(u01)));
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1RowUpdate(const uint8_t * src0, size_t size, int32_t a0, int32_t * dst)
        {
            int16x4_t _a0 = vdup_n_s16(a0);
            size_t sizeA = AlignLo(size, A), i = 0;
            for (; i < sizeA; i += A, dst += A)
                ResizerByteArea1x1Update<update>(Load<false>(src0 + i), _a0, dst);
            if (i < size)
                ResizerByteArea1x1Update<update>(ResizerByteArea1x1LoadTail(src0, size), _a0, dst);
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1Update(uint8x16_t s0, uint8x16_t s1, int16x4_t a0, int16x4_t a1, int32_t * dst)
        {
            int16x8_t u00 = UnpackU8s<0>(s0);
            int16x8_t u01 = UnpackU8s<1>(s0);
            int16x8_t u10 = UnpackU8s<0>(s1);
            int16x8_t u11 = UnpackU8s<1>(s1);
            Update<update, true>(dst + 0 * F, vmlal_s16(vmull_s16(a0, Half<0>(u00)), a1, Half<0>(u10)));
            Update<update, true>(dst + 1 * F, vmlal_s16(vmull_s16(a0, Half<1>(u00)), a1, Half<1>(u10)));
            Update<update, true>(dst + 2 * F, vmlal_s16(vmull_s16(a0, Half<0>(u01)), a1, Half<0>(u11)));
            Update<update, true>(dst + 3 * F, vmlal_s16(vmull_s16(a0, Half<1>(u01)), a1, Half<1>(u11)));
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1RowUpdate(const uint8_t * src0, size_t stride, size_t size, int32_t a0, int32_t a1, int32_t * dst)
        {
            int16x4_t _a0 = vdup_n_s16(a0);
            int16x4_t _a1 = vdup_n_s16(a1);
            const uint8_t * src1 = src0 + stride;
            size_t sizeA = AlignLo(size, A), i = 0;
            for (; i < sizeA; i += A, dst += A)
                ResizerByteArea1x1Update<update>(Load<false>(src0 + i), Load<false>(src1 + i), _a0, _a1, dst);
            if (i < size)
                ResizerByteArea1x1Update<update>(ResizerByteArea1x1LoadTail(src0, size),
                    ResizerByteArea1x1LoadTail(src1, size), _a0, _a1, dst);
        }

        SIMD_INLINE void ResizerByteArea1x1RowSum(const uint8_t * src, size_t stride, size_t count, size_t size, int32_t curr, int32_t zero, int32_t next, int32_t * dst)
        {
            if (count)
            {
                size_t i = 0;
                ResizerByteArea1x1RowUpdate<UpdateSet>(src, stride, size, curr, count == 1 ? zero - next : zero, dst), src += 2 * stride, i += 2;
                for (; i < count; i += 2, src += 2 * stride)
                    ResizerByteArea1x1RowUpdate<UpdateAdd>(src, stride, size, zero, i == count - 1 ? zero - next : zero, dst);
                if (i == count)
                    ResizerByteArea1x1RowUpdate<UpdateAdd>(src, size, zero - next, dst);
            }
            else
                ResizerByteArea1x1RowUpdate<UpdateSet>(src, size, curr - next, dst);
        }

        template<size_t N> SIMD_INLINE void ResizerByteAreaResult(const int32_t * src, size_t count, int32_t curr, int32_t zero, int32_t next, uint8_t * dst)
        {
            int32_t sum[N];
            Base::ResizerByteAreaSet<N>(src, curr, sum);
            for (size_t i = 0; i < count; ++i)
                src += N, Base::ResizerByteAreaAdd<N>(src, zero, sum);
            Base::ResizerByteAreaAdd<N>(src, -next, sum);
            Base::ResizerByteAreaRes<N>(sum, dst);
        }

        template<size_t N> SIMD_INLINE void ResizerByteAreaResult34(const int32_t * src, size_t count, int32_t curr, int32_t zero, int32_t next, uint8_t * dst)
        {
            int32x4_t sum = vmulq_s32(Load<false>(src), vdupq_n_s32(curr));
            for (size_t i = 0; i < count; ++i)
                src += N, sum = vmlaq_s32(sum, Load<false>(src), vdupq_n_s32(zero));
            sum = vmlaq_s32(sum, Load<false>(src), vdupq_n_s32(-next));
            int32x4_t res = vshrq_n_s32(vaddq_s32(sum, vdupq_n_s32(Base::AREA_ROUND)), Base::AREA_SHIFT);
            *(uint32_t*)dst = vget_lane_u32((uint32x2_t)vqmovn_u16(vcombine_u16(vqmovun_s32(res), vdup_n_u16(0))), 0);
        }

        template<> SIMD_INLINE void ResizerByteAreaResult<4>(const int32_t * src, size_t count, int32_t curr, int32_t zero, int32_t next, uint8_t * dst)
        {
            ResizerByteAreaResult34<4>(src, count, curr, zero, next, dst);
        }

        template<> SIMD_INLINE void ResizerByteAreaResult<3>(const int32_t * src, size_t count, int32_t curr, int32_t zero, int32_t next, uint8_t * dst)
        {
            ResizerByteAreaResult34<3>(src, count, curr, zero, next, dst);
        }

        template<size_t N> void ResizerByteArea1x1::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            size_t dstW = _param.dstW, bodyW = dstW - (N == 3 ? 1 : 0), rowSize = _param.srcW*N, rowRest = dstStride - dstW * N;
            const int32_t * iy = _iy.data, *ix = _ix.data, *ay = _ay.data, *ax = _ax.data;
            int32_t ay0 = ay[0], ax0 = ax[0];
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += rowRest)
            {
                int32_t * buf = _by.data;
                size_t yn = iy[dy + 1] - iy[dy];
                ResizerByteArea1x1RowSum(src, srcStride, yn, rowSize, ay[dy], ay0, ay[dy + 1], buf), src += yn * srcStride;
                size_t dx = 0;
                for (; dx < bodyW; dx++, dst += N)
                {
                    size_t xn = ix[dx + 1] - ix[dx];
                    Neon::ResizerByteAreaResult<N>(buf, xn, ax[dx], ax0, ax[dx + 1], dst), buf += xn * N;
                }
                for (; dx < dstW; dx++, dst += N)
                {
                    size_t xn = ix[dx + 1] - ix[dx];
                    Base::ResizerByteAreaResult<N>(buf, xn, ax[dx], ax0, ax[dx + 1], dst), buf += xn * N;
                }
            }
        }

        void ResizerByteArea1x1::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            switch (_param.channels)
            {
            case 1: Run<1>(src, srcStride, dst, dstStride); return;
            case 2: Run<2>(src, srcStride, dst, dstStride); return;
            case 3: Run<3>(src, srcStride, dst, dstStride); return;
            case 4: Run<4>(src, srcStride, dst, dstStride); return;
            default:
                assert(0);
            }
        }
    }
#endif// SIMD_NEON_ENABLE
}
