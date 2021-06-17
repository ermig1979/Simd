/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdResizer.h"
#include "Simd/SimdUpdate.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        ResizerByteBilinear::ResizerByteBilinear(const ResParam & param)
            : Base::ResizerByteBilinear(param)
            , _blocks(0)
        {
        }

        size_t ResizerByteBilinear::BlockCountMax(size_t align)
        {
            return (size_t)Simd::Max(::ceil(float(_param.srcW) / (align - 1)), ::ceil(float(_param.dstW) * 2.0f / align));
        }

        void ResizerByteBilinear::EstimateParams()
        {
            if (_ax.data)
                return;
            if (_param.channels == 1 && _param.srcW < 4 * _param.dstW)
                _blocks = BlockCountMax(A);
            float scale = (float)_param.srcW / _param.dstW;
            _ax.Resize(_param.dstW * _param.channels * 2, false, _param.align);
            uint8_t * alphas = _ax.data;
            if (_blocks)
            {
                _ixg.Resize(_blocks);
                int block = 0;
                _ixg[0].src = 0;
                _ixg[0].dst = 0;
                for (int dstIndex = 0; dstIndex < _param.dstW; ++dstIndex)
                {
                    float alpha = (float)((dstIndex + 0.5)*scale - 0.5);
                    int srcIndex = (int)::floor(alpha);
                    alpha -= srcIndex;

                    if (srcIndex < 0)
                    {
                        srcIndex = 0;
                        alpha = 0;
                    }

                    if (srcIndex > _param.srcW - 2)
                    {
                        srcIndex = (int)_param.srcW - 2;
                        alpha = 1;
                    }

                    int dst = 2 * dstIndex - _ixg[block].dst;
                    int src = srcIndex - _ixg[block].src;
                    if (src >= A - 1 || dst >= A)
                    {
                        block++;
                        _ixg[block].src = Simd::Min(srcIndex, int(_param.srcW - A));
                        _ixg[block].dst = 2 * dstIndex;
                        dst = 0;
                        src = srcIndex - _ixg[block].src;
                    }
                    _ixg[block].shuffle[dst] = src;
                    _ixg[block].shuffle[dst + 1] = src + 1;

                    alphas[1] = (uint8_t)(alpha * Base::FRACTION_RANGE + 0.5);
                    alphas[0] = (uint8_t)(Base::FRACTION_RANGE - alphas[1]);
                    alphas += 2;
                }
                _blocks = block + 1;
            }
            else
            {
                _ix.Resize(_param.dstW);
                for (size_t i = 0; i < _param.dstW; ++i)
                {
                    float alpha = (float)((i + 0.5)*scale - 0.5);
                    ptrdiff_t index = (ptrdiff_t)::floor(alpha);
                    alpha -= index;

                    if (index < 0)
                    {
                        index = 0;
                        alpha = 0;
                    }

                    if (index >(ptrdiff_t)_param.srcW - 2)
                    {
                        index = _param.srcW - 2;
                        alpha = 1;
                    }

                    _ix[i] = (int)index;
                    alphas[1] = (uint8_t)(alpha * Base::FRACTION_RANGE + 0.5);
                    alphas[0] = (uint8_t)(Base::FRACTION_RANGE - alphas[1]);
                    for (size_t channel = 1; channel < _param.channels; channel++)
                        ((uint16_t*)alphas)[channel] = *(uint16_t*)alphas;
                    alphas += 2 * _param.channels;
                }
            }
            size_t size = AlignHi(_param.dstW, _param.align)*_param.channels * 2;
            _bx[0].Resize(size, false, _param.align);
            _bx[1].Resize(size, false, _param.align);
}

        template <size_t N> void ResizerByteBilinearInterpolateX(const uint8_t * alpha, uint8_t * buffer);

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<1>(const uint8_t * alpha, uint8_t * buffer)
        {
            uint8x8x2_t a = vld2_u8(alpha);
            uint8x8x2_t b = vld2_u8(buffer);
            Store<true>(buffer, (uint8x16_t)vaddq_u16(vmull_u8(a.val[0], b.val[0]), vmull_u8(a.val[1], b.val[1])));
        }

        SIMD_INLINE void ResizerByteBilinearInterpolateX2(const uint8_t * alpha, uint8_t * buffer)
        {
            uint8x8x2_t a = vld2_u8(alpha);
            uint16x4x2_t b = vld2_u16((uint16_t*)buffer);
            Store<true>(buffer, (uint8x16_t)vaddq_u16(vmull_u8(a.val[0], (uint8x8_t)b.val[0]), vmull_u8(a.val[1], (uint8x8_t)b.val[1])));
        }

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<2>(const uint8_t * alpha, uint8_t * buffer)
        {
            ResizerByteBilinearInterpolateX2(alpha + 0, buffer + 0);
            ResizerByteBilinearInterpolateX2(alpha + A, buffer + A);
        }

        SIMD_INLINE void ResizerByteBilinearInterpolateX3(const uint8_t * alpha, const uint8_t * src, uint8_t * dst)
        {
            uint8x8x2_t a = vld2_u8(alpha);
            uint8x8x2_t b = vld2_u8(src);
            Store<true>(dst, (uint8x16_t)vaddq_u16(vmull_u8(a.val[0], b.val[0]), vmull_u8(a.val[1], b.val[1])));
        }

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<3>(const uint8_t * alpha, uint8_t * buffer)
        {
            uint8_t b[3 * A];
            uint8x16x3_t _b = vld3q_u8(buffer);
            vst3q_u16((uint16_t*)b, *(uint16x8x3_t*)&_b);
            ResizerByteBilinearInterpolateX3(alpha + 0 * A, b + 0 * A, buffer + 0 * A);
            ResizerByteBilinearInterpolateX3(alpha + 1 * A, b + 1 * A, buffer + 1 * A);
            ResizerByteBilinearInterpolateX3(alpha + 2 * A, b + 2 * A, buffer + 2 * A);
        }

        SIMD_INLINE void ResizerByteBilinearInterpolateX4(const uint8_t * alpha, uint8_t * buffer)
        {
            uint8x8x2_t a = vld2_u8(alpha);
            uint32x2x2_t b = vld2_u32((uint32_t*)buffer);
            Store<true>(buffer, (uint8x16_t)vaddq_u16(vmull_u8(a.val[0], (uint8x8_t)b.val[0]), vmull_u8(a.val[1], (uint8x8_t)b.val[1])));
        }

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<4>(const uint8_t * alpha, uint8_t * buffer)
        {
            ResizerByteBilinearInterpolateX4(alpha + 0 * A, buffer + 0 * A);
            ResizerByteBilinearInterpolateX4(alpha + 1 * A, buffer + 1 * A);
            ResizerByteBilinearInterpolateX4(alpha + 2 * A, buffer + 2 * A);
            ResizerByteBilinearInterpolateX4(alpha + 3 * A, buffer + 3 * A);
        }

        const uint16x8_t K16_FRACTION_ROUND_TERM = SIMD_VEC_SET1_EPI16(Base::BILINEAR_ROUND_TERM);

        template<bool align> SIMD_INLINE uint16x8_t ResizerByteBilinearInterpolateY(const uint16_t * pbx0, const uint16_t * pbx1, uint16x8_t alpha[2])
        {
            uint16x8_t sum = vaddq_u16(vmulq_u16(Load<align>(pbx0), alpha[0]), vmulq_u16(Load<align>(pbx1), alpha[1]));
            return vshrq_n_u16(vaddq_u16(sum, K16_FRACTION_ROUND_TERM), Base::BILINEAR_SHIFT);
        }

        template<bool align> SIMD_INLINE void ResizerByteBilinearInterpolateY(const uint8_t * bx0, const uint8_t * bx1, uint16x8_t alpha[2], uint8_t * dst)
        {
            uint16x8_t lo = ResizerByteBilinearInterpolateY<align>((uint16_t*)(bx0 + 0), (uint16_t*)(bx1 + 0), alpha);
            uint16x8_t hi = ResizerByteBilinearInterpolateY<align>((uint16_t*)(bx0 + A), (uint16_t*)(bx1 + A), alpha);
            Store<false>(dst, PackU16(lo, hi));
        }

        template<size_t N> void ResizerByteBilinear::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            struct One { uint8_t val[N * 1]; };
            struct Two { uint8_t val[N * 2]; };

            size_t size = 2 * _param.dstW*N;
            size_t aligned = AlignHi(size, DA) - DA;
            const size_t step = A * N;
            ptrdiff_t previous = -2;
            uint16x8_t a[2];
            uint8_t * bx[2] = { _bx[0].data, _bx[1].data };
            const uint8_t * ax = _ax.data;
            const int32_t * ix = _ix.data;
            size_t dstW = _param.dstW;

            for (size_t yDst = 0; yDst < _param.dstH; yDst++, dst += dstStride)
            {
                a[0] = vdupq_n_u16(int16_t(Base::FRACTION_RANGE - _ay[yDst]));
                a[1] = vdupq_n_u16(int16_t(_ay[yDst]));

                ptrdiff_t sy = _iy[yDst];
                int k = 0;

                if (sy == previous)
                    k = 2;
                else if (sy == previous + 1)
                {
                    Swap(bx[0], bx[1]);
                    k = 1;
                }

                previous = sy;

                for (; k < 2; k++)
                {
                    Two * pb = (Two *)bx[k];
                    const One * psrc = (const One *)(src + (sy + k)*srcStride);
                    for (size_t x = 0; x < dstW; x++)
                        pb[x] = *(Two *)(psrc + ix[x]);

                    uint8_t * pbx = bx[k];
                    for (size_t i = 0; i < size; i += step)
                        ResizerByteBilinearInterpolateX<N>(ax + i, pbx + i);
                }

                for (size_t ib = 0, id = 0; ib < aligned; ib += DA, id += A)
                    ResizerByteBilinearInterpolateY<true>(bx[0] + ib, bx[1] + ib, a, dst + id);
                size_t i = size - DA;
                ResizerByteBilinearInterpolateY<false>(bx[0] + i, bx[1] + i, a, dst + i / 2);
            }
        }

        union ResizerByteBilinearLoadGrayInterpolatedHelper
        {
            uint8x16_t full;
            uint8x8x2_t half;
        }; 

        template <class Idx> SIMD_INLINE void ResizerByteBilinearLoadGrayInterpolated(const uint8_t * src, const Idx & index, const uint8_t * alpha, uint8_t * dst)
        {
            ResizerByteBilinearLoadGrayInterpolatedHelper _src, _shuffle, _alpha, unpacked;
            _src.full = vld1q_u8(src + index.src);
            _shuffle.full = vld1q_u8(index.shuffle);
            unpacked.half.val[0] = vtbl2_u8(_src.half, _shuffle.half.val[0]);
            unpacked.half.val[1] = vtbl2_u8(_src.half, _shuffle.half.val[1]);
            _alpha.full = vld1q_u8(alpha + index.dst);
            uint16x8_t lo = vmull_u8(unpacked.half.val[0], _alpha.half.val[0]);
            uint16x8_t hi = vmull_u8(unpacked.half.val[1], _alpha.half.val[1]);
            vst1q_u8(dst + index.dst, vreinterpretq_u8_u16(Hadd(lo, hi)));
        }

        template <class Idx> SIMD_INLINE void ResizerByteBilinearLoadGray(const uint8_t * src, const Idx & index, uint8_t * dst)
        {
            ResizerByteBilinearLoadGrayInterpolatedHelper _src, _shuffle, _alpha, unpacked;
            _src.full = vld1q_u8(src + index.src);
            _shuffle.full = vld1q_u8(index.shuffle);
            unpacked.half.val[0] = vtbl2_u8(_src.half, _shuffle.half.val[0]);
            unpacked.half.val[1] = vtbl2_u8(_src.half, _shuffle.half.val[1]);
            vst1q_u8(dst + index.dst, unpacked.full);
        }

//#define MERGE_LOADING_AND_INTERPOLATION

        void ResizerByteBilinear::RunG(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            size_t bufW = AlignHi(_param.dstW, A) * 2;
            size_t size = 2 * _param.dstW;
            size_t aligned = AlignHi(size, DA) - DA;
            size_t blocks = _blocks;
            ptrdiff_t previous = -2;
            uint16x8_t a[2];
            uint8_t * bx[2] = { _bx[0].data, _bx[1].data };
            const uint8_t * ax = _ax.data;
            const Idx * ixg = _ixg.data;

            for (size_t yDst = 0; yDst < _param.dstH; yDst++, dst += dstStride)
            {
                a[0] = vdupq_n_u16(int16_t(Base::FRACTION_RANGE - _ay[yDst]));
                a[1] = vdupq_n_u16(int16_t(_ay[yDst]));

                ptrdiff_t sy = _iy[yDst];
                int k = 0;

                if (sy == previous)
                    k = 2;
                else if (sy == previous + 1)
                {
                    Swap(bx[0], bx[1]);
                    k = 1;
                }

                previous = sy;

                for (; k < 2; k++)
                {
#ifdef MERGE_LOADING_AND_INTERPOLATION
                    const uint8_t * psrc = src + (sy + k)*srcStride;
                    uint8_t * pdst = bx[k];
                    for (size_t i = 0; i < blocks; ++i)
                        ResizerByteBilinearLoadGrayInterpolated(psrc, ixg[i], ax, pdst);
#else
                    const uint8_t * psrc = src + (sy + k)*srcStride;
                    uint8_t * pdst = bx[k];
                    for (size_t i = 0; i < blocks; ++i)
                        ResizerByteBilinearLoadGray(psrc, ixg[i], pdst);

                    uint8_t * pbx = bx[k];
                    for (size_t i = 0; i < size; i += A)
                        ResizerByteBilinearInterpolateX<1>(ax + i, pbx + i);
#endif
                }

                for (size_t ib = 0, id = 0; ib < aligned; ib += DA, id += A)
                    ResizerByteBilinearInterpolateY<true>(bx[0] + ib, bx[1] + ib, a, dst + id);
                size_t i = size - DA;
                ResizerByteBilinearInterpolateY<false>(bx[0] + i, bx[1] + i, a, dst + i / 2);
            }
        }

        void ResizerByteBilinear::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            assert(_param.dstW >= A);

            EstimateParams();
            switch (_param.channels)
            {
            case 1:
                if (_blocks)
                    RunG(src, srcStride, dst, dstStride);
                else
                    Run<1>(src, srcStride, dst, dstStride);
                break;
            case 2: Run<2>(src, srcStride, dst, dstStride); break;
            case 3: Run<3>(src, srcStride, dst, dstStride); break;
            case 4: Run<4>(src, srcStride, dst, dstStride); break;
            default:
                assert(0);
            }
        }
        //---------------------------------------------------------------------

        ResizerByteArea::ResizerByteArea(const ResParam & param)
            : Base::ResizerByteArea(param)
        {
            _by.Resize(AlignHi(_param.srcW*_param.channels, _param.align), false, _param.align);
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteAreaRowUpdate(const uint8_t * src0, size_t size, int32_t a0, int32_t * dst)
        {
            int16x4_t _a0 = vdup_n_s16(a0);
            for (size_t i = 0; i < size; i += A, dst += A, src0 += A)
            {
                uint8x16_t s0 = Load<false>(src0);
                int16x8_t u00 = UnpackU8s<0>(s0);
                int16x8_t u01 = UnpackU8s<1>(s0);
                Update<update, true>(dst + 0 * F, vmull_s16(_a0, Half<0>(u00)));
                Update<update, true>(dst + 1 * F, vmull_s16(_a0, Half<1>(u00)));
                Update<update, true>(dst + 2 * F, vmull_s16(_a0, Half<0>(u01)));
                Update<update, true>(dst + 3 * F, vmull_s16(_a0, Half<1>(u01)));
            }
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteAreaRowUpdate(const uint8_t * src0, size_t stride, size_t size, int32_t a0, int32_t a1, int32_t * dst)
        {
            int16x4_t _a0 = vdup_n_s16(a0);
            int16x4_t _a1 = vdup_n_s16(a1);
            const uint8_t * src1 = src0 + stride;
            for (size_t i = 0; i < size; i += A, dst += A)
            {
                uint8x16_t s0 = Load<false>(src0 + i);
                uint8x16_t s1 = Load<false>(src1 + i);
                int16x8_t u00 = UnpackU8s<0>(s0);
                int16x8_t u01 = UnpackU8s<1>(s0);
                int16x8_t u10 = UnpackU8s<0>(s1);
                int16x8_t u11 = UnpackU8s<1>(s1);
                Update<update, true>(dst + 0 * F, vmlal_s16(vmull_s16(_a0, Half<0>(u00)), _a1, Half<0>(u10)));
                Update<update, true>(dst + 1 * F, vmlal_s16(vmull_s16(_a0, Half<1>(u00)), _a1, Half<1>(u10)));
                Update<update, true>(dst + 2 * F, vmlal_s16(vmull_s16(_a0, Half<0>(u01)), _a1, Half<0>(u11)));
                Update<update, true>(dst + 3 * F, vmlal_s16(vmull_s16(_a0, Half<1>(u01)), _a1, Half<1>(u11)));
            }
        }

        SIMD_INLINE void ResizerByteAreaRowSum(const uint8_t * src, size_t stride, size_t count, size_t size, int32_t curr, int32_t zero, int32_t next, int32_t * dst)
        {
            if (count)
            {
                size_t i = 0;
                ResizerByteAreaRowUpdate<UpdateSet>(src, stride, size, curr, count == 1 ? zero - next : zero, dst), src += 2 * stride, i += 2;
                for (; i < count; i += 2, src += 2 * stride)
                    ResizerByteAreaRowUpdate<UpdateAdd>(src, stride, size, zero, i == count - 1 ? zero - next : zero, dst);
                if (i == count)
                    ResizerByteAreaRowUpdate<UpdateAdd>(src, size, zero - next, dst);
            }
            else
                ResizerByteAreaRowUpdate<UpdateSet>(src, size, curr - next, dst);
        }

        template<size_t N> SIMD_INLINE void ResizerByteAreaSet(const int32_t * src, int32_t value, int32_t * dst)
        {
            for (size_t c = 0; c < N; ++c)
                dst[c] = src[c] * value;
        }

        template<size_t N> SIMD_INLINE void ResizerByteAreaAdd(const int32_t * src, int32_t value, int32_t * dst)
        {
            for (size_t c = 0; c < N; ++c)
                dst[c] += src[c] * value;
        }

        template<size_t N> SIMD_INLINE void ResizerByteAreaRes(const int32_t * src, uint8_t * dst)
        {
            for (size_t c = 0; c < N; ++c)
                dst[c] = uint8_t((src[c] + Base::AREA_ROUND) >> Base::AREA_SHIFT);
        }

        template<size_t N> SIMD_INLINE void ResizerByteAreaResult(const int32_t * src, size_t count, int32_t curr, int32_t zero, int32_t next, uint8_t * dst)
        {
            int32_t sum[N];
            ResizerByteAreaSet<N>(src, curr, sum);
            for (size_t i = 0; i < count; ++i)
                src += N, ResizerByteAreaAdd<N>(src, zero, sum);
            ResizerByteAreaAdd<N>(src, -next, sum);
            ResizerByteAreaRes<N>(sum, dst);
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

        template<size_t N> void ResizerByteArea::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            size_t dstW = _param.dstW, rowSize = _param.srcW*N, rowRest = dstStride - dstW * N;
            const int32_t * iy = _iy.data, *ix = _ix.data, *ay = _ay.data, *ax = _ax.data;
            int32_t ay0 = ay[0], ax0 = ax[0];
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += rowRest)
            {
                int32_t * buf = _by.data;
                size_t yn = iy[dy + 1] - iy[dy];
                ResizerByteAreaRowSum(src, srcStride, yn, rowSize, ay[dy], ay0, ay[dy + 1], buf), src += yn * srcStride;
                for (size_t dx = 0; dx < dstW; dx++, dst += N)
                {
                    size_t xn = ix[dx + 1] - ix[dx];
                    ResizerByteAreaResult<N>(buf, xn, ax[dx], ax0, ax[dx + 1], dst), buf += xn * N;
                }
            }
        }

        void ResizerByteArea::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
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

        //---------------------------------------------------------------------

        ResizerShortBilinear::ResizerShortBilinear(const ResParam& param)
            : Base::ResizerShortBilinear(param)
        {
        }

        SIMD_INLINE float32x4_t BilColS1(const uint16_t* src, const int32_t* idx, float32x4_t fx0, float32x4_t fx1)
        {
            const uint32_t buf[4] = { 
                *(uint32_t*)(src + idx[0]), *(uint32_t*)(src + idx[1]),
                *(uint32_t*)(src + idx[2]), *(uint32_t*)(src + idx[3]) };
            uint16x4x2_t _buf =  LoadHalf2<false>((uint16_t*)buf);
            float32x4_t m0 = vmulq_f32(fx0, vcvtq_f32_u32(vmovl_u16(_buf.val[0])));
            float32x4_t m1 = vmulq_f32(fx1, vcvtq_f32_u32(vmovl_u16(_buf.val[1])));
            return vaddq_f32(m0, m1);
        }

        SIMD_INLINE float32x4_t BilColS2(const uint16_t* src, const int32_t* idx, float32x4_t fx0, float32x4_t fx1)
        {
            const uint64_t buf[2] = { *(uint64_t*)(src + idx[0]), *(uint64_t*)(src + idx[2]) };
            uint32x2x2_t _buf = LoadHalf2<false>((uint32_t*)buf);
            float32x4_t m0 = vmulq_f32(fx0, vcvtq_f32_u32(vmovl_u16((uint16x4_t)_buf.val[0])));
            float32x4_t m1 = vmulq_f32(fx1, vcvtq_f32_u32(vmovl_u16((uint16x4_t)_buf.val[1])));
            return vaddq_f32(m0, m1);
        }

        SIMD_INLINE float32x4_t BilColS3(const uint16_t* src, const int32_t* idx, float32x4_t fx0, float32x4_t fx1)
        {
            float32x4_t m0 = vmulq_f32(fx0, vcvtq_f32_u32(vmovl_u16(LoadHalf<false>(src + idx[0]))));
            float32x4_t m1 = vmulq_f32(fx1, vcvtq_f32_u32(vmovl_u16(LoadHalf<false>(src + idx[3]))));
            return vaddq_f32(m0, m1);
        }

        SIMD_INLINE float32x4_t BilColS4(const uint16_t* src, const int32_t* idx, float32x4_t fx0, float32x4_t fx1)
        {
            float32x4_t m0 = vmulq_f32(fx0, vcvtq_f32_u32(vmovl_u16(LoadHalf<false>(src + idx[0]))));
            float32x4_t m1 = vmulq_f32(fx1, vcvtq_f32_u32(vmovl_u16(LoadHalf<false>(src + idx[4]))));
            return vaddq_f32(m0, m1);
        }

        template<size_t N> void ResizerShortBilinear::RunB(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            size_t rs = _param.dstW * N;
            float* pbx[2] = { _bx[0].data, _bx[1].data };
            int32_t prev = -2;
            size_t rs3 = AlignLoAny(rs - 1, 3);
            size_t rs4 = AlignLo(rs, 4);
            size_t rs8 = AlignLo(rs, 8);
            float32x4_t _1 = vdupq_n_f32(1.0f);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                float fy1 = _ay[dy];
                float fy0 = 1.0f - fy1;
                int32_t sy = _iy[dy];
                int32_t k = 0;

                if (sy == prev)
                    k = 2;
                else if (sy == prev + 1)
                {
                    Swap(pbx[0], pbx[1]);
                    k = 1;
                }

                prev = sy;

                for (; k < 2; k++)
                {
                    float* pb = pbx[k];
                    const uint16_t* ps = src + (sy + k) * srcStride;
                    size_t dx = 0;
                    if (N == 1)
                    {
                        for (; dx < rs4; dx += 4)
                        {
                            float32x4_t fx1 = Load<false>(_ax.data + dx);
                            float32x4_t fx0 = vsubq_f32(_1, fx1);
                            Store<false>(pb + dx, BilColS1(ps, _ix.data + dx, fx0, fx1));
                        }
                    }
                    if (N == 2)
                    {
                        for (; dx < rs4; dx += 4)
                        {
                            float32x4_t fx1 = Load<false>(_ax.data + dx);
                            float32x4_t fx0 = vsubq_f32(_1, fx1);
                            Store<false>(pb + dx, BilColS2(ps, _ix.data + dx, fx0, fx1));
                        }
                    }
                    if (N == 3)
                    {
                        for (; dx < rs3; dx += 3)
                        {
                            float32x4_t fx1 = Load<false>(_ax.data + dx);
                            float32x4_t fx0 = vsubq_f32(_1, fx1);
                            Store<false>(pb + dx, BilColS3(ps, _ix.data + dx, fx0, fx1));
                        }
                    }
                    if (N == 4)
                    {
                        for (; dx < rs4; dx += 4)
                        {
                            float32x4_t fx1 = Load<false>(_ax.data + dx);
                            float32x4_t fx0 = vsubq_f32(_1, fx1);
                            Store<false>(pb + dx, BilColS4(ps, _ix.data + dx, fx0, fx1));
                        }
                    }
                    for (; dx < rs; dx++)
                    {
                        int32_t sx = _ix[dx];
                        float fx = _ax[dx];
                        pb[dx] = ps[sx] * (1.0f - fx) + ps[sx + N] * fx;
                    }
                }

                size_t dx = 0;
                float32x4_t _fy0 = vdupq_n_f32(fy0);
                float32x4_t _fy1 = vdupq_n_f32(fy1);
                for (; dx < rs8; dx += 8)
                {
                    float32x4_t m00 = vmulq_f32(Load<false>(pbx[0] + dx + 0), _fy0);
                    float32x4_t m01 = vmulq_f32(Load<false>(pbx[1] + dx + 0), _fy1);
                    uint32x4_t i0 = (uint32x4_t)Round(vaddq_f32(m00, m01));
                    float32x4_t m10 = vmulq_f32(Load<false>(pbx[0] + dx + 4), _fy0);
                    float32x4_t m11 = vmulq_f32(Load<false>(pbx[1] + dx + 4), _fy1);
                    uint32x4_t i1 = (uint32x4_t)Round(vaddq_f32(m10, m11));
                    Store<false>(dst + dx, PackU32(i0, i1));
                }
                for (; dx < rs4; dx += 4)
                {
                    float32x4_t m0 = vmulq_f32(Load<false>(pbx[0] + dx), _fy0);
                    float32x4_t m1 = vmulq_f32(Load<false>(pbx[1] + dx), _fy1);
                    uint32x4_t i0 = (uint32x4_t)Round(vaddq_f32(m0, m1));
                    Store<false>(dst + dx, vmovn_u32(i0));
                }
                for (; dx < rs; dx++)
                    dst[dx] = Simd::Round(pbx[0][dx] * fy0 + pbx[1][dx] * fy1);
            }
        }

        template<size_t N> void ResizerShortBilinear::RunS(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            size_t rs = _param.dstW * N;
            size_t rs3 = AlignLoAny(rs - 1, 3);
            size_t rs6 = AlignLoAny(rs - 1, 6);
            size_t rs4 = AlignLo(rs, 4);
            size_t rs8 = AlignLo(rs, 8);
            float32x4_t _1 = vdupq_n_f32(1.0f);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                float fy1 = _ay[dy];
                float fy0 = 1.0f - fy1;
                int32_t sy = _iy[dy];
                const uint16_t* ps0 = src + (sy + 0) * srcStride;
                const uint16_t* ps1 = src + (sy + 1) * srcStride;
                size_t dx = 0;
                float32x4_t _fy0 = vdupq_n_f32(fy0);
                float32x4_t _fy1 = vdupq_n_f32(fy1);
                if (N == 1)
                {
                    for (; dx < rs8; dx += 8)
                    {
                        float32x4_t fx01 = Load<false>(_ax.data + dx + 0);
                        float32x4_t fx00 = vsubq_f32(_1, fx01);
                        float32x4_t m00 = vmulq_f32(BilColS1(ps0, _ix.data + dx + 0, fx00, fx01), _fy0);
                        float32x4_t m01 = vmulq_f32(BilColS1(ps1, _ix.data + dx + 0, fx00, fx01), _fy1);
                        uint32x4_t i0 = (uint32x4_t)Round(vaddq_f32(m00, m01));
                        float32x4_t fx11 = Load<false>(_ax.data + dx + 4);
                        float32x4_t fx10 = vsubq_f32(_1, fx11);
                        float32x4_t m10 = vmulq_f32(BilColS1(ps0, _ix.data + dx + 4, fx10, fx11), _fy0);
                        float32x4_t m11 = vmulq_f32(BilColS1(ps1, _ix.data + dx + 4, fx10, fx11), _fy1);
                        uint32x4_t i1 = (uint32x4_t)Round(vaddq_f32(m10, m11));
                        Store<false>(dst + dx, PackU32(i0, i1));
                    }
                    for (; dx < rs4; dx += 4)
                    {
                        float32x4_t fx1 = Load<false>(_ax.data + dx);
                        float32x4_t fx0 = vsubq_f32(_1, fx1);
                        float32x4_t m0 = vmulq_f32(BilColS1(ps0, _ix.data + dx, fx0, fx1), _fy0);
                        float32x4_t m1 = vmulq_f32(BilColS1(ps1, _ix.data + dx, fx0, fx1), _fy1);
                        uint32x4_t i0 = (uint32x4_t)Round(vaddq_f32(m0, m1));
                        Store<false>(dst + dx, vmovn_u32(i0));
                    }
                }
                if (N == 2)
                {
                    for (; dx < rs8; dx += 8)
                    {
                        float32x4_t fx01 = Load<false>(_ax.data + dx + 0);
                        float32x4_t fx00 = vsubq_f32(_1, fx01);
                        float32x4_t m00 = vmulq_f32(BilColS2(ps0, _ix.data + dx + 0, fx00, fx01), _fy0);
                        float32x4_t m01 = vmulq_f32(BilColS2(ps1, _ix.data + dx + 0, fx00, fx01), _fy1);
                        uint32x4_t i0 = (uint32x4_t)Round(vaddq_f32(m00, m01));
                        float32x4_t fx11 = Load<false>(_ax.data + dx + 4);
                        float32x4_t fx10 = vsubq_f32(_1, fx11);
                        float32x4_t m10 = vmulq_f32(BilColS2(ps0, _ix.data + dx + 4, fx10, fx11), _fy0);
                        float32x4_t m11 = vmulq_f32(BilColS2(ps1, _ix.data + dx + 4, fx10, fx11), _fy1);
                        uint32x4_t i1 = (uint32x4_t)Round(vaddq_f32(m10, m11));
                        Store<false>(dst + dx, PackU32(i0, i1));
                    }
                    for (; dx < rs4; dx += 4)
                    {
                        float32x4_t fx1 = Load<false>(_ax.data + dx);
                        float32x4_t fx0 = vsubq_f32(_1, fx1);
                        float32x4_t m0 = vmulq_f32(BilColS2(ps0, _ix.data + dx, fx0, fx1), _fy0);
                        float32x4_t m1 = vmulq_f32(BilColS2(ps1, _ix.data + dx, fx0, fx1), _fy1);
                        uint32x4_t i0 = (uint32x4_t)Round(vaddq_f32(m0, m1));
                        Store<false>(dst + dx, vmovn_u32(i0));
                    }
                }
                if (N == 3)
                {
                    for (; dx < rs3; dx += 3)
                    {
                        float32x4_t fx1 = Load<false>(_ax.data + dx);
                        float32x4_t fx0 = vsubq_f32(_1, fx1);
                        float32x4_t m0 = vmulq_f32(BilColS3(ps0, _ix.data + dx, fx0, fx1), _fy0);
                        float32x4_t m1 = vmulq_f32(BilColS3(ps1, _ix.data + dx, fx0, fx1), _fy1);
                        uint32x4_t i0 = (uint32x4_t)Round(vaddq_f32(m0, m1));
                        Store<false>(dst + dx, vmovn_u32(i0));
                    }
                }
                if (N == 4)
                {
                    for (; dx < rs8; dx += 8)
                    {
                        float32x4_t fx01 = Load<false>(_ax.data + dx + 0);
                        float32x4_t fx00 = vsubq_f32(_1, fx01);
                        float32x4_t m00 = vmulq_f32(BilColS4(ps0, _ix.data + dx + 0, fx00, fx01), _fy0);
                        float32x4_t m01 = vmulq_f32(BilColS4(ps1, _ix.data + dx + 0, fx00, fx01), _fy1);
                        uint32x4_t i0 = (uint32x4_t)Round(vaddq_f32(m00, m01));
                        float32x4_t fx11 = Load<false>(_ax.data + dx + 4);
                        float32x4_t fx10 = vsubq_f32(_1, fx11);
                        float32x4_t m10 = vmulq_f32(BilColS4(ps0, _ix.data + dx + 4, fx10, fx11), _fy0);
                        float32x4_t m11 = vmulq_f32(BilColS4(ps1, _ix.data + dx + 4, fx10, fx11), _fy1);
                        uint32x4_t i1 = (uint32x4_t)Round(vaddq_f32(m10, m11));
                        Store<false>(dst + dx, PackU32(i0, i1));
                    }
                    for (; dx < rs4; dx += 4)
                    {
                        float32x4_t fx1 = Load<false>(_ax.data + dx);
                        float32x4_t fx0 = vsubq_f32(_1, fx1);
                        float32x4_t m0 = vmulq_f32(BilColS4(ps0, _ix.data + dx, fx0, fx1), _fy0);
                        float32x4_t m1 = vmulq_f32(BilColS4(ps1, _ix.data + dx, fx0, fx1), _fy1);
                        uint32x4_t i0 = (uint32x4_t)Round(vaddq_f32(m0, m1));
                        Store<false>(dst + dx, vmovn_u32(i0));
                    }
                }
                for (; dx < rs; dx++)
                {
                    int32_t sx = _ix[dx];
                    float fx1 = _ax[dx];
                    float fx0 = 1.0f - fx1;
                    float r0 = ps0[sx] * fx0 + ps0[sx + N] * fx1;
                    float r1 = ps1[sx] * fx0 + ps1[sx + N] * fx1;
                    dst[dx] = Simd::Round(r0 * fy0 + r1 * fy1);
                }
            }
        }

        void ResizerShortBilinear::Run(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            bool sparse = _param.dstH * 2.0 <= _param.srcH;
            switch (_param.channels)
            {
            case 1: sparse ? RunS<1>(src, srcStride, dst, dstStride) : RunB<1>(src, srcStride, dst, dstStride); return;
            case 2: sparse ? RunS<2>(src, srcStride, dst, dstStride) : RunB<2>(src, srcStride, dst, dstStride); return;
            case 3: sparse ? RunS<3>(src, srcStride, dst, dstStride) : RunB<3>(src, srcStride, dst, dstStride); return;
            case 4: sparse ? RunS<4>(src, srcStride, dst, dstStride) : RunB<4>(src, srcStride, dst, dstStride); return;
            default:
                assert(0);
            }
        }

        //---------------------------------------------------------------------

        ResizerFloatBilinear::ResizerFloatBilinear(const ResParam & param)
            : Base::ResizerFloatBilinear(param)
        {
        }

        void ResizerFloatBilinear::Run(const float * src, size_t srcStride, float * dst, size_t dstStride) 
        {
            size_t cn = _param.channels;
            size_t rs = _param.dstW * cn;
            float * pbx[2] = { _bx[0].data, _bx[1].data };
            int32_t prev = -2;
            size_t rsa = AlignLo(rs, F);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                float fy1 = _ay[dy];
                float fy0 = 1.0f - fy1;
                int32_t sy = _iy[dy];
                int32_t k = 0;

                if (sy == prev)
                    k = 2;
                else if (sy == prev + 1)
                {
                    Swap(pbx[0], pbx[1]);
                    k = 1;
                }

                prev = sy;

                for (; k < 2; k++)
                {
                    float * pb = pbx[k];
                    const float * ps = src + (sy + k)*srcStride;
                    size_t dx = 0;
                    if (cn == 1)
                    {
                        float32x4_t _1 = vdupq_n_f32(1.0f);
                        for (; dx < rsa; dx += F)
                        {
                            float32x4_t s01 = Load(ps + _ix[dx + 0], ps + _ix[dx + 1]);
                            float32x4_t s23 = Load(ps + _ix[dx + 2], ps + _ix[dx + 3]);
                            float32x4_t fx1 = Load<true>(_ax.data + dx);
                            float32x4_t fx0 = vsubq_f32(_1, fx1);
                            float32x4x2_t us = vuzpq_f32(s01, s23);
                            Store<true>(pb + dx, vmlaq_f32(vmulq_f32(us.val[0], fx0), us.val[1], fx1));
                        }
                    }
                    if (cn == 3 && rs > 3)
                    {
                        float32x4_t _1 = vdupq_n_f32(1.0f);
                        size_t rs3 = rs - 3;
                        for (; dx < rs3; dx += 3)
                        {
                            float32x4_t s0 = Load<false>(ps + _ix[dx] + 0);
                            float32x4_t s1 = Load<false>(ps + _ix[dx] + 3);
                            float32x4_t fx1 = vdupq_n_f32(_ax.data[dx]);
                            float32x4_t fx0 = vsubq_f32(_1, fx1);
                            Store<false>(pb + dx, vmlaq_f32(vmulq_f32(fx0, s0), fx1, s1));
                        }
                    }
                    for (; dx < rs; dx++)
                    {
                        int32_t sx = _ix[dx];
                        float fx = _ax[dx];
                        pb[dx] = ps[sx] * (1.0f - fx) + ps[sx + cn] * fx;
                    }
                }

                size_t dx = 0;
                float32x4_t _fy0 = vdupq_n_f32(fy0);
                float32x4_t _fy1 = vdupq_n_f32(fy1);
                for (; dx < rsa; dx += F)
                    Store<false>(dst + dx, vmlaq_f32(vmulq_f32(Load<true>(pbx[0] + dx), _fy0), Load<true>(pbx[1] + dx), _fy1));
                for (; dx < rs; dx++)
                    dst[dx] = pbx[0][dx] * fy0 + pbx[1][dx] * fy1;
            }
        }

        //---------------------------------------------------------------------

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method)
        {
            ResParam param(srcX, srcY, dstX, dstY, channels, type, method, sizeof(float32x4_t));
            if (param.IsByteBilinear() && dstX >= A)
                return new ResizerByteBilinear(param);
            else if (param.IsByteArea())
                return new ResizerByteArea(param);
            else if (param.IsShortBilinear() && channels == 1)
                return new ResizerShortBilinear(param);
            else if (param.IsFloatBilinear())
                return new ResizerFloatBilinear(param);
            else
                return Base::ResizerInit(srcX, srcY, dstX, dstY, channels, type, method);
        }
    }
#endif// SIMD_NEON_ENABLE
}
