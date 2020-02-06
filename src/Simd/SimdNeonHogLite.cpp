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
#include "Simd/SimdBase.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        const uint8x16_t K8_KX4 = SIMD_VEC_SETR_EPI8(1, 3, 5, 7, 7, 5, 3, 1, 1, 3, 5, 7, 7, 5, 3, 1);
        const uint8x16_t K8_KX8 = SIMD_VEC_SETR_EPI8(1, 3, 5, 7, 9, 11, 13, 15, 15, 13, 11, 9, 7, 5, 3, 1);

        SIMD_INLINE uint16x8_t Hadd16u(uint16x8_t a, uint16x8_t b)
        {
            return vcombine_u16(vpadd_u16(Half<0>(a), Half<1>(a)), vpadd_u16(Half<0>(b), Half<1>(b)));
        }

        SIMD_INLINE uint32x4_t Hadd32u(uint32x4_t a, uint32x4_t b)
        {
            return vcombine_u32(vpadd_u32(Half<0>(a), Half<1>(a)), vpadd_u32(Half<0>(b), Half<1>(b)));
        }

        SIMD_INLINE uint16x8_t Madd8u(uint8x16_t a, uint8x16_t b)
        {
            return Hadd16u(vmull_u8(Half<0>(a), Half<0>(b)), vmull_u8(Half<1>(a), Half<1>(b)));
        }

        SIMD_INLINE int32x4_t Madd16u(uint16x8_t a, uint16x8_t b)
        {
            return (int32x4_t)Hadd32u(vmull_u16(Half<0>(a), Half<0>(b)), vmull_u16(Half<1>(a), Half<1>(b)));
        }

        const uint8x8_t K8_I40 = SIMD_VEC_SETR_PI8(16, 17, 18, 19, 0, 1, 2, 3);
        const uint8x8_t K8_I51 = SIMD_VEC_SETR_PI8(20, 21, 22, 23, 4, 5, 6, 7);
        const uint8x8_t K8_I62 = SIMD_VEC_SETR_PI8(24, 25, 26, 27, 8, 9, 10, 11);
        const uint8x8_t K8_I73 = SIMD_VEC_SETR_PI8(28, 29, 30, 31, 12, 13, 14, 15);

        SIMD_INLINE float32x2_t Permute(float32x4x2_t v, uint8x8_t i)
        {
            return vreinterpret_f32_u8(vtbl4_u8(*(uint8x8x4_t*)&v, i));
        }

        SIMD_INLINE void UzpAs32(const uint16x8_t * src, uint16x8_t * dst)
        {
            *(uint32x4x2_t*)dst = vuzpq_u32(vreinterpretq_u32_u16(src[0]), vreinterpretq_u32_u16(src[1]));
        }

        template <size_t cell> class HogLiteFeatureExtractor
        {
            static const size_t FQ = 8;
            static const size_t HQ = FQ / 2;
            static const size_t DQ = FQ * 2;

            typedef Array<uint8_t> Bytes;
            typedef Array<int> Ints;
            typedef Array<float> Floats;

            size_t _hx, _fx, _w, _aw;
            Bytes _value, _index;
            Ints _hi[2];
            Floats _hf[2], _nf[4], _nb;
            int _k0[cell], _k1[cell];
            //__m128i _kx4, _kx8;
            float32x4_t _k, _02, _05, _02357, _eps;

            SIMD_INLINE void Init(size_t width)
            {
                _w = (width / cell - 1)*cell;
                _aw = AlignLo(_w, A);
                _hx = width / cell;
                _fx = _hx - 2;
                _value.Resize(_aw + 3 * A, true);
                _index.Resize(_aw + 3 * A, true);
                for (size_t i = 0; i < cell; ++i)
                {
                    _k0[i] = int(cell - i - 1) * 2 + 1;
                    _k1[i] = int(i) * 2 + 1;
                }
                for (size_t i = 0; i < 2; ++i)
                {
                    _hi[i].Resize((_hx + 4)*FQ, true);
                    _hf[i].Resize(_hx*FQ);
                }
                for (size_t i = 0; i < 4; ++i)
                    _nf[i].Resize(_hx + DF);
                _nb.Resize(_hx * 4);
                _k = vdupq_n_f32(1.0f / Simd::Square(cell * 2));
                _02 = vdupq_n_f32(0.2f);
                _05 = vdupq_n_f32(0.5f);
                _02357 = vdupq_n_f32(0.2357f);
                _eps = vdupq_n_f32(0.0001f);
            }

            template<bool align> static SIMD_INLINE void SetIndexAndValue(const uint8_t * src, size_t stride, uint8_t * value, uint8_t * index)
            {
                uint8x16_t y0 = Load<false>(src - stride);
                uint8x16_t y1 = Load<false>(src + stride);
                uint8x16_t x0 = Load<false>(src - 1);
                uint8x16_t x1 = Load<false>(src + 1);

                uint8x16_t ady = vabdq_u8(y0, y1);
                uint8x16_t adx = vabdq_u8(x0, x1);

                uint8x16_t max = vmaxq_u8(ady, adx);
                uint8x16_t min = vminq_u8(ady, adx);
                uint8x16_t val = vqaddq_u8(max, vrhaddq_u8(min, K8_00));
                Store<align>(value, val);

                uint8x16_t idx = vbslq_u8(Compare8u<SimdCompareGreater>(adx, ady), K8_00, K8_01);
                idx = vbslq_u8(Compare8u<SimdCompareGreater>(x1, x0), idx, vsubq_u8(K8_03, idx));
                idx = vbslq_u8(Compare8u<SimdCompareGreater>(y1, y0), idx, vsubq_u8(K8_07, idx));
                Store<align>(index, idx);
            }

            SIMD_INLINE void SetIndexAndValue(const uint8_t * src, size_t stride)
            {
                uint8_t * value = _value.data + A;
                uint8_t * index = _index.data + A;
                for (size_t col = 0; col < _aw; col += A)
                    SetIndexAndValue<true>(src + col, stride, value + col, index + col);
                if (_aw < _w)
                {
                    size_t col = _w - A;
                    SetIndexAndValue<false>(src + col, stride, value + col, index + col);
                }
            }

            static SIMD_INLINE void UpdateIntegerHistogram4x4(uint8_t * value, uint8_t * index, const uint16x8_t & ky0, const uint16x8_t & ky1, int * h0, int * h1)
            {
                uint8x16_t val = Load<false>(value);
                uint8x16_t idx = Load<false>(index);
                uint8x16_t cur0 = K8_00;
                uint8x16_t cur1 = K8_01;
                uint16x8_t dirs[4];
                for (size_t i = 0; i < 4; ++i)
                {
                    uint16x8_t dir0 = Madd8u(vandq_u8(vceqq_u8(idx, cur0), val), K8_KX4);
                    uint16x8_t dir1 = Madd8u(vandq_u8(vceqq_u8(idx, cur1), val), K8_KX4);
                    dirs[i] = Hadd16u(dir0, dir1);
                    cur0 = vqaddq_u8(cur0, K8_02);
                    cur1 = vqaddq_u8(cur1, K8_02);
                }
                UzpAs32(dirs + 0, dirs + 0);
                UzpAs32(dirs + 2, dirs + 2);
                Store<true>(h0 + 0 * F, vaddq_s32(Load<true>(h0 + 0 * F), Madd16u(dirs[0], ky0)));
                Store<true>(h0 + 1 * F, vaddq_s32(Load<true>(h0 + 1 * F), Madd16u(dirs[2], ky0)));
                Store<true>(h0 + 4 * F, vaddq_s32(Load<true>(h0 + 4 * F), Madd16u(dirs[1], ky0)));
                Store<true>(h0 + 5 * F, vaddq_s32(Load<true>(h0 + 5 * F), Madd16u(dirs[3], ky0)));
                Store<true>(h1 + 0 * F, vaddq_s32(Load<true>(h1 + 0 * F), Madd16u(dirs[0], ky1)));
                Store<true>(h1 + 1 * F, vaddq_s32(Load<true>(h1 + 1 * F), Madd16u(dirs[2], ky1)));
                Store<true>(h1 + 4 * F, vaddq_s32(Load<true>(h1 + 4 * F), Madd16u(dirs[1], ky1)));
                Store<true>(h1 + 5 * F, vaddq_s32(Load<true>(h1 + 5 * F), Madd16u(dirs[3], ky1)));
            }

            SIMD_INLINE void UpdateIntegerHistogram4x4(size_t rowI, size_t rowF)
            {
                int * h0 = _hi[(rowI + 0) & 1].data;
                int * h1 = _hi[(rowI + 1) & 1].data;
                uint8_t * value = _value.data + A - cell;
                uint8_t * index = _index.data + A - cell;
                uint16x8_t ky0 = vdupq_n_u16(_k0[rowF]);
                uint16x8_t ky1 = vdupq_n_u16(_k1[rowF]);
                for (size_t col = 0; col <= _w;)
                {
                    UpdateIntegerHistogram4x4(value + col, index + col, ky0, ky1, h0, h1);
                    col += cell;
                    h0 += FQ;
                    h1 += FQ;
                    UpdateIntegerHistogram4x4(value + col, index + col, ky0, ky1, h0, h1);
                    col += 3 * cell;
                    h0 += 3 * FQ;
                    h1 += 3 * FQ;
                }
            }

            SIMD_INLINE void UpdateIntegerHistogram8x8(size_t rowI, size_t rowF)
            {
                int * h0 = _hi[(rowI + 0) & 1].data;
                int * h1 = _hi[(rowI + 1) & 1].data;
                uint8_t * value = _value.data + A - cell;
                uint8_t * index = _index.data + A - cell;
                uint16x8_t ky0 = vdupq_n_u16(_k0[rowF]);
                uint16x8_t ky1 = vdupq_n_u16(_k1[rowF]);
                for (size_t col = 0; col <= _w; col += cell)
                {
                    uint8x16_t val = Load<false>(value + col);
                    uint8x16_t idx = Load<false>(index + col);
                    uint8x16_t cur0 = K8_00;
                    uint8x16_t cur1 = K8_01;
                    uint16x8_t dirs[4];
                    for (size_t i = 0; i < 4; ++i)
                    {
                        uint16x8_t dir0 = Madd8u(vandq_u8(vceqq_u8(idx, cur0), val), K8_KX8);
                        uint16x8_t dir1 = Madd8u(vandq_u8(vceqq_u8(idx, cur1), val), K8_KX8);
                        dirs[i] = Hadd16u(dir0, dir1);
                        cur0 = vqaddq_u8(cur0, K8_02);
                        cur1 = vqaddq_u8(cur1, K8_02);
                    }
                    dirs[0] = Hadd16u(dirs[0], dirs[1]);
                    dirs[1] = Hadd16u(dirs[2], dirs[3]);
                    Store<true>(h0 + 0, vaddq_s32(Load<true>(h0 + 0), Madd16u(dirs[0], ky0)));
                    Store<true>(h0 + F, vaddq_s32(Load<true>(h0 + F), Madd16u(dirs[1], ky0)));
                    Store<true>(h1 + 0, vaddq_s32(Load<true>(h1 + 0), Madd16u(dirs[0], ky1)));
                    Store<true>(h1 + F, vaddq_s32(Load<true>(h1 + F), Madd16u(dirs[1], ky1)));
                    h0 += FQ;
                    h1 += FQ;
                }
            }

            SIMD_INLINE void UpdateFloatHistogram(size_t rowI)
            {
                Ints & hi = _hi[rowI & 1];
                Floats & hf = _hf[rowI & 1];
                Floats & nf = _nf[rowI & 3];

                for (size_t i = 0; i < hf.size; i += DF)
                {
                    Store<true>(hf.data + i + 0, vmulq_f32(_k, vcvtq_f32_s32(Load<true>(hi.data + i + 0))));
                    Store<true>(hf.data + i + F, vmulq_f32(_k, vcvtq_f32_s32(Load<true>(hi.data + i + F))));
                }
                hi.Clear();

                const float * h = hf.data;
                for (size_t x = 0; x < _hx; ++x, h += FQ)
                {
                    float32x4_t h0 = Load<true>(h + 00);
                    float32x4_t h1 = Load<true>(h + HQ);
                    float32x4_t s1 = vaddq_f32(h0, h1);
                    float32x4_t s2 = vmulq_f32(s1, s1);
                    nf.data[x] = ExtractSum32f(s2);
                }
            }

            SIMD_INLINE void BlockNorm(size_t rowI)
            {
                const float * src0 = _nf[(rowI - 2) & 3].data;
                const float * src1 = _nf[(rowI - 1) & 3].data;
                const float * src2 = _nf[(rowI - 0) & 3].data;
                float * dst = _nb.data;
                for (size_t x = 0; x < _fx; x += 3, src0 += 3, src1 += 3, src2 += 3, dst += 3 * F)
                {
                    float32x4_t s00 = Load<false>(src0 + 0);
                    float32x4_t s01 = Load<false>(src0 + 1);
                    float32x4_t s10 = Load<false>(src1 + 0);
                    float32x4_t s11 = Load<false>(src1 + 1);
                    float32x4_t s20 = Load<false>(src2 + 0);
                    float32x4_t s21 = Load<false>(src2 + 1);
                    float32x4_t v00 = vaddq_f32(s00, s10);
                    float32x4_t v01 = vaddq_f32(s01, s11);
                    float32x4_t v10 = vaddq_f32(s10, s20);
                    float32x4_t v11 = vaddq_f32(s11, s21);
                    float32x4x2_t h;
                    h.val[0] = Hadd(v00, v01);
                    h.val[1] = Hadd(v10, v11);
                    float32x2_t p40 = Permute(h, K8_I40);
                    float32x2_t p51 = Permute(h, K8_I51);
                    float32x2_t p62 = Permute(h, K8_I62);
                    float32x2_t p73 = Permute(h, K8_I73);
                    Store<true>(dst + 0 * HF, p62);
                    Store<true>(dst + 1 * HF, p40);
                    Store<true>(dst + 2 * HF, p51);
                    Store<true>(dst + 3 * HF, p62);
                    Store<true>(dst + 4 * HF, p73);
                    Store<true>(dst + 5 * HF, p51);
                }
            }

            SIMD_INLINE void SetFeatures(size_t rowI, float * dst)
            {
                const float * hf = _hf[(rowI - 1) & 1].data + FQ;
                const float * nb = _nb.data;
                for (size_t x = 0; x < _fx; ++x, nb += 4)
                {
                    float32x4_t n = ReciprocalSqrt<1>(vaddq_f32(Load<true>(nb), _eps));
                    float32x4_t t = vdupq_n_f32(0.0f);
                    const float * src = hf + x * FQ;
                    for (int o = 0; o < FQ; o += 4)
                    {
                        float32x4_t s = Load<false>(src);
                        float32x4_t h0 = vminq_f32(vmulq_f32(Broadcast<0>(s), n), _02);
                        float32x4_t h1 = vminq_f32(vmulq_f32(Broadcast<1>(s), n), _02);
                        float32x4_t h2 = vminq_f32(vmulq_f32(Broadcast<2>(s), n), _02);
                        float32x4_t h3 = vminq_f32(vmulq_f32(Broadcast<3>(s), n), _02);
                        t = vaddq_f32(t, vaddq_f32(vaddq_f32(h0, h1), vaddq_f32(h2, h3)));
                        Store<false>(dst, vmulq_f32(_05, Hadd(Hadd(h0, h1), Hadd(h2, h3))));
                        dst += F;
                        src += F;
                    }
                    src = hf + x * FQ;
                    float32x4_t s = vaddq_f32(Load<false>(src), Load<false>(src + HQ));
                    float32x4_t h0 = vminq_f32(vmulq_f32(Broadcast<0>(s), n), _02);
                    float32x4_t h1 = vminq_f32(vmulq_f32(Broadcast<1>(s), n), _02);
                    float32x4_t h2 = vminq_f32(vmulq_f32(Broadcast<2>(s), n), _02);
                    float32x4_t h3 = vminq_f32(vmulq_f32(Broadcast<3>(s), n), _02);
                    Store<false>(dst, vmulq_f32(_05, Hadd(Hadd(h0, h1), Hadd(h2, h3))));
                    dst += 4;
                    Store<false>(dst, vmulq_f32(t, _02357));
                    dst += 4;
                }
            }

        public:

            void Run(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * features, size_t featuresStride)
            {
                assert(cell == 8 || cell == 4);
                assert(width >= cell * 3 && height >= cell * 3);

                Init(width);

                src += (srcStride + 1)*cell / 2;
                height = (height / cell - 1)*cell;

                for (size_t row = 0; row < height; ++row)
                {
                    SetIndexAndValue(src, srcStride);
                    size_t rowI = row / cell;
                    size_t rowF = row & (cell - 1);
                    if (cell == 4)
                        UpdateIntegerHistogram4x4(rowI, rowF);
                    else
                        UpdateIntegerHistogram8x8(rowI, rowF);
                    if (rowF == cell - 1)
                    {
                        UpdateFloatHistogram(rowI);
                        if (rowI >= 2)
                        {
                            BlockNorm(rowI);
                            SetFeatures(rowI, features);
                            features += featuresStride;
                        }
                    }
                    src += srcStride;
                }
                size_t rowI = height / cell;
                UpdateFloatHistogram(rowI);
                BlockNorm(rowI);
                SetFeatures(rowI, features);
            }
        };

        void HogLiteExtractFeatures(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t cell, float * features, size_t featuresStride)
        {
            if (cell == 4)
            {
                HogLiteFeatureExtractor<4> extractor;
                extractor.Run(src, srcStride, width, height, features, featuresStride);
            }
            else
            {
                HogLiteFeatureExtractor<8> extractor;
                extractor.Run(src, srcStride, width, height, features, featuresStride);
            }
        }

        class HogLiteFeatureFilter
        {
            template<bool align> SIMD_INLINE void ProductSum1x1(const float * src, const float * filter, float32x4_t & sum)
            {
                float32x4_t _src = Load<align>(src);
                float32x4_t _filter = Load<align>(filter);
                sum = vmlaq_f32(sum, _src, _filter);
            }

            template<bool align, size_t step> SIMD_INLINE void ProductSum1x4(const float * src, const float * filter, float32x4_t * sums)
            {
                float32x4_t _filter = Load<align>(filter);
                sums[0] = vmlaq_f32(sums[0], Load<align>(src + 0 * step), _filter);
                sums[1] = vmlaq_f32(sums[1], Load<align>(src + 1 * step), _filter);
                sums[2] = vmlaq_f32(sums[2], Load<align>(src + 2 * step), _filter);
                sums[3] = vmlaq_f32(sums[3], Load<align>(src + 3 * step), _filter);
            }

            template <bool align, size_t featureSize> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, const float * filter, size_t filterWidth, size_t filterHeight, float * dst, size_t dstStride)
            {
                size_t filterStride = featureSize * filterWidth;
                size_t alignedDstWidth = AlignLo(dstWidth, 4);
                for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                {
                    size_t dstCol = 0;
                    for (; dstCol < alignedDstWidth; dstCol += 4)
                    {
                        float32x4_t sums[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
                        const float * pSrc = src + dstRow * srcStride + dstCol * featureSize;
                        const float * pFilter = filter;
                        for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                        {
                            size_t filterCol = 0;
                            for (; filterCol < filterStride; filterCol += F)
                                ProductSum1x4<align, featureSize>(pSrc + filterCol, pFilter + filterCol, sums);
                            pSrc += srcStride;
                            pFilter += filterStride;
                        }
                        Store<false>(dst + dstCol, Extract4Sums(sums));
                    }
                    for (; dstCol < dstWidth; ++dstCol)
                    {
                        float32x4_t sum = vdupq_n_f32(0.0f);
                        const float * pSrc = src + dstRow * srcStride + dstCol * featureSize;
                        const float * pFilter = filter;
                        for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                        {
                            for (size_t filterCol = 0; filterCol < filterStride; filterCol += F)
                                ProductSum1x1<align>(pSrc + filterCol, pFilter + filterCol, sum);
                            pSrc += srcStride;
                            pFilter += filterStride;
                        }
                        dst[dstCol] = ExtractSum32f(sum);
                    }
                    dst += dstStride;
                }
            }

            template <bool align, size_t featureSize> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
            {
                size_t filterStride = featureSize * filterWidth;
                size_t alignedDstWidth = AlignLo(dstWidth, 4);
                float32x4_t _min = vdupq_n_f32(-FLT_MAX);
                for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                {
                    size_t dstCol = 0;
                    for (; dstCol < alignedDstWidth; dstCol += 4)
                    {
                        uint32x4_t _mask = Load<false>(mask + dstCol);
                        if (TestZ(_mask))
                            Store<false>(dst + dstCol, _min);
                        else
                        {
                            float32x4_t sums[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
                            const float * pSrc = src + dstRow * srcStride + dstCol * featureSize;
                            const float * pFilter = filter;
                            for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                            {
                                size_t filterCol = 0;
                                for (; filterCol < filterStride; filterCol += F)
                                    ProductSum1x4<align, featureSize>(pSrc + filterCol, pFilter + filterCol, sums);
                                pSrc += srcStride;
                                pFilter += filterStride;
                            }
                            Store<false>(dst + dstCol, vbslq_f32(_mask, Extract4Sums(sums), _min));
                        }
                    }
                    for (; dstCol < dstWidth; ++dstCol)
                    {
                        if (mask[dstCol])
                        {
                            float32x4_t sum = vdupq_n_f32(0.0f);
                            const float * pSrc = src + dstRow * srcStride + dstCol * featureSize;
                            const float * pFilter = filter;
                            for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                            {
                                for (size_t filterCol = 0; filterCol < filterStride; filterCol += F)
                                    ProductSum1x1<align>(pSrc + filterCol, pFilter + filterCol, sum);
                                pSrc += srcStride;
                                pFilter += filterStride;
                            }
                            dst[dstCol] = ExtractSum32f(sum);
                        }
                        else
                            dst[dstCol] = -FLT_MAX;
                    }
                    dst += dstStride;
                    mask += maskStride;
                }
            }

            template <bool align> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, float * dst, size_t dstStride)
            {
                if (featureSize == 16)
                    Filter<align, 16>(src, srcStride, dstWidth, dstHeight, filter, filterWidth, filterHeight, dst, dstStride);
                else
                    Filter<align, 8>(src, srcStride, dstWidth, dstHeight, filter, filterWidth, filterHeight, dst, dstStride);
            }

            template <bool align> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
            {
                if (featureSize == 16)
                    Filter<align, 16>(src, srcStride, dstWidth, dstHeight, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
                else
                    Filter<align, 8>(src, srcStride, dstWidth, dstHeight, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
            }

        public:

            void Run(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
            {
                assert(featureSize == 8 || featureSize == 16);
                assert(srcWidth >= filterWidth && srcHeight >= filterHeight);

                size_t dstWidth = srcWidth - filterWidth + 1;
                size_t dstHeight = srcHeight - filterHeight + 1;

                if (mask)
                {
                    if (Aligned(src) && Aligned(srcStride) && Aligned(filter))
                        Filter<true>(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
                    else
                        Filter<false>(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
                }
                else
                {
                    if (Aligned(src) && Aligned(srcStride) && Aligned(filter))
                        Filter<true>(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterWidth, filterHeight, dst, dstStride);
                    else
                        Filter<false>(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterWidth, filterHeight, dst, dstStride);
                }
            }
        };

        void HogLiteFilterFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
        {
            HogLiteFeatureFilter featureFilter;
            featureFilter.Run(src, srcStride, srcWidth, srcHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
        }

        namespace HogLiteFeatureResizerDetail
        {
            template <int size> struct Feature
            {
                template <bool align> static SIMD_INLINE void Interpolate(const float * src0, const float * src1, const float32x4_t k[2][2], float * dst);
            };

            template <> struct Feature<8>
            {
                template <bool align> static SIMD_INLINE void Interpolate(const float * src0, const float * src1, const float32x4_t k[2][2], float * dst)
                {
                    Store<align>(dst + 0 * F, vaddq_f32(
                        vmlaq_f32(vmulq_f32(Load<align>(src0 + 0 * F), k[0][0]), Load<align>(src0 + 2 * F), k[0][1]),
                        vmlaq_f32(vmulq_f32(Load<align>(src1 + 0 * F), k[1][0]), Load<align>(src1 + 2 * F), k[1][1])));
                    Store<align>(dst + 1 * F, vaddq_f32(
                        vmlaq_f32(vmulq_f32(Load<align>(src0 + 1 * F), k[0][0]), Load<align>(src0 + 3 * F), k[0][1]),
                        vmlaq_f32(vmulq_f32(Load<align>(src1 + 1 * F), k[1][0]), Load<align>(src1 + 3 * F), k[1][1])));
                }
            };

            template <> struct Feature<16>
            {
                template <bool align> static SIMD_INLINE void Interpolate(const float * src0, const float * src1, const float32x4_t k[2][2], float * dst)
                {
                    Store<align>(dst + 0 * F, vaddq_f32(
                        vmlaq_f32(vmulq_f32(Load<align>(src0 + 0 * F), k[0][0]), Load<align>(src0 + 4 * F), k[0][1]),
                        vmlaq_f32(vmulq_f32(Load<align>(src1 + 0 * F), k[1][0]), Load<align>(src1 + 4 * F), k[1][1])));
                    Store<align>(dst + 1 * F, vaddq_f32(
                        vmlaq_f32(vmulq_f32(Load<align>(src0 + 1 * F), k[0][0]), Load<align>(src0 + 5 * F), k[0][1]),
                        vmlaq_f32(vmulq_f32(Load<align>(src1 + 1 * F), k[1][0]), Load<align>(src1 + 5 * F), k[1][1])));
                    Store<align>(dst + 2 * F, vaddq_f32(
                        vmlaq_f32(vmulq_f32(Load<align>(src0 + 2 * F), k[0][0]), Load<align>(src0 + 6 * F), k[0][1]),
                        vmlaq_f32(vmulq_f32(Load<align>(src1 + 2 * F), k[1][0]), Load<align>(src1 + 6 * F), k[1][1])));
                    Store<align>(dst + 3 * F, vaddq_f32(
                        vmlaq_f32(vmulq_f32(Load<align>(src0 + 3 * F), k[0][0]), Load<align>(src0 + 7 * F), k[0][1]),
                        vmlaq_f32(vmulq_f32(Load<align>(src1 + 3 * F), k[1][0]), Load<align>(src1 + 7 * F), k[1][1])));
                }
            };
        }

        class HogLiteFeatureResizer
        {
            typedef Array<int> Ints;
            typedef Array<float> Floats;

            Ints _iy, _ix;
            Floats _ky, _kx;

            void InitIndexWeight(size_t srcSize, size_t dstSize, size_t dstStep, Ints & indexes, Floats & weights)
            {
                indexes.Resize(dstSize);
                weights.Resize(dstSize);

                float scale = float(srcSize) / float(dstSize);
                for (size_t i = 0; i < dstSize; ++i)
                {
                    float weight = (float)((i + 0.5f)*scale - 0.5f);
                    int index = (int)::floor(weight);
                    weight -= index;
                    if (index < 0)
                    {
                        index = 0;
                        weight = 0.0f;
                    }
                    if (index > (int)srcSize - 2)
                    {
                        index = (int)srcSize - 2;
                        weight = 1.0f;
                    }
                    indexes[i] = int(index*dstStep);
                    weights[i] = weight;
                }
            }

            template<bool align, size_t featureSize> void Resize(const float * src, size_t srcStride, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
            {
                float32x4_t _1 = vdupq_n_f32(1.0f);
                for (size_t rowDst = 0; rowDst < dstHeight; ++rowDst)
                {
                    float32x4_t ky1 = vdupq_n_f32(_ky[rowDst]);
                    float32x4_t ky0 = vsubq_f32(_1, ky1);
                    const float * pSrc = src + _iy[rowDst];
                    float * pDst = dst + rowDst * dstStride;
                    for (size_t colDst = 0; colDst < dstWidth; ++colDst, pDst += featureSize)
                    {
                        float32x4_t kx1 = vdupq_n_f32(_kx[colDst]);
                        float32x4_t kx0 = vsubq_f32(_1, kx1);
                        float32x4_t k[2][2];
                        k[0][0] = vmulq_f32(ky0, kx0);
                        k[0][1] = vmulq_f32(ky0, kx1);
                        k[1][0] = vmulq_f32(ky1, kx0);
                        k[1][1] = vmulq_f32(ky1, kx1);
                        const float * pSrc0 = pSrc + _ix[colDst];
                        const float * pSrc1 = pSrc0 + srcStride;
                        HogLiteFeatureResizerDetail::Feature<featureSize>:: template Interpolate<align>(pSrc0, pSrc1, k, pDst);
                    }
                }
            }

            template<bool align> void Resize(const float * src, size_t srcStride, size_t featureSize, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
            {
                if (featureSize == 8)
                    Resize<align, 8>(src, srcStride, dst, dstStride, dstWidth, dstHeight);
                else
                    Resize<align, 16>(src, srcStride, dst, dstStride, dstWidth, dstHeight);
            }

        public:
            void Run(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
            {
                assert(featureSize == 8 || featureSize == 16);

                if (srcWidth == dstWidth && srcHeight == dstHeight)
                {
                    size_t size = sizeof(float)*srcWidth*featureSize;
                    for (size_t row = 0; row < dstHeight; ++row)
                        memcpy(dst + row * dstStride, src + row * srcStride, size);
                    return;
                }

                InitIndexWeight(srcWidth, dstWidth, featureSize, _ix, _kx);
                InitIndexWeight(srcHeight, dstHeight, srcStride, _iy, _ky);

                if (Aligned(src) && Aligned(dst))
                    Resize<true>(src, srcStride, featureSize, dst, dstStride, dstWidth, dstHeight);
                else
                    Resize<false>(src, srcStride, featureSize, dst, dstStride, dstWidth, dstHeight);
            }
        };

        void HogLiteResizeFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
        {
            HogLiteFeatureResizer featureResizer;
            featureResizer.Run(src, srcStride, srcWidth, srcHeight, featureSize, dst, dstStride, dstWidth, dstHeight);
        }

        template<bool align> void HogLiteCompressFeatures(const float * src, size_t srcStride, size_t width, size_t height, const float * pca, float * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                const float * s = src;
                float * d = dst;
                for (size_t col = 0; col < width; ++col)
                {
                    const float * p = pca;
                    for (size_t i = 0; i < 8; i += 4, p += 64)
                    {
                        float32x4_t sums[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
                        for (size_t j = 0; j < 16; j += F)
                        {
                            float32x4_t _s = Load<align>(s + j);
                            sums[0] = vmlaq_f32(sums[0], _s, Load<align>(p + j + 00));
                            sums[1] = vmlaq_f32(sums[1], _s, Load<align>(p + j + 16));
                            sums[2] = vmlaq_f32(sums[2], _s, Load<align>(p + j + 32));
                            sums[3] = vmlaq_f32(sums[3], _s, Load<align>(p + j + 48));
                        }
                        Store<align>(d + i, Extract4Sums(sums));
                    }
                    s += 16;
                    d += 8;
                }
                src += srcStride;
                dst += dstStride;
            }

        }

        void HogLiteCompressFeatures(const float * src, size_t srcStride, size_t width, size_t height, const float * pca, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(pca) && Aligned(dst))
                HogLiteCompressFeatures<true>(src, srcStride, width, height, pca, dst, dstStride);
            else
                HogLiteCompressFeatures<false>(src, srcStride, width, height, pca, dst, dstStride);
        }

        class HogLiteSeparableFilter
        {
            size_t _dstWidth, _dstHeight, _dstStride;
            Array32f _buffer;
            Array128f _filter;

            void Init(size_t srcWidth, size_t srcHeight, size_t hSize, size_t vSize)
            {
                _dstWidth = srcWidth - hSize + 1;
                _dstStride = AlignHi(_dstWidth, F);
                _dstHeight = srcHeight - vSize + 1;
                _buffer.Resize(_dstStride*srcHeight);
            }

            template<bool align> static SIMD_INLINE void FilterHx1(const float * src, const float * filter, float32x4_t & sum)
            {
                float32x4_t _src = Load<align>(src);
                float32x4_t _filter = Load<align>(filter);
                sum = vmlaq_f32(sum, _src, _filter);
            }

            template<bool align, size_t step> static SIMD_INLINE void FilterHx4(const float * src, const float * filter, float32x4_t * sums)
            {
                float32x4_t _filter = Load<align>(filter);
                sums[0] = vmlaq_f32(sums[0], Load<align>(src + 0 * step), _filter);
                sums[1] = vmlaq_f32(sums[1], Load<align>(src + 1 * step), _filter);
                sums[2] = vmlaq_f32(sums[2], Load<align>(src + 2 * step), _filter);
                sums[3] = vmlaq_f32(sums[3], Load<align>(src + 3 * step), _filter);
            }

            template <bool align, size_t step> void FilterH(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                size_t alignedWidth = AlignLo(width, 4);
                for (size_t row = 0; row < height; ++row)
                {
                    size_t col = 0;
                    for (; col < alignedWidth; col += 4)
                    {
                        float32x4_t sums[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
                        const float * s = src + col * step;
                        for (size_t i = 0; i < size; i += F)
                            FilterHx4<align, step>(s + i, filter + i, sums);
                        Store<true>(dst + col, Extract4Sums(sums));
                    }
                    for (; col < width; ++col)
                    {
                        float32x4_t sum = vdupq_n_f32(0);
                        const float * s = src + col * step;
                        for (size_t i = 0; i < size; i += F)
                            FilterHx1<align>(s + i, filter + i, sum);
                        dst[col] = ExtractSum32f(sum);
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template <bool align> void FilterH(const float * src, size_t srcStride, size_t width, size_t height, size_t step, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                if (step == 16)
                    FilterH<align, 16>(src, srcStride, width, height, filter, size, dst, dstStride);
                else
                    FilterH<align, 8>(src, srcStride, width, height, filter, size, dst, dstStride);
            }

            template <bool srcAlign, bool dstAlign, UpdateType update, bool masked> static SIMD_INLINE void FilterV(const float * src, size_t stride, const float32x4_t * filter, size_t size, float * dst, const float32x4_t & mask)
            {
                float32x4_t sum = vdupq_n_f32(0);
                for (size_t i = 0; i < size; ++i, src += stride)
                    sum = vmlaq_f32(sum, Load<srcAlign>(src), filter[i]);
                Update<update, dstAlign>(dst, Masked<masked && update != UpdateSet>(sum, mask));
            }

            template <UpdateType update, bool align> void FilterV(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = vdupq_n_f32(filter[i]);

                size_t alignedWidth = AlignLo(width, F);
                float32x4_t tailMask = RightNotZero32f(width - alignedWidth);

                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += F)
                        FilterV<true, align, update, false>(src + col, srcStride, _filter.data, size, dst + col, tailMask);
                    if (alignedWidth != width)
                        FilterV<false, false, update, true>(src + width - F, srcStride, _filter.data, size, dst + width - F, tailMask);
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template <UpdateType update> void FilterV(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                if (Aligned(dst) && Aligned(dstStride))
                    FilterV<update, true>(src, srcStride, width, height, filter, size, dst, dstStride);
                else
                    FilterV<update, false>(src, srcStride, width, height, filter, size, dst, dstStride);
            }

        public:

            void Run(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * hFilter, size_t hSize, const float * vFilter, size_t vSize, float * dst, size_t dstStride, int add)
            {
                assert(featureSize == 8 || featureSize == 16);
                assert(srcWidth >= hSize && srcHeight >= vSize);

                Init(srcWidth, srcHeight, hSize, vSize);

                if (Aligned(src) && Aligned(srcStride) && Aligned(hFilter))
                    FilterH<true>(src, srcStride, _dstWidth, srcHeight, featureSize, hFilter, hSize*featureSize, _buffer.data, _dstStride);
                else
                    FilterH<false>(src, srcStride, _dstWidth, srcHeight, featureSize, hFilter, hSize*featureSize, _buffer.data, _dstStride);

                if (add)
                    FilterV<UpdateAdd>(_buffer.data, _dstStride, _dstWidth, _dstHeight, vFilter, vSize, dst, dstStride);
                else
                    FilterV<UpdateSet>(_buffer.data, _dstStride, _dstWidth, _dstHeight, vFilter, vSize, dst, dstStride);
            }
        };

        void HogLiteFilterSeparable(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * hFilter, size_t hSize, const float * vFilter, size_t vSize, float * dst, size_t dstStride, int add)
        {
            HogLiteSeparableFilter filter;
            filter.Run(src, srcStride, srcWidth, srcHeight, featureSize, hFilter, hSize, vFilter, vSize, dst, dstStride, add);
        }

        void HogLiteFindMax7x7(const float * a, size_t aStride, const float * b, size_t bStride, size_t height, float * pValue, size_t * pCol, size_t * pRow)
        {
            float32x4_t max = vdupq_n_f32(-FLT_MAX), val;
            uint32x4_t idx = vdupq_n_u32(0);
            uint32x4_t cur = K32_0123;
            for (size_t row = 0; row < height; ++row)
            {
                val = vaddq_f32(Load<false>(a + 0), Load<false>(b + 0));
                idx = vbslq_u32(vcgtq_f32(val, max), cur, idx);
                max = vmaxq_f32(max, val);
                cur = vaddq_u32(cur, K32_00000003);
                val = vaddq_f32(Load<false>(a + 3), Load<false>(b + 3));
                idx = vbslq_u32(vcgtq_f32(val, max), cur, idx);
                max = vmaxq_f32(max, val);
                cur = vaddq_u32(cur, K32_00000005);
                a += aStride;
                b += bStride;
            }

            uint32_t _idx[F];
            float _max[F];
            Store<false>(_max, max);
            Store<false>(_idx, idx);
            *pValue = -FLT_MAX;
            for (size_t i = 0; i < F; ++i)
            {
                if (_max[i] > *pValue)
                {
                    *pValue = _max[i];
                    *pCol = _idx[i] & 7;
                    *pRow = _idx[i] / 8;
                }
                else if (_max[i] == *pValue && *pRow > _idx[i] / 8)
                {
                    *pCol = _idx[i] & 7;
                    *pRow = _idx[i] / 8;
                }
            }
        }

        SIMD_INLINE void Fill7x7(uint32_t * dst, size_t stride)
        {
            for (size_t row = 0; row < 7; ++row)
            {
                Store<false>(dst + 0, K32_FFFFFFFF);
                Store<false>(dst + 3, K32_FFFFFFFF);
                dst += stride;
            }
        }

        template <size_t scale> void HogLiteCreateMask7x7(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, uint32_t * dst, size_t dstStride)
        {
            size_t dstStartEnd = 7 - scale;
            size_t dstRowSize = (srcWidth*scale + 7 - scale) * sizeof(uint32_t);
            for (size_t dstRow = 0; dstRow < dstStartEnd; ++dstRow)
                memset(dst + dstRow * dstStride, 0, dstRowSize);

            size_t alignedSrcWidth = AlignLo(srcWidth, F);
            float32x4_t _threshold = vdupq_n_f32(*threshold);
            for (size_t srcRow = 0; srcRow < srcHeight; ++srcRow)
            {
                for (size_t dstRow = 0; dstRow < scale; ++dstRow)
                    memset(dst + (dstStartEnd + dstRow)*dstStride, 0, dstRowSize);

                size_t srcCol = 0;
                for (; srcCol < alignedSrcWidth; srcCol += F)
                {
                    uint32x4_t mask = vcgtq_f32(Load<false>(src + srcCol), _threshold);
                    uint32_t * pDst = dst + srcCol * scale;
                    if (vgetq_lane_u32(mask, 0))
                        Fill7x7(pDst + 0 * scale, dstStride);
                    if (vgetq_lane_u32(mask, 1))
                        Fill7x7(pDst + 1 * scale, dstStride);
                    if (vgetq_lane_u32(mask, 2))
                        Fill7x7(pDst + 2 * scale, dstStride);
                    if (vgetq_lane_u32(mask, 3))
                        Fill7x7(pDst + 3 * scale, dstStride);
                }
                for (; srcCol < srcWidth; ++srcCol)
                {
                    if (src[srcCol] > *threshold)
                        Fill7x7(dst + srcCol * scale, dstStride);
                }
                src += srcStride;
                dst += dstStride * scale;
            }
        }

        void HogLiteCreateMask(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, size_t scale, size_t size, uint32_t * dst, size_t dstStride)
        {
            if (scale == 1 && size == 7)
                HogLiteCreateMask7x7<1>(src, srcStride, srcWidth, srcHeight, threshold, dst, dstStride);
            else if (scale == 2 && size == 7)
                HogLiteCreateMask7x7<2>(src, srcStride, srcWidth, srcHeight, threshold, dst, dstStride);
            else
                Base::HogLiteCreateMask(src, srcStride, srcWidth, srcHeight, threshold, scale, size, dst, dstStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}
