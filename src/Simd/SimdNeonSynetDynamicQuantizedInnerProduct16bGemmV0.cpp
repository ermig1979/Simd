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
#include "Simd/SimdSynetDynamicQuantizedInnerProduct16b.h"
#include "Simd/SimdSynetDynamicQuantizedActivation.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdShuffle.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Neon
    {
        typedef Simd::DynamicQuantizedInnerProduct16bParam DqipParam;
        typedef Base::SynetDynamicQuantizedInnerProduct16bGemmV0::AlgParam AlgParam;
        typedef Base::SynetDynamicQuantizedInnerProduct16bGemmV0::GemmLastPtr GemmLastPtr;

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE float ReduceMin(float32x4_t value)
        {
            float32x2_t half = vpmin_f32(vget_low_f32(value), vget_high_f32(value));
            return vget_lane_f32(vpmin_f32(half, half), 0);
        }

        SIMD_INLINE float ReduceMax(float32x4_t value)
        {
            float32x2_t half = vpmax_f32(vget_low_f32(value), vget_high_f32(value));
            return vget_lane_f32(vpmax_f32(half, half), 0);
        }

        SIMD_INLINE float32x4_t LoadTail(const float* ptr, size_t size)
        {
            float buf[F] = { 0, 0, 0, 0 };
            for (size_t i = 0; i < size; ++i)
                buf[i] = ptr[i];
            return vld1q_f32(buf);
        }

        SIMD_INLINE void StoreTail(int32_t* ptr, int32x4_t val, size_t size)
        {
            int32_t buf[F];
            vst1q_s32(buf, val);
            for (size_t i = 0; i < size; ++i)
                ptr[i] = buf[i];
        }

        SIMD_INLINE int32x4_t QuantizeAf(float32x4_t src, float32x4_t norm, int32x4_t zero)
        {
            return vaddq_s32(NearbyInt(vmulq_f32(src, norm)), zero);
        }

        SIMD_INLINE void QuantizeAf1(const float* src, float32x4_t norm, int32x4_t zero, uint8_t* dst)
        {
            int32x4_t i32 = QuantizeAf(vdupq_n_f32(src[0]), norm, zero);
            dst[0] = vget_lane_u8(vqmovun_s16(vcombine_s16(vqmovn_s32(i32), vdup_n_s16(0))), 0);
        }

        SIMD_INLINE void QuantizeAf4(const float* src, float32x4_t norm, int32x4_t zero, uint8_t* dst)
        {
            int32x4_t i32 = QuantizeAf(vld1q_f32(src), norm, zero);
            uint8x8_t u8 = vqmovun_s16(vcombine_s16(vqmovn_s32(i32), vdup_n_s16(0)));
            vst1_lane_u32((uint32_t*)dst, vreinterpret_u32_u8(u8), 0);
        }

        SIMD_INLINE void QuantizeAf16(const float* src, float32x4_t norm, int32x4_t zero, uint8_t* dst)
        {
            int32x4_t i0 = QuantizeAf(vld1q_f32(src + 0 * F), norm, zero);
            int32x4_t i1 = QuantizeAf(vld1q_f32(src + 1 * F), norm, zero);
            int32x4_t i2 = QuantizeAf(vld1q_f32(src + 2 * F), norm, zero);
            int32x4_t i3 = QuantizeAf(vld1q_f32(src + 3 * F), norm, zero);
            vst1q_u8(dst, vcombine_u8(
                vqmovun_s16(vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1))),
                vqmovun_s16(vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3)))));
        }

        void Dqip16bGemmV0_ConvertAf(const uint8_t* src8, const DynamicQuantizedInnerProduct16bParam& p,
            const AlgParam& a, size_t M, uint8_t* dst, float* scale, int32_t* zero)
        {
            const float* src = (float*)src8;
            size_t K = p.K, K4 = AlignLo(K, 4), K16 = AlignLo(K, 16), KA = a.aK, k;
            for (size_t i = 0; i < M; ++i)
            {
                float32x4_t _min = vdupq_n_f32(0.0f);
                float32x4_t _max = vdupq_n_f32(0.0f);
                for (k = 0; k < K4; k += 4)
                {
                    float32x4_t _src = vld1q_f32(src + k);
                    _min = vminq_f32(_src, _min);
                    _max = vmaxq_f32(_src, _max);
                }
                float min = ReduceMin(_min);
                float max = ReduceMax(_max);
                for (; k < K; ++k)
                {
                    min = Simd::Min(min, src[k]);
                    max = Simd::Max(max, src[k]);
                }
                float rowScale = (min == max) ? 1.0f : (max - min) / 255.0f;
                scale[i] = rowScale;
                float z = -min / rowScale;
                zero[i] = Base::NearByInt(Simd::Max(0.0f, Simd::Min(255.0f, z)));
                float32x4_t _norm = vdupq_n_f32(1.0f / rowScale);
                int32x4_t _zero = vdupq_n_s32(zero[i]);

                for (k = 0; k < K16; k += 16)
                    QuantizeAf16(src + k, _norm, _zero, dst + k);
                for (; k < K4; k += 4)
                    QuantizeAf4(src + k, _norm, _zero, dst + k);
                for (; k < K; k += 1)
                    QuantizeAf1(src + k, _norm, _zero, dst + k);
                for (; k < KA; ++k)
                    dst[k] = 0;
                src += K;
                dst += KA;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE float32x4_t LoadBf16ToFp32(const uint16_t* ptr)
        {
            return BFloat16ToFloat32(vmovl_u16(vld1_u16(ptr)));
        }

        SIMD_INLINE float32x4_t LoadBf16ToFp32(const uint16_t* ptr, size_t size)
        {
            uint16_t buf[F] = { 0, 0, 0, 0 };
            for (size_t i = 0; i < size; ++i)
                buf[i] = ptr[i];
            return BFloat16ToFloat32(vmovl_u16(vld1_u16(buf)));
        }

        SIMD_INLINE void QuantizeAb1(const uint16_t* src, float32x4_t norm, int32x4_t zero, uint8_t* dst)
        {
            int32x4_t i32 = QuantizeAf(LoadBf16ToFp32(src, 1), norm, zero);
            dst[0] = vget_lane_u8(vqmovun_s16(vcombine_s16(vqmovn_s32(i32), vdup_n_s16(0))), 0);
        }

        SIMD_INLINE void QuantizeAb4(const uint16_t* src, float32x4_t norm, int32x4_t zero, uint8_t* dst)
        {
            int32x4_t i32 = QuantizeAf(LoadBf16ToFp32(src), norm, zero);
            uint8x8_t u8 = vqmovun_s16(vcombine_s16(vqmovn_s32(i32), vdup_n_s16(0)));
            vst1_lane_u32((uint32_t*)dst, vreinterpret_u32_u8(u8), 0);
        }

        SIMD_INLINE void QuantizeAb16(const uint16_t* src, float32x4_t norm, int32x4_t zero, uint8_t* dst)
        {
            uint16x8_t s0 = vld1q_u16(src + 0);
            uint16x8_t s1 = vld1q_u16(src + 8);
            int32x4_t i0 = QuantizeAf(BFloat16ToFloat32(vmovl_u16(vget_low_u16(s0))), norm, zero);
            int32x4_t i1 = QuantizeAf(BFloat16ToFloat32(vmovl_u16(vget_high_u16(s0))), norm, zero);
            int32x4_t i2 = QuantizeAf(BFloat16ToFloat32(vmovl_u16(vget_low_u16(s1))), norm, zero);
            int32x4_t i3 = QuantizeAf(BFloat16ToFloat32(vmovl_u16(vget_high_u16(s1))), norm, zero);
            vst1q_u8(dst, vcombine_u8(
                vqmovun_s16(vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1))),
                vqmovun_s16(vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3)))));
        }

        void Dqip16bGemmV0_ConvertAb(const uint8_t* src8, const DynamicQuantizedInnerProduct16bParam& p,
            const AlgParam& a, size_t M, uint8_t* dst, float* scale, int32_t* zero)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t K = p.K, K4 = AlignLo(K, 4), K16 = AlignLo(K, 16), KA = a.aK, k;
            for (size_t i = 0; i < M; ++i)
            {
                float32x4_t _min = vdupq_n_f32(0.0f);
                float32x4_t _max = vdupq_n_f32(0.0f);
                for (k = 0; k < K4; k += 4)
                {
                    float32x4_t _src = LoadBf16ToFp32(src + k);
                    _min = vminq_f32(_src, _min);
                    _max = vmaxq_f32(_src, _max);
                }
                float min = ReduceMin(_min);
                float max = ReduceMax(_max);
                for (; k < K; ++k)
                {
                    float val = Base::BFloat16ToFloat32(src[k]);
                    min = Simd::Min(min, val);
                    max = Simd::Max(max, val);
                }
                float rowScale = (min == max) ? 1.0f : (max - min) / 255.0f;
                scale[i] = rowScale;
                float z = -min / rowScale;
                zero[i] = Base::NearByInt(Simd::Max(0.0f, Simd::Min(255.0f, z)));
                float32x4_t _norm = vdupq_n_f32(1.0f / rowScale);
                int32x4_t _zero = vdupq_n_s32(zero[i]);

                for (k = 0; k < K16; k += 16)
                    QuantizeAb16(src + k, _norm, _zero, dst + k);
                for (; k < K4; k += 4)
                    QuantizeAb4(src + k, _norm, _zero, dst + k);
                for (; k < K; k += 1)
                    QuantizeAb1(src + k, _norm, _zero, dst + k);
                for (; k < KA; ++k)
                    dst[k] = 0;
                src += K;
                dst += KA;
            }
        }

        //-----------------------------------------------------------------------------------------

        const uint8x16_t K8_SFL_IDX = SIMD_VEC_SETR_EPI8(0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);

        SIMD_INLINE int8x16_t PackI32ToI8(int32x4_t v0, int32x4_t v1, int32x4_t v2, int32x4_t v3)
        {
            int8x16_t packed = vcombine_s8(
                vqmovn_s16(vcombine_s16(vqmovn_s32(v0), vqmovn_s32(v1))),
                vqmovn_s16(vcombine_s16(vqmovn_s32(v2), vqmovn_s32(v3))));
            return vreinterpretq_s8_u8(Shuffle(vreinterpretq_u8_s8(packed), K8_SFL_IDX));
        }

        SIMD_INLINE void QuantizeB4(float32x4_t s0, float32x4_t s1, float32x4_t s2, float32x4_t s3,
            float32x4_t norm, int32x4_t lo, int32x4_t hi, int32x4_t& sums, int8_t* dst)
        {
            int32x4_t v0 = vmaxq_s32(lo, vminq_s32(NearbyInt(vmulq_f32(s0, norm)), hi));
            int32x4_t v1 = vmaxq_s32(lo, vminq_s32(NearbyInt(vmulq_f32(s1, norm)), hi));
            int32x4_t v2 = vmaxq_s32(lo, vminq_s32(NearbyInt(vmulq_f32(s2, norm)), hi));
            int32x4_t v3 = vmaxq_s32(lo, vminq_s32(NearbyInt(vmulq_f32(s3, norm)), hi));
            sums = vsubq_s32(sums, vaddq_s32(vaddq_s32(v0, v1), vaddq_s32(v2, v3)));
            vst1q_s8(dst, PackI32ToI8(v0, v1, v2, v3));
        }

        void Dqip16bGemmV0_ConvertBf(const uint8_t* src8, const DynamicQuantizedInnerProduct16bParam& p,
            const AlgParam& a, size_t N, int8_t* dst, float* scale, int32_t* sums)
        {
            const float* src = (float*)src8;
            size_t K = p.K, K4 = AlignLo(K, 4), NF = AlignLo(N, F), pN = p.N, j = 0, k;
            int32x4_t _lo = vdupq_n_s32(-64), _hi = vdupq_n_s32(63);
            float32x4_t _eps = vdupq_n_f32(0.000001f), _q = vdupq_n_f32(63.0f);
            for (; j < NF; j += F)
            {
                const float* ps = src + j;
                float32x4_t _max = _eps;
                for (k = 0; k < K; ++k, ps += pN)
                    _max = vmaxq_f32(_max, vabsq_f32(vld1q_f32(ps)));
                float32x4_t _norm = Div<-1>(_q, _max);
                vst1q_f32(scale + j, Div<-1>(_max, _q));
                int32x4_t _sums = vdupq_n_s32(0);
                ps = src + j;
                for (k = 0; k < K4; k += 4, ps += pN * 4)
                {
                    QuantizeB4(vld1q_f32(ps + 0 * pN), vld1q_f32(ps + 1 * pN), vld1q_f32(ps + 2 * pN), vld1q_f32(ps + 3 * pN), _norm, _lo, _hi, _sums, dst);
                    dst += F * 4;
                }
                for (; k < K; k += 4, ps += pN * 4)
                {
                    size_t tK = Simd::Min<size_t>(4, K - k);
                    float32x4_t s0 = vld1q_f32(ps + 0 * pN);
                    float32x4_t s1 = tK > 1 ? vld1q_f32(ps + 1 * pN) : vdupq_n_f32(0.0f);
                    float32x4_t s2 = tK > 2 ? vld1q_f32(ps + 2 * pN) : vdupq_n_f32(0.0f);
                    float32x4_t s3 = tK > 3 ? vld1q_f32(ps + 3 * pN) : vdupq_n_f32(0.0f);
                    QuantizeB4(s0, s1, s2, s3, _norm, _lo, _hi, _sums, dst);
                    dst += F * 4;
                }
                vst1q_s32(sums + j, _sums);
            }
            for (; j < N; j += F)
            {
                size_t tN = Simd::Min<size_t>(F, N - j);
                const float* ps = src + j;
                float32x4_t _max = _eps;
                for (k = 0; k < K; ++k, ps += pN)
                    _max = vmaxq_f32(_max, vabsq_f32(LoadTail(ps, tN)));
                float32x4_t _norm = Div<-1>(_q, _max);
                Store(scale + j, Div<-1>(_max, _q), tN);
                int32x4_t _sums = vdupq_n_s32(0);
                ps = src + j;
                for (k = 0; k < K4; k += 4, ps += pN * 4)
                {
                    QuantizeB4(LoadTail(ps + 0 * pN, tN), LoadTail(ps + 1 * pN, tN), LoadTail(ps + 2 * pN, tN), LoadTail(ps + 3 * pN, tN), _norm, _lo, _hi, _sums, dst);
                    dst += F * 4;
                }
                for (; k < K; k += 4, ps += pN * 4)
                {
                    size_t tK = Simd::Min<size_t>(4, K - k);
                    float32x4_t s0 = LoadTail(ps + 0 * pN, tN);
                    float32x4_t s1 = tK > 1 ? LoadTail(ps + 1 * pN, tN) : vdupq_n_f32(0.0f);
                    float32x4_t s2 = tK > 2 ? LoadTail(ps + 2 * pN, tN) : vdupq_n_f32(0.0f);
                    float32x4_t s3 = tK > 3 ? LoadTail(ps + 3 * pN, tN) : vdupq_n_f32(0.0f);
                    QuantizeB4(s0, s1, s2, s3, _norm, _lo, _hi, _sums, dst);
                    dst += F * 4;
                }
                StoreTail(sums + j, _sums, tN);
            }
        }

        //-----------------------------------------------------------------------------------------

        void Dqip16bGemmV0_ConvertBb(const uint8_t* src8, const DynamicQuantizedInnerProduct16bParam& p,
            const AlgParam& a, size_t N, int8_t* dst, float* scale, int32_t* sums)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t K = p.K, K4 = AlignLo(K, 4), NF = AlignLo(N, F), pN = p.N, j = 0, k;
            int32x4_t _lo = vdupq_n_s32(-64), _hi = vdupq_n_s32(63);
            float32x4_t _eps = vdupq_n_f32(0.000001f), _q = vdupq_n_f32(63.0f);
            for (; j < NF; j += F)
            {
                const uint16_t* ps = src + j;
                float32x4_t _max = _eps;
                for (k = 0; k < K; ++k, ps += pN)
                    _max = vmaxq_f32(_max, vabsq_f32(LoadBf16ToFp32(ps)));
                float32x4_t _norm = Div<-1>(_q, _max);
                vst1q_f32(scale + j, Div<-1>(_max, _q));
                int32x4_t _sums = vdupq_n_s32(0);
                ps = src + j;
                for (k = 0; k < K4; k += 4, ps += pN * 4)
                {
                    QuantizeB4(LoadBf16ToFp32(ps + 0 * pN), LoadBf16ToFp32(ps + 1 * pN), LoadBf16ToFp32(ps + 2 * pN), LoadBf16ToFp32(ps + 3 * pN), _norm, _lo, _hi, _sums, dst);
                    dst += F * 4;
                }
                for (; k < K; k += 4, ps += pN * 4)
                {
                    size_t tK = Simd::Min<size_t>(4, K - k);
                    float32x4_t s0 = LoadBf16ToFp32(ps + 0 * pN);
                    float32x4_t s1 = tK > 1 ? LoadBf16ToFp32(ps + 1 * pN) : vdupq_n_f32(0.0f);
                    float32x4_t s2 = tK > 2 ? LoadBf16ToFp32(ps + 2 * pN) : vdupq_n_f32(0.0f);
                    float32x4_t s3 = tK > 3 ? LoadBf16ToFp32(ps + 3 * pN) : vdupq_n_f32(0.0f);
                    QuantizeB4(s0, s1, s2, s3, _norm, _lo, _hi, _sums, dst);
                    dst += F * 4;
                }
                vst1q_s32(sums + j, _sums);
            }
            for (; j < N; j += F)
            {
                size_t tN = Simd::Min<size_t>(F, N - j);
                const uint16_t* ps = src + j;
                float32x4_t _max = _eps;
                for (k = 0; k < K; ++k, ps += pN)
                    _max = vmaxq_f32(_max, vabsq_f32(LoadBf16ToFp32(ps, tN)));
                float32x4_t _norm = Div<-1>(_q, _max);
                Store(scale + j, Div<-1>(_max, _q), tN);
                int32x4_t _sums = vdupq_n_s32(0);
                ps = src + j;
                for (k = 0; k < K4; k += 4, ps += pN * 4)
                {
                    QuantizeB4(LoadBf16ToFp32(ps + 0 * pN, tN), LoadBf16ToFp32(ps + 1 * pN, tN), LoadBf16ToFp32(ps + 2 * pN, tN), LoadBf16ToFp32(ps + 3 * pN, tN), _norm, _lo, _hi, _sums, dst);
                    dst += F * 4;
                }
                for (; k < K; k += 4, ps += pN * 4)
                {
                    size_t tK = Simd::Min<size_t>(4, K - k);
                    float32x4_t s0 = LoadBf16ToFp32(ps + 0 * pN, tN);
                    float32x4_t s1 = tK > 1 ? LoadBf16ToFp32(ps + 1 * pN, tN) : vdupq_n_f32(0.0f);
                    float32x4_t s2 = tK > 2 ? LoadBf16ToFp32(ps + 2 * pN, tN) : vdupq_n_f32(0.0f);
                    float32x4_t s3 = tK > 3 ? LoadBf16ToFp32(ps + 3 * pN, tN) : vdupq_n_f32(0.0f);
                    QuantizeB4(s0, s1, s2, s3, _norm, _lo, _hi, _sums, dst);
                    dst += F * 4;
                }
                StoreTail(sums + j, _sums, tN);
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void StoreBody1(int32_t* ptr, int32x4_t val0)
        {
            vst1q_s32(ptr, val0);
        }

        SIMD_INLINE void StoreBody2(int32_t* ptr, int32x4_t val0, int32x4_t val1)
        {
            vst1q_s32(ptr + 0, val0);
            vst1q_s32(ptr + F, val1);
        }

        template<int M> void Dqip16bGemmV0_GemmBody_2xM(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum)
        {
            int32x4_t c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71, c80, c81, c90, c91, cA0, cA1, cB0, cB1;
            uint8x16_t a0;
            int8x16_t b0, b1;
            size_t dS = a.cN, dA = a.aK;
            const int8_t* B1 = B0 + a.aK * F;
            const uint8_t* A1 = A0 + 0x1 * dA;
            const uint8_t* A2 = A0 + 0x2 * dA;
            const uint8_t* A3 = A0 + 0x3 * dA;
            const uint8_t* A4 = A0 + 0x4 * dA;
            const uint8_t* A5 = A0 + 0x5 * dA;
            const uint8_t* A6 = A0 + 0x6 * dA;
            const uint8_t* A7 = A0 + 0x7 * dA;
            const uint8_t* A8 = A0 + 0x8 * dA;
            const uint8_t* A9 = A0 + 0x9 * dA;
            const uint8_t* AA = A0 + 0xA * dA;
            const uint8_t* AB = A0 + 0xB * dA;
            if (N > F)
            {
                if (update)
                {
                    if (M > 0x0) c00 = vld1q_s32(sum + 0x0 * dS + 0), c01 = vld1q_s32(sum + 0x0 * dS + F);
                    if (M > 0x1) c10 = vld1q_s32(sum + 0x1 * dS + 0), c11 = vld1q_s32(sum + 0x1 * dS + F);
                    if (M > 0x2) c20 = vld1q_s32(sum + 0x2 * dS + 0), c21 = vld1q_s32(sum + 0x2 * dS + F);
                    if (M > 0x3) c30 = vld1q_s32(sum + 0x3 * dS + 0), c31 = vld1q_s32(sum + 0x3 * dS + F);
                    if (M > 0x4) c40 = vld1q_s32(sum + 0x4 * dS + 0), c41 = vld1q_s32(sum + 0x4 * dS + F);
                    if (M > 0x5) c50 = vld1q_s32(sum + 0x5 * dS + 0), c51 = vld1q_s32(sum + 0x5 * dS + F);
                    if (M > 0x6) c60 = vld1q_s32(sum + 0x6 * dS + 0), c61 = vld1q_s32(sum + 0x6 * dS + F);
                    if (M > 0x7) c70 = vld1q_s32(sum + 0x7 * dS + 0), c71 = vld1q_s32(sum + 0x7 * dS + F);
                    if (M > 0x8) c80 = vld1q_s32(sum + 0x8 * dS + 0), c81 = vld1q_s32(sum + 0x8 * dS + F);
                    if (M > 0x9) c90 = vld1q_s32(sum + 0x9 * dS + 0), c91 = vld1q_s32(sum + 0x9 * dS + F);
                    if (M > 0xA) cA0 = vld1q_s32(sum + 0xA * dS + 0), cA1 = vld1q_s32(sum + 0xA * dS + F);
                    if (M > 0xB) cB0 = vld1q_s32(sum + 0xB * dS + 0), cB1 = vld1q_s32(sum + 0xB * dS + F);
                }
                else
                {
                    if (M > 0x0) c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
                    if (M > 0x1) c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
                    if (M > 0x2) c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0);
                    if (M > 0x3) c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0);
                    if (M > 0x4) c40 = vdupq_n_s32(0), c41 = vdupq_n_s32(0);
                    if (M > 0x5) c50 = vdupq_n_s32(0), c51 = vdupq_n_s32(0);
                    if (M > 0x6) c60 = vdupq_n_s32(0), c61 = vdupq_n_s32(0);
                    if (M > 0x7) c70 = vdupq_n_s32(0), c71 = vdupq_n_s32(0);
                    if (M > 0x8) c80 = vdupq_n_s32(0), c81 = vdupq_n_s32(0);
                    if (M > 0x9) c90 = vdupq_n_s32(0), c91 = vdupq_n_s32(0);
                    if (M > 0xA) cA0 = vdupq_n_s32(0), cA1 = vdupq_n_s32(0);
                    if (M > 0xB) cB0 = vdupq_n_s32(0), cB1 = vdupq_n_s32(0);
                }
                for (size_t k = 0; k < K; k += 4)
                {
                    b0 = vld1q_s8(B0);
                    b1 = vld1q_s8(B1);
                    if (M > 0x0) a0 = Set4(A0 + k), Madd4<true>(c00, a0, b0), Madd4<true>(c01, a0, b1);
                    if (M > 0x1) a0 = Set4(A1 + k), Madd4<true>(c10, a0, b0), Madd4<true>(c11, a0, b1);
                    if (M > 0x2) a0 = Set4(A2 + k), Madd4<true>(c20, a0, b0), Madd4<true>(c21, a0, b1);
                    if (M > 0x3) a0 = Set4(A3 + k), Madd4<true>(c30, a0, b0), Madd4<true>(c31, a0, b1);
                    if (M > 0x4) a0 = Set4(A4 + k), Madd4<true>(c40, a0, b0), Madd4<true>(c41, a0, b1);
                    if (M > 0x5) a0 = Set4(A5 + k), Madd4<true>(c50, a0, b0), Madd4<true>(c51, a0, b1);
                    if (M > 0x6) a0 = Set4(A6 + k), Madd4<true>(c60, a0, b0), Madd4<true>(c61, a0, b1);
                    if (M > 0x7) a0 = Set4(A7 + k), Madd4<true>(c70, a0, b0), Madd4<true>(c71, a0, b1);
                    if (M > 0x8) a0 = Set4(A8 + k), Madd4<true>(c80, a0, b0), Madd4<true>(c81, a0, b1);
                    if (M > 0x9) a0 = Set4(A9 + k), Madd4<true>(c90, a0, b0), Madd4<true>(c91, a0, b1);
                    if (M > 0xA) a0 = Set4(AA + k), Madd4<true>(cA0, a0, b0), Madd4<true>(cA1, a0, b1);
                    if (M > 0xB) a0 = Set4(AB + k), Madd4<true>(cB0, a0, b0), Madd4<true>(cB1, a0, b1);
                    B0 += A, B1 += A;
                }
                if (M > 0x0) StoreBody2(sum + 0x0 * dS, c00, c01);
                if (M > 0x1) StoreBody2(sum + 0x1 * dS, c10, c11);
                if (M > 0x2) StoreBody2(sum + 0x2 * dS, c20, c21);
                if (M > 0x3) StoreBody2(sum + 0x3 * dS, c30, c31);
                if (M > 0x4) StoreBody2(sum + 0x4 * dS, c40, c41);
                if (M > 0x5) StoreBody2(sum + 0x5 * dS, c50, c51);
                if (M > 0x6) StoreBody2(sum + 0x6 * dS, c60, c61);
                if (M > 0x7) StoreBody2(sum + 0x7 * dS, c70, c71);
                if (M > 0x8) StoreBody2(sum + 0x8 * dS, c80, c81);
                if (M > 0x9) StoreBody2(sum + 0x9 * dS, c90, c91);
                if (M > 0xA) StoreBody2(sum + 0xA * dS, cA0, cA1);
                if (M > 0xB) StoreBody2(sum + 0xB * dS, cB0, cB1);
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) c00 = vld1q_s32(sum + 0x0 * dS + 0);
                    if (M > 0x1) c10 = vld1q_s32(sum + 0x1 * dS + 0);
                    if (M > 0x2) c20 = vld1q_s32(sum + 0x2 * dS + 0);
                    if (M > 0x3) c30 = vld1q_s32(sum + 0x3 * dS + 0);
                    if (M > 0x4) c40 = vld1q_s32(sum + 0x4 * dS + 0);
                    if (M > 0x5) c50 = vld1q_s32(sum + 0x5 * dS + 0);
                    if (M > 0x6) c60 = vld1q_s32(sum + 0x6 * dS + 0);
                    if (M > 0x7) c70 = vld1q_s32(sum + 0x7 * dS + 0);
                    if (M > 0x8) c80 = vld1q_s32(sum + 0x8 * dS + 0);
                    if (M > 0x9) c90 = vld1q_s32(sum + 0x9 * dS + 0);
                    if (M > 0xA) cA0 = vld1q_s32(sum + 0xA * dS + 0);
                    if (M > 0xB) cB0 = vld1q_s32(sum + 0xB * dS + 0);
                }
                else
                {
                    if (M > 0x0) c00 = vdupq_n_s32(0);
                    if (M > 0x1) c10 = vdupq_n_s32(0);
                    if (M > 0x2) c20 = vdupq_n_s32(0);
                    if (M > 0x3) c30 = vdupq_n_s32(0);
                    if (M > 0x4) c40 = vdupq_n_s32(0);
                    if (M > 0x5) c50 = vdupq_n_s32(0);
                    if (M > 0x6) c60 = vdupq_n_s32(0);
                    if (M > 0x7) c70 = vdupq_n_s32(0);
                    if (M > 0x8) c80 = vdupq_n_s32(0);
                    if (M > 0x9) c90 = vdupq_n_s32(0);
                    if (M > 0xA) cA0 = vdupq_n_s32(0);
                    if (M > 0xB) cB0 = vdupq_n_s32(0);
                }
                for (size_t k = 0; k < K; k += 4)
                {
                    b0 = vld1q_s8(B0);
                    if (M > 0x0) a0 = Set4(A0 + k), Madd4<true>(c00, a0, b0);
                    if (M > 0x1) a0 = Set4(A1 + k), Madd4<true>(c10, a0, b0);
                    if (M > 0x2) a0 = Set4(A2 + k), Madd4<true>(c20, a0, b0);
                    if (M > 0x3) a0 = Set4(A3 + k), Madd4<true>(c30, a0, b0);
                    if (M > 0x4) a0 = Set4(A4 + k), Madd4<true>(c40, a0, b0);
                    if (M > 0x5) a0 = Set4(A5 + k), Madd4<true>(c50, a0, b0);
                    if (M > 0x6) a0 = Set4(A6 + k), Madd4<true>(c60, a0, b0);
                    if (M > 0x7) a0 = Set4(A7 + k), Madd4<true>(c70, a0, b0);
                    if (M > 0x8) a0 = Set4(A8 + k), Madd4<true>(c80, a0, b0);
                    if (M > 0x9) a0 = Set4(A9 + k), Madd4<true>(c90, a0, b0);
                    if (M > 0xA) a0 = Set4(AA + k), Madd4<true>(cA0, a0, b0);
                    if (M > 0xB) a0 = Set4(AB + k), Madd4<true>(cB0, a0, b0);
                    B0 += A;
                }
                if (M > 0x0) StoreBody1(sum + 0x0 * dS, c00);
                if (M > 0x1) StoreBody1(sum + 0x1 * dS, c10);
                if (M > 0x2) StoreBody1(sum + 0x2 * dS, c20);
                if (M > 0x3) StoreBody1(sum + 0x3 * dS, c30);
                if (M > 0x4) StoreBody1(sum + 0x4 * dS, c40);
                if (M > 0x5) StoreBody1(sum + 0x5 * dS, c50);
                if (M > 0x6) StoreBody1(sum + 0x6 * dS, c60);
                if (M > 0x7) StoreBody1(sum + 0x7 * dS, c70);
                if (M > 0x8) StoreBody1(sum + 0x8 * dS, c80);
                if (M > 0x9) StoreBody1(sum + 0x9 * dS, c90);
                if (M > 0xA) StoreBody1(sum + 0xA * dS, cA0);
                if (M > 0xB) StoreBody1(sum + 0xB * dS, cB0);
            }
        }

        typedef void(*Dqip16bGemmV0_GemmBody_2xM_Ptr)(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum);

        SIMD_INLINE Dqip16bGemmV0_GemmBody_2xM_Ptr GetDqip16bGemmV0_GemmBody_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return Dqip16bGemmV0_GemmBody_2xM<1>;
            case 2: return Dqip16bGemmV0_GemmBody_2xM<2>;
            case 3: return Dqip16bGemmV0_GemmBody_2xM<3>;
            case 4: return Dqip16bGemmV0_GemmBody_2xM<4>;
            case 5: return Dqip16bGemmV0_GemmBody_2xM<5>;
            case 6: return Dqip16bGemmV0_GemmBody_2xM<6>;
            case 7: return Dqip16bGemmV0_GemmBody_2xM<7>;
            case 8: return Dqip16bGemmV0_GemmBody_2xM<8>;
            case 9: return Dqip16bGemmV0_GemmBody_2xM<9>;
            case 10: return Dqip16bGemmV0_GemmBody_2xM<10>;
            case 11: return Dqip16bGemmV0_GemmBody_2xM<11>;
            case 12: return Dqip16bGemmV0_GemmBody_2xM<12>;
            }
            assert(0);
            return NULL;
        }

        static void Dqip16bGemmV0_GemmBody(const uint8_t* A, const DqipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum)
        {
            size_t n = 12;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dS = a.cN, dA = a.aK;
            Dqip16bGemmV0_GemmBody_2xM_Ptr gemm_2xN = GetDqip16bGemmV0_GemmBody_2xM(n);
            Dqip16bGemmV0_GemmBody_2xM_Ptr gemm_2xM = GetDqip16bGemmV0_GemmBody_2xM(m);

            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                size_t i = 0;
                for (; i < Mn; i += n)
                    gemm_2xN(A + i * dA, p, a, K, dN, update, B, sum + i * dS);
                for (; i < M; i += m)
                    gemm_2xM(A + i * dA, p, a, K, dN, update, B, sum + i * dS);
                B += a.aK * DF;
                sum += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int M> void Dqip16bGemmV0_GemmLast_2xM(const uint8_t* A0,
            const DqipParam& p, const AlgParam& a, size_t K, size_t N, int update, const int8_t* B0, int32_t* sum, int32_t* buf,
            const int32_t* aZero, const int32x4_t* bSums, const float* aScale, const float32x4_t* bScale, const float32x4_t* bias, const float32x4_t* params, uint8_t* C)
        {
            int32x4_t c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71, c80, c81, c90, c91, cA0, cA1, cB0, cB1;
            uint8x16_t a0;
            int8x16_t b0, b1;
            size_t dS = a.cN, dA = a.aK, dC = p.N * a.eC;
            const int8_t* B1 = B0 + a.aK * F;
            const uint8_t* A1 = A0 + 0x1 * dA;
            const uint8_t* A2 = A0 + 0x2 * dA;
            const uint8_t* A3 = A0 + 0x3 * dA;
            const uint8_t* A4 = A0 + 0x4 * dA;
            const uint8_t* A5 = A0 + 0x5 * dA;
            const uint8_t* A6 = A0 + 0x6 * dA;
            const uint8_t* A7 = A0 + 0x7 * dA;
            const uint8_t* A8 = A0 + 0x8 * dA;
            const uint8_t* A9 = A0 + 0x9 * dA;
            const uint8_t* AA = A0 + 0xA * dA;
            const uint8_t* AB = A0 + 0xB * dA;
            if (N > F)
            {
                if (update)
                {
                    if (M > 0x0) c00 = vld1q_s32(sum + 0x0 * dS + 0), c01 = vld1q_s32(sum + 0x0 * dS + F);
                    if (M > 0x1) c10 = vld1q_s32(sum + 0x1 * dS + 0), c11 = vld1q_s32(sum + 0x1 * dS + F);
                    if (M > 0x2) c20 = vld1q_s32(sum + 0x2 * dS + 0), c21 = vld1q_s32(sum + 0x2 * dS + F);
                    if (M > 0x3) c30 = vld1q_s32(sum + 0x3 * dS + 0), c31 = vld1q_s32(sum + 0x3 * dS + F);
                    if (M > 0x4) c40 = vld1q_s32(sum + 0x4 * dS + 0), c41 = vld1q_s32(sum + 0x4 * dS + F);
                    if (M > 0x5) c50 = vld1q_s32(sum + 0x5 * dS + 0), c51 = vld1q_s32(sum + 0x5 * dS + F);
                    if (M > 0x6) c60 = vld1q_s32(sum + 0x6 * dS + 0), c61 = vld1q_s32(sum + 0x6 * dS + F);
                    if (M > 0x7) c70 = vld1q_s32(sum + 0x7 * dS + 0), c71 = vld1q_s32(sum + 0x7 * dS + F);
                    if (M > 0x8) c80 = vld1q_s32(sum + 0x8 * dS + 0), c81 = vld1q_s32(sum + 0x8 * dS + F);
                    if (M > 0x9) c90 = vld1q_s32(sum + 0x9 * dS + 0), c91 = vld1q_s32(sum + 0x9 * dS + F);
                    if (M > 0xA) cA0 = vld1q_s32(sum + 0xA * dS + 0), cA1 = vld1q_s32(sum + 0xA * dS + F);
                    if (M > 0xB) cB0 = vld1q_s32(sum + 0xB * dS + 0), cB1 = vld1q_s32(sum + 0xB * dS + F);
                }
                else
                {
                    if (M > 0x0) c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
                    if (M > 0x1) c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
                    if (M > 0x2) c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0);
                    if (M > 0x3) c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0);
                    if (M > 0x4) c40 = vdupq_n_s32(0), c41 = vdupq_n_s32(0);
                    if (M > 0x5) c50 = vdupq_n_s32(0), c51 = vdupq_n_s32(0);
                    if (M > 0x6) c60 = vdupq_n_s32(0), c61 = vdupq_n_s32(0);
                    if (M > 0x7) c70 = vdupq_n_s32(0), c71 = vdupq_n_s32(0);
                    if (M > 0x8) c80 = vdupq_n_s32(0), c81 = vdupq_n_s32(0);
                    if (M > 0x9) c90 = vdupq_n_s32(0), c91 = vdupq_n_s32(0);
                    if (M > 0xA) cA0 = vdupq_n_s32(0), cA1 = vdupq_n_s32(0);
                    if (M > 0xB) cB0 = vdupq_n_s32(0), cB1 = vdupq_n_s32(0);
                }
                for (size_t k = 0; k < K; k += 4)
                {
                    b0 = vld1q_s8(B0);
                    b1 = vld1q_s8(B1);
                    if (M > 0x0) a0 = Set4(A0 + k), Madd4<true>(c00, a0, b0), Madd4<true>(c01, a0, b1);
                    if (M > 0x1) a0 = Set4(A1 + k), Madd4<true>(c10, a0, b0), Madd4<true>(c11, a0, b1);
                    if (M > 0x2) a0 = Set4(A2 + k), Madd4<true>(c20, a0, b0), Madd4<true>(c21, a0, b1);
                    if (M > 0x3) a0 = Set4(A3 + k), Madd4<true>(c30, a0, b0), Madd4<true>(c31, a0, b1);
                    if (M > 0x4) a0 = Set4(A4 + k), Madd4<true>(c40, a0, b0), Madd4<true>(c41, a0, b1);
                    if (M > 0x5) a0 = Set4(A5 + k), Madd4<true>(c50, a0, b0), Madd4<true>(c51, a0, b1);
                    if (M > 0x6) a0 = Set4(A6 + k), Madd4<true>(c60, a0, b0), Madd4<true>(c61, a0, b1);
                    if (M > 0x7) a0 = Set4(A7 + k), Madd4<true>(c70, a0, b0), Madd4<true>(c71, a0, b1);
                    if (M > 0x8) a0 = Set4(A8 + k), Madd4<true>(c80, a0, b0), Madd4<true>(c81, a0, b1);
                    if (M > 0x9) a0 = Set4(A9 + k), Madd4<true>(c90, a0, b0), Madd4<true>(c91, a0, b1);
                    if (M > 0xA) a0 = Set4(AA + k), Madd4<true>(cA0, a0, b0), Madd4<true>(cA1, a0, b1);
                    if (M > 0xB) a0 = Set4(AB + k), Madd4<true>(cB0, a0, b0), Madd4<true>(cB1, a0, b1);
                    B0 += A, B1 += A;
                }
                if (N == DF)
                {
                    if (M > 0x0) Save2<term, type>(C + 0x0 * dC, c00, c01, aZero[0x0], bSums, aScale[0x0], bScale, bias, params);
                    if (M > 0x1) Save2<term, type>(C + 0x1 * dC, c10, c11, aZero[0x1], bSums, aScale[0x1], bScale, bias, params);
                    if (M > 0x2) Save2<term, type>(C + 0x2 * dC, c20, c21, aZero[0x2], bSums, aScale[0x2], bScale, bias, params);
                    if (M > 0x3) Save2<term, type>(C + 0x3 * dC, c30, c31, aZero[0x3], bSums, aScale[0x3], bScale, bias, params);
                    if (M > 0x4) Save2<term, type>(C + 0x4 * dC, c40, c41, aZero[0x4], bSums, aScale[0x4], bScale, bias, params);
                    if (M > 0x5) Save2<term, type>(C + 0x5 * dC, c50, c51, aZero[0x5], bSums, aScale[0x5], bScale, bias, params);
                    if (M > 0x6) Save2<term, type>(C + 0x6 * dC, c60, c61, aZero[0x6], bSums, aScale[0x6], bScale, bias, params);
                    if (M > 0x7) Save2<term, type>(C + 0x7 * dC, c70, c71, aZero[0x7], bSums, aScale[0x7], bScale, bias, params);
                    if (M > 0x8) Save2<term, type>(C + 0x8 * dC, c80, c81, aZero[0x8], bSums, aScale[0x8], bScale, bias, params);
                    if (M > 0x9) Save2<term, type>(C + 0x9 * dC, c90, c91, aZero[0x9], bSums, aScale[0x9], bScale, bias, params);
                    if (M > 0xA) Save2<term, type>(C + 0xA * dC, cA0, cA1, aZero[0xA], bSums, aScale[0xA], bScale, bias, params);
                    if (M > 0xB) Save2<term, type>(C + 0xB * dC, cB0, cB1, aZero[0xB], bSums, aScale[0xB], bScale, bias, params);
                }
                else
                {
                    if (M > 0x0) Save2<term, type>(C + 0x0 * dC, c00, c01, aZero[0x0], bSums, aScale[0x0], bScale, bias, params, N);
                    if (M > 0x1) Save2<term, type>(C + 0x1 * dC, c10, c11, aZero[0x1], bSums, aScale[0x1], bScale, bias, params, N);
                    if (M > 0x2) Save2<term, type>(C + 0x2 * dC, c20, c21, aZero[0x2], bSums, aScale[0x2], bScale, bias, params, N);
                    if (M > 0x3) Save2<term, type>(C + 0x3 * dC, c30, c31, aZero[0x3], bSums, aScale[0x3], bScale, bias, params, N);
                    if (M > 0x4) Save2<term, type>(C + 0x4 * dC, c40, c41, aZero[0x4], bSums, aScale[0x4], bScale, bias, params, N);
                    if (M > 0x5) Save2<term, type>(C + 0x5 * dC, c50, c51, aZero[0x5], bSums, aScale[0x5], bScale, bias, params, N);
                    if (M > 0x6) Save2<term, type>(C + 0x6 * dC, c60, c61, aZero[0x6], bSums, aScale[0x6], bScale, bias, params, N);
                    if (M > 0x7) Save2<term, type>(C + 0x7 * dC, c70, c71, aZero[0x7], bSums, aScale[0x7], bScale, bias, params, N);
                    if (M > 0x8) Save2<term, type>(C + 0x8 * dC, c80, c81, aZero[0x8], bSums, aScale[0x8], bScale, bias, params, N);
                    if (M > 0x9) Save2<term, type>(C + 0x9 * dC, c90, c91, aZero[0x9], bSums, aScale[0x9], bScale, bias, params, N);
                    if (M > 0xA) Save2<term, type>(C + 0xA * dC, cA0, cA1, aZero[0xA], bSums, aScale[0xA], bScale, bias, params, N);
                    if (M > 0xB) Save2<term, type>(C + 0xB * dC, cB0, cB1, aZero[0xB], bSums, aScale[0xB], bScale, bias, params, N);
                }
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) c00 = vld1q_s32(sum + 0x0 * dS + 0);
                    if (M > 0x1) c10 = vld1q_s32(sum + 0x1 * dS + 0);
                    if (M > 0x2) c20 = vld1q_s32(sum + 0x2 * dS + 0);
                    if (M > 0x3) c30 = vld1q_s32(sum + 0x3 * dS + 0);
                    if (M > 0x4) c40 = vld1q_s32(sum + 0x4 * dS + 0);
                    if (M > 0x5) c50 = vld1q_s32(sum + 0x5 * dS + 0);
                    if (M > 0x6) c60 = vld1q_s32(sum + 0x6 * dS + 0);
                    if (M > 0x7) c70 = vld1q_s32(sum + 0x7 * dS + 0);
                    if (M > 0x8) c80 = vld1q_s32(sum + 0x8 * dS + 0);
                    if (M > 0x9) c90 = vld1q_s32(sum + 0x9 * dS + 0);
                    if (M > 0xA) cA0 = vld1q_s32(sum + 0xA * dS + 0);
                    if (M > 0xB) cB0 = vld1q_s32(sum + 0xB * dS + 0);
                }
                else
                {
                    if (M > 0x0) c00 = vdupq_n_s32(0);
                    if (M > 0x1) c10 = vdupq_n_s32(0);
                    if (M > 0x2) c20 = vdupq_n_s32(0);
                    if (M > 0x3) c30 = vdupq_n_s32(0);
                    if (M > 0x4) c40 = vdupq_n_s32(0);
                    if (M > 0x5) c50 = vdupq_n_s32(0);
                    if (M > 0x6) c60 = vdupq_n_s32(0);
                    if (M > 0x7) c70 = vdupq_n_s32(0);
                    if (M > 0x8) c80 = vdupq_n_s32(0);
                    if (M > 0x9) c90 = vdupq_n_s32(0);
                    if (M > 0xA) cA0 = vdupq_n_s32(0);
                    if (M > 0xB) cB0 = vdupq_n_s32(0);
                }
                for (size_t k = 0; k < K; k += 4)
                {
                    b0 = vld1q_s8(B0);
                    if (M > 0x0) a0 = Set4(A0 + k), Madd4<true>(c00, a0, b0);
                    if (M > 0x1) a0 = Set4(A1 + k), Madd4<true>(c10, a0, b0);
                    if (M > 0x2) a0 = Set4(A2 + k), Madd4<true>(c20, a0, b0);
                    if (M > 0x3) a0 = Set4(A3 + k), Madd4<true>(c30, a0, b0);
                    if (M > 0x4) a0 = Set4(A4 + k), Madd4<true>(c40, a0, b0);
                    if (M > 0x5) a0 = Set4(A5 + k), Madd4<true>(c50, a0, b0);
                    if (M > 0x6) a0 = Set4(A6 + k), Madd4<true>(c60, a0, b0);
                    if (M > 0x7) a0 = Set4(A7 + k), Madd4<true>(c70, a0, b0);
                    if (M > 0x8) a0 = Set4(A8 + k), Madd4<true>(c80, a0, b0);
                    if (M > 0x9) a0 = Set4(A9 + k), Madd4<true>(c90, a0, b0);
                    if (M > 0xA) a0 = Set4(AA + k), Madd4<true>(cA0, a0, b0);
                    if (M > 0xB) a0 = Set4(AB + k), Madd4<true>(cB0, a0, b0);
                    B0 += A;
                }
                if (N == F)
                {
                    if (M > 0x0) Save1<term, type>(C + 0x0 * dC, c00, aZero[0x0], bSums, aScale[0x0], bScale, bias, params);
                    if (M > 0x1) Save1<term, type>(C + 0x1 * dC, c10, aZero[0x1], bSums, aScale[0x1], bScale, bias, params);
                    if (M > 0x2) Save1<term, type>(C + 0x2 * dC, c20, aZero[0x2], bSums, aScale[0x2], bScale, bias, params);
                    if (M > 0x3) Save1<term, type>(C + 0x3 * dC, c30, aZero[0x3], bSums, aScale[0x3], bScale, bias, params);
                    if (M > 0x4) Save1<term, type>(C + 0x4 * dC, c40, aZero[0x4], bSums, aScale[0x4], bScale, bias, params);
                    if (M > 0x5) Save1<term, type>(C + 0x5 * dC, c50, aZero[0x5], bSums, aScale[0x5], bScale, bias, params);
                    if (M > 0x6) Save1<term, type>(C + 0x6 * dC, c60, aZero[0x6], bSums, aScale[0x6], bScale, bias, params);
                    if (M > 0x7) Save1<term, type>(C + 0x7 * dC, c70, aZero[0x7], bSums, aScale[0x7], bScale, bias, params);
                    if (M > 0x8) Save1<term, type>(C + 0x8 * dC, c80, aZero[0x8], bSums, aScale[0x8], bScale, bias, params);
                    if (M > 0x9) Save1<term, type>(C + 0x9 * dC, c90, aZero[0x9], bSums, aScale[0x9], bScale, bias, params);
                    if (M > 0xA) Save1<term, type>(C + 0xA * dC, cA0, aZero[0xA], bSums, aScale[0xA], bScale, bias, params);
                    if (M > 0xB) Save1<term, type>(C + 0xB * dC, cB0, aZero[0xB], bSums, aScale[0xB], bScale, bias, params);
                }
                else
                {
                    if (M > 0x0) Save1<term, type>(C + 0x0 * dC, c00, aZero[0x0], bSums, aScale[0x0], bScale, bias, params, N);
                    if (M > 0x1) Save1<term, type>(C + 0x1 * dC, c10, aZero[0x1], bSums, aScale[0x1], bScale, bias, params, N);
                    if (M > 0x2) Save1<term, type>(C + 0x2 * dC, c20, aZero[0x2], bSums, aScale[0x2], bScale, bias, params, N);
                    if (M > 0x3) Save1<term, type>(C + 0x3 * dC, c30, aZero[0x3], bSums, aScale[0x3], bScale, bias, params, N);
                    if (M > 0x4) Save1<term, type>(C + 0x4 * dC, c40, aZero[0x4], bSums, aScale[0x4], bScale, bias, params, N);
                    if (M > 0x5) Save1<term, type>(C + 0x5 * dC, c50, aZero[0x5], bSums, aScale[0x5], bScale, bias, params, N);
                    if (M > 0x6) Save1<term, type>(C + 0x6 * dC, c60, aZero[0x6], bSums, aScale[0x6], bScale, bias, params, N);
                    if (M > 0x7) Save1<term, type>(C + 0x7 * dC, c70, aZero[0x7], bSums, aScale[0x7], bScale, bias, params, N);
                    if (M > 0x8) Save1<term, type>(C + 0x8 * dC, c80, aZero[0x8], bSums, aScale[0x8], bScale, bias, params, N);
                    if (M > 0x9) Save1<term, type>(C + 0x9 * dC, c90, aZero[0x9], bSums, aScale[0x9], bScale, bias, params, N);
                    if (M > 0xA) Save1<term, type>(C + 0xA * dC, cA0, aZero[0xA], bSums, aScale[0xA], bScale, bias, params, N);
                    if (M > 0xB) Save1<term, type>(C + 0xB * dC, cB0, aZero[0xB], bSums, aScale[0xB], bScale, bias, params, N);
                }
            }
        }

        typedef void(*Dqip16bGemmV0_GemmLast_2xM_Ptr)(const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, size_t N, int update, const int8_t* B0, int32_t* sum,
            int32_t* buf, const int32_t* aZero, const int32x4_t* bSums, const float* aScale, const float32x4_t* bScale, const float32x4_t* bias, const float32x4_t* params, uint8_t* C);

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE Dqip16bGemmV0_GemmLast_2xM_Ptr GetDqip16bGemmV0_GemmLast_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return Dqip16bGemmV0_GemmLast_2xM<term, type, 1>;
            case 2: return Dqip16bGemmV0_GemmLast_2xM<term, type, 2>;
            case 3: return Dqip16bGemmV0_GemmLast_2xM<term, type, 3>;
            case 4: return Dqip16bGemmV0_GemmLast_2xM<term, type, 4>;
            case 5: return Dqip16bGemmV0_GemmLast_2xM<term, type, 5>;
            case 6: return Dqip16bGemmV0_GemmLast_2xM<term, type, 6>;
            case 7: return Dqip16bGemmV0_GemmLast_2xM<term, type, 7>;
            case 8: return Dqip16bGemmV0_GemmLast_2xM<term, type, 8>;
            case 9: return Dqip16bGemmV0_GemmLast_2xM<term, type, 9>;
            case 10: return Dqip16bGemmV0_GemmLast_2xM<term, type, 10>;
            case 11: return Dqip16bGemmV0_GemmLast_2xM<term, type, 11>;
            case 12: return Dqip16bGemmV0_GemmLast_2xM<term, type, 12>;
            }
            assert(0);
            return NULL;
        }

        template<Term16bType term, SimdConvolutionActivationType type> void Dqip16bGemmV0_GemmLast(const uint8_t* A, const DqipParam& p,
            const AlgParam& a, size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum, int32_t* buf,
            const int32_t* aZero, const int32_t* bSums, const float* aScale, const float* bScale, const float* bias, const float* params, uint8_t* C)
        {
            size_t n = 12;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dS = a.cN, dA = a.aK, dC = p.N * a.eC;
            Dqip16bGemmV0_GemmLast_2xM_Ptr gemm_2xN = GetDqip16bGemmV0_GemmLast_2xM<term, type>(n);
            Dqip16bGemmV0_GemmLast_2xM_Ptr gemm_2xM = GetDqip16bGemmV0_GemmLast_2xM<term, type>(m);

            int32x4_t _bSums[2];
            float32x4_t _bScale[2], _bias[2], _params[2];
            if (type != SimdConvolutionActivationIdentity)
            {
                _params[0] = vdupq_n_f32(params[0]);
                _params[1] = vdupq_n_f32(params[1]);
            }
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                _bSums[0] = vld1q_s32(bSums + j + 0);
                _bScale[0] = vld1q_f32(bScale + j + 0);
                _bias[0] = vld1q_f32(bias + j + 0);
                if (dN > F)
                {
                    _bSums[1] = vld1q_s32(bSums + j + F);
                    _bScale[1] = vld1q_f32(bScale + j + F);
                    _bias[1] = vld1q_f32(bias + j + F);
                }
                if (type == SimdConvolutionActivationPrelu)
                {
                    _params[0] = vld1q_f32(params + j + 0);
                    if (dN > F)
                        _params[1] = vld1q_f32(params + j + F);
                }
                size_t i = 0;
                for (; i < Mn; i += n)
                    gemm_2xN(A + i * dA, p, a, K, dN, update, B, sum + i * dS, buf, aZero + i, _bSums, aScale + i, _bScale, _bias, _params, C + i * dC);
                for (; i < M; i += m)
                    gemm_2xM(A + i * dA, p, a, K, dN, update, B, sum + i * dS, buf, aZero + i, _bSums, aScale + i, _bScale, _bias, _params, C + i * dC);
                B += a.aK * DF;
                sum += DF;
                C += DF * a.eC;
            }
        }

        template<Term16bType term> void SetGemmLast(const DqipParam& p, GemmLastPtr& gemmLast)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationHswish>; break;
            case SimdConvolutionActivationMish: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationMish>; break;
            case SimdConvolutionActivationHardSigmoid: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationHardSigmoid>; break;
            case SimdConvolutionActivationSwish: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationSwish>; break;
            case SimdConvolutionActivationGelu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationGelu>; break;
            default:
                assert(0);
            }
        }

        void SetGemmLast(const DqipParam& p, GemmLastPtr& gemmLast)
        {
            if (p.typeC == SimdTensorData16b)
                SetGemmLast<Term16bLast16b>(p, gemmLast);
            else if (p.typeC == SimdTensorData32f)
                SetGemmLast<Term16bLast32f>(p, gemmLast);
        }

        //-----------------------------------------------------------------------------------------

        SynetDynamicQuantizedInnerProduct16bGemmV0::SynetDynamicQuantizedInnerProduct16bGemmV0(const DynamicQuantizedInnerProduct16bParam& p)
            : Base::SynetDynamicQuantizedInnerProduct16bGemmV0(p)
        {
            SetAlgParam(F, 12, F * 2, 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (p.typeA == SimdTensorData32f)
                _convertA = Dqip16bGemmV0_ConvertAf;
            else if (p.typeA == SimdTensorData16b)
                _convertA = Dqip16bGemmV0_ConvertAb;
            if (p.typeB == SimdTensorData32f)
                _convertB = Dqip16bGemmV0_ConvertBf;
            else if (p.typeB == SimdTensorData16b)
                _convertB = Dqip16bGemmV0_ConvertBb;
            _gemmBody = Dqip16bGemmV0_GemmBody;
            SetGemmLast(p, _gemmLast);
        }
    }
#endif
}
