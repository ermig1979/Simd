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
#include "Simd/SimdTranspose.h"
#include "Simd/SimdConversion.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Neon
    {
        template <bool align, bool nofma> SIMD_INLINE void SynetConvert32fTo8u(const float* src, float32x4_t scale, float32x4_t shift, uint8x8_t upper, uint8_t* dst)
        {
            int32x4_t i32 = Round(Fmadd<nofma>(Load<align>(src), scale, shift));
            *((int32_t*)dst) = vget_lane_s32(vreinterpret_s32_u8(vmin_u8(vqmovun_s16(vcombine_s16(vmovn_s32(i32), vcreate_s16(0))), upper)), 0);
        }

        template <bool align, bool nofma> void SynetConvert32fTo8uNchw(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst) && Aligned(spatial, F));

            size_t spatialF = AlignLo(spatial, F);
            uint8x8_t _upper = vdup_n_u8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float32x4_t _scale = vdupq_n_f32(scale[c]);
                    float32x4_t _shift = vdupq_n_f32(shift[c]);
                    size_t s = 0;
                    for (; s < spatialF; s += F)
                        SynetConvert32fTo8u<align, nofma>(src + s, _scale, _shift, _upper, dst + s);
                    for (; s < spatial; s += 1)
                        dst[s] = Base::SynetConvert32fTo8u(src[s], scale[c], shift[c], 0, upper);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template <bool align, bool nofma> void SynetConvert32fTo8uNhwc(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst) && Aligned(channels, F) && Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            uint8x8_t _upper = vdup_n_u8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                        SynetConvert32fTo8u<align, nofma>(src + c, Load<align>(scale + c), Load<align>(shift + c), _upper, dst + c);
                    for (; c < channels; ++c)
                        dst[c] = Base::SynetConvert32fTo8u(src[c], scale[c], shift[c], 0, upper);
                    src += channels;
                    dst += channels;
                }
            }
        }

        template <bool align, bool nofma> void SynetConvert32fTo8uNhwc3(const float* src, size_t batch, size_t spatial, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst) && Aligned(spatial, A));

            size_t spatial3 = spatial * 3;
            size_t spatial3F = AlignLo(spatial, F) * 3;
            uint8x8_t _upper = vdup_n_u8(upper);
            float _scale[F * 3], _shift[F * 3];
            for (size_t i = 0; i < F; ++i)
                for (size_t c = 0; c < 3; ++c)
                    _scale[i * 3 + c] = scale[c], _shift[i * 3 + c] = shift[c];

            float32x4_t _scale0 = Load<false>(_scale + 0 * F);
            float32x4_t _scale1 = Load<false>(_scale + 1 * F);
            float32x4_t _scale2 = Load<false>(_scale + 2 * F);
            float32x4_t _shift0 = Load<false>(_shift + 0 * F);
            float32x4_t _shift1 = Load<false>(_shift + 1 * F);
            float32x4_t _shift2 = Load<false>(_shift + 2 * F);

            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatial3F; s += 3 * F)
                {
                    SynetConvert32fTo8u<align, nofma>(src + 0 * F, _scale0, _shift0, _upper, dst + 0 * F);
                    SynetConvert32fTo8u<align, nofma>(src + 1 * F, _scale1, _shift1, _upper, dst + 1 * F);
                    SynetConvert32fTo8u<align, nofma>(src + 2 * F, _scale2, _shift2, _upper, dst + 2 * F);
                    src += 3 * F;
                    dst += 3 * F;
                }
                for (; s < spatial3; s += 3)
                {
                    dst[0] = Base::SynetConvert32fTo8u(src[0], scale[0], shift[0], 0, upper);
                    dst[1] = Base::SynetConvert32fTo8u(src[1], scale[1], shift[1], 0, upper);
                    dst[2] = Base::SynetConvert32fTo8u(src[2], scale[2], shift[2], 0, upper);
                    src += 3;
                    dst += 3;
                }
            }
        }

        template<bool nofma> void SynetConvert32fTo8u(const float* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            size_t spatial = height * width;
            if (Base::NchwCompatible(channels, spatial, format))
            {
                if (Aligned(src) && Aligned(dst) && Aligned(spatial, A))
                    SynetConvert32fTo8uNchw<true, nofma>(src, batch, channels, spatial, scale, shift, upper, dst);
                else
                    SynetConvert32fTo8uNchw<false, nofma>(src, batch, channels, spatial, scale, shift, upper, dst);
            }
            else if (Base::NhwcCompatible(channels, spatial, format))
            {
                if (channels == 3)
                {
                    if (Aligned(src) && Aligned(dst) && Aligned(spatial, A))
                        SynetConvert32fTo8uNhwc3<true, nofma>(src, batch, spatial, scale, shift, upper, dst);
                    else
                        SynetConvert32fTo8uNhwc3<false, nofma>(src, batch, spatial, scale, shift, upper, dst);
                }
                else
                {
                    if (Aligned(src) && Aligned(dst) && Aligned(channels, A) && Aligned(scale) && Aligned(shift))
                        SynetConvert32fTo8uNhwc<true, nofma>(src, batch, channels, spatial, scale, shift, upper, dst);
                    else
                        SynetConvert32fTo8uNhwc<false, nofma>(src, batch, channels, spatial, scale, shift, upper, dst);
                }
            }
            else
                assert(0);
        }

        void SynetConvert32fTo8u(const float* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, uint8_t* dst, SimdSynetCompatibilityType compatibility)
        {
            int upper = Base::Narrowed(compatibility) ? Base::U8_NARROWED_MAX : Base::U8_PRECISE_MAX;
            if (Base::FmaAvoid(compatibility))
                SynetConvert32fTo8u<true>(src, batch, channels, height, width, format, scale, shift, upper, dst);
            else
                SynetConvert32fTo8u<false>(src, batch, channels, height, width, format, scale, shift, upper, dst);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void StoreScaled(float * ptr, uint32x4_t value32, float32x4_t scale, float32x4_t shift)
        {
            Store<align>(ptr, vmlaq_f32(shift, vcvtq_f32_u32(value32), scale));
        }

        template<SimdPixelFormatType format> SIMD_INLINE void SynetSetInput1(const uint8_t * src, float32x4_t scale, float32x4_t shift, float * dst);

        SIMD_INLINE void SynetSetInput1Gray16(uint16x8_t gray, float32x4_t scale, float32x4_t shift, float * dst)
        {
            StoreScaled<false>(dst + 0 * F, UnpackU16<0>(gray), scale, shift);
            StoreScaled<false>(dst + 1 * F, UnpackU16<1>(gray), scale, shift);
        }       
        
        SIMD_INLINE void SynetSetInput1Gray8(uint8x16_t gray, float32x4_t scale, float32x4_t shift, float * dst)
        {
            SynetSetInput1Gray16(UnpackU8<0>(gray), scale, shift, dst + 0 * F);
            SynetSetInput1Gray16(UnpackU8<1>(gray), scale, shift, dst + 2 * F);
        }

        template<> SIMD_INLINE void SynetSetInput1<SimdPixelFormatGray8>(const uint8_t * src, float32x4_t scale, float32x4_t shift, float * dst)
        {
            SynetSetInput1Gray8(Load<false>(src), scale, shift, dst);
        }

        SIMD_INLINE void SynetSetInput1Bgr16(uint16x8_t blue, uint16x8_t green, uint16x8_t red, float32x4_t scale, float32x4_t shift, float * dst)
        {
            StoreScaled<false>(dst + 0 * F, BgrToGray<0>(blue, green, red), scale, shift);
            StoreScaled<false>(dst + 1 * F, BgrToGray<1>(blue, green, red), scale, shift);
        }

        template<> SIMD_INLINE void SynetSetInput1<SimdPixelFormatBgr24>(const uint8_t * src, float32x4_t scale, float32x4_t shift, float * dst)
        {
            uint8x16x3_t bgr = Load3<false>(src);
            SynetSetInput1Bgr16(UnpackU8<0>(bgr.val[0]), UnpackU8<0>(bgr.val[1]), UnpackU8<0>(bgr.val[2]), scale, shift, dst + 0 * F);
            SynetSetInput1Bgr16(UnpackU8<1>(bgr.val[0]), UnpackU8<1>(bgr.val[1]), UnpackU8<1>(bgr.val[2]), scale, shift, dst + 2 * F);
        }

        template<> SIMD_INLINE void SynetSetInput1<SimdPixelFormatBgra32>(const uint8_t * src, float32x4_t scale, float32x4_t shift, float * dst)
        {
            uint8x16x4_t bgra = Load4<false>(src);
            SynetSetInput1Bgr16(UnpackU8<0>(bgra.val[0]), UnpackU8<0>(bgra.val[1]), UnpackU8<0>(bgra.val[2]), scale, shift, dst + 0 * F);
            SynetSetInput1Bgr16(UnpackU8<1>(bgra.val[0]), UnpackU8<1>(bgra.val[1]), UnpackU8<1>(bgra.val[2]), scale, shift, dst + 2 * F);
        }

        template<> SIMD_INLINE void SynetSetInput1<SimdPixelFormatRgb24>(const uint8_t * src, float32x4_t scale, float32x4_t shift, float * dst)
        {
            uint8x16x3_t rgb = Load3<false>(src);
            SynetSetInput1Bgr16(UnpackU8<0>(rgb.val[2]), UnpackU8<0>(rgb.val[1]), UnpackU8<0>(rgb.val[0]), scale, shift, dst + 0 * F);
            SynetSetInput1Bgr16(UnpackU8<1>(rgb.val[2]), UnpackU8<1>(rgb.val[1]), UnpackU8<1>(rgb.val[0]), scale, shift, dst + 2 * F);
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInput1(const uint8_t * src, size_t width, size_t height, size_t stride, const float * scale, const float * shift, float * dst)
        {
            float32x4_t _scale = vdupq_n_f32(scale[0]);
            float32x4_t _shift = vdupq_n_f32(shift[0]);
            size_t aligned = AlignLo(width, A);
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < aligned; x += A)
                    SynetSetInput1<format>(src + step * x, _scale, _shift, dst + x);
                if (aligned < width)
                    SynetSetInput1<format>(src + step * (width - A), _scale, _shift, dst + width - A);
                src += stride;
                dst += width;
            }
        }

        template<SimdPixelFormatType format> SIMD_INLINE void SynetSetInputNchw3(const uint8_t * src, const float32x4_t * scale, const float32x4_t * shift, float * dst, size_t channel);

        template<> SIMD_INLINE void SynetSetInputNchw3<SimdPixelFormatGray8>(const uint8_t * src, const float32x4_t * scale, const float32x4_t * shift, float * dst, size_t channel)
        {
            uint8x16_t _src = Load<false>(src);
            uint16x8_t src0 = UnpackU8<0>(_src);
            uint16x8_t src1 = UnpackU8<1>(_src);
            uint32x4_t gray0 = UnpackU16<0>(src0);
            uint32x4_t gray1 = UnpackU16<1>(src0);
            uint32x4_t gray2 = UnpackU16<0>(src1);
            uint32x4_t gray3 = UnpackU16<1>(src1);
            StoreScaled<false>(dst + 0 * F, gray0, scale[0], shift[0]);
            StoreScaled<false>(dst + 1 * F, gray1, scale[0], shift[0]);
            StoreScaled<false>(dst + 2 * F, gray2, scale[0], shift[0]);
            StoreScaled<false>(dst + 3 * F, gray3, scale[0], shift[0]);
            dst += channel;
            StoreScaled<false>(dst + 0 * F, gray0, scale[1], shift[1]);
            StoreScaled<false>(dst + 1 * F, gray1, scale[1], shift[1]);
            StoreScaled<false>(dst + 2 * F, gray2, scale[1], shift[1]);
            StoreScaled<false>(dst + 3 * F, gray3, scale[1], shift[1]);
            dst += channel;
            StoreScaled<false>(dst + 0 * F, gray0, scale[2], shift[2]);
            StoreScaled<false>(dst + 1 * F, gray1, scale[2], shift[2]);
            StoreScaled<false>(dst + 2 * F, gray2, scale[2], shift[2]);
            StoreScaled<false>(dst + 3 * F, gray3, scale[2], shift[2]);
        }

        template<> SIMD_INLINE void SynetSetInputNchw3<SimdPixelFormatBgr24>(const uint8_t * src, const float32x4_t * scale, const float32x4_t * shift, float * dst, size_t channel)
        {
            uint8x16x3_t bgr = Load3<false>(src);
            SynetSetInput1Gray8(bgr.val[0], scale[0], shift[0], dst + 0 * channel);
            SynetSetInput1Gray8(bgr.val[1], scale[1], shift[1], dst + 1 * channel);
            SynetSetInput1Gray8(bgr.val[2], scale[2], shift[2], dst + 2 * channel);
        }

        template<> SIMD_INLINE void SynetSetInputNchw3<SimdPixelFormatBgra32>(const uint8_t * src, const float32x4_t * scale, const float32x4_t * shift, float * dst, size_t channel)
        {
            uint8x16x4_t bgra = Load4<false>(src);
            SynetSetInput1Gray8(bgra.val[0], scale[0], shift[0], dst + 0 * channel);
            SynetSetInput1Gray8(bgra.val[1], scale[1], shift[1], dst + 1 * channel);
            SynetSetInput1Gray8(bgra.val[2], scale[2], shift[2], dst + 2 * channel);
        }

        template<> SIMD_INLINE void SynetSetInputNchw3<SimdPixelFormatRgb24>(const uint8_t * src, const float32x4_t * scale, const float32x4_t * shift, float * dst, size_t channel)
        {
            uint8x16x3_t rgb = Load3<false>(src);
            SynetSetInput1Gray8(rgb.val[2], scale[0], shift[0], dst + 0 * channel);
            SynetSetInput1Gray8(rgb.val[1], scale[1], shift[1], dst + 1 * channel);
            SynetSetInput1Gray8(rgb.val[0], scale[2], shift[2], dst + 2 * channel);
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInputNchw3(const uint8_t * src, size_t width, size_t height, size_t stride, const float * scale, const float * shift, float * dst)
        {
            size_t aligned = AlignLo(width, A), channel = width * height;
            float32x4_t _scale[3], _shift[3];
            for (size_t i = 0; i < 3; ++i)
            {
                _scale[i] = vdupq_n_f32(scale[i]);
                _shift[i] = vdupq_n_f32(shift[i]);
            }
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < aligned; x += A)
                    SynetSetInputNchw3<format>(src + step * x, _scale, _shift, dst + x, channel);
                if (aligned < width)
                    SynetSetInputNchw3<format>(src + step * (width - A), _scale, _shift, dst + width - A, channel);
                src += stride;
                dst += width;
            }
        }

        template<SimdPixelFormatType format> SIMD_INLINE void SynetSetInputNhwc3(const uint8_t * src, const float32x4_t * scale, const float32x4_t * shift, float * dst);

        SIMD_INLINE uint8x8_t Shuffle(const uint8x16_t & src, const uint8x8_t & idx)
        {
            return vtbl2_u8((const uint8x8x2_t &)src, idx);
        }

        const uint8x8_t K8_TBL_GRAY_TO_BGR_0 = SIMD_VEC_SETR_PI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x2, 0x2);
        const uint8x8_t K8_TBL_GRAY_TO_BGR_1 = SIMD_VEC_SETR_PI8(0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4, 0x5);
        const uint8x8_t K8_TBL_GRAY_TO_BGR_2 = SIMD_VEC_SETR_PI8(0x5, 0x5, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7);

        template<> SIMD_INLINE void SynetSetInputNhwc3<SimdPixelFormatGray8>(const uint8_t * src, const float32x4_t * scale, const float32x4_t * shift, float * dst)
        {
            uint8x8_t gray0 = LoadHalf<false>(src + 0);
            uint16x8_t bgr0 = vmovl_u8(vtbl1_u8(gray0, K8_TBL_GRAY_TO_BGR_0));
            StoreScaled<false>(dst + 0x0 * F, UnpackU16<0>(bgr0), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x1 * F, UnpackU16<1>(bgr0), scale[1], shift[1]);
            uint16x8_t bgr1 = vmovl_u8(vtbl1_u8(gray0, K8_TBL_GRAY_TO_BGR_1));
            StoreScaled<false>(dst + 0x2 * F, UnpackU16<0>(bgr1), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x3 * F, UnpackU16<1>(bgr1), scale[0], shift[0]);
            uint16x8_t bgr2 = vmovl_u8(vtbl1_u8(gray0, K8_TBL_GRAY_TO_BGR_2));
            StoreScaled<false>(dst + 0x4 * F, UnpackU16<0>(bgr2), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x5 * F, UnpackU16<1>(bgr2), scale[2], shift[2]);
            uint8x8_t gray1 = LoadHalf<false>(src + 8);
            uint16x8_t bgr3 = vmovl_u8(vtbl1_u8(gray1, K8_TBL_GRAY_TO_BGR_0));
            StoreScaled<false>(dst + 0x6 * F, UnpackU16<0>(bgr3), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x7 * F, UnpackU16<1>(bgr3), scale[1], shift[1]);
            uint16x8_t bgr4 = vmovl_u8(vtbl1_u8(gray1, K8_TBL_GRAY_TO_BGR_1));
            StoreScaled<false>(dst + 0x8 * F, UnpackU16<0>(bgr4), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x9 * F, UnpackU16<1>(bgr4), scale[0], shift[0]);
            uint16x8_t bgr5 = vmovl_u8(vtbl1_u8(gray1, K8_TBL_GRAY_TO_BGR_2));
            StoreScaled<false>(dst + 0xA * F, UnpackU16<0>(bgr5), scale[1], shift[1]);
            StoreScaled<false>(dst + 0xB * F, UnpackU16<1>(bgr5), scale[2], shift[2]);
        }

        template<> SIMD_INLINE void SynetSetInputNhwc3<SimdPixelFormatBgr24>(const uint8_t * src, const float32x4_t * scale, const float32x4_t * shift, float * dst)
        {
            uint8x16_t bgr0 = Load<false>(src + 0 * A);
            uint16x8_t bgr00 = UnpackU8<0>(bgr0);
            StoreScaled<false>(dst + 0x0 * F, UnpackU16<0>(bgr00), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x1 * F, UnpackU16<1>(bgr00), scale[1], shift[1]);
            uint16x8_t bgr01 = UnpackU8<1>(bgr0);
            StoreScaled<false>(dst + 0x2 * F, UnpackU16<0>(bgr01), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x3 * F, UnpackU16<1>(bgr01), scale[0], shift[0]);
            uint8x16_t bgr1 = Load<false>(src + 1 * A);
            uint16x8_t bgr10 = UnpackU8<0>(bgr1);
            StoreScaled<false>(dst + 0x4 * F, UnpackU16<0>(bgr10), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x5 * F, UnpackU16<1>(bgr10), scale[2], shift[2]);
            uint16x8_t bgr11 = UnpackU8<1>(bgr1);
            StoreScaled<false>(dst + 0x6 * F, UnpackU16<0>(bgr11), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x7 * F, UnpackU16<1>(bgr11), scale[1], shift[1]);
            uint8x16_t bgr2 = Load<false>(src + 2 * A);
            uint16x8_t bgr20 = UnpackU8<0>(bgr2);
            StoreScaled<false>(dst + 0x8 * F, UnpackU16<0>(bgr20), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x9 * F, UnpackU16<1>(bgr20), scale[0], shift[0]);
            uint16x8_t bgr21 = UnpackU8<1>(bgr2);
            StoreScaled<false>(dst + 0xA * F, UnpackU16<0>(bgr21), scale[1], shift[1]);
            StoreScaled<false>(dst + 0xB * F, UnpackU16<1>(bgr21), scale[2], shift[2]);
        }

        const uint8x8_t K8_TBL_BGRA_TO_BGR_0 = SIMD_VEC_SETR_PI8(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9);
        const uint8x8_t K8_TBL_BGRA_TO_BGR_1 = SIMD_VEC_SETR_PI8(0x0, 0x2, 0x3, 0x4, 0x6, 0x7, 0x8, 0xA);
        const uint8x8_t K8_TBL_BGRA_TO_BGR_2 = SIMD_VEC_SETR_PI8(0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE);

        template<> SIMD_INLINE void SynetSetInputNhwc3<SimdPixelFormatBgra32>(const uint8_t * src, const float32x4_t * scale, const float32x4_t * shift, float * dst)
        {
            uint16x8_t bgr0 = vmovl_u8(Shuffle(Load<false>(src + 0), K8_TBL_BGRA_TO_BGR_0));
            StoreScaled<false>(dst + 0x0 * F, UnpackU16<0>(bgr0), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x1 * F, UnpackU16<1>(bgr0), scale[1], shift[1]);
            uint16x8_t bgr1 = vmovl_u8(Shuffle(Load<false>(src + 10), K8_TBL_BGRA_TO_BGR_1));
            StoreScaled<false>(dst + 0x2 * F, UnpackU16<0>(bgr1), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x3 * F, UnpackU16<1>(bgr1), scale[0], shift[0]);
            uint16x8_t bgr2 = vmovl_u8(Shuffle(Load<false>(src + 16), K8_TBL_BGRA_TO_BGR_2));
            StoreScaled<false>(dst + 0x4 * F, UnpackU16<0>(bgr2), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x5 * F, UnpackU16<1>(bgr2), scale[2], shift[2]);
            uint16x8_t bgr3 = vmovl_u8(Shuffle(Load<false>(src + 32), K8_TBL_BGRA_TO_BGR_0));
            StoreScaled<false>(dst + 0x6 * F, UnpackU16<0>(bgr3), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x7 * F, UnpackU16<1>(bgr3), scale[1], shift[1]);
            uint16x8_t bgr4 = vmovl_u8(Shuffle(Load<false>(src + 42), K8_TBL_BGRA_TO_BGR_1));
            StoreScaled<false>(dst + 0x8 * F, UnpackU16<0>(bgr4), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x9 * F, UnpackU16<1>(bgr4), scale[0], shift[0]);
            uint16x8_t bgr5 = vmovl_u8(Shuffle(Load<false>(src + 48), K8_TBL_BGRA_TO_BGR_2));
            StoreScaled<false>(dst + 0xA * F, UnpackU16<0>(bgr5), scale[1], shift[1]);
            StoreScaled<false>(dst + 0xB * F, UnpackU16<1>(bgr5), scale[2], shift[2]);
        }

        const uint8x8_t K8_TBL_RGB_TO_BGR_0 = SIMD_VEC_SETR_PI8(0x2, 0x1, 0x0, 0x5, 0x4, 0x3, 0x8, 0x7);
        const uint8x8_t K8_TBL_RGB_TO_BGR_1 = SIMD_VEC_SETR_PI8(0x0, 0x5, 0x4, 0x3, 0x8, 0x7, 0x6, 0xB);
        const uint8x8_t K8_TBL_RGB_TO_BGR_2 = SIMD_VEC_SETR_PI8(0x8, 0x7, 0xC, 0xB, 0xA, 0xF, 0xE, 0xD);

        template<> SIMD_INLINE void SynetSetInputNhwc3<SimdPixelFormatRgb24>(const uint8_t * src, const float32x4_t * scale, const float32x4_t * shift, float * dst)
        {
            uint16x8_t bgr0 = vmovl_u8(Shuffle(Load<false>(src + 0), K8_TBL_RGB_TO_BGR_0));
            StoreScaled<false>(dst + 0x0 * F, UnpackU16<0>(bgr0), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x1 * F, UnpackU16<1>(bgr0), scale[1], shift[1]);
            uint16x8_t bgr1 = vmovl_u8(Shuffle(Load<false>(src + 6), K8_TBL_RGB_TO_BGR_1));
            StoreScaled<false>(dst + 0x2 * F, UnpackU16<0>(bgr1), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x3 * F, UnpackU16<1>(bgr1), scale[0], shift[0]);
            uint16x8_t bgr2 = vmovl_u8(Shuffle(Load<false>(src + 8), K8_TBL_RGB_TO_BGR_2));
            StoreScaled<false>(dst + 0x4 * F, UnpackU16<0>(bgr2), scale[1], shift[1]);
            StoreScaled<false>(dst + 0x5 * F, UnpackU16<1>(bgr2), scale[2], shift[2]);
            uint16x8_t bgr3 = vmovl_u8(Shuffle(Load<false>(src + 24), K8_TBL_RGB_TO_BGR_0));
            StoreScaled<false>(dst + 0x6 * F, UnpackU16<0>(bgr3), scale[0], shift[0]);
            StoreScaled<false>(dst + 0x7 * F, UnpackU16<1>(bgr3), scale[1], shift[1]);
            uint16x8_t bgr4 = vmovl_u8(Shuffle(Load<false>(src + 30), K8_TBL_RGB_TO_BGR_1));
            StoreScaled<false>(dst + 0x8 * F, UnpackU16<0>(bgr4), scale[2], shift[2]);
            StoreScaled<false>(dst + 0x9 * F, UnpackU16<1>(bgr4), scale[0], shift[0]);
            uint16x8_t bgr5 = vmovl_u8(Shuffle(Load<false>(src + 32), K8_TBL_RGB_TO_BGR_2));
            StoreScaled<false>(dst + 0xA * F, UnpackU16<0>(bgr5), scale[1], shift[1]);
            StoreScaled<false>(dst + 0xB * F, UnpackU16<1>(bgr5), scale[2], shift[2]);
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInputNhwc3(const uint8_t * src, size_t width, size_t height, size_t stride, const float * scale, const float * shift, float * dst)
        {
            size_t aligned = AlignLo(width, A);
            float32x4_t _scale[3], _shift[3];
            _scale[0] = SetF32(scale[0], scale[1], scale[2], scale[0]);
            _scale[1] = SetF32(scale[1], scale[2], scale[0], scale[1]);
            _scale[2] = SetF32(scale[2], scale[0], scale[1], scale[2]);
            _shift[0] = SetF32(shift[0], shift[1], shift[2], shift[0]);
            _shift[1] = SetF32(shift[1], shift[2], shift[0], shift[1]);
            _shift[2] = SetF32(shift[2], shift[0], shift[1], shift[2]);
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < aligned; x += A)
                    SynetSetInputNhwc3<format>(src + step * x, _scale, _shift, dst + 3 * x);
                if (aligned < width)
                    SynetSetInputNhwc3<format>(src + step * (width - A), _scale, _shift, dst + 3 * (width - A));
                src += stride;
                dst += 3 * width;
            }
        }

        void SynetSetInput(const uint8_t * src, size_t width, size_t height, size_t stride, SimdPixelFormatType srcFormat,
            const float * lower, const float * upper, float * dst, size_t channels, SimdTensorFormatType dstFormat)
        {
            assert(width >= A);

            float scale[3];
            for (size_t i = 0; i < channels; ++i)
                scale[i] = (upper[i] - lower[i]) / 255.0f;
            switch (channels)
            {
            case 1:
                switch (srcFormat)
                {
                case SimdPixelFormatGray8: SynetSetInput1<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatBgr24: SynetSetInput1<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatBgra32: SynetSetInput1<SimdPixelFormatBgra32, 4>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatRgb24: SynetSetInput1<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                default: assert(0);
                }
                break;
            case 3:
                switch (dstFormat)
                {
                case SimdTensorFormatNchw:
                    switch (srcFormat)
                    {
                    case SimdPixelFormatGray8: SynetSetInputNchw3<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgr24: SynetSetInputNchw3<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgra32: SynetSetInputNchw3<SimdPixelFormatBgra32, 4>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgb24: SynetSetInputNchw3<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                    default: assert(0);
                    }
                    break;
                case SimdTensorFormatNhwc:
                    switch (srcFormat)
                    {
                    case SimdPixelFormatGray8: SynetSetInputNhwc3<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgr24: SynetSetInputNhwc3<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgra32: SynetSetInputNhwc3<SimdPixelFormatBgra32, 4>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgb24: SynetSetInputNhwc3<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                    default: return Base::SynetSetInput(src, width, height, stride, srcFormat, lower, upper, dst, channels, dstFormat); assert(0);
                    }
                    break;
                default: assert(0);
                }
                break;
            default: assert(0);
            }
        }
    }
#endif// SIMD_NEON_ENABLE
}
