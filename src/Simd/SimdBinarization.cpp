/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
#include "Simd/SimdEnable.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdBinarization.h"

namespace Simd
{
	namespace Base
	{
        template <CompareType type> SIMD_INLINE bool Compare(const uchar & src, const uchar & b);

        template <> SIMD_INLINE bool Compare<CompareGreaterThen>(const uchar & a, const uchar & b)
        {
            return a > b;
        }

        template <> SIMD_INLINE bool Compare<CompareLesserThen>(const uchar & a, const uchar & b)
        {
            return a < b;
        }

        template <> SIMD_INLINE bool Compare<CompareEqualTo>(const uchar & a, const uchar & b)
        {
            return a == b;
        }

        template <CompareType compareType> 
		void Binarization(const uchar * src, size_t srcStride, size_t width, size_t height, 
			uchar value, uchar positive, uchar negative, uchar * dst, size_t dstStride)
		{
			for(size_t row = 0; row < height; ++row)
			{
				for(size_t col = 0; col < width; ++col)
					dst[col] = Compare<compareType>(src[col], value) ? positive : negative;
				src += srcStride;
				dst += dstStride;
			}
		}

        void Binarization(const uchar * src, size_t srcStride, size_t width, size_t height, 
            uchar value, uchar positive, uchar negative, uchar * dst, size_t dstStride, CompareType compareType)
        {
            switch(compareType)
            {
            case CompareGreaterThen:
                return Binarization<CompareGreaterThen>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case CompareLesserThen:
                return Binarization<CompareLesserThen>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case CompareEqualTo:
                return Binarization<CompareEqualTo>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            default:
                assert(0);
            }
        }

        namespace
        {
            struct Buffer
            {
                Buffer(size_t width, size_t edge)
                {
                    size_t size = sizeof(uint)*(width + 2*edge);
                    _p = Allocate(size);
                    memset(_p, 0, size);
                    sa = (uint*)_p + edge;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint * sa;
            private:
                void *_p;
            };
        }

        template <CompareType compareType>
        void AveragingBinarization(const uchar * src, size_t srcStride, size_t width, size_t height,
            uchar value, size_t neighborhood, uchar threshold, uchar positive, uchar negative, uchar * dst, size_t dstStride)
        {
            assert(width > neighborhood && height > neighborhood && neighborhood < 0x80);

            Buffer buffer(width, neighborhood + 1);

            union SaSum
            {
                uint sum;
                ushort sa[2];
            };

            for(size_t row = 0; row < neighborhood; ++row)
            {
                const uchar * s = src + row*srcStride;
                for(size_t col = 0; col < width; ++col)
                { 
                    buffer.sa[col] += Compare<compareType>(s[col], value) ? 0x10001 : 0x10000;
                }
            }

            for(size_t row = 0; row < height; ++row)
            {
                if(row < height - neighborhood)
                {
                    const uchar * s = src + (row + neighborhood)*srcStride;
                    for(size_t col = 0; col < width; ++col)
                    {
                        buffer.sa[col] += Compare<compareType>(s[col], value) ? 0x10001 : 0x10000;
                    }
                }

                if(row > neighborhood)
                {
                    const uchar * s = src + (row - neighborhood - 1)*srcStride;
                    for(size_t col = 0; col < width; ++col)
                    {
                        buffer.sa[col] -= Compare<compareType>(s[col], value) ? 0x10001 : 0x10000;
                    }
                }

                SaSum saSum = {0};
                for(size_t col = 0; col < neighborhood; ++col)
                    saSum.sum += buffer.sa[col];
                for(size_t col = 0; col < width; ++col)
                {
                    saSum.sum += buffer.sa[col + neighborhood];
                    saSum.sum -= buffer.sa[col - neighborhood - 1];
                    dst[col] = (saSum.sa[0]*0xFF > threshold*saSum.sa[1]) ? positive : negative;
                }
                dst += dstStride;
            }
        }

        void AveragingBinarization(const uchar * src, size_t srcStride, size_t width, size_t height,
            uchar value, size_t neighborhood, uchar threshold, uchar positive, uchar negative, 
            uchar * dst, size_t dstStride, CompareType compareType)
        {
            switch(compareType)
            {
            case CompareGreaterThen:
                return AveragingBinarization<CompareGreaterThen>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case CompareLesserThen:
                return AveragingBinarization<CompareLesserThen>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case CompareEqualTo:
                return AveragingBinarization<CompareEqualTo>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            default:
                assert(0);
            }
        }
	}

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
        template<CompareType compareType> SIMD_INLINE __m128i Compare(__m128i a, __m128i b);

        template<> SIMD_INLINE __m128i Compare<CompareGreaterThen>(__m128i a, __m128i b)
        {
            return _mm_andnot_si128(_mm_cmpeq_epi8(_mm_min_epu8(a, b), a), K_INV_ZERO);
        }

        template<> SIMD_INLINE __m128i Compare<CompareLesserThen>(__m128i a, __m128i b)
        {
            return _mm_andnot_si128(_mm_cmpeq_epi8(_mm_max_epu8(a, b), a), K_INV_ZERO);
        }

        template<> SIMD_INLINE __m128i Compare<CompareEqualTo>(__m128i a, __m128i b)
        {
            return _mm_cmpeq_epi8(a, b);
        }

        SIMD_INLINE __m128i Combine(__m128i mask, __m128i positive, __m128i negative)
        {
            return _mm_or_si128(_mm_and_si128(mask, positive), _mm_andnot_si128(mask, negative));
        }

        template <bool align, CompareType compareType> 
        void Binarization(const uchar * src, size_t srcStride, size_t width, size_t height, 
            uchar value, uchar positive, uchar negative, uchar * dst, size_t dstStride)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = Simd::AlignLo(width, A);

            __m128i value_ = _mm_set1_epi8(value);
            __m128i positive_ = _mm_set1_epi8(positive);
            __m128i negative_ = _mm_set1_epi8(negative);
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    const __m128i mask = Compare<compareType>(Load<align>((__m128i*)(src + col)), value_);
                    Store<align>((__m128i*)(dst + col), Combine(mask, positive_, negative_));
                }
                if(alignedWidth != width)
                {
                    const __m128i mask = Compare<compareType>(Load<false>((__m128i*)(src + width - A)), value_);
                    Store<false>((__m128i*)(dst + width - A), Combine(mask, positive_, negative_));
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        template <CompareType compareType> 
        void Binarization(const uchar * src, size_t srcStride, size_t width, size_t height, 
            uchar value, uchar positive, uchar negative, uchar * dst, size_t dstStride)
        {
            if(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                Binarization<true, compareType>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            else
                Binarization<false, compareType>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
        }

        void Binarization(const uchar * src, size_t srcStride, size_t width, size_t height, 
            uchar value, uchar positive, uchar negative, uchar * dst, size_t dstStride, CompareType compareType)
        {
            switch(compareType)
            {
            case CompareGreaterThen:
                return Binarization<CompareGreaterThen>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case CompareLesserThen:
                return Binarization<CompareLesserThen>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case CompareEqualTo:
                return Binarization<CompareEqualTo>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            default:
                assert(0);
            }
        }

        namespace
        {
            struct Buffer
            {
                Buffer(size_t width, size_t edge)
                {
                    size_t size = sizeof(ushort)*(width + 2*edge) + sizeof(uint)*(2*width + 2*edge);
                    _p = Allocate(size);
                    memset(_p, 0, size);
                    sa = (ushort*)_p + edge;
                    s0a0 = (uint*)(sa + width + edge) + edge;
                    sum = (uint*)(s0a0 + width + edge);
                }

                ~Buffer()
                {
                    Free(_p);
                }

                ushort * sa;
                uint * s0a0;
                uint * sum;
            private:
                void *_p;
            };
        }

        template <bool srcAlign, bool dstAlign, CompareType compareType>
        SIMD_INLINE void AddRows(const uchar * src, ushort * sa, const __m128i & value, const __m128i & mask)
        {
            const __m128i inc = _mm_and_si128(Compare<compareType>(Load<srcAlign>((__m128i*)src), value), mask);
            Store<dstAlign>((__m128i*)sa + 0, _mm_add_epi8(Load<dstAlign>((__m128i*)sa + 0), _mm_unpacklo_epi8(inc, mask)));
            Store<dstAlign>((__m128i*)sa + 1, _mm_add_epi8(Load<dstAlign>((__m128i*)sa + 1), _mm_unpackhi_epi8(inc, mask)));
        }

        template <bool srcAlign, bool dstAlign, CompareType compareType>
        SIMD_INLINE void SubRows(const uchar * src, ushort * sa, const __m128i & value, const __m128i & mask)
        {
            const __m128i dec = _mm_and_si128(Compare<compareType>(Load<srcAlign>((__m128i*)src), value), mask);
            Store<dstAlign>((__m128i*)sa + 0, _mm_sub_epi8(Load<dstAlign>((__m128i*)sa + 0), _mm_unpacklo_epi8(dec, mask)));
            Store<dstAlign>((__m128i*)sa + 1, _mm_sub_epi8(Load<dstAlign>((__m128i*)sa + 1), _mm_unpackhi_epi8(dec, mask)));
        }

        template <bool align>
        SIMD_INLINE __m128i CompareSum(const uint * sum, const __m128i & ff_threshold)
        {
            const __m128i mask0 = _mm_cmpgt_epi32(_mm_madd_epi16(Load<align>((__m128i*)sum + 0), ff_threshold), K_ZERO);
            const __m128i mask1 = _mm_cmpgt_epi32(_mm_madd_epi16(Load<align>((__m128i*)sum + 1), ff_threshold), K_ZERO);
            const __m128i mask2 = _mm_cmpgt_epi32(_mm_madd_epi16(Load<align>((__m128i*)sum + 2), ff_threshold), K_ZERO);
            const __m128i mask3 = _mm_cmpgt_epi32(_mm_madd_epi16(Load<align>((__m128i*)sum + 3), ff_threshold), K_ZERO);
            return _mm_packs_epi16(_mm_packs_epi32(mask0, mask1), _mm_packs_epi32(mask2, mask3));
        }

        template <bool align, CompareType compareType>
        void AveragingBinarization(const uchar * src, size_t srcStride, size_t width, size_t height,
            uchar value, size_t neighborhood, uchar threshold, uchar positive, uchar negative, uchar * dst, size_t dstStride)
        {
            assert(width > neighborhood && height > neighborhood && neighborhood < 0x7F);

            const size_t alignedWidth = AlignLo(width, A);

            const __m128i tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            const __m128i ff_threshold = SetInt16(0xFF, -threshold);
            const __m128i _value = _mm_set1_epi8(value);
            const __m128i _positive = _mm_set1_epi8(positive);
            const __m128i _negative = _mm_set1_epi8(negative);

            Buffer buffer(AlignHi(width, A), AlignHi(neighborhood + 1, A));

            for(size_t row = 0; row < neighborhood; ++row)
            {
                const uchar * s = src + row*srcStride;
                for(size_t col = 0; col < alignedWidth; col += A)
                    AddRows<align, true, compareType>(s + col, buffer.sa + col, _value, K8_01);
                if(alignedWidth != width)
                    AddRows<false, false, compareType>(s + width - A, buffer.sa + width - A, _value, tailMask);
            }

            for(size_t row = 0; row < height; ++row)
            {
                if(row < height - neighborhood)
                {
                    const uchar * s = src +  (row + neighborhood)*srcStride;
                    for(size_t col = 0; col < alignedWidth; col += A)
                        AddRows<align, true, compareType>(s + col, buffer.sa + col, _value, K8_01);
                    if(alignedWidth != width)
                        AddRows<false, false, compareType>(s + width - A, buffer.sa + width - A, _value, tailMask);
                }
                if(row > neighborhood)
                {
                    const uchar * s = src + (row - neighborhood - 1)*srcStride;
                    for(size_t col = 0; col < alignedWidth; col += A)
                        SubRows<align, true, compareType>(s + col, buffer.sa + col, _value, K8_01);
                    if(alignedWidth != width)
                        SubRows<false, false, compareType>(s + width - A, buffer.sa + width - A, _value, tailMask);
                }

                for(size_t col = 0; col < width; col += HA)
                {
                    const __m128i sa = Load<true>((__m128i*)(buffer.sa + col));
                    Store<true>((__m128i*)(buffer.s0a0 + col) + 0, _mm_unpacklo_epi8(sa, K_ZERO));
                    Store<true>((__m128i*)(buffer.s0a0 + col) + 1, _mm_unpackhi_epi8(sa, K_ZERO));
                }

                uint sum = 0;
                for(size_t col = 0; col < neighborhood; ++col)
                {
                    sum += buffer.s0a0[col];
                }
                for(size_t col = 0; col < width; ++col)
                {
                    sum += buffer.s0a0[col + neighborhood];
                    sum -= buffer.s0a0[col - neighborhood - 1];
                    buffer.sum[col] = sum;
                }

                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    const __m128i mask = CompareSum<true>(buffer.sum + col, ff_threshold);
                    Store<align>((__m128i*)(dst + col), Combine(mask, _positive, _negative));
                }
                if(alignedWidth != width)
                {
                    const __m128i mask = CompareSum<false>(buffer.sum + width - A, ff_threshold);
                    Store<false>((__m128i*)(dst + width - A), Combine(mask, _positive, _negative));
                }
                dst += dstStride;
            }
        }

        template <CompareType compareType> 
        void AveragingBinarization(const uchar * src, size_t srcStride, size_t width, size_t height,
            uchar value, size_t neighborhood, uchar threshold, uchar positive, uchar negative, uchar * dst, size_t dstStride)
        {
            if(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                AveragingBinarization<true, compareType>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            else
                AveragingBinarization<false, compareType>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
        }

        void AveragingBinarization(const uchar * src, size_t srcStride, size_t width, size_t height,
            uchar value, size_t neighborhood, uchar threshold, uchar positive, uchar negative, 
            uchar * dst, size_t dstStride, CompareType compareType)
        {
            switch(compareType)
            {
            case CompareGreaterThen:
                return AveragingBinarization<CompareGreaterThen>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case CompareLesserThen:
                return AveragingBinarization<CompareLesserThen>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case CompareEqualTo:
                return AveragingBinarization<CompareEqualTo>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            default:
                assert(0);
            }
        }
    }
#endif// SIMD_SSE2_ENABLE

	void Binarization(const uchar * src, size_t srcStride, size_t width, size_t height, 
		uchar value, uchar positive, uchar negative, uchar * dst, size_t dstStride, CompareType compareType)
	{
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::Binarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
		if(Sse2::Enable && width >= Sse2::A)
			Sse2::Binarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
		else
#endif// SIMD_SSE2_ENABLE
			Base::Binarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
	}

    void AveragingBinarization(const uchar * src, size_t srcStride, size_t width, size_t height,
        uchar value, size_t neighborhood, uchar threshold, uchar positive, uchar negative, 
        uchar * dst, size_t dstStride, CompareType compareType)
    {
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::AveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
        if(Sse2::Enable && width >= Sse2::A)
            Sse2::AveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
        else
#endif// SIMD_SSE2_ENABLE
            Base::AveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
    }
}