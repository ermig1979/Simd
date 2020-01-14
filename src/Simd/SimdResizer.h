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
#ifndef __SimdResizer_h__
#define __SimdResizer_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdMath.h"

namespace Simd
{
    struct ResParam
    {
        SimdResizeChannelType type;
        SimdResizeMethodType method;
        size_t srcW, srcH, dstW, dstH, channels, align;

        ResParam(size_t srcW, size_t srcH, size_t dstW, size_t dstH, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method, size_t align)
        {
            this->type = type;
            this->method = method;
            this->srcW = srcW;
            this->srcH = srcH;
            this->dstW = dstW;
            this->dstH = dstH;
            this->channels = channels;
            this->align = align;
        }

        bool IsByteBilinear() const
        {
            return type == SimdResizeChannelByte && method == SimdResizeMethodBilinear;
        }

        bool IsByteArea() const
        {
            return type == SimdResizeChannelByte && method == SimdResizeMethodArea;
        }

        bool IsFloatBilinear() const
        {
            return type == SimdResizeChannelFloat && 
                (method == SimdResizeMethodBilinear || method == SimdResizeMethodCaffeInterp || method == SimdResizeMethodInferenceEngineInterp);
        }
    };

    class Resizer : Deletable
    {
    public:
        Resizer(const ResParam & param)
            : _param(param)
        {
        }

        virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride) = 0;

    protected:
        ResParam _param;
    };

    namespace Base
    {
        class ResizerByteBilinear : public Resizer
        {
        protected:
            Array32i _ax, _ix, _ay, _iy, _bx[2];

            void EstimateIndexAlpha(size_t srcSize, size_t dstSize, size_t channels, int32_t * indices, int32_t * alphas);
        public:
            ResizerByteBilinear(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        const int32_t AREA_SHIFT = 22;
        const int32_t AREA_RANGE = 1 << 11;
        const int32_t AREA_ROUND = 1 << 21;

        class ResizerByteArea : public Resizer
        {
        protected:
            Array32i _ax, _ix, _ay, _iy;

            template<size_t N> void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        public:
            ResizerByteArea(const ResParam & param);

            void EstimateParams(size_t srcSize, size_t dstSize, size_t range, int32_t * alpha, int32_t * index);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        class ResizerFloatBilinear : public Resizer
        {
        protected:
            Array32i _ix, _iy;
            Array32f _ax, _ay, _bx[2];

            void EstimateIndexAlpha(size_t srcSize, size_t dstSize, size_t channels, int32_t * indices, float * alphas);

            virtual void Run(const float * src, size_t srcStride, float * dst, size_t dstStride);

        public:
            ResizerFloatBilinear(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }

#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        class ResizerFloatBilinear : public Base::ResizerFloatBilinear
        {
            virtual void Run(const float * src, size_t srcStride, float * dst, size_t dstStride);
        public:
            ResizerFloatBilinear(const ResParam & param);
        };

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_SSE_ENABLE 

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        class ResizerByteBilinear : public Base::ResizerByteBilinear
        {
        protected:
            Array16i _ax;
            Array8u _bx[2];

            void EstimateParams();
            template<size_t N> void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        public:
            ResizerByteBilinear(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        class ResizerByteArea : public Base::ResizerByteArea
        {
        protected:
            Array32i _by;

            template<size_t N> void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        public:
            ResizerByteArea(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_SSE2_ENABLE 

#ifdef SIMD_SSSE3_ENABLE    
    namespace Ssse3
    {
        class ResizerByteBilinear : public Sse2::ResizerByteBilinear
        {
        protected:
            Array8u _ax;
            size_t _blocks;
            struct Idx
            {
                int32_t src, dst;
                uint8_t shuffle[A];
            };
            Array<Idx> _ixg;

            size_t BlockCountMax(size_t align);
            void EstimateParams();
            template<size_t N> void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
            void RunG(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        public:
            ResizerByteBilinear(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_SSSE3_ENABLE 

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class ResizerByteArea : public Sse2::ResizerByteArea
        {
        protected:
            template<size_t N> void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        public:
            ResizerByteArea(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_SSE41_ENABLE

#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        class ResizerFloatBilinear : public Base::ResizerFloatBilinear
        {
            virtual void Run(const float * src, size_t srcStride, float * dst, size_t dstStride);
        public:
            ResizerFloatBilinear(const ResParam & param);
        };

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_AVX_ENABLE 

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <class Idx> SIMD_INLINE void ResizerByteBilinearLoadGrayInterpolated(const uint8_t * src, const Idx & index, const uint8_t * alpha, uint8_t * dst)
        {
            __m256i _src = _mm256_loadu_si256((__m256i*)(src + index.src));
            __m256i _shuffle = _mm256_loadu_si256((__m256i*)&index.shuffle);
            __m256i _alpha = _mm256_loadu_si256((__m256i*)(alpha + index.dst));
            _mm256_storeu_si256((__m256i*)(dst + index.dst), _mm256_maddubs_epi16(Avx2::Shuffle(_src, _shuffle), _alpha));
        }

        class ResizerByteBilinear : public Ssse3::ResizerByteBilinear
        {
        protected:
            struct Idx
            {
                int32_t src, dst;
                uint8_t shuffle[A];
            };
            Array<Idx> _ixg;

            void EstimateParams();
            template<size_t N> void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
            void RunG(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        public:
            ResizerByteBilinear(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        class ResizerByteArea : public Sse41::ResizerByteArea
        {
        protected:
            template<size_t N> void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        public:
            ResizerByteArea(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        class ResizerFloatBilinear : public Base::ResizerFloatBilinear
        {
            virtual void Run(const float * src, size_t srcStride, float * dst, size_t dstStride);
        public:
            ResizerFloatBilinear(const ResParam & param);
        };

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_AVX2_ENABLE 

#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        class ResizerFloatBilinear : public Base::ResizerFloatBilinear
        {
            virtual void Run(const float * src, size_t srcStride, float * dst, size_t dstStride);
        public:
            ResizerFloatBilinear(const ResParam & param);
        };

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_AVX512F_ENABLE 

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        class ResizerByteBilinear : public Avx2::ResizerByteBilinear
        {
        protected:
            template<size_t N> void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
            void RunG(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        public:
            ResizerByteBilinear(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        class ResizerByteArea : public Avx2::ResizerByteArea
        {
        protected:
            template<size_t N> void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        public:
            ResizerByteArea(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_AVX512BW_ENABLE 

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        class ResizerByteBilinear : public Base::ResizerByteBilinear
        {
        protected:
            Array8u _ax, _bx[2];
            size_t _blocks;
            struct Idx
            {
                int32_t src, dst;
                uint8_t shuffle[A];
            };
            Array<Idx> _ixg;

            size_t BlockCountMax(size_t align);
            void EstimateParams();
            template<size_t N> void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
            void RunG(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        public:
            ResizerByteBilinear(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        class ResizerByteArea : public Base::ResizerByteArea
        {
        protected:
            Array32i _by;

            template<size_t N> void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        public:
            ResizerByteArea(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        class ResizerFloatBilinear : public Base::ResizerFloatBilinear
        {
            virtual void Run(const float * src, size_t srcStride, float * dst, size_t dstStride);
        public:
            ResizerFloatBilinear(const ResParam & param);
        };

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_NEON_ENABLE 
}
#endif//__SimdResizer_h__
