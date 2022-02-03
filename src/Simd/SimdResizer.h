/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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

#define SIMD_RESIZER_BICUBIC_BITS 7 // 7, 11

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

        bool IsNearest() const
        {
            return method == SimdResizeMethodNearest || method == SimdResizeMethodNearestPytorch || srcW == 1 || srcH == 1;
        }

        bool IsByteBilinear() const
        {
            return type == SimdResizeChannelByte && (method == SimdResizeMethodBilinear || 
                (method == SimdResizeMethodBicubic && (srcW == 2 || srcH == 2)));
        }

        bool IsShortBilinear() const
        {
            return type == SimdResizeChannelShort && method == SimdResizeMethodBilinear;
        }

        bool IsFloatBilinear() const
        {
            return type == SimdResizeChannelFloat && 
                (method == SimdResizeMethodBilinear || method == SimdResizeMethodBilinearCaffe || method == SimdResizeMethodBilinearPytorch);
        }

        bool IsByteBicubic() const
        {
            return type == SimdResizeChannelByte && method == SimdResizeMethodBicubic;
        }

        bool IsByteArea() const
        {
            return type == SimdResizeChannelByte && method == SimdResizeMethodArea;
        }

        size_t ChannelSize() const
        {
            static const size_t sizes[3] = { 1, 2, 4 };
            return sizes[(int)type];
        }

        size_t PixelSize() const
        {
            return ChannelSize() * channels;
        }
    };

    //---------------------------------------------------------------------------------------------

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

    //---------------------------------------------------------------------------------------------

    namespace Base
    {
        class ResizerNearest : public Resizer
        {
            void Resize(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
            template<size_t N> void Resize(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        protected:
            size_t _pixelSize;
            Array32i _ix, _iy;

            void EstimateIndex(size_t srcSize, size_t dstSize, size_t channelSize, size_t channels, int32_t* indices);

            void EstimateParams();
        public:
            ResizerNearest(const ResParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        };

        //---------------------------------------------------------------------------------------------

        class ResizerByteBilinear : public Resizer
        {
        protected:
            Array32i _ax, _ix, _ay, _iy, _bx[2];

            void EstimateIndexAlpha(size_t srcSize, size_t dstSize, size_t channels, int32_t * indices, int32_t * alphas);
        public:
            ResizerByteBilinear(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        //---------------------------------------------------------------------------------------------

        class ResizerShortBilinear : public Resizer
        {
        protected:
            Array32i _ix, _iy;
            Array32f _ax, _ay, _bx[2];

            void EstimateIndexAlpha(size_t srcSize, size_t dstSize, size_t channels, int32_t* indices, float* alphas);

            template<size_t N> void RunB(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride);
            template<size_t N> void RunS(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride);

            virtual void Run(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride);

        public:
            ResizerShortBilinear(const ResParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        };

        //---------------------------------------------------------------------------------------------

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

        //---------------------------------------------------------------------------------------------

        const int32_t BICUBIC_RANGE = 1 << SIMD_RESIZER_BICUBIC_BITS;
        const int32_t BICUBIC_SHIFT = SIMD_RESIZER_BICUBIC_BITS * 2;
        const int32_t BICUBIC_ROUND = 1 << (BICUBIC_SHIFT - 1);

        class ResizerByteBicubic : public Resizer
        {
        protected:
            Array32i _ix, _iy, _ax, _ay, _bx[4];
            size_t _xn, _xt, _sxl;

            void EstimateIndexAlpha(size_t sizeS, size_t sizeD, size_t N, Array32i& index, Array32i& alpha);

            void Init(bool sparse);

            template<int N> void RunS(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
            template<int N> void RunB(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        public:
            ResizerByteBicubic(const ResParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        };

        //---------------------------------------------------------------------------------------------

        const int32_t AREA_SHIFT = 22;
        const int32_t AREA_RANGE = 1 << 11;
        const int32_t AREA_ROUND = 1 << 21;

        class ResizerByteArea : public Resizer
        {
        protected:
            Array32i _ax, _ix, _ay, _iy;

            template<size_t N> void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        public:
            ResizerByteArea(const ResParam& param);

            void EstimateParams(size_t srcSize, size_t dstSize, size_t range, int32_t* alpha, int32_t* index);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        };
        
        //---------------------------------------------------------------------------------------------

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }

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

        //---------------------------------------------------------------------------------------------

        class ResizerFloatBilinear : public Base::ResizerFloatBilinear
        {
            virtual void Run(const float * src, size_t srcStride, float * dst, size_t dstStride);
        public:
            ResizerFloatBilinear(const ResParam & param);
        };

        //---------------------------------------------------------------------------------------------

        class ResizerByteArea : public Base::ResizerByteArea
        {
        protected:
            Array32i _by;

            template<size_t N> void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        public:
            ResizerByteArea(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        //---------------------------------------------------------------------------------------------

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_SSE2_ENABLE 

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class ResizerNearest : public Base::ResizerNearest
        {
        protected:
            size_t _blocks;
            struct IndexShuffle16x1
            {
                int32_t src, dst;
                uint8_t shuffle[A];
            };
            Array<IndexShuffle16x1> _ix16x1;
            __m128i _tail16x1;

            size_t BlockCountMax(size_t align);
            void EstimateParams();
            void Shuffle16x1(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
            void Resize12(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        public:
            ResizerNearest(const ResParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        };   

        //---------------------------------------------------------------------------------------------
        
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

        //---------------------------------------------------------------------------------------------

        class ResizerShortBilinear : public Base::ResizerShortBilinear
        {
        protected:
            template<size_t N> void RunB(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride);
            template<size_t N> void RunS(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride);

            virtual void Run(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride);
        public:
            ResizerShortBilinear(const ResParam& param);
        };
        
        //---------------------------------------------------------------------------------------------

        class ResizerByteBicubic : public Base::ResizerByteBicubic
        {
        protected:
            Array8i _ax;

            void EstimateIndexAlphaY();
            void EstimateIndexAlphaX();

            void Init(bool sparse);

            template<int N> void RunS(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
            template<int N> void RunB(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        public:
            ResizerByteBicubic(const ResParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        };

        //---------------------------------------------------------------------------------------------

        class ResizerByteArea : public Sse2::ResizerByteArea
        {
        protected:
            template<size_t N> void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        public:
            ResizerByteArea(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        //---------------------------------------------------------------------------------------------

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

        //---------------------------------------------------------------------------------------------

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_AVX_ENABLE 

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class ResizerNearest : public Sse41::ResizerNearest
        {
        protected:
            void EstimateParams();
            void Gather2(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
            void Gather3(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
            void Gather4(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
            void Gather8(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        public:
            ResizerNearest(const ResParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        };

        //---------------------------------------------------------------------------------------------

        class ResizerByteBilinear : public Sse41::ResizerByteBilinear
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

        class ResizerShortBilinear : public Sse41::ResizerShortBilinear
        {
        protected:
            template<size_t N> void RunB(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride);
            template<size_t N> void RunS(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride);

            virtual void Run(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride);
        public:
            ResizerShortBilinear(const ResParam& param);
        };

        class ResizerFloatBilinear : public Base::ResizerFloatBilinear
        {
            virtual void Run(const float * src, size_t srcStride, float * dst, size_t dstStride);
        public:
            ResizerFloatBilinear(const ResParam & param);
        };

        //---------------------------------------------------------------------------------------------

        class ResizerByteBicubic : public Sse41::ResizerByteBicubic
        {
        protected:
            template<int N> void RunS(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
            template<int N> void RunB(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        public:
            ResizerByteBicubic(const ResParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        };

        //---------------------------------------------------------------------------------------------

        class ResizerByteArea : public Sse41::ResizerByteArea
        {
        protected:
            template<size_t N> void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        public:
            ResizerByteArea(const ResParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        };

        //---------------------------------------------------------------------------------------------

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_AVX2_ENABLE 

#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        class ResizerNearest : public Avx2::ResizerNearest
        {
        protected:
            void Gather4(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
            void Gather8(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        public:
            ResizerNearest(const ResParam& param);

            static bool Preferable(const ResParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        };

        //---------------------------------------------------------------------------------------------

        class ResizerFloatBilinear : public Base::ResizerFloatBilinear
        {
            virtual void Run(const float * src, size_t srcStride, float * dst, size_t dstStride);
        public:
            ResizerFloatBilinear(const ResParam & param);
        };

        //---------------------------------------------------------------------------------------------

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_AVX512F_ENABLE 

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        class ResizerNearest : public Avx512f::ResizerNearest
        {
        protected:
            struct IndexShuffle32x2
            {
                int32_t src, dst;
                uint16_t shuffle[HA];
            };
            Array<IndexShuffle32x2> _ix32x2;
            __mmask32 _tail32x2;

            void EstimateParams();
            void Shuffle32x2(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        public:
            ResizerNearest(const ResParam& param);

            static bool Preferable(const ResParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        };

        //---------------------------------------------------------------------------------------------

        class ResizerByteBilinear : public Avx2::ResizerByteBilinear
        {
        protected:
            template<size_t N> void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
            void RunG(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        public:
            ResizerByteBilinear(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        class ResizerShortBilinear : public Avx2::ResizerShortBilinear
        {
        protected:
            template<size_t N> void RunB(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride);
            template<size_t N> void RunS(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride);

            virtual void Run(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride);
        public:
            ResizerShortBilinear(const ResParam& param);
        };

        //---------------------------------------------------------------------------------------------

        class ResizerByteBicubic : public Avx2::ResizerByteBicubic
        {
        protected:
            template<int N> void RunS(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
            template<int N> void RunB(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        public:
            ResizerByteBicubic(const ResParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        };

        //---------------------------------------------------------------------------------------------

        class ResizerByteArea : public Avx2::ResizerByteArea
        {
        protected:
            template<size_t N> void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        public:
            ResizerByteArea(const ResParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        //---------------------------------------------------------------------------------------------

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

        class ResizerShortBilinear : public Base::ResizerShortBilinear
        {
        protected:
            template<size_t N> void RunB(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride);
            template<size_t N> void RunS(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride);

            virtual void Run(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride);
        public:
            ResizerShortBilinear(const ResParam& param);
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
