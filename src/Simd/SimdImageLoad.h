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
#ifndef __SimdImageLoad_h__
#define __SimdImageLoad_h__

#include "Simd/SimdMemoryStream.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdAlignment.h"

#include "Simd/SimdView.hpp"

#include <vector>

namespace Simd
{
    typedef uint8_t* (*ImageLoadFromMemoryPtr)(const uint8_t* data, size_t size, size_t* stride, size_t* width, size_t* height, SimdPixelFormatType* format);

    uint8_t* ImageLoadFromFile(const ImageLoadFromMemoryPtr loader, const char* path, size_t* stride, size_t* width, size_t* height, SimdPixelFormatType* format);

    //-------------------------------------------------------------------------

    struct ImageLoaderParam
    {
        const uint8_t* data;
        size_t size;
        SimdImageFileType file;
        SimdPixelFormatType format;

        ImageLoaderParam(const uint8_t* d, size_t s, SimdPixelFormatType f);

        bool Validate();
    };

    class ImageLoader
    {
    protected:
        typedef Simd::View<Simd::Allocator> Image;

        ImageLoaderParam _param;
        InputMemoryStream _stream;
        Image _image;
        
    public:
        ImageLoader(const ImageLoaderParam& param)
            : _param(param)
            , _stream(_param.data, _param.size)
        {
        }

        virtual ~ImageLoader()
        {
        }

        virtual bool FromStream() = 0;

        SIMD_INLINE uint8_t* Release(size_t* stride, size_t* width, size_t* height, SimdPixelFormatType* format)
        {
            *stride = _image.stride;
            *width = _image.width;
            *height = _image.height;
            *format = (SimdPixelFormatType)_image.format;
            return _image.Release();
        }
    };

    namespace Base
    {
        class ImagePxmLoader : public ImageLoader
        {
        public:
            ImagePxmLoader(const ImageLoaderParam& param);

        protected:
            typedef void (*ToAnyPtr)(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride);
            typedef void (*ToBgraPtr)(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha);
            ToAnyPtr _toAny;
            ToBgraPtr _toBgra;
            Array8u _buffer;
            size_t _block, _size;

            bool ReadHeader(size_t version);
            virtual void SetConverters() = 0;
        };

        class ImagePgmTxtLoader : public ImagePxmLoader
        {
        public:
            ImagePgmTxtLoader(const ImageLoaderParam& param);

            virtual bool FromStream();

        protected:
            virtual void SetConverters();
        };

        class ImagePgmBinLoader : public ImagePxmLoader
        {
        public:
            ImagePgmBinLoader(const ImageLoaderParam& param);

            virtual bool FromStream();

        protected:
            virtual void SetConverters();
        };

        class ImagePpmTxtLoader : public ImagePxmLoader
        {
        public:
            ImagePpmTxtLoader(const ImageLoaderParam& param);

            virtual bool FromStream();

        protected:
            virtual void SetConverters();
        };

        class ImagePpmBinLoader : public ImagePxmLoader
        {
        public:
            ImagePpmBinLoader(const ImageLoaderParam& param);

            virtual bool FromStream();

        protected:
            virtual void SetConverters();
        };

        class ImagePngLoader : public ImageLoader
        {
        public:
            ImagePngLoader(const ImageLoaderParam& param);

            virtual bool FromStream();

        protected:
            typedef void (*ToAny8Ptr)(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride);
            typedef void (*ToBgra8Ptr)(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha);
            typedef void (*ToAny16Ptr)(const uint16_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride);
            typedef void (*ToBgra16Ptr)(const uint16_t* src, size_t width, size_t height, size_t srcStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha);
            ToAny8Ptr _toAny8;
            ToBgra8Ptr _toBgra8, _bgrToBgra;
            ToAny16Ptr _toAny16;
            ToBgra16Ptr _toBgra16;

            virtual void SetConverters();
        private:
            bool _first, _hasTrans, _iPhone;
            uint32_t _width, _height, _channels;
            uint16_t _tc16[3];
            uint8_t _depth, _color, _interlace, _paletteChannels, _tc[3];
            Array8u _palette, _idat;

            struct Chunk
            {
                uint32_t size;
                uint32_t type;
                uint32_t offs;
            };
            typedef std::vector<Chunk> Chunks;
            Chunks _idats;

            bool ParseFile();
            bool CheckHeader();
            bool ReadChunk(Chunk& chunk);
            bool ReadHeader(const Chunk & chunk);
            bool ReadPalette(const Chunk& chunk);
            bool ReadTransparency(const Chunk& chunk);
            bool ReadData(const Chunk& chunk);
            InputMemoryStream MergedDataStream();
        };

        class ImageJpegLoader : public ImageLoader
        {
        public:
            ImageJpegLoader(const ImageLoaderParam& param);

            virtual bool FromStream();
        };

        //---------------------------------------------------------------------

        uint8_t* ImageLoadFromMemory(const uint8_t* data, size_t size, size_t* stride, size_t* width, size_t* height, SimdPixelFormatType* format);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class ImagePgmTxtLoader : public Base::ImagePgmTxtLoader
        {
        public:
            ImagePgmTxtLoader(const ImageLoaderParam& param);

        protected:
            virtual void SetConverters();
        };

        class ImagePgmBinLoader : public Base::ImagePgmBinLoader
        {
        public:
            ImagePgmBinLoader(const ImageLoaderParam& param);

        protected:
            virtual void SetConverters();
        };

        class ImagePpmTxtLoader : public Base::ImagePpmTxtLoader
        {
        public:
            ImagePpmTxtLoader(const ImageLoaderParam& param);

        protected:
            virtual void SetConverters();
        };

        class ImagePpmBinLoader : public Base::ImagePpmBinLoader
        {
        public:
            ImagePpmBinLoader(const ImageLoaderParam& param);

        protected:
            virtual void SetConverters();
        };

        class ImagePngLoader : public Base::ImagePngLoader
        {
        public:
            ImagePngLoader(const ImageLoaderParam& param);

            virtual bool FromStream();
        };

        //---------------------------------------------------------------------

        uint8_t* ImageLoadFromMemory(const uint8_t* data, size_t size, size_t* stride, size_t* width, size_t* height, SimdPixelFormatType* format);
    }
#endif// SIMD_SSE41_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class ImagePgmTxtLoader : public Sse41::ImagePgmTxtLoader
        {
        public:
            ImagePgmTxtLoader(const ImageLoaderParam& param);

        protected:
            virtual void SetConverters();
        };

        class ImagePgmBinLoader : public Sse41::ImagePgmBinLoader
        {
        public:
            ImagePgmBinLoader(const ImageLoaderParam& param);

        protected:
            virtual void SetConverters();
        };

        class ImagePpmTxtLoader : public Sse41::ImagePpmTxtLoader
        {
        public:
            ImagePpmTxtLoader(const ImageLoaderParam& param);

        protected:
            virtual void SetConverters();
        };

        class ImagePpmBinLoader : public Sse41::ImagePpmBinLoader
        {
        public:
            ImagePpmBinLoader(const ImageLoaderParam& param);

        protected:
            virtual void SetConverters();
        };

        //---------------------------------------------------------------------

        uint8_t* ImageLoadFromMemory(const uint8_t* data, size_t size, size_t* stride, size_t* width, size_t* height, SimdPixelFormatType* format);
    }
#endif// SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        class ImagePgmTxtLoader : public Avx2::ImagePgmTxtLoader
        {
        public:
            ImagePgmTxtLoader(const ImageLoaderParam& param);

        protected:
            virtual void SetConverters();
        };

        class ImagePgmBinLoader : public Avx2::ImagePgmBinLoader
        {
        public:
            ImagePgmBinLoader(const ImageLoaderParam& param);

        protected:
            virtual void SetConverters();
        };

        class ImagePpmTxtLoader : public Avx2::ImagePpmTxtLoader
        {
        public:
            ImagePpmTxtLoader(const ImageLoaderParam& param);

        protected:
            virtual void SetConverters();
        };

        class ImagePpmBinLoader : public Avx2::ImagePpmBinLoader
        {
        public:
            ImagePpmBinLoader(const ImageLoaderParam& param);

        protected:
            virtual void SetConverters();
        };

        //---------------------------------------------------------------------

        uint8_t* ImageLoadFromMemory(const uint8_t* data, size_t size, size_t* stride, size_t* width, size_t* height, SimdPixelFormatType* format);
    }
#endif// SIMD_AVX512BW_ENABLE

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        class ImagePgmTxtLoader : public Base::ImagePgmTxtLoader
        {
        public:
            ImagePgmTxtLoader(const ImageLoaderParam& param);

        protected:
            virtual void SetConverters();
        };

        class ImagePgmBinLoader : public Base::ImagePgmBinLoader
        {
        public:
            ImagePgmBinLoader(const ImageLoaderParam& param);

        protected:
            virtual void SetConverters();
        };

        class ImagePpmTxtLoader : public Base::ImagePpmTxtLoader
        {
        public:
            ImagePpmTxtLoader(const ImageLoaderParam& param);

        protected:
            virtual void SetConverters();
        };

        class ImagePpmBinLoader : public Base::ImagePpmBinLoader
        {
        public:
            ImagePpmBinLoader(const ImageLoaderParam& param);

        protected:
            virtual void SetConverters();
        };

        //---------------------------------------------------------------------

        uint8_t* ImageLoadFromMemory(const uint8_t* data, size_t size, size_t* stride, size_t* width, size_t* height, SimdPixelFormatType* format);
    }
#endif// SIMD_NEON_ENABLE
}

#endif//__SimdImageLoad_h__
