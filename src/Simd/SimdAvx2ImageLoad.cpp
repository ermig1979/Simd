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
#include "Simd/SimdImageLoad.h"
#include "Simd/SimdAvx2.h"

#include <memory>

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        ImagePgmTxtLoader::ImagePgmTxtLoader(const ImageLoaderParam& param)
            : Sse41::ImagePgmTxtLoader(param)
        {
        }

        void ImagePgmTxtLoader::SetConverters()
        {
            Sse41::ImagePgmTxtLoader::SetConverters();
            if (_image.width >= A)
            {
                switch (_param.format)
                {
                case SimdPixelFormatBgr24: _toAny = Avx2::GrayToBgr; break;
                case SimdPixelFormatBgra32: _toBgra = Avx2::GrayToBgra; break;
                case SimdPixelFormatRgb24: _toAny = Avx2::GrayToBgr; break;
                case SimdPixelFormatRgba32: _toBgra = Avx2::GrayToBgra; break;
                default: break;
                }
            }
        }

        //---------------------------------------------------------------------

        ImagePgmBinLoader::ImagePgmBinLoader(const ImageLoaderParam& param)
            : Sse41::ImagePgmBinLoader(param)
        {
        }

        void ImagePgmBinLoader::SetConverters()
        {
            Sse41::ImagePgmBinLoader::SetConverters();
            if (_image.width >= A)
            {
                switch (_param.format)
                {
                case SimdPixelFormatBgr24: _toAny = Avx2::GrayToBgr; break;
                case SimdPixelFormatBgra32: _toBgra = Avx2::GrayToBgra; break;
                case SimdPixelFormatRgb24: _toAny = Avx2::GrayToBgr; break;
                case SimdPixelFormatRgba32: _toBgra = Avx2::GrayToBgra; break;
                default: break;
                }
            }
        }

        //---------------------------------------------------------------------

        ImagePpmTxtLoader::ImagePpmTxtLoader(const ImageLoaderParam& param)
            : Sse41::ImagePpmTxtLoader(param)
        {
        }

        void ImagePpmTxtLoader::SetConverters()
        {
            Sse41::ImagePpmTxtLoader::SetConverters();
            if (_image.width >= A)
            {
                switch (_param.format)
                {
                case SimdPixelFormatGray8: _toAny = Avx2::RgbToGray; break;
                case SimdPixelFormatBgr24: _toAny = Avx2::BgrToRgb; break;
                case SimdPixelFormatBgra32: _toBgra = Avx2::RgbToBgra; break;
                case SimdPixelFormatRgba32: _toBgra = Avx2::BgrToBgra; break;
                default: break;
                }
            }
        }

        //---------------------------------------------------------------------

        ImagePpmBinLoader::ImagePpmBinLoader(const ImageLoaderParam& param)
            : Sse41::ImagePpmBinLoader(param)
        {
        }

        void ImagePpmBinLoader::SetConverters()
        {
            Sse41::ImagePpmBinLoader::SetConverters();
            if (_image.width >= A)
            {
                switch (_param.format)
                {
                case SimdPixelFormatGray8: _toAny = Avx2::RgbToGray; break;
                case SimdPixelFormatBgr24: _toAny = Avx2::BgrToRgb; break;
                case SimdPixelFormatBgra32: _toBgra = Avx2::RgbToBgra; break;
                case SimdPixelFormatRgba32: _toBgra = Avx2::BgrToBgra; break;
                default: break;
                }
            }
        }

        //---------------------------------------------------------------------

        ImageLoader* CreateImageLoader(const ImageLoaderParam& param)
        {
            switch (param.file)
            {
            case SimdImageFilePgmTxt: return new ImagePgmTxtLoader(param);
            case SimdImageFilePgmBin: return new ImagePgmBinLoader(param);
            case SimdImageFilePpmTxt: return new ImagePpmTxtLoader(param);
            case SimdImageFilePpmBin: return new ImagePpmBinLoader(param);
            case SimdImageFilePng: return new Sse41::ImagePngLoader(param);
            case SimdImageFileJpeg: return new Base::ImageJpegLoader(param);
            default:
                return NULL;
            }
        }

        uint8_t* ImageLoadFromMemory(const uint8_t* data, size_t size, size_t* stride, size_t* width, size_t* height, SimdPixelFormatType* format)
        {
            ImageLoaderParam param(data, size, *format);
            if (param.Validate())
            {
                Holder<ImageLoader> loader(CreateImageLoader(param));
                if (loader)
                {
                    if (loader->FromStream())
                        return loader->Release(stride, width, height, format);
                }
            }
            return NULL;
        }
    }
#endif// SIMD_AVX2_ENABLE
}
