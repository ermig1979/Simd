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
#include "Simd/SimdMemory.h"
#include "Simd/SimdImageLoad.h"
#include "Simd/SimdSve2.h"

#include <memory>

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE    
    namespace Sve2
    {
        ImagePgmTxtLoader::ImagePgmTxtLoader(const ImageLoaderParam& param)
            : Neon::ImagePgmTxtLoader(param)
        {
        }

        void ImagePgmTxtLoader::SetConverters()
        {
            Neon::ImagePgmTxtLoader::SetConverters();
            if (_image.width >= svcntb())
            {
                switch (_param.format)
                {
                case SimdPixelFormatBgr24: _toAny = Sve2::GrayToBgr; break;
                case SimdPixelFormatBgra32: _toBgra = Sve2::GrayToBgra; break;
                case SimdPixelFormatRgb24: _toAny = Sve2::GrayToBgr; break;
                case SimdPixelFormatRgba32: _toBgra = Sve2::GrayToBgra; break;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        ImagePgmBinLoader::ImagePgmBinLoader(const ImageLoaderParam& param)
            : Neon::ImagePgmBinLoader(param)
        {
        }

        void ImagePgmBinLoader::SetConverters()
        {
            Neon::ImagePgmBinLoader::SetConverters();
            if (_image.width >= svcntb())
            {
                switch (_param.format)
                {
                case SimdPixelFormatBgr24: _toAny = Sve2::GrayToBgr; break;
                case SimdPixelFormatBgra32: _toBgra = Sve2::GrayToBgra; break;
                case SimdPixelFormatRgb24: _toAny = Sve2::GrayToBgr; break;
                case SimdPixelFormatRgba32: _toBgra = Sve2::GrayToBgra; break;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        ImagePpmTxtLoader::ImagePpmTxtLoader(const ImageLoaderParam& param)
            : Neon::ImagePpmTxtLoader(param)
        {
        }

        void ImagePpmTxtLoader::SetConverters()
        {
            Neon::ImagePpmTxtLoader::SetConverters();
            if (_image.width >= svcntb())
            {
                switch (_param.format)
                {
                case SimdPixelFormatGray8: _toAny = Sve2::RgbToGray; break;
                case SimdPixelFormatBgr24: _toAny = Sve2::BgrToRgb; break;
                case SimdPixelFormatBgra32: _toBgra = Sve2::RgbToBgra; break;
                case SimdPixelFormatRgba32: _toBgra = Sve2::BgrToBgra; break;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        ImagePpmBinLoader::ImagePpmBinLoader(const ImageLoaderParam& param)
            : Neon::ImagePpmBinLoader(param)
        {
        }

        void ImagePpmBinLoader::SetConverters()
        {
            Neon::ImagePpmBinLoader::SetConverters();
            if (_image.width >= svcntb())
            {
                switch (_param.format)
                {
                case SimdPixelFormatGray8: _toAny = Sve2::RgbToGray; break;
                case SimdPixelFormatBgr24: _toAny = Sve2::BgrToRgb; break;
                case SimdPixelFormatBgra32: _toBgra = Sve2::RgbToBgra; break;
                case SimdPixelFormatRgba32: _toBgra = Sve2::BgrToBgra; break;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        ImageBmpLoader::ImageBmpLoader(const ImageLoaderParam& param)
            : Neon::ImageBmpLoader(param)
        {
        }

        void ImageBmpLoader::SetConverters()
        {
            Neon::ImageBmpLoader::SetConverters();
            if (_width < svcntb())
                return;
            if (_bpp == 8)
            {
                switch (_param.format)
                {
                case SimdPixelFormatBgr24: _toAny = Sve2::GrayToBgr; break;
                case SimdPixelFormatRgb24: _toAny = Sve2::GrayToBgr; break;
                case SimdPixelFormatBgra32: _toBgra = Sve2::GrayToBgra; break;
                case SimdPixelFormatRgba32: _toBgra = Sve2::GrayToBgra; break;
                default: break;
                }
                return;
            }
            switch (_param.format)
            {
            case SimdPixelFormatGray8: _toAny = (_bpp == 32 ? Sve2::BgraToGray : (_bpp == 24 ? Sve2::BgrToGray : NULL)); break;
            case SimdPixelFormatBgr24: _toAny = (_bpp == 32 ? (ToAnyPtr)Sve2::BgraToBgr : NULL); break;
            case SimdPixelFormatRgb24: _toAny = (_bpp == 32 ? Sve2::BgraToRgb : Sve2::BgrToRgb); break;
            case SimdPixelFormatBgra32: _toBgra = (_bpp == 32 ? NULL : (ToBgraPtr)Sve2::BgrToBgra); break;
            case SimdPixelFormatRgba32:
                if (_bpp == 32)
                    _toAny = Sve2::BgraToRgba;
                else
                    _toBgra = (ToBgraPtr)Sve2::RgbToBgra;
                break;
            default: break;
            }
        }

        //-------------------------------------------------------------------------------------------------

        ImageLoader* CreateImageLoader(const ImageLoaderParam& param)
        {
            switch (param.file)
            {
            case SimdImageFilePgmTxt: return new ImagePgmTxtLoader(param);
            case SimdImageFilePgmBin: return new ImagePgmBinLoader(param);
            case SimdImageFilePpmTxt: return new ImagePpmTxtLoader(param);
            case SimdImageFilePpmBin: return new ImagePpmBinLoader(param);
            case SimdImageFilePng: return new Base::ImagePngLoader(param);
            case SimdImageFileJpeg: return new Base::ImageJpegLoader(param);
            case SimdImageFileBmp: return new ImageBmpLoader(param);
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
#endif
}
