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
#include "Simd/SimdAvx512bw.h"

#include <memory>

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        ImagePgmTxtLoader::ImagePgmTxtLoader(const ImageLoaderParam& param)
            : Avx2::ImagePgmTxtLoader(param)
        {
        }

        void ImagePgmTxtLoader::SetConverters()
        {
            Avx2::ImagePgmTxtLoader::SetConverters();
            if (_image.width >= A)
            {
                switch (_param.format)
                {
                case SimdPixelFormatBgr24: _toAny = Avx512bw::GrayToBgr; break;
                case SimdPixelFormatBgra32: _toBgra = Avx512bw::GrayToBgra; break;
                case SimdPixelFormatRgb24: _toAny = Avx512bw::GrayToBgr; break;
                default: break;
                }
            }
        }

        //---------------------------------------------------------------------

        ImagePgmBinLoader::ImagePgmBinLoader(const ImageLoaderParam& param)
            : Avx2::ImagePgmBinLoader(param)
        {
        }

        void ImagePgmBinLoader::SetConverters()
        {
            Avx2::ImagePgmBinLoader::SetConverters();
            if (_image.width >= A)
            {
                switch (_param.format)
                {
                case SimdPixelFormatBgr24: _toAny = Avx512bw::GrayToBgr; break;
                case SimdPixelFormatBgra32: _toBgra = Avx512bw::GrayToBgra; break;
                case SimdPixelFormatRgb24: _toAny = Avx512bw::GrayToBgr; break;
                default: break;
                }
            }
        }

        //---------------------------------------------------------------------

        ImagePpmTxtLoader::ImagePpmTxtLoader(const ImageLoaderParam& param)
            : Avx2::ImagePpmTxtLoader(param)
        {
        }

        void ImagePpmTxtLoader::SetConverters()
        {
            Avx2::ImagePpmTxtLoader::SetConverters();
            if (_image.width >= A)
            {
                switch (_param.format)
                {
                case SimdPixelFormatGray8: _toAny = Avx512bw::RgbToGray; break;
                case SimdPixelFormatBgr24: _toAny = Avx512bw::BgrToRgb; break;
                case SimdPixelFormatBgra32: _toBgra = Avx512bw::RgbToBgra; break;
                default: break;
                }
            }
        }

        //---------------------------------------------------------------------

        ImagePpmBinLoader::ImagePpmBinLoader(const ImageLoaderParam& param)
            : Avx2::ImagePpmBinLoader(param)
        {
        }

        void ImagePpmBinLoader::SetConverters()
        {
            Avx2::ImagePpmBinLoader::SetConverters();
            if (_image.width >= A)
            {
                switch (_param.format)
                {
                case SimdPixelFormatGray8: _toAny = Avx512bw::RgbToGray; break;
                case SimdPixelFormatBgr24: _toAny = Avx512bw::BgrToRgb; break;
                case SimdPixelFormatBgra32: _toBgra = Avx512bw::RgbToBgra; break;
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
            case SimdImageFilePng: return new Base::ImagePngLoader(param);
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
#endif// SIMD_AVX512BW_ENABLE
}
