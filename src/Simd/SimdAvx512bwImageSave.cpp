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
#include "Simd/SimdImageSave.h"
#include "Simd/SimdAvx512bw.h"

#include <memory>

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        ImagePgmTxtSaver::ImagePgmTxtSaver(const ImageSaverParam& param)
            : Avx2::ImagePgmTxtSaver(param)
        {
            switch (_param.format)
            {
            case SimdPixelFormatBgr24: _convert = Avx512bw::BgrToGray; break;
            case SimdPixelFormatBgra32: _convert = Avx512bw::BgraToGray; break;
            case SimdPixelFormatRgb24: _convert = Avx512bw::RgbToGray; break;
            case SimdPixelFormatRgba32: _convert = Avx512bw::RgbaToGray; break;
            default: break;
            }
        }

        //---------------------------------------------------------------------

        ImagePgmBinSaver::ImagePgmBinSaver(const ImageSaverParam& param)
            : Avx2::ImagePgmBinSaver(param)
        {
            switch (_param.format)
            {
            case SimdPixelFormatBgr24: _convert = Avx512bw::BgrToGray; break;
            case SimdPixelFormatBgra32: _convert = Avx512bw::BgraToGray; break;
            case SimdPixelFormatRgb24: _convert = Avx512bw::RgbToGray; break;
            case SimdPixelFormatRgba32: _convert = Avx512bw::RgbaToGray; break;
            default: break;
            }
        }

        //---------------------------------------------------------------------

        ImagePpmTxtSaver::ImagePpmTxtSaver(const ImageSaverParam& param)
            : Avx2::ImagePpmTxtSaver(param)
        {
            switch (_param.format)
            {
            case SimdPixelFormatGray8: _convert = Avx512bw::GrayToBgr; break;
            case SimdPixelFormatBgr24: _convert = Avx512bw::BgrToRgb; break;
            case SimdPixelFormatBgra32: _convert = Avx512bw::BgraToRgb; break;
            case SimdPixelFormatRgba32: _convert = Avx512bw::BgraToBgr; break;
            default: break;
            }
        }

        //---------------------------------------------------------------------

        ImagePpmBinSaver::ImagePpmBinSaver(const ImageSaverParam& param)
            : Avx2::ImagePpmBinSaver(param)
        {
            switch (_param.format)
            {
            case SimdPixelFormatGray8: _convert = Avx512bw::GrayToBgr; break;
            case SimdPixelFormatBgr24: _convert = Avx512bw::BgrToRgb; break;
            case SimdPixelFormatBgra32: _convert = Avx512bw::BgraToRgb; break;
            case SimdPixelFormatRgba32: _convert = Avx512bw::BgraToBgr; break;
            default: break;
            }
        }

        //---------------------------------------------------------------------

        ImageSaver* CreateImageSaver(const ImageSaverParam& param)
        {
            switch (param.file)
            {
            case SimdImageFilePgmTxt: return new ImagePgmTxtSaver(param);
            case SimdImageFilePgmBin: return new ImagePgmBinSaver(param);
            case SimdImageFilePpmTxt: return new ImagePpmTxtSaver(param);
            case SimdImageFilePpmBin: return new ImagePpmBinSaver(param);
            case SimdImageFilePng: return new ImagePngSaver(param);
            case SimdImageFileJpeg: return new ImageJpegSaver(param);
            default:
                return NULL;
            }
        }

        uint8_t* ImageSaveToMemory(const uint8_t* src, size_t stride, size_t width, size_t height, SimdPixelFormatType format, SimdImageFileType file, int quality, size_t* size)
        {
            ImageSaverParam param(width, height, format, file, quality);
            if (param.Validate())
            {
                Holder<ImageSaver> saver(CreateImageSaver(param));
                if (saver)
                {
                    if (saver->ToStream(src, stride))
                        return saver->Release(size);
                }
            }
            return NULL;
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
