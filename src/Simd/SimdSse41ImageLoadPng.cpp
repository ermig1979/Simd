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
#include "Simd/SimdImageLoad.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) 
    namespace Sse41
    {
        template<size_t pixelSize> SIMD_INLINE void Copy(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            Base::Copy(src, srcStride, width, height, pixelSize, dst, dstStride);
        }

        SIMD_INLINE void GrayToBgra(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            GrayToBgra(src, width, height, srcStride, dst, dstStride, 0xFF);
        }

        SIMD_INLINE void BgrToBgra(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            BgrToBgra(src, width, height, srcStride, dst, dstStride, 0xFF);
        }


        SIMD_INLINE void RgbToBgra(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            RgbToBgra(src, width, height, srcStride, dst, dstStride, 0xFF);
        }

        //-------------------------------------------------------------------------------------------------

        ImagePngLoader::ImagePngLoader(const ImageLoaderParam& param)
            : Base::ImagePngLoader(param)
        {
            if (_param.format == SimdPixelFormatNone)
                _param.format = SimdPixelFormatRgb24;
        }

        void ImagePngLoader::SetHandlers()
        {
            Base::ImagePngLoader::SetHandlers();
            if (_width >= A)
            {
                if (_depth <= 8)
                {
                    if (_outN == 1)
                    {
                        switch (_param.format)
                        {
                        case SimdPixelFormatGray8: _converter = Copy<1>; break;
                        case SimdPixelFormatBgr24: _converter = GrayToBgr; break;
                        case SimdPixelFormatRgb24: _converter = GrayToBgr; break;
                        case SimdPixelFormatBgra32: _converter = GrayToBgra; break;
                        case SimdPixelFormatRgba32: _converter = GrayToBgra; break;
                        }
                    }
                    else if (_outN == 3)
                    {
                        switch (_param.format)
                        {
                        case SimdPixelFormatGray8: _converter = RgbToGray; break;
                        case SimdPixelFormatBgr24: _converter = BgrToRgb; break;
                        case SimdPixelFormatRgb24: _converter = Copy<3>; break;
                        case SimdPixelFormatBgra32: _converter = RgbToBgra; break;
                        case SimdPixelFormatRgba32: _converter = BgrToBgra; break;
                        }
                    }
                    else if (_outN == 4)
                    {
                        switch (_param.format)
                        {
                        case SimdPixelFormatGray8: _converter = RgbaToGray; break;
                        case SimdPixelFormatBgr24: _converter = BgraToRgb; break;
                        case SimdPixelFormatRgb24: _converter = BgraToBgr; break;
                        case SimdPixelFormatBgra32: _converter = BgraToRgba; break;
                        case SimdPixelFormatRgba32: _converter = Copy<4>; break;
                        }
                    }
                }
            }
        }
    }
#endif
}
