#include "Simd/SimdImageSave.h"
#include "Simd/SimdLib.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace
{
    size_t Dimension(uint8_t value)
    {
        const size_t boundaries[] = { 1, 2, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128 };
        if (value >= '0' && value <= '=')
            return boundaries[value - '0'];
        return 1 + value % 128;
    }

    size_t Selector(uint8_t value, size_t count)
    {
        return uint8_t(value - '0') % count;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    const size_t header = 7;
    if (size <= header)
        return 0;

    const SimdPixelFormatType formats[] = {
        SimdPixelFormatGray8, SimdPixelFormatBgr24, SimdPixelFormatBgra32,
        SimdPixelFormatRgb24, SimdPixelFormatRgba32
    };
    const size_t channels[] = { 1, 3, 4, 3, 4 };
    const SimdImageFileType files[] = {
        SimdImageFilePgmTxt, SimdImageFilePgmBin, SimdImageFilePpmTxt,
        SimdImageFilePpmBin, SimdImageFilePng, SimdImageFileJpeg, SimdImageFileBmp
    };
    const int qualities[] = { 10, 49, 50, 65, 90, 91, 100 };
    const size_t width = Dimension(data[0]);
    const size_t height = Dimension(data[1]);
    const size_t formatIndex = Selector(data[2], 5);
    const SimdImageFileType file = files[Selector(data[3], 7)];
    const int quality = data[4] >= '0' && data[4] <= '6'
        ? qualities[data[4] - '0'] : 1 + data[4] % 100;
    const size_t stride = width * channels[formatIndex] + Selector(data[5], 33);
    const size_t offset = Selector(data[6], 32);
    std::vector<uint8_t> storage(offset + stride * height);
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = data[header + index % (size - header)];
    const uint8_t* source = storage.data() + offset;

    size_t baseSize = 0, publicSize = 0;
    typedef std::unique_ptr<uint8_t, decltype(&SimdFree)> Image;
    Image base(Simd::Base::ImageSaveToMemory(source, stride, width, height,
        formats[formatIndex], file, quality, &baseSize), &SimdFree);
    Image dispatched(SimdImageSaveToMemory(source, stride, width, height,
        formats[formatIndex], file, quality, &publicSize), &SimdFree);
    if (file != SimdImageFileJpeg && base && dispatched &&
        (baseSize != publicSize || std::memcmp(base.get(), dispatched.get(), baseSize) != 0))
        std::abort();
    return 0;
}
