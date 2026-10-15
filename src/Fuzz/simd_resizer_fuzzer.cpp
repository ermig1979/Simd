#include "Simd/SimdLib.h"
#include "Simd/SimdResizer.h"

#include <cstdlib>
#include <cstring>
#include <memory>

namespace
{
    size_t Dimension(uint8_t value)
    {
        const size_t boundaries[] = { 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 255, 256 };
        if (value >= '0' && value <= '?')
            return boundaries[value - '0'];
        return 1 + size_t(value);
    }

    size_t Selector(uint8_t value, size_t count)
    {
        return uint8_t(value - '0') % count;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    const size_t header = 10;
    if (size <= header)
        return 0;

    const SimdResizeMethodType methods[] = {
        SimdResizeMethodNearest, SimdResizeMethodBilinear, SimdResizeMethodBicubic,
        SimdResizeMethodArea, SimdResizeMethodAreaFast
    };
    const size_t sourceWidth = Dimension(data[0]);
    const size_t sourceHeight = Dimension(data[1]);
    const size_t destinationWidth = Dimension(data[2]);
    const size_t destinationHeight = Dimension(data[3]);
    const size_t channels = 1 + Selector(data[4], 4);
    const SimdResizeMethodType method = methods[Selector(data[5], 5)];
    if ((method == SimdResizeMethodArea || method == SimdResizeMethodAreaFast) &&
        (sourceWidth == 1 || sourceHeight == 1))
        return 0;
    if (method == SimdResizeMethodBicubic && sourceWidth == 3 && destinationWidth == 1)
        return 0;
    const size_t sourceStride = sourceWidth * channels + Selector(data[6], 65);
    const size_t destinationStride = destinationWidth * channels + Selector(data[7], 65);
    const size_t sourceOffset = Selector(data[8], 64);
    const size_t destinationOffset = Selector(data[9], 64);
    const size_t sourceSize = sourceOffset + sourceStride * sourceHeight;
    const size_t destinationSize = destinationOffset + destinationStride * destinationHeight;
    typedef std::unique_ptr<uint8_t, decltype(&SimdFree)> Buffer;
    Buffer source(static_cast<uint8_t*>(SimdAllocate(sourceSize, 64)), &SimdFree);
    Buffer baseOutput(static_cast<uint8_t*>(SimdAllocate(destinationSize, 64)), &SimdFree);
    Buffer publicOutput(static_cast<uint8_t*>(SimdAllocate(destinationSize, 64)), &SimdFree);
    if (!source || !baseOutput || !publicOutput)
        return 0;

    typedef std::unique_ptr<void, decltype(&SimdRelease)> Context;
    Context base(Simd::Base::ResizerInit(sourceWidth, sourceHeight, destinationWidth,
        destinationHeight, channels, SimdResizeChannelByte, method), &SimdRelease);
    Context dispatched(SimdResizerInit(sourceWidth, sourceHeight, destinationWidth,
        destinationHeight, channels, SimdResizeChannelByte, method), &SimdRelease);
    for (size_t pass = 0; pass < 2; ++pass)
    {
        for (size_t index = 0; index < sourceSize; ++index)
            source.get()[index] = data[header + index % (size - header)] ^ (pass ? 0xFF : 0);
        std::memset(baseOutput.get(), 0x11, destinationSize);
        std::memset(publicOutput.get(), 0x22, destinationSize);
        if (base)
            SimdResizerRun(base.get(), source.get() + sourceOffset, sourceStride,
                baseOutput.get() + destinationOffset, destinationStride);
        if (dispatched)
            SimdResizerRun(dispatched.get(), source.get() + sourceOffset, sourceStride,
                publicOutput.get() + destinationOffset, destinationStride);
        if (method == SimdResizeMethodNearest && base && dispatched)
        {
            for (size_t row = 0; row < destinationHeight; ++row)
            {
                const size_t offset = destinationOffset + row * destinationStride;
                if (std::memcmp(baseOutput.get() + offset, publicOutput.get() + offset,
                    destinationWidth * channels) != 0)
                    std::abort();
            }
        }
    }
    return 0;
}
