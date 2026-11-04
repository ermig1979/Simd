#include "Simd/SimdArray.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdLib.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace
{
    unsigned Decode(uint8_t value)
    {
        return static_cast<uint8_t>(value - 32);
    }

    size_t OutputSize(size_t source, int kernel, int stride, int pad, bool ceil)
    {
        const int extent = static_cast<int>(source);
        const int numerator = extent + 2 * pad - kernel;
        if (numerator < 0)
            return 0;
        const int output = (numerator + (ceil ? stride - 1 : 0)) / stride + 1;
        if (output <= 0 || output > 64)
            return 0;
        for (int position = 0; position < output; ++position)
        {
            const int start = position * stride - pad;
            if (std::max(start, 0) >= std::min(start + kernel, extent))
                return 0;
        }
        return static_cast<size_t>(output);
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    if (size < 8)
        return 0;

    const bool maximum = Decode(data[0]) % 2 != 0;
    const SimdTensorFormatType format = Decode(data[1]) % 2 ? SimdTensorFormatNhwc : SimdTensorFormatNchw;
    const size_t channels = 1 + Decode(data[2]) % 65;
    const size_t height = 1 + Decode(data[3]) % 64;
    const size_t width = 1 + Decode(data[4]) % 64;
    const unsigned configuration = Decode(data[5]) % 5;
    const int kernel = configuration == 0 ? 2 : 3;
    const int stride = configuration == 1 || configuration == 2 ? 1 : 2;
    const int pad = configuration == 2 || configuration == 4 ? 1 : 0;
    const unsigned flags = Decode(data[6]);
    const SimdBool excludePad = flags & 1 ? SimdTrue : SimdFalse;
    const bool nchwMaximum = maximum && format == SimdTensorFormatNchw;
    if (maximum && format == SimdTensorFormatNhwc && channels < 8)
        return 0;
    if (nchwMaximum && configuration == 2 && height == 1)
        return 0;
    if (nchwMaximum && configuration == 0 && (width == 7 || width == 15 || width == 31))
        return 0;
    const bool ceil = (flags & 2) != 0 || (nchwMaximum && stride == 2 && pad == 0);
    const size_t dstH = OutputSize(height, kernel, stride, pad, ceil);
    const size_t dstW = OutputSize(width, kernel, stride, pad, ceil);
    if (dstH == 0 || dstW == 0)
        return 0;

    const size_t sourceCount = channels * height * width;
    const size_t destinationCount = channels * dstH * dstW;
    const size_t payloadSize = std::min(size - 7, size_t(4096));
    Simd::Array32f source(sourceCount);
    Simd::Array32f reference(destinationCount);
    Simd::Array32f actual(destinationCount);
    if (!source.data || !reference.data || !actual.data)
        return 0;
    for (size_t index = 0; index < sourceCount; ++index)
        source[index] = (static_cast<int>(data[7 + index % payloadSize] % 129) - 64) / 64.0f;
    std::fill(reference.data, reference.data + destinationCount, -2.0f);
    std::fill(actual.data, actual.data + destinationCount, 2.0f);

    if (maximum)
    {
        Simd::Base::SynetPoolingMax32f(source.data, channels, height, width, 1, kernel, kernel,
            1, stride, stride, 0, pad, pad, reference.data, channels, dstH, dstW, format);
        SimdSynetPoolingMax32f(source.data, channels, height, width, 1, kernel, kernel,
            1, stride, stride, 0, pad, pad, actual.data, channels, dstH, dstW, format);
    }
    else
    {
        Simd::Base::SynetPoolingAverage(source.data, channels, height, width, kernel, kernel,
            stride, stride, pad, pad, reference.data, dstH, dstW, excludePad, format);
        SimdSynetPoolingAverage(source.data, channels, height, width, kernel, kernel,
            stride, stride, pad, pad, actual.data, dstH, dstW, excludePad, format);
    }

    for (size_t index = 0; index < destinationCount; ++index)
    {
        if (!std::isfinite(reference[index]) || !std::isfinite(actual[index]))
            std::abort();
        if (maximum ? reference[index] != actual[index] : std::fabs(reference[index] - actual[index]) > 0.001f)
            std::abort();
    }
    return 0;
}
