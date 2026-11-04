#include "Simd/SimdGaussianBlur.h"
#include "Simd/SimdLib.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <vector>

namespace
{
    size_t Value(uint8_t byte)
    {
        return static_cast<uint8_t>(byte - 32);
    }

    size_t Dimension(uint8_t byte)
    {
        const size_t dimensions[] = { 1, 2, 3, 4, 7, 8, 15, 16,
            17, 30, 32, 34, 62, 64, 66, 128 };
        const size_t value = Value(byte);
        return value < 16 ? dimensions[value] : 1 + value % 128;
    }

    struct Image
    {
        size_t stride;
        size_t rowBytes;
        size_t height;
        std::vector<uint8_t> storage;
        uint8_t* pixels;

        Image(size_t bytes, size_t rows, size_t layout)
            : stride(layout % 4 == 1 ? (bytes + 63) / 64 * 64 :
                layout % 4 == 2 ? bytes + 1 + (layout / 4) % 31 : bytes)
            , rowBytes(bytes)
            , height(rows)
            , storage(stride * rows + 128, 0xA5)
        {
            const uintptr_t address = reinterpret_cast<uintptr_t>(storage.data());
            pixels = reinterpret_cast<uint8_t*>((address + 63) & ~uintptr_t(63));
            if (layout % 4 >= 2)
                pixels += 1 + (layout / 4) % 31;
        }

        void Fill(const uint8_t* payload, size_t length, size_t pattern)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t column = 0; column < rowBytes; ++column)
                {
                    const size_t index = row * rowBytes + column;
                    const uint8_t sample = payload[index % length];
                    pixels[row * stride + column] = pattern % 3 == 0 ? sample :
                        pattern % 3 == 1 ? uint8_t(sample + index * 37) :
                        uint8_t(sample ^ (index % 2 ? 255 : 0));
                }
            }
        }

        void Change()
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t column = 0; column < rowBytes; ++column)
                    pixels[row * stride + column] ^= 255;
            }
        }

        void Compare(const Image& reference) const
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t column = 0; column < rowBytes; ++column)
                {
                    const int difference = int(pixels[row * stride + column]) -
                        int(reference.pixels[row * reference.stride + column]);
                    if (difference < -1 || difference > 1)
                        std::abort();
                }
            }
        }
    };
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    const size_t header = 8;
    if (size <= header)
        return 0;

    const size_t width = Dimension(data[0]);
    const size_t height = Dimension(data[1]);
    const size_t channels = 1 + Value(data[2]) % 4;
    const float sigmas[] = { 0.3f, 0.5f, 1.0f, 1.2f, 1.6f, 2.0f, 3.0f };
    const float sigma = sigmas[Value(data[3]) % 7];
    const float epsilon = 0.001f;
    const float* epsilonPointer = Value(data[7]) % 2 == 0 ? &epsilon : nullptr;
    Image source(width * channels, height, Value(data[4]));
    Image result(width * channels, height, Value(data[5]));
    Image reference(width * channels, height, Value(data[5]));
    source.Fill(data + header, size - header, Value(data[6]));

    typedef std::unique_ptr<void, void (*)(void*)> Context;
    Context base(Simd::Base::GaussianBlurInit(width, height, channels,
        &sigma, epsilonPointer), SimdRelease);
    Context dispatch(SimdGaussianBlurInit(width, height, channels,
        &sigma, epsilonPointer), SimdRelease);
    if (!base || !dispatch)
        return 0;

    for (size_t pass = 0; pass < 2; ++pass)
    {
        if (pass != 0)
            source.Change();
        std::fill(result.storage.begin(), result.storage.end(), 0xA5);
        std::fill(reference.storage.begin(), reference.storage.end(), 0x5A);
        SimdGaussianBlurRun(base.get(), source.pixels, source.stride,
            reference.pixels, reference.stride);
        SimdGaussianBlurRun(dispatch.get(), source.pixels, source.stride,
            result.pixels, result.stride);
        result.Compare(reference);
    }
    return 0;
}
