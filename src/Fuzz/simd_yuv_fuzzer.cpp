#include "Simd/SimdBase.h"
#include "Simd/SimdLib.h"

#include <cstdlib>
#include <cstring>
#include <vector>

namespace
{
    size_t Value(uint8_t byte)
    {
        return static_cast<uint8_t>(byte - 32);
    }

    struct Plane
    {
        size_t stride;
        size_t rowBytes;
        size_t height;
        std::vector<uint8_t> storage;
        uint8_t* pixels;

        Plane(size_t bytes, size_t rows, size_t layout)
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

        void Fill(const uint8_t* payload, size_t length, size_t pattern, size_t salt)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t column = 0; column < rowBytes; ++column)
                {
                    const size_t index = row * rowBytes + column;
                    const uint8_t sample = payload[(index + salt) % length];
                    pixels[row * stride + column] = pattern % 3 == 0 ? sample :
                        pattern % 3 == 1 ? uint8_t(sample + index * 37 + salt) :
                        uint8_t(sample ^ (index % 2 ? 255 : 0));
                }
            }
        }

        void Compare(const Plane& reference) const
        {
            for (size_t row = 0; row < height; ++row)
            {
                if (std::memcmp(pixels + row * stride,
                    reference.pixels + row * reference.stride, rowBytes) != 0)
                    std::abort();
            }
        }
    };

    typedef void (*ToYuv)(const uint8_t*, size_t, size_t, size_t,
        uint8_t*, size_t, uint8_t*, size_t, uint8_t*, size_t, SimdYuvType);
    typedef void (*ToBgr)(const uint8_t*, size_t, const uint8_t*, size_t,
        const uint8_t*, size_t, size_t, size_t, uint8_t*, size_t, SimdYuvType);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    const size_t header = 9;
    if (size <= header)
        return 0;

    const size_t widths[] = { 1, 2, 3, 14, 16, 18, 30, 32, 34,
        62, 64, 66, 94, 96, 98, 128 };
    const size_t mode = Value(data[0]) % 6;
    const size_t family = mode % 3;
    const size_t horizontal = family == 2 ? 1 : 2;
    const size_t vertical = family == 0 ? 2 : 1;
    const size_t widthCode = Value(data[1]);
    size_t width = widthCode < 16 ? widths[widthCode] : 1 + widthCode % 128;
    size_t height = 1 + Value(data[2]) % 128;
    width = (width + horizontal - 1) / horizontal * horizontal;
    height = (height + vertical - 1) / vertical * vertical;

    const SimdYuvType types[] = { SimdYuvBt601, SimdYuvBt709,
        SimdYuvBt2020, SimdYuvTrect871 };
    const SimdYuvType type = types[Value(data[3]) % 4];
    Plane bgr(width * 3, height, Value(data[4]));
    Plane bgrReference(width * 3, height, Value(data[4]));
    Plane y(width, height, Value(data[5]));
    Plane yReference(width, height, Value(data[5]));
    Plane u(width / horizontal, height / vertical, Value(data[6]));
    Plane uReference(width / horizontal, height / vertical, Value(data[6]));
    Plane v(width / horizontal, height / vertical, Value(data[7]));
    Plane vReference(width / horizontal, height / vertical, Value(data[7]));
    const size_t pattern = Value(data[8]);
    const uint8_t* payload = data + header;
    const size_t length = size - header;

    if (mode < 3)
    {
        const ToYuv base[] = { Simd::Base::BgrToYuv420pV2,
            Simd::Base::BgrToYuv422pV2, Simd::Base::BgrToYuv444pV2 };
        const ToYuv dispatch[] = { SimdBgrToYuv420pV2,
            SimdBgrToYuv422pV2, SimdBgrToYuv444pV2 };
        bgr.Fill(payload, length, pattern, 0);
        base[family](bgr.pixels, bgr.stride, width, height,
            yReference.pixels, yReference.stride, uReference.pixels, uReference.stride,
            vReference.pixels, vReference.stride, type);
        dispatch[family](bgr.pixels, bgr.stride, width, height,
            y.pixels, y.stride, u.pixels, u.stride, v.pixels, v.stride, type);
        y.Compare(yReference);
        u.Compare(uReference);
        v.Compare(vReference);
    }
    else
    {
        const ToBgr base[] = { Simd::Base::Yuv420pToBgrV2,
            Simd::Base::Yuv422pToBgrV2, Simd::Base::Yuv444pToBgrV2 };
        const ToBgr dispatch[] = { SimdYuv420pToBgrV2,
            SimdYuv422pToBgrV2, SimdYuv444pToBgrV2 };
        y.Fill(payload, length, pattern, 0);
        u.Fill(payload, length, pattern, 17);
        v.Fill(payload, length, pattern, 53);
        base[family](y.pixels, y.stride, u.pixels, u.stride, v.pixels, v.stride,
            width, height, bgrReference.pixels, bgrReference.stride, type);
        dispatch[family](y.pixels, y.stride, u.pixels, u.stride, v.pixels, v.stride,
            width, height, bgr.pixels, bgr.stride, type);
        bgr.Compare(bgrReference);
    }
    return 0;
}
