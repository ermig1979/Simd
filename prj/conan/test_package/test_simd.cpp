#include "Simd/SimdLib.hpp"
#include "Simd/SimdFrame.hpp"
#include "Simd/SimdPyramid.hpp"

#include <iostream>

static bool test_cpu_info()
{
    const char* version = SimdVersion();
    if (!version || version[0] == '\0')
        return false;
    const char* cpu_model = SimdCpuDesc(SimdCpuDescModel);
    std::cout << "Simd Library: " << version << std::endl;
    std::cout << "CPU: " << (cpu_model ? cpu_model : "unknown") << std::endl;
    std::cout << "Cores: " << SimdCpuInfo(SimdCpuInfoCores) << std::endl;
    return true;
}

static bool test_view()
{
    typedef Simd::View<Simd::Allocator> View;

    View src(64, 64, View::Bgra32);
    View dst(64, 64, View::Gray8);
    if (src.data == nullptr || dst.data == nullptr)
        return false;
    Simd::Convert(src, dst);
    return true;
}

static bool test_frame()
{
    typedef Simd::Frame<Simd::Allocator> Frame;

    Frame src(2, 2, Frame::Yuv420p);
    Frame dst(2, 2, Frame::Bgr24);
    Simd::Convert(src, dst);
    return true;
}

static bool test_resize()
{
    typedef Simd::View<Simd::Allocator> View;

    View src(128, 96, View::Bgr24);
    View dst(40, 30, View::Bgr24);
    Simd::Resize(src, dst, SimdResizeMethodArea);
    return dst.data != nullptr;
}

int main()
{
    bool ok = true;

    std::cout << "=== Simd test_package ===" << std::endl;

    if (!test_cpu_info()) {
        std::cerr << "FAIL: test_cpu_info" << std::endl;
        ok = false;
    }

    if (!test_view()) {
        std::cerr << "FAIL: test_view" << std::endl;
        ok = false;
    }

    if (!test_frame()) {
        std::cerr << "FAIL: test_frame" << std::endl;
        ok = false;
    }

    if (!test_resize()) {
        std::cerr << "FAIL: test_resize" << std::endl;
        ok = false;
    }

    if (ok)
        std::cout << "All tests passed." << std::endl;

    return ok ? 0 : 1;
}
