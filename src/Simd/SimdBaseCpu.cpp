/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdCpu.h"

#include <vector>
#include <thread>
#include <sstream>
#include <iostream>

#if defined(_MSC_VER)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>

#elif defined(__GNUC__)
#include <unistd.h>
#include <stdbool.h>
#include <stdlib.h>

#if defined(SIMD_X86_ENABLE) || defined(SIMD_X64_ENABLE)
#include <cpuid.h>
#endif

#if defined(SIMD_PPC_ENABLE) || defined(SIMD_PPC64_ENABLE) || defined(SIMD_ARM_ENABLE) || defined(SIMD_ARM64_ENABLE)
#include <fcntl.h>
#include <sys/auxv.h>
#if defined(SIMD_ARM_ENABLE) || defined(SIMD_ARM64_ENABLE)
#include <asm/hwcap.h>
#endif
#endif

#else
# error Do not know how to detect CPU info
#endif

namespace Simd
{
    namespace Base
    {
#if defined(SIMD_X86_ENABLE) || defined(SIMD_X64_ENABLE)
        bool CheckBit(Cpuid::Level level, Cpuid::Register index, Cpuid::Bit bit)
        {
            unsigned int registers[4] = { 0, 0, 0, 0 };
#if defined(_MSC_VER)
            __cpuid((int*)registers, level);
#elif (defined __GNUC__)
            if (__get_cpuid_max(0, NULL) < level)
                return false;
            __cpuid_count(level, 0, 
                registers[Cpuid::Eax], 
                registers[Cpuid::Ebx], 
                registers[Cpuid::Ecx], 
                registers[Cpuid::Edx]);
#else
#error Do not know how to detect CPU info!
#endif
            return (registers[index] & bit) == bit;
        }
#endif//defined(SIMD_X86_ENABLE) || defined(SIMD_X64_ENABLE)

#if defined(__GNUC__) && (defined(SIMD_PPC_ENABLE) || defined(SIMD_PPC64_ENABLE) || defined(SIMD_ARM_ENABLE) || defined(SIMD_ARM64_ENABLE))
        bool CheckBit(int at, int bit)
        {
            bool result = false;
            int file = ::open("/proc/self/auxv", O_RDONLY);
            if (file < 0)
                return false;
            const ssize_t size = 64;
            unsigned long buffer[size];
            for (ssize_t count = size; count == size;)
            {
                count = ::read(file, buffer, sizeof(buffer)) / sizeof(unsigned long);
                for (int i = 0; i < count; i += 2)
                {
                    if (buffer[i] == (unsigned)at)
                    {
                        result = !!(buffer[i + 1] & bit);
                        count = 0;
                    }
                    if (buffer[i] == AT_NULL)
                        count = 0;
                }
            }
            ::close(file);
            return result;
        }
#endif//defined(__GNUC__) && (defined(SIMD_PPC_ENABLE) || defined(SIMD_PPC64_ENABLE) || defined(SIMD_ARM_ENABLE) || defined(SIMD_ARM64_ENABLE))

        size_t CpuThreadNumber()
        {
            return std::thread::hardware_concurrency();
        }

#if defined(_MSC_VER)
        typedef SYSTEM_LOGICAL_PROCESSOR_INFORMATION Info;

        void GetLogicalProcessorInformation(std::vector<Info> & info)
        {
            DWORD size = 0;
            ::GetLogicalProcessorInformation(0, &size); 
            info.resize(size / sizeof(Info));
            ::GetLogicalProcessorInformation(info.data(), &size);
        }

        size_t CpuSocketNumber()
        {
            std::vector<Info> info;
            GetLogicalProcessorInformation(info);
            size_t number = 0;
            for (size_t i = 0; i < info.size(); ++i)
                if (info[i].Relationship == ::RelationNumaNode)
                    number++;
            return number;
        }            

        size_t CpuCoreNumber()
        {
            std::vector<Info> info;
            GetLogicalProcessorInformation(info);
            size_t number = 0;
            for (size_t i = 0; i < info.size(); ++i)
                if (info[i].Relationship == ::RelationProcessorCore)
                    number++;
            return number;
        }

        size_t CpuCacheSize(size_t level)
        {
            std::vector<Info> info;
            GetLogicalProcessorInformation(info);
            for (size_t i = 0; i < info.size(); ++i)
                if (info[i].Relationship == ::RelationCache && info[i].Cache.Level == level && (info[i].Cache.Type == ::CacheData || info[i].Cache.Type == CacheUnified))
                    return info[i].Cache.Size;
            return 0;
        }
#elif defined(__GNUC__)
        size_t CpuSocketNumber()
        {
            uint32_t number = 0;
            ::FILE * p = ::popen("lscpu -b -p=Socket | grep -v '^#' | sort -u | wc -l", "r");
            if (p)
            {
                char buffer[PATH_MAX];
                while (::fgets(buffer, PATH_MAX, p));
                number = ::atoi(buffer);
                ::pclose(p);
            }
            return number;
        }

        size_t CpuCoreNumber()
        {
            uint32_t number = 0;
            ::FILE * p = ::popen("lscpu -b -p=Core | grep -v '^#' | sort -u | wc -l", "r");
            if (p)
            {
                char buffer[PATH_MAX];
                while (::fgets(buffer, PATH_MAX, p));
                number = ::atoi(buffer);
                ::pclose(p);
            }
            return number;
        }

        SIMD_INLINE size_t CorrectIfZero(size_t value, size_t otherwise)
        {
            return value ? value : otherwise;
        }

#if defined(_SC_LEVEL1_DCACHE_SIZE) && defined(_SC_LEVEL2_CACHE_SIZE) && defined(_SC_LEVEL3_CACHE_SIZE)
        size_t CpuCacheSize(size_t level)
        {
            switch (level)
            {
            case 1: return CorrectIfZero(::sysconf(_SC_LEVEL1_DCACHE_SIZE), 32 * 1024);
            case 2: return CorrectIfZero(::sysconf(_SC_LEVEL2_CACHE_SIZE), 256 * 1024);
            case 3: return CorrectIfZero(::sysconf(_SC_LEVEL3_CACHE_SIZE), 2048 * 1024);
            default:
                return 0;
            }
        }
#else
        size_t CpuCacheSize(size_t level)
        {
            switch (level)
            {
            case 1: return 32 * 1024;
            case 2: return 256 * 1024;
            case 3: return 2048 * 1024;
            default:
                return 0;
            }
        }
#endif

#else
#error This platform is unsupported!
#endif
    }

    namespace Cpu
    {
        const size_t SOCKET_NUMBER = Base::CpuSocketNumber();
        const size_t CORE_NUMBER = Base::CpuCoreNumber();
        const size_t THREAD_NUMBER = Base::CpuThreadNumber();
        const size_t L1_CACHE_SIZE = Base::CpuCacheSize(1);
        const size_t L2_CACHE_SIZE = Base::CpuCacheSize(2);
        const size_t L3_CACHE_SIZE = Base::CpuCacheSize(3);
    }
}
