/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdEnable.h"

#include <vector>
#include <thread>
#include <sstream>
#include <iostream>

#ifdef __GNUC__
#include <unistd.h>
#include <stdbool.h>
#include <stdlib.h>
#endif

namespace Simd
{
    namespace Base
    {
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
}
