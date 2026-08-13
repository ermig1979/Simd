/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar,
*               2022-2022 Souriya Trinh,
*               2022-2022 Fabien Spindler,
*               2024-2024 Jan Rysavy,
*               2024-2024 Jamil Halabi.
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
#include "Simd/SimdMath.h"

#include <vector>
#ifdef SIMD_CPP_2011_ENABLE
#include <thread>
#endif
#include <sstream>
#include <iostream>
#include <fstream>

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>

#elif defined(__GNUC__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <unistd.h>
#include <stdbool.h>
#include <stdlib.h>
#include <sched.h>

#if defined(SIMD_X86_ENABLE) || defined(SIMD_X64_ENABLE)
#include <cpuid.h>
#endif

#if defined(SIMD_ARM_ENABLE) || defined(SIMD_ARM64_ENABLE)
#include <fcntl.h>
#if !defined(__APPLE__)
#include <sys/auxv.h>
#if (defined(SIMD_ARM_ENABLE) || defined(SIMD_ARM64_ENABLE)) && !defined(__FreeBSD__)
#include <asm/hwcap.h>
#endif
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
        SIMD_INLINE bool CpuId(int eax, int ecx, unsigned int *registers)
        {
#if defined(_WIN32)
            __cpuidex((int*)registers, eax, ecx);
#elif (defined __GNUC__)
            if (__get_cpuid_max(0, NULL) < eax)
                return false;
            __cpuid_count(eax, ecx,
                registers[Cpuid::Eax],
                registers[Cpuid::Ebx],
                registers[Cpuid::Ecx],
                registers[Cpuid::Edx]);
#else
#error Do not know how to detect CPU info!
#endif
            return true;
        }

        bool CheckBit(int eax, int ecx, Cpuid::Register index, Cpuid::Bit bit)
        {
            unsigned int registers[4] = { 0, 0, 0, 0 };
            if (!CpuId(eax, ecx, registers))
                return false;
            return (registers[index] & bit) == bit;
        }

        const char* VendorId()
        {
            unsigned int regs[4] = { 0, 0, 0, 0};
            CpuId(0, 0, regs);
            static unsigned int vendorId[4] = { regs[1], regs[3], regs[2], 0 };
            return (char*)vendorId;
        }
#endif//defined(SIMD_X86_ENABLE) || defined(SIMD_X64_ENABLE)

#if defined(__GNUC__) && (defined(SIMD_ARM_ENABLE) || defined(SIMD_ARM64_ENABLE)) && !defined(__APPLE__)
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
#endif//defined(__GNUC__) && (defined(SIMD_ARM_ENABLE) || defined(SIMD_ARM64_ENABLE)) && !defined(__APPLE__)

#ifdef SIMD_CPP_2011_ENABLE
        size_t CpuThreadNumber()
        {
            return std::thread::hardware_concurrency();
        }
#endif

#if defined(_WIN32)
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

        uint64_t CpuRamSize()
        {
            MEMORYSTATUSEX memorystatusex;
            memorystatusex.dwLength = sizeof(memorystatusex);
            if (GlobalMemoryStatusEx(&memorystatusex) == TRUE)
                return memorystatusex.ullTotalPhys;
            return 0;
        }

        static std::string Execute(const char* cmd)
        {
            std::string result;

            SECURITY_ATTRIBUTES saAttr;
            saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
            saAttr.bInheritHandle = TRUE;
            saAttr.lpSecurityDescriptor = NULL;

            HANDLE hReadPipe = NULL;
            HANDLE hWritePipe = NULL;
            if (!CreatePipe(&hReadPipe, &hWritePipe, &saAttr, 0))
                return "";
            SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0);

            PROCESS_INFORMATION piProcInfo;
            ZeroMemory(&piProcInfo, sizeof(PROCESS_INFORMATION));

            STARTUPINFOA siStartInfo;
            ZeroMemory(&siStartInfo, sizeof(STARTUPINFO));
            siStartInfo.cb = sizeof(STARTUPINFO);
            siStartInfo.hStdError = hWritePipe;
            siStartInfo.hStdOutput = hWritePipe;
            siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

            BOOL bSuccess = CreateProcessA(NULL, (LPSTR)cmd, NULL, NULL, TRUE, 
                CREATE_NO_WINDOW, NULL,  NULL, &siStartInfo, &piProcInfo);

            CloseHandle(hWritePipe);

            if (bSuccess)
            {
                char buffer[4096];
                DWORD dwRead = 0;
                while (ReadFile(hReadPipe, buffer, sizeof(buffer), &dwRead, NULL) && dwRead > 0)
                    result.append(buffer, dwRead);
                CloseHandle(piProcInfo.hProcess);
                CloseHandle(piProcInfo.hThread);
            }

            CloseHandle(hReadPipe);
            return result;
        }

        std::string CpuModel()
        {
            int cpuInfo[4] = { -1 };
            char cpuModel[64] = { 0 };
            __cpuid(cpuInfo, 0x80000000);
            for (unsigned i = 0x80000000, n = cpuInfo[0]; i <= n; ++i)
            {
                __cpuid(cpuInfo, i);
                if (i == 0x80000002)
                    memcpy(cpuModel, cpuInfo, sizeof(cpuInfo));
                else if (i == 0x80000003)
                    memcpy(cpuModel + 16, cpuInfo, sizeof(cpuInfo));
                else if (i == 0x80000004)
                    memcpy(cpuModel + 32, cpuInfo, sizeof(cpuInfo));
            }
            for (int i = 63; i >= 0 && (cpuModel[i] == ' ' || cpuModel[i] == 0); --i)
                cpuModel[i] = 0;
            return cpuModel;
        }

        uint64_t CpuCurrentFrequency()
        {
            return 0;
        }

#elif defined(__GNUC__)

        size_t CpuSocketNumber()
        {
            uint32_t number = 0;
#if defined(__APPLE__)
#elif defined(__ANDROID__)
#else
            ::FILE * p = ::popen("lscpu -b -p=Socket 2>/dev/null | grep -v '^#' | sort -u 2>/dev/null | wc -l 2>/dev/null", "r");
            if (p)
            {
                char buffer[PATH_MAX];
                while (::fgets(buffer, PATH_MAX, p));
                number = ::atoi(buffer);
                ::pclose(p);
            }
#endif
            return number;
        }

        size_t CpuCoreNumber()
        {
            uint32_t number = 0;
#if defined(__APPLE__)
#elif defined(__ANDROID__)
#else
            ::FILE * p = ::popen("lscpu -b -p=Core 2>/dev/null | grep -v '^#' | sort -u 2>/dev/null | wc -l 2>/dev/null", "r");
            if (p)
            {
                char buffer[PATH_MAX];
                while (::fgets(buffer, PATH_MAX, p));
                number = ::atoi(buffer);
                ::pclose(p);
            }
#endif
            return number;
        }

        size_t CpuCacheSizeFromLscpu(size_t row)
        {
            char buffer[PATH_MAX];
            ::FILE* pv = ::popen("lscpu -V", "r");
            if (pv)
            {
                while (::fgets(buffer, PATH_MAX, pv));
                char* beg = buffer;
                while (beg < buffer + PATH_MAX && *beg++ != '.');
                size_t ver = ::atoi(beg);
                ::pclose(pv);
                if (ver < 34)
                    return 0;
            }
            std::stringstream ss;
            ss << "lscpu -C=ONE-SIZE -B | sed -n '" << row << "p'";
            size_t size = 0;
            ::FILE* pc = ::popen(ss.str().c_str(), "r");
            if (pc)
            {
                while (::fgets(buffer, PATH_MAX, pc));
                size = ::atoi(buffer);
                ::pclose(pc);
            }
            return size;
        }

        size_t CpuCacheSize(size_t level)
        {
            ptrdiff_t size = 0;
            switch (level)
            {
            case 1:
            {
#if defined(_SC_LEVEL1_DCACHE_SIZE)
                size = ::sysconf(_SC_LEVEL1_DCACHE_SIZE);
                if (size < 0)
                    size = 0;
#endif
                if (size == 0)
                {
                    ::FILE* p = ::popen("getconf -a | grep LEVEL1_DCACHE_SIZE | grep -oE '[^ ]+$'", "r");
                    if (p)
                    {
                        char buffer[PATH_MAX];
                        while (::fgets(buffer, PATH_MAX, p));
                        size = ::atoi(buffer);
                        ::pclose(p);
                    }
                }
                if (size == 0)
                    size = CpuCacheSizeFromLscpu(2);
                if (size == 0)
                    size = 32 * 1024;
                break;
            }
            case 2:
            {
#if defined(_SC_LEVEL2_CACHE_SIZE)
                size = ::sysconf(_SC_LEVEL2_CACHE_SIZE);
                if (size < 0)
                    size = 0;
#endif
                if (size == 0)
                {
                    ::FILE* p = ::popen("getconf -a | grep LEVEL2_CACHE_SIZE | grep -oE '[^ ]+$'", "r");
                    if (p)
                    {
                        char buffer[PATH_MAX];
                        while (::fgets(buffer, PATH_MAX, p));
                        size = ::atoi(buffer);
                        ::pclose(p);
                    }
                }
                if (size == 0)
                    size = CpuCacheSizeFromLscpu(4);
                if (size == 0)
                    size = 256 * 1024;
                break;
            }
            case 3:
            {
#if defined(_SC_LEVEL3_CACHE_SIZE)
                size = ::sysconf(_SC_LEVEL3_CACHE_SIZE);
                if (size < 0)
                    size = 0;
#endif
                if (size == 0)
                {
                    ::FILE* p = ::popen("getconf -a | grep LEVEL3_CACHE_SIZE | grep -oE '[^ ]+$'", "r");
                    if (p)
                    {
                        char buffer[PATH_MAX];
                        while (::fgets(buffer, PATH_MAX, p));
                        size = ::atoi(buffer);
                        ::pclose(p);
                    }
                }
                if (size == 0)
                    size = CpuCacheSizeFromLscpu(5);
                if (size == 0)
                    size = 2048 * 1024;
                break;
            }
            default:
                return 0;
            }
            return size;
        }

        uint64_t CpuRamSize()
        {
            uint64_t size = 0;
#if defined(__APPLE__)
#elif defined(__ANDROID__)
#else
            ::FILE* file = ::popen("grep MemTotal /proc/meminfo | awk '{printf \"%d\", $2 }'", "r");
            if (file)
            {
                char buf[PATH_MAX];
                while (::fgets(buf, PATH_MAX, file));
                size = atoll(buf) * 1024;
                ::pclose(file);
            }
#endif
            return size;
        }

        std::string CpuModel()
        {
            std::string model;
#if defined(__APPLE__)
#elif defined(__ANDROID__)
#else
            ::FILE* file = ::popen("lscpu | grep 'Model name:' | sed -r 's/Model name:\\s{1,}//g'", "r");
            if (file)
            {
                char buffer[PATH_MAX];
                while (::fgets(buffer, PATH_MAX, file));
                model = buffer;
                model = model.substr(0, model.find('\n'));
                ::pclose(file);
            }
#endif
            return model;
        }

        uint64_t CpuFrequencyFromFile(const std::string & path, uint64_t valueScale)
        {
            uint64_t freq = 0;
            std::ifstream file(path.c_str());
            if (file.is_open())
            {
                double value = 0;
                file >> value;
                if (value > 0)
                    freq = (uint64_t)Round(value * double(valueScale));
            }
            return freq;
        }

        uint64_t CpuFrequencyFromSysfs(int core)
        {
            if (core < 0)
                core = 0;
            std::string cpu = "/sys/devices/system/cpu/cpu" + std::to_string(core);
            std::string cpufreq = cpu + "/cpufreq/";
            std::string policy = "/sys/devices/system/cpu/cpufreq/policy" + std::to_string(core) + "/";
            const char* names[5] = { "scaling_cur_freq", "cpuinfo_cur_freq", "cpuinfo_avg_freq", "cpuinfo_max_freq", "scaling_max_freq" };
            for (int i = 0; i < 5; ++i)
            {
                uint64_t freq = CpuFrequencyFromFile(cpufreq + names[i], 1000);
                if (freq == 0)
                    freq = CpuFrequencyFromFile(policy + names[i], 1000);
                if (freq > 0)
                    return freq;
            }
            // ACPI CPPC (common on ARM/ARM64 servers, including Neoverse): values are in MHz.
            uint64_t freq = CpuFrequencyFromFile(cpu + "/acpi_cppc/nominal_freq", 1000000);
            if (freq == 0)
                freq = CpuFrequencyFromFile(cpu + "/acpi_cppc/lowest_freq", 1000000);
            return freq;
        }

        uint64_t CpuFrequencyFromSmbios()
        {
            // Prefer per-structure sysfs nodes when present.
            for (int i = 0; i < 8; ++i)
            {
                std::string path = "/sys/firmware/dmi/entries/4-" + std::to_string(i) + "/raw";
                std::ifstream file(path.c_str(), std::ios::binary);
                if (!file.is_open())
                    continue;
                unsigned char buf[0x18];
                file.read((char*)buf, 0x18);
                if (file.gcount() < 0x18 || buf[0] != 4 || buf[1] < 0x18)
                    continue;
                unsigned int current = unsigned(buf[0x16]) | (unsigned(buf[0x17]) << 8);
                unsigned int maxSpeed = unsigned(buf[0x14]) | (unsigned(buf[0x15]) << 8);
                unsigned int mhz = current ? current : maxSpeed;
                if (mhz > 0 && mhz < 100000)
                    return uint64_t(mhz) * 1000000;
            }

            // Fallback: scan raw SMBIOS table (usually world-readable; no root dmidecode needed).
            std::ifstream table("/sys/firmware/dmi/tables/DMI", std::ios::binary);
            if (!table.is_open())
                return 0;
            std::vector<unsigned char> data((std::istreambuf_iterator<char>(table)), std::istreambuf_iterator<char>());
            size_t off = 0;
            while (off + 4 < data.size())
            {
                unsigned char type = data[off];
                unsigned char length = data[off + 1];
                if (length < 4 || off + length > data.size())
                    break;
                if (type == 4 && length >= 0x18)
                {
                    unsigned int current = unsigned(data[off + 0x16]) | (unsigned(data[off + 0x17]) << 8);
                    unsigned int maxSpeed = unsigned(data[off + 0x14]) | (unsigned(data[off + 0x15]) << 8);
                    unsigned int mhz = current ? current : maxSpeed;
                    if (mhz > 0 && mhz < 100000)
                        return uint64_t(mhz) * 1000000;
                }
                size_t next = off + length;
                while (next + 1 < data.size() && !(data[next] == 0 && data[next + 1] == 0))
                    ++next;
                if (next + 1 >= data.size())
                    break;
                off = next + 2;
                if (type == 127)
                    break;
            }
            return 0;
        }

        uint64_t CpuFrequencyFromDeviceTree()
        {
            ::FILE* p = ::popen("find /sys/firmware/devicetree/base/cpus -name clock-frequency 2>/dev/null | head -n1", "r");
            if (!p)
                return 0;
            char path[PATH_MAX];
            path[0] = 0;
            if (!::fgets(path, PATH_MAX, p))
            {
                ::pclose(p);
                return 0;
            }
            ::pclose(p);
            std::string filePath = path;
            filePath = filePath.substr(0, filePath.find('\n'));
            if (filePath.empty())
                return 0;
            std::ifstream file(filePath.c_str(), std::ios::binary);
            if (!file.is_open())
                return 0;
            unsigned char b[4] = { 0, 0, 0, 0 };
            file.read((char*)b, 4);
            if (file.gcount() < 4)
                return 0;
            uint64_t hz = (uint64_t(b[0]) << 24) | (uint64_t(b[1]) << 16) | (uint64_t(b[2]) << 8) | uint64_t(b[3]);
            return hz > 0 ? hz : 0;
        }

        uint64_t CpuCurrentFrequency()
        {
#if defined(__APPLE__)
#elif defined(__ANDROID__)
#else
            int core = sched_getcpu();
            if (core < 0)
                core = 0;

            uint64_t freq = CpuFrequencyFromSysfs(core);
            // AMU-based scaling_cur_freq can be 0 on an idle core; try cpu0 and a few neighbors.
            for (int i = 0; freq == 0 && i < 8; ++i)
                if (i != core)
                    freq = CpuFrequencyFromSysfs(i);

            if (freq == 0)
            {
                std::stringstream args;
                args << "cat /proc/cpuinfo 2>/dev/null | grep -i \"MHz\" | head -n" << core + 1 << " | tail -1";
                ::FILE* p = ::popen(args.str().c_str(), "r");
                if (p)
                {
                    char buffer[PATH_MAX];
                    buffer[0] = 0;
                    while (::fgets(buffer, PATH_MAX, p));
                    ::pclose(p);
                    std::string output = buffer;
                    std::size_t beg = output.find(":");
                    if (beg != std::string::npos)
                    {
                        int mhz = Round(::atof(output.substr(beg + 1).c_str()));
                        if (mhz > 0)
                            freq = uint64_t(mhz) * 1000000;
                    }
                }
            }

            // ARM/ARM64 Neoverse-N2 and similar: often no cpufreq and no MHz in /proc/cpuinfo.
            if (freq == 0)
                freq = CpuFrequencyFromSmbios();
            if (freq == 0)
                freq = CpuFrequencyFromDeviceTree();
            if (freq == 0)
            {
                // Non-zero speeds only (skip "Current Speed: 0" / "Unknown").
                ::FILE* p = ::popen("dmidecode -t processor 2>/dev/null | grep -E 'Current Speed|Max Speed' | grep -oE '[1-9][0-9]*' | head -n1", "r");
                if (p)
                {
                    char buffer[PATH_MAX];
                    buffer[0] = 0;
                    while (::fgets(buffer, PATH_MAX, p));
                    ::pclose(p);
                    int mhz = ::atoi(buffer);
                    if (mhz > 0)
                        freq = uint64_t(mhz) * 1000000;
                }
            }
            if (freq == 0)
            {
                ::FILE* p = ::popen("dmidecode -s processor-frequency 2>/dev/null | grep -oE '[1-9][0-9]*' | head -n1", "r");
                if (p)
                {
                    char buffer[PATH_MAX];
                    buffer[0] = 0;
                    while (::fgets(buffer, PATH_MAX, p));
                    ::pclose(p);
                    int mhz = ::atoi(buffer);
                    if (mhz > 0)
                        freq = uint64_t(mhz) * 1000000;
                }
            }
            if (freq == 0)
            {
                ::FILE* p = ::popen("lscpu 2>/dev/null | grep -E 'CPU max MHz|CPU MHz' | head -n1 | grep -oE '[0-9]+(\\.[0-9]+)?' | head -n1", "r");
                if (p)
                {
                    char buffer[PATH_MAX];
                    buffer[0] = 0;
                    while (::fgets(buffer, PATH_MAX, p));
                    ::pclose(p);
                    double mhz = ::atof(buffer);
                    if (mhz > 0)
                        freq = (uint64_t)Round(mhz * 1000000.0);
                }
            }
            if (freq == 0)
            {
                ::FILE* p = ::popen("lscpu 2>/dev/null | grep -oE '@[ ]*[0-9]+(\\.[0-9]+)?[ ]*GHz' | head -n1 | grep -oE '[0-9]+(\\.[0-9]+)?'", "r");
                if (p)
                {
                    char buffer[PATH_MAX];
                    buffer[0] = 0;
                    while (::fgets(buffer, PATH_MAX, p));
                    ::pclose(p);
                    double ghz = ::atof(buffer);
                    if (ghz > 0)
                        freq = (uint64_t)Round(ghz * 1000000000.0);
                }
            }
            return freq;
#endif
            return 0;
        }
#else
#error This platform is unsupported!
#endif
    }

    namespace Cpu
    {
        const std::string CPU_MODEL = Base::CpuModel();
        const size_t SOCKET_NUMBER = Base::CpuSocketNumber();
        const size_t CORE_NUMBER = Base::CpuCoreNumber();
#ifdef SIMD_CPP_2011_ENABLE
        const size_t THREAD_NUMBER = Base::CpuThreadNumber();
#endif
        const size_t L1_CACHE_SIZE = Base::CpuCacheSize(1);
        const size_t L2_CACHE_SIZE = Base::CpuCacheSize(2);
        const size_t L3_CACHE_SIZE = Base::CpuCacheSize(3);
        const uint64_t RAM_SIZE = Base::CpuRamSize();
    }
}
