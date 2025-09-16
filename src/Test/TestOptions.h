/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#ifndef __TestOptions_h__
#define __TestOptions_h__

#include "Test/TestConfig.h"
#include "Test/TestLog.h"
#include "Test/TestString.h"

namespace Test
{
    struct ArgsParser
    {
    public:
        ArgsParser(int argc, char* argv[], bool alt = false)
            : _argc(argc)
            , _argv(argv)
            , _alt(alt)
        {
        }

        const int Argc() const { return _argc; }
        char** Argv() const { return (char**)_argv; }

        int* ArgcPtr() { return &_argc; }
        char*** ArgvPtr() { return &_argv; }

    protected:
        String GetArg(const String& name, const String& default_ = String(), bool exit = true, const Strings& valids = Strings())
        {
            return GetArgs({ name }, { default_ }, exit, valids)[0];
        }

        String GetArg2(const String& name1, const String& name2, const String& default_ = String(), bool exit = true, const Strings& valids = Strings())
        {
            return GetArgs({ name1, name2 }, { default_ }, exit, valids)[0];
        }

        Strings GetArgs(const String& name, const Strings& defaults = Strings(), bool exit = true, const Strings& valids = Strings())
        {
            return GetArgs(Strings({ name }), defaults, exit, valids);
        }

        Strings GetArgs(const Strings& names, const Strings& defaults = Strings(), bool exit = true, const Strings& valids = Strings())
        {
            Strings values;
            for (int a = 1; a < _argc; ++a)
            {
                String arg = _argv[a];
                for (size_t n = 0; n < names.size(); ++n)
                {
                    const String& name = names[n];
                    if (arg.substr(0, name.size()) == name)
                    {
                        String value;
                        if (_alt)
                        {
                            if (arg.substr(name.size(), 1) == "=")
                                value = arg.substr(name.size() + 1);
                        }
                        else
                        {
                            a++;
                            if (a < _argc)
                                value = _argv[a];
                        }
                        if (valids.size())
                        {
                            bool found = false;
                            for (size_t v = 0; v < valids.size() && !found; ++v)
                                if (valids[v] == value)
                                    found = true;
                            if (!found)
                            {
                                std::cout << "Argument '";
                                for (size_t i = 0; i < names.size(); ++i)
                                    std::cout << (i ? " | " : "") << names[i];
                                std::cout << "' is equal to " << value << " ! Its valid values : { ";
                                for (size_t i = 0; i < valids.size(); ++i)
                                    std::cout << (i ? " | " : "") << valids[i];
                                std::cout << " }." << std::endl;
                                ::exit(1);
                            }
                        }
                        values.push_back(value);
                    }
                }
            }
            if (values.empty())
            {
                if (defaults.empty() && exit)
                {
                    std::cout << "Argument '";
                    for (size_t n = 0; n < names.size(); ++n)
                        std::cout << (n ? " | " : "") << names[n];
                    std::cout << "' is absent!" << std::endl;
                    ::exit(1);
                }
                else
                    return defaults;
            }

            return values;
        }

        String AppName() const
        {
            return _argv[0];
        }

        bool HasArg(const Strings& names) const
        {
            for (int a = 1; a < _argc; ++a)
            {
                String arg = _argv[a];
                for (size_t n = 0; n < names.size(); ++n)
                {
                    const String& name = names[n];
                    if (arg.substr(0, name.size()) == name)
                        return true;
                }
            }
            return false;
        }

        bool HasArg(const String& name) const
        {
            return HasArg({ name });
        }

        bool HasArg(const String& name0, const String& name1) const
        {
            return HasArg({ name0, name1 });
        }

    private:
        int _argc;
        char** _argv;
        bool _alt;
    };

    //-------------------------------------------------------------------------------------------------

    struct Options : public ArgsParser
    {
        enum Mode
        {
            Auto,
            Special,
        } mode;

        bool help;

        Strings include, exclude;

        String text, html, source, output;

        size_t workThreads, testRepeats, testStatistics, testThreads;

        bool printAlign, printInternal, checkCpp, pinThreads;

        double warmUpTime;

        uint32_t disabledExtensions;

        Options(int argc, char* argv[])
            : ArgsParser(argc, argv, true)
            , mode(Auto)
        {
            help = HasArg("--help", "-?");
            include = GetArgs("-fi", Strings(), false);
            exclude = GetArgs("-fe", Strings(), false);
            testThreads = std::min<int>(std::max<int>(FromString<int>(GetArg("-tt", "0", false)), 0), std::thread::hardware_concurrency());
            workThreads = std::min<int>(std::max<int>(FromString<int>(GetArg("-wt", "1", false)), 1), std::thread::hardware_concurrency());
            testRepeats = std::min<int>(FromString<int>(GetArg("-tr", "1", false)), 1);
            testStatistics = std::min<int>(FromString<int>(GetArg("-ts", "0", false)), 0);
            checkCpp = FromString<bool>(GetArg("-cc", "0", false));
            printAlign = FromString<bool>(GetArg("-pa", "0", false));
            printInternal = FromString<bool>(GetArg("-pi", "1", false));
            pinThreads = FromString<bool>(GetArg("-pt", "1", false));
            text = GetArg("-ot", "", false);
            html = GetArg("-oh", "", false);
            source = GetArg("-s", "", false);
            output = GetArg("-o", "", false);
            warmUpTime = FromString<int>(GetArg("-wu", "0", false)) * 0.001;
            disabledExtensions = FromString<uint32_t>(GetArg("-de", "0", false));

            for (int i = 1; i < argc; ++i)
            {
                String arg = argv[i];
                if (arg.substr(0, 5) == "-help" || arg.substr(0, 2) == "-?")
                {
                    //help = true;
                    break;
                }
                else if (arg.find("-m=") == 0)
                {
                    switch (arg[3])
                    {
                    case 'a': mode = Auto; break;
                    case 's': mode = Special; break;
                    default:
                        TEST_LOG_SS(Error, "Unknown command line options: '" << arg << "'!" << std::endl);
                        exit(1);
                    }
                }
                else if (arg.find("-tt=") == 0)
                {
                    //TEST_THREADS = std::min<int>(FromString<int>(arg.substr(4, arg.size() - 4)), (size_t)std::thread::hardware_concurrency());
                }
                else if (arg.find("-tr=") == 0)
                {
                    //testRepeats = FromString<size_t>(arg.substr(4, arg.size() - 4));
                }
                else if (arg.find("-ts=") == 0)
                {
                    //testStatistics = FromString<int>(arg.substr(4, arg.size() - 4));
                }
                else if (arg.find("-fi=") == 0)
                {
                    //include.push_back(arg.substr(4, arg.size() - 4));
                }
                else if (arg.find("-fe=") == 0)
                {
                    //exclude.push_back(arg.substr(4, arg.size() - 4));
                }
                else if (arg.find("-ot=") == 0)
                {
                    //text = arg.substr(4, arg.size() - 4);
                }
                else if (arg.find("-oh=") == 0)
                {
                    //html = arg.substr(4, arg.size() - 4);
                }
                else if (arg.find("-r=") == 0)
                {
                    ROOT_PATH = arg.substr(3, arg.size() - 3);
                }
                else if (arg.find("-s=") == 0)
                {
                    //SOURCE = arg.substr(3, arg.size() - 3);
                }
                else if (arg.find("-o=") == 0)
                {
                    //OUTPUT = arg.substr(3, arg.size() - 3);
                }
                else if (arg.find("-c=") == 0)
                {
                    C = FromString<int>(arg.substr(3, arg.size() - 3));
                }
                else if (arg.find("-h=") == 0)
                {
                    H = FromString<int>(arg.substr(3, arg.size() - 3));
                }
                else if (arg.find("-w=") == 0)
                {
                    W = FromString<int>(arg.substr(3, arg.size() - 3));
                }
                else if (arg.find("-pa=") == 0)
                {
                    //printAlign = FromString<bool>(arg.substr(4, arg.size() - 4));
                }
                else if (arg.find("-pi=") == 0)
                {
                    //printInternal = FromString<bool>(arg.substr(4, arg.size() - 4));
                }
                else if (arg.find("-wt=") == 0)
                {
                    //workThreads = FromString<size_t>(arg.substr(4, arg.size() - 4));
                }
                else if (arg.find("-mt=") == 0)
                {
                    MINIMAL_TEST_EXECUTION_TIME = FromString<int>(arg.substr(4, arg.size() - 4)) * 0.001;
                }
                else if (arg.find("-lc=") == 0)
                {
                    LITTER_CPU_CACHE = FromString<int>(arg.substr(4, arg.size() - 4));
                }
                else if (arg.find("-ri=") == 0)
                {
                    REAL_IMAGE = arg.substr(4, arg.size() - 4);
                }
                else if (arg.find("-cc=") == 0)
                {
                    //checkCpp = FromString<bool>(arg.substr(4, arg.size() - 4));
                }
                else if (arg.find("-de=") == 0)
                {
                    //DISABLED_EXTENSIONS = FromString<uint32_t>(arg.substr(4, arg.size() - 4));
                }
                else if (arg.find("-wu=") == 0)
                {
                    //WARM_UP_TIME = FromString<int>(arg.substr(4, arg.size() - 4)) * 0.001;
                }
                else if (arg.find("-pt=") == 0)
                {
                    //PIN_THREAD = FromString<bool>(arg.substr(4, arg.size() - 4));
                }
                else
                {
                    TEST_LOG_SS(Error, "Unknown command line options: '" << arg << "'!" << std::endl);
                    exit(1);
                }
            }
        }
    };

    //-------------------------------------------------------------------------------------------------

    SIMD_INLINE bool TestBase(const Options& options)
    {
        return (options.disabledExtensions & 0x000000001) == 0;
    }

    SIMD_INLINE bool TestSse41(const Options& options)
    {
        return (options.disabledExtensions & 0x000000002) == 0;
    }

    SIMD_INLINE bool TestAvx2(const Options& options)
    {
        return (options.disabledExtensions & 0x000000004) == 0;
    }

    SIMD_INLINE bool TestAvx512bw(const Options& options)
    {
        return (options.disabledExtensions & 0x000000008) == 0;
    }

    SIMD_INLINE bool TestAvx512vnni(const Options& options)
    {
        return (options.disabledExtensions & 0x000000010) == 0;
    }

    SIMD_INLINE bool TestAmxBf16(const Options& options)
    {
        return (options.disabledExtensions & 0x000000020) == 0;
    }

    SIMD_INLINE bool TestNeon(const Options& options)
    {
        return (options.disabledExtensions & 0x000000002) == 0;
    }
}

#endif
