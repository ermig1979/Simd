/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Test/TestPerformance.h"
#include "Test/TestUtils.h"
#include "Test/TestTable.h"

#if defined(_MSC_VER)
#define NOMINMAX
#include <windows.h>
#elif defined(__GNUC__)
#include <sys/time.h>
#else
#error Platform is not supported!
#endif

namespace Test
{
#if defined(_MSC_VER)
    double GetFrequency()
    {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        return double(frequency.QuadPart);
    }

    double g_frequency = GetFrequency();

    double GetTime()
    {
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        return double(counter.QuadPart) / g_frequency;
    }
#elif defined(__GNUC__)
    double GetTime()
    {
        timeval t1;
        gettimeofday(&t1, NULL);
        return t1.tv_sec + t1.tv_usec / 1000000.0;
    }
#else
#error Platform is not supported!
#endif

    //-------------------------------------------------------------------------

    PerformanceMeasurer::PerformanceMeasurer(const String & description)
        : _description(description)
        , _count(0)
        , _total(0)
        , _min(std::numeric_limits<double>::max())
        , _max(std::numeric_limits<double>::min())
        , _entered(false)
        , _size(0)
    {
    }

    PerformanceMeasurer::PerformanceMeasurer(const PerformanceMeasurer & pm)
        : _description(pm._description)
        , _count(pm._count)
        , _total(pm._total)
        , _min(pm._min)
        , _max(pm._max)
        , _entered(pm._entered)
        , _size(pm._size)
    {
    }

    void PerformanceMeasurer::Enter()
    {
        if (!_entered)
        {
            _entered = true;
            _start = GetTime();
        }
    }

    void PerformanceMeasurer::Leave(size_t size)
    {
        if (_entered)
        {
            _entered = false;
            double difference = double(GetTime() - _start);
            _total += difference;
            _min = std::min(_min, difference);
            _max = std::max(_max, difference);
            ++_count;
            _size += std::max<size_t>(1, size);
        }
    }

    double PerformanceMeasurer::Average() const
    {
        return _count ? (_total / _count) : 0;
    }

    String PerformanceMeasurer::Statistic() const
    {
        std::stringstream ss;
        ss << _description << ": ";
        ss << std::setprecision(0) << std::fixed << _total * 1000 << " ms";
        ss << " / " << _count << " = ";
        ss << std::setprecision(3) << std::fixed << Average()*1000.0 << " ms";
        ss << std::setprecision(3) << " {min=" << _min*1000.0 << "; max=" << _max*1000.0 << "}";
        if (_size > (long long)_count)
        {
            double size = double(_size);
            ss << std::setprecision(3) << " [<s>=" << size / _count*0.001 << " kb; <t>=" << _total / size * 1000000000 << " ns]";
        }
        return ss.str();
    }

    void PerformanceMeasurer::Combine(const PerformanceMeasurer & other)
    {
        _count += other._count;
        _total += other._total;
        _min = std::min(_min, other._min);
        _max = std::max(_max, other._max);
        _size += other._size;
    }

    //-------------------------------------------------------------------------

    PerformanceMeasurerStorage PerformanceMeasurerStorage::s_storage;

    PerformanceMeasurerStorage::PerformanceMeasurerStorage()
    {
    }

    PerformanceMeasurerStorage::~PerformanceMeasurerStorage()
    {
    }

    PerformanceMeasurerStorage::Thread & PerformanceMeasurerStorage::ThisThread()
    {
        std::lock_guard<std::recursive_mutex> lock(_mutex);
        return _map[std::this_thread::get_id()];
    }

    PerformanceMeasurer* PerformanceMeasurerStorage::Get(String name)
    {
        Thread & thread = ThisThread();
        name = name + (thread.align ? "{a}" : "{u}");
        PerformanceMeasurer * pm = NULL;
        FunctionMap::iterator it = thread.map.find(name);
        if (it == thread.map.end())
        {
            pm = new PerformanceMeasurer(name);
            thread.map[name].reset(pm);
        }
        else
            pm = it->second.get();
        return pm;
    }

    size_t PerformanceMeasurerStorage::Align(size_t size)
    {
        Thread & thread = ThisThread();
        thread.align = size%SIMD_ALIGN == 0;
        return thread.align ? SIMD_ALIGN : sizeof(void*);
    }

    static String FunctionShortName(const String & description)
    {
        bool isApi = description.find("Simd::") == std::string::npos;
        if (isApi)
        {
            return description.substr(4, description.size() - 7);
        }
        else
        {
            size_t pos = description.rfind("::");
            return description.substr(pos + 2, description.size() - pos - 5);
        }
    }

    template <class Measurer> double Relation(const Measurer & a, const Measurer & b)
    {
        return b.Average() > 0 ? a.Average() / b.Average() : 0;
    }

    class CommonPerformance
    {
        int _count;
        double _value;

    public:
        CommonPerformance()
            : _count(0)
            , _value(0)
        {
        }

        void Add(const PerformanceMeasurer & pm)
        {
            if (pm.Average() > 0)
            {
                _count++;
                _value += ::log(pm.Average());
            }
        }

        double Average() const
        {
            return _count > 0 ? ::exp(_value / _count) : 0;
        }
    };

    struct Name
    {
        const char * full;
        const char * brief;
    };

    template <class T> struct Statistic
    {
        T simd;
        T base;
        T sse;
        T sse2;
        T ssse3;
        T sse41;
        T avx;
        T avx2;
        T avx512;
        T vmx;
        T vsx;
        T neon;

        size_t Size() const { return sizeof(Statistic) / sizeof(T); };
        T & operator [] (size_t i) { return (&simd)[i]; }
        const T & operator [] (size_t i) const { return (&simd)[i]; }
    };
    typedef std::pair<PerformanceMeasurer, PerformanceMeasurer> Function;
    typedef Statistic<Function> FunctionStatistic;
    typedef std::map<std::string, FunctionStatistic> FunctionStatisticMap;
    typedef std::pair<CommonPerformance, CommonPerformance> Common;
    typedef Statistic<Common> CommonStatistic;
    typedef Statistic<bool> StatisticEnable;
    typedef Statistic<Name> StatisticNames;

    template <class T> const T & Previous(const T & f)
    {
        return (&f)[-1].first.Average() > 0 ? (&f)[-1] : Previous((&f)[-1]);
    }

    template<class Printer>
    static String Print(const String & name, const Printer & printer, const StatisticEnable & enable, bool align)
    {
        std::stringstream ss;

        ss << name << " " << printer.Separator();

        ss << printer.Average(printer.data.simd);
        ss << printer.Average(printer.data.base);
        if (enable.sse) ss << printer.Average(printer.data.sse);
        if (enable.sse2) ss << printer.Average(printer.data.sse2);
        if (enable.ssse3) ss << printer.Average(printer.data.ssse3);
        if (enable.sse41) ss << printer.Average(printer.data.sse41);
        if (enable.avx) ss << printer.Average(printer.data.avx);
        if (enable.avx2) ss << printer.Average(printer.data.avx2);
        if (enable.avx512) ss << printer.Average(printer.data.avx512);
        if (enable.vmx) ss << printer.Average(printer.data.vmx);
        if (enable.vsx) ss << printer.Average(printer.data.vsx);
        if (enable.neon) ss << printer.Average(printer.data.neon);
        ss << printer.Separator();

        if (enable.sse || enable.sse2 || enable.ssse3 || enable.sse41 || enable.avx || enable.avx2 || enable.avx512 || enable.vmx || enable.vsx || enable.neon)
        {
            if (enable.sse) ss << printer.Relation(printer.data.base, printer.data.sse);
            if (enable.sse2) ss << printer.Relation(printer.data.base, printer.data.sse2);
            if (enable.ssse3) ss << printer.Relation(printer.data.base, printer.data.ssse3);
            if (enable.sse41) ss << printer.Relation(printer.data.base, printer.data.sse41);
            if (enable.avx) ss << printer.Relation(printer.data.base, printer.data.avx);
            if (enable.avx2) ss << printer.Relation(printer.data.base, printer.data.avx2);
            if (enable.avx512) ss << printer.Relation(printer.data.base, printer.data.avx512);
            if (enable.vmx) ss << printer.Relation(printer.data.base, printer.data.vmx);
            if (enable.vsx) ss << printer.Relation(printer.data.base, printer.data.vsx);
            if (enable.neon) ss << printer.Relation(printer.data.base, printer.data.neon);
            ss << printer.Separator();
        }

        if (enable.sse || enable.sse2 || enable.ssse3 || enable.sse41 || enable.avx || enable.avx2 || enable.avx512)
        {
            if (enable.sse) ss << printer.Improving(printer.data.sse);
            if (enable.sse2) ss << printer.Improving(printer.data.sse2);
            if (enable.ssse3) ss << printer.Improving(printer.data.ssse3);
            if (enable.sse41) ss << printer.Improving(printer.data.sse41);
            if (enable.avx) ss << printer.Improving(printer.data.avx);
            if (enable.avx2) ss << printer.Improving(printer.data.avx2);
            if (enable.avx512) ss << printer.Improving(printer.data.avx512);
            if (enable.vmx) ss << printer.Improving(printer.data.vmx);
            if (enable.vsx) ss << printer.Improving(printer.data.vsx);
            if (enable.neon) ss << printer.Improving(printer.data.neon);
            ss << printer.Separator();
        }

        if (align)
        {
            ss << printer.Alignment(printer.data.simd);
            ss << printer.Alignment(printer.data.base);
            if (enable.sse) ss << printer.Alignment(printer.data.sse);
            if (enable.sse2) ss << printer.Alignment(printer.data.sse2);
            if (enable.ssse3) ss << printer.Alignment(printer.data.ssse3);
            if (enable.sse41) ss << printer.Alignment(printer.data.sse41);
            if (enable.avx) ss << printer.Alignment(printer.data.avx);
            if (enable.avx2) ss << printer.Alignment(printer.data.avx2);
            if (enable.avx512) ss << printer.Alignment(printer.data.avx512);
            if (enable.vmx) ss << printer.Alignment(printer.data.vmx);
            if (enable.vsx) ss << printer.Alignment(printer.data.vsx);
            if (enable.neon) ss << printer.Alignment(printer.data.neon);
            ss << printer.Separator();
        }

        return ss.str();
    }

    struct TextHeaderPrinter
    {
        const StatisticNames & data;
        size_t average, relation, fraction;

        TextHeaderPrinter(const StatisticNames & d, size_t a, size_t r, size_t f)
            : data(d), average(a), relation(r), fraction(f) {}

        String Average(const Name & a) const
        {
            return ExpandToLeft(std::string(a.full), average + fraction + 1) + " ";
        }

        String Relation(const Name & a, const Name & b) const
        {
            return ExpandToLeft(std::string(a.brief) + "/" + std::string(b.brief), relation + fraction) + " ";
        }

        String Improving(const Name & a) const
        {
            return ExpandToLeft("P/" + std::string(a.brief), relation + fraction) + " ";
        }

        String Alignment(const Name & a) const
        {
            return ExpandToLeft(std::string(a.brief) + ":U/A", relation + fraction + 1) + " ";
        }

        String Separator() const
        {
            return "| ";
        }
    };

    template<class Value> struct TextValuePrinter
    {
        const Statistic<Value> & data;
        size_t average, relation, fraction;

        TextValuePrinter(const Statistic<Value> & d, size_t a, size_t r, size_t f)
            : data(d), average(a), relation(r), fraction(f) {}

        String Average(const Value & a) const
        {
            return ToString(a.first.Average()*1000.0, average, fraction) + " ";
        }

        String Relation(const Value & a, const Value & b) const
        {
            return ToString(Test::Relation(a.first, b.first), relation, fraction - 1) + " ";
        }

        String Improving(const Value & a) const
        {
            return ToString(Test::Relation(Previous(a).first, a.first), relation, fraction - 1) + " ";
        }

        String Alignment(const Value & a) const
        {
            return ToString(Test::Relation(a.second, a.first), relation, fraction) + " ";
        }

        String Separator() const
        {
            return "| ";
        }
    };

    static inline void AddToFunction(const PerformanceMeasurer & src, Function & dst, bool & enable)
    {
        const String & desc = src.Description();
        bool align = desc[desc.size() - 2] == 'a';
        (align ? dst.first : dst.second) = src;
        enable = true;
    }

    static inline void AddToFunction(const PerformanceMeasurer & src, FunctionStatistic & dst, StatisticEnable & enable)
    {
        const String & desc = src.Description();
        if (desc.find("Simd::") == std::string::npos && desc.find("Simd") == 0)
            AddToFunction(src, dst.simd, enable.simd);
        if (desc.find("Simd::Base::") != std::string::npos)
            AddToFunction(src, dst.base, enable.base);
        if (desc.find("Simd::Sse::") != std::string::npos)
            AddToFunction(src, dst.sse, enable.sse);
        if (desc.find("Simd::Sse2::") != std::string::npos)
            AddToFunction(src, dst.sse2, enable.sse2);
        if (desc.find("Simd::Ssse3::") != std::string::npos || desc.find("Simd::Sse3::") != std::string::npos)
            AddToFunction(src, dst.ssse3, enable.ssse3);
        if (desc.find("Simd::Sse41::") != std::string::npos || desc.find("Simd::Sse42::") != std::string::npos)
            AddToFunction(src, dst.sse41, enable.sse41);
        if (desc.find("Simd::Avx::") != std::string::npos)
            AddToFunction(src, dst.avx, enable.avx);
        if (desc.find("Simd::Avx2::") != std::string::npos)
            AddToFunction(src, dst.avx2, enable.avx2);
        if (desc.find("Simd::Avx512f::") != std::string::npos || desc.find("Simd::Avx512bw::") != std::string::npos)
            AddToFunction(src, dst.avx512, enable.avx512);
        if (desc.find("Simd::Vmx::") != std::string::npos)
            AddToFunction(src, dst.vmx, enable.vmx);
        if (desc.find("Simd::Vsx::") != std::string::npos)
            AddToFunction(src, dst.vsx, enable.vsx);
        if (desc.find("Simd::Neon::") != std::string::npos)
            AddToFunction(src, dst.neon, enable.neon);
    }

    static inline const Function & Cond(const Function & a, const Function & b)
    {
        return a.first.Average() > 0 ? a : b;
    }

    static inline void Add(const Function & src, Common & dst)
    {
        dst.first.Add(src.first);
        dst.second.Add(src.second);
    }

    static void AddToCommon(const FunctionStatistic & s, const StatisticEnable & enable, CommonStatistic & d)
    {
        Add(s.simd, d.simd);
        Add(s.base, d.base);
        if (enable.sse) Add(Cond(s.sse, s.base), d.sse);
        if (enable.sse2) Add(Cond(s.sse2, Cond(s.sse, s.base)), d.sse2);
        if (enable.ssse3) Add(Cond(s.ssse3, Cond(s.sse2, Cond(s.sse, s.base))), d.ssse3);
        if (enable.sse41) Add(Cond(s.sse41, Cond(s.ssse3, Cond(s.sse2, Cond(s.sse, s.base)))), d.sse41);
        if (enable.avx) Add(Cond(s.avx, Cond(s.sse41, Cond(s.ssse3, Cond(s.sse2, Cond(s.sse, s.base))))), d.avx);
        if (enable.avx2) Add(Cond(s.avx2, Cond(s.avx, Cond(s.sse41, Cond(s.ssse3, Cond(s.sse2, Cond(s.sse, s.base)))))), d.avx2);
        if (enable.avx512) Add(Cond(s.avx512, Cond(s.avx2, Cond(s.avx, Cond(s.sse41, Cond(s.ssse3, Cond(s.sse2, Cond(s.sse, s.base))))))), d.avx512);
        if (enable.vmx) Add(Cond(s.vmx, s.base), d.vmx);
        if (enable.vsx) Add(Cond(s.vsx, Cond(s.vmx, s.base)), d.vsx);
        if (enable.neon) Add(Cond(s.neon, s.base), d.neon);
    }

    String PerformanceMeasurerStorage::TextReport(bool align, bool raw) const
    {
        FunctionMap map;
        {
            std::lock_guard<std::recursive_mutex> lock(_mutex);
            for (ThreadMap::const_iterator thread = _map.begin(); thread != _map.end(); ++thread)
            {
                for (FunctionMap::const_iterator function = thread->second.map.begin(); function != thread->second.map.end(); ++function)
                {
                    if (map.find(function->first) == map.end())
                        map[function->first].reset(new PerformanceMeasurer(function->first));
                    map[function->first]->Combine(*function->second);
                }
            }
        }

        FunctionStatisticMap functions;
        CommonStatistic common;
        StatisticEnable enable = { false, false, false, false, false, false, false, false, false, false, false, false };
        StatisticNames names = { {"Simd", "S"}, {"Base", "B"}, { "Sse", "S1" }, {"Sse2", "S2"}, {"Ssse3", "S3"}, {"Sse41", "S4"}, { "Avx", "A1" }, {"Avx2", "A2"}, { "Avx5", "A5" }, {"Vmx", "Vm"}, {"Vsx", "Vs"}, { "Neon", "N"} };
        double timeMax = 0;
        size_t sizeMax = 8;
        for (FunctionMap::const_iterator it = map.begin(); it != map.end(); ++it)
        {
            const PerformanceMeasurer & pm = *it->second;
            String name = FunctionShortName(pm.Description());
            AddToFunction(pm, functions[name], enable);
            timeMax = std::max(timeMax, pm.Average());
            sizeMax = std::max(name.size(), sizeMax);
        }

        for (FunctionStatisticMap::const_iterator it = functions.begin(); it != functions.end(); ++it)
            AddToCommon(it->second, enable, common);

        const size_t average = 1 + (size_t)::log10(std::max(timeMax * 1000, 1.0));
        const size_t relative = 3;
        const size_t fraction = 3;

        String header = Print(ExpandToRight("Function", sizeMax), TextHeaderPrinter(names, average, relative, fraction), enable, align);

        std::vector<std::string> statistics;
        for (FunctionStatisticMap::const_iterator it = functions.begin(); it != functions.end(); ++it)
            statistics.push_back(Print(ExpandToRight(it->first, sizeMax), TextValuePrinter<Function>(it->second, average, relative, fraction), enable, align));
        std::sort(statistics.begin(), statistics.end());

        std::stringstream separator;
        for (size_t i = 0; i < header.size(); ++i)
            separator << "-";

        std::stringstream report;

        if (raw)
        {
            report << std::endl << std::endl << "Performance report:" << std::endl << std::endl;
            for (FunctionMap::const_iterator it = map.begin(); it != map.end(); ++it)
                report << it->second->Statistic() << std::endl;
        }
        else
        {
            report << std::endl << std::endl << "Performance report:" << std::endl << std::endl;
            report << separator.str() << std::endl;
            report << header << std::endl;
            report << separator.str() << std::endl;
            report << Print(ExpandToRight("Common", sizeMax), TextValuePrinter<Common>(common, average, relative, fraction), enable, align) << std::endl;
            report << separator.str() << std::endl;
            for (size_t i = 0; i < statistics.size(); ++i)
                report << statistics[i] << std::endl;
            report << separator.str() << std::endl;
        }

        return report.str();
    }

    void PerformanceMeasurerStorage::Clear()
    {
        _map.clear();
    }

    static const char * INDENT = "  ";
    const char * STYLE_HEADER = "background-color:#eeeeee; font-weight:bold";

    typedef std::pair<String, String> Attribute;
    typedef std::vector<Attribute> Attributes;

    static inline Attributes Attr()
    {
        return Attributes();
    }

    static inline Attributes Attr(
        const String & name0, const String & value0)
    {
        Attributes attrbutes;
        attrbutes.push_back(Attribute(name0, value0));
        return attrbutes;
    }

    static inline Attributes Attr(
        const String & name0, const String & value0,
        const String & name1, const String & value1)
    {
        Attributes attrbutes;
        attrbutes.push_back(Attribute(name0, value0));
        attrbutes.push_back(Attribute(name1, value1));
        return attrbutes;
    }

    static inline Attributes Attr(
        const String & name0, const String & value0,
        const String & name1, const String & value1,
        const String & name2, const String & value2)
    {
        Attributes attrbutes;
        attrbutes.push_back(Attribute(name0, value0));
        attrbutes.push_back(Attribute(name1, value1));
        attrbutes.push_back(Attribute(name2, value2));
        return attrbutes;
    }

    struct Html
    {
        Html(const String & path)
            : _indent(0)
        {
            _stream.open(path.c_str());
        }

        ~Html()
        {
            if(_stream.is_open())
                _stream.close();
        }

        bool Good() const
        {
            return _stream.is_open();
        }

        void WriteIndent()
        {
            for (int i = 0; i < _indent; ++i)
                _stream << INDENT;
        }

        void WriteAtribute(const Attribute & attribute)
        {
            _stream << " " << attribute.first << "=\"" << attribute.second << "\"";
        }

        void WriteBegin(const String & name, const Attributes & attributes, bool indent, bool line)
        {
            WriteIndent();
            _stream << "<" << name;
            for (size_t i = 0; i < attributes.size(); ++i)
                WriteAtribute(attributes[i]);
            _stream << ">";
            if (line)
                _stream << std::endl;
            if (indent)
                _indent++;
        }

        void WriteEnd(const String & name, bool indent, bool line)
        {
            if (indent)
            {
                _indent--;
                WriteIndent();
            }
            _stream << "</" << name << ">";
            if (line)
                _stream << std::endl;
        }

        void WriteValue(const String & name, const Attributes & attributes, const String & value, bool line)
        {
            WriteBegin(name, attributes, false, false);
            _stream << value;
            WriteEnd(name, false, line);
        }

        void WriteText(const String & text, bool indent, bool line)
        {
            if (indent)
                WriteIndent();
            _stream << text;
            if (line)
                _stream << std::endl;
        }
       
    private:
        std::ofstream _stream;
        int _indent;
    };

    struct HtmlHeaderPrinter
    {
        const StatisticNames & data;

        HtmlHeaderPrinter(const StatisticNames & d)
            : data(d) {}

        String Average(const Name & a) const
        {
            return std::string(a.full) + "</td><td>";
        }

        String Relation(const Name & a, const Name & b) const
        {
            return std::string(a.brief) + "/" + std::string(b.brief)+ "</td><td>";
        }

        String Improving(const Name & a) const
        {
            return "P/" + std::string(a.brief) + "</td><td>";
        }

        String Alignment(const Name & a) const
        {
            return std::string(a.brief) + ":U/A" + "</td><td>";
        }

        String Separator() const
        {
            return "</td><td>";
        }
    };

    template<class Value> struct HtmlValuePrinter
    {
        const Statistic<Value> & data;

        HtmlValuePrinter(const Statistic<Value> & d)
            : data(d) {}

        String Average(const Value & a) const
        {
            return ToString(a.first.Average()*1000.0) + "</td><td>";
        }

        String Relation(const Value & a, const Value & b) const
        {
            return ToString(Test::Relation(a.first, b.first)) + "</td><td>";
        }

        String Improving(const Value & a) const
        {
            return ToString(Test::Relation(Previous(a).first, a.first)) + "</td><td>";
        }

        String Alignment(const Value & a) const
        {
            return ToString(Test::Relation(a.second, a.first)) + "</td><td>";
        }

        String Separator() const
        {
            return "</td><td>";
        }
    };

    template <class Value> void AddRow(Table & table, size_t row, const String & name, const Statistic<Value> & statistic, const StatisticEnable & enable, bool align)
    {
        size_t col = 0;
        table.SetCell(col++, row, name);
        for (size_t i = 0; i < statistic.Size(); ++i)
            if (enable[i])
                table.SetCell(col++, row, statistic[i].first.Average()*1000.0);
        for (size_t i = 2; i < statistic.Size(); ++i)
            if (enable[i])
                table.SetCell(col++, row, Test::Relation(statistic[1].first, statistic[i].first));
        for (size_t i = 2; i < statistic.Size(); ++i)
            if (enable[i])
                table.SetCell(col++, row, Test::Relation(Previous(statistic[i]).first, statistic[i].first));
        if (align)
        {
            for (size_t i = 0; i < statistic.Size(); ++i)
                if (enable[i])
                    table.SetCell(col++, row, Test::Relation(statistic[i].second, statistic[i].first));
        }
    }

    bool PerformanceMeasurerStorage::HtmlReport(const String & path, bool align) const
    {
        Html html(path);
        if (!html.Good())
            return false;

        html.WriteBegin("html", Attr(), true, true);
        html.WriteValue("title", Attr(), "Simd Library Perfromance Report", true);
        html.WriteBegin("body", Attr(), true, true);

        html.WriteValue("h1", Attr("id", "home"), "Simd Library Perfromance Report", true);

        html.WriteText("Test generation time: " + GetCurrentDateTimeString(), true, true);

        Attributes attributes;
        attributes.push_back(Attribute("align", "center"));
        attributes.push_back(Attribute("cellpadding", "2"));
        attributes.push_back(Attribute("cellspacing", "0"));
        attributes.push_back(Attribute("border", "1"));
        attributes.push_back(Attribute("cellpadding", "2"));
        attributes.push_back(Attribute("width", "100%"));
        html.WriteBegin("table", attributes, true, true);

        FunctionMap map;
        {
            std::lock_guard<std::recursive_mutex> lock(_mutex);
            for (ThreadMap::const_iterator thread = _map.begin(); thread != _map.end(); ++thread)
            {
                for (FunctionMap::const_iterator function = thread->second.map.begin(); function != thread->second.map.end(); ++function)
                {
                    if (map.find(function->first) == map.end())
                        map[function->first].reset(new PerformanceMeasurer(function->first));
                    map[function->first]->Combine(*function->second);
                }
            }
        }

        FunctionStatisticMap functions;
        CommonStatistic common;
        StatisticEnable enable = { false, false, false, false, false, false, false, false, false, false, false, false };
        StatisticNames names = { { "Simd", "S" },{ "Base", "B" },{ "Sse", "S1" },{ "Sse2", "S2" },{ "Ssse3", "S3" },{ "Sse41", "S4" },{ "Avx", "A1" },{ "Avx2", "A2" },{ "Avx5", "A5" },{ "Vmx", "Vm" },{ "Vsx", "Vs" },{ "Neon", "N" } };
        for (FunctionMap::const_iterator it = map.begin(); it != map.end(); ++it)
        {
            const PerformanceMeasurer & pm = *it->second;
            String name = FunctionShortName(pm.Description());
            AddToFunction(pm, functions[name], enable);
        }

        for (FunctionStatisticMap::const_iterator it = functions.begin(); it != functions.end(); ++it)
            AddToCommon(it->second, enable, common);

        html.WriteBegin("tr", Attr("style", STYLE_HEADER), true, true);
        html.WriteBegin("td", Attr(), false, false);
        html.WriteText(Print("Function", HtmlHeaderPrinter(names), enable, align), false, false);
        html.WriteEnd("td", false, false);
        html.WriteEnd("tr", true, true);

        html.WriteBegin("tr", Attr(), true, true);
        html.WriteBegin("td", Attr(), false, false);
        html.WriteText(Print("Common", HtmlValuePrinter<Common>(common), enable, align), false, false);
        html.WriteEnd("td", false, false);
        html.WriteEnd("tr", true, true);

        for (FunctionStatisticMap::const_iterator it = functions.begin(); it != functions.end(); ++it)
        {
            html.WriteBegin("tr", Attr(), true, true);
            html.WriteBegin("td", Attr(), false, false);
            html.WriteText(Print(it->first, HtmlValuePrinter<Function>(it->second), enable, align), false, false);
            html.WriteEnd("td", false, false);
            html.WriteEnd("tr", true, true);

        }

        html.WriteEnd("table", true, true);
        html.WriteEnd("body", true, true);
        html.WriteEnd("html", true, true);

        {
            size_t n = 0;
            for (size_t i = 2; i < enable.Size(); ++i)
            {
                if (enable[i])
                    n++;
            }
            size_t w = 1 + 2 + n * 3 + (align ? 2 + n : 0);
            size_t h = 1 + 1 + functions.size();

            Table table(w, h);

            size_t row = 1;
            AddRow(table, row++, "Common", common, enable, align);
            for (FunctionStatisticMap::const_iterator it = functions.begin(); it != functions.end(); ++it)
                AddRow(table, row++, it->first, it->second, enable, align);

        }

        return true;
    }
}
