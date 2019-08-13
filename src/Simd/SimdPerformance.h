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
#ifndef __SimdPerformance_h__
#define __SimdPerformance_h__

#include "Simd/SimdDefs.h"

#include <string>
#include <sstream>

namespace Simd
{
    typedef std::string String;

    template <class T> SIMD_INLINE String ToStr(const T & value)
    {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }
}

#if defined(SIMD_PERFORMANCE_STATISTIC)// && defined(NDEBUG)

#include "Simd/SimdTime.h"

#include <limits>
#include <iostream>
#include <iomanip>
#include <memory>
#include <map>
#include <thread>
#include <mutex>
#include <algorithm>

namespace Simd
{
    namespace Base
    {
        class PerformanceMeasurer
        {
            String	_name;
            double _start, _current, _total, _min, _max;
            long long _count, _flop;
            bool _entered, _paused;

        public:
            SIMD_INLINE PerformanceMeasurer(const String & name = "Unknown", long long flop = 0)
                : _name(name)
                , _flop(flop)
                , _count(0)
                , _current(0)
                , _total(0)
                , _min(std::numeric_limits<double>::max())
                , _max(std::numeric_limits<double>::min())
                , _entered(false)
                , _paused(false)
            {
            }

            SIMD_INLINE PerformanceMeasurer(const PerformanceMeasurer & pm)
                : _name(pm._name)
                , _flop(pm._flop)
                , _count(pm._count)
                , _start(pm._start)
                , _current(pm._current)
                , _total(pm._total)
                , _min(pm._min)
                , _max(pm._max)
                , _entered(pm._entered)
                , _paused(pm._paused)
            {
            }

            SIMD_INLINE void Enter()
            {
                if (!_entered)
                {
                    _entered = true;
                    _paused = false;
                    _start = Time();
                }
            }

            SIMD_INLINE void Leave(bool pause = false)
            {
                if (_entered || _paused)
                {
                    if (_entered)
                    {
                        _entered = false;
                        _current += Time() - _start;
                    }
                    if (!pause)
                    {
                        _total += _current;
                        _min = std::min(_min, _current);
                        _max = std::max(_max, _current);
                        ++_count;
                        _current = 0;
                    } 
                    _paused = pause;
                }
            }

            double Average() const
            {
                return _count ? (_total / _count) : 0;
            }

            double GFlops() const
            {
                return _count && _flop && _total > 0 ? (double(_flop * _count) / _total / 1000000000.0f) : 0;
            }

            String Statistic() const
            {
                std::stringstream ss;
                ss << _name << ": ";
                ss << std::setprecision(0) << std::fixed << _total * 1000 << " ms";
                ss << " / " << _count << " = ";
                ss << std::setprecision(3) << std::fixed << Average()*1000.0 << " ms";
                ss << std::setprecision(3) << " {min=" << _min * 1000.0 << "; max=" << _max * 1000.0 << "}";
                if(_flop)
                    ss << " " << std::setprecision(1) << GFlops() << " GFlops";
                return ss.str();
            }

            void Combine(const PerformanceMeasurer & other)
            {
                _count += other._count;
                _total += other._total;
                _min = std::min(_min, other._min);
                _max = std::max(_max, other._max);
            }
        };

        class PerformanceMeasurerHolder
        {
            PerformanceMeasurer * _pm;

        public:
            SIMD_INLINE PerformanceMeasurerHolder(PerformanceMeasurer * pm, bool enter = true)
                : _pm(pm)
            {
                if (_pm && enter)
                    _pm->Enter();
            }

            SIMD_INLINE void Enter()
            {
                if (_pm)
                    _pm->Enter();
            }

            SIMD_INLINE void Leave(bool pause)
            {
                if (_pm)
                    _pm->Leave(pause);
            }

            SIMD_INLINE ~PerformanceMeasurerHolder()
            {
                if (_pm)
                    _pm->Leave();
            }
        };

        class PerformanceMeasurerStorage
        {
            typedef PerformanceMeasurer Pm;
            typedef std::shared_ptr<Pm> PmPtr;
            typedef std::map<String, PmPtr> FunctionMap;
            typedef std::map<std::thread::id, FunctionMap> ThreadMap;

            ThreadMap _map;
            mutable std::recursive_mutex _mutex;
            String _report;

            SIMD_INLINE FunctionMap & ThisThread()
            {
                static thread_local FunctionMap * thread = NULL;
                if (thread == NULL)
                {
                    std::lock_guard<std::recursive_mutex> lock(_mutex);
                    thread = &_map[std::this_thread::get_id()];
                }
                return *thread;
            }

        public:
            static PerformanceMeasurerStorage s_storage;

            PerformanceMeasurerStorage()
            {
            }

            SIMD_INLINE PerformanceMeasurer * Get(const String & name, long long flop = 0)
            {
                FunctionMap & thread = ThisThread();
                PerformanceMeasurer * pm = NULL;
                FunctionMap::iterator it = thread.find(name);
                if (it == thread.end())
                {
                    pm = new PerformanceMeasurer(name, flop);
                    thread[name].reset(pm);
                }
                else
                    pm = it->second.get();
                return pm;
            }

            SIMD_INLINE PerformanceMeasurer * Get(const String func, const String & desc, long long flop = 0)
            {
                return Get(func + "{ " + desc + " }", flop);
            }

            const char * PerformanceStatistic()
            {
                if (_map.empty())
                    return "";
                FunctionMap combined;
                std::lock_guard<std::recursive_mutex> lock(_mutex);
                for (ThreadMap::const_iterator thread = _map.begin(); thread != _map.end(); ++thread)
                {
                    for (FunctionMap::const_iterator function = thread->second.begin(); function != thread->second.end(); ++function)
                    {
                        if (combined.find(function->first) == combined.end())
                            combined[function->first].reset(new PerformanceMeasurer(*function->second));
                        else
                            combined[function->first]->Combine(*function->second);
                    }
                }
                std::stringstream report;
                report << std::endl << "Simd Library Internal Performance Statistics:" << std::endl;
                for (FunctionMap::const_iterator it = combined.begin(); it != combined.end(); ++it)
                    report << it->second->Statistic() << std::endl;
                _report = report.str();
                return _report.c_str();
            }
        };
    }
}
#define SIMD_PERF_FUNCF(flop) Simd::Base::PerformanceMeasurerHolder SIMD_CAT(__pmh, __LINE__)(Simd::Base::PerformanceMeasurerStorage::s_storage.Get(SIMD_FUNCTION, (long long)(flop)))
#define SIMD_PERF_FUNC() SIMD_PERF_FUNCF(0)
#define SIMD_PERF_BEGF(desc, flop) Simd::Base::PerformanceMeasurerHolder SIMD_CAT(__pmh, __LINE__)(Simd::Base::PerformanceMeasurerStorage::s_storage.Get(SIMD_FUNCTION, desc, (long long)(flop)))
#define SIMD_PERF_BEG(desc) SIMD_PERF_BEGF(desc, 0)
#define SIMD_PERF_IFF(cond, desc, flop) Simd::Base::PerformanceMeasurerHolder SIMD_CAT(__pmh, __LINE__)((cond) ? Simd::Base::PerformanceMeasurerStorage::s_storage.Get(SIMD_FUNCTION, desc, (long long)(flop)) : NULL)
#define SIMD_PERF_IF(cond, desc) SIMD_PERF_IFF(cond, desc, 0)
#define SIMD_PERF_END(desc) Simd::Base::PerformanceMeasurerStorage::s_storage.Get(SIMD_FUNCTION, desc)->Leave();
#define SIMD_PERF_INITF(name, desc, flop) Simd::Base::PerformanceMeasurerHolder name(Simd::Base::PerformanceMeasurerStorage::s_storage.Get(SIMD_FUNCTION, desc, (long long)(flop)), false);
#define SIMD_PERF_INIT(name, desc)  SIMD_PERF_INITF(name, desc, 0);
#define SIMD_PERF_START(name) name.Enter(); 
#define SIMD_PERF_PAUSE(name) name.Leave(true);
#define SIMD_PERF_EXT(ext) Simd::Base::PerformanceMeasurerHolder SIMD_CAT(__pmh, __LINE__)((ext)->Perf(SIMD_FUNCTION)) 
#else//SIMD_PERFORMANCE_STATISTIC
#define SIMD_PERF_FUNCF(flop)
#define SIMD_PERF_FUNC()
#define SIMD_PERF_BEGF(desc, flop)
#define SIMD_PERF_BEG(desc)
#define SIMD_PERF_IFF(cond, desc, flop)
#define SIMD_PERF_IF(cond, desc)
#define SIMD_PERF_END(desc)
#define SIMD_PERF_INITF(name, desc, flop)
#define SIMD_PERF_INIT(name, desc)
#define SIMD_PERF_START(name)
#define SIMD_PERF_PAUSE(name)
#define SIMD_PERF_EXT(ext)
#endif//SIMD_PERFORMANCE_STATISTIC 

#endif//__SimdPerformance_h__
