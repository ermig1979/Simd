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

#ifdef SIMD_PERFORMANCE_STATISTIC

#include "Simd/SimdTime.h"

#include <string>
#include <sstream>
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
    typedef std::string String;

    template <class T> SIMD_INLINE String ToStr(const T & value)
    {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }

    class PerformanceMeasurer
    {
        String	_name;
        double _start;
        int _count;
        double _total;
        double _min;
        double _max;
        bool _entered;

    public:
        SIMD_INLINE PerformanceMeasurer(const String & name = "Unknown")
            : _name(name)
            , _count(0)
            , _total(0)
            , _min(std::numeric_limits<double>::max())
            , _max(std::numeric_limits<double>::min())
            , _entered(false)
        {
        }

        SIMD_INLINE PerformanceMeasurer(const PerformanceMeasurer & pm)
            : _name(pm._name)
            , _count(pm._count)
            , _total(pm._total)
            , _min(pm._min)
            , _max(pm._max)
            , _entered(pm._entered)
        {
        }

        SIMD_INLINE void Enter()
        {
            if (!_entered)
            {
                _entered = true;
                _start = Time();
            }
        }

        SIMD_INLINE void Leave()
        {
            if (_entered)
            {
                _entered = false;
                double difference = Time() - _start;
                _total += difference;
                _min = std::min(_min, difference);
                _max = std::max(_max, difference);
                ++_count;
            }
        }

        double Average() const
        {
            return _count ? (_total / _count) : 0;
        }

        String Statistic() const
        {
            std::stringstream ss;
            ss << _name << ": ";
            ss << std::setprecision(0) << std::fixed << _total * 1000 << " ms";
            ss << " / " << _count << " = ";
            ss << std::setprecision(3) << std::fixed << Average()*1000.0 << " ms";
            ss << std::setprecision(3) << " {min=" << _min * 1000.0 << "; max=" << _max * 1000.0 << "}";
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
        SIMD_INLINE PerformanceMeasurerHolder(PerformanceMeasurer * pm)
            : _pm(pm)
        {
            if (_pm)
                _pm->Enter();
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

        SIMD_INLINE FunctionMap & ThisThread()
        {
            std::lock_guard<std::recursive_mutex> lock(_mutex);
            return _map[std::this_thread::get_id()];
        }

    public:
        static PerformanceMeasurerStorage s_storage;

        PerformanceMeasurerStorage()
        {
        }

        ~PerformanceMeasurerStorage()
        {
            if (_map.empty())
                return;
            FunctionMap combined;
            {
                std::lock_guard<std::recursive_mutex> lock(_mutex);
                for (ThreadMap::const_iterator thread = _map.begin(); thread != _map.end(); ++thread)
                {
                    for (FunctionMap::const_iterator function = thread->second.begin(); function != thread->second.end(); ++function)
                    {
                        if (combined.find(function->first) == combined.end())
                            combined[function->first].reset(new PerformanceMeasurer(function->first));
                        combined[function->first]->Combine(*function->second);
                    }
                }
            }
            std::cout << std::endl << "Simd Library Internal Performance Statistic:" << std::endl;
            for (FunctionMap::const_iterator it = combined.begin(); it != combined.end(); ++it)
                std::cout << it->second->Statistic() << std::endl;
        }

        SIMD_INLINE PerformanceMeasurer * Get(const String & name)
        {
            FunctionMap & thread = ThisThread();
            PerformanceMeasurer * pm = NULL;
            FunctionMap::iterator it = thread.find(name);
            if (it == thread.end())
            {
                pm = new PerformanceMeasurer(name);
                thread[name].reset(pm);
            }
            else
                pm = it->second.get();
            return pm;
        }

        SIMD_INLINE PerformanceMeasurer * Get(const String func, const String & block)
        {
            return Get(func + "{ " + block + " }");
        }
    };
}

#define SIMD_PERF_FUNC() Simd::PerformanceMeasurerHolder SIMD_CAT(__pmh, __LINE__)(Simd::PerformanceMeasurerStorage::s_storage.Get(SIMD_FUNCTION))
#define SIMD_PERF_BEG(block) Simd::PerformanceMeasurerHolder SIMD_CAT(__pmh, __LINE__)(Simd::PerformanceMeasurerStorage::s_storage.Get(SIMD_FUNCTION, block))
#define SIMD_PERF_IF(cond, block) Simd::PerformanceMeasurerHolder SIMD_CAT(__pmh, __LINE__)((cond) ? Simd::PerformanceMeasurerStorage::s_storage.Get(SIMD_FUNCTION, block) : NULL)
#define SIMD_PERF_END(block) Simd::PerformanceMeasurerStorage::s_storage.Get(SIMD_FUNCTION, block)->Leave();
#else//SIMD_PERFORMANCE_STATISTIC
#define SIMD_PERF_FUNC()
#define SIMD_PERF_BEG(block)
#define SIMD_PERF_IF(cond, block)
#define SIMD_PERF_END(block)
#endif//SIMD_PERFORMANCE_STATISTIC 

#endif//__SimdPerformance_h__
