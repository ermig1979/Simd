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
#ifndef __SimdRuntime_h__
#define __SimdRuntime_h__

#include "Simd/SimdTime.h"

#include <vector>
#include <limits>
#include <algorithm>
#include <string>
#ifdef SIMD_RUNTIME_STATISTIC
#include <sstream>
#include <iostream>
#include <iomanip>
#endif

namespace Simd
{
    typedef ::std::string String;

    template <class Func, class Args> struct Runtime
    {
        SIMD_INLINE Runtime()
            : _best(NULL)
        {
        }

        SIMD_INLINE ~Runtime()
        {
#ifdef SIMD_RUNTIME_STATISTIC
            if (!_info.empty())
            {
                std::sort(_candidates.begin(), _candidates.end(), [](const Candidate & a, const Candidate & b) { return a.Mean() < b.Mean(); });
                std::cout << std::setprecision(3) << std::fixed;
                std::cout << "Simd::Runtime " << _info << " : ";
                for (size_t i = 0; i < _candidates.size(); ++i)
                    std::cout << _candidates[i].func.Name() << ": " << _candidates[i].Mean()*1000.0 << "  ";
                std::cout << std::endl;
            }
#endif
        }

        SIMD_INLINE void Init(const Func & func)
        {
            _candidates.clear();
            _candidates.push_back(Candidate(func));
            _best = &_candidates[0].func;
        }

        SIMD_INLINE void Init(const std::vector<Func> & funcs)
        {
            assert(funcs.size() >= 1);
            _candidates.clear();
            for (size_t i = 0; i < funcs.size(); ++i)
                _candidates.push_back(Candidate(funcs[i]));
            _best = funcs.size() == 1 ? &_candidates[0].func : NULL;
        }

        SIMD_INLINE void Run(const Args & args)
        {
            if (_best)
                _best->Run(args);
            else
                Test(args);
        }

    private:
        static const size_t TEST_COUNT = 3 + 2;

        struct Candidate
        {
            Func func;
            size_t count;
            double sum, min, max;

            SIMD_INLINE Candidate(const Func & f)
                : func(f)
                , count(0)
                , sum(0)
                , min(std::numeric_limits<double>::max())
                , max(std::numeric_limits<double>::min())
            {
            }

            SIMD_INLINE void Update(const double & value)
            {
                count += 1;
                sum += value;
                min = std::min(min, value);
                max = std::max(max, value);
            }

            SIMD_INLINE double Mean() const
            {
                return (sum - min - max) / (count - 2);
            }
        };
        typedef std::vector<Candidate> Candidates;

        Func * _best;
        Candidates _candidates;
        String _info;

        SIMD_INLINE void Test(const Args & args)
        {
            assert(_candidates.size());
            Candidate * current = Current();
            if (current)
            {
#ifdef SIMD_RUNTIME_STATISTIC
                if (_info.empty())
                    _info = current->Info(args);
#endif
                double start = Simd::Time();
                current->func.Run(args);
                current->Update(Simd::Time() - start);
            }
            else
            {
                _best = &Best()->func;
                _best->Run(args);
            }
        }

        SIMD_INLINE Candidate * Current()
        {
            size_t min = TEST_COUNT;
            Candidate * current = NULL;
            for (size_t i = 0; i < _candidates.size(); ++i)
            {
                if (_candidates[i].count < min)
                {
                    min = _candidates[i].count;
                    current = &_candidates[i];
                }
            }
            return current;
        }

        SIMD_INLINE Candidate * Best()
        {
            Candidate * best = &_candidates[0];
            double min = best->Mean();
            for (size_t i = 1; i < _candidates.size(); ++i)
            {
                double mean = _candidates[i].Mean();
                if (mean < min)
                {
                    min = mean;
                    best = &_candidates[i];
                }
            }
            return best;
        }
    };

    struct GemmArgs
    {
        size_t M; size_t N; size_t K; const float * alpha; const float * A; size_t lda; const float * B; size_t ldb; const float * beta; float * C; size_t ldc;
        SIMD_INLINE GemmArgs(size_t M_, size_t N_, size_t K_, const float * alpha_, const float * A_, size_t lda_, const float * B_, size_t ldb_, const float * beta_, float * C_, size_t ldc_)
            :M(M_), N(N_), K(K_), alpha(alpha_), A(A_), lda(lda_), B(B_), ldb(ldb_), beta(beta_), ldc(ldc_), C(C_) 
        {}
    };

    struct GemmFunc
    {
        typedef SimdGemm32fNNPtr Func;

        SIMD_INLINE GemmFunc(const Func & func, const String & name)
            : _func(func)
            , _name(name)
        {
        }

        SIMD_INLINE String Name() const { return _name; }

        SIMD_INLINE void Run(const GemmArgs & args)
        {
            _func(args.M, args.N, args.K, args.alpha, args.A, args.lda, args.B, args.ldb, args.beta, args.C, args.ldc);
        }

#ifdef SIMD_RUNTIME_STATISTIC
        SIMD_INLINE String Info(const GemmArgs & args) const
        {
            std::stringstream ss;
            ss << "Gemm [" << args.M << ", " << args.N << ", " << args.K << "]";
            return ss.str();
        }
#endif

    private:
        Func _func;
        String _name;
    };
    typedef std::vector<GemmFunc> GemmFuncs;

    SIMD_INLINE GemmFuncs InitGemmFuncs(const GemmFunc::Func & func1, const String & name1, const GemmFunc::Func & func2 = NULL, const String & name2 = String())
    {
        GemmFuncs funcs;
        funcs.push_back(GemmFunc(func1, name1));
        if (func2)
            funcs.push_back(GemmFunc(func2, name2));
        return funcs;
    }

    typedef Runtime<GemmFunc, GemmArgs> RuntimeGemm;
}

#endif//__SimdRuntime_h__
