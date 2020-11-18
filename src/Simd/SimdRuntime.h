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
#ifndef __SimdRuntime_h__
#define __SimdRuntime_h__

#include "Simd/SimdTime.h"
#include "Simd/SimdGemm.h"

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
                std::cout << "Simd::Runtime " << _info << " : ";
                int64_t f = TimeFrequency();
                for (size_t i = 0; i < _candidates.size(); ++i)
                {
                    int64_t t = _candidates[i].Mean();
                    std::cout << _candidates[i].func.Name() << ": " << t * 1000 / f << "." << (t * 1000000 / f) % 1000 << "  ";
                }
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

        SIMD_INLINE size_t Size() const
        {
            return _candidates.size();
        }

        SIMD_INLINE const Func & At(size_t index) const
        {
            return _candidates[index].func;
        }

    private:
        static const size_t TEST_COUNT = 3 + 2;

        struct Candidate
        {
            Func func;
            size_t count;
            int64_t sum, min, max;

            SIMD_INLINE Candidate(const Func & f)
                : func(f)
                , count(0)
                , sum(0)
                , min(std::numeric_limits<int64_t>::max())
                , max(0)
            {
            }

            SIMD_INLINE void Update(int64_t value)
            {
                count += 1;
                sum += value;
                min = std::min(min, value);
                max = std::max(max, value);
            }

            SIMD_INLINE int64_t Mean() const
            {
                if( count > 2)
                    return (sum - min - max) / (count - 2);
                else if (count > 0)
                    return sum / count;
                else
                    return sum;
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
                    _info = current->func.Info(args);
#endif
                int64_t start = Simd::TimeCounter();
                current->func.Run(args);
                current->Update(Simd::TimeCounter() - start);
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
            int64_t min = best->Mean();
            for (size_t i = 1; i < _candidates.size(); ++i)
            {
                int64_t mean = _candidates[i].Mean();
                if (mean < min)
                {
                    min = mean;
                    best = &_candidates[i];
                }
            }
            return best;
        }
    };

    //-------------------------------------------------------------------------

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

    //-------------------------------------------------------------------------

    struct GemmCbArgs
    {
        size_t M; size_t N; size_t K; const float * A; const float * pB; float * C;
        SIMD_INLINE GemmCbArgs(size_t M_, size_t N_, size_t K_, const float * A_, const float * pB_, float * C_)
            :M(M_), N(N_), K(K_), A(A_), pB(pB_), C(C_)
        {}
    };

    struct GemmCbFunc
    {
        typedef size_t (*BufferSizePtr)(size_t M, size_t N, size_t K, GemmKernelType type, bool compatibility);
        typedef void (*ReorderBPtr)(size_t M, size_t N, size_t K, const float * B, float * pB, GemmKernelType type, bool compatibility);
        typedef void (*RunPtr)(size_t M, size_t N, size_t K, const float * A, const float * pB, float * C, GemmKernelType type, bool compatibility);

        SIMD_INLINE GemmCbFunc(BufferSizePtr bufferSize, ReorderBPtr reorderB, RunPtr run, GemmKernelType type, const String & name)
            : _bufferSize(bufferSize)
            , _reorderB(reorderB)
            , _run(run)
            , _type(type)
            , _name(name)
        {
        }

        SIMD_INLINE String Name() const { return _name; }

        SIMD_INLINE void Run(const GemmCbArgs & args)
        {
            _run(args.M, args.N, args.K, args.A, args.pB, args.C, _type, _type != GemmKernelAny);
        }

#ifdef SIMD_RUNTIME_STATISTIC
        SIMD_INLINE String Info(const GemmCbArgs & args) const
        {
            std::stringstream ss;
            ss << "GemmCb [" << args.M << ", " << args.N << ", " << args.K << "]";
            return ss.str();
        }
#endif 
        
        SIMD_INLINE GemmKernelType Type() const { return _type; }

        SIMD_INLINE size_t BufferSize(size_t M, size_t N, size_t K) const
        {
            return _bufferSize(M, N, K, _type, _type != GemmKernelAny);
        }

        SIMD_INLINE void ReorderB(size_t M, size_t N, size_t K, const float * B, float * pB) const
        {
            _reorderB(M, N, K, B, pB, _type, _type != GemmKernelAny);
        }

    private:
        BufferSizePtr _bufferSize;
        ReorderBPtr _reorderB;
        RunPtr _run;
        GemmKernelType _type;
        String _name;
    };
    typedef std::vector<GemmCbFunc> GemmCbFuncs;

    SIMD_INLINE GemmCbFuncs InitGemmCbFuncs(GemmCbFunc::BufferSizePtr bufferSize, GemmCbFunc::ReorderBPtr reorderB, GemmCbFunc::RunPtr run, 
        const String & name, GemmKernelType begin, GemmKernelType end)
    {
        GemmCbFuncs funcs;
        for (int i = (int)begin, n = (int)end; i <= n; ++i)
            funcs.push_back(GemmCbFunc(bufferSize, reorderB, run, GemmKernelType(i), name + "-" + ToStr(i)));
        return funcs;
    }

    typedef Runtime<GemmCbFunc, GemmCbArgs> RuntimeGemmCb;
}

#endif//__SimdRuntime_h__
