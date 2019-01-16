/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#ifndef __SimdRuntime_h__
#define __SimdRuntime_h__

#include "Simd/SimdTime.h"

#include <vector>
#include <limits>
#include <algorithm>
#include <string>

namespace Simd
{
    struct RuntimeGemm
    {
        typedef SimdGemm32fNNPtr Func;
        typedef std::string Name;

        RuntimeGemm()
            : _best(NULL)
        {
        }

        void Init(const Func & func1, const Name & name1, const Func & func2 = NULL, const Name & name2 = Name())
        {
            _candidates.clear();
            _candidates.push_back(Candidate(func1, name1));
            if (func2)
                _candidates.push_back(Candidate(func2, name2));
            _best = _candidates.size() > 1 ? NULL : _candidates[0].func;
        }

        SIMD_INLINE void Run(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            if (_best)
                _best(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            else
                Test(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }

    private:
        static const size_t TEST_COUNT = 3 + 2;

        struct Candidate
        {
            Func func;
            Name name;
            size_t count;
            double sum, sqsum, min, max;

            Candidate(const Func & f, const Name & n)
                : func(f)
                , name(n)
                , count(0)
                , sum(0)
                , sqsum(0)
                , min(std::numeric_limits<double>::max())
                , max(std::numeric_limits<double>::min())
            {
            }

            SIMD_INLINE void Update(const double & value)
            {
                count += 1;
                sum += value;
                sqsum += value * value;
                min = std::min(min, value);
                max = std::max(max, value);
            }

            SIMD_INLINE double Mean() const
            {
                return (sum - min - max) / (count - 2);
            }
        };
        typedef std::vector<Candidate> Candidates;

        Func _best;
        Candidates _candidates;

        SIMD_INLINE void Test(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            assert(_candidates.size());
            Candidate * current = Current();
            if (current)
            {
                double start = Simd::Time();
                current->func(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
                current->Update(Simd::Time() - start);
            }
            else
            {
                _best = Best()->func;
                _best(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
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
}

#endif//__SimdRuntime_h__
