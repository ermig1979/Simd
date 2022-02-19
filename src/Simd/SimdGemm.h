/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#ifndef __SimdGemm_h__
#define __SimdGemm_h__

#if defined(__GNUC__) && ((__GNUC__ > 10) || ((__GNUC__ == 10) && (__GNUC_MINOR__ >= 1) && (__GNUC_MINOR__ <= 3)))
#define SIMD_FUTURE_DISABLE
#endif

#include "Simd/SimdArray.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdParallel.hpp"
#include "Simd/SimdPerformance.h"

#ifdef _N
#undef _N
#endif

namespace Simd
{
    template <class T, size_t F, class TM> class GemmNN
    {
    public:
        typedef void(*Main)(size_t K, T alpha, const T * A, size_t lda, const T * B, size_t ldb, size_t sb, T * C, size_t ldc, TM tail);
        typedef void(*Tail)(size_t M, size_t K, T alpha, const T * A, size_t lda, const T * B, size_t ldb, size_t sb, T * C, size_t ldc, TM tail);
        typedef void(*PackA)(const T * A, size_t lda, size_t M, size_t K, size_t microM, T * pA);
        typedef void(*PackB)(const T * B, size_t ldb, size_t K, size_t N, size_t microN, T * pB);
        typedef void(*ScaleC)(size_t M, size_t N, T beta, T * C, size_t ldc);
        typedef TM(*TailMask)(ptrdiff_t tail);

        GemmNN(size_t M, size_t N, size_t K, size_t microM, size_t microN, size_t L1, size_t L2, size_t L3,
            Main kernelMM, Main kernelMT, Tail kernelTM, Tail kernelTT, PackA packA, PackB packB, ScaleC scaleC, TailMask tailMask)
            : _M(M)
            , _N(N)
            , _K(K)
            , _microM(microM)
            , _microN(microN)
            , _threadNumber(Base::GetThreadNumber())
            , _kernelMM(kernelMM)
            , _kernelMT(kernelMT)
            , _kernelTM(kernelTM)
            , _kernelTT(kernelTT)
            , _scaleC(scaleC)
            , _packB(packB)
            , _packA(packA)
        {
            _macroK = Simd::Min(L1 / sizeof(T) / _microN, _K);
            _macroM = Simd::RestrictRange(AlignLoAny(L2 / sizeof(T) / _macroK, _microM), _microM, AlignHiAny(_M, _microM));
            _macroN = Simd::RestrictRange(AlignLoAny(L3 / sizeof(T) / _macroK, _microN), _microN, AlignHiAny(_N, _microN));
            if (_N * _M * _K < 256 * 256 * 256 * 2)
                _threadNumber = 1;
            _pA.resize(_threadNumber);
            _pB.resize(_threadNumber);
            for (size_t t = 0; t < _threadNumber; ++t) 
            {
                _pA[t].Resize(_macroM * _macroK);
                _pB[t].Resize(_macroN * _macroK);
            }
            size_t NF = AlignLo(_N, F);
            if (tailMask)
            {
                _main = TM(-1);
                _tail = NF == _N ? TM(-1) : tailMask(_N - NF);
            }
            else
            {
                _main = TM(F);
                _tail = NF == _N ? TM(F) : TM(_N - NF);
            }
        }

        void Run(const T * alpha, const T * A, size_t lda, const T * B, size_t ldb, const T * beta, T * C, size_t ldc)
        {
            Simd::Parallel(0, _N, [&](size_t thread, size_t begin, size_t end)
            {
                ThreadKernel(end - begin, *alpha, A, lda, B + begin, ldb, *beta, C + begin, ldc, thread);
            }, _threadNumber, _microN);
        }

    private:

        void ThreadKernel(size_t N, T alpha, const T * A, size_t lda, const T * B, size_t ldb, T beta, T * C, size_t ldc, size_t thread)
        {
            for (size_t j = 0; j < N; j += _macroN)
            {
                size_t macroN = Simd::Min(N, j + _macroN) - j;
                for (size_t k = 0; k < _K; k += _macroK)
                {
                    size_t macroK = Simd::Min(_K, k + _macroK) - k;
                    for (size_t i = 0; i < _M; i += _macroM)
                    {
                        size_t macroM = Simd::Min(_M, i + _macroM) - i;
                        if (k == 0)
                            _scaleC(macroM, macroN, beta, C + i * ldc + j, ldc);
                        MacroKernel(macroM, macroN, macroK, alpha, A + i * lda + k, lda, B + k * ldb + j, ldb, beta, C + i * ldc + j, ldc, i == 0, thread);
                    }
                }
            }
        }

        void MacroKernel(size_t M, size_t N, size_t K, T alpha, const T * A, size_t lda, const T * B, size_t ldb, T beta, T * C, size_t ldc, bool packB, size_t thread)
        {
            size_t klda = lda;
            if (_packA)
            {
                _packA(A, lda, M, K, _microM, _pA[thread].data);
                A = _pA[thread].data;
                lda = K;
                klda = 1;
            }
            size_t MA = AlignLoAny(M, _microM);
            size_t NA = AlignLoAny(N, _microN);
            size_t j = 0;
            for (; j < NA; j += _microN)
            {
                T * pB = _pB[thread].data + j * _macroK;
                if (packB)
                    _packB(B + j, ldb, K, _microN, _microN, pB);
                size_t i = 0;
                for (; i < MA; i += _microM)
                    _kernelMM(K, alpha, A + i * lda, klda, pB, F, _microN, C + i * ldc + j, ldc, _main);
                if (i < M)
                    _kernelTM(M - i, K, alpha, A + i * lda, klda, pB, F, _microN, C + i * ldc + j, ldc, _main);
            }
            if (j < N)
            {
                T * pB = _pB[thread].data + j * _macroK;
                if (packB)
                    _packB(B + j, ldb, K, N - j, _microN, pB);
                size_t i = 0;
                for (; i < MA; i += _microM)
                    _kernelMT(K, alpha, A + i * lda, klda, pB, F, _microN, C + i * ldc + j, ldc, _tail);
                if (i < M)
                    _kernelTT(M - i, K, alpha, A + i * lda, klda, pB, F, _microN, C + i * ldc + j, ldc, _tail);
            }
        }

        typedef std::vector<Simd::Array<T>> Arrays;

        Arrays _pA, _pB;
        size_t _M, _N, _K, _microM, _microN, _macroM, _macroN, _macroK, _threadNumber;
        TM _main, _tail;
        Main _kernelMM, _kernelMT;
        Tail _kernelTM, _kernelTT;
        ScaleC _scaleC;
        PackB _packB;
        PackA _packA;
    };

    template <class T, size_t F> class GemmNT
    {
    public:
        typedef void(*Kernel)(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc);
        typedef void(*ScaleC)(size_t M, size_t N, T beta, T * C, size_t ldc);

        GemmNT(size_t M, size_t N, size_t K, size_t L1, size_t L2, size_t L3, ScaleC scaleC,
            Kernel k1x1, Kernel k1x4, Kernel k2x1, Kernel k2x4, Kernel k3x1, Kernel k3x4, Kernel k6x1, Kernel k6x4)
            : _M(M)
            , _N(N)
            , _K(K)
            , _threadNumber(Base::GetThreadNumber())
            , _scaleC(scaleC)
            , _k1x1(k1x1)
            , _k1x4(k1x4)
            , _k2x1(k2x1)
            , _k2x4(k2x4)
            , _k3x1(k3x1)
            , _k3x4(k3x4)
            , _k6x1(k6x1)
            , _k6x4(k6x4)
        {
            _microN = 4;
            _microM = _k6x4 ? 6 : 3;
            _macroK = AlignLo(L1 / sizeof(T) / _microN, F);
            _macroM = AlignLoAny(L2 / sizeof(T) / _macroK, _microM);
            _macroN = AlignLoAny(L3 / sizeof(T) / _macroK, _microN);
            if (_N * _M * _K < 256 * 256 * 256 * 2)
                _threadNumber = 1;
        }

        void Run(const T * alpha, const T * A, size_t lda, const T * B, size_t ldb, const T * beta, T * C, size_t ldc)
        {
            Simd::Parallel(0, _N, [&](size_t thread, size_t begin, size_t end)
            {
                ThreadKernel(end - begin, *alpha, A, lda, B + begin*ldb, ldb, *beta, C + begin, ldc, thread);
            }, _threadNumber, _microN);
        }

    private:

        void ThreadKernel(size_t N, T alpha, const T * A, size_t lda, const T * B, size_t ldb, T beta, T * C, size_t ldc, size_t thread)
        {
            for (size_t j = 0; j < N; j += _macroN)
            {
                size_t macroN = Simd::Min(N, j + _macroN) - j;
                for (size_t k = 0; k < _K; k += _macroK)
                {
                    size_t macroK = Simd::Min(_K, k + _macroK) - k;
                    for (size_t i = 0; i < _M; i += _macroM)
                    {
                        size_t macroM = Simd::Min(_M, i + _macroM) - i;
                        if (k == 0)
                            _scaleC(macroM, macroN, beta, C + i * ldc + j, ldc);
                        MacroKernel(macroM, macroN, macroK, alpha, A + i * lda + k, lda, B + j * ldb + k, ldb, beta, C + i * ldc + j, ldc);
                    }
                }
            }
        }

        void MacroKernel(size_t M, size_t N, size_t K, T alpha, const T * A, size_t lda, const T * B, size_t ldb, T beta, T * C, size_t ldc)
        {
            size_t N4 = Simd::AlignLo(N, 4);
            size_t i = 0;
            if (_k6x4)
            {
                size_t M6 = Simd::AlignLoAny(M, 6);
                for (; i < M6; i += 6)
                {
                    const float * pA = A + i * lda;
                    float * pC = C + i * ldc;
                    size_t j = 0;
                    for (; j < N4; j += 4)
                        _k6x4(K, alpha, pA, lda, B + j * ldb, ldb, pC + j, ldc);
                    for (; j < N; ++j)
                        _k6x1(K, alpha, pA, lda, B + j * ldb, ldb, pC + j, ldc);
                }
            }
            if (_k3x4)
            {
                size_t M3 = Simd::AlignLoAny(M, 3);
                for (; i < M3; i += 3)
                {
                    const float * pA = A + i * lda;
                    float * pC = C + i * ldc;
                    size_t j = 0;
                    for (; j < N4; j += 4)
                        _k3x4(K, alpha, pA, lda, B + j * ldb, ldb, pC + j, ldc);
                    for (; j < N; ++j)
                        _k3x1(K, alpha, pA, lda, B + j * ldb, ldb, pC + j, ldc);
                }
                for (; i < M - 1; i += 2)
                {
                    const float * pA = A + i * lda;
                    float * pC = C + i * ldc;
                    size_t j = 0;
                    for (; j < N4; j += 4)
                        _k2x4(K, alpha, pA, lda, B + j * ldb, ldb, pC + j, ldc);
                    for (; j < N; ++j)
                        _k2x1(K, alpha, pA, lda, B + j * ldb, ldb, pC + j, ldc);
                }            
            }
            for (; i < M; i++)
            {
                const float * pA = A + i * lda;
                float * pC = C + i * ldc;
                size_t j = 0;
                for (; j < N4; j += 4)
                    _k1x4(K, alpha, pA, lda, B + j * ldb, ldb, pC + j, ldc);
                for (; j < N; ++j)
                    _k1x1(K, alpha, pA, lda, B + j * ldb, ldb, pC + j, ldc);
            }
        }

        size_t _M, _N, _K, _microM, _microN, _macroM, _macroN, _macroK, _F, _threadNumber;
        ScaleC _scaleC;
        Kernel _k1x1, _k1x4, _k2x1, _k2x4, _k3x1, _k3x4, _k6x1, _k6x4;
    };

    template <class T, size_t F, class TM> class GemmNNcb
    {
    public:
        typedef void(*Main)(size_t K, T alpha, const T * A, size_t lda, const T * B, size_t ldb, size_t sb, T * C, size_t ldc, TM tail);
        typedef void(*Tail)(size_t M, size_t K, T alpha, const T * A, size_t lda, const T * B, size_t ldb, size_t sb, T * C, size_t ldc, TM tail);
        typedef void(*PackA)(const T * A, size_t lda, size_t M, size_t K, size_t microM, T * pA);
        typedef void(*PackB)(const T * B, size_t ldb, size_t K, size_t N, size_t microN, T * pB);
        typedef void(*ScaleC)(size_t M, size_t N, T beta, T * C, size_t ldc);
        typedef TM(*TailMask)(ptrdiff_t tail);

        GemmNNcb(size_t M, size_t N, size_t K, size_t microM, size_t microN, size_t L1, size_t L2, size_t L3, 
            Main kernelMM, Main kernelMT, Tail kernelTM, Tail kernelTT, PackA packA, PackB packB, ScaleC scaleC, TailMask tailMask, bool compatible = false)
            : _0(0)
            , _1(1)
        {
            L2 = Simd::RestrictRange(size_t(::sqrt(double(L1 * L3))), L2/4, L2);
            _compatible = compatible;
            _M = M;
            _N = N;
            _K = K;
            _microM = microM;
            _microN = microN;
            _kernelMM = kernelMM;
            _kernelMT = kernelMT;
            _kernelTM = kernelTM;
            _kernelTT = kernelTT;
            _scaleC = scaleC;
            _packB = packB;
            _packA = packA;
            _macroK = Simd::Min(L1 / sizeof(T) / _microN, _K);
            _macroM = Simd::RestrictRange(AlignLoAny(L2 / sizeof(T) / _macroK, _microM), 
                _microM, AlignHiAny(_M, _microM));
            _macroN = Simd::RestrictRange(AlignLoAny(L3 / sizeof(T) / _macroK, _microN), 
                compatible ? F : _microN, AlignHiAny(_N, _compatible ? F : _microN));
            if (_packA && NeedPackA())
                _pA.Resize(_macroM * _macroK);
            size_t NF = AlignLo(_N, F);
            if (tailMask)
            {
                _main = TM(-1);
                _tail = NF == _N ? TM(-1) : tailMask(_N - NF);
            }
            else
            {
                _main = TM(F);
                _tail = NF == _N ? TM(F) : TM(_N - NF);
            }        
        }

        SIMD_INLINE size_t BufferSize() const 
        {
            return AlignHiAny(_N, _compatible ? F : _microN)*_K;
        }

        void ReorderB(const T * B, size_t ldb, T * pB)
        {
            if (_compatible)
            {
                _packB(B, ldb, _K, _N, F, pB);
            }
            else
            {
                for (size_t j = 0; j < _N; j += _macroN)
                {
                    size_t macroN = Simd::Min(_N, j + _macroN) - j;
                    for (size_t k = 0; k < _K; k += _macroK)
                    {
                        size_t macroK = Simd::Min(_K, k + _macroK) - k;
                        _packB(B + k * ldb + j, ldb, macroK, macroN, _microN, pB);
                        pB += AlignHiAny(macroN, _microN)*macroK;
                    }
                }
            }
        }

        SIMD_INLINE void Run(const T * A, size_t lda, const T * pB, T * C, size_t ldc)
        {
            Run(_M, A, lda, pB, C, ldc);
        }

        void Run(size_t M, const T * A, size_t lda, const T * pB, T * C, size_t ldc)
        {
            assert(M <= _M);
            for (size_t j = 0; j < _N; j += _macroN)
            {
                size_t macroN = Simd::Min(_N, j + _macroN) - j;
                for (size_t k = 0; k < _K; k += _macroK)
                {
                    size_t macroK = Simd::Min(_K, k + _macroK) - k;
                    for (size_t i = 0; i < M; i += _macroM)
                    {
                        size_t macroM = Simd::Min(M, i + _macroM) - i;
                        if (k == 0)
                            _scaleC(macroM, macroN, _0, C + i * ldc + j, ldc);
                        if (_compatible)
                            MacroKernelCompatible(macroM, macroN, macroK, A + i * lda + k, lda, pB + j * _K + k * F, C + i * ldc + j, ldc);
                        else
                            MacroKernelSpecific(macroM, macroN, macroK, A + i * lda + k, lda, pB, C + i * ldc + j, ldc);
                    }
                    if(!_compatible)
                        pB += AlignHiAny(macroN, _microN)*macroK;
                }
            }
        }

    private:

        void MacroKernelSpecific(size_t M, size_t N, size_t K, const T * A, size_t lda, const T * pB, T * C, size_t ldc)
        {
            size_t klda = lda;
            if (_pA.data)
            {
                _packA(A, lda, M, K, _microM, _pA.data);
                A = _pA.data;
                lda = K;
                klda = 1;
            }
            size_t MA = AlignLoAny(M, _microM);
            size_t NA = AlignLoAny(N, _microN);
            size_t j = 0;
            for (; j < NA; j += _microN)
            {
                size_t i = 0;
                for (; i < MA; i += _microM)
                    _kernelMM(K, _1, A + i * lda, klda, pB, F, _microN, C + i * ldc + j, ldc, _main);
                if (i < M)
                    _kernelTM(M - i, K, _1, A + i * lda, klda, pB, F, _microN, C + i * ldc + j, ldc, _main);
                pB += _microN * K;
            }
            if (j < N)
            {
                size_t i = 0;
                for (; i < MA; i += _microM)
                    _kernelMT(K, _1, A + i * lda, klda, pB, F, _microN, C + i * ldc + j, ldc, _tail);
                if (i < M)
                    _kernelTT(M - i, K, _1, A + i * lda, klda, pB, F, _microN, C + i * ldc + j, ldc, _tail);
            }
        }

        void MacroKernelCompatible(size_t M, size_t N, size_t K, const T * A, size_t lda, const T * pB, T * C, size_t ldc)
        {
            size_t klda = lda, plda = lda;
            T * pA = (T*)A;
            if (_pA.data)
            {
                pA = _pA.data;
                plda = K;
                klda = 1;
            }
            size_t MA = AlignLoAny(M, _microM);
            size_t NA = AlignLoAny(N, _microN);
            size_t j = 0;
            for (; j < NA; j += _microN)
            {
                size_t i = 0;
                for (; i < MA; i += _microM)
                {
                    if (_pA.data && j == 0)
                        _packA(A + i * lda, lda, _microM, K, _microM, pA + i * plda);
                    _kernelMM(K, _1, pA + i * plda, klda, pB, F * _K, F, C + i * ldc + j, ldc, _main);
                }
                if (i < M)
                {
                    if (_pA.data && j == 0)
                        _packA(A + i * lda, lda, M - i, K, _microM, pA + i * plda);
                    _kernelTM(M - i, K, _1, pA + i * plda, klda, pB, F * _K, F, C + i * ldc + j, ldc, _main);
                }
                pB += _microN * _K;
            }
            if (j < N)
            {
                size_t i = 0;
                for (; i < MA; i += _microM)
                {
                    if (_pA.data && j == 0)
                        _packA(A + i * lda, lda, _microM, K, _microM, pA + i * plda);
                    _kernelMT(K, _1, pA + i * plda, klda, pB, F * _K, F, C + i * ldc + j, ldc, _tail);
                }
                if (i < M)
                {
                    if (_pA.data && j == 0)
                        _packA(A + i * lda, lda, M - i, K, _microM, pA + i * plda);
                    _kernelTT(M - i, K, _1, pA + i * plda, klda, pB, F * _K, F, C + i * ldc + j, ldc, _tail);
                }
            }
        }

        bool NeedPackA() const
        {
            if (_K >= 256 && _M > 256 && _N > _microN * 4)
                return true;
            if (_M * 3 < _N && _N >= 512 && _K >= 128 && _M > 16 && _microN >= 32)
                return true;
            return false;
        }

        typedef Simd::Array<T> Array;

        size_t _M, _N, _K, _microM, _microN, _macroM, _macroN, _macroK;
        TM _main, _tail;
        Main _kernelMM, _kernelMT;
        Tail _kernelTM, _kernelTT;
        ScaleC _scaleC;
        PackB _packB;
        PackA _packA;
        Array _pA;
        T _0, _1;
        bool _compatible;
    };

    enum GemmKernelType
    {
        GemmKernelAny = 0,
        GemmKernelF1,
        GemmKernelF2,
        GemmKernelF3,
        GemmKernelF4,
    };

#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        void GemmKernel4x12nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernel4x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernel4x4nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);

        void GemmKernel6x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernel6x4nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);

        void GemmKernelMx12nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernelMx8nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernelMx4nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);

        void GemmPackA(const float * A, size_t lda, size_t M, size_t K, size_t microM, float * pA);
        void GemmPackB(const float * B, size_t ldb, size_t K, size_t N, size_t microN, float * pB);
        void GemmScaleC(size_t M, size_t N, float beta, float * C, size_t ldc);

        size_t Gemm32fNNcbBufferSize(size_t M, size_t N, size_t K, GemmKernelType type, bool compatibility);
        void Gemm32fNNcbReorderB(size_t M, size_t N, size_t K, const float * B, float * pB, GemmKernelType type, bool compatibility);
        void Gemm32fNNcbRun(size_t M, size_t N, size_t K, const float * A, const float * pB, float * C, GemmKernelType type, bool compatibility);
    }
#endif//SIMD_SSE2_ENABLE

#ifdef SIMD_AVX_ENABLE
    namespace Avx
    {
        void GemmKernel4x24nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernel4x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernel4x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);

        void GemmKernel6x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernel6x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);

        void GemmKernelMx24nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernelMx16nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernelMx8nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);

        void GemmPackA(const float * A, size_t lda, size_t M, size_t K, size_t microM, float * pA);
        void GemmPackB(const float * B, size_t ldb, size_t K, size_t N, size_t microN, float * pB);
        void GemmScaleC(size_t M, size_t N, float beta, float * C, size_t ldc);

        size_t Gemm32fNNcbBufferSize(size_t M, size_t N, size_t K, GemmKernelType type, bool compatibility);
        void Gemm32fNNcbReorderB(size_t M, size_t N, size_t K, const float * B, float * pB, GemmKernelType type, bool compatibility);
        void Gemm32fNNcbRun(size_t M, size_t N, size_t K, const float * A, const float * pB, float * C, GemmKernelType type, bool compatibility);
    }
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        void GemmKernel4x24nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernel4x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernel4x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);

        void GemmKernel6x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernel6x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);

        void GemmKernelMx24nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernelMx16nn(size_t M,size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernelMx8nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);

        size_t Gemm32fNNcbBufferSize(size_t M, size_t N, size_t K, GemmKernelType type, bool compatibility);
        void Gemm32fNNcbReorderB(size_t M, size_t N, size_t K, const float * B, float * pB, GemmKernelType type, bool compatibility);
        void Gemm32fNNcbRun(size_t M, size_t N, size_t K, const float * A, const float * pB, float * C, GemmKernelType type, bool compatibility);
    }
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX512F_ENABLE
    namespace Avx512f
    {
        void GemmKernel4x48nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask);
        void GemmKernel4x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask);
        void GemmKernel4x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask);

        void GemmKernel6x64nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, __mmask16 mask);
        void GemmKernel6x48nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, __mmask16 mask);
        void GemmKernel6x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask);
        void GemmKernel6x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask);

        void GemmKernel8x48nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask);
        void GemmKernel8x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask);
        void GemmKernel8x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask);

        void GemmKernel9x48nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask);
        void GemmKernel9x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask);
        void GemmKernel9x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask);

        void GemmKernel12x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask);
        void GemmKernel12x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask);

        void GemmKernel14x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask);
        void GemmKernel14x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask);

        void GemmPackA(const float * A, size_t lda, size_t M, size_t K, size_t microM, float * pA);
        void GemmPackB(const float * B, size_t ldb, size_t K, size_t N, size_t microN, float * pB);
        void GemmScaleC(size_t M, size_t N, float beta, float * C, size_t ldc);

        size_t Gemm32fNNcbBufferSize(size_t M, size_t N, size_t K, GemmKernelType type, bool compatibility);
        void Gemm32fNNcbReorderB(size_t M, size_t N, size_t K, const float * B, float * pB, GemmKernelType type, bool compatibility);
        void Gemm32fNNcbRun(size_t M, size_t N, size_t K, const float * A, const float * pB, float * C, GemmKernelType type, bool compatibility);
    }
#endif//SIMD_AVX512F_ENABLE

#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        void GemmKernel4x12nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernel4x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernel4x4nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);

        void GemmKernel6x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernel6x4nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);

        void GemmKernelMx12nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernelMx8nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);
        void GemmKernelMx4nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail);

        void GemmPackA(const float * A, size_t lda, size_t M, size_t K, size_t microM, float * pA);
        void GemmPackB(const float * B, size_t ldb, size_t K, size_t N, size_t microN, float * pB);
        void GemmScaleC(size_t M, size_t N, float beta, float * C, size_t ldc);

        size_t Gemm32fNNcbBufferSize(size_t M, size_t N, size_t K, GemmKernelType type, bool compatibility);
        void Gemm32fNNcbReorderB(size_t M, size_t N, size_t K, const float * B, float * pB, GemmKernelType type, bool compatibility);
        void Gemm32fNNcbRun(size_t M, size_t N, size_t K, const float * A, const float * pB, float * C, GemmKernelType type, bool compatibility);
    }
#endif//SIMD_NEON_ENABLE
}

#endif//__SimdGemm_h__
