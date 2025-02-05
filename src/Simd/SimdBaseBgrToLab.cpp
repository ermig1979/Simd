/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#include "Simd/SimdBgrToLab.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdLog.h"

namespace Simd
{
    namespace Base
    {
#define SIMD_BGR_TO_LAB_OPENCV_COMPATIBILITY

        namespace LabV1
        {
            inline double FromRaw(uint64_t raw)
            {
                return *((double*)&raw);
            }

            SIMD_INLINE float mullAdd(float a, float b, float c)
            {
                return a * b + c;
            }

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))
            const int xyz_shift = 12;

            //const double D65[3] = { 0.950456, 1., 1.088754 };
            const double D65[3] = { FromRaw(0x3fee6a22b3892ee8),
                                 1.0 ,
                                 FromRaw(0x3ff16b8950763a19) };

            //static const double sRGB2XYZ_D65[] =
            //{
            //    0.412453, 0.357580, 0.180423,
            //    0.212671, 0.715160, 0.072169,
            //    0.019334, 0.119193, 0.950227
            //};

            const double sRGB2XYZ_D65[] =
            {
                FromRaw(0x3fda65a14488c60d),
                FromRaw(0x3fd6e297396d0918),
                FromRaw(0x3fc71819d2391d58),
                FromRaw(0x3fcb38cda6e75ff6),
                FromRaw(0x3fe6e297396d0918),
                FromRaw(0x3fb279aae6c8f755),
                FromRaw(0x3f93cc4ac6cdaf4b),
                FromRaw(0x3fbe836eb4e98138),
                FromRaw(0x3fee68427418d691)
            };

            enum { LAB_CBRT_TAB_SIZE = 1024, GAMMA_TAB_SIZE = 1024 };

            static const float GammaTabScale((int)GAMMA_TAB_SIZE);

            static uint16_t sRGBGammaTab_b[256];

#undef lab_shift
#define lab_shift xyz_shift
#define gamma_shift 3
#define lab_shift2 (lab_shift + gamma_shift)
#define LAB_CBRT_TAB_SIZE_B (256*3/2*(1<<gamma_shift))
            static uint16_t LabCbrtTab_b[LAB_CBRT_TAB_SIZE_B];

//all constants should be presented through integers to keep bit-exactness
            static const double gammaThreshold = double(809) / double(20000);    //  0.04045
            static const double gammaInvThreshold = double(7827) / double(2500000); //  0.0031308
            static const double gammaLowScale = double(323) / double(25);       // 12.92
            static const double gammaPower = double(12) / double(5);         //  2.4
            static const double gammaXshift = double(11) / double(200);       //  0.055

            static const float lthresh = float(216) / float(24389); // 0.008856f = (6/29)^3
            static const float lscale = float(841) / float(108); // 7.787f = (29/3)^3/(29*4)
            static const float lbias = float(16) / float(116);

            static inline float applyGamma(float x)
            {
                //return x <= 0.04045f ? x*(1.f/12.92f) : (float)std::pow((double)(x + 0.055)*(1./1.055), 2.4);

                double xd = x;
                return (xd <= gammaThreshold ?
                    xd / gammaLowScale :
                    pow((xd + gammaXshift) / (1.0 + gammaXshift), gammaPower));
            }

            static void initLabTabs()
            {
                static bool initialized = false;
                if (!initialized)
                {
                    float scale = 1.0f / float(GammaTabScale);
                    static const float intScale(255 * (1 << gamma_shift));
                    for (int i = 0; i < 256; i++)
                    {
                        float x = float(i) / 255.0f;
                        sRGBGammaTab_b[i] = (uint16_t)(Round(intScale * applyGamma(x)));
                    }

                    static const float cbTabScale(1.0 / (255.0 * (1 << gamma_shift)));
                    static const float lshift2(1 << lab_shift2);
                    for (int i = 0; i < LAB_CBRT_TAB_SIZE_B; i++)
                    {
                        float x = cbTabScale * float(i);
                        LabCbrtTab_b[i] = (uint16_t)(Round(lshift2 * (x < lthresh ? mullAdd(x,  lscale,  lbias) : cbrt(x))));
                    }
#if defined(SIMD_BGR_TO_LAB_OPENCV_COMPATIBILITY)
                    if(LabCbrtTab_b[324] == 17746)
                        LabCbrtTab_b[324] = 17745;
                    if (LabCbrtTab_b[49] == 9455)
                        LabCbrtTab_b[49] = 9454;
#endif
                    initialized = true;
                }
            }

            struct RGB2Lab_b
            {
                typedef uint8_t channel_type;

                RGB2Lab_b(int _srccn, int blueIdx)
                    : srccn(_srccn)
                {
                    initLabTabs();

                    double whitePt[3];
                    for (int i = 0; i < 3; i++)
                        whitePt[i] = D65[i];

                    const double lshift(1 << lab_shift);
                    for (int i = 0; i < 3; i++)
                    {
                        double c[3];
                        for (int j = 0; j < 3; j++)
                            c[j] = sRGB2XYZ_D65[i * 3 + j];
                        coeffs[i * 3 + (blueIdx ^ 2)] = Round(lshift * c[0] / whitePt[i]);
                        coeffs[i * 3 + 1] = Round(lshift * c[1] / whitePt[i]);
                        coeffs[i * 3 + blueIdx] = Round(lshift * c[2] / whitePt[i]);

                        assert(coeffs[i * 3] >= 0 && coeffs[i * 3 + 1] >= 0 && coeffs[i * 3 + 2] >= 0 &&
                            coeffs[i * 3] + coeffs[i * 3 + 1] + coeffs[i * 3 + 2] < 2 * (1 << lab_shift));
                    }
                }

                void operator()(const uint8_t* src, uint8_t* dst, int n) const
                {
                    const int Lscale = (116 * 255 + 50) / 100;
                    const int Lshift = -((16 * 255 * (1 << lab_shift2) + 50) / 100);
                    const uint16_t* tab = sRGBGammaTab_b;
                    int i, scn = srccn;
                    int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
                        C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
                        C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

                    for (int i = 0; i < n; i++, src += scn, dst += 3)
                    {
                        int R = tab[src[0]], G = tab[src[1]], B = tab[src[2]];
                        int fX = LabCbrtTab_b[CV_DESCALE(R * C0 + G * C1 + B * C2, lab_shift)];
                        int fY = LabCbrtTab_b[CV_DESCALE(R * C3 + G * C4 + B * C5, lab_shift)];
                        int fZ = LabCbrtTab_b[CV_DESCALE(R * C6 + G * C7 + B * C8, lab_shift)];

                        int L = CV_DESCALE(Lscale * fY + Lshift, lab_shift2);
                        int a = CV_DESCALE(500 * (fX - fY) + 128 * (1 << lab_shift2), lab_shift2);
                        int b = CV_DESCALE(200 * (fY - fZ) + 128 * (1 << lab_shift2), lab_shift2);

#if defined(_WIN32) && 0
                        if (L == 45 && a == 165)
                        {
                            std::cout << i << " old Lab = {" << L << ", " << a << ", " << b << "}";

                            int R = tab[src[0]], G = tab[src[1]], B = tab[src[2]];
                            int fX = LabCbrtTab_b[CV_DESCALE(R * C0 + G * C1 + B * C2, lab_shift)];
                            int fY = LabCbrtTab_b[CV_DESCALE(R * C3 + G * C4 + B * C5, lab_shift)] - 1;
                            int fZ = LabCbrtTab_b[CV_DESCALE(R * C6 + G * C7 + B * C8, lab_shift)];

                            int L = CV_DESCALE(Lscale * fY + Lshift, lab_shift2);
                            int a = CV_DESCALE(500 * (fX - fY) + 128 * (1 << lab_shift2), lab_shift2);
                            int b = CV_DESCALE(200 * (fY - fZ) + 128 * (1 << lab_shift2), lab_shift2);

                            std::cout << " new Lab = {" << L << ", " << a << ", " << b << "}";
                            std::cout << " index " << CV_DESCALE(R * C3 + G * C4 + B * C5, lab_shift);
                            std::cout << " value " << LabCbrtTab_b[CV_DESCALE(R * C3 + G * C4 + B * C5, lab_shift)];
                            std::cout << std::endl;
                            exit(0);
                        }
#endif
                        dst[0] = Base::RestrictRange(L);
                        dst[1] = Base::RestrictRange(a);
                        dst[2] = Base::RestrictRange(b);
                    }
                }

                int srccn;
                int coeffs[9];
            };
        }

        void BgrToLab(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* lab, size_t labStride)
        {
#if 1
            LabV1::RGB2Lab_b _lab(3, 0);
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t* pBgr = bgr + row * bgrStride;
                uint8_t* pLab = lab + row * labStride;
                _lab(pBgr, pLab, width);
            }

#else
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t* pBgr = bgr + row * bgrStride;
                uint8_t* pLab = lab + row * labStride;
                for (const uint8_t* pBgrEnd = pBgr + width * 3; pBgr < pBgrEnd; pBgr += 3, pLab += 3)
                    BgrToLab(pBgr[0], pBgr[1], pBgr[2], pLab);
            }
#endif
        }
    }
}

