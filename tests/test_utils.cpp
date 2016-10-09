/** \file    test_utils.cpp
    \date    Oct 2016
    \author  Eugene Vasiliev

    Test the number-to-string conversion routine (should add more tests for the utils here)
*/
#include "utils.h"
#include "math_core.h"
#include <iostream>
#include <iomanip>
#include <cmath>

int main()
{
    bool ok=true;
    const int NUM = 50;
    double values[NUM] = {0, NAN, INFINITY, -INFINITY, 1., 
        9.4999999999999, 9.95, 0.95000000000001, 9.5e8, 9.96e9, 9.5e-5};
    for(int i=11; i<NUM; i++) {
        double val = math::random();
        int type   = (int)(math::random()*8);
        if(type & 1)
            val = pow(10, 10*val);
        else
            val = pow(10, 150*val);
        if(type & 2)
            val = 1/val;
        if(type & 4)
            val = -val;
        values[i] = val;
    }
    for(int i=0; i<NUM; i++) {
        std::cout << std::setw(24) << std::setprecision(16) << values[i];
        for(unsigned int w=1; w<12; w++) {
            std::string result = utils::pp(values[i], w);
            double val = utils::toDouble(result);
            bool fail  = result.size() != w;
            if(val)  // conversion ok
                fail |= (val>0 ^ values[i]>0) ||  // not the same sign
                val / values[i] < 0.5 || val / values[i] > 2;  // differs by more than a factor of 2
            std::cout << ' ';
            if(fail) std::cout << "\033[1;31m";
            std::cout << result;
            if(fail) std::cout << "\033[0m";
            ok &= !fail;
        }
        std::cout << '\n';
    }
    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}