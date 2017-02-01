/** \file   raga.cpp
    \brief  The Monte Carlo stellar-dynamical code Raga
    \author Eugene Vasiliev
    \date   2013-2017
*/
#include "raga_core.h"
#include "utils_config.h"
#include <iostream>

int main(int argc, char *argv[])
{
    if(argc<=1) {
        std::cout << "Raga v2.0 build " __DATE__ "\n"
        "Usage: raga file.ini\n"
        "See the description of the method and the INI parameters in readme_raga.pdf\n";
        return 0;
    }
    raga::RagaCore core(utils::ConfigFile(argv[1]).findSection("Raga"));
    core.run();
}
