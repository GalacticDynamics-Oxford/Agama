/** \file   raga.cpp
    \brief  The Monte Carlo stellar-dynamical code Raga
    \author Eugene Vasiliev
    \date   2013-2020
*/
#include "raga_core.h"
#include "utils_config.h"
#include <iostream>
#include <algorithm>

int main(int argc, char *argv[])
{
    if(argc<=1) {
        std::cout << "Raga v3.0 build " __DATE__ "\n"
        "Usage: raga file.ini\n"
        "See the description of the method and the INI parameters in readme_raga.pdf\n";
        return 0;
    }
    raga::RagaCore core;
    core.init(utils::ConfigFile(argv[1]).findSection("Raga"));
    if(core.paramsRaga.timeEnd <= core.paramsRaga.timeCurr || core.paramsRaga.episodeLength <= 0) {
        std::cout << "Total simulation time and episode length should be positive "
            "([Raga]/timeTotal, [Raga]/episodeLength)\n";
        return 1;
    }
    while(core.paramsRaga.timeCurr < core.paramsRaga.timeEnd)
        core.doEpisode(std::min<double>(
            core.paramsRaga.episodeLength, core.paramsRaga.timeEnd-core.paramsRaga.timeCurr) );
}
