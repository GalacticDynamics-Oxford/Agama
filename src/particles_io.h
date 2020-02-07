/** \file    particles_io.h
    \brief   Input/output of Nbody snapshots in various formats
    \author  EV
    \date    2010-2020

    The routines 'readSnapshot' and 'writeSnapshot' provide the top-level interface
    to a variety of file formats (Text, NEMO, Gadget).
*/

#pragma once
#include "particles_base.h"
#include "units.h"
#include <string>

namespace particles {

/** Read an N-body snapshot in arbitrary format.
    \param[in]  fileName  is the file to read, its format is determined automatically;
    \param[in]  unitConverter  is the instance of unit conversion object (may be a trivial one);
    \returns    a new instance of ParticleArray containing the particles read from the file.
*/
ParticleArrayAux readSnapshot(
    const std::string& fileName,
    const units::ExternalUnits& unitConverter = units::ExternalUnits());

/** Write an N-body snapshot in the given format.
    \param[in]  fileName is the file to write;
    \param[in]  particles  is the array of particles to write;
    \param[in]  fileFormat  is the string specifying the output file format
    (only the first letter matters, case-insensitive: 't' - Text, 'n' - Nemo, 'g' - Gadget);
    \param[in]  unitConverter is the optional instance of unit conversion (may be a trivial one);
    \param[in]  header  is the optional header string  (not for all formats);
    \param[in]  time  is the timestamp of the snapshot (not for all formats);
    \param[in]  append  is the flag specifying whether to append to an existing file or
    overwrite the file (only for Nemo format, other formats always overwrite).
    \throw  std::runtime_error if the format name string is incorrect, or file name is empty,
    or if the file could not be written.
    \tparam     particle type: could be position in some coordinate system (e.g. PosCar),
    or position+velocity (e.g. PosVelCyl), or particle with additional attributes (ParticleAux)
*/
template<typename ParticleT>
void writeSnapshot(
    const std::string& fileName,
    const ParticleArray<ParticleT>& particles,
    const std::string &fileFormat="Text",
    const units::ExternalUnits& unitConverter = units::ExternalUnits(),
    const std::string& header="",
    const double time=NAN,
    const bool append=false);

}  // namespace
