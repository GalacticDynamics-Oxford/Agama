/** \file    particles_io.h
    \brief   Input/output of Nbody snapshots in various formats
    \author  EV
    \date    2010-2015

    The base class, particles::BaseIOSnapshot, is used as the common interface 
    for reading and writing Nbody snapshots to disk. 
    The snapshots are provided by particles::ParticleArray.
    Derived classes implement the data storage in various formats.
    Helper routines create an instance of the class corresponding to a given 
    format string or to the actual file format.
*/

#pragma once
#include "particles_base.h"
#include "units.h"
#include "smart.h"
#include <string>

namespace particles {

/** The abstract class implementing reading and writing snapshots.
    Derived classes take the filename as the argument of the constructor,
    and an instance of unit converter for transforming between 
    the "external" units of the file and the "internal" units of the library 
    (a trivial conversion is also possible).
*/
class BaseIOSnapshot {
public:
    virtual ~BaseIOSnapshot() {};
    /** read a snapshot from the file;
        \returns a new instance of ParticleArray class.
        \throws  std::runtime_error in case of error (e.g., file doesn't exist).
    */
    virtual ParticleArrayCar readSnapshot() const=0;
    /** write a snapshot to the file;  
        \param[in] particles is an instance of ParticleArray class to be stored;
        \throws  std::runtime_error in case of error (e.g., file is not writable)
    */
    virtual void writeSnapshot(const ParticleArrayCar& particles) const=0;
};

/// Text file with three coordinates, possibly three velocities and mass, space or tab-separated.
class IOSnapshotText: public BaseIOSnapshot {
public:
    IOSnapshotText(const std::string &_fileName, const units::ExternalUnits& unitConverter): 
        fileName(_fileName), conv(unitConverter) {};
    virtual ParticleArrayCar readSnapshot() const;
    virtual void writeSnapshot(const ParticleArrayCar& particles) const;
private:
    const std::string fileName;
    const units::ExternalUnits conv;
};

/// NEMO snapshot format.
/// reading is supported only if compiled with UNSIO library; 
/// writing is implemented by builtin routines.
class IOSnapshotNemo: public BaseIOSnapshot {
public:
    /// create the class to read or write to the file; 
    /// if writing is intended, may provide a header string and timestamp
    /// and choose whether to append to file if it already exists
    IOSnapshotNemo(const std::string &_fileName, const units::ExternalUnits& unitConverter,
        const std::string &_header="", double _time=0, bool _append=false) :
        fileName(_fileName), conv(unitConverter), header(_header), time(_time), append(_append) {};
    virtual ParticleArrayCar readSnapshot() const;
    virtual void writeSnapshot(const ParticleArrayCar& particles) const;
private:
    const std::string fileName;
    const units::ExternalUnits conv;
    const std::string header;    ///< header string which will be written to the file
    const double time;           ///< timestamp of the snapshot to write
    const bool append;           ///< whether to append to the end of file or overwrite it
};

/// GADGET snapshot format; needs UNSIO library.
class IOSnapshotGadget: public BaseIOSnapshot {
public:
    IOSnapshotGadget(const std::string &_fileName, const units::ExternalUnits& unitConverter):
        fileName(_fileName), conv(unitConverter) {};
    virtual ParticleArrayCar readSnapshot() const;
    virtual void writeSnapshot(const ParticleArrayCar& particles) const;
private:
    const std::string fileName;
    const units::ExternalUnits conv;
};

/// smart pointer to snapshot interface
typedef unique_ptr<BaseIOSnapshot> PtrIOSnapshot;

/// creates an instance of appropriate snapshot reader, according to the file format 
/// determined by reading first few bytes, or throw a std::runtime_error if a file doesn't exist
PtrIOSnapshot createIOSnapshotRead (const std::string &fileName, 
    const units::ExternalUnits& unitConverter = units::ExternalUnits());

/// creates an instance of snapshot writer for a given format name,
/// or throw a std::runtime_error if the format name string is incorrect or file name is empty
PtrIOSnapshot createIOSnapshotWrite(const std::string &fileName, 
    const std::string &fileFormat="Text",
    const units::ExternalUnits& unitConverter = units::ExternalUnits(),
    const std::string& header="", const double time=0, const bool append=false);

/** convenience function for reading an N-body snapshot in arbitrary format.
    \param[in]  fileName  is the file to read, its format is determined automatically;
    \param[in]  unitConverter  is the instance of unit conversion object (may be a trivial one);
    \returns    a new instance of ParticleArray containing the particles read from the file.
*/
inline ParticleArrayCar readSnapshot(const std::string& fileName, 
    const units::ExternalUnits& unitConverter = units::ExternalUnits())
{
    return createIOSnapshotRead(fileName, unitConverter)->readSnapshot();
}

/** convenience function for writing an N-body snapshot in the given format.
    \param[in]  fileName is the file to write;
    \param[in]  particles  is the array of particles (positions,velocities and masses) to write;
    \param[in]  fileFormat  is the output format (optional; default is 'Text');
    \param[in]  unitConverter is the instance of unit conversion (may be a trivial one).
*/
inline void writeSnapshot(const std::string& fileName, 
    const ParticleArrayCar& particles,
    const std::string &fileFormat="Text",
    const units::ExternalUnits& unitConverter = units::ExternalUnits())
{
    createIOSnapshotWrite(fileName, fileFormat, unitConverter)->writeSnapshot(particles);
}

/** convenience function for writing an N-body snapshot that contains only positions.
    The automatic conversion pipeline does not apply in this case, 
    so zero velocities are assigned manually.
    \tparam   CoordT is the coordinate system name (positions given in this system);
    the rest of parameters are the same as in `writeSnapshot()`.
*/
template<typename CoordT>
inline void writeSnapshot(const std::string& fileName, 
    const ParticleArray<coord::PosT<CoordT> >& particles,
    const std::string &fileFormat="Text",
    const units::ExternalUnits& unitConverter = units::ExternalUnits())
{
    ParticleArrayCar tmpParticles;
    tmpParticles.data.reserve(particles.size());
    for(unsigned int i=0; i<particles.size(); i++)
        tmpParticles.data.push_back(std::make_pair(coord::PosVelCar(toPosCar(particles[i].first),
            coord::VelCar(0,0,0)), particles[i].second));  // convert the position and assign zero velocity
    createIOSnapshotWrite(fileName, fileFormat, unitConverter)->writeSnapshot(tmpParticles);
}

/* ------ Correspondence between file format names and types ------- */
#if 0
/// list of all available IO snapshot formats, initialized at module start 
/// according to the file format supported at compile time
extern std::vector< std::string > formatsIOSnapshot;
#endif

}  // namespace
