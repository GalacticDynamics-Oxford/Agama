#include "particles_io.h"
#ifdef HAVE_UNSIO
#include <uns.h>
#endif
#include <fstream>
#include <cassert>
#include <stdexcept>
#include "utils.h"

namespace particles {

namespace{  // internal

//----- text file format -----//

template<typename ParticleT>
const char* formatHeader();

template<typename ParticleT>
std::string formatParticle(
    const typename ParticleArray<ParticleT>::ElemType& point, const units::ExternalUnits& unitConverter);

template<> inline const char* formatHeader<coord::PosCar>() {
    return "#x\ty\tz\tmass\n";
}

template<> inline const char* formatHeader<coord::PosVelCar>() {
    return "#x\ty\tz\tvx\tvy\tvz\tmass\n";
}

template<> inline const char* formatHeader<ParticleAux>() {
    return "#x\ty\tz\tvx\tvy\tvz\tparticleMass\tstellarMass\tstellarRadius\n";
}

template<> inline std::string formatParticle<coord::PosCar>(
    const ParticleArray<coord::PosCar>::ElemType& point, const units::ExternalUnits& conv)
{
    return
    utils::toString(point.first.x  / conv.lengthUnit, 8) + '\t' +
    utils::toString(point.first.y  / conv.lengthUnit, 8) + '\t' +
    utils::toString(point.first.z  / conv.lengthUnit, 8) + '\t' +
    utils::toString(point.second   / conv.massUnit  , 8) + '\n';
}

template<> inline std::string formatParticle<coord::PosVelCar>(
    const ParticleArray<coord::PosVelCar>::ElemType& point, const units::ExternalUnits& conv)
{
    return
    utils::toString(point.first.x  / conv.lengthUnit  , 8) + '\t' +
    utils::toString(point.first.y  / conv.lengthUnit  , 8) + '\t' +
    utils::toString(point.first.z  / conv.lengthUnit  , 8) + '\t' +
    utils::toString(point.first.vx / conv.velocityUnit, 8) + '\t' +
    utils::toString(point.first.vy / conv.velocityUnit, 8) + '\t' +
    utils::toString(point.first.vz / conv.velocityUnit, 8) + '\t' +
    utils::toString(point.second   / conv.massUnit    , 8) + '\n';
}

template<> inline std::string formatParticle<ParticleAux>(
    const ParticleArray<ParticleAux>::ElemType& point, const units::ExternalUnits& conv)
{
    return
    utils::toString(point.first.x  / conv.lengthUnit  , 8) + '\t' +
    utils::toString(point.first.y  / conv.lengthUnit  , 8) + '\t' +
    utils::toString(point.first.z  / conv.lengthUnit  , 8) + '\t' +
    utils::toString(point.first.vx / conv.velocityUnit, 8) + '\t' +
    utils::toString(point.first.vy / conv.velocityUnit, 8) + '\t' +
    utils::toString(point.first.vz / conv.velocityUnit, 8) + '\t' +
    utils::toString(point.second   / conv.massUnit    , 8) + '\t' +
    utils::toString(point.first.stellarMass   / conv.massUnit  , 8) + '\t' +
    utils::toString(point.first.stellarRadius / conv.lengthUnit, 8) + '\n';
}

template<typename ParticleT>
void writeSnapshotText(
    const std::string& fileName,
    const ParticleArray<ParticleT>& points,
    const units::ExternalUnits& conv,
    const std::string& header,
    const double time)
{
    std::ofstream strm(fileName.c_str(), std::ios::out);
    if(!strm) 
        throw std::runtime_error("writeSnapshotText: cannot write to file "+fileName);
    if(!header.empty())
        strm << "#" << header << "\n";
    if(isFinite(time))
        strm << "#time: " << time / conv.timeUnit << "\n";
    strm << formatHeader<ParticleT>();
    for(size_t indx=0; indx<points.size(); indx++)
        strm << formatParticle<ParticleT>(points[indx], conv);
    if(!strm.good())
        throw std::runtime_error("writeSnapshotText: cannot write to file "+fileName);
}

ParticleArrayAux readSnapshotText(const std::string& fileName, const units::ExternalUnits& conv)
{
    std::ifstream strm(fileName.c_str(), std::ios::in);
    if(!strm) 
        throw std::runtime_error("readSnapshotText: cannot read from file "+fileName);
    char firstbyte = strm.peek();
    if(firstbyte<32)
        throw std::runtime_error("readSnapshotText: "+fileName+" is not a valid text file");
    ParticleArrayAux points;
    std::string buffer;
    std::vector<std::string> fields;
    while(std::getline(strm, buffer) && !strm.eof())
    {
        if(!buffer.empty() && utils::isComment(buffer[0]))  // commented line
            continue;
        fields = utils::splitString(buffer, "#;, \t");
        size_t numFields = fields.size();
        if(numFields < 4 ||
            !((fields[0][0]>='0' && fields[0][0]<='9') || fields[0][0]=='-' || fields[0][0]=='+'))
            continue;
        bool haveVel = numFields >= 7;
        double particleMass = utils::toDouble(fields[haveVel ? 6 : 3])  * conv.massUnit;
        double stellarMass  = numFields>=8 ? utils::toDouble(fields[7]) * conv.massUnit : particleMass;
        double stellarRadius= numFields>=9 ? utils::toDouble(fields[8]) * conv.lengthUnit : 0;
        points.add(ParticleAux(coord::PosVelCar(
            utils::toDouble(fields[0]) * conv.lengthUnit,
            utils::toDouble(fields[1]) * conv.lengthUnit,
            utils::toDouble(fields[2]) * conv.lengthUnit,
            haveVel ? utils::toDouble(fields[3]) * conv.velocityUnit : 0,
            haveVel ? utils::toDouble(fields[4]) * conv.velocityUnit : 0,
            haveVel ? utils::toDouble(fields[5]) * conv.velocityUnit : 0),
            stellarMass,
            stellarRadius),
            particleMass);
    }
    return points;
}


//----- NEMO file format -----//

/// helper class for writing snapshots in NEMO format
class NemoSnapshotWriter
{
    std::ofstream snap;  ///< data stream
    int level;           ///< index of current level (root level is 0)
public:
    static const int COORDSYS = 66306;  ///< magic code for the courtesian coordinate system

    /// create class instance and open file (append if necessary)
    NemoSnapshotWriter(const std::string &filename, bool append=false) {
        snap.open(filename.c_str(), std::ios::binary | (append? std::ios_base::app : std::ios_base::trunc));
        level=0;
    }

    /// return a letter corresponding to the given type
    template<typename T> char typeLetter();

    /// store a named quantity
    template<typename T> void putVal(const std::string &name, T val) {
        snap.put(-110);
        snap.put(9);
        snap.put(typeLetter<T>());
        putZString(name);
        snap.write(reinterpret_cast<const char*>(&val), sizeof(T));
    }

    /// write array of T; ndim - number of dimensions, dim - length of array for each dimension
    template<typename T> void putArray(const std::string &name, int ndim, const int dim[], const T* data) {
        snap.put(-110);
        snap.put(11);
        snap.put(typeLetter<T>());
        putZString(name);
        snap.write(reinterpret_cast<const char*>(dim), ndim*sizeof(int));
        int buf=0;
        snap.write(reinterpret_cast<const char*>(&buf), sizeof(int));
        size_t size=1;
        for(int i=0; i<ndim; i++)
            size *= dim[i];   // compute the array size
        snap.write(reinterpret_cast<const char*>(data), size*sizeof(T));
    }

    /// begin a new nested array
    void startLevel(const std::string &name) {
        level++;
        snap.put(-110);
        snap.put(9);
        snap.put('(');
        putZString(name);
    }

    /// end a nested array
    void endLevel() {
        level--;
        assert(level>=0);
        snap.put(-110);
        snap.put(9);
        snap.put(')');
        snap.put(0);
    }

    /// store an array of char
    void putString(const std::string &name, const std::string &str) {
        int len=static_cast<int>(str.size());
        putArray(name, 1, &len, str.c_str());
    }

    /// store a string enclosed by zeros
    void putZString(const std::string &name) {
        snap.put(0);
        snap.write(name.c_str(), static_cast<std::streamsize>(name.size()));
        snap.put(0);
    }

    /// write phase space
    template<typename ParticleT>
    void writeParticles(const ParticleArray<ParticleT>& points, const units::ExternalUnits& conv);

    /// check if any i/o errors occured
    bool ok() const { return snap.good(); }
};

template<> char NemoSnapshotWriter::typeLetter<int>()   { return 'i'; }
template<> char NemoSnapshotWriter::typeLetter<float>() { return 'f'; }
template<> char NemoSnapshotWriter::typeLetter<double>(){ return 'd'; }
template<> char NemoSnapshotWriter::typeLetter<char>()  { return 'c'; }

template<> void NemoSnapshotWriter::writeParticles<coord::PosCar>(
    const ParticleArray<coord::PosCar>& points, const units::ExternalUnits& conv)
{
    int nbody = static_cast<int>(points.size()), dim[2] = {nbody, 3};
    std::vector<float> pos(nbody * 3);
    std::vector<float> mass(nbody);
    for(int i=0; i<nbody; i++) {
        pos [i*3  ] = static_cast<float>(points.point(i).x  / conv.lengthUnit);
        pos [i*3+1] = static_cast<float>(points.point(i).y  / conv.lengthUnit);
        pos [i*3+2] = static_cast<float>(points.point(i).z  / conv.lengthUnit);
        mass[i]     = static_cast<float>(points.mass (i)    / conv.massUnit);
    }
    putArray("Position", 2, dim, &pos[0]);
    putArray("Mass",     1, dim, &mass[0]);
}

template<> void NemoSnapshotWriter::writeParticles<coord::PosVelCar>(
    const ParticleArray<coord::PosVelCar>& points, const units::ExternalUnits& conv)
{
    int nbody = static_cast<int>(points.size()), dim[2] = {nbody, 3};
    std::vector<float> pos(nbody*3), vel(nbody*3), mass(nbody);
    for(int i=0; i<nbody; i++) {
        pos [i*3  ] = static_cast<float>(points.point(i).x  / conv.lengthUnit);
        pos [i*3+1] = static_cast<float>(points.point(i).y  / conv.lengthUnit);
        pos [i*3+2] = static_cast<float>(points.point(i).z  / conv.lengthUnit);
        vel [i*3  ] = static_cast<float>(points.point(i).vx / conv.velocityUnit);
        vel [i*3+1] = static_cast<float>(points.point(i).vy / conv.velocityUnit);
        vel [i*3+2] = static_cast<float>(points.point(i).vz / conv.velocityUnit);
        mass[i]     = static_cast<float>(points.mass (i)    / conv.massUnit);
    }
    putArray("Position", 2, dim, &pos[0]);
    putArray("Velocity", 2, dim, &vel[0]);
    putArray("Mass",     1, dim, &mass[0]);
}

template<> void NemoSnapshotWriter::writeParticles<ParticleAux>(
    const ParticleArray<ParticleAux>& points, const units::ExternalUnits& conv)
{
    int nbody = static_cast<int>(points.size()), dim[2] = {nbody, 3};
    std::vector<float> pos(nbody*3), vel(nbody*3), mass(nbody), aux(nbody), eps(nbody);
    for(int i=0; i<nbody; i++) {
        pos [i*3  ] = static_cast<float>(points.point(i).x  / conv.lengthUnit);
        pos [i*3+1] = static_cast<float>(points.point(i).y  / conv.lengthUnit);
        pos [i*3+2] = static_cast<float>(points.point(i).z  / conv.lengthUnit);
        vel [i*3  ] = static_cast<float>(points.point(i).vx / conv.velocityUnit);
        vel [i*3+1] = static_cast<float>(points.point(i).vy / conv.velocityUnit);
        vel [i*3+2] = static_cast<float>(points.point(i).vz / conv.velocityUnit);
        mass[i]     = static_cast<float>(points.mass (i)    / conv.massUnit);
        aux [i]     = static_cast<float>(points.point(i).stellarMass   / conv.massUnit);
        eps [i]     = static_cast<float>(points.point(i).stellarRadius / conv.lengthUnit);
    }
    putArray("Position", 2, dim, &pos[0]);
    putArray("Velocity", 2, dim, &vel[0]);
    putArray("Mass",     1, dim, &mass[0]);
    putArray("Aux",      1, dim, &aux[0]);
    putArray("Eps",      1, dim, &eps[0]);
}

template<typename ParticleT>
void writeSnapshotNEMO(
    const std::string& fileName,
    const ParticleArray<ParticleT>& points,
    const units::ExternalUnits& conv,
    const std::string& header,
    const double time,
    const bool append)
{
    NemoSnapshotWriter snapshotWriter(fileName, append);
    bool result = snapshotWriter.ok();
    if(result) {
        if(!header.empty())
            snapshotWriter.putString("History", header.c_str());
        int nbody = static_cast<int>(points.size());
        snapshotWriter.startLevel("SnapShot");
        snapshotWriter.startLevel("Parameters");
        snapshotWriter.putVal("Nobj", nbody);
        if(isFinite(time))
            snapshotWriter.putVal("Time", time / conv.timeUnit);
        snapshotWriter.endLevel();
        snapshotWriter.startLevel("Particles");
        snapshotWriter.putVal("CoordSystem", snapshotWriter.COORDSYS);
        snapshotWriter.writeParticles(points, conv);
        snapshotWriter.endLevel();
        snapshotWriter.endLevel();
        result = snapshotWriter.ok();
    }
    if(!result)
        throw std::runtime_error("writeSnapshotNEMO: cannot write to file "+fileName);
}


//----- file formats supported by the UNSIO library -----//

#ifdef HAVE_UNSIO

ParticleArrayAux readSnapshotUNSIO(const std::string& fileName,
    const units::ExternalUnits& conv) 
{ 
    uns::CunsIn input(fileName.c_str(), "all", "all");
    if(input.isValid() && input.snapshot->nextFrame("")) {
        FILTERMSG(utils::VL_DEBUG, "readSnapshotUNSIO",
            "input snapshot file "+fileName+" of type "+input.snapshot->getInterfaceType());
        float *pos=NULL, *vel=NULL, *mass=NULL, *aux=NULL, *eps=NULL;
        int nbodyp, nbodyv, nbodym, nbodya, nbodye;
        input.snapshot->getData("pos", &nbodyp, &pos);
        input.snapshot->getData("vel", &nbodyv, &vel);
        input.snapshot->getData("mass",&nbodym, &mass);
        input.snapshot->getData("aux", &nbodya, &aux);
        input.snapshot->getData("eps", &nbodye, &eps);
        if(nbodyp==0) pos=NULL;
        if(nbodyv==0) vel=NULL;
        if(nbodym==0) mass=NULL;
        if(nbodya==0) aux=NULL;
        if(nbodye==0) eps=NULL;
        ParticleArrayAux points;
        if(nbodyp>0) {
            points.data.reserve(nbodyp);
            for(int i=0; i<nbodyp; i++) {
                double particleMass = mass? mass[i] * conv.massUnit : 0;
                double stellarMass  = aux ? aux [i] * conv.massUnit : particleMass;
                double stellarRadius= eps ? eps [i] * conv.lengthUnit : 0;
                points.add(ParticleAux(coord::PosVelCar(
                    pos[i*3]   * conv.lengthUnit, 
                    pos[i*3+1] * conv.lengthUnit, 
                    pos[i*3+2] * conv.lengthUnit,
                    vel ? vel[i*3]   * conv.velocityUnit : 0, 
                    vel ? vel[i*3+1] * conv.velocityUnit : 0, 
                    vel ? vel[i*3+2] * conv.velocityUnit : 0),
                    stellarMass,
                    stellarRadius),
                    particleMass);
            }
        }
        return points;
    } else
        throw std::runtime_error("readSnapshotUNSIO: cannot read from file "+fileName);
}

// write an UNSIO snapshot (essentially, this function is only used for the Gadget format,
// hence this weird 'halo' argument and the inability to store extra attributes)
template<typename ParticleT>
bool writeParticlesUNSIO(
    uns::CSnapshotInterfaceOut& file,
    const ParticleArray<ParticleT>& points,
    const units::ExternalUnits& conv)
{
    int nbody = static_cast<int>(points.size());
    std::vector<float> pos(nbody*3), vel(nbody*3), mass(nbody);
    for(int i=0; i<nbody; i++) {
        pos[i*3]  = static_cast<float>(points.point(i).x  / conv.lengthUnit);
        pos[i*3+1]= static_cast<float>(points.point(i).y  / conv.lengthUnit);
        pos[i*3+2]= static_cast<float>(points.point(i).z  / conv.lengthUnit);
        vel[i*3]  = static_cast<float>(points.point(i).vx / conv.velocityUnit);
        vel[i*3+1]= static_cast<float>(points.point(i).vy / conv.velocityUnit);
        vel[i*3+2]= static_cast<float>(points.point(i).vz / conv.velocityUnit);
        mass[i]   = static_cast<float>(points.mass (i)    / conv.massUnit);
    }
    return
    file.setData("halo","pos", nbody, &(pos.front()), true) > 0  &&
    file.setData("halo","vel", nbody, &(vel.front()), true) > 0  &&
    file.setData("halo","mass",nbody, &(mass.front()),true) > 0  &&
    file.save() > 0;
}

// specialization of the above function for the case when only coordinates are available
template<> inline bool writeParticlesUNSIO<coord::PosCar>(
    uns::CSnapshotInterfaceOut& file,
    const ParticleArray<coord::PosCar>& points,
    const units::ExternalUnits& conv)
{
    int nbody = static_cast<int>(points.size());
    std::vector<float> pos(nbody*3), mass(nbody);
    for(int i=0; i<nbody; i++) {
        pos[i*3]  = static_cast<float>(points.point(i).x  / conv.lengthUnit);
        pos[i*3+1]= static_cast<float>(points.point(i).y  / conv.lengthUnit);
        pos[i*3+2]= static_cast<float>(points.point(i).z  / conv.lengthUnit);
        mass[i]   = static_cast<float>(points.mass (i)    / conv.massUnit);
    }
    return
    file.setData("halo","pos", nbody, &(pos.front()), true) > 0  &&
    file.setData("halo","mass",nbody, &(mass.front()),true) > 0  &&
    file.save() > 0;
}

template<typename ParticleT>
void writeSnapshotUNSIO(
    const std::string& fileName,
    const ParticleArray<ParticleT>& points,
    const units::ExternalUnits& conv,
    const std::string& /*header - ignored*/,
    const double time,
    const char* type)
{
    uns::CunsOut output(fileName.c_str(), type);
    bool result = true;
    if(isFinite(time))
        result &= (bool)output.snapshot->setData("time", static_cast<float>(time));
    result &= writeParticlesUNSIO<ParticleT>(*output.snapshot, points, conv);
    if(!result) 
        throw std::runtime_error("writeSnapshotUNSIO: cannot write to file "+fileName);
}

#endif

}  // internal namespace


// 'readSnapshot' always returns the richest possible particle flavour (ParticleAux), which
// can then be 'downgraded' to any desired level and converted to a different coordinate system.
ParticleArrayAux readSnapshot(
    const std::string& fileName,
    const units::ExternalUnits& unitConverter)
{
    std::ifstream strm(fileName.c_str(), std::ios::in);
    if(!strm)
        throw std::runtime_error("readSnapshot: cannot read from "+fileName);
    char buffer[8];
    strm.read(buffer, 8);
    strm.close();
    if( (buffer[0]==-110 &&  // NEMO signature: open block of a particular type
        (buffer[2]=='c' || buffer[2]=='i' || buffer[2]=='f' || buffer[2]=='d' || buffer[2]=='(')) ||
        // GADGET signature (depending on byte order and version)
        (buffer[0]==8 && buffer[1]==0 && buffer[2]==0 && buffer[3]==0) ||
        (buffer[0]==0 && buffer[1]==1 && buffer[2]==0 && buffer[3]==0) ||
        (buffer[0]==0 && buffer[1]==0 && buffer[2]==8 && buffer[3]==0) ||
        (buffer[0]==0 && buffer[1]==0 && buffer[2]==0 && buffer[3]==1) )
    {
#ifdef HAVE_UNSIO
        return readSnapshotUNSIO(fileName, unitConverter);  // NEMO or Gadget
#else
        throw std::runtime_error("readSnapshot: compiled without support for reading NEMO or GADGET files");
#endif
    }
    else if(buffer[0]>=32)
    {
        return readSnapshotText(fileName, unitConverter);
    }
    throw std::runtime_error("readSnapshot: file format not recognized");
}


template<typename ParticleT>
inline void writeSnapshot(
    const std::string& fileName,
    const ParticleArray<ParticleT>& particles,
    const std::string &fileFormat,
    const units::ExternalUnits& unitConverter,
    const std::string& header,
    const double time,
    const bool append)
{
    if(fileFormat.empty() || fileName.empty())
        throw std::runtime_error("writeSnapshot: file name or format is empty");
    if(tolower(fileFormat[0])=='t') {
        writeSnapshotText(fileName, particles, unitConverter, header, time);
    }
    else if(tolower(fileFormat[0])=='n')
    {
        writeSnapshotNEMO(fileName, particles, unitConverter, header, time, append);
    }
    else if(tolower(fileFormat[0])=='g')
    {
#ifdef HAVE_UNSIO
        writeSnapshotUNSIO(fileName, particles, unitConverter, header, time, "gadget2");
#else
        throw std::runtime_error("writeSnapshot: compiled without support for writing GADGET files");
#endif
    }
    else
        throw std::runtime_error("writeSnapshot: file format not recognized");
}

// 'writeSnapshot' is a templated function which accepts several kinds of particle 'identities'
// (position only, or position/velocity, or ParticleAux) in different coordinate systems.
// It is actually implemented only for the Cartesian system (for all three particle identities),
// and other coordinate systems are converted to Cartesian with the same particle identity.

// the actual instantiations for Cartesian system and various identities
template void writeSnapshot(
    const std::string& fileName, const ParticleArray<coord::PosCar>& particles,
    const std::string& fileFormat, const units::ExternalUnits& unitConverter,
    const std::string& header, const double time, const bool append);

template void writeSnapshot(
    const std::string& fileName, const ParticleArray<coord::PosVelCar>& particles,
    const std::string& fileFormat, const units::ExternalUnits& unitConverter,
    const std::string& header, const double time, const bool append);

template void writeSnapshot(
    const std::string& fileName, const ParticleArray<ParticleAux>& particles,
    const std::string& fileFormat, const units::ExternalUnits& unitConverter,
    const std::string& header, const double time, const bool append);

// instantiations for other coordinate systems with conversion into Cartesian
template<> void writeSnapshot(
    const std::string& fileName, const ParticleArray<coord::PosCyl>& particles,
    const std::string& fileFormat, const units::ExternalUnits& unitConverter,
    const std::string& header, const double time, const bool append)
{
    writeSnapshot(fileName, ParticleArray<coord::PosCar>(particles),
        fileFormat, unitConverter, header, time, append);
}

template<> void writeSnapshot(
    const std::string& fileName, const ParticleArray<coord::PosSph>& particles,
    const std::string& fileFormat, const units::ExternalUnits& unitConverter,
    const std::string& header, const double time, const bool append)
{
    writeSnapshot(fileName, ParticleArray<coord::PosCar>(particles),
        fileFormat, unitConverter, header, time, append);
}

template<> void writeSnapshot(
    const std::string& fileName, const ParticleArray<coord::PosVelCyl>& particles,
    const std::string& fileFormat, const units::ExternalUnits& unitConverter,
    const std::string& header, const double time, const bool append)
{
    writeSnapshot(fileName, ParticleArray<coord::PosVelCar>(particles),
        fileFormat, unitConverter, header, time, append);
}

template<> void writeSnapshot(
    const std::string& fileName, const ParticleArray<coord::PosVelSph>& particles,
    const std::string& fileFormat, const units::ExternalUnits& unitConverter,
    const std::string& header, const double time, const bool append)
{
    writeSnapshot(fileName, ParticleArray<coord::PosVelCar>(particles),
        fileFormat, unitConverter, header, time, append);
}

}  // namespace particles
