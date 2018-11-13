#include "particles_io.h"
#ifdef HAVE_UNSIO
#include <uns.h>
#endif
#include <fstream>
#include <cassert>
#include <stdexcept>
#include "utils.h"

namespace particles {

ParticleArrayCar IOSnapshotText::readSnapshot() const
{
    std::ifstream strm(fileName.c_str(), std::ios::in);
    if(!strm) 
        throw std::runtime_error("IOSnapshotText: cannot read from file "+fileName);
    ParticleArrayCar points;
    std::string buffer;
    std::vector<std::string> fields;
    while(std::getline(strm, buffer) && !strm.eof())
    {
        fields = utils::splitString(buffer, "%# \t");
        size_t numFields = fields.size();
        if(numFields>=7 && 
            ((fields[0][0]>='0' && fields[0][0]<='9') || fields[0][0]=='-' || fields[0][0]=='+'))
        {
            points.add(coord::PosVelCar(
                utils::toDouble(fields[0]) * conv.lengthUnit, 
                utils::toDouble(fields[1]) * conv.lengthUnit, 
                utils::toDouble(fields[2]) * conv.lengthUnit, 
                utils::toDouble(fields[3]) * conv.velocityUnit, 
                utils::toDouble(fields[4]) * conv.velocityUnit, 
                utils::toDouble(fields[5]) * conv.velocityUnit),
                utils::toDouble(fields[6]) * conv.massUnit);
        }
    }
    return points;
}

void IOSnapshotText::writeSnapshot(const ParticleArrayCar& points) const
{
    std::ofstream strm(fileName.c_str(), std::ios::out);
    if(!strm) 
        throw std::runtime_error("IOSnapshotText: cannot write to file "+fileName);
    strm << "#x\ty\tz\tvx\tvy\tvz\tm" << std::endl;
    for(size_t indx=0; indx<points.size(); indx++)
    {
        const coord::PosVelCar& pt = points.point(indx);
        strm <<
            utils::toString(pt.x  / conv.lengthUnit  ) + '\t' +
            utils::toString(pt.y  / conv.lengthUnit  ) + '\t' +
            utils::toString(pt.z  / conv.lengthUnit  ) + '\t' +
            utils::toString(pt.vx / conv.velocityUnit) + '\t' +
            utils::toString(pt.vy / conv.velocityUnit) + '\t' +
            utils::toString(pt.vz / conv.velocityUnit) + '\t' +
            utils::toString(points.mass(indx) / conv.massUnit) + '\n';
    }
    if(!strm.good())
        throw std::runtime_error("IOSnapshotText: cannot read from file "+fileName);
}

#ifdef HAVE_UNSIO
namespace{  // internal

ParticleArrayCar readSnapshotUNSIO(const std::string& fileName, 
    const units::ExternalUnits& conv) 
{ 
    uns::CunsIn input(fileName, "all", "all");
    if(input.isValid() && input.snapshot->nextFrame("xvm")) {
        utils::msg(utils::VL_DEBUG, FUNCNAME,
            "input snapshot file "+fileName+" of type "+input.snapshot->getInterfaceType());
        float *pos=NULL, *vel=NULL, *mass=NULL;
        int nbodyp, nbodyv, nbodym;
        input.snapshot->getData("pos", &nbodyp, &pos);
        input.snapshot->getData("vel", &nbodyv, &vel);
        input.snapshot->getData("mass", &nbodym, &mass);
        if(nbodyp==0) pos=NULL;
        if(nbodyv==0) vel=NULL;
        if(nbodym==0) mass=NULL;
        ParticleArrayCar points;
        if(nbodyp>0) {
            points.data.reserve(nbodyp);
            for(int i=0; i<nbodyp; i++)
                points.add(coord::PosVelCar(
                    pos[i*3]   * conv.lengthUnit, 
                    pos[i*3+1] * conv.lengthUnit, 
                    pos[i*3+2] * conv.lengthUnit,
                    vel ? vel[i*3]   * conv.velocityUnit : 0, 
                    vel ? vel[i*3+1] * conv.velocityUnit : 0, 
                    vel ? vel[i*3+2] * conv.velocityUnit : 0),
                    mass? mass[i] * conv.massUnit : 0 );
        }
        return points;
    } else
        throw std::runtime_error("IOSnapshotUNSIO: cannot read from file "+fileName);
}

void writeSnapshotUNSIO(const std::string& fileName,
    const units::ExternalUnits& conv, const ParticleArrayCar& points, const std::string& type)
{
    uns::CunsOut output(fileName, type);
    bool result = true;
    int nbody=static_cast<int>(points.size());
    std::vector<float> pos(nbody*3), vel(nbody*3), mass(nbody);
    for(int i=0; i<nbody; i++) {
        const coord::PosVelCar& pt = points.point(i);
        pos[i*3]  = static_cast<float>(pt.x  / conv.lengthUnit);
        pos[i*3+1]= static_cast<float>(pt.y  / conv.lengthUnit);
        pos[i*3+2]= static_cast<float>(pt.z  / conv.lengthUnit);
        vel[i*3]  = static_cast<float>(pt.vx / conv.velocityUnit);
        vel[i*3+1]= static_cast<float>(pt.vy / conv.velocityUnit);
        vel[i*3+2]= static_cast<float>(pt.vz / conv.velocityUnit);
        mass[i]   = static_cast<float>(points.mass(i) / conv.massUnit);
    }
    result &= output.snapshot->setData("halo", "pos", nbody, &(pos.front()), true)>0;
    result &= output.snapshot->setData("halo", "vel", nbody, &(vel.front()), true)>0;
    result &= output.snapshot->setData("halo", "mass",nbody, &(mass.front()),true)>0;
    result &= output.snapshot->save()>0;
    if(!result) 
        throw std::runtime_error("IOSnapshotUNSIO: cannot write to file "+fileName);
}
}  // internal ns

ParticleArrayCar IOSnapshotGadget::readSnapshot() const {
    return readSnapshotUNSIO(fileName, conv);
}

void IOSnapshotGadget::writeSnapshot(const ParticleArrayCar& points) const {
    writeSnapshotUNSIO(fileName, conv, points, "gadget2");
}

ParticleArrayCar IOSnapshotNemo::readSnapshot() const {
    return readSnapshotUNSIO(fileName, conv);
}

#else
// no UNSIO
ParticleArrayCar IOSnapshotNemo::readSnapshot() const
{
    throw std::runtime_error("Error, compiled without support for reading NEMO snapshots");
}
#endif

namespace {   // internal
/// helper class that writes NEMO-compatible snapshot file
class NemoSnapshotWriter
{
    std::ofstream snap;  ///< data stream
    int level;           ///< index of current level (root level is 0)
    static const int COORDSYS = 66306;  ///< magic code for the cartesian coordinate system
public:
    /// create class instance and open file (append if necessary)
    NemoSnapshotWriter(const std::string &filename, bool append=false) {
        snap.open(filename.c_str(), std::ios::binary | (append? std::ios_base::app : std::ios_base::trunc));
        level=0;
    }
    /// close file
    ~NemoSnapshotWriter() { snap.close(); }
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
    /// store history line in header
    void writeHistory(const std::string &text) {
        if(!text.empty())
            putString("History", text.c_str());
    }
    /// write phase space (positions and velocities); NumT may be float or double
    template<typename NumT> 
    void writePhase(const ParticleArrayCar& points, double time, const units::ExternalUnits& conv) 
    {
        size_t nbody = static_cast<int>(points.size());
        // check if the velocity information is present
        bool noVel = true;
        for(size_t i=0; noVel && i<nbody; i++)
            noVel &= pow_2(points.point(i).vx) + pow_2(points.point(i).vy) + pow_2(points.point(i).vz) == 0;
        int nvar = noVel ? 3 : 6, inbody = nbody;
        std::vector<NumT> phase(nbody * nvar);
        std::vector<NumT> mass(nbody);
        for(size_t i = 0 ; i < nbody ; i++) {
            mass [i]        = static_cast<NumT>(points.mass(i)     / conv.massUnit);
            phase[i*nvar  ] = static_cast<NumT>(points.point(i).x  / conv.lengthUnit);
            phase[i*nvar+1] = static_cast<NumT>(points.point(i).y  / conv.lengthUnit);
            phase[i*nvar+2] = static_cast<NumT>(points.point(i).z  / conv.lengthUnit);
            if(noVel) continue;
            phase[i*nvar+3] = static_cast<NumT>(points.point(i).vx / conv.velocityUnit);
            phase[i*nvar+4] = static_cast<NumT>(points.point(i).vy / conv.velocityUnit);
            phase[i*nvar+5] = static_cast<NumT>(points.point(i).vz / conv.velocityUnit);
        }
        startLevel("SnapShot");
        startLevel("Parameters");
        putVal("Nobj", inbody);
        putVal("Time", time);
        endLevel();
        startLevel("Particles");
        putVal("CoordSystem", COORDSYS);
        putArray("Mass", 1, &inbody, &mass[0]);
        if(noVel) {
            int tmp_dim[2] = {inbody, 3};
            putArray("Position", 2, tmp_dim, &phase[0]);
        } else {
            int tmp_dim[3] = {inbody, 2, 3};
            putArray("PhaseSpace", 3, tmp_dim, &phase[0]);
        }
        endLevel();
        endLevel();
    }
    /// check if any i/o errors occured
    bool ok() const { return snap.good(); }
};
template<> char NemoSnapshotWriter::typeLetter<int>()   { return 'i'; }
template<> char NemoSnapshotWriter::typeLetter<float>() { return 'f'; }
template<> char NemoSnapshotWriter::typeLetter<double>(){ return 'd'; }
template<> char NemoSnapshotWriter::typeLetter<char>()  { return 'c'; }
}  // end internal namespace

void IOSnapshotNemo::writeSnapshot(const ParticleArrayCar& points) const
{
    NemoSnapshotWriter snapshotWriter(fileName, append);
    bool result = snapshotWriter.ok();
    if(result) {
        snapshotWriter.writeHistory(header);
        snapshotWriter.writePhase<float>(points, time, conv); 
        result = snapshotWriter.ok();
    }
    if(!result) 
        throw std::runtime_error("IOSnapshotNEMO: cannot write to file "+fileName);
}


// creates an instance of appropriate snapshot reader, according to the file format 
// determined by reading first few bytes, or throw an exception if a file doesn't exist
PtrIOSnapshot createIOSnapshotRead (const std::string &fileName, 
    const units::ExternalUnits& unitConverter)
{
    std::ifstream strm(fileName.c_str(), std::ios::in);
    if(!strm)
        throw std::runtime_error("Cannot read snapshot from "+fileName);
    char buffer[8];
    strm.read(buffer, 8);
    strm.close();
    if(buffer[0]==-110 &&  // NEMO signature: open block of a particular type
        (buffer[2]=='c' || buffer[2]=='i' || buffer[2]=='f' || buffer[2]=='d' || buffer[2]=='('))
    {
        return PtrIOSnapshot(new IOSnapshotNemo(fileName, unitConverter));
    } else
#ifdef HAVE_UNSIO
    if( (buffer[0]==8 && buffer[1]==0 && buffer[2]==0 && buffer[3]==0) ||
        (buffer[0]==0 && buffer[1]==1 && buffer[2]==0 && buffer[3]==0) ||
        (buffer[0]==0 && buffer[1]==0 && buffer[2]==8 && buffer[3]==0) ||
        (buffer[0]==0 && buffer[1]==0 && buffer[2]==0 && buffer[3]==1) )
    {
        return PtrIOSnapshot(new IOSnapshotGadget(fileName, unitConverter));
    } else
#endif
    {   // anything else is a text file by default (might not work out anyway)
        return PtrIOSnapshot(new IOSnapshotText(fileName, unitConverter));
    }
}

// creates an instance of snapshot writer for a given format name, 
// or throw an exception if the format name string is incorrect
PtrIOSnapshot createIOSnapshotWrite(const std::string &fileName, 
    const std::string &fileFormat, const units::ExternalUnits& unitConverter,
    const std::string& header, const double time, const bool append)
{
    if(fileFormat.empty() || fileName.empty())
        throw std::runtime_error("Snapshot file name or format is empty");
    if(tolower(fileFormat[0])=='t')
        return PtrIOSnapshot(new IOSnapshotText(fileName, unitConverter) );
    else 
    if(tolower(fileFormat[0])=='n')
        return PtrIOSnapshot(new IOSnapshotNemo(fileName, unitConverter, header, time, append));
#ifdef HAVE_UNSIO
    else 
    if(tolower(fileFormat[0])=='g')
        return PtrIOSnapshot(new IOSnapshotGadget(fileName, unitConverter) );
#endif
    else
        throw std::runtime_error("Snapshot file format not recognized");   // error - format name not found
}

}  // namespace particles
