#include "particles_io.h"
#ifdef HAVE_UNSIO
#include <uns.h>
#endif
#include <fstream>
#include <stdexcept>
#include "utils.h"

namespace particles {

void IOSnapshotText::readSnapshot(PointMassArrayCar& points)
{
    std::ifstream strm(fileName.c_str(), std::ios::in);
    if(!strm) 
        throw std::runtime_error("IOSnapshotText: cannot read from file "+fileName);
    points.data.clear();
    std::string buffer;
    std::vector<std::string> fields;
    bool noMasses=false, nonzeroMasses=false;
    while(std::getline(strm, buffer) && !strm.eof())
    {
        utils::splitString(buffer, "%# \t", fields);
        size_t numFields=fields.size();
        if(numFields>=3 && ((fields[0][0]>='0' && fields[0][0]<='9') || fields[0][0]=='-' || fields[0][0]=='+'))
        {
            double mass=numFields>6 ? utils::convertToDouble(fields[6]):(noMasses=true,1.0);
            points.add(coord::PosVelCar(
                utils::convertToDouble(fields[0]), 
                utils::convertToDouble(fields[1]), 
                utils::convertToDouble(fields[2]), 
                numFields>=6 ? utils::convertToDouble(fields[3]) : 0, 
                numFields>=6 ? utils::convertToDouble(fields[4]) : 0, 
                numFields>=6 ? utils::convertToDouble(fields[5]) : 0),
                mass);
            if(mass>0) nonzeroMasses=true;
        }
    }
    if(noMasses || !nonzeroMasses)
    {
        double mass=1.0/points.size();
        for(size_t i=0; i<points.size(); i++)
            points[i].second=mass;
    }
};

void IOSnapshotText::writeSnapshot(const PointMassArrayCar& points)
{
    std::ofstream strm(fileName.c_str(), std::ios::out);
    if(!strm) 
        throw std::runtime_error("IOSnapshotText: cannot write to file "+fileName);
    strm << "x\ty\tz\tvx\tvy\tvz\tm" << std::endl;
    for(size_t indx=0; indx<points.size(); indx++)
    {
        const coord::PosVelCar& pt=points[indx].first;
        strm << 
            pt.x  << "\t" << pt.y  << "\t" << pt.z  << "\t" << 
            pt.vx << "\t" << pt.vy << "\t" << pt.vz << "\t" <<
            points[indx].second << std::endl;
    }
    if(!strm.good())
        throw std::runtime_error("IOSnapshotText: cannot read from file "+fileName);
}

#ifdef HAVE_UNSIO

void readSnapshotUNSIO(const std::string& fileName, PointMassArrayCar& points) 
{ 
    uns::CunsIn input(fileName, "all", "all");
    if(input.isValid() && input.snapshot->nextFrame("xvm")) {
#ifdef DEBUGPRINT
        my_message(FUNCNAME, 
            "input snapshot file "+fileName+" of type "+input.snapshot->getInterfaceType());
#endif
        float *pos=NULL, *vel=NULL, *mass=NULL;
        int nbodyp, nbodyv, nbodym;
        input.snapshot->getData("pos", &nbodyp, &pos);
        input.snapshot->getData("vel", &nbodyv, &vel);
        input.snapshot->getData("mass", &nbodym, &mass);
        if(nbodyp==0) pos=NULL;
        if(nbodyv==0) vel=NULL;
        if(nbodym==0) mass=NULL;
        if(nbodyp>0) {
            points.data.clear();
            points.data.reserve(nbodyp);
            for(int i=0; i<nbodyp; i++)
                points.add(coord::PosVelCar(pos[i*3], pos[i*3+1], pos[i*3+2],
                    vel ? vel[i*3] : 0, vel ? vel[i*3+1] : 0, vel ? vel[i*3+2] : 0),
                    mass? mass[i] : 1.0/nbodyp );
        }
    } else
        throw std::runtime_error("IOSnapshotUNSIO: cannot read from file "+fileName);
};

void writeSnapshotUNSIO(const std::string& fileName, const std::string& type,
    const PointMassArrayCar& points)
{
    uns::CunsOut output(fileName, type);
    bool result = 1 || output.isValid();  // this flag is apparently not initialized properly
    if(result) {
        int nbody=static_cast<int>(points.size());
        std::vector<float> pos(nbody*3), vel(nbody*3), mass(nbody);
        for(int i=0; i<nbody; i++) {
            const coord::PosVelCar& pt=points[i].first;
            pos[i*3]  = static_cast<float>(pt.x);
            pos[i*3+1]= static_cast<float>(pt.y);
            pos[i*3+2]= static_cast<float>(pt.z);
            vel[i*3]  = static_cast<float>(pt.vx);
            vel[i*3+1]= static_cast<float>(pt.vy);
            vel[i*3+2]= static_cast<float>(pt.vz);
            mass[i]   = static_cast<float>(points[i].second);
        }
        result &= output.snapshot->setData("halo", "pos", nbody, &(pos.front()), true)>0;
        result &= output.snapshot->setData("halo", "vel", nbody, &(vel.front()), true)>0;
        result &= output.snapshot->setData("halo", "mass",nbody, &(mass.front()),true)>0;
        result &= output.snapshot->save()>0;
    }
    if(!result) 
        throw std::runtime_error("IOSnapshotUNSIO: cannot write to file "+fileName);
};

void IOSnapshotGadget::readSnapshot(PointMassArrayCar& points) {
    readSnapshotUNSIO(fileName, points);
}

void IOSnapshotGadget::writeSnapshot(const PointMassArrayCar& points) {
    writeSnapshotUNSIO(fileName, "gadget2", points);
}

void IOSnapshotNemo::readSnapshot(PointMassArrayCar& points) {
    readSnapshotUNSIO(fileName, points);
}

#else
// no UNSIO
void IOSnapshotNemo::readSnapshot(PointMassArrayCar&)
{
    throw std::runtime_error("Error, compiled without support for reading NEMO snapshots");
};
#endif

/// helper class that writes NEMO-compatible snapshot file
class CNemoSnapshotWriter
{
public:
    /// create class instance and open file (append if necessary)
    CNemoSnapshotWriter(const std::string &filename, bool append=false) {
        snap.open(filename.c_str(), std::ios::binary | (append?std::ios_base::app : std::ios_base::trunc));
        level=0;
    }
    /// close file
    ~CNemoSnapshotWriter() { snap.close(); }
    /// return a letter corresponding to the given type
    template<typename T> char typeLetter();
    /// store a named quantity
    template<typename T> void putVal(const std::string &name, T val) {
        snap.put(-110);
        snap.put(9);
        snap.put(typeLetter<T>());
        putZString(name);
        snap.write(reinterpret_cast<const char*>(&val), sizeof(T));
    };
    /// write array of T; ndim - number of dimensions, dim - length of array for each dimension
    template<typename T> void putArray(const std::string &name, int ndim, const int dim[], const T* data) {
        snap.put(-110);
        snap.put(11);
        snap.put(typeLetter<T>());
        putZString(name);
        snap.write(reinterpret_cast<const char*>(dim), ndim*sizeof(int));
        int buf=0;
        snap.write(reinterpret_cast<const char*>(&buf), sizeof(int));
        buf=1;
        for(int i=0; i<ndim; i++)
            buf*=dim[i];   // compute the array size
        snap.write(reinterpret_cast<const char*>(data), sizeof(T)*buf);
    };
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
        if (level<0)
        {   // should raise error
            throw std::runtime_error("Error writing NEMO snapshot: level<0 in endLevel()");
        }
        snap.put(-110);
        snap.put(9);
        snap.put(')');
        snap.put(0);
    }
    /// store an array of char
    void putString(const std::string &name,const std::string &str) {
        int len=static_cast<int>(str.size());
        putArray(name, 1, &len, str.c_str());
    }
    /// store a string enclosed by zeros
    void putZString(const std::string &name) {
        snap.put(0);
        snap.write(name.c_str(),static_cast<std::streamsize>(name.size()));
        snap.put(0);
    }
    /// store history line in header
    void writeHistory(const std::string &new_h) {
        if (new_h.size() > 0)
            putString("History",new_h.c_str());
    }
    /// write phase space (positions and velocities)
    template<typename NumT> void writePhase(const PointMassArrayCar& points, double time) {
        int nbody    = static_cast<int>(points.size());
        NumT* phase = new NumT[nbody * 6];
        NumT* mass  = new NumT[nbody];
        for(int i = 0 ; i < nbody ; i++) {
            phase[i*6  ] = static_cast<NumT>(points[i].first.x);
            phase[i*6+1] = static_cast<NumT>(points[i].first.y);
            phase[i*6+2] = static_cast<NumT>(points[i].first.z);
            phase[i*6+3] = static_cast<NumT>(points[i].first.vx);
            phase[i*6+4] = static_cast<NumT>(points[i].first.vy);
            phase[i*6+5] = static_cast<NumT>(points[i].first.vz);
            mass[i] = points[i].second;
        }
        startLevel("SnapShot");
        startLevel("Parameters");
        putVal("Nobj", nbody);
        putVal("Time", time);
        endLevel();
        startLevel("Particles");
        putVal("CoordSystem",static_cast<int>(0201402));
        int tmp_dim[3];
        tmp_dim[0] = nbody;
        putArray("Mass",1,tmp_dim,mass);
        tmp_dim[0] = nbody;
        tmp_dim[1] = 2;
        tmp_dim[2] = 3;
        putArray("PhaseSpace",3,tmp_dim,phase);
        endLevel();
        endLevel();
        delete [] phase;
        delete [] mass;
    }
    /// check if any i/o errors occured
    bool ok() const { return snap.good(); }
private:
    std::ofstream snap;  ///< data stream
    int level;           ///< index of current level (root level is 0)
};
template<> char CNemoSnapshotWriter::typeLetter<int>()   { return 'i'; };
template<> char CNemoSnapshotWriter::typeLetter<float>() { return 'f'; };
template<> char CNemoSnapshotWriter::typeLetter<double>(){ return 'd'; };
template<> char CNemoSnapshotWriter::typeLetter<char>()  { return 'c'; };

void IOSnapshotNemo::writeSnapshot(const PointMassArrayCar& points)
{
    CNemoSnapshotWriter SnapshotWriter(fileName, append);
    bool result = SnapshotWriter.ok();
    if(result) {
        SnapshotWriter.writeHistory(header);
        SnapshotWriter.writePhase<float>(points, time); 
        result = SnapshotWriter.ok();
    }
    if(!result) 
        throw std::runtime_error("IOSnapshotNEMO: cannot write to file "+fileName);
};

#if 0
/// create a list of all IO snapshot formats available at compile time
std::vector< std::string > initFormatsIOSnapshot()
{
    std::vector< std::string > formats;
    formats.push_back("Text");
    formats.push_back("Nemo");
#ifdef HAVE_UNSIO
    formats.push_back("Gadget");
#endif
    return formats;
};

// list of all available IO snapshot formats, initialized at module start 
// according to the file formats supported at compile time
std::vector< std::string > formatsIOSnapshot = initFormatsIOSnapshot();
#endif

// creates an instance of appropriate snapshot reader, according to the file format 
// determined by reading first few bytes, or throw an exception if a file doesn't exist
BaseIOSnapshot* createIOSnapshotRead (const std::string &fileName)
{
    std::ifstream strm(fileName.c_str(), std::ios::in);
    if(!strm)
        throw std::runtime_error("Cannot read snapshot from "+fileName);
    char buffer[8];
    strm.read(buffer, 8);
    strm.close();
    if(buffer[0]==-110 &&  // NEMO signature: open block
        (buffer[2]=='c' || buffer[2]=='i' || buffer[2]=='f' || buffer[2]=='d' || buffer[2]=='('))  // nemo block type
    {
        return new IOSnapshotNemo(fileName);
    } else
#ifdef HAVE_UNSIO
    if( (buffer[0]==8 && buffer[1]==0 && buffer[2]==0 && buffer[3]==0) ||
        (buffer[0]==0 && buffer[1]==1 && buffer[2]==0 && buffer[3]==0) ||
        (buffer[0]==0 && buffer[1]==0 && buffer[2]==8 && buffer[3]==0) ||
        (buffer[0]==0 && buffer[1]==0 && buffer[2]==0 && buffer[3]==1) )
    {
        return new IOSnapshotGadget(fileName);
    } else
#endif
    {   // anything else is a text file by default (might not work out anyway)
        return new IOSnapshotText(fileName);
    }
};

// creates an instance of snapshot writer for a given format name, 
// or throw an exception if the format name string is incorrect
BaseIOSnapshot* createIOSnapshotWrite(const std::string &fileFormat, const std::string &fileName, 
    const std::string& header, const double time, const bool append)
{
    if(fileFormat.empty() || fileName.empty())
        throw std::runtime_error("Snapshot file name or format is empty");
    if(tolower(fileFormat[0])=='t')
        return new IOSnapshotText(time==0 ? fileName : (fileName + utils::convertToString(time)));
    else 
    if(tolower(fileFormat[0])=='n')
        return new IOSnapshotNemo(fileName, header, time, append);
#ifdef HAVE_UNSIO
    else 
    if(tolower(fileFormat[0])=='g')
        return new IOSnapshotGadget(time==0 ? fileName : (fileName + utils::convertToString(time)));
#endif
    else
        throw std::runtime_error("Snapshot file format not recognized");   // error - format name not found
};

};  // namespace
