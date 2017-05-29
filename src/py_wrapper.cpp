/** \file   py_wrapper.cpp
    \brief  Python wrapper for the Agama library
    \author Eugene Vasiliev
    \date   2014-2017

    This is a Python extension module that provides the interface to
    some of the classes and functions from the Agama C++ library.
    It needs to be compiled into a dynamic library and placed in a folder
    that Python is aware of (e.g., through the PYTHONPATH= environment variable).

    Currently this module provides access to potential classes, orbit integration
    routine, action finders, distribution functions, N-dimensional integration
    and sampling routines, and smoothing splines.
    Unit conversion is also part of the calling convention: the quantities
    received from Python are assumed to be in some physical units and converted
    into internal units inside this module, and the output from the Agama library
    routines is converted back to physical units. The physical units are assigned
    by `setUnits` and `resetUnits` functions.

    Type `help(agama)` in Python to get a list of exported routines and classes,
    and `help(agama.whatever)` to get the usage syntax for each of them.
*/
#ifdef HAVE_PYTHON
#include <Python.h>
#include <structmember.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// note: for some versions of NumPy, it seems necessary to replace constants
// starting with NPY_ARRAY_*** by NPY_***
#include <numpy/arrayobject.h>
#include <stdexcept>
#include <signal.h>
#ifdef _OPENMP
#include "omp.h"
#endif
// include almost everything!
#include "units.h"
#include "potential_factory.h"
#include "potential_composite.h"
#include "potential_multipole.h"
#include "potential_utils.h"
#include "actions_spherical.h"
#include "actions_staeckel.h"
#include "df_factory.h"
#include "df_interpolated.h"
#include "df_pseudoisotropic.h"
#include "galaxymodel.h"
#include "galaxymodel_selfconsistent.h"
#include "galaxymodel_target.h"
#include "galaxymodel_velocitysampler.h"
#include "orbit.h"
#include "math_core.h"
#include "math_optimization.h"
#include "math_sample.h"
#include "math_spline.h"
#include "utils.h"
#include "utils_config.h"

/// classes and routines for the Python interface
namespace pywrapper {  // internal namespace

//  ---------------------------------------------------
/// \name  Helper class to manage the OpenMP behaviour
//  ---------------------------------------------------
///@{

/// This is a lock-type object that temporarily switches off OpenMP parallelization
/// during its existence; this is needed when a Python callback function is provided
/// to the C++ library, because it is not possible to call Python routines concurrently.
/// It remembers the OpenMP setting for the maximum number of threads that was effective
/// at the moment of construction, and restores it upon destruction.
/// If OpenMP is not used, this is a no-op.
class OmpDisabler {
#ifdef _OPENMP
    int origMaxThreads;  ///< the value effective before this object was constructed
public:
    OmpDisabler()
    {
        origMaxThreads = omp_get_max_threads();
        utils::msg(utils::VL_DEBUG, "Agama", "OpenMP is now disabled "
            "(original max # of threads was "+utils::toString(origMaxThreads)+")");
        omp_set_num_threads(1);
    }
    ~OmpDisabler()
    {
        utils::msg(utils::VL_DEBUG, "Agama", "OpenMP is now enabled "
            "(max # of threads is "+utils::toString(origMaxThreads)+")");
        omp_set_num_threads(origMaxThreads);
    }
#else
public:
    OmpDisabler() {}
#endif
};

///@}
//  ----------------------------------------------------------------------------
/// \name  Mechanism for capturing the Control-C signal in lengthy calculations
//  ----------------------------------------------------------------------------
///@{

/// flag that is set to one if the keyboard interrupt has occurred
volatile sig_atomic_t keyboardInterruptTriggered = 0;
/// signal handler installed during lengthy computations that triggers the flag
void customKeyboardInterruptHandler(int) { keyboardInterruptTriggered = 1; }
/// previous signal handler restored after the computation is finished
void (*defaultKeyboardInterruptHandler)(int) = NULL;

///@}
//  ------------------------------------------------------------------
/// \name  Helper routines for type conversions and argument checking
//  ------------------------------------------------------------------
///@{

/// return a string representation of a Python object
std::string toString(PyObject* obj)
{
    if(obj==NULL)
        return "";
    if(PyString_Check(obj))
        return std::string(PyString_AsString(obj));
    PyObject* s = PyObject_Str(obj);
    std::string str = PyString_AsString(s);
    Py_DECREF(s);
    return str;
}

/// return an integer representation of a Python object, or a default value in case of error
int toInt(PyObject* obj, int defaultValue=-1)
{
    if(obj==NULL)
        return defaultValue;
    if(PyNumber_Check(obj)) {
        int value = PyInt_AsLong(obj);
        if(PyErr_Occurred())
            return defaultValue;
        return value;
    }
    // it wasn't a number, but may be it can be converted to a number
    PyObject* l = PyNumber_Long(obj);
    if(l) {
        int value = PyInt_AsLong(l);
        Py_DECREF(l);
        return value;
    }
    return defaultValue;
}

/// return a float representation of a Python object, or a default value in case of error
double toDouble(PyObject* obj, double defaultValue=NAN)
{
    if(obj==NULL)
        return defaultValue;
    if(PyNumber_Check(obj)) {
        double value = PyFloat_AsDouble(obj);
        if(PyErr_Occurred())
            return defaultValue;
        return value;
    }
    PyObject* d = PyNumber_Float(obj);
    if(d) {
        double value = PyFloat_AsDouble(d);
        Py_DECREF(d);
        return value;
    }
    return defaultValue;
}

/// a convenience function for accessing an element of a PyArrayObject with the given data type
template<typename DataType>
inline DataType& pyArrayElem(void* arr, npy_intp ind)
{
    return *static_cast<DataType*>(PyArray_GETPTR1(static_cast<PyArrayObject*>(arr), ind));
}

/// same as above, but for a 2d array
template<typename DataType>
inline DataType& pyArrayElem(void* arr, npy_intp ind1, npy_intp ind2)
{
    return *static_cast<DataType*>(PyArray_GETPTR2(static_cast<PyArrayObject*>(arr), ind1, ind2));
}

/// convert a Python array of floats to std::vector, or return empty vector in case of error
std::vector<double> toDoubleArray(PyObject* obj)
{
    if(!obj)
        return std::vector<double>();
    PyObject *arr = PyArray_FROM_OTF(obj, NPY_DOUBLE, 0/*no special requirements*/);
    if(!arr || PyArray_NDIM((PyArrayObject*)arr) != 1) {
        Py_XDECREF(arr);
        return std::vector<double>();
    }
    int size = PyArray_DIM((PyArrayObject*)arr, 0);
    std::vector<double> vec(size);
    for(int i=0; i<size; i++)
        vec[i] = pyArrayElem<double>(arr, i);
    Py_DECREF(arr);
    return vec;
}

/// convert a C++ vector into a NumPy array
PyObject* toPyArray(const std::vector<double>& vec)
{
    npy_intp size = vec.size();
    PyObject* arr = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
    if(!arr)
        return arr;
    for(npy_intp i=0; i<size; i++)
        pyArrayElem<double>(arr, i) = vec[i];
    return arr;
}

/// convert a Python tuple or list into an array of borrowed PyObject* references
std::vector<PyObject*> toPyObjectArray(PyObject* obj)
{
    std::vector<PyObject*> result;
    if(!obj) return result;
    if(PyTuple_Check(obj)) {
        Py_ssize_t size = PyTuple_Size(obj);
        for(Py_ssize_t i=0; i<size; i++)
            result.push_back(PyTuple_GET_ITEM(obj, i));
    } else
    if(PyList_Check(obj)) {
        Py_ssize_t size = PyList_Size(obj);
        for(Py_ssize_t i=0; i<size; i++)
            result.push_back(PyList_GET_ITEM(obj, i));
    } else
        result.push_back(obj);  // return an array consisting of a single object
    return result;
}

/// convert a Python dictionary to its C++ analog
utils::KeyValueMap convertPyDictToKeyValueMap(PyObject* dict)
{
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    utils::KeyValueMap params;
    while (PyDict_Next(dict, &pos, &key, &value))
        params.set(toString(key), toString(value));
    return params;
}

/// check that the list of arguments provided to a Python function
/// contains only named args and no positional args
inline bool onlyNamedArgs(PyObject* args, PyObject* namedArgs)
{
    if((args!=NULL && PyTuple_Check(args) && PyTuple_Size(args)>0) ||
        namedArgs==NULL || !PyDict_Check(namedArgs) || PyDict_Size(namedArgs)==0)
    {
        PyErr_SetString(PyExc_ValueError, "Should only provide named arguments");
        return false;
    }
    return true;
}

/// find an item in the Python dictionary using case-insensitive key comparison
PyObject* getItemFromPyDict(PyObject* dict, const char* itemkey)
{
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(dict, &pos, &key, &value))
        if(utils::stringsEqual(toString(key), itemkey))
            return value;
    return NULL;
}

// forward declaration for a routine that constructs a Python cubic spline object
PyObject* createCubicSpline(const std::vector<double>& x, const std::vector<double>& y);

/// NumPy data type corresponding to the storage container type of additive models
static const int STORAGE_NUM_T = 
    sizeof(galaxymodel::StorageNumT) == sizeof(float)  ? NPY_FLOAT  :
    sizeof(galaxymodel::StorageNumT) == sizeof(double) ? NPY_DOUBLE : NPY_NOTYPE;
    
///@}
//  ------------------------------
/// \name  Unit handling routines
//  ------------------------------
///@{

/// internal working units (arbitrary!)
static const units::InternalUnits unit(2.7183 * units::Kpc, 3.1416 * units::Myr);

/// external units that are used in the calling code, set by the user,
/// (or remaining at default values (no conversion) if not set explicitly
static unique_ptr<const units::ExternalUnits> conv;

/// description of setUnits function
static const char* docstringSetUnits = 
    "Inform the library about the physical units that are used in Python code\n"
    "Arguments should be any three independent physical quantities that define "
    "'mass', 'length', 'velocity' or 'time' scales "
    "(note that the latter three are not all independent).\n"
    "Their values specify the units in terms of "
    "'Solar mass', 'Kiloparsec', 'km/s' and 'Megayear', correspondingly.\n"
    "Example: standard GADGET units are defined as\n"
    "    setUnits(mass=1e10, length=1, velocity=1)\n";

/// define the unit conversion
PyObject* setUnits(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"mass", "length", "velocity", "time", NULL};
    double mass = 0, length = 0, velocity = 0, time = 0;
    if(!onlyNamedArgs(args, namedArgs))
        return NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "|dddd", const_cast<char**>(keywords),
        &mass, &length, &velocity, &time))
        return NULL;
    if(mass<0 || length<0 || velocity<0 || time<0) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to setUnits()");
        return NULL;
    }
    if(length>0 && velocity>0 && time>0) {
        PyErr_SetString(PyExc_ValueError, 
            "You may not assign length, velocity and time units simultaneously");
        return NULL;
    }
    if(mass==0) {
        PyErr_SetString(PyExc_ValueError, "You must specify mass unit");
        return NULL;
    }
    if(length>0 && time>0)
        conv.reset(new units::ExternalUnits(unit,
            length*units::Kpc, length/time * units::Kpc/units::Myr, mass*units::Msun));
    else if(length>0 && velocity>0)
        conv.reset(new units::ExternalUnits(unit,
            length*units::Kpc, velocity*units::kms, mass*units::Msun));
    else if(time>0 && velocity>0)
        conv.reset(new units::ExternalUnits(unit,
            velocity*time * units::kms*units::Myr, velocity*units::kms, mass*units::Msun));
    else {
        PyErr_SetString(PyExc_ValueError,
            "You must specify exactly two out of three units: length, time and velocity");
        return NULL;
    }
    utils::msg(utils::VL_VERBOSE, "Agama",   // internal unit conversion factors not for public eye
        "length unit: "  +utils::toString(conv->lengthUnit)+", "
        "velocity unit: "+utils::toString(conv->velocityUnit)+", "
        "time unit: "    +utils::toString(conv->timeUnit)+", "
        "mass unit: "    +utils::toString(conv->massUnit));
    return Py_BuildValue("s", 
        ("Length unit: " +utils::toString(conv->lengthUnit   * unit.to_Kpc)+ " Kpc, "
        "velocity unit: "+utils::toString(conv->velocityUnit * unit.to_kms)+ " km/s, "
        "time unit: "    +utils::toString(conv->timeUnit     * unit.to_Myr)+ " Myr, "
        "mass unit: "    +utils::toString(conv->massUnit     * unit.to_Msun)+" Msun, "
        "gravitational constant: "+utils::toString(units::Grav *
            (conv->massUnit * unit.to_Msun * units::Msun) /
            pow_2(conv->velocityUnit * unit.to_kms * units::kms) /
            (conv->lengthUnit * unit.to_Kpc * units::Kpc) ) ).c_str());
}

/// description of resetUnits function
static const char* docstringResetUnits = 
    "Reset the unit conversion system to a trivial one "
    "(i.e., no conversion involved and all quantities are assumed to be in N-body units, "
    "with the gravitational constant equal to 1.\n"
    "Note that this is NOT equivalent to setUnits(mass=1, length=1, velocity=1).\n";

/// reset the unit conversion
PyObject* resetUnits(PyObject* /*self*/, PyObject* /*args*/)
{
    conv.reset(new units::ExternalUnits());
    Py_INCREF(Py_None);
    return Py_None;
}

/// helper function for converting position to internal units
inline coord::PosCar convertPos(const double input[]) {
    return coord::PosCar(
        input[0] * conv->lengthUnit,
        input[1] * conv->lengthUnit,
        input[2] * conv->lengthUnit);
}

/// helper function for converting position/velocity to internal units
inline coord::PosVelCar convertPosVel(const double input[]) {
    return coord::PosVelCar(
        input[0] * conv->lengthUnit,
        input[1] * conv->lengthUnit,
        input[2] * conv->lengthUnit,
        input[3] * conv->velocityUnit,
        input[4] * conv->velocityUnit,
        input[5] * conv->velocityUnit);
}

/// helper function for converting actions to internal units
inline actions::Actions convertActions(const double input[]) {
    return actions::Actions(
        input[0] * conv->lengthUnit * conv->velocityUnit,
        input[1] * conv->lengthUnit * conv->velocityUnit,
        input[2] * conv->lengthUnit * conv->velocityUnit);
}

/// helper function to convert position from internal units back to user units
inline void unconvertPos(const coord::PosCar& point, double dest[])
{
    dest[0] = point.x / conv->lengthUnit;
    dest[1] = point.y / conv->lengthUnit;
    dest[2] = point.z / conv->lengthUnit;
}

/// helper function to convert position/velocity from internal units back to user units
inline void unconvertPosVel(const coord::PosVelCar& point, double dest[])
{
    dest[0] = point.x / conv->lengthUnit;
    dest[1] = point.y / conv->lengthUnit;
    dest[2] = point.z / conv->lengthUnit;
    dest[3] = point.vx / conv->velocityUnit;
    dest[4] = point.vy / conv->velocityUnit;
    dest[5] = point.vz / conv->velocityUnit;
}

/// helper function to convert actions from internal units back to user units
inline void unconvertActions(const actions::Actions& act, double dest[])
{
    dest[0] = act.Jr   / (conv->lengthUnit * conv->velocityUnit);
    dest[1] = act.Jz   / (conv->lengthUnit * conv->velocityUnit);
    dest[2] = act.Jphi / (conv->lengthUnit * conv->velocityUnit);
}


///@}
//  --------------------------------------------------------------
/// \name  A truly general interface for evaluating some function
///        for some input data and storing its output somewhere
//  --------------------------------------------------------------
///@{

/// any function that evaluates something for a given object and an `input` array of floats,
/// and stores one or more values in the `result` array of floats
typedef void (*anyFunction) 
    (void* obj, const double input[], double *result);

/// anyFunction input type
enum INPUT_VALUE {
    INPUT_VALUE_SINGLE = 1,  ///< a single number
    INPUT_VALUE_TRIPLET= 3,  ///< three numbers
    INPUT_VALUE_SEXTET = 6   ///< six numbers
};

/// anyFunction output type; numerical value is arbitrary
enum OUTPUT_VALUE {
    OUTPUT_VALUE_SINGLE              = 1,  ///< scalar value
    OUTPUT_VALUE_TRIPLET             = 3,  ///< a triplet of numbers
    OUTPUT_VALUE_SEXTET              = 6,  ///< a sextet of numbers
    OUTPUT_VALUE_SINGLE_AND_SINGLE   = 11, ///< a single number and another single number
    OUTPUT_VALUE_SINGLE_AND_TRIPLET  = 13, ///< a single number and a triplet
    OUTPUT_VALUE_SINGLE_AND_SEXTET   = 16, ///< a single number and a sextet
    OUTPUT_VALUE_TRIPLET_AND_TRIPLET = 33, ///< a triplet and another triplet -- two separate arrays
    OUTPUT_VALUE_TRIPLET_AND_SEXTET  = 36, ///< a triplet and a sextet
    OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET   = 136, ///< all wonders at once
    OUTPUT_VALUE_TRIPLET_AND_TRIPLET_AND_TRIPLET = 333, ///< another all-in-one
};

/// size of input array for a single point
template<int numArgs>
size_t inputLength();

/// parse a list of numArgs floating-point arguments for a Python function, 
/// and store them in inputArray[]; return 1 on success, 0 on failure 
template<int numArgs>
int parseTuple(PyObject* args, double inputArray[]);

/// check that the input array is of right dimensions, and return its length
template<int numArgs>
inline int parseArray(PyArrayObject* arr)
{
    if(PyArray_NDIM(arr) == 2 && PyArray_DIM(arr, 1) == numArgs)
        return PyArray_DIM(arr, 0);
    else
        return 0;
}

/// error message for an input array of incorrect size
template<int numArgs>
const char* errStrInvalidArrayDim();

/// error message for an input array of incorrect size or an invalid list of arguments
template<int numArgs>
const char* errStrInvalidInput();

/// size of output array for a single point
template<int numOutput>
size_t outputLength();

/// construct an output tuple containing the given result data computed for a single input point
template<int numOutput>
PyObject* formatTuple(const double result[]);

/// construct an output array, or several arrays, that will store the output for many input points
template<int numOutput>
PyObject* allocOutputArr(int size);

/// store the 'result' data computed for a single input point in an output array 'resultObj' at 'index'
template<int numOutput>
void formatOutputArr(const double result[], const int index, PyObject* resultObj);

// ---- template instantiations for input parameters ----

template<> inline size_t inputLength<INPUT_VALUE_SINGLE> () {return 1;}
template<> inline size_t inputLength<INPUT_VALUE_TRIPLET>() {return 3;}
template<> inline size_t inputLength<INPUT_VALUE_SEXTET> () {return 6;}

template<> inline int parseTuple<INPUT_VALUE_SINGLE>(PyObject* args, double input[]) {
    if(PyTuple_Check(args) && PyTuple_Size(args)==1)
        args = PyTuple_GET_ITEM(args, 0);
    input[0] = PyFloat_AsDouble(args);
    return PyErr_Occurred() ? 0 : 1;
}
template<> inline int parseTuple<INPUT_VALUE_TRIPLET>(PyObject* args, double input[]) {
    return PyArg_ParseTuple(args, "ddd", &input[0], &input[1], &input[2]);
}
template<> inline int parseTuple<INPUT_VALUE_SEXTET>(PyObject* args, double input[]) {
    return PyArg_ParseTuple(args, "dddddd",
        &input[0], &input[1], &input[2], &input[3], &input[4], &input[5]);
}

template<> inline int parseArray<INPUT_VALUE_SINGLE>(PyArrayObject* arr)
{
    if(PyArray_NDIM(arr) == 1)
        return PyArray_DIM(arr, 0);
    else
        return 0;
}

template<> inline const char* errStrInvalidArrayDim<INPUT_VALUE_SINGLE>() {
    return "Input does not contain a valid one-dimensional array";
}
template<> inline const char* errStrInvalidArrayDim<INPUT_VALUE_TRIPLET>() {
    return "Input does not contain a valid Nx3 array";
}
template<> inline const char* errStrInvalidArrayDim<INPUT_VALUE_SEXTET>() {
    return "Input does not contain a valid Nx6 array";
}

template<> inline const char* errStrInvalidInput<INPUT_VALUE_SINGLE>() {
    return "Input does not contain valid data (either a single number or a one-dimensional array)";
}
template<> inline const char* errStrInvalidInput<INPUT_VALUE_TRIPLET>() {
    return "Input does not contain valid data (either 3 numbers for a single point or a Nx3 array)";
}
template<> inline const char* errStrInvalidInput<INPUT_VALUE_SEXTET>() {
    return "Input does not contain valid data (either 6 numbers for a single point or a Nx6 array)";
}

// ---- template instantiations for output parameters ----

template<> inline size_t outputLength<OUTPUT_VALUE_SINGLE>()  {return 1;}
template<> inline size_t outputLength<OUTPUT_VALUE_TRIPLET>() {return 3;}
template<> inline size_t outputLength<OUTPUT_VALUE_SEXTET>()  {return 6;}
template<> inline size_t outputLength<OUTPUT_VALUE_SINGLE_AND_SINGLE>()   {return 2;}
template<> inline size_t outputLength<OUTPUT_VALUE_SINGLE_AND_TRIPLET>()  {return 4;}
template<> inline size_t outputLength<OUTPUT_VALUE_SINGLE_AND_SEXTET>()   {return 7;}
template<> inline size_t outputLength<OUTPUT_VALUE_TRIPLET_AND_TRIPLET>() {return 6;}
template<> inline size_t outputLength<OUTPUT_VALUE_TRIPLET_AND_SEXTET>()  {return 9;}
template<> inline size_t outputLength<OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET>()   {return 10;}
template<> inline size_t outputLength<OUTPUT_VALUE_TRIPLET_AND_TRIPLET_AND_TRIPLET>() {return 9;}

template<> inline PyObject* formatTuple<OUTPUT_VALUE_SINGLE>(const double result[]) {
    return Py_BuildValue("d", result[0]);
}
template<> inline PyObject* formatTuple<OUTPUT_VALUE_TRIPLET>(const double result[]) {
    return Py_BuildValue("ddd", result[0], result[1], result[2]);
}
template<> inline PyObject* formatTuple<OUTPUT_VALUE_SEXTET>(const double result[]) {
    return Py_BuildValue("dddddd",
        result[0], result[1], result[2], result[3], result[4], result[5]);
}
template<> inline PyObject* formatTuple<OUTPUT_VALUE_SINGLE_AND_SINGLE>(const double result[]) {
    return Py_BuildValue("(dd)", result[0], result[1]);
}
template<> inline PyObject* formatTuple<OUTPUT_VALUE_SINGLE_AND_TRIPLET>(const double result[]) {
    return Py_BuildValue("d(ddd)", result[0], result[1], result[2], result[3]);
}
template<> inline PyObject* formatTuple<OUTPUT_VALUE_SINGLE_AND_SEXTET>(const double result[]) {
    return Py_BuildValue("d(dddddd)", result[0],
        result[1], result[2], result[3], result[4], result[5], result[6]);
}
template<> inline PyObject* formatTuple<OUTPUT_VALUE_TRIPLET_AND_TRIPLET>(const double result[]) {
    return Py_BuildValue("(ddd)(ddd)", result[0], result[1], result[2],
        result[3], result[4], result[5]);
}
template<> inline PyObject* formatTuple<OUTPUT_VALUE_TRIPLET_AND_SEXTET>(const double result[]) {
    return Py_BuildValue("(ddd)(dddddd)", result[0], result[1], result[2],
        result[3], result[4], result[5], result[6], result[7], result[8]);
}
template<> inline PyObject* formatTuple<OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET>(
    const double result[])
{
    return Py_BuildValue("d(ddd)(dddddd)", result[0], result[1], result[2], result[3],
        result[4], result[5], result[6], result[7], result[8], result[9]);
}
template<> inline PyObject* formatTuple<OUTPUT_VALUE_TRIPLET_AND_TRIPLET_AND_TRIPLET>(
    const double result[])
{
    return Py_BuildValue("(ddd)(ddd)(ddd)", result[0], result[1], result[2],
        result[3], result[4], result[5], result[6], result[7], result[8]);
}
    
template<> inline PyObject* allocOutputArr<OUTPUT_VALUE_SINGLE>(int size) {
    npy_intp dims[] = {size};
    return PyArray_SimpleNew(1, dims, NPY_DOUBLE);
}
template<> inline PyObject* allocOutputArr<OUTPUT_VALUE_TRIPLET>(int size) {
    npy_intp dims[] = {size, 3};
    return PyArray_SimpleNew(2, dims, NPY_DOUBLE);
}
template<> inline PyObject* allocOutputArr<OUTPUT_VALUE_SEXTET>(int size) {
    npy_intp dims[] = {size, 6};
    return PyArray_SimpleNew(2, dims, NPY_DOUBLE);
}
template<> inline PyObject* allocOutputArr<OUTPUT_VALUE_SINGLE_AND_SINGLE>(int size) {
    npy_intp dims[] = {size};
    PyObject* arr1 = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    return Py_BuildValue("NN", arr1, arr2);
}
template<> inline PyObject* allocOutputArr<OUTPUT_VALUE_SINGLE_AND_TRIPLET>(int size) {
    npy_intp dims1[] = {size};
    npy_intp dims2[] = {size, 3};
    PyObject* arr1 = PyArray_SimpleNew(1, dims1, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(2, dims2, NPY_DOUBLE);
    return Py_BuildValue("NN", arr1, arr2);
}
template<> inline PyObject* allocOutputArr<OUTPUT_VALUE_SINGLE_AND_SEXTET>(int size) {
    npy_intp dims1[] = {size};
    npy_intp dims2[] = {size, 6};
    PyObject* arr1 = PyArray_SimpleNew(1, dims1, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(2, dims2, NPY_DOUBLE);
    return Py_BuildValue("NN", arr1, arr2);
}
template<> inline PyObject* allocOutputArr<OUTPUT_VALUE_TRIPLET_AND_TRIPLET>(int size) {
    npy_intp dims[] = {size, 3};
    PyObject* arr1 = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    return Py_BuildValue("NN", arr1, arr2);
}
template<> inline PyObject* allocOutputArr<OUTPUT_VALUE_TRIPLET_AND_SEXTET>(int size) {
    npy_intp dims1[] = {size, 3};
    npy_intp dims2[] = {size, 6};
    PyObject* arr1 = PyArray_SimpleNew(2, dims1, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(2, dims2, NPY_DOUBLE);
    return Py_BuildValue("NN", arr1, arr2);
}
template<> inline PyObject* allocOutputArr<OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET>(int size) {
    npy_intp dims1[] = {size};
    npy_intp dims2[] = {size, 3};
    npy_intp dims3[] = {size, 6};
    PyObject* arr1 = PyArray_SimpleNew(1, dims1, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(2, dims2, NPY_DOUBLE);
    PyObject* arr3 = PyArray_SimpleNew(2, dims3, NPY_DOUBLE);
    return Py_BuildValue("NNN", arr1, arr2, arr3);
}
template<> inline PyObject* allocOutputArr<OUTPUT_VALUE_TRIPLET_AND_TRIPLET_AND_TRIPLET>(int size) {
    npy_intp dims[] = {size, 3};
    PyObject* arr1 = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    PyObject* arr3 = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    return Py_BuildValue("NNN", arr1, arr2, arr3);
}

template<> inline void formatOutputArr<OUTPUT_VALUE_SINGLE>(
    const double result[], const int index, PyObject* resultObj) 
{
    pyArrayElem<double>(resultObj, index) = result[0];
}
template<> inline void formatOutputArr<OUTPUT_VALUE_TRIPLET>(
    const double result[], const int index, PyObject* resultObj) 
{
    for(int d=0; d<3; d++)
        pyArrayElem<double>(resultObj, index, d) = result[d];
}
template<> inline void formatOutputArr<OUTPUT_VALUE_SEXTET>(
    const double result[], const int index, PyObject* resultObj) 
{
    for(int d=0; d<6; d++)
        pyArrayElem<double>(resultObj, index, d) = result[d];
}
template<> inline void formatOutputArr<OUTPUT_VALUE_SINGLE_AND_SINGLE>(
    const double result[], const int index, PyObject* resultObj)
{
    pyArrayElem<double>(PyTuple_GET_ITEM(resultObj, 0), index) = result[0];
    pyArrayElem<double>(PyTuple_GET_ITEM(resultObj, 1), index) = result[1];
}
template<> inline void formatOutputArr<OUTPUT_VALUE_SINGLE_AND_TRIPLET>(
    const double result[], const int index, PyObject* resultObj)
{
    pyArrayElem<double>(PyTuple_GET_ITEM(resultObj, 0), index) = result[0];
    PyObject* arr2 = PyTuple_GET_ITEM(resultObj, 1);
    for(int d=0; d<3; d++)
        pyArrayElem<double>(arr2, index, d) = result[d+1];
}
template<> inline void formatOutputArr<OUTPUT_VALUE_SINGLE_AND_SEXTET>(
    const double result[], const int index, PyObject* resultObj)
{
    pyArrayElem<double>(PyTuple_GET_ITEM(resultObj, 0), index) = result[0];
    PyObject* arr2 = PyTuple_GET_ITEM(resultObj, 1);
    for(int d=0; d<6; d++)
        pyArrayElem<double>(arr2, index, d) = result[d+1];
}
template<> inline void formatOutputArr<OUTPUT_VALUE_TRIPLET_AND_TRIPLET>(
    const double result[], const int index, PyObject* resultObj)
{
    PyObject* arr1 = PyTuple_GET_ITEM(resultObj, 0);
    PyObject* arr2 = PyTuple_GET_ITEM(resultObj, 1);
    for(int d=0; d<3; d++) {
        pyArrayElem<double>(arr1, index, d) = result[d];
        pyArrayElem<double>(arr2, index, d) = result[d+3];
    }
}
template<> inline void formatOutputArr<OUTPUT_VALUE_TRIPLET_AND_SEXTET>(
    const double result[], const int index, PyObject* resultObj)
{
    PyObject* arr1 = PyTuple_GET_ITEM(resultObj, 0);
    PyObject* arr2 = PyTuple_GET_ITEM(resultObj, 1);
    for(int d=0; d<3; d++)
        pyArrayElem<double>(arr1, index, d) = result[d];
    for(int d=0; d<6; d++)
        pyArrayElem<double>(arr2, index, d) = result[d+3];
}
template<> inline void formatOutputArr<OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET>(
    const double result[], const int index, PyObject* resultObj)
{
    pyArrayElem<double>(PyTuple_GET_ITEM(resultObj, 0), index) = result[0];
    PyObject* arr1 = PyTuple_GET_ITEM(resultObj, 1);
    PyObject* arr2 = PyTuple_GET_ITEM(resultObj, 2);
    for(int d=0; d<3; d++)
        pyArrayElem<double>(arr1, index, d) = result[d+1];
    for(int d=0; d<6; d++)
        pyArrayElem<double>(arr2, index, d) = result[d+4];
}
template<> inline void formatOutputArr<OUTPUT_VALUE_TRIPLET_AND_TRIPLET_AND_TRIPLET>(
    const double result[], const int index, PyObject* resultObj)
{
    PyObject* arr1 = PyTuple_GET_ITEM(resultObj, 0);
    PyObject* arr2 = PyTuple_GET_ITEM(resultObj, 1);
    PyObject* arr3 = PyTuple_GET_ITEM(resultObj, 2);
    for(int d=0; d<3; d++) {
        pyArrayElem<double>(arr1, index, d) = result[d];
        pyArrayElem<double>(arr2, index, d) = result[d+3];
        pyArrayElem<double>(arr3, index, d) = result[d+6];
    }
}

/** A general function that computes something for one or many input points.
    \tparam numArgs  is the size of array that contains the value of a single input point.
    \tparam numOutput is the identifier (not literally the size) of output data format
    for a single input point: it may be a single number, an array of floats, or even several arrays.
    \param[in]  fnc  is the function pointer to the routine that actually computes something,
    taking a pointer to an instance of Python object, an array of floats as the input point,
    and producing another array of floats as the output.
    \param[in] params  is the pointer to auxiliary parameters that is passed to the 'fnc' routine
    \param[in] args  is the arguments of the function call: it may be a sequence of numArg floats
    that represents a single input point, or a 1d array of the same length and same meaning,
    or a 2d array of dimensions N * numArgs, representing N input points.
    \returns  the result of applying 'fnc' to one or many input points, in the form determined
    both by the number of input points, and the output data format.
    The output for a single point may be a sequence of numbers (tuple or 1d array),
    or several such arrays forming a tuple (e.g., [ [1,2,3], [1,2,3,4,5,6] ]).
    The output for an array of input points would be one or several 2d arrays of length N and
    shape determined by the output format, i.e., for the above example it would be ([N,3], [N,6]).
*/
template<int numArgs, int numOutput>
PyObject* callAnyFunctionOnArray(void* params, PyObject* args, anyFunction fnc)
{
    if(args==NULL) {
        PyErr_SetString(PyExc_ValueError, "No input data provided");
        return NULL;
    }
    try{
        double input[inputLength<numArgs>()];
        double output[outputLength<numOutput>()];
        if(parseTuple<numArgs>(args, input)) {  // one point
            fnc(params, input, output);
            return formatTuple<numOutput>(output);
        }
        PyErr_Clear();  // clear error if the argument list is not a tuple of a proper type
        PyObject* obj=NULL;
        if(PyArray_Check(args) || PyList_Check(args))
            obj = args;
        else if(PyTuple_Check(args)) {
            if(PyTuple_Size(args)==1)
                obj = PyTuple_GET_ITEM(args, 0);
            else if(numArgs == INPUT_VALUE_SINGLE)
                obj = args;   // the entire tuple of arguments is the input array
        }
        if(obj) {
            PyArrayObject *arr = (PyArrayObject*) PyArray_FROM_OTF(obj,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            if(arr == NULL) {
                PyErr_SetString(PyExc_ValueError, "Input does not contain a valid array");
                return NULL;
            }
            int numpt = 0;
            if(PyArray_NDIM(arr) == 1 && PyArray_DIM(arr, 0) == numArgs)
            {   // 1d array of length numArgs - a single point
                fnc(params, &pyArrayElem<double>(arr, 0), output);
                Py_DECREF(arr);
                return formatTuple<numOutput>(output);
            }
            // check the shape of input array
            numpt = parseArray<numArgs>(arr);
            if(numpt == 0) {
                PyErr_SetString(PyExc_ValueError, errStrInvalidArrayDim<numArgs>());
                Py_DECREF(arr);
                return NULL;
            }
            keyboardInterruptTriggered = 0;
            defaultKeyboardInterruptHandler = signal(SIGINT, customKeyboardInterruptHandler);
            // allocate an appropriate output object
            PyObject* outputObj = allocOutputArr<numOutput>(numpt);
            // loop over input array
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i=0; i<numpt; i++) {
                if(keyboardInterruptTriggered) continue;
                double local_output[outputLength<numOutput>()];  // separate variable in each thread
                fnc(params, &pyArrayElem<double>(arr, i, 0), local_output);
                formatOutputArr<numOutput>(local_output, i, outputObj);
            }
            Py_DECREF(arr);
            signal(SIGINT, defaultKeyboardInterruptHandler);
            if(keyboardInterruptTriggered) {
                PyErr_SetObject(PyExc_KeyboardInterrupt, NULL);
                return NULL;
            }
            return outputObj;
        }
        PyErr_SetString(PyExc_ValueError, errStrInvalidInput<numArgs>());
        return NULL;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, (std::string("Exception occurred: ")+e.what()).c_str());
        signal(SIGINT, defaultKeyboardInterruptHandler);
        return NULL;
    }
}


///@}
//  ---------------------
/// \name  Density class
//  ---------------------
///@{

/// \cond INTERNAL_DOCS
/// Python type corresponding to Density class
typedef struct {
    PyObject_HEAD
    potential::PtrDensity dens;
} DensityObject;
/// \endcond

void Density_dealloc(DensityObject* self)
{
    if(self->dens)
        utils::msg(utils::VL_VERBOSE, "Agama", "Deleted "+std::string(self->dens->name())+
            " density at "+utils::toString(self->dens.get()));
    else
        utils::msg(utils::VL_VERBOSE, "Agama", "Deleted an empty density");
    self->dens.reset();
    self->ob_type->tp_free(self);
}

/// common fragment of docstring for Density and Potential classes
#define DOCSTRING_DENSITY_PARAMS \
    "  mass=...   total mass of the model, if applicable.\n" \
    "  scaleRadius=...   scale radius of the model (if applicable).\n" \
    "  scaleHeight=...   scale height of the model (currently applicable to " \
    "Dehnen, MiyamotoNagai and DiskDensity).\n" \
    "  p=...   or  axisRatioY=...   axis ratio y/x, i.e., intermediate to long axis " \
    "(applicable to triaxial potential models such as Dehnen and Ferrers, " \
    "and to SpheroidDensity and SersicDensity models).\n" \
    "  q=...   or  axisRatioZ=...   short to long axis (z/x).\n" \
    "  gamma=...  central cusp slope (applicable for Dehnen and SpheroidDensity).\n" \
    "  beta=...   outer density slope (SpheroidDensity).\n" \
    "  alpha=...  strength of transition from the inner to the outer slopes (SpheroidDensity).\n" \
    "  sersicIndex=...   profile shape parameter 'n' (SersicDensity).\n" \
    "  innerCutoffRadius=...   radius of inner hole (DiskDensity).\n" \
    "  outerCutoffRadius=...   radius of outer exponential cutoff (SpheroidDensity).\n" \
    "  cutoffStrength=...   strength of outer exponential cutoff  (SpheroidDensity).\n" \
    "  surfaceDensity=...   central surface density (DiskDensity or SersicDensity).\n" \
    "  densityNorm=...   normalization of density profile for SpheroidDensity.\n"

/// description of Density class
static const char* docstringDensity =
    "Density is a class representing a variety of density profiles "
    "that do not necessarily have a corresponding potential defined.\n"
    "An instance of Density class is constructed using the following keyword arguments:\n"
    "  type='...' or density='...'   the name of density profile (required), can be one of the following:\n"
    "    Denhen, Plummer, OblatePerfectEllipsoid, Ferrers, MiyamotoNagai, "
    "NFW, DiskDensity, SpheroidDensity, SersicDensity.\n"
    DOCSTRING_DENSITY_PARAMS
    "Most of these parameters have reasonable default values.\n"
    "Alternatively, one may construct a spherically-symmetric density model from a cumulative "
    "mass profile by providing a single argument\n"
    "  cumulmass=...  which should contain a table with two columns: radius and enclosed mass, "
    "both strictly positive and monotonically increasing.\n"
    "One may also load density expansion coefficients that weree previously written to a text file "
    "using the `export()` method, by providing the file name as an argument.\n"
    "Finally, one may create a composite density from several Density objects by providing them as "
    "unnamed arguments to the constructor:  densum = Density(den1, den2, den3)\n\n"
    "An instance of Potential class may be used in all contexts when a Density object is required;\n"
    "moreover, an arbitrary Python object with a method 'density(x,y,z)' that returns a single value "
    "may also be used in these contexts (i.e., an object presenting a Density interface).";

// extract a pointer to C++ Density class from a Python object, or return an empty pointer on error.
// Declared here, implemented after the PotentialObject definition becomes available.
potential::PtrDensity getDensity(PyObject* dens_obj, coord::SymmetryType sym=coord::ST_TRIAXIAL);

// extract a pointer to C++ Potential class from a Python object, or return an empty pointer on error
// (also a forward declaration, the function will be defined later)
potential::PtrPotential getPotential(PyObject* pot_obj);

// create a Python Density object and initialize it with an existing instance of C++ density class
PyObject* createDensityObject(const potential::PtrDensity& dens);

// create a Python Potential object and initialize it with an existing instance of C++ potential class
PyObject* createPotentialObject(const potential::PtrPotential& pot);

/// attempt to construct a composite density from a tuple of Density objects
potential::PtrDensity Density_initFromTuple(PyObject* tuple)
{
    // if we have one string parameter, it could be the file name
    if(PyTuple_Size(tuple) == 1 && PyString_Check(PyTuple_GET_ITEM(tuple, 0)))
        return potential::readDensity(PyString_AsString(PyTuple_GET_ITEM(tuple, 0)), *conv);
    std::vector<potential::PtrDensity> components;
    for(Py_ssize_t i=0; i<PyTuple_Size(tuple); i++) {
        potential::PtrDensity comp = getDensity(PyTuple_GET_ITEM(tuple, i));
        if(!comp)
            throw std::invalid_argument("Tuple should contain only valid Density objects "
                "or functions providing that interface");
        components.push_back(comp);
    }
    return potential::PtrDensity(new potential::CompositeDensity(components));
}

/// attempt to construct a spherically-symmetric density from a cumulative mass profile
potential::PtrDensity Density_initFromCumulMass(PyObject* cumulMass)
{
    PyArrayObject *cumulMassArr = (PyArrayObject*)
        PyArray_FROM_OTF(cumulMass, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(cumulMassArr == NULL || PyArray_NDIM(cumulMassArr) != 2 || PyArray_DIM(cumulMassArr, 1) != 2) {
        Py_XDECREF(cumulMassArr);
        throw std::invalid_argument("'cumulmass' does not contain a valid Nx2 array");
    }
    int size = PyArray_DIM(cumulMassArr, 0);
    std::vector<double> radius(size), mass(size);
    for(int i=0; i<size; i++) {
        radius[i] = pyArrayElem<double>(cumulMassArr, i, 0) * conv->lengthUnit;
        mass  [i] = pyArrayElem<double>(cumulMassArr, i, 1) * conv->massUnit;
    }
    Py_DECREF(cumulMassArr);
    std::vector<double> dens = galaxymodel::densityFromCumulativeMass(radius, mass);
    return potential::PtrDensity(new potential::DensitySphericalHarmonic(
        radius, std::vector<std::vector<double> >(1, dens)));
}

/// attempt to construct a density from key=value parameters
potential::PtrDensity Density_initFromDict(PyObject* namedArgs)
{
    PyObject* cumulmass = getItemFromPyDict(namedArgs, "cumulmass");
    if(cumulmass)
        return Density_initFromCumulMass(cumulmass);
    utils::KeyValueMap params = convertPyDictToKeyValueMap(namedArgs);
    // for convenience, may specify the type of density model in type="..." argument
    if(params.contains("type") && !params.contains("density"))
        params.set("density", params.getString("type"));
    if(!params.contains("density") && !params.contains("file"))
        throw std::invalid_argument("Should provide the name of density model "
            "in type='...' or density='...', or the file name to load in file='...' arguments");
    return potential::createDensity(params, *conv);
}
    
/// constructor of Density class
int Density_init(DensityObject* self, PyObject* args, PyObject* namedArgs)
{
    try{
        // check if we have only a tuple of density components as arguments
        if(args!=NULL && PyTuple_Check(args) && PyTuple_Size(args)>0 &&
            (namedArgs==NULL || PyDict_Size(namedArgs)==0))
            self->dens = Density_initFromTuple(args);
        else if(namedArgs!=NULL && PyDict_Check(namedArgs) && PyDict_Size(namedArgs)>0)
            self->dens = Density_initFromDict(namedArgs);
        else {
            throw std::invalid_argument(
                "Invalid parameters passed to the constructor, type help(Density) for details");
        }
        assert(self->dens);
        utils::msg(utils::VL_VERBOSE, "Agama", "Created "+std::string(self->dens->name())+
            " density at "+utils::toString(self->dens.get()));
        return 0;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, (std::string("Error in creating density: ")+e.what()).c_str());
        return -1;
    }
}

void fncDensity_density(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    result[0] = ((DensityObject*)obj)->dens->density(point)
        / (conv->massUnit / pow_3(conv->lengthUnit));  // unit of density is M/L^3
}
PyObject* Density_density(PyObject* self, PyObject* args) {
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE>
        (self, args, fncDensity_density);
}

PyObject* Density_totalMass(PyObject* self)
{
    try{
        return Py_BuildValue("d", ((DensityObject*)self)->dens->totalMass() / conv->massUnit);
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError,
            (std::string("Error in Density.totalMass(): ")+e.what()).c_str());
        return NULL;
    }
}

PyObject* Density_export(PyObject* self, PyObject* args)
{
    const char* filename=NULL;
    if(!PyArg_ParseTuple(args, "s", &filename))
        return NULL;
    try{
        writeDensity(filename, *((DensityObject*)self)->dens, *conv);
        Py_INCREF(Py_None);
        return Py_None;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, (std::string("Error writing file: ")+e.what()).c_str());
        return NULL;
    }
}

/// shared between Density and Potential classes
PyObject* sampleDensity(const potential::BaseDensity& dens, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"n", "potential", "beta", "kappa", NULL};
    int numPoints=0;
    PyObject* pot_obj=NULL;
    double beta=NAN, kappa=NAN;  // undefined by default, if no argument is provided
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "i|Odd", const_cast<char**>(keywords),
        &numPoints, &pot_obj, &beta, &kappa))
    {
        //PyErr_SetString(PyExc_ValueError,
        //    "sample() takes at least one integer argument (the number of particles)");
        return NULL;
    }
    if(numPoints<=0) {
        PyErr_SetString(PyExc_TypeError, "number of sampling points 'n' must be positive");
        return NULL;
    }
    potential::PtrPotential pot = getPotential(pot_obj);  // if not NULL, will assign velocity as well
    if(pot_obj!=NULL && !pot) {
        PyErr_SetString(PyExc_TypeError,
            "'potential' must be a valid instance of Potential class");
        return NULL;
    }
    try{
        // do the sampling of the density profile
        particles::ParticleArray<coord::PosCyl> points =
            galaxymodel::generateDensitySamples(dens, numPoints);

        // assign the velocities if needed
        particles::ParticleArrayCar pointsvel;
        if(pot)
            pointsvel = galaxymodel::assignVelocity(points, dens, *pot, beta, kappa);

        // convert output to NumPy array
        numPoints = points.size();
        npy_intp dims[] = {numPoints, pot? 6 : 3};  // either position or position+velocity
        PyArrayObject* point_arr = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        PyArrayObject* mass_arr  = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        for(int i=0; i<numPoints; i++) {
            if(pot)
                unconvertPosVel(pointsvel.point(i), &pyArrayElem<double>(point_arr, i, 0));
            else
                unconvertPos(coord::toPosCar(points.point(i)), &pyArrayElem<double>(point_arr, i, 0));
            pyArrayElem<double>(mass_arr, i) = points.mass(i) / conv->massUnit;
        }
        return Py_BuildValue("NN", point_arr, mass_arr);
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, 
            (std::string("Error in sample(): ")+e.what()).c_str());
        return NULL;
    }
}

PyObject* Density_sample(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    return sampleDensity(*((DensityObject*)self)->dens, args, namedArgs);
}

PyObject* Density_name(PyObject* self)
{
    const char* name = ((DensityObject*)self)->dens->name();
    if(name == potential::CompositeDensity::myName()) {
        try{
            const potential::CompositeDensity& dens =
                dynamic_cast<const potential::CompositeDensity&>(*((DensityObject*)self)->dens);
            std::string tmp = std::string(name) + ": ";
            for(unsigned int i=0; i<dens.size(); i++) {
                if(i>0) tmp += ", ";
                tmp += dens.component(i)->name();
            }
            return Py_BuildValue("s", tmp.c_str());
        }
        catch(std::exception& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        }
    } else
        return Py_BuildValue("s", name);
}

PyObject* Density_elem(PyObject* self, Py_ssize_t index)
{
    try{
        const potential::CompositeDensity& dens =
            dynamic_cast<const potential::CompositeDensity&>(*((DensityObject*)self)->dens);
        if(index<0 || index >= (Py_ssize_t)dens.size()) {
            PyErr_SetString(PyExc_IndexError, "Density component index out of range");
            return NULL;
        }
        return createDensityObject(dens.component(index));
    }
    catch(std::exception&) {
        PyErr_SetString(PyExc_TypeError, "Density is not a composite");
        return NULL;
    }
}

Py_ssize_t Density_len(PyObject* self)
{
    try{
        return dynamic_cast<const potential::CompositeDensity&>(*((DensityObject*)self)->dens).size();
    }
    catch(std::exception&) {
        PyErr_SetString(PyExc_TypeError, "Density is not a composite");
        return -1;
    }
}

static PySequenceMethods Density_sequence_methods = {
    Density_len, 0, 0, Density_elem,
};

static PyMethodDef Density_methods[] = {
    { "name", (PyCFunction)Density_name, METH_NOARGS,
      "Return the name of the density model\n"
      "No arguments\n"
      "Returns: string" },
    { "density", Density_density, METH_VARARGS, 
      "Compute density at a given point or array of points\n"
      "Arguments: a triplet of floats (x,y,z) or a 2d Nx3 array\n"
      "Returns: float or array of floats" },
    { "export", Density_export, METH_VARARGS,
      "Export density expansion coefficients to a text file\n"
      "Arguments: filename (string)\n"
      "Returns: none" },
    { "sample", (PyCFunction)Density_sample, METH_VARARGS | METH_KEYWORDS, 
      "Sample the density profile with N point masses (assign particle coordinates and masses), "
      "and optionally assign velocities using one of three possible methods: "
      "Eddington, spherical or axisymmetric Jeans equations.\n"
      "The choice of method depends on the provided optional parameters: "
      "Eddington requires only the potential, "
      "spherical Jeans -- additionally the velocity anisotropy coefficient 'beta', "
      "axisymmetric Jeans -- additionally the rotation parameter 'kappa', "
      "and 'beta' has a different meaning in this case, but still required.\n"
      "Arguments: \n"
      "  n - the number of particles (required)\n"
      "  potential - an instance of Potential class providing the total potential "
      "used to assign particle velocities (optional - if not provided, assign only the coordinates)\n"
      "  beta - velocity anisotropy coefficient for the spherical or axisymmetric Jeans models; "
      "in the spherical case it specifies the ratio between radial and tangential velocity dispersions, "
      "assumed to be constant with radius: beta = 1 - sigma_t^2 / (2 sigma_r^2) as usual, "
      "and in the axisymmetric case - the ratio between R and z dispersions, "
      "again assumed to be constant: beta = 1 - sigma_z^2 / sigma_R^2. "
      "If not provided, this means using the Eddington method (spherical isotropic velocity "
      "distribution function, but not necessarily Gaussian as assumed in the Jeans approach).\n"
      "  kappa - the degree of net rotation (controls the decomposition of <v_phi^2> - total second "
      "moment of azimuthal velocity - into the mean streaming velocity <v_phi> and the velocity "
      "dispersion sigma_phi). kappa=0 means no net rotation, kappa=1 corresponds to sigma_phi=sigma_R). "
      "If this argument is provided, this triggers the use of the axisymmetric Jeans method.\n"
      "Returns: a tuple of two arrays: "
      "a 2d array of size Nx3 (in case of positions only) or Nx6 (in case of velocity assignment), "
      "and a 1d array of N point masses." },
    { "totalMass", (PyCFunction)Density_totalMass, METH_NOARGS,
      "Return the total mass of the density model\n"
      "No arguments\n"
      "Returns: float number" },
    { NULL }
};

static PyTypeObject DensityType = {
    PyObject_HEAD_INIT(NULL)
    0, "agama.Density",
    sizeof(DensityObject), 0, (destructor)Density_dealloc,
    0, 0, 0, 0, 0, 0, &Density_sequence_methods, 0, 0, 0, Density_name, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringDensity,
    0, 0, 0, 0, 0, 0, Density_methods, 0, 0, 0, 0, 0, 0, 0,
    (initproc)Density_init
};

/// create a Python Density object and initialize it with an existing instance of C++ density class
PyObject* createDensityObject(const potential::PtrDensity& dens)
{
    DensityObject* dens_obj = PyObject_New(DensityObject, &DensityType);
    if(!dens_obj)
        return NULL;

    // this is a DIRTY HACK!!! we have allocated a new instance of Python class,
    // but have not initialized its extra fields in any way, so they contain garbage.
    // We can't simply assign a new value to its 'dens' member variable,
    // because this is an object (smart pointer) with undefined state,
    // and it would attempt to deallocate its managed pointer before being assigned
    // a new value, which results in a crash.
    // Therefore, we use the "placement new" syntax to construct an empty smart pointer in-place,
    // in the already allocated chunk of memory (which is just the address of this member variable).
    // Note that we don't have these problems in the standard workflow when a Python object 
    // is allocated from Python, because the tp_alloc routine used in its default or custom tp_new
    // method fills the entire memory block of the corresponding struct with zeros,
    // which results in a correct initialization of both POD types and smart pointers
    // (perhaps accidentally, and would not be valid for more complex classes with virtual tables).
    new (&(dens_obj->dens)) potential::PtrDensity;
    // now we may safely assign a new value to the smart pointer
    dens_obj->dens = dens;
    utils::msg(utils::VL_VERBOSE, "Agama", "Created a Python wrapper for "+
        std::string(dens->name())+" density at "+utils::toString(dens.get()));
    return (PyObject*)dens_obj;
}

/// Helper class for providing a BaseDensity interface
/// to a Python function that returns density at one or several point
class DensityWrapper: public potential::BaseDensity{
    OmpDisabler ompDisabler;
    PyObject* fnc;
    coord::SymmetryType sym;
    std::string fncname;
public:
    DensityWrapper(PyObject* _fnc, coord::SymmetryType _sym): fnc(_fnc), sym(_sym)
    {
        Py_INCREF(fnc);
        fncname = toString(fnc);
        utils::msg(utils::VL_VERBOSE, "Agama",
            "Created a C++ density wrapper for Python function "+fncname);
    }
    ~DensityWrapper()
    {
        utils::msg(utils::VL_VERBOSE, "Agama",
            "Deleted a C++ density wrapper for Python function "+fncname);
        Py_DECREF(fnc);
    }
    virtual coord::SymmetryType symmetry() const { return sym; }
    virtual const char* name() const { return fncname.c_str(); };
    virtual double densityCyl(const coord::PosCyl &pos) const {
        return densityCar(toPosCar(pos)); }
    virtual double densitySph(const coord::PosSph &pos) const {
        return densityCar(toPosCar(pos)); }
    virtual double densityCar(const coord::PosCar &pos) const {
        double xyz[3];
        unconvertPos(pos, xyz);
        npy_intp dims[]  = {1, 3};
        PyObject* args   = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, xyz);
        PyObject* result = PyObject_CallFunctionObjArgs(fnc, args, NULL);
        Py_DECREF(args);
        double value;
        if(result == NULL) {
            PyErr_Print();
            throw std::runtime_error("Call to user-defined density function failed");
        }
        if(PyArray_Check(result))
            value = pyArrayElem<double>(result, 0);
        else if(PyNumber_Check(result))
            value = PyFloat_AsDouble(result);
        else {
            Py_DECREF(result);
            throw std::runtime_error("Invalid data type returned from user-defined density function");
        }
        Py_DECREF(result);
        return value * conv->massUnit / pow_3(conv->lengthUnit);
    }
};


///@}
//  -----------------------
/// \name  Potential class
//  -----------------------
///@{

/// \cond INTERNAL_DOCS
/// Python type corresponding to Potential class
typedef struct {
    PyObject_HEAD
    potential::PtrPotential pot;
} PotentialObject;
/// \endcond

void Potential_dealloc(PotentialObject* self)
{
    if(self->pot)
        utils::msg(utils::VL_VERBOSE, "Agama", "Deleted "+std::string(self->pot->name())+
        " potential at "+utils::toString(self->pot.get()));
    else
        utils::msg(utils::VL_VERBOSE, "Agama", "Deleted an empty potential");
    self->pot.reset();
    self->ob_type->tp_free(self);
}

/// pointer to the Potential type object (will be initialized below to &PotentialType,
/// this is necessary because it is used in Potential_init which is defined before PotentialType)
static PyTypeObject* PotentialTypePtr;

/// description of Potential class
static const char* docstringPotential = 
    "Potential is a class that represents a wide range of gravitational potentials.\n"
    "There are several ways of initializing the potential instance:\n"
    "  - from a list of key=value arguments that specify an elementary potential class;\n"
    "  - from a tuple of dictionary objects that contain the same list of possible "
    "key/value pairs for each component of a composite potential;\n"
    "  - from an INI file with these parameters for one or several components;\n"
    "  - from a file with potential expansion coefficients or an N-body snapshot;\n"
    "  - from a tuple of existing Potential objects created previously "
    "(in this case a composite potential is created from these components).\n"
    "Note that all keywords and their values are not case-sensitive.\n\n"
    "List of possible keywords for a single component:\n"
    "  type='...'   the type of potential, can be one of the following 'basic' types:\n"
    "    Harmonic, Logarithmic, Plummer, MiyamotoNagai, NFW, Ferrers, Dehnen, "
    "OblatePerfectEllipsoid, DiskDensity, SpheroidDensity;\n"
    "    or one of the expansion types:  Multipole or CylSpline - "
    "in these cases, one should provide either a density model, file name, "
    "or an array of particles.\n"
    DOCSTRING_DENSITY_PARAMS
    "Parameters for potential expansions:\n"
    "  density=...   the density model for a potential expansion.\n  It may be a string "
    "with the name of density profile (most of the elementary potentials listed above "
    "can be used as density models, except those with infinite mass; "
    "in addition, there are other density models without a corresponding potential).\n"
    "  Alternatively, it may be an object providing an appropriate interface -- "
    "either an instance of Density or Potential class, or a user-defined function "
    "'my_density(xyz)' returning the value of density computed simultaneously at N points, "
    "where xyz is a Nx3 array of points in cartesian coordinates (even if N=1, it's a 2d array).\n"
    "  file='...'   the name of a file with potential coefficients for a potential "
    "expansion (an alternative to density='...'), or with an N-body snapshot that "
    "will be used to compute the coefficients.\n"
    "  particles=(coords, mass)   array of point masses to be used in construction "
    "of a potential expansion (an alternative to density='...' or file='...' options): "
    "should be a tuple with two arrays - coordinates and mass, where the first one is "
    "a two-dimensional Nx3 array and the second one is a one-dimensional array of length N.\n"
    "  symmetry='...'   assumed symmetry for potential expansion constructed from "
    "an N-body snapshot (possible options, in order of decreasing symmetry: "
    "'Spherical', 'Axisymmetric', 'Triaxial', 'None', or a numerical code).\n"
    "  gridSizeR=...   number of radial grid points in Multipole and CylSpline potentials.\n"
    "  gridSizeZ=...   number of grid points in z-direction for CylSpline potential.\n"
    "  rmin=...   radius of the innermost grid node for Multipole and CylSpline; zero(default) "
    "means auto-detect.\n"
    "  rmax=...   same for the outermost grid node.\n"
    "  zmin=...   z-coordinate of the innermost grid node in CylSpline (zero means autodetect).\n"
    "  zmax=...   same for the outermost grid node.\n"
    "  lmax=...   order of spherical-harmonic expansion (max.index of angular harmonic "
    "coefficient) in Multipole.\n"
    "  mmax=...   order of azimuthal-harmonic expansion (max.index of Fourier coefficient in "
    "phi angle) in Multipole and CylSpline.\n"
    "  smoothing=...   amount of smoothing in Multipole initialized from an N-body snapshot.\n\n"
    "Most of these parameters have reasonable default values; the only necessary ones are "
    "`type`, and for a potential expansion, `density` or `file` or `particles`.\n"
    "If the coefficiens of a potential expansion are loaded from a file, then the `type` argument "
    "is not required (it will be inferred from the first line of the file).\n"
    "Examples:\n\n"
    ">>> pot_halo = Potential(type='Dehnen', mass=1e12, gamma=1, scaleRadius=100, p=0.8, q=0.6)\n"
    ">>> pot_disk = Potential(type='MiyamotoNagai', mass=5e10, scaleRadius=5, scaleHeight=0.5)\n"
    ">>> pot_composite = Potential(pot_halo, pot_disk)\n"
    ">>> pot_from_ini  = Potential('my_potential.ini')\n"
    ">>> pot_from_coef = Potential(file='stored_coefs')\n"
    ">>> pot_from_particles = Potential(type='Multipole', particles=(coords, masses))\n"
    ">>> pot_user = Potential(type='Multipole', density=lambda x: (numpy.sum(x**2,axis=1)+1)**-2)\n"
    ">>> disk_par = dict(type='DiskDensity', surfaceDensity=1e9, scaleRadius=3, scaleHeight=0.4)\n"
    ">>> halo_par = dict(type='SpheroidDensity', densityNorm=2e7, scaleRadius=15, gamma=1, beta=3, "
    "outerCutoffRadius=150, axisRatioZ=0.8)\n"
    ">>> pot_exp = Potential(type='Multipole', density=Density(**halo_par), "
    "gridSizeR=20, Rmin=1, Rmax=500, lmax=4)\n"
    ">>> pot_galpot = Potential(disk_par, halo_par)\n\n"
    "The latter example illustrates the use of GalPot components (exponential disks and spheroids) "
    "from Dehnen&Binney 1998; these are internally implemented using a Multipole potential expansion "
    "and a special variant of disk potential, but may also be combined with any other components "
    "if needed.\n"
    "The numerical values in the above examples are given in solar masses and kiloparsecs; "
    "a call to `setUnits` should precede the construction of potentials in this approach. "
    "Alternatively, one may provide no units at all, and use the `N-body` convention G=1 "
    "(this is the default regime and is restored by `resetUnits`).\n";

/// attempt to construct potential from an array of particles
potential::PtrPotential Potential_initFromParticles(
    const utils::KeyValueMap& params, PyObject* points)
{
    if(params.contains("file"))
        throw std::invalid_argument("Cannot provide both 'particles' and 'file' arguments");
    if(params.contains("density"))
        throw std::invalid_argument("Cannot provide both 'particles' and 'density' arguments");
    if(!params.contains("type"))
        throw std::invalid_argument("Must provide 'type=\"...\"' argument");
    PyObject *pointCoordObj, *pointMassObj;
    if(!PyArg_ParseTuple(points, "OO", &pointCoordObj, &pointMassObj)) {
        throw std::invalid_argument("'particles' must be a tuple with two arrays - "
            "coordinates and mass, where the first one is a two-dimensional Nx3 array "
            "and the second one is a one-dimensional array of length N");
    }
    PyArrayObject *pointCoordArr = (PyArrayObject*)
        PyArray_FROM_OTF(pointCoordObj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pointMassArr  = (PyArrayObject*)
        PyArray_FROM_OTF(pointMassObj,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(pointCoordArr == NULL || pointMassArr == NULL) {
        Py_XDECREF(pointCoordArr);
        Py_XDECREF(pointMassArr);
        throw std::invalid_argument("'particles' does not contain valid arrays");
    }
    int numpt = 0;
    if(PyArray_NDIM(pointMassArr) == 1)
        numpt = PyArray_DIM(pointMassArr, 0);
    if(numpt == 0 || PyArray_NDIM(pointCoordArr) != 2 || 
        PyArray_DIM(pointCoordArr, 0) != numpt || PyArray_DIM(pointCoordArr, 1) != 3)
    {
        Py_DECREF(pointCoordArr);
        Py_DECREF(pointMassArr);
        throw std::invalid_argument("'particles' does not contain valid arrays "
            "(the first one must be 2d array of shape Nx3, "
            "and the second one must be 1d array of length N)");
    }
    particles::ParticleArray<coord::PosCar> pointArray;
    pointArray.data.reserve(numpt);
    for(int i=0; i<numpt; i++) {
        pointArray.add(convertPos(&pyArrayElem<double>(pointCoordArr, i, 0)),
            pyArrayElem<double>(pointMassArr, i) * conv->massUnit);
    }
    Py_DECREF(pointCoordArr);
    Py_DECREF(pointMassArr);
    return potential::createPotential(params, pointArray, *conv);
}

/// attempt to construct an elementary potential from the parameters provided in dictionary
potential::PtrPotential Potential_initFromDict(PyObject* args)
{
    utils::KeyValueMap params = convertPyDictToKeyValueMap(args);
    // check if the list of arguments contains points
    PyObject* points = getItemFromPyDict(args, "particles");
    if(points) {
        params.unset("particles");
        return Potential_initFromParticles(params, points);
    }
    // check if the list of arguments contains a density object
    // or a string specifying the name of density model
    PyObject* dens_obj = getItemFromPyDict(args, "density");
    if(dens_obj) {
        if(params.contains("file"))
            throw std::invalid_argument("Cannot provide both 'file' and 'density' arguments");
        potential::PtrDensity dens = getDensity(dens_obj,
            potential::getSymmetryTypeByName(toString(getItemFromPyDict(args, "symmetry"))));
        if(dens) {
            /// attempt to construct a potential expansion from a user-provided density model
            if(params.getString("type").empty())
                throw std::invalid_argument("'type' argument must be provided");
            params.unset("density");
            return potential::createPotential(params, *dens, *conv);
        } else if(!PyString_Check(dens_obj)) {
            throw std::invalid_argument(
                "'density' argument should be the name of density profile "
                "or an object that provides an appropriate interface (e.g., an instance of "
                "Density or Potential class, or a user-defined function of 3 coordinates)");
        }
    }
    return potential::createPotential(params, *conv);
}

/// attempt to construct a composite potential from a tuple of Potential objects
/// or dictionaries with potential parameters
potential::PtrPotential Potential_initFromTuple(PyObject* tuple)
{
    // if we have one string parameter, it could be the name of an INI file or a coefs file
    if(PyTuple_Size(tuple) == 1 && PyString_Check(PyTuple_GET_ITEM(tuple, 0))) {
        std::string name(PyString_AsString(PyTuple_GET_ITEM(tuple, 0)));
        try{
            // first attempt to treat it as a name of a coefficients file
            return potential::readPotential(name, *conv);
        }
        catch(std::exception&) {
            // if that failed, treat it as an INI file (this may also fail - then return an error)
            return potential::createPotential(name, *conv);
        }
    }
    bool onlyPot = true, onlyDict = true;
    // first check the types of tuple elements
    for(Py_ssize_t i=0; i<PyTuple_Size(tuple); i++) {
        onlyPot &= PyObject_TypeCheck(PyTuple_GET_ITEM(tuple, i), PotentialTypePtr) &&
             ((PotentialObject*)PyTuple_GET_ITEM(tuple, i))->pot;  // an existing Potential object
        onlyDict &= PyDict_Check(PyTuple_GET_ITEM(tuple, i));      // a dictionary with param=value pairs
    }
    if(onlyPot) {
        std::vector<potential::PtrPotential> components;
        for(Py_ssize_t i=0; i<PyTuple_Size(tuple); i++) {
            components.push_back(((PotentialObject*)PyTuple_GET_ITEM(tuple, i))->pot);
        }
        return potential::PtrPotential(new potential::CompositeCyl(components));
    } else if(onlyDict) {
        std::vector<utils::KeyValueMap> paramsArr;
        for(Py_ssize_t i=0; i<PyTuple_Size(tuple); i++) {
            paramsArr.push_back(convertPyDictToKeyValueMap(PyTuple_GET_ITEM(tuple, i)));
        }
        return potential::createPotential(paramsArr, *conv);
    } else
        throw std::invalid_argument(
            "The tuple should contain either Potential objects or dictionaries with potential parameters");
}

/// the generic constructor of Potential object
int Potential_init(PotentialObject* self, PyObject* args, PyObject* namedArgs)
{
    try{
        // check if we have only a tuple of potential components as arguments
        if(args!=NULL && PyTuple_Check(args) && PyTuple_Size(args)>0 && 
            (namedArgs==NULL || PyDict_Size(namedArgs)==0))
            self->pot = Potential_initFromTuple(args);
        else if(namedArgs!=NULL && PyDict_Check(namedArgs) && PyDict_Size(namedArgs)>0)
            self->pot = Potential_initFromDict(namedArgs);
        else {
            utils::msg(utils::VL_WARNING, "Agama",
                "Received "+utils::toString((int)PyTuple_Size(args))+" positional arguments "+
                (namedArgs==NULL ? "and no named arguments" :
                "and "+utils::toString((int)PyDict_Size(namedArgs))+" named arguments"));
            throw std::invalid_argument(
                "Invalid parameters passed to the constructor, type help(Potential) for details");
        }
        assert(self->pot);
        utils::msg(utils::VL_VERBOSE, "Agama", "Created "+std::string(self->pot->name())+
            " potential at "+utils::toString(self->pot.get()));
        return 0;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, (std::string("Error in creating potential: ")+e.what()).c_str());
        return -1;
    }
}

// this check seems to be unnecessary, but let it remain for historical reasons
bool Potential_isCorrect(PyObject* self)
{
    if(self==NULL) {
        PyErr_SetString(PyExc_ValueError, "Should be called as method of Potential object");
        return false;
    }
    if(!((PotentialObject*)self)->pot) {
        PyErr_SetString(PyExc_ValueError, "Potential is not initialized properly");
        return false;
    }
    return true;
}

// functions that do actually compute something from the potential object,
// applying appropriate unit conversions

void fncPotential_potential(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    result[0] = ((PotentialObject*)obj)->pot->value(point)
        / pow_2(conv->velocityUnit);   // unit of potential is V^2
}

PyObject* Potential_potential(PyObject* self, PyObject* args) {
    if(!Potential_isCorrect(self))
        return NULL;
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE>
        (self, args, fncPotential_potential);
}

void fncPotential_density(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    result[0] = ((PotentialObject*)obj)->pot->density(point)
        / (conv->massUnit / pow_3(conv->lengthUnit));  // unit of density is M/L^3
}

PyObject* Potential_density(PyObject* self, PyObject* args) {
    if(!Potential_isCorrect(self))
        return NULL;
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE>
        (self, args, fncPotential_density);
}

void fncPotential_force(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    coord::GradCar grad;
    ((PotentialObject*)obj)->pot->eval(point, NULL, &grad);
    // unit of force per unit mass is V/T
    const double convF = 1 / (conv->velocityUnit/conv->timeUnit);
    result[0] = -grad.dx * convF;
    result[1] = -grad.dy * convF;
    result[2] = -grad.dz * convF;
}

PyObject* Potential_force(PyObject* self, PyObject* args) {
    if(!Potential_isCorrect(self))
        return NULL;
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_TRIPLET>
        (self, args, fncPotential_force);
}

void fncPotential_forceDeriv(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    coord::GradCar grad;
    coord::HessCar hess;
    ((PotentialObject*)obj)->pot->eval(point, NULL, &grad, &hess);
    // unit of force per unit mass is V/T
    const double convF = 1 / (conv->velocityUnit / conv->timeUnit);
    // unit of force deriv per unit mass is V/T/L
    const double convD = 1 / (conv->velocityUnit / conv->timeUnit / conv->lengthUnit);
    result[0] = -grad.dx   * convF;
    result[1] = -grad.dy   * convF;
    result[2] = -grad.dz   * convF;
    result[3] = -hess.dx2  * convD;
    result[4] = -hess.dy2  * convD;
    result[5] = -hess.dz2  * convD;
    result[6] = -hess.dxdy * convD;
    result[7] = -hess.dydz * convD;
    result[8] = -hess.dxdz * convD;
}

PyObject* Potential_forceDeriv(PyObject* self, PyObject* args) {
    if(!Potential_isCorrect(self))
        return NULL;
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_TRIPLET_AND_SEXTET>
        (self, args, fncPotential_forceDeriv);
}

void fncPotential_Rcirc_from_L(void* obj, const double input[], double *result) {
    const double L = input[0] * conv->lengthUnit * conv->velocityUnit;
    result[0] = R_from_Lz(*((PotentialObject*)obj)->pot, L) / conv->lengthUnit;
}

void fncPotential_Rcirc_from_E(void* obj, const double input[], double *result) {
    const double E = input[0] * pow_2(conv->velocityUnit);
    result[0] = R_circ(*((PotentialObject*)obj)->pot, E) / conv->lengthUnit;
}

PyObject* Potential_Rcirc(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!Potential_isCorrect(self))
        return NULL;
    static const char* keywords[] = {"L", "E", NULL};
    PyObject *L_obj=NULL, *E_obj=NULL;
    if(PyArg_ParseTupleAndKeywords(args, namedArgs, "|OO", const_cast<char**>(keywords), &L_obj, &E_obj)
        && ((L_obj!=NULL) ^ (E_obj!=NULL) /*exactly one of them should be non-NULL*/) )
    {
        if(L_obj)
            return callAnyFunctionOnArray<INPUT_VALUE_SINGLE, OUTPUT_VALUE_SINGLE>
                (self, L_obj, fncPotential_Rcirc_from_L);
        else
            return callAnyFunctionOnArray<INPUT_VALUE_SINGLE, OUTPUT_VALUE_SINGLE>
                (self, E_obj, fncPotential_Rcirc_from_E);
    } else {
        PyErr_SetString(PyExc_ValueError, "Rcirc() takes exactly one argument (either L or E)");
        return NULL;
    }
}

void fncPotential_Tcirc_from_E(void* obj, const double input[], double *result) {
    const double
    E = input[0] * pow_2(conv->velocityUnit),
    R = R_circ(*((PotentialObject*)obj)->pot, E),
    T = R / v_circ(*((PotentialObject*)obj)->pot, R) * 2*M_PI;
    result[0] = T / conv->timeUnit;
}

void fncPotential_Tcirc_from_xv(void* obj, const double input[], double *result) {
    const coord::PosVelCar point = convertPosVel(input);
    const double
    E = totalEnergy(*((PotentialObject*)obj)->pot, point),
    R = R_circ(*((PotentialObject*)obj)->pot, E),
    T = R / v_circ(*((PotentialObject*)obj)->pot, R) * 2*M_PI;
    result[0] = T / conv->timeUnit;
}

PyObject* Potential_Tcirc(PyObject* self, PyObject* args)
{
    PyObject* input = NULL;
    if(!Potential_isCorrect(self) || !PyArg_ParseTuple(args, "O", &input))
        return NULL;
    // find out the shape of input array or anything that could be converted into an array
    PyArrayObject* arr = NULL;
    PyArray_Descr* dtype = NULL;
    int ndim = 0;
    npy_intp inputdims[NPY_MAXDIMS], *dims=inputdims;
    if(PyArray_GetArrayParamsFromObject(input, NULL, 0, &dtype, &ndim, inputdims, &arr, NULL) < 0)
        return NULL;
    // if it was indeed an array, need to obtain its shape manually
    if(arr) {
        ndim = PyArray_NDIM(arr);
        dims = PyArray_DIMS(arr);
    }        
    if((ndim == 1 && dims[0] == 6) || (ndim == 2 && dims[1] == 6))
        return callAnyFunctionOnArray<INPUT_VALUE_SEXTET, OUTPUT_VALUE_SINGLE>
            (self, input, fncPotential_Tcirc_from_xv);
    else
        return callAnyFunctionOnArray<INPUT_VALUE_SINGLE, OUTPUT_VALUE_SINGLE>
            (self, input, fncPotential_Tcirc_from_E);
}

void fncPotential_Rmax(void* obj, const double input[], double *result) {
    const double E = input[0] * pow_2(conv->velocityUnit);
    result[0] = R_max(*((PotentialObject*)obj)->pot, E) / conv->lengthUnit;
}

PyObject* Potential_Rmax(PyObject* self, PyObject* args) {
    if(!Potential_isCorrect(self))
        return NULL;
    return callAnyFunctionOnArray<INPUT_VALUE_SINGLE, OUTPUT_VALUE_SINGLE>
        (self, args, fncPotential_Rmax);
}

PyObject* Potential_export(PyObject* self, PyObject* args)
{
    const char* filename=NULL;
    if(!Potential_isCorrect(self) || !PyArg_ParseTuple(args, "s", &filename))
        return NULL;
    try{
        writePotential(filename, *((PotentialObject*)self)->pot, *conv);
        Py_INCREF(Py_None);
        return Py_None;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, (std::string("Error writing file: ")+e.what()).c_str());
        return NULL;
    }
}

PyObject* Potential_totalMass(PyObject* self)
{
    if(!Potential_isCorrect(self))
        return NULL;
    try{
        return Py_BuildValue("d", ((PotentialObject*)self)->pot->totalMass() / conv->massUnit);
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError,
            (std::string("Error in Potential.totalMass(): ")+e.what()).c_str());
        return NULL;
    }
}

PyObject* Potential_sample(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!Potential_isCorrect(self))
        return NULL;
    return sampleDensity(*((PotentialObject*)self)->pot, args, namedArgs);
}

PyObject* Potential_name(PyObject* self)
{
    if(!Potential_isCorrect(self))
        return NULL;
    const char* name = ((PotentialObject*)self)->pot->name();
    if(name == potential::CompositeCyl::myName()) {
        try{
            const potential::CompositeCyl& pot =
                dynamic_cast<const potential::CompositeCyl&>(*((PotentialObject*)self)->pot);
            std::string tmp = std::string(name) + ": ";
            for(unsigned int i=0; i<pot.size(); i++) {
                if(i>0) tmp += ", ";
                tmp += pot.component(i)->name();
            }
            return Py_BuildValue("s", tmp.c_str());
        }
        catch(std::exception& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        }
    } else
        return Py_BuildValue("s", name);
}

PyObject* Potential_elem(PyObject* self, Py_ssize_t index)
{
    if(!Potential_isCorrect(self))
        return NULL;
    try{
        const potential::CompositeCyl& pot =
            dynamic_cast<const potential::CompositeCyl&>(*((PotentialObject*)self)->pot);
        if(index<0 || index >= (Py_ssize_t)pot.size()) {
            PyErr_SetString(PyExc_IndexError, "Potential component index out of range");
            return NULL;
        }
        return createPotentialObject(pot.component(index));
    }
    catch(std::exception&) {
        PyErr_SetString(PyExc_TypeError, "Potential is not a composite");
        return NULL;
    }
}

Py_ssize_t Potential_len(PyObject* self)
{
    if(!Potential_isCorrect(self))
        return -1;
    try{
        return dynamic_cast<const potential::CompositeCyl&>(*((PotentialObject*)self)->pot).size();
    }
    catch(std::exception&) {
        PyErr_SetString(PyExc_TypeError, "Potential is not a composite");
        return -1;
    }
}

static PySequenceMethods Potential_sequence_methods = {
    Potential_len, 0, 0, Potential_elem,
};

static PyMethodDef Potential_methods[] = {
    { "name", (PyCFunction)Potential_name, METH_NOARGS,
      "Return the name of the potential\n"
      "No arguments\n"
      "Returns: string" },
    { "potential", Potential_potential, METH_VARARGS,
      "Compute potential at a given point or array of points\n"
      "Arguments: a triplet of floats (x,y,z) or array of such triplets\n"
      "Returns: float or array of floats" },
    { "density", Potential_density, METH_VARARGS,
      "Compute density at a given point or array of points\n"
      "Arguments: a triplet of floats (x,y,z) or array of such triplets\n"
      "Returns: float or array of floats" },
    { "force", Potential_force, METH_VARARGS,
      "Compute force at a given point or array of points\n"
      "Arguments: a triplet of floats (x,y,z) or array of such triplets\n"
      "Returns: float[3] - x,y,z components of force, or array of such triplets" },
    { "forceDeriv", Potential_forceDeriv, METH_VARARGS,
      "Compute force and its derivatives at a given point or array of points\n"
      "Arguments: a triplet of floats (x,y,z) or array of such triplets\n"
      "Returns: (float[3],float[6]) - x,y,z components of force, "
      "and the matrix of force derivatives stored as dFx/dx,dFy/dy,dFz/dz,dFx/dy,dFy/dz,dFz/dx; "
      "or if the input was an array of N points, then both items in the tuple are 2d arrays "
      "with sizes Nx3 and Nx6, respectively"},
    { "sample", (PyCFunction)Potential_sample, METH_VARARGS | METH_KEYWORDS, 
      "Sample the density profile with N point masses (see help(Density.sample) for description)\n"
      "Arguments: the number of points\n"
      "Returns: a tuple of two arrays: "
      "2d Nx3 array of point cartesian coordinates and 1d array of N point masses" },
    { "export", Potential_export, METH_VARARGS,
      "Export potential expansion coefficients to a text file\n"
      "Arguments: filename (string)\n"
      "Returns: none" },
    { "totalMass", (PyCFunction)Potential_totalMass, METH_NOARGS,
      "Return the total mass of the density model\n"
      "No arguments\n"
      "Returns: float number" },
    { "Rcirc", (PyCFunction)Potential_Rcirc, METH_VARARGS | METH_KEYWORDS,
      "Find the radius of a circular orbit in the equatorial plane corresponding to "
      "either the given z-component of angular momentum L or energy E; "
      "the potential is assumed to be axisymmetric (all quantities are evaluated along x axis)\n"
      "Arguments:\n"
      "  L=... (a single number or an array of numbers) - the values of angular momentum;\n"
      "  E=... (same) - the values of energy; the arguments are mutually exclusive, "
      "and L is the default one if no name is provided\n"
      "Returns: a single number or an array of numbers - the radii of corresponding orbits\n" },
    { "Tcirc", Potential_Tcirc, METH_VARARGS,
      "Compute the period of a circular orbit for the given energy (a) or the (x,v) point (b)\n"
      "Arguments:\n"
      "  (a) a single value of energy or an array of N such values, or\n"
      "  (b) a single point (6 numbers - position and velocity) or a Nx6 array of points\n"
      "Returns: a single value or N values of orbital periods\n" },
    { "Rmax", Potential_Rmax, METH_VARARGS,
      "Find the maximum radius accessible to the given energy (i.e. the root of Phi(Rmax,0,0)=E)\n"
      "Arguments: a single number or an array of numbers - the values of energy\n"
      "Returns: corresponding values of radii\n" },
    { NULL }
};

static PyTypeObject PotentialType = {
    PyObject_HEAD_INIT(NULL)
    0, "agama.Potential",
    sizeof(PotentialObject), 0, (destructor)Potential_dealloc,
    0, 0, 0, 0, 0, 0, &Potential_sequence_methods, 0, 0, 0, Potential_name, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringPotential,
    0, 0, 0, 0, 0, 0, Potential_methods, 0, 0, 0, 0, 0, 0, 0,
    (initproc)Potential_init
};

/// create a Python Potential object and initialize it with an existing instance of C++ potential class
PyObject* createPotentialObject(const potential::PtrPotential& pot)
{
    PotentialObject* pot_obj = PyObject_New(PotentialObject, &PotentialType);
    if(!pot_obj)
        return NULL;
    // same hack as in 'createDensityObject()'
    new (&(pot_obj->pot)) potential::PtrPotential;
    pot_obj->pot = pot;
    utils::msg(utils::VL_VERBOSE, "Agama",
        "Created a Python wrapper for "+std::string(pot->name())+" potential");
    return (PyObject*)pot_obj;
}

/// extract a pointer to C++ Potential class from a Python object, or return an empty pointer on error
potential::PtrPotential getPotential(PyObject* pot_obj)
{
    if(pot_obj == NULL || !PyObject_TypeCheck(pot_obj, &PotentialType) ||
        !((PotentialObject*)pot_obj)->pot)
        return potential::PtrPotential();    // empty pointer
    return ((PotentialObject*)pot_obj)->pot; // pointer to an existing instance of C++ Potential class
}

/// extract a pointer to C++ Density class from a Python object, or return an empty pointer on error
potential::PtrDensity getDensity(PyObject* dens_obj, coord::SymmetryType sym)
{
    if(dens_obj == NULL)
        return potential::PtrDensity();

    // check if this is a Python wrapper for a C++ Density object
    if(PyObject_TypeCheck(dens_obj, &DensityType) && ((DensityObject*)dens_obj)->dens)
        return ((DensityObject*)dens_obj)->dens;

    // check if this is a Python wrapper for a C++ Potential object,
    // which also provides a 'density()' method
    if(PyObject_TypeCheck(dens_obj, &PotentialType) && ((PotentialObject*)dens_obj)->pot)
        return ((PotentialObject*)dens_obj)->pot;

    // otherwise this could be an arbitrary Python function
    if(PyCallable_Check(dens_obj))
    {   // then create a C++ wrapper for this Python function
        // (don't check if it accepts a single Nx3 array as the argument...)
        return potential::PtrDensity(new DensityWrapper(dens_obj, sym));
    }

    // none of the above succeeded -- return an empty pointer
    return potential::PtrDensity();
}


///@}
//  --------------------------
/// \name  ActionFinder class
//  --------------------------
///@{

/// create a spherical or non-spherical action finder
actions::PtrActionFinder createActionFinder(const potential::PtrPotential& pot, bool interpolate)
{
    assert(pot);
    actions::PtrActionFinder af = isSpherical(*pot) ?
        actions::PtrActionFinder(new actions::ActionFinderSpherical(*pot)) :
        actions::PtrActionFinder(new actions::ActionFinderAxisymFudge(pot, interpolate));
    utils::msg(utils::VL_VERBOSE, "Agama",
        "Created " +
        std::string(isSpherical(*pot) ? "Spherical" : interpolate ? "Interpolated Fudge" : "Fudge") +
        " action finder for " + pot->name() + " potential at " + utils::toString(af.get()));
    return af;
}

/// \cond INTERNAL_DOCS
/// Python type corresponding to ActionFinder class
typedef struct {
    PyObject_HEAD
    actions::PtrActionFinder af;  // C++ object for action finder
} ActionFinderObject;
/// \endcond

void ActionFinder_dealloc(ActionFinderObject* self)
{
    utils::msg(utils::VL_VERBOSE, "Agama", "Deleted an action finder at "+
        utils::toString(self->af.get()));
    self->af.reset();
    self->ob_type->tp_free(self);
}

static const char* docstringActionFinder =
    "ActionFinder object is created for a given potential (provided as the first argument "
    "to the constructor); if the potential is axisymmetric, there is a further option to use "
    "interpolation tables for actions (optional second argument 'interp=...', True by default), "
    "which speeds up computation of actions (but not actions and angles) at the expense of "
    "a somewhat lower accuracy.\n"
    "The () operator computes actions for a given position/velocity point, or array of points.\n"
    "Arguments: a sextet of floats (x,y,z,vx,vy,vz) or an Nx6 array of N such sextets, "
    "and optionally an 'angles=True' argument if frequencies and angles are also needed "
    "(requires extra computations).\n"
    "Returns: if angles are not computed, a single Nx3 array of floats "
    "(for each point: Jr, Jz, Jphi); in the opposite case, a tuple of three Nx3 arrays: "
    "actions, angles, and frequencies (in the same order - r,z,phi).";

int ActionFinder_init(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"potential", "interp", NULL};
    PyObject* pot_obj=NULL;
    int interpolate=1;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "O|i", const_cast<char**>(keywords),
        &pot_obj, &interpolate))
    {
        PyErr_SetString(PyExc_ValueError, "Incorrect parameters for ActionFinder constructor: "
            "must provide an instance of Potential to work with.");
        return -1;
    }
    potential::PtrPotential pot = getPotential(pot_obj);
    if(!pot) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a valid instance of Potential class");
        return -1;
    }
    try{
        ((ActionFinderObject*)self)->af = createActionFinder(pot, interpolate);
        return 0;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, 
            (std::string("Error in ActionFinder initialization: ")+e.what()).c_str());
        return -1;
    }
}

template<bool Angles>
void fncActions(void* obj, const double input[], double *result) {
    try{
        const coord::PosVelCyl point = coord::toPosVelCyl(convertPosVel(input));
        actions::Frequencies freq;
        actions::ActionAngles actang = Angles ?
            ((ActionFinderObject*)obj)->af->actionAngles(point, &freq) :
            actions::ActionAngles(
            ((ActionFinderObject*)obj)->af->actions(point), actions::Angles(0,0,0));
        // unit of action is V*L
        const double convA = 1 / (conv->velocityUnit * conv->lengthUnit);
        result[0] = actang.Jr   * convA;
        result[1] = actang.Jz   * convA;
        result[2] = actang.Jphi * convA;
        if(Angles) {
            // unit of frequency is V/L
            const double convF = conv->lengthUnit / conv->velocityUnit;
            result[3] = actang.thetar;
            result[4] = actang.thetaz;
            result[5] = actang.thetaphi;
            result[6] = freq.Omegar   * convF;
            result[7] = freq.Omegaz   * convF;
            result[8] = freq.Omegaphi * convF;
        }
    }
    catch(std::exception& ) {  // indicates an error, e.g., positive value of energy
        result[0] = result[1] = result[2] = NAN;
        if(Angles)  result[3] = result[4] = result[5] = result[6] = result[7] = result[8] =NAN;
    }
}

PyObject* ActionFinder_value(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!((ActionFinderObject*)self)->af) {
        PyErr_SetString(PyExc_ValueError, "ActionFinder object is not properly initialized");
        return NULL;
    }
    static const char* keywords[] = {"point", "angles", NULL};
    PyObject *points = NULL, *angles = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "O|O", const_cast<char**>(keywords),
        &points, &angles))
    {
        PyErr_SetString(PyExc_ValueError,
            "Must provide an array of points and optionally the 'angles=True/False' flag");
        return NULL;
    }
    if(angles==NULL || !PyObject_IsTrue(angles))
        return callAnyFunctionOnArray<INPUT_VALUE_SEXTET, OUTPUT_VALUE_TRIPLET>
            (self, points, fncActions<false>);
    else
        return callAnyFunctionOnArray<INPUT_VALUE_SEXTET, OUTPUT_VALUE_TRIPLET_AND_TRIPLET_AND_TRIPLET>
            (self, points, fncActions<true>);
}

static PyTypeObject ActionFinderType = {
    PyObject_HEAD_INIT(NULL)
    0, "agama.ActionFinder",
    sizeof(ActionFinderObject), 0, (destructor)ActionFinder_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, ActionFinder_value, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringActionFinder, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ActionFinder_init
};

/// create a Python ActionFinder object and initialize it
/// with an existing instance of C++ action finder class
PyObject* createActionFinderObject(actions::PtrActionFinder af)
{
    ActionFinderObject* af_obj = PyObject_New(ActionFinderObject, &ActionFinderType);
    if(!af_obj)
        return NULL;
    // same trickery as in 'createDensityObject()'
    new (&(af_obj->af)) actions::PtrActionFinder;
    af_obj->af = af;
    utils::msg(utils::VL_VERBOSE, "Agama", "Created a Python wrapper for action finder at "+
        utils::toString(af.get()));
    return (PyObject*)af_obj;
}

/// \cond INTERNAL_DOCS
/// standalone action finder
typedef struct {
    potential::PtrPotential pot;
    double ifd;
} ActionFinderParams;
/// \endcond

template<bool Angles>
void fncActionsStandalone(void* obj, const double input[], double *result) {
    try{
        const coord::PosVelCyl point = coord::toPosVelCyl(convertPosVel(input));
        const ActionFinderParams* params = static_cast<const ActionFinderParams*>(obj);
        double ifd = params->ifd * conv->lengthUnit;
        actions::Frequencies freq;
        actions::ActionAngles actang = Angles ?
            (isSpherical(*params->pot) ?
            actions::actionAnglesSpherical  (*params->pot, point, &freq) :
            actions::actionAnglesAxisymFudge(*params->pot, point, ifd, &freq)) :
            actions::ActionAngles(
            (isSpherical(*params->pot) ?
            actions::actionsSpherical  (*params->pot, point) :
            actions::actionsAxisymFudge(*params->pot, point, ifd)), actions::Angles(0,0,0));
        const double convA = 1 / (conv->velocityUnit * conv->lengthUnit);  // inverse action unit
        result[0] = actang.Jr   * convA;
        result[1] = actang.Jz   * convA;
        result[2] = actang.Jphi * convA;
        if(Angles) {
            const double convF = conv->lengthUnit / conv->velocityUnit;  // inverse frequency unit
            result[3] = actang.thetar;
            result[4] = actang.thetaz;
            result[5] = actang.thetaphi;
            result[6] = freq.Omegar   * convF;
            result[7] = freq.Omegaz   * convF;
            result[8] = freq.Omegaphi * convF;
        }
    }
    catch(std::exception& ) {
        result[0] = result[1] = result[2] = NAN;
        if(Angles)  result[3] = result[4] = result[5] = result[6] = result[7] = result[8] = NAN;
    }
}

static const char* docstringActions =
    "Compute actions for a given position/velocity point, or array of points\n"
    "Arguments: \n"
    "    point - a sextet of floats (x,y,z,vx,vy,vz) or array of such sextets;\n"
    "    potential - Potential object that defines the gravitational potential;\n"
    "    fd (float) - focal distance for the prolate spheroidal coordinate system "
    "(not necessary if the potential is spherical);\n"
    "    angles (boolean, default False) - whether to compute angles and frequencies as well.\n"
    "Returns: if angles are not computed, a single Nx3 array of floats "
    "(for each point: Jr, Jz, Jphi); in the opposite case, a tuple of three Nx3 arrays: "
    "actions, angles, and frequencies (in the same order - r,z,phi).";
    
PyObject* actions(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"point", "potential", "fd", "angles", NULL};
    double ifd = 0;
    PyObject *points_obj = NULL, *pot_obj = NULL, *angles = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "|OOdO", const_cast<char**>(keywords),
        &points_obj, &pot_obj, &ifd, &angles) || ifd<0)
    {
        //PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to actions()");
        return NULL;
    }
    ActionFinderParams params;
    params.pot = getPotential(pot_obj);
    params.ifd = ifd;
    if(!params.pot) {
        PyErr_SetString(PyExc_TypeError, "Argument 'potential' must be a valid instance of Potential class");
        return NULL;
    }
    if(angles==NULL || !PyObject_IsTrue(angles))
        return callAnyFunctionOnArray<INPUT_VALUE_SEXTET, OUTPUT_VALUE_TRIPLET>
            (&params, points_obj, fncActionsStandalone<false>);
    else
        return callAnyFunctionOnArray<INPUT_VALUE_SEXTET, OUTPUT_VALUE_TRIPLET_AND_TRIPLET_AND_TRIPLET>
            (&params, points_obj, fncActionsStandalone<true>);
}


///@}
//  ----------------------------------
/// \name  DistributionFunction class
//  ----------------------------------
///@{

/// \cond INTERNAL_DOCS
/// Python type corresponding to DistributionFunction class
typedef struct {
    PyObject_HEAD
    df::PtrDistributionFunction df;
} DistributionFunctionObject;
/// \endcond

void DistributionFunction_dealloc(DistributionFunctionObject* self)
{
    utils::msg(utils::VL_VERBOSE, "Agama", "Deleted a distribution function at "+
        utils::toString(self->df.get()));
    self->df.reset();
    self->ob_type->tp_free(self);
}

// pointer to the DistributionFunctionType object (will be initialized below)
static PyTypeObject* DistributionFunctionTypePtr;
// forward declaration
df::PtrDistributionFunction getDistributionFunction(PyObject* df_obj);

static const char* docstringDistributionFunction =
    "DistributionFunction class represents an action-based distribution function.\n\n"
    "The constructor accepts several key=value arguments that describe the parameters "
    "of distribution function.\n"
    "Required parameter is type='...', specifying the type of DF: currently available types are "
    "'DoublePowerLaw' (for the halo), 'PseudoIsothermal' (for the disk component), "
    "'PseudoIsotropic' (for the isotropic DF corresponding to a given density profile), "
    "'Interp1', 'Interp3' (for interpolated DFs).\n"
    "For some of them, one also needs to provide the potential to initialize the table of epicyclic "
    "frequencies (potential=... argument), and for the PseudoIsotropic DF one needs to provide "
    "an instance of density profile (density=...) and the potential (if they are the same, then only "
    "potential=... is needed).\n"
    "Other parameters are specific to each DF type.\n"
    "Alternatively, a composite DF may be created from an array of previously constructed DFs:\n"
    ">>> df = DistributionFunction(df1, df2, df3)\n\n"
    "The () operator computes the value of distribution function for the given triplet of actions.\n"
    "The totalMass() function computes the total mass in the entire phase space.\n\n"
    "A user-defined Python function that takes a single argument - Nx3 array "
    "(with columns representing Jr, Jz, Jphi at N>=1 points) and returns an array of length N "
    "may be provided in all contexts where a DistributionFunction object is required.";

/// attempt to construct an interpolated distribution function from the parameters provided in dictionary
template<int N>
df::PtrDistributionFunction DistributionFunction_initInterpolated(PyObject* namedArgs)
{
    PyObject *u_obj = getItemFromPyDict(namedArgs, "u");  // borrowed reference or NULL
    PyObject *v_obj = getItemFromPyDict(namedArgs, "v");
    PyObject *w_obj = getItemFromPyDict(namedArgs, "w");
    PyObject *ampl_obj = getItemFromPyDict(namedArgs, "ampl");
    if(!u_obj || !v_obj || !w_obj || !ampl_obj)
        throw std::invalid_argument("Interpolated DF requires 4 array arguments: u, v, w, ampl");
    std::vector<double>
        ampl (toDoubleArray(ampl_obj)),
        gridU(toDoubleArray(u_obj)),
        gridV(toDoubleArray(v_obj)),
        gridW(toDoubleArray(w_obj));
    if(gridU.empty() || gridV.empty() || gridW.empty() || ampl.empty())
    {
        throw std::invalid_argument("Input arguments do not contain valid arrays");
    }
    // convert units and implement scaling for U and ampl arrays
    df::PtrActionSpaceScaling scaling(new df::ActionSpaceScalingTriangLog());
    const double convJ = conv->velocityUnit * conv->lengthUnit;  // dimension of actions
    const double convF = conv->massUnit / pow_3(convJ);  // dimension of distribution function
    for(unsigned int i=0; i<gridU.size(); i++) {
        double v[3];
        scaling->toScaled(actions::Actions(0, 0, gridU[i] * convJ), v);
        gridU[i] = v[0];
    }
    for(unsigned int i=0; i<ampl.size(); i++) {
        ampl[i] *= convF;
    }
    return df::PtrDistributionFunction(new df::InterpolatedDF<N>(scaling, gridU, gridV, gridW, ampl));
}

/// attempt to construct an elementary distribution function from the parameters provided in dictionary
df::PtrDistributionFunction DistributionFunction_initFromDict(PyObject* namedArgs)
{
    PyObject *pot_obj = PyDict_GetItemString(namedArgs, "potential");  // borrowed reference or NULL
    potential::PtrPotential pot;
    if(pot_obj!=NULL) {
        pot = getPotential(pot_obj);
        if(!pot)
            throw std::invalid_argument("Argument 'potential' must be a valid instance of Potential class");
        PyDict_DelItemString(namedArgs, "potential");
    }
    utils::KeyValueMap params = convertPyDictToKeyValueMap(namedArgs);
    if(!params.contains("type"))
        throw std::invalid_argument("Should provide the type='...' argument");
    std::string type = params.getString("type");
    if(utils::stringsEqual(type, "Interp1"))
        return DistributionFunction_initInterpolated<1>(namedArgs);
    else if(utils::stringsEqual(type, "Interp3"))
        return DistributionFunction_initInterpolated<3>(namedArgs);
    else if(utils::stringsEqual(type, "PseudoIsotropic")) {
        if(!pot)
            throw std::invalid_argument("Must provide a potential in 'potential=...'");
        PyObject *dens_obj = PyDict_GetItemString(namedArgs, "density");
        potential::PtrDensity dens;
        if(dens_obj!=NULL) {
            dens = getDensity(dens_obj);
            if(!dens)
                throw std::invalid_argument("Argument 'density' must be a valid Density object");
        } else
            dens = pot;
        return df::PtrDistributionFunction(new df::PseudoIsotropic(
            galaxymodel::makeEddingtonDF(potential::DensityWrapper(*dens),
            potential::PotentialWrapper(*pot)), *pot));
    }
    return df::createDistributionFunction(params, pot.get(), *conv);  // any other DF type
}

/// attempt to construct a composite distribution function from a tuple of DistributionFunction objects
df::PtrDistributionFunction DistributionFunction_initFromTuple(PyObject* tuple)
{
    std::vector<df::PtrDistributionFunction> components;
    for(Py_ssize_t i=0; i<PyTuple_Size(tuple); i++) {
        df::PtrDistributionFunction comp = getDistributionFunction(PyTuple_GET_ITEM(tuple, i));
        if(!comp)
            throw std::invalid_argument("Tuple should contain only valid DistributionFunction objects "
                "or functions providing that interface");
        components.push_back(comp);
    }
    return df::PtrDistributionFunction(new df::CompositeDF(components));
}

/// the generic constructor of DistributionFunction object
int DistributionFunction_init(DistributionFunctionObject* self, PyObject* args, PyObject* namedArgs)
{
    try{
        // check if we have only a tuple of DF components as arguments
        if(args!=NULL && PyTuple_Check(args) && PyTuple_Size(args)>0 && 
            (namedArgs==NULL || !PyDict_Check(namedArgs) || PyDict_Size(namedArgs)==0))
        {
            self->df = DistributionFunction_initFromTuple(args);
        }
        // otherwise we must have only key=value arguments for an elementary DF
        else if(namedArgs!=NULL && PyDict_Check(namedArgs) && PyDict_Size(namedArgs)>0 &&
            (args==NULL || !PyTuple_Check(args) || PyTuple_Size(args)==0))
        {
            self->df = DistributionFunction_initFromDict(namedArgs);
        } else {
            throw std::invalid_argument(
                "Should provide either a list of key=value arguments to create an elementary DF, "
                "or a tuple of existing DistributionFunction objects to create a composite DF");
        }
        assert(self->df);
        utils::msg(utils::VL_VERBOSE, "Agama", "Created a distribution function at "+
            utils::toString(self->df.get()));
        return 0;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError,
            (std::string("Error in creating distribution function: ")+e.what()).c_str());
        return -1;
    }
}

void fncDistributionFunction(void* obj, const double input[], double *result)
{
    const actions::Actions acts = convertActions(input);
    // dimension of distribution function is M L^-3 V^-3
    const double dim = pow_3(conv->velocityUnit * conv->lengthUnit) / conv->massUnit;
    try{
        result[0] = ((DistributionFunctionObject*)obj)->df->value(acts) * dim;
    }
    catch(std::exception& ) {
        result[0] = NAN;
    }
}

PyObject* DistributionFunction_value(PyObject* self, PyObject* args, PyObject* /*namedArgs*/)
{
    if(((DistributionFunctionObject*)self)->df==NULL) {
        PyErr_SetString(PyExc_ValueError, "DistributionFunction object is not properly initialized");
        return NULL;
    }
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE>
        (self, args, fncDistributionFunction);
}

PyObject* DistributionFunction_totalMass(PyObject* self)
{
    if(((DistributionFunctionObject*)self)->df==NULL) {
        PyErr_SetString(PyExc_ValueError, "DistributionFunction object is not properly initialized");
        return NULL;
    }
    try{
        double err;
        double val = ((DistributionFunctionObject*)self)->df->totalMass(1e-5, 1e6, &err);
        if(err>1e-5*val)
            utils::msg(utils::VL_WARNING, "Agama", "can't reach tolerance in df->totalMass: "
            "rel.err="+utils::toString(err/val));
        return Py_BuildValue("d", val / conv->massUnit);
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError,
            (std::string("Error in DistributionFunction.totalMass(): ")+e.what()).c_str());
        return NULL;
    }
}

static PyMethodDef DistributionFunction_methods[] = {
    { "totalMass", (PyCFunction)DistributionFunction_totalMass, METH_NOARGS,
      "Return the total mass of the model (integral of the distribution function "
      "over the entire phase space of actions)\n"
      "No arguments\n"
      "Returns: float number" },
    { NULL, NULL, 0, NULL }
};

static PyTypeObject DistributionFunctionType = {
    PyObject_HEAD_INIT(NULL)
    0, "agama.DistributionFunction",
    sizeof(DistributionFunctionObject), 0, (destructor)DistributionFunction_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, DistributionFunction_value, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringDistributionFunction, 
    0, 0, 0, 0, 0, 0, DistributionFunction_methods, 0, 0, 0, 0, 0, 0, 0,
    (initproc)DistributionFunction_init
};


/// Helper class for providing a BaseDistributionFunction interface
/// to a Python function that returns the value of df at a point in action space
class DistributionFunctionWrapper: public df::BaseDistributionFunction{
    OmpDisabler ompDisabler;
    PyObject* fnc;
public:
    DistributionFunctionWrapper(PyObject* _fnc): fnc(_fnc)
    {
        Py_INCREF(fnc);
        utils::msg(utils::VL_VERBOSE, "Agama",
            "Created a C++ df wrapper for Python function "+toString(fnc));
    }
    ~DistributionFunctionWrapper()
    {
        utils::msg(utils::VL_VERBOSE, "Agama",
            "Deleted a C++ df wrapper for Python function "+toString(fnc));
        Py_DECREF(fnc);
    }
    virtual double value(const actions::Actions &J) const {
        double act[3];
        unconvertActions(J, act);
        npy_intp dims[]  = {1, 3};
        PyObject* args   = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, act);
        PyObject* result = PyObject_CallFunctionObjArgs(fnc, args, NULL);
        Py_DECREF(args);
        double value;
        if(result == NULL) {
            PyErr_Print();
            throw std::runtime_error("Call to user-defined distribution function failed");
        }
        if(PyArray_Check(result))
            value = pyArrayElem<double>(result, 0);  // TODO: ensure that it's an array of doubles?
        else if(PyNumber_Check(result))
            value = PyFloat_AsDouble(result);
        else {
            Py_DECREF(result);
            throw std::runtime_error("Invalid data type returned from user-defined distribution function");
        }
        Py_DECREF(result);
        return value * conv->massUnit / pow_3(conv->velocityUnit * conv->lengthUnit);
    }
};

/// extract a pointer to C++ DistributionFunction class from a Python object,
/// or return an empty pointer on error
df::PtrDistributionFunction getDistributionFunction(PyObject* df_obj)
{
    if(df_obj == NULL)
        return df::PtrDistributionFunction();
    // check if this is a Python wrapper for a genuine C++ DF object
    if(PyObject_TypeCheck(df_obj, &DistributionFunctionType) && ((DistributionFunctionObject*)df_obj)->df)
        return ((DistributionFunctionObject*)df_obj)->df;
    // otherwise this could be an arbitrary callable Python object
    if(PyCallable_Check(df_obj))
    {   // then create a C++ wrapper for this Python function
        // (don't check if it accepts a single Nx3 array as the argument...)
        return df::PtrDistributionFunction(new DistributionFunctionWrapper(df_obj));
    }
    // none succeeded - return an empty pointer
    return df::PtrDistributionFunction();
}

/// create a Python DistributionFunction object from an existing instance of C++ DF class
PyObject* createDistributionFunctionObject(df::PtrDistributionFunction df)
{
    DistributionFunctionObject* df_obj =
        PyObject_New(DistributionFunctionObject, &DistributionFunctionType);
    if(!df_obj)
        return NULL;
    // same hack as in 'createDensityObject()'
    new (&(df_obj->df)) df::PtrDistributionFunction;
    df_obj->df = df;
    utils::msg(utils::VL_VERBOSE, "Agama", "Created a Python wrapper for distribution function");
    return (PyObject*)df_obj;
}


///@}
//  -------------------------
/// \name  GalaxyModel class
//  -------------------------
///@{

/// \cond INTERNAL_DOCS
/// Python type corresponding to GalaxyModel class
typedef struct {
    PyObject_HEAD
    PotentialObject* pot_obj;
    DistributionFunctionObject* df_obj;
    ActionFinderObject* af_obj;
} GalaxyModelObject;
/// \endcond

void GalaxyModel_dealloc(GalaxyModelObject* self)
{
    Py_XDECREF(self->pot_obj);
    Py_XDECREF(self->df_obj);
    Py_XDECREF(self->af_obj);
    self->ob_type->tp_free((PyObject*)self);
}

bool GalaxyModel_isCorrect(GalaxyModelObject* self)
{
    if(self==NULL) {
        PyErr_SetString(PyExc_ValueError, "Should be called as method of GalaxyModel object");
        return false;
    }
    if( !self->pot_obj || !self->pot_obj->pot ||
        !self->af_obj || !self->af_obj->af ||
        !self->df_obj || !self->df_obj->df)
    {
        PyErr_SetString(PyExc_ValueError, "GalaxyModel is not properly initialized");
        return false;
    }
    return true;
}

static const char* docstringGalaxyModel =
    "GalaxyModel is a class that takes together a Potential, "
    "a DistributionFunction, and an ActionFinder objects, "
    "and provides methods to compute moments and projections of the distribution function "
    "at a given point in the ordinary phase space (coordinate/velocity), as well as "
    "methods for drawing samples from the distribution function in the given potential.\n"
    "The constructor takes the following arguments:\n"
    "  potential - a Potential object;\n"
    "  df  - a DistributionFunction object;\n"
    "  af (optional) - an ActionFinder object; "
    "if not provided then the action finder is created internally.\n";

int GalaxyModel_init(GalaxyModelObject* self, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"potential", "df", "af", NULL};
    PyObject *pot_obj = NULL, *df_obj = NULL, *af_obj = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "OO|O", const_cast<char**>(keywords),
        &pot_obj, &df_obj, &af_obj))
    {
        PyErr_SetString(PyExc_ValueError,
            "GalaxyModel constructor takes two or three arguments: potential, df, [af]");
        return -1;
    }

    // check and store the potential
    if(!getPotential(pot_obj)) {
        PyErr_SetString(PyExc_TypeError, "Argument 'potential' must be a valid instance of Potential class");
        return -1;
    }
    Py_XDECREF(self->pot_obj);
    Py_INCREF(pot_obj);
    self->pot_obj = (PotentialObject*)pot_obj;

    // check and store the DF
    df::PtrDistributionFunction df = getDistributionFunction(df_obj);
    if(!df) {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'df' must be a valid instance of DistributionFunction class");
        return -1;
    }
    Py_XDECREF(self->df_obj);
    if(PyObject_TypeCheck(df_obj, &DistributionFunctionType))
    {   // it is a true Python DF object
        Py_INCREF(df_obj);
        self->df_obj = (DistributionFunctionObject*)df_obj;
    } else {
        // it is a Python function that was wrapped in a C++ class,
        // which now in turn will be wrapped in a new Python DF object
        self->df_obj = (DistributionFunctionObject*)createDistributionFunctionObject(df);
    }

    // af_obj might be NULL (then create a new one); if not NULL then check its validity
    if(af_obj!=NULL && (!PyObject_TypeCheck(af_obj, &ActionFinderType) ||
       ((ActionFinderObject*)af_obj)->af==NULL))
    {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'af' must be a valid instance of ActionFinder class "
            "corresponding to the given potential");
        return -1;
    }
    Py_XDECREF(self->af_obj);
    if(af_obj==NULL) {  // no action finder provided - create one internally
        PyObject *args = Py_BuildValue("(O)", pot_obj);
        self->af_obj = (ActionFinderObject*)PyObject_CallObject((PyObject*)&ActionFinderType, args);
        Py_DECREF(args);
        if(!self->af_obj)
            return -1;
    } else {  // use an existing action finder and increase its refcount
        Py_INCREF(af_obj);
        self->af_obj = (ActionFinderObject*)af_obj;
    }

    assert(GalaxyModel_isCorrect(self));
    return 0;
}

/// generate samples in position/velocity space
PyObject* GalaxyModel_sample_posvel(GalaxyModelObject* self, PyObject* args)
{
    if(!GalaxyModel_isCorrect(self))
        return NULL;
    int numPoints=0;
    if(!PyArg_ParseTuple(args, "i", &numPoints) || numPoints<=0)
    {
        PyErr_SetString(PyExc_ValueError, "sample() takes one integer argument - the number of points");
        return NULL;
    }
    try{
        // do the sampling
        galaxymodel::GalaxyModel galmod(*self->pot_obj->pot, *self->af_obj->af, *self->df_obj->df);
        particles::ParticleArrayCyl points = galaxymodel::generatePosVelSamples(galmod, numPoints);

        // convert output to NumPy array
        numPoints = points.size();
        npy_intp dims[] = {numPoints, 6};
        PyObject* posvel_arr = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        PyObject* mass_arr   = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        for(int i=0; i<numPoints; i++) {
            unconvertPosVel(coord::toPosVelCar(points.point(i)), &pyArrayElem<double>(posvel_arr, i, 0));
            pyArrayElem<double>(mass_arr, i) = points.mass(i) / conv->massUnit;
        }
        return Py_BuildValue("NN", posvel_arr, mass_arr);
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, 
            (std::string("Error in sample(): ")+e.what()).c_str());
        return NULL;
    }
}

/// \cond INTERNAL_DOCS
struct GalaxyModelParams{
    const galaxymodel::GalaxyModel model;
    bool needDens;
    bool needVel;
    bool needVel2;
    double vz_error;
    GalaxyModelParams(
        const potential::BasePotential& pot,
        const actions::BaseActionFinder& af,
        const df::BaseDistributionFunction& df,
        bool dens=true, bool vel=false, bool vel2=true) :
        model(pot, af, df), needDens(dens), needVel(vel), needVel2(vel2) {};
};
/// \endcond

void fncGalaxyModelMoments(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    GalaxyModelParams* params = static_cast<GalaxyModelParams*>(obj);
    double dens;
    coord::VelCyl vel;
    coord::Vel2Cyl vel2;
    try{
        computeMoments(params->model, coord::toPosCyl(point),
            params->needDens ? &dens : NULL,
            params->needVel  ? &vel  : NULL,
            params->needVel2 ? &vel2 : NULL/*, NULL, NULL, NULL,
            params->accuracy, params->maxNumEval*/);
    }
    catch(std::exception& e) {
        dens = NAN;
        vel.vR = vel.vz = vel.vphi = NAN;
        vel2.vR2 = vel2.vz2 = vel2.vphi2 = vel2.vRvz = vel2.vRvphi = vel2.vzvphi = NAN;
        utils::msg(utils::VL_WARNING, "GalaxyModel.moments", e.what());
    }
    unsigned int offset=0;
    if(params->needDens) {
        result[offset] = dens * pow_3(conv->lengthUnit) / conv->massUnit;  // dimension of density is M L^-3
        offset += 1;
    }
    if(params->needVel) {
        result[offset  ] = vel.vR   / conv->velocityUnit;
        result[offset+1] = vel.vz   / conv->velocityUnit;
        result[offset+2] = vel.vphi / conv->velocityUnit;
        offset += 3;
    }
    if(params->needVel2) {
        result[offset  ] = vel2.vR2    / pow_2(conv->velocityUnit);
        result[offset+1] = vel2.vz2    / pow_2(conv->velocityUnit);
        result[offset+2] = vel2.vphi2  / pow_2(conv->velocityUnit);
        result[offset+3] = vel2.vRvz   / pow_2(conv->velocityUnit);
        result[offset+4] = vel2.vRvphi / pow_2(conv->velocityUnit);
        result[offset+5] = vel2.vzvphi / pow_2(conv->velocityUnit);
    }
}

/// compute moments of DF at a given 3d point
PyObject* GalaxyModel_moments(GalaxyModelObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!GalaxyModel_isCorrect(self))
        return NULL;
    static const char* keywords[] = {"point","dens", "vel", "vel2", NULL};
    PyObject *points_obj = NULL, *dens_flag = NULL, *vel_flag = NULL, *vel2_flag = NULL;
    if(!PyArg_ParseTupleAndKeywords(
        args, namedArgs, "O|OOO", const_cast<char**>(keywords),
        &points_obj, &dens_flag, &vel_flag, &vel2_flag))
    {
        //PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to moments()");
        return NULL;
    }
    try{
        GalaxyModelParams params(*self->pot_obj->pot, *self->af_obj->af, *self->df_obj->df,
            dens_flag==NULL || PyObject_IsTrue(dens_flag),
            vel_flag !=NULL && PyObject_IsTrue(vel_flag),
            vel2_flag==NULL || PyObject_IsTrue(vel2_flag) );
        if(params.needDens) {
            if(params.needVel) {
                if(params.needVel2)
                    return callAnyFunctionOnArray
                    <INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET>
                    (&params, points_obj, fncGalaxyModelMoments);
                else
                    return callAnyFunctionOnArray
                    <INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE_AND_TRIPLET>
                    (&params, points_obj, fncGalaxyModelMoments);
            } else {
                if(params.needVel2)
                    return callAnyFunctionOnArray
                    <INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE_AND_SEXTET>
                    (&params, points_obj, fncGalaxyModelMoments);
                else
                    return callAnyFunctionOnArray
                    <INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE>
                    (&params, points_obj, fncGalaxyModelMoments);
            }
        } else {
            if(params.needVel) {
                if(params.needVel2)
                    return callAnyFunctionOnArray
                    <INPUT_VALUE_TRIPLET, OUTPUT_VALUE_TRIPLET_AND_SEXTET>
                    (&params, points_obj, fncGalaxyModelMoments);
                else
                    return callAnyFunctionOnArray
                    <INPUT_VALUE_TRIPLET, OUTPUT_VALUE_TRIPLET>
                    (&params, points_obj, fncGalaxyModelMoments);
            } else {
                if(params.needVel2)
                    return callAnyFunctionOnArray
                    <INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SEXTET>
                    (&params, points_obj, fncGalaxyModelMoments);
                else {
                    PyErr_SetString(PyExc_ValueError, "Nothing to compute!");
                    return NULL;
                }
            }
        }
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError,
            (std::string("Error in moments(): ")+e.what()).c_str());
        return NULL;
    }
}

void fncGalaxyModelProjectedMoments(void* obj, const double input[], double *result) {
    GalaxyModelParams* params = static_cast<GalaxyModelParams*>(obj);
    try{
        double surfaceDensity, losvdisp;
        computeProjectedMoments(params->model, input[0] * conv->lengthUnit,
            surfaceDensity, losvdisp/*, NULL, NULL, params->accuracy, params->maxNumEval*/);
        result[0] = surfaceDensity * pow_2(conv->lengthUnit) / conv->massUnit;
        result[1] = losvdisp / pow_2(conv->velocityUnit);
    }
    catch(std::exception& ) {
        result[0] = result[1] = NAN;
    }
}

/// compute projected moments of distribution function
PyObject* GalaxyModel_projectedMoments(GalaxyModelObject* self, PyObject* args)
{
    if(!GalaxyModel_isCorrect(self))
        return NULL;
    PyObject *points_obj = NULL;
    if(!PyArg_ParseTuple(args, "O", &points_obj))
    {
        //PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to projectedMoments()");
        return NULL;
    }
    try{
        GalaxyModelParams params(*self->pot_obj->pot, *self->af_obj->af, *self->df_obj->df);
        return callAnyFunctionOnArray<INPUT_VALUE_SINGLE, OUTPUT_VALUE_SINGLE_AND_SINGLE>
            (&params, points_obj, fncGalaxyModelProjectedMoments);
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError,
            (std::string("Error in projectedMoments(): ")+e.what()).c_str());
        return NULL;
    }
}

void fncGalaxyModelProjectedDF(void* obj, const double input[], double *result) {
    const double R = sqrt(pow_2(input[0]) + pow_2(input[1])) * conv->lengthUnit;
    const double vz = input[2] * conv->velocityUnit;
    // dimension of projected distribution function is M L^-2 V^-1
    const double dim = conv->velocityUnit * pow_2(conv->lengthUnit) / conv->massUnit;
    GalaxyModelParams* params = static_cast<GalaxyModelParams*>(obj);
    try{
        result[0] = computeProjectedDF(params->model, R, vz, params->vz_error) * dim;
    }
    catch(std::exception& ) {
        result[0] = NAN;
    }
}

/// compute projected distribution function
PyObject* GalaxyModel_projectedDF(GalaxyModelObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!GalaxyModel_isCorrect(self))
        return NULL;
    static const char* keywords[] = {"point", "vz_error", NULL};
    PyObject *points_obj = NULL;
    double vz_error = 0;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "O|d", const_cast<char**>(keywords),
        &points_obj, &vz_error))
    {
        //PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to projectedDF()");
        return NULL;
    }
    try{
        GalaxyModelParams params(*self->pot_obj->pot, *self->af_obj->af, *self->df_obj->df);
        params.vz_error = vz_error * conv->velocityUnit;
        PyObject* result = callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE>
            (&params, points_obj, fncGalaxyModelProjectedDF);
        return result;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError,
            (std::string("Error in projectedDF(): ")+e.what()).c_str());
        return NULL;
    }
}

bool computeVDFatPoint(const galaxymodel::GalaxyModel& model, const coord::PosCyl& point,
    const bool projected, const std::vector<double>& gridvR_ext,
    const std::vector<double>& gridvz_ext, const std::vector<double>& gridvphi_ext,
    /*storage for output interpolators */ PyObject*& splinevR, PyObject*& splinevz, PyObject*& splinevphi)
{
    try{
        // create a default grid in velocity space (if not provided by the user), in internal units
        double v_max = sqrt(-2*model.potential.value(point));
        std::vector<double> defaultgrid =
            math::mirrorGrid(math::createNonuniformGrid(51, v_max*0.01, v_max, true));

        // convert the user-provided grids to internal units, or use the default grid
        std::vector<double> gridvR(defaultgrid), gridvz(defaultgrid), gridvphi(defaultgrid);
        if(!gridvR_ext.empty()) {
            gridvR.resize(gridvR_ext.size());
            for(unsigned int i=0; i<gridvR_ext.size(); i++)
                gridvR[i] = gridvR_ext[i] * conv->velocityUnit;
        }
        if(!gridvz_ext.empty()) {
            gridvz.resize(gridvz_ext.size());
            for(unsigned int i=0; i<gridvz_ext.size(); i++)
                gridvz[i] = gridvz_ext[i] * conv->velocityUnit;
        }
        if(!gridvphi_ext.empty()) {
            gridvphi.resize(gridvphi_ext.size());
            for(unsigned int i=0; i<gridvphi_ext.size(); i++)
                gridvphi[i] = gridvphi_ext[i] * conv->velocityUnit;
        }

        // compute the distributions
        std::vector<double> amplvR, amplvz, amplvphi;
        const int ORDER = 3;
        math::BsplineInterpolator1d<ORDER> intvR(gridvR), intvz(gridvz), intvphi(gridvphi);
        galaxymodel::computeVelocityDistribution<ORDER>(model, point, projected,
            gridvR, gridvz, gridvphi, /*output*/ amplvR, amplvz, amplvphi,
            /*accuracy*/ 1e-2, /*maxNumEval*/1e6);

        // convert the units for the abscissae (velocity)
        for(unsigned int i=0; i<gridvR.  size(); i++)
            gridvR  [i] /= conv->velocityUnit;
        for(unsigned int i=0; i<gridvz.  size(); i++)
            gridvz  [i] /= conv->velocityUnit;
        for(unsigned int i=0; i<gridvphi.size(); i++)
            gridvphi[i] /= conv->velocityUnit;

        // convert the units for the ordinates (f(v) ~ 1/velocity)
        for(unsigned int i=0; i<amplvR.  size(); i++)
            amplvR  [i] *= conv->velocityUnit;
        for(unsigned int i=0; i<amplvz.  size(); i++)
            amplvz  [i] *= conv->velocityUnit;
        for(unsigned int i=0; i<amplvphi.size(); i++)
            amplvphi[i] *= conv->velocityUnit;

        // construct three interpolating spline objects
        splinevR   = createCubicSpline(gridvR,   amplvR);
        splinevz   = createCubicSpline(gridvz,   amplvz);
        splinevphi = createCubicSpline(gridvphi, amplvphi);
        return true;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError,
            (std::string("Error in vdf(): ")+e.what()).c_str());
        return false;
    }
}

/// compute velocity distribution functions at point(s)
PyObject* GalaxyModel_vdf(GalaxyModelObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!GalaxyModel_isCorrect(self))
        return NULL;
    static const char* keywords[] = {"point", "gridvR", "gridvz", "gridvphi", NULL};
    PyObject *points_obj = NULL, *gridvR_obj = NULL, *gridvz_obj = NULL, *gridvphi_obj = NULL;
    if(!PyArg_ParseTupleAndKeywords(
        args, namedArgs, "O|OOO", const_cast<char**>(keywords),
        &points_obj, &gridvR_obj, &gridvz_obj, &gridvphi_obj))
    {
        //PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to vdf()");
        return NULL;
    }

    // retrieve the input point(s)
    PyArrayObject *points_arr =
        (PyArrayObject*) PyArray_FROM_OTF(points_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    npy_intp npoints = 0;  // # of points at which the VDFs should be computed
    npy_intp ndim    = 0;  // dimensions of points: 2 for projected VDF at (R,phi), 3 for (R,phi,z)
    if(points_arr) {
        if(PyArray_NDIM(points_arr) == 1) {
            ndim    = PyArray_DIM(points_arr, 0);
            npoints = 1;
        } else if(PyArray_NDIM(points_arr) == 2) {
            ndim    = PyArray_DIM(points_arr, 1);
            npoints = PyArray_DIM(points_arr, 0);
        }
    }
    if(npoints==0 || !(ndim==2 || ndim==3)) {
        Py_XDECREF(points_arr);
        PyErr_SetString(PyExc_ValueError, "Argument 'point' should be a 2d/3d point or an array of points");
        return NULL;
    }

    // retrieve the input grids, if they are provided
    std::vector<double> gridvR_arr   = toDoubleArray(gridvR_obj);   // empty array if gridvR_obj==NULL
    std::vector<double> gridvz_arr   = gridvz_obj   ? toDoubleArray(gridvz_obj)   : gridvR_arr;
    std::vector<double> gridvphi_arr = gridvphi_obj ? toDoubleArray(gridvphi_obj) : gridvR_arr;

    galaxymodel::GalaxyModel model(*self->pot_obj->pot, *self->af_obj->af, *self->df_obj->df);
    bool allok = true;

    // in the case of several input points, the output will contain three arrays with spline objects
    PyObject *splvR = NULL, *splvz = NULL, *splvphi = NULL;
    if(npoints>1) {
        splvR   = PyArray_SimpleNew(1, &npoints, NPY_OBJECT);
        splvz   = PyArray_SimpleNew(1, &npoints, NPY_OBJECT);
        splvphi = PyArray_SimpleNew(1, &npoints, NPY_OBJECT);
        if(!splvR || !splvz || !splvphi) {
            Py_DECREF(points_arr);
            Py_XDECREF(splvR);
            Py_XDECREF(splvz);
            Py_XDECREF(splvphi);
            return NULL;
        }
        // initialize the arrays with NULL pointers
        for(int ind=0; ind<npoints; ind++) {
            pyArrayElem<PyObject*>(splvR,   ind) = NULL;
            pyArrayElem<PyObject*>(splvz,   ind) = NULL;
            pyArrayElem<PyObject*>(splvphi, ind) = NULL;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
        for(int ind=0; ind<npoints; ind++) {
            const coord::PosCar point(
                pyArrayElem<double>(points_arr, ind, 0) * conv->lengthUnit,
                pyArrayElem<double>(points_arr, ind, 1) * conv->lengthUnit,
                ndim==3 ? pyArrayElem<double>(points_arr, ind, 2) * conv->lengthUnit : 0);
            if(allok) {  // if even a single point failed, don't continue
                if(!computeVDFatPoint(model, toPosCyl(point), ndim==2,
                    gridvR_arr, gridvz_arr, gridvphi_arr,
                    pyArrayElem<PyObject*>(splvR,   ind),
                    pyArrayElem<PyObject*>(splvz,   ind),
                    pyArrayElem<PyObject*>(splvphi, ind)))
                        allok = false;
            }
        }
        // if something went wrong, release any successfully created objects
        if(!allok) {
            for(int ind=npoints-1; ind>=0; ind--) {
                Py_XDECREF(pyArrayElem<PyObject*>(splvR,   ind));
                Py_XDECREF(pyArrayElem<PyObject*>(splvz,   ind));
                Py_XDECREF(pyArrayElem<PyObject*>(splvphi, ind));
            }
        }
    } else {   // a single input point -- don't create arrays, just construct three PyObject's
        const coord::PosCar point(
            pyArrayElem<double>(points_arr, 0) * conv->lengthUnit,
            pyArrayElem<double>(points_arr, 1) * conv->lengthUnit,
            ndim==3 ? pyArrayElem<double>(points_arr, 2) * conv->lengthUnit : 0);
        allok &= computeVDFatPoint(model, toPosCyl(point), ndim==2,
            gridvR_arr, gridvz_arr, gridvphi_arr,
            /*output*/ splvR, splvz, splvphi);
    }
    Py_DECREF(points_arr);
    if(!allok) {
        Py_XDECREF(splvR);
        Py_XDECREF(splvz);
        Py_XDECREF(splvphi);
        return NULL;
    }
    return Py_BuildValue("NNN", splvR, splvz, splvphi);
}

static PyMemberDef GalaxyModel_members[] = {
    { const_cast<char*>("potential"), T_OBJECT_EX, offsetof(GalaxyModelObject, pot_obj), READONLY,
      const_cast<char*>("Potential (read-only)") },
    { const_cast<char*>("af"),  T_OBJECT_EX, offsetof(GalaxyModelObject, af_obj ), READONLY,
      const_cast<char*>("Action finder (read-only)") },
    { const_cast<char*>("df"),  T_OBJECT_EX, offsetof(GalaxyModelObject, df_obj ), READONLY,
      const_cast<char*>("Distribution function (read-only)") },
    { NULL }
};

static PyMethodDef GalaxyModel_methods[] = {
    { "sample", (PyCFunction)GalaxyModel_sample_posvel, METH_VARARGS,
      "Sample distribution function in the given potential by N particles.\n"
      "Arguments:\n"
      "  Number of particles to sample.\n"
      "Returns:\n"
      "  A tuple of two arrays: position/velocity (2d array of size Nx6) and mass (1d array of length N)." },
    { "moments", (PyCFunction)GalaxyModel_moments, METH_VARARGS | METH_KEYWORDS,
      "Compute moments of distribution function in the given potential.\n"
      "Arguments:\n"
      "  point -- a single point or an array of points specifying the position "
      "in cartesian coordinates at which the moments need to be computed "
      "(a triplet of numbers or an Nx3 array);\n"
      "  dens (boolean, default True)  -- flag telling whether the density (0th moment) "
      "needs to be computed;\n"
      "  vel  (boolean, default False) -- same for streaming velocity (1st moment);\n"
      "  vel2 (boolean, default True)  -- same for 2nd moment of velocity.\n"
      "Returns:\n"
      "  For each input point, return the requested moments (one value for density, "
      "a triplet for velocity, and 6 components of the 2nd moment tensor)." },
    { "projectedMoments", (PyCFunction)GalaxyModel_projectedMoments, METH_VARARGS,
      "Compute projected moments of distribution function in the given potential.\n"
      "Arguments:\n"
      "  A single value or an array of values of cylindrical radius at which to compute moments.\n"
      "Returns:\n"
      "  A tuple of two floats or arrays: surface density and line-of-sight velocity dispersion "
      "at each input radius.\n" },
    { "projectedDF", (PyCFunction)GalaxyModel_projectedDF, METH_VARARGS | METH_KEYWORDS,
      "Compute projected distribution function (integrated over z-coordinate and x- and y-velocities)\n"
      "Named arguments:\n"
      "  point -- a single point or an array of points specifying the x,y- components of position "
      "in cartesian coordinates and z-component of velocity "
      "(a triplet of numbers or an Nx3 array);\n"
      "  vz_error -- optional error on z-component of velocity "
      "(DF will be convolved with a Gaussian if this error is non-zero)\n"
      "Returns:\n"
      "  The value of projected DF (integrated over the missing components of position and velocity) "
      "at each point." },
    { "vdf", (PyCFunction)GalaxyModel_vdf, METH_VARARGS | METH_KEYWORDS,
      "Compute the velocity distribution functions in three directions at one or several "
      "points in 3d (x,y,z), or projected velocity distributions at the given 2d points (x,y), "
      "integrated over z.\n"
      "Arguments:\n"
      "  point -- a single point or an array of points specifying the position "
      "in cartesian coordinates (x,y,z) in the case of intrinsic VDF, or the (x,y) components "
      "of position in the case of projected VDF (a Nx3 or Nx2 array); \n"
      "  gridvR -- (optional) array of grid points in the velocity space, specifying the x-nodes "
      "of the interpolating cubic spline for f(v_R). Typically the grid should span the range "
      "(-v_escape, +v_escape), although it might be narrower; the nodes do not need to be equally "
      "spaced. If this argument is omitted, a default grid with 100 nodes and a denser spacing "
      "near origin will be used. Note that in the case of user-provided grid, it will be used for "
      "all input points, while the automatic grid will be scaled with local escape velocity.\n"
      "  gridvz -- (optional) same for the interpolated f(v_z); if omitted, gridvR is used instead.\n"
      "  gridvphi -- (optional) same for f(v_phi), with gridvR as default.\n"
      "Returns:\n"
      "  A tuple of three functions (in case of one input point) or arrays of functions (if N>1), "
      "which represent spline-interpolated VDFs f(v_R), f(v_z), f(v_phi) at each input point. "
      "Note that the points are specified in cartesian coordinates but the VDFs are given in "
      "terms of velocity components in cylindrical coordinates. "
      "Also keep in mind that the interpolated values may be negative, especially at the wings of "
      "distribution, and that by default the spline is linearly extrapolated beyond its domain; "
      "to extrapolate as zero use `f(v, ext=False)` when evaluating the spline function.\n"
      "The VDFs are normalized such that the integral of f(v_k) d v_k  over the interval "
      "(-v_escape, v_escape) is unity for each component v_k." },
    { NULL }
};

static PyTypeObject GalaxyModelType = {
    PyObject_HEAD_INIT(NULL)
    0, "agama.GalaxyModel",
    sizeof(GalaxyModelObject), 0, (destructor)GalaxyModel_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringGalaxyModel,
    0, 0, 0, 0, 0, 0, GalaxyModel_methods, GalaxyModel_members, 0, 0, 0, 0, 0, 0,
    (initproc)GalaxyModel_init
};


///@}
//  -----------------------------------------------
/// \name  Component class for SelfConsistentModel
//  -----------------------------------------------
///@{

/// \cond INTERNAL_DOCS
/// Python type corresponding to Component class
typedef struct {
    PyObject_HEAD
    galaxymodel::PtrComponent comp;
    const char* name;
} ComponentObject;
/// \endcond

void Component_dealloc(ComponentObject* self)
{
    if(self->comp)
        utils::msg(utils::VL_VERBOSE, "Agama", "Deleted " + std::string(self->name) + " at " +
            utils::toString(self->comp.get()));
    else
        utils::msg(utils::VL_VERBOSE, "Agama", "Deleted an empty component");
    self->comp.reset();
    // self->name is either NULL or points to a constant string that does not require deallocation
    self->ob_type->tp_free(self);
}

static const char* docstringComponent = 
    "Represents a single component of a self-consistent model.\n"
    "It can be either a static component with a fixed density or potential profile, "
    "or a DF-based component whose density profile is recomputed iteratively "
    "in the self-consistent modelling procedure.\n"
    "Constructor takes only named arguments:\n"
    "  df --  an instance of DistributionFunction class for a dynamically-updated component;\n"
    "  if not provided then the component is assumed to be static.\n"
    "  potential --  an instance of Potential class for a static component with a known potential;\n"
    "  it is mutually exclusive with the 'df' argument.\n"
    "  density --  an object providing a Density interface (e.g., an instance of "
    "Density or Potential class) that specifies the initial guess for the density profile "
    "for DF-based components (needed to compute the potential on the first iteration), "
    "or a fixed density profile for a static component (optional, and may be combined with "
    "the 'potential' argument).\n"
    "  disklike (boolean) --  a flag tagging the density profile to be attributed to either "
    "the CylSpline or Multipole potential expansions in the SelfConsistentModel "
    "(required for DF-based components and for fixed components specified by their density).\n"
    "  Depending on this flag, other arguments must be provided.\n"
    "  For spheroidal components:\n"
    "    rminSph, rmaxSph --  inner- and outermost radii of the logarithmic radial grid.\n"
    "    sizeRadialSph --  the number of nodes in the radial grid.\n"
    "    lmaxAngularSph --  the order of expansion in angular harmonics.\n"
    "  For disklike components:\n"
    "    gridR, gridz (array-like) --  the nodes of 2d grid in cylindrical coordinates, "
    "first elements must be zeros (may be constructed using 'agama.createNonuniformGrid()').\n";

int Component_init(ComponentObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!onlyNamedArgs(args, namedArgs))
        return -1;
    // check if a potential object was provided
    PyObject* pot_obj = getItemFromPyDict(namedArgs, "potential");
    potential::PtrPotential pot = getPotential(pot_obj);
    if(pot_obj!=NULL && !pot) {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'potential' must be a valid instance of Potential class");
        return -1;
    }
    // check if a density object was provided
    PyObject* dens_obj = getItemFromPyDict(namedArgs, "density");
    potential::PtrDensity dens = getDensity(dens_obj);
    if(dens_obj!=NULL && !dens) {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'density' must be a valid Density instance");
        return -1;
    }
    // check if a df object was provided
    PyObject* df_obj = getItemFromPyDict(namedArgs, "df");
    df::PtrDistributionFunction df = getDistributionFunction(df_obj);
    if(df_obj!=NULL && !df) {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'df' must be a valid instance of DistributionFunction class");
        return -1;
    }
    // check if a 'disklike' flag was provided
    PyObject* disklike_obj = getItemFromPyDict(namedArgs, "disklike");
    int disklike = disklike_obj ? PyObject_IsTrue(disklike_obj) : -1;

    // choose the variant of component: static or DF-based
    if((pot_obj!=NULL && df_obj!=NULL) || (pot_obj==NULL && df_obj==NULL && dens_obj==NULL)) {
        PyErr_SetString(PyExc_ValueError,
            "Should provide either a 'potential' and/or 'density' argument for a static component, "
            "or a 'df' argument for a component specified by a distribution function");
        return -1;
    }
    // if density and/or DF is provided, it must be tagged to be either disk-like or spheroidal
    if((dens!=NULL || df!=NULL) && disklike == -1) {
        PyErr_SetString(PyExc_ValueError, "Should provide a boolean argument 'disklike'");
        return -1;
    }
    if(!df_obj) {   // static component with potential and optionally density
        try {
            if(!dens)  // only potential
                self->comp.reset(new galaxymodel::ComponentStatic(pot));
            else       // both potential and density
                self->comp.reset(new galaxymodel::ComponentStatic(dens, disklike, pot));
            self->name = "Static component";
            utils::msg(utils::VL_VERBOSE, "Agama", "Created a " + std::string(self->name) + " at "+
                utils::toString(self->comp.get()));
            return 0;
        }
        catch(std::exception& e) {
            PyErr_SetString(PyExc_ValueError,
                (std::string("Error in creating a static component: ")+e.what()).c_str());
            return -1;
        }
    } else if(disklike == 0) {   // spheroidal component
        double rmin = toDouble(getItemFromPyDict(namedArgs, "rminSph"), -1) * conv->lengthUnit;
        double rmax = toDouble(getItemFromPyDict(namedArgs, "rmaxSph"), -1) * conv->lengthUnit;
        int numRad  = toInt(getItemFromPyDict(namedArgs, "sizeRadialSph"), -1);
        int numAng  = toInt(getItemFromPyDict(namedArgs, "lmaxAngularSph"), -1);
        if(rmin<=0 || rmax<=rmin || numRad<=0 || numAng<0) {
            PyErr_SetString(PyExc_ValueError,
                "For spheroidal components, should provide correct values for the following arguments: "
                "rminSph, rmaxSph, sizeRadialSph, lmaxAngularSph");
            return -1;
        }
        try {
            self->comp.reset(new galaxymodel::ComponentWithSpheroidalDF(
                df, dens, rmin, rmax, numRad, numAng));
            self->name = "Spheroidal component";
            utils::msg(utils::VL_VERBOSE, "Agama", "Created a " + std::string(self->name) + " at "+
                utils::toString(self->comp.get()));
            return 0;
        }
        catch(std::exception& e) {
            PyErr_SetString(PyExc_ValueError,
                (std::string("Error in creating a spheroidal component: ")+e.what()).c_str());
            return -1;
        }
    } else {   // disk-like component
        std::vector<double> gridR(toDoubleArray(getItemFromPyDict(namedArgs, "gridR")));
        std::vector<double> gridz(toDoubleArray(getItemFromPyDict(namedArgs, "gridz")));
        math::blas_dmul(conv->lengthUnit, gridR);
        math::blas_dmul(conv->lengthUnit, gridz);
        if(gridR.empty() || gridz.empty()) {
            PyErr_SetString(PyExc_ValueError,
                "For disklike components, should provide two array arguments: gridR, gridz");
            return -1;
        }
        try {
            self->comp.reset(new galaxymodel::ComponentWithDisklikeDF(
                df, dens, gridR, gridz));
            self->name = "Disklike component";
            utils::msg(utils::VL_VERBOSE, "Agama", "Created a " + std::string(self->name) + " at "+
                utils::toString(self->comp.get()));
            return 0;
        }
        catch(std::exception& e) {
            PyErr_SetString(PyExc_ValueError,
                (std::string("Error in creating a disklike component: ")+e.what()).c_str());
            return -1;
        }
    }
    // shouldn't reach here
    PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to Component()");
    return -1;
}

PyObject* Component_name(PyObject* self)
{
    return Py_BuildValue("s", ((ComponentObject*)self)->name);
}

PyObject* Component_getPotential(ComponentObject* self)
{
    potential::PtrPotential pot = self->comp->getPotential();
    if(pot)
        return createPotentialObject(pot);
    // otherwise no potential is available (e.g. for a df-based component)
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* Component_getDensity(ComponentObject* self)
{
    potential::PtrDensity dens = self->comp->getDensity();
    if(dens)
        return createDensityObject(dens);
    // otherwise no density is available (e.g. for a static component)
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef Component_methods[] = {
    { "getPotential", (PyCFunction)Component_getPotential, METH_NOARGS,
      "Return a Potential object associated with a static component, or None.\n"
      "No arguments.\n" },
    { "getDensity", (PyCFunction)Component_getDensity, METH_NOARGS,
      "Return a Density object representing the fixed density profile for a static component "
      "(or None if it has only a potential profile), "
      "or the current density profile from the previous iteration of the self-consistent "
      "modelling procedure for a DF-based component.\n"
      "No arguments.\n" },
    { NULL }
};

static PyTypeObject ComponentType = {
    PyObject_HEAD_INIT(NULL)
    0, "agama.Component",
    sizeof(ComponentObject), 0, (destructor)Component_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Component_name, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringComponent,
    0, 0, 0, 0, 0, 0, Component_methods, 0, 0, 0, 0, 0, 0, 0,
    (initproc)Component_init
};


///@}
//  ---------------------------------
/// \name  SelfConsistentModel class
//  ---------------------------------
///@{

/// \cond INTERNAL_DOCS
/// Python type corresponding to SelfConsistentModel class
typedef struct {
    PyObject_HEAD
    PyObject* components;
    PotentialObject* pot;
    ActionFinderObject* af;
    /// members of galaxymodel::SelfConsistentModel structure listed here
    double rminSph, rmaxSph;      ///< range of radii for the logarithmic grid
    unsigned int sizeRadialSph;   ///< number of grid points in radius
    unsigned int lmaxAngularSph;  ///< maximum order of angular-harmonic expansion (l_max)
    double RminCyl, RmaxCyl;      ///< innermost (non-zero) and outermost grid nodes in cylindrical radius
    double zminCyl, zmaxCyl;      ///< innermost and outermost grid nodes in vertical direction
    unsigned int sizeRadialCyl;   ///< number of grid nodes in cylindrical radius
    unsigned int sizeVerticalCyl; ///< number of grid nodes in vertical (z) direction
} SelfConsistentModelObject;
/// \endcond

void SelfConsistentModel_dealloc(SelfConsistentModelObject* self)
{
    Py_XDECREF(self->components);
    Py_XDECREF(self->pot);
    Py_XDECREF(self->af);
    self->ob_type->tp_free(self);
}

static const char* docstringSelfConsistentModel =
    "A class for performing self-consistent modelling procedure.\n"
    "A full model consists of one or more instances of Component class "
    "representing either static density or potential profiles, or distribution function-based "
    "components with iteratively recomputed density profiles, plus the overall potential "
    "and the associated action finder object.\n"
    "The SelfConsistentModel object contains parameters for two kinds of potential "
    "expansions used in the procedure -- Multipole potential for spheroidal components "
    "and CylSpline potential for disk-like components, the list of Component objects, "
    "and read-only references to the total potential and the action finder.\n"
    "The constructor takes named arguments describing the potential expansion parameters -- "
    "a full list is given by 'dir(SelfConsistentModel)', and they may be modified at any time.\n"
    "The list of components is initially empty and should be filled by the user; "
    "it may also be modified between iterations.\n"
    "The potential and action finder member variables are initially empty, "
    "and are initialized after the first call to the 'iterate()' method.\n";

int SelfConsistentModel_init(SelfConsistentModelObject* self, PyObject* args, PyObject* namedArgs)
{
    Py_XDECREF(self->components);
    Py_XDECREF(self->pot);
    Py_XDECREF(self->af);
    if(!onlyNamedArgs(args, namedArgs))
        return -1;
    // allocate a new empty list of components, but not potential or action finder
    self->components  = PyList_New(0);
    self->pot         = NULL;
    self->af          = NULL;
    self->rminSph     = toDouble(getItemFromPyDict(namedArgs, "rminSph"), -2);
    self->rmaxSph     = toDouble(getItemFromPyDict(namedArgs, "rmaxSph"), -2);
    self->sizeRadialSph  = toInt(getItemFromPyDict(namedArgs, "sizeRadialSph"), -1);
    self->lmaxAngularSph = toInt(getItemFromPyDict(namedArgs, "lmaxAngularSph"), -1);
    self->RminCyl     = toDouble(getItemFromPyDict(namedArgs, "RminCyl"), -1);
    self->RmaxCyl     = toDouble(getItemFromPyDict(namedArgs, "RmaxCyl"), -1);
    self->zminCyl     = toDouble(getItemFromPyDict(namedArgs, "zminCyl"), -1);
    self->zmaxCyl     = toDouble(getItemFromPyDict(namedArgs, "zmaxCyl"), -1);
    self->sizeRadialCyl  = toInt(getItemFromPyDict(namedArgs, "sizeRadialCyl"), -1);
    self->sizeVerticalCyl= toInt(getItemFromPyDict(namedArgs, "sizeVerticalCyl"), -1);
    return 0;
}

PyObject* SelfConsistentModel_iterate(SelfConsistentModelObject* self)
{
    galaxymodel::SelfConsistentModel model;
    // parse the Python list of components
    if(self->components==NULL || !PyList_Check(self->components) || PyList_Size(self->components)==0)
    {
        PyErr_SetString(PyExc_ValueError,
            "SelfConsistentModel.components should be a non-empty list of Component objects");
        return NULL;
    }
    int numComp = PyList_Size(self->components);
    for(int i=0; i<numComp; i++)
    {
        PyObject* elem = PyList_GetItem(self->components, i);
        if(!PyObject_TypeCheck(elem, &ComponentType)) {
            PyErr_SetString(PyExc_ValueError,
                "SelfConsistentModel.components should contain only Component objects");
            return NULL;
        }
        model.components.push_back(((ComponentObject*)elem)->comp);
    }
    model.rminSph = self->rminSph * conv->lengthUnit;
    model.rmaxSph = self->rmaxSph * conv->lengthUnit;
    model.sizeRadialSph = self->sizeRadialSph;
    model.lmaxAngularSph = self->lmaxAngularSph;
    model.RminCyl = self->RminCyl * conv->lengthUnit;
    model.RmaxCyl = self->RmaxCyl * conv->lengthUnit;
    model.zminCyl = self->zminCyl * conv->lengthUnit;
    model.zmaxCyl = self->zmaxCyl * conv->lengthUnit;
    model.sizeRadialCyl = self->sizeRadialCyl;
    model.sizeVerticalCyl = self->sizeVerticalCyl;
    if(self->pot!=NULL && PyObject_TypeCheck(self->pot, &PotentialType))
        model.totalPotential = ((PotentialObject*)self->pot)->pot;
    if(self->af!=NULL && PyObject_TypeCheck(self->af, &ActionFinderType))
        model.actionFinder = ((ActionFinderObject*)self->af)->af;
    try {
        doIteration(model);
        // update the total potential and action finder by copying the C++ smart pointers into
        // Python objects; old Python objects are released (and destroyed if no one else uses them)
        Py_XDECREF(self->pot);
        Py_XDECREF(self->af);
        self->pot = (PotentialObject*)createPotentialObject(model.totalPotential);
        self->af  = (ActionFinderObject*)createActionFinderObject(model.actionFinder);
        Py_INCREF(Py_None);
        return Py_None;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError,
            (std::string("Error in SelfConsistentModel.iterate(): ")+e.what()).c_str());
        return NULL;
    }
}

static PyMemberDef SelfConsistentModel_members[] = {
    { const_cast<char*>("components"), T_OBJECT_EX, offsetof(SelfConsistentModelObject, components), 0,
      const_cast<char*>("List of Component objects (may be modified by the user, but should be "
      "non-empty and contain only instances of Component class upon a call to 'iterate()' method)") },
    { const_cast<char*>("potential"), T_OBJECT, offsetof(SelfConsistentModelObject, pot), READONLY,
      const_cast<char*>("Total potential of the model (read-only)") },
    { const_cast<char*>("af"), T_OBJECT, offsetof(SelfConsistentModelObject, af), READONLY,
      const_cast<char*>("Action finder associated with the total potential (read-only)") },
    { const_cast<char*>("rminSph"), T_DOUBLE, offsetof(SelfConsistentModelObject, rminSph), 0,
      const_cast<char*>("Spherical radius of innermost grid node for Multipole potential") },
    { const_cast<char*>("rmaxSph"), T_DOUBLE, offsetof(SelfConsistentModelObject, rmaxSph), 0,
      const_cast<char*>("Spherical radius of outermost grid node for Multipole potential") },
    { const_cast<char*>("sizeRadialSph"), T_INT, offsetof(SelfConsistentModelObject, sizeRadialSph), 0,
      const_cast<char*>("Number of radial grid points for Multipole potential") },
    { const_cast<char*>("lmaxAngularSph"), T_INT, offsetof(SelfConsistentModelObject, lmaxAngularSph), 0,
      const_cast<char*>("Order of angular-harmonic expansion for Multipole potential") },
    { const_cast<char*>("RminCyl"), T_DOUBLE, offsetof(SelfConsistentModelObject, RminCyl), 0,
      const_cast<char*>("Cylindrical radius of first (nonzero) grid node for CylSpline potential") },
    { const_cast<char*>("RmaxCyl"), T_DOUBLE, offsetof(SelfConsistentModelObject, RmaxCyl), 0,
      const_cast<char*>("Cylindrical radius of outermost grid node for CylSpline potential") },
    { const_cast<char*>("zminCyl"), T_DOUBLE, offsetof(SelfConsistentModelObject, zminCyl), 0,
      const_cast<char*>("z-coordinate of first (nonzero) grid node for CylSpline potential") },
    { const_cast<char*>("zmaxCyl"), T_DOUBLE, offsetof(SelfConsistentModelObject, zmaxCyl), 0,
      const_cast<char*>("z-coordinate of outermost grid node for CylSpline potential") },
    { const_cast<char*>("sizeRadialCyl"), T_INT, offsetof(SelfConsistentModelObject, sizeRadialCyl), 0,
      const_cast<char*>("Grid size in cylindrical radius for CylSpline potential") },
    { const_cast<char*>("sizeVerticalCyl"), T_INT, offsetof(SelfConsistentModelObject, sizeVerticalCyl), 0,
      const_cast<char*>("Grid size in z-coordinate for CylSpline potential") },
    { NULL }
};

static PyMethodDef SelfConsistentModel_methods[] = {
    { "iterate", (PyCFunction)SelfConsistentModel_iterate, METH_NOARGS,
      "Perform one iteration of self-consistent modelling procedure, "
      "recomputing density profiles of all DF-based components, "
      "and then updating the total potential.\n" },
    { NULL }
};

static PyTypeObject SelfConsistentModelType = {
    PyObject_HEAD_INIT(NULL)
    0, "agama.SelfConsistentModel",
    sizeof(SelfConsistentModelObject), 0, (destructor)SelfConsistentModel_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringSelfConsistentModel,
    0, 0, 0, 0, 0, 0, SelfConsistentModel_methods, SelfConsistentModel_members, 0, 0, 0, 0, 0, 0,
    (initproc)SelfConsistentModel_init
};


///@}
//  ----------------------------------------
/// \name  Target class for additive models
//  ----------------------------------------
///@{


/// \cond INTERNAL_DOCS
/// Python type corresponding to Target class
typedef struct {
    PyObject_HEAD
    galaxymodel::PtrTarget target;
} TargetObject;
/// \endcond

void Target_dealloc(TargetObject* self)
{
    if(self->target)
        utils::msg(utils::VL_VERBOSE, "Agama", "Deleted " + std::string(self->target->name()) +
            " target at " + utils::toString(self->target.get()));
    else
        utils::msg(utils::VL_VERBOSE, "Agama", "Deleted an empty target");
    self->target.reset();
    self->ob_type->tp_free(self);
}

static const char* docstringTarget =
    "Target objects represent various targets that need to be satisfied by an additive model.\n";

int Target_init(TargetObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!onlyNamedArgs(args, namedArgs))
        return -1;
    // check if a potential object was provided
    PyObject* pot_obj = getItemFromPyDict(namedArgs, "potential");
    potential::PtrPotential pot = getPotential(pot_obj);
    if(pot_obj!=NULL && !pot) {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'potential' must be a valid instance of Potential class");
        return -1;
    }
    // check if a density object was provided
    PyObject* dens_obj = getItemFromPyDict(namedArgs, "density");
    potential::PtrDensity dens = getDensity(dens_obj);
    if(dens_obj!=NULL && !dens) {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'density' must be a valid Density instance");
        return -1;
    }

    PyObject* type_obj = getItemFromPyDict(namedArgs, "type");
    if(type_obj==NULL || !PyString_Check(type_obj)) {
        PyErr_SetString(PyExc_ValueError, "Must provide a type='...' argument");
        return -1;
    }
    std::string type_str(PyString_AsString(type_obj));
    try{
        // check if a DensityGrid is requested
        galaxymodel::DensityGridParams densParams;
        densParams.type = galaxymodel::DG_UNKNOWN;
        if(utils::stringsEqual(type_str, "DensityClassicTopHat"))
            densParams.type = galaxymodel::DG_CLASSIC_TOPHAT;
        if(utils::stringsEqual(type_str, "DensityClassicLinear"))
            densParams.type = galaxymodel::DG_CLASSIC_LINEAR;
        if(utils::stringsEqual(type_str, "DensitySphHarm"))
            densParams.type = galaxymodel::DG_SPH_HARM;
        if(utils::stringsEqual(type_str, "DensityCylindricalTopHat"))
            densParams.type = galaxymodel::DG_CYLINDRICAL_TOPHAT;
        if(utils::stringsEqual(type_str, "DensityCylindricalLinear"))
            densParams.type = galaxymodel::DG_CYLINDRICAL_LINEAR;
        if(densParams.type != galaxymodel::DG_UNKNOWN) {
            densParams.gridSizeR = toInt(getItemFromPyDict(namedArgs, "gridSizeR"), densParams.gridSizeR);
            densParams.gridSizez = toInt(getItemFromPyDict(namedArgs, "gridSizez"), densParams.gridSizez);
            densParams.lmax = toInt(getItemFromPyDict(namedArgs, "lmax"), densParams.lmax);
            densParams.mmax = toInt(getItemFromPyDict(namedArgs, "mmax"), densParams.mmax);
            densParams.stripsPerPane =
                toInt(getItemFromPyDict(namedArgs, "stripsPerPane"), densParams.stripsPerPane);
            densParams.innerShellMass =
                toDouble(getItemFromPyDict(namedArgs, "innerShellMass"), densParams.innerShellMass);
            densParams.outerShellMass =
                toDouble(getItemFromPyDict(namedArgs, "outerShellMass"), densParams.outerShellMass);
            densParams.axisRatioY =
                toDouble(getItemFromPyDict(namedArgs, "axisRatioY"), densParams.axisRatioY);
            densParams.axisRatioZ =
                toDouble(getItemFromPyDict(namedArgs, "axisRatioZ"), densParams.axisRatioZ);
            if(!dens) {
                PyErr_SetString(PyExc_ValueError, "Must provide a density=... argument");
                return -1;
            }
            self->target.reset(new galaxymodel::TargetDensity(*dens, densParams));
            utils::msg(utils::VL_VERBOSE, "Agama", "Created a " + std::string(self->target->name()) +
                " at " + utils::toString(self->target.get()));
            return 0;
        }

        // check if a KinemJeans is being requested
        if(utils::stringsEqual(type_str, "KinemJeans")) {
            double beta   = toDouble(getItemFromPyDict(namedArgs, "beta"), 0.);
            int gridSizeR = toInt(getItemFromPyDict(namedArgs, "gridSizeR"), 0);
            int degree    = toInt(getItemFromPyDict(namedArgs, "degree"), 0);
            if(!dens || !pot) {
                PyErr_SetString(PyExc_ValueError, "Must provide density=... and potential=... arguments");
                return -1;
            }
            self->target.reset(new galaxymodel::TargetKinemJeans(*dens, *pot, beta, gridSizeR, degree));
            utils::msg(utils::VL_VERBOSE, "Agama", "Created a " + std::string(self->target->name()) +
                " at " + utils::toString(self->target.get()));
            return 0;
        }
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError,
            (std::string("Error in creating a Target object: ")+e.what()).c_str());
        return -1;
    }
    // shouldn't reach here
    PyErr_SetString(PyExc_ValueError, "Unknown type='...' argument");
    return -1;
}

PyObject* Target_name(PyObject* self)
{
    return Py_BuildValue("s", ((TargetObject*)self)->target->name());
}

PyObject* Target_elem(PyObject* self, Py_ssize_t index)
{
    std::vector<double> constraints = ((TargetObject*)self)->target->constraints();
    if(index<0 || index >= (Py_ssize_t)constraints.size()) {
        PyErr_SetString(PyExc_IndexError, "element index out of range in Target");
        return NULL;
    }
    return Py_BuildValue("sd", ((TargetObject*)self)->target->elemName(index).c_str(), constraints[index]);
}

Py_ssize_t Target_len(PyObject* self)
{
    return ((TargetObject*)self)->target->numConstraints();
}

PyObject* Target_values(PyObject* self)
{
    return toPyArray(((TargetObject*)self)->target->constraints());
}

static PySequenceMethods Target_sequence_methods = {
    Target_len, 0, 0, Target_elem,
};
static PyMethodDef Target_methods[] = {
    { "values", (PyCFunction)Target_values, METH_NOARGS,
      "Return the 1d array of constraint values" },
    { NULL }
};

static PyTypeObject TargetType = {
    PyObject_HEAD_INIT(NULL)
    0, "agama.Target",
    sizeof(TargetObject), 0, (destructor)Target_dealloc,
    0, 0, 0, 0, 0, 0, &Target_sequence_methods, 0, 0, 0, Target_name, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringTarget,
    0, 0, 0, 0, 0, 0, Target_methods, 0, 0, 0, 0, 0, 0, 0,
    (initproc)Target_init
};


///@}
//  ----------------------------------------------
/// \name  CubicSpline class and related routines
//  ----------------------------------------------
///@{

/// \cond INTERNAL_DOCS
/// Python type corresponding to CubicSpline class
typedef struct {
    PyObject_HEAD
    math::CubicSpline spl;
} CubicSplineObject;
/// \endcond

void CubicSpline_dealloc(PyObject* self)
{
    utils::msg(utils::VL_VERBOSE, "Agama", "Deleted a cubic spline of size "+
        utils::toString(((CubicSplineObject*)self)->spl.xvalues().size())+" at "+
        utils::toString(&((CubicSplineObject*)self)->spl));
    // dirty hack: manually call the destructor for an object that was
    // constructed not in a normal way, but rather with a placement new operator
    ((CubicSplineObject*)self)->spl.~CubicSpline();
    self->ob_type->tp_free(self);
}

int CubicSpline_init(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    // "dirty hack" (see above) to construct a C++ object in an already allocated chunk of memory
    new (&(((CubicSplineObject*)self)->spl)) math::CubicSpline;
    PyObject* x_obj=NULL;
    PyObject* y_obj=NULL;
    double derivLeft=NAN, derivRight=NAN;  // undefined by default
    int regularize=0;
    static const char* keywords[] = {"x","y","left","right","reg",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "OO|ddi", const_cast<char **>(keywords),
        &x_obj, &y_obj, &derivLeft, &derivRight, &regularize))
    {
        PyErr_SetString(PyExc_ValueError, "CubicSpline: "
            "must provide two arrays of equal length (input x and y points), "
            "optionally one or both endpoint derivatives (left, right), or a regularization flag");
        return -1;
    }
    std::vector<double>
        xvalues(toDoubleArray(x_obj)),
        yvalues(toDoubleArray(y_obj));
    if(xvalues.size() != yvalues.size() || xvalues.size() < 2) {
        PyErr_SetString(PyExc_ValueError, "CubicSpline: input does not contain valid arrays");
        return -1;
    }
    if(regularize && (derivLeft==derivLeft || derivRight==derivRight)) {
        PyErr_SetString(PyExc_ValueError, "CubicSpline: cannot provide endpoint derivatives "
            "and apply a regularization filter at the same time");
        return -1;
    }
    try {
        ((CubicSplineObject*)self)->spl = regularize ?
            math::CubicSpline(xvalues, yvalues, regularize) :
            math::CubicSpline(xvalues, yvalues, derivLeft, derivRight);
        utils::msg(utils::VL_VERBOSE, "Agama", "Created a cubic spline of size "+
            utils::toString(((CubicSplineObject*)self)->spl.xvalues().size())+" at "+
            utils::toString(&((CubicSplineObject*)self)->spl));
        return 0;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return -1;
    }
}

inline double splEval(const math::CubicSpline& spl, double x, int der)
{
    double result;
    switch(der) {
        case 0: return spl.value(x);
        case 1: spl.evalDeriv(x, NULL, &result); return result;
        case 2: spl.evalDeriv(x, NULL, NULL, &result); return result;
        default: return NAN;  // shouldn't occur
    }
}

PyObject* CubicSpline_value(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    if(self==NULL || ((CubicSplineObject*)self)->spl.empty()) {
        PyErr_SetString(PyExc_ValueError, "CubicSpline object is not properly initialized");
        return NULL;
    }
    const math::CubicSpline& spl = ((CubicSplineObject*)self)->spl;
    static const char* keywords[] = {"x","der","ext",NULL};
    PyObject* ptx=NULL;
    int der=0;
    PyObject* extrapolate_obj=NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "O|iO", const_cast<char **>(keywords),
        &ptx, &der, &extrapolate_obj))
    {
        //PyErr_SetString(PyExc_ValueError, "Invalid arguments");
        return NULL;
    }
    if(der<0 || der>2) {
        PyErr_SetString(PyExc_ValueError, "Can only compute derivatives up to 2nd");
        return NULL;
    }
    // check if we should extrapolate the spline (default behaviour),
    // or replace the output with the given value if it's out of range (if ext=... argument was given)
    double extrapolate_val = extrapolate_obj == NULL ? 0 : toDouble(extrapolate_obj);
    double xmin = spl.xmin(), xmax = spl.xmax();

    // if the input is a single value, just do it
    if(PyFloat_Check(ptx)) { // one value
        double x = PyFloat_AsDouble(ptx);
        if(PyErr_Occurred())
            return NULL;
        if(extrapolate_obj!=NULL && (x<xmin || x>xmax))
            return Py_BuildValue("O", extrapolate_obj);
        else
            return Py_BuildValue("d", splEval(spl, x, der) );
    }
    // otherwise the input should be an array, and the output will be an array of the same shape
    PyArrayObject *arr = (PyArrayObject*)
        PyArray_FROM_OTF(ptx, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY | NPY_ARRAY_ENSURECOPY);
    if(arr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Argument must be either float, list or numpy array");
        return NULL;
    }

    // replace elements of the copy of input array with computed values
    npy_intp size = PyArray_SIZE(arr);
    for(int i=0; i<size; i++) {
        // reference to the array element to be replaced
        double& x = static_cast<double*>(PyArray_DATA(arr))[i];
        if(extrapolate_obj!=NULL && (x<xmin || x>xmax))
            x = extrapolate_val;
        else
            x = splEval(spl, x, der);
    }
    return PyArray_Return(arr);
}

static PyMethodDef CubicSpline_methods[] = {
    { NULL, NULL, 0, NULL }  // no named methods
};

static const char* docstringCubicSpline =
    "Cubic spline with natural or clamped boundary conditions.\n"
    "Arguments:\n"
    "    x (array of floats) -- grid nodes in x (at least two), "
    "must be sorted in increasing order.\n"
    "    y (array of floats) -- values of spline at grid nodes, "
    "same length as x.\n"
    "    left (float, optional) -- derivative at the leftmost endpoint; "
    "if not provided or is NAN, a natural boundary condition is used "
    "(i.e., second derivative is zero).\n"
    "    right (float, optional) -- derivative at the rightmost endpoint, "
    "same default behaviour.\n"
    "    reg (boolean, default False) -- apply a regularization filter to "
    "reduce overshooting in the case of sharp discontinuities in input data "
    "and preserve monotonic trend of input points; "
    "cannot be used when an endpoint derivative is provided manually.\n\n"
    "Values of the spline and up to its second derivative are computed using the () "
    "operator with the first argument being a single x-point or an array of points, "
    "the optional second argument (der=...) is the derivative index (0, 1, or 2), "
    "and the optional third argument (ext=...) specifies the value returned for "
    "points outside the definition region; if the latter is not provided, "
    "the spline is linearly extrapolated outside its definition region.";

static PyTypeObject CubicSplineType = {
    PyObject_HEAD_INIT(NULL)
    0, "agama.CubicSpline",
    sizeof(CubicSplineObject), 0, CubicSpline_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, CubicSpline_value, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringCubicSpline,
    0, 0, 0, 0, 0, 0, CubicSpline_methods, 0, 0, 0, 0, 0, 0, 0,
    CubicSpline_init
};

/// Construct a Python cubic spline object from the provided x and y arrays
PyObject* createCubicSpline(const std::vector<double>& x, const std::vector<double>& y)
{
    // allocate a new Python CubicSpline object
    CubicSplineObject* spl_obj = PyObject_New(CubicSplineObject, &CubicSplineType);
    if(!spl_obj)
        return NULL;
    // same dirty hack to construct a C++ object in already allocated memory
    new (&(spl_obj->spl)) math::CubicSpline(x, y);
    utils::msg(utils::VL_VERBOSE, "Agama", "Constructed a cubic spline of size "+
        utils::toString(spl_obj->spl.xvalues().size())+" at "+
        utils::toString(&(spl_obj->spl)));
    return (PyObject*)spl_obj;
}


static const char* docstringSplineApprox =
    "splineApprox constructs a smoothing cubic spline from a set of points.\n"
    "It approximates a large set of (x,y) points by a smooth curve with "
    "a moderate number of knots.\n"
    "Arguments:\n"
    "    knots -- array of nodes of the grid that will be used to represent "
    "the smoothing spline; must be sorted in order of increase. "
    "The knots should preferrably encompass the range of x values of all points, "
    "and each interval between knots should contain at least one points; "
    "however, both these conditions are not obligatory.\n"
    "    x -- x-coordinates of points (1d array), "
    "should preferrably be in the range covered by knots, ordering does not matter.\n"
    "    y -- y-coordinates of points, same length as x.\n"
    "    w -- (1d array of the same length as x, optional) are weights of "
    "each input point used in least-square fitting, assumed uniform if omitted.\n"
    "    smooth -- (float, default 0) is the parameter controlling the tradeoff "
    "between smoothness and approximation error; zero means no additional smoothing "
    "(beyond the one resulting from discreteness of the spacing of knots), "
    "and values around unity usually yield a reasonable smoothing of noise without "
    "significantly increasing the rms error in the approximation.\n"
    "Returns: a CubicSpline object.\n";

PyObject* splineApprox(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"knots","x","y","w","smooth",NULL};
    PyObject* k_obj=NULL;
    PyObject* x_obj=NULL;
    PyObject* y_obj=NULL;
    PyObject* w_obj=NULL;
    double smoothfactor=0;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "OOO|Od", const_cast<char **>(keywords),
        &k_obj, &x_obj, &y_obj, &w_obj, &smoothfactor))
    {
        PyErr_SetString(PyExc_ValueError, "splineApprox: "
            "must provide an array of grid nodes and two arrays of equal length "
            "(input x and y points), and optionally an array of weights "
            "and a float (smooth factor)");
        return NULL;
    }
    std::vector<double>
        knots  (toDoubleArray(k_obj)),
        xvalues(toDoubleArray(x_obj)),
        yvalues(toDoubleArray(y_obj)),
        weights(toDoubleArray(w_obj));
    if(xvalues.empty() || yvalues.empty() || knots.empty()) {
        PyErr_SetString(PyExc_ValueError, "Input does not contain valid arrays");
        return NULL;
    }
    if(knots.size() < 2 || xvalues.size() != yvalues.size()) {
        PyErr_SetString(PyExc_ValueError,
            "Arguments must be an array of grid nodes (at least 2) "
            "and two arrays of equal length (x and y)");
        return NULL;
    }
    if(!weights.empty() && weights.size() != xvalues.size()) {
        PyErr_SetString(PyExc_ValueError,
            "Length of the array of weights must be the same as the number of x and y points");
        return NULL;
    }
    try{
        math::SplineApprox spl(knots, xvalues, weights);
        std::vector<double> amplitudes;
        if(smoothfactor>0)
            amplitudes = spl.fitOversmooth(yvalues, smoothfactor);
        else
            amplitudes = spl.fit(yvalues, -smoothfactor);
        return createCubicSpline(knots, amplitudes);
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
}


static const char* docstringSplineLogDensity =
    "splineLogDensity performs a non-parametric density estimate  "
    "from a set of sample points.\n"
    "Let rho(x) be an arbitrary density distribution over a finite or infinite "
    "interval, and let {x_i, w_i} be a set of sample points and weights, "
    "drawn from this distribution.\n"
    "This routine reconstructs log(rho(x)) approximated as a cubic spline defined "
    "by the given grid of nodes X_k, using a penalized density estimation approach.\n"
    "Arguments:\n"
    "    knots -- array of nodes of the grid that will be used to represent "
    "the smoothing spline; must be sorted in order of increase. "
    "Ideally, the knots should encompass all or most of the sample points "
    "and be spaced such that each segment contains at least a few samples.\n"
    "    x -- coordinates of sample points (1d array), ordering does not matter.\n"
    "    w (optional) -- weights of sample points (1d array with the same length as x); "
    "by default set all weights to 1/len(x).\n"
    "    infLeft (boolean, default False) specifies whether the density is assumed to "
    "extend to x=-infinity (True) or is taken to be zero for all x<knots[0] (False). "
    "In the former case, any points to the left of the first knot are ignored during "
    "the estimate, while in the latter case they are taken into account; "
    "note that log(rho(x)) is linearly extrapolated for x<knots[0], so it will "
    "obviously be declining towards -infinity for the integral over rho(x) to be finite.\n"
    "    infRight (boolean, default False) is the same option for extrapolating "
    "the estimated density to x>knots[-1].\n"
    "    smooth (float, default 0) -- optional extra smoothing.\n"
    "Returns: a CubicSpline object representing log(rho(x)).\n";

PyObject* splineLogDensity(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"knots","x","w","infLeft","infRight","smooth",NULL};
    PyObject* k_obj=NULL;
    PyObject* x_obj=NULL;
    PyObject* w_obj=NULL;
    int infLeft=0, infRight=0;  // should rather be bool, but python doesn't handle it in arg list
    double smooth=0;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "OO|Oiid", const_cast<char **>(keywords),
        &k_obj, &x_obj, &w_obj, &infLeft, &infRight, &smooth))
    {
        PyErr_SetString(PyExc_ValueError, "splineLogDensity: "
            "must provide an array of grid nodes and two arrays of equal length "
            "(x coordinates of input points and their weights)");
        return NULL;
    }
    std::vector<double>
        knots  (toDoubleArray(k_obj)),
        xvalues(toDoubleArray(x_obj)),
        weights;
    if(w_obj)
        weights = toDoubleArray(w_obj);
    else
        weights.assign(xvalues.size(), 1./xvalues.size());
    if(xvalues.empty() || weights.empty() || knots.empty()) {
        PyErr_SetString(PyExc_ValueError, "Input does not contain valid arrays");
        return NULL;
    }
    if(knots.size() < 2|| xvalues.size() != weights.size()) {
        PyErr_SetString(PyExc_ValueError,
            "Arguments must be an array of grid nodes (at least 2) "
            "and two arrays of equal length (x and w), "
            "plus optionally two boolean parameters (infLeft, infRight)");
        return NULL;
    }
    if(!(smooth>=0)) {
        PyErr_SetString(PyExc_ValueError, "smooth factor must be non-negative");
        return NULL;
    }
    try{
        std::vector<double> amplitudes = math::splineLogDensity<3>(knots, xvalues, weights,
            math::FitOptions(
            (infLeft ? math::FO_INFINITE_LEFT : 0) |
            (infRight? math::FO_INFINITE_RIGHT: 0) ),
            smooth );
        return createCubicSpline(knots, amplitudes);
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
}


///@}
//  ---------------------------------
/// \name  Orbit integration routine
//  ---------------------------------
///@{

/// description of orbit function
static const char* docstringOrbit =
    "Compute a single orbit or a bunch of orbits in the given potential\n"
    "Named arguments:\n"
    "  ic:  initial conditions - either an array of 6 numbers (3 positions and 3 velocities in "
    "Cartesian coordinates) for a single orbit, or a 2d array of Nx6 numbers for a bunch of orbits.\n"
    "  potential:  a Potential object or a compatible interface.\n"
    "  Omega (optional):  pattern speed of the rotating frame (default 0).\n"
    "  time:  integration time - for a single orbit, just one number; "
    "for a bunch of orbits, an array of length N.\n"
    "  targets (optional):  zero or more instances of Target class (a tuple/list if more than one); "
    "each target collects its own data for each orbit.\n"
    "  trajsize (optional):  if given, turns on the recording of trajectory for each orbit "
    "(should be either a single integer or an array of integers with length N). "
    "The trajectory of each orbit is stored at regular intervals of time (`dt=time/(trajsize-1)`, "
    "so that the number of points is `trajsize`; both time and trajsize may differ between orbits.\n"
    "  accuracy (optional, default 1e-8):  relative accuracy of ODE integrator.\n"
    "Returns:\n"
    "  depending on the arguments, one or a tuple of several data containers (one for each target, "
    "plus an extra one for trajectories). \n"
    "  Each target produces a 2d array of floats with shape NxC, where N is the number of orbits, "
    "and C is the number of constraints in the target (varies between targets); "
    "if there was a single orbit, then this would be a 1d array of length C. "
    "These data storage arrays should be provided to the `optsolve()` routine. \n"
    "  Trajectory output is represented as a Nx2 array (or, in case of a single orbit, a 1d array "
    "of length 2), with elements being NumPy arrays themselves: "
    "each row stands for one orbit, the first element in each row is a 1d array of length "
    "`trajsize` containing the timestamps, and the second is a 2d array of size `trajsize`x6 "
    "containing the position+velocity at corresponding timestamps.\n"
    "Examples:\n"
    "# compute a single orbit and output the trajectory in a 2d array of size 1001x6:\n"
    ">>> times,points = orbit(potential=mypot, ic=[x,y,z,vx,vy,vz], time=100, trajsize=1001)\n"
    "# integrate a bunch of orbits with initial conditions taken from a Nx6 array `initcond`, "
    "for a time equivalent to 50 periods for each orbit, collecting the data for two targets "
    "`target1` and `target2` and also storing their trajectories in a Nx2 array of "
    "time and position/velocity arrays:\n"
    ">>> stor1, stor2, trajectories = orbit(potential=mypot, ic=initcond, time=50*mypot.Tcirc(initcond), "
    "trajsize=500, targets=(target1, target2))";

/// run a single orbit or the entire orbit library for a Schwarzschild model
PyObject* orbit(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    if(!onlyNamedArgs(args, namedArgs))
        return NULL;

    // parse input arguments
    orbit::OrbitIntParams params;
    double Omega = 0.;
    PyObject *ic_obj = NULL, *time_obj = NULL, *pot_obj = NULL, *targets_obj = NULL, *trajsize_obj = NULL;
    static const char* keywords[] =
        {"ic", "time", "potential", "targets", "trajsize", "Omega", "accuracy", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "|OOOOOdd", const_cast<char**>(keywords),
        &ic_obj, &time_obj, &pot_obj, &targets_obj, &trajsize_obj, &Omega, &params.accuracy))
    {
        //PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to orbit()");
        return NULL;
    }

    // ensure that a potential object was provided
    potential::PtrPotential pot = getPotential(pot_obj);
    if(!pot) {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'potential' must be a valid instance of Potential class");
        return NULL;
    }

    // ensure that initial conditions were provided
    PyArrayObject *ic_arr = ic_obj==NULL ? NULL :
        (PyArrayObject*) PyArray_FROM_OTF(ic_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(ic_arr == NULL || !(
        (PyArray_NDIM(ic_arr) == 1 && PyArray_DIM(ic_arr, 0) == 6) ||
        (PyArray_NDIM(ic_arr) == 2 && PyArray_DIM(ic_arr, 1) == 6) ) )
    {
        Py_XDECREF(ic_arr);
        PyErr_SetString(PyExc_ValueError, "Argument 'ic' does not contain a valid array of length Nx6");
        return NULL;
    }
    bool singleOrbit = PyArray_NDIM(ic_arr) == 1; // in this case all output arrays have one dimension less

    // unit-convert initial conditions
    int numOrbits = singleOrbit ? 1 : (int)PyArray_DIM(ic_arr, 0);
    std::vector<coord::PosVelCar> initCond(numOrbits);
    for(int i=0; i<numOrbits; i++)
        initCond[i] = convertPosVel(PyArray_NDIM(ic_arr) == 1 ?
            &pyArrayElem<double>(ic_arr, 0) : &pyArrayElem<double>(ic_arr, i, 0));
    Py_DECREF(ic_arr);

    // check that integration time(s) were provided
    PyArrayObject *time_arr = time_obj==NULL ? NULL :
        (PyArrayObject*) PyArray_FROM_OTF(time_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(time_arr == NULL || !( PyArray_NDIM(time_arr) == 0 ||
        (PyArray_NDIM(time_arr) == 1 && (int)PyArray_DIM(time_arr, 0) == numOrbits) ) )
    {
        Py_XDECREF(time_arr);
        PyErr_SetString(PyExc_ValueError,
            "Argument 'time' must either be a scalar or have the same length "
            "as the number of points in the initial conditions");
        return NULL;
    }

    // unit-convert integration times
    std::vector<double> integrTimes(numOrbits);
    if(PyArray_NDIM(time_arr) == 0)
        integrTimes.assign(numOrbits, PyFloat_AsDouble(time_obj));
    else
        for(int i=0; i<numOrbits; i++)
            integrTimes[i] = pyArrayElem<double>(time_arr, i) * conv->timeUnit;
    Py_DECREF(time_arr);
    for(int orb=0; orb<numOrbits; orb++)
        if(integrTimes[orb] <= 0) {
            PyErr_SetString(PyExc_ValueError, "Argument 'time' must be positive");
            return NULL;
        }

    // check that valid targets were provided
    std::vector<PyObject*> targets_vec = toPyObjectArray(targets_obj);
    std::vector<galaxymodel::PtrTarget> targets;
    size_t numTargets = targets_vec.size();
    for(size_t t=0; t<numTargets; t++) {
        if(!PyObject_TypeCheck(targets_vec[t], &TargetType)) {
            PyErr_SetString(PyExc_ValueError, "Argument 'targets' must contain "
                "an instance of Target class or a tuple/list of such instances");
            return NULL;
        }
        targets.push_back(((TargetObject*)targets_vec[t])->target);
    }

    // check if trajectory needs to be recorded
    std::vector<int> trajSizes;
    bool haveTraj = trajsize_obj!=NULL;  // in this case the output tuple contains one extra item
    if(haveTraj) {
        PyArrayObject *trajsize_arr =
            (PyArrayObject*) PyArray_FROM_OTF(trajsize_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
        if(!trajsize_arr)
            return NULL;
        if(PyArray_NDIM(trajsize_arr) == 0) {
            long val = PyInt_AsLong(trajsize_obj);
            if(val > 0)
                trajSizes.assign(numOrbits, val);
        } else if(PyArray_NDIM(trajsize_arr) == 1 && (int)PyArray_DIM(trajsize_arr, 0) == numOrbits) {
            trajSizes.resize(numOrbits);
            for(int i=0; i<numOrbits; i++)
                trajSizes[i] = pyArrayElem<int>(trajsize_arr, i);
        }
        Py_DECREF(trajsize_arr);
        if((int)trajSizes.size() != numOrbits) {
            PyErr_SetString(PyExc_ValueError,
                "Argument 'trajsize', if provided, must either be an integer or an array of integers "
                "with the same length as the number of points in the initial conditions");
            return NULL;
        }
    }

    // the output is a tuple with the following items:
    // each target corresponds to a NumPy array where the collected information for all orbits is stored,
    // plus optionally a list containing the trajectories of all orbits if they are requested
    if(numTargets + haveTraj == 0) {
        PyErr_SetString(PyExc_ValueError, "No output is requested");
        return NULL;
    }
    PyObject* result = PyTuple_New(numTargets + haveTraj);
    if(!result)
        return NULL;

    // allocate the arrays for storing the information for each target,
    // and optionally for the output trajectory(ies) - the last item in the output tuple;
    // the latter one is a Nx2 array of Python objects
    bool fail = false;  // error flag (e.g., insufficient memory)
    for(size_t t=0; !fail && t < numTargets + haveTraj; t++) {
        npy_intp numCols = t==numTargets ? 2 : targets[t]->numConstraints();
        int datatype     = t==numTargets ? NPY_OBJECT : STORAGE_NUM_T;
        npy_intp size[2] = {numOrbits, numCols};
        // if there is only a single orbit, the output array is 1-dimensional,
        // otherwise 2-dimensional (numOrbits rows, numConstraints columns)
        PyObject* storage_arr = singleOrbit ?
            PyArray_SimpleNew(1, &size[1], datatype) :
            PyArray_SimpleNew(2, size, datatype);
        if(storage_arr)
            PyTuple_SetItem(result, t, storage_arr);
        else fail = true;
    }

    // set up signal handler to stop the integration on a keyboard interrupt
    keyboardInterruptTriggered = 0;
    defaultKeyboardInterruptHandler = signal(SIGINT, customKeyboardInterruptHandler);

    // finally, run the orbit integration
    if(!fail) {
        const orbit::OrbitIntegratorRot orbitIntegrator(*pot, Omega / conv->timeUnit);
        int numComplete = 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for(int orb = 0; orb < numOrbits; orb++) {
            if(fail || keyboardInterruptTriggered) continue;
            try{
                double integrTime = integrTimes.at(orb);
                // slightly reduce the output interval for trajectory to ensure that
                // the last point is stored (otherwise it may be left out due to roundoff)
                double trajStep = haveTraj && trajSizes[orb]>0 ?
                    integrTime / (trajSizes[orb]-1+1e-10) : INFINITY;
                std::vector<coord::PosVelCar> traj;

                // construct runtime functions for each target that store the collected data
                // in the respective row of each target's matrix,
                // plus optionally the trajectory recording function
                orbit::RuntimeFncArray fncs(numTargets + haveTraj);
                for(size_t t=0; t<numTargets; t++) {
                    PyObject* storage_arr = PyTuple_GET_ITEM(result, t);
                    galaxymodel::StorageNumT* output = singleOrbit ?
                        &pyArrayElem<galaxymodel::StorageNumT>(storage_arr, 0) :
                        &pyArrayElem<galaxymodel::StorageNumT>(storage_arr, orb, 0);
                    fncs[t] = targets[t]->getOrbitRuntimeFnc(output);
                }
                if(haveTraj)
                    fncs[numTargets].reset(new orbit::RuntimeTrajectory<coord::Car>(traj, trajStep));

                // integrate the orbit
                orbit::integrate(initCond.at(orb), integrTime, orbitIntegrator, fncs);

                // if the trajectory was recorded, store it in the last item of the output tuple
                if(haveTraj) {
                    const npy_intp size = traj.size();
                    npy_intp dims[] = {size, 6};
                    PyObject* time_arr = PyArray_SimpleNew(1, dims, STORAGE_NUM_T);
                    PyObject* traj_arr = PyArray_SimpleNew(2, dims, STORAGE_NUM_T);
                    if(!time_arr || !traj_arr) {
                        fail = true;
                        continue;
                    }

                    // convert the units and numerical type
                    for(npy_intp index=0; index<size; index++) {
                        double point[6];
                        unconvertPosVel(traj[index], point);
                        for(int c=0; c<6; c++)
                            pyArrayElem<galaxymodel::StorageNumT>(traj_arr, index, c) =
                            static_cast<galaxymodel::StorageNumT>(point[c]);
                        pyArrayElem<galaxymodel::StorageNumT>(time_arr, index) =
                        static_cast<galaxymodel::StorageNumT>(trajStep * index / conv->timeUnit);
                    }

                    // store these arrays in the last element of the output tuple
                    PyObject* elem = PyTuple_GET_ITEM(result, numTargets);
                    if(singleOrbit) {
                        pyArrayElem<PyObject*>(elem, 0) = time_arr;
                        pyArrayElem<PyObject*>(elem, 1) = traj_arr;
                    } else {
                        pyArrayElem<PyObject*>(elem, orb, 0) = time_arr;
                        pyArrayElem<PyObject*>(elem, orb, 1) = traj_arr;
                    }
                }

                // status update
                ++numComplete;
                if(numOrbits != 1) {
                    printf("%i orbits complete\r", numComplete);
                    fflush(stdout);
                }
            }
            catch(std::exception& e) {
                PyErr_SetString(PyExc_ValueError, e.what());
                fail = true;
            }
        }
    }
    if(numOrbits != 1)
        printf("%i orbits complete\n", numOrbits);
    signal(SIGINT, defaultKeyboardInterruptHandler);  // restore signal handler
    if(keyboardInterruptTriggered) {
        PyErr_SetObject(PyExc_KeyboardInterrupt, NULL);
        fail = true;
    }
    if(fail) {
        Py_XDECREF(result);
        return NULL;
    }
    // return a tuple of storage matrices (numpy-compatible) and/or a list of orbit trajectories,
    // but if this tuple only contains one element, return simply this element
    if(PyTuple_Size(result) == 1) {
        PyObject* item = PyTuple_GET_ITEM(result, 0);
        Py_INCREF(item);
        Py_DECREF(result);
        return item;
    } else
        return result;
}


static const char* docstringSampleOrbitLibrary =
    "Construct an N-body snapshot from the orbit library\n"
    "Arguments:\n"
    "  n:  the required number of particles in the output snapshot.\n"
    "  traj:  an array of trajectories returned by the `orbit()` routine.\n"
    "  amplitudes:  an array of orbit weights, returned by the `optsolve()` routine.\n"
    "Returns: a tuple of two elements: the flag indicating success or failure, and the result.\n"
    "  In case of success, the result is a tuple of two arrays: particle coordinates/velocities "
    "(2d Nx6 array) and particle weights (1d array of length N).\n"
    "  In case of failure, the result is a different tuple of two arrays: "
    "list of orbit indices which did not have enough trajectory samples (length is anywhere from 1 to N), "
    "and corresponding required numbers of samples for each orbit from this list.\n";

/// convert the orbit library to an N-body model
PyObject* sampleOrbitLibrary(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    long Nbody = 0;
    PyObject *traj_obj = NULL, *ampl_obj = NULL;
    static const char* keywords[] = {"n", "traj", "amplitudes", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "lOO", const_cast<char**>(keywords),
        &Nbody, &traj_obj, &ampl_obj))
    {
        //PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to sampleOrbitLibrary()");
        return NULL;
    }
    if(Nbody <= 0) {
        PyErr_SetString(PyExc_ValueError, "Argument 'n' must be a positive integer");
        return NULL;
    }

    // check that amplitudes are correct and their sum is positive
    std::vector<double> ampl(toDoubleArray(ampl_obj));
    npy_intp numOrbits = ampl.size();
    if(numOrbits <= 0) {
        PyErr_SetString(PyExc_ValueError, "Argument 'amplitudes' must be a non-empty array of floats");
        return NULL;
    }
    double totalMass = 0.;
    for(npy_intp orb=0; orb<numOrbits; orb++) {
        if(ampl[orb]>=0)
            totalMass += ampl[orb];
        else
            totalMass = -INFINITY;
    }
    if(!(totalMass > 0)) {
        PyErr_SetString(PyExc_ValueError, "The sum of amplitudes must be positive");
        return NULL;
    }

    // check that the trajectories are provided: it should be an array with PyObjects as elements
    // (they must be arrays themselves, which will be checked later); the shape of this array should be
    // either numOrbits or numOrbits x 2 (the latter variant is returned by the `orbit()` routine).
    PyArrayObject* traj_arr = NULL;
    PyArray_Descr* dtype = NULL;
    int ndim = 0;
    npy_intp inputdims[NPY_MAXDIMS];
    if(PyArray_GetArrayParamsFromObject(traj_obj, NULL, 0, &dtype, &ndim, inputdims, &traj_arr, NULL) < 0 ||
        traj_arr == NULL ||                        // the argument must be a genuine NumPy array,
        PyArray_TYPE(traj_arr) != NPY_OBJECT ||    // with PyObjects as elements,
        !( PyArray_NDIM(traj_arr) == 1 ||          // either 1d, or 2d and the second dimension is 2,
        (PyArray_NDIM(traj_arr) == 2 && PyArray_DIM(traj_arr, 1) == 2) ) ||
        PyArray_DIM(traj_arr, 0) != numOrbits )    // first dimension equal to the number of orbits
    {
        PyErr_SetString(PyExc_ValueError,
            "'traj' must be an array of numpy arrays with the same length as 'amplitudes'");
        return NULL;
    }
    // if this array has shape numOrbits x 2, as returned by the `orbit()` routine,
    // then the first column contains timestamps (not used here), and the second column - trajectories
    bool useSecondCol = PyArray_NDIM(traj_arr) == 2;

    // allocate the output arrays of particles and masses
    npy_intp dims[2] = {Nbody, 6};
    PyObject* posvel_arr = PyArray_SimpleNew(2, dims, STORAGE_NUM_T);
    PyObject* mass_arr   = PyArray_SimpleNew(1, dims, STORAGE_NUM_T);

    // this array will store indices of orbits that failed to produce required number of samples
    std::vector<std::pair<int, int> > badOrbits;
    
    // scan the array of orbits and sample appropriate number of points from each trajectory
    double cumulMass   = 0.;  // total mass of output points sampled so far
    long outPointIndex = 0;   // total number of output samples constructed so far
    for(int orb=0; orb<numOrbits; orb++) {
        cumulMass += ampl[orb];
        long newPointIndex = static_cast<long int>(cumulMass / totalMass * Nbody);
        // required number of points to sample from this orbit
        int pointsToSample = newPointIndex - outPointIndex;

        // obtain the trajectory of the current orbit and check its correctness
        PyArrayObject* traj_elem = useSecondCol ?
            pyArrayElem<PyArrayObject*>(traj_arr, orb, 1) :
            pyArrayElem<PyArrayObject*>(traj_arr, orb);
        if(!PyArray_Check(traj_elem) || PyArray_NDIM(traj_elem) != 2 || PyArray_DIM(traj_elem, 1) != 6) {
            PyErr_SetString(PyExc_ValueError, "'traj' must contain arrays with shape Lx6");
            Py_DECREF(posvel_arr);
            Py_DECREF(mass_arr);
            return NULL;
        }
        int pointsInTrajectory = PyArray_DIM(traj_elem, 0);

        // check if this orbit has enough points to be sampled
        if(pointsInTrajectory >= pointsToSample) {
            // construct a random permutation of indices
            std::vector<int> permutation(pointsInTrajectory);
            math::getRandomPermutation(pointsInTrajectory, &permutation[0]);

            // draw the given number of samples with random indices from the trajectory
            for(int i=0; i<pointsToSample; i++) {
                const galaxymodel::StorageNumT* from =
                    &pyArrayElem<galaxymodel::StorageNumT>(traj_elem, permutation[i], 0);
                std::copy(from, from+6,
                    &pyArrayElem<galaxymodel::StorageNumT>(posvel_arr, outPointIndex+i, 0));
                pyArrayElem<galaxymodel::StorageNumT>(mass_arr, outPointIndex+i) =
                    static_cast<galaxymodel::StorageNumT>(totalMass / Nbody);
            }
        } else {
            // insufficient samples: do not draw anything, add to the list of bad orbits
            badOrbits.push_back(std::make_pair(orb, pointsToSample));
        }
        outPointIndex = newPointIndex;
    }
    assert(outPointIndex == Nbody);
    if(badOrbits.empty()) {
        return Py_BuildValue("O(NN)", Py_True, posvel_arr, mass_arr);
    } else {
        npy_intp size = badOrbits.size();
        PyObject* indices_arr   = PyArray_SimpleNew(1, &size, NPY_INT);
        PyObject* trajsizes_arr = PyArray_SimpleNew(1, &size, NPY_INT);
        for(npy_intp i=0; i<size; i++) {
            pyArrayElem<int>(indices_arr,   i) = badOrbits[i].first;
            pyArrayElem<int>(trajsizes_arr, i) = badOrbits[i].second;
        }
        Py_DECREF(posvel_arr);
        Py_DECREF(mass_arr);
        return Py_BuildValue("O(NN)", Py_False, indices_arr, trajsizes_arr);
    }
}


///@}
//  ----------------------------
/// \name  Optimization routine
//  ----------------------------
///@{

/** helper routine for combining together the elements of a tuple of 1d arrays.
    \param[in]  obj  is a single Python array or a tuple of arrays
    \param[in]  nRow is the required size of each input array
    \param[out] out  will be filled by the data from input arrays stacked together;
    if the input is empty, out will remain empty
    \return  true if the number and sizes of input arrays match the requirements in nRow,
    or if the input is empty; false otherwise
*/
bool stackVectors(PyObject* obj, const std::vector<int> nRow, std::vector<double>& out)
{
    out.clear();
    if(obj == NULL)
        return true;
    std::vector<PyObject*> arr = toPyObjectArray(obj);
    if(arr.size() != nRow.size())
        return false;
    for(size_t i=0; i<arr.size(); i++) {
        std::vector<double> tmp = toDoubleArray(arr[i]);
        if((int)tmp.size() != nRow[i])
            return false;
        out.insert(out.end(), tmp.begin(), tmp.end());
    }
    return true;
}

/// interface class for accessing the values of a 2d Python array or a stack of such arrays
class StackedMatrix: public math::IMatrix<double> {
    const std::vector<PyObject*>& stack;
    const std::vector<int>& nRow;
    std::vector<int> dataTypes;
public:
    StackedMatrix(const std::vector<PyObject*>& _stack,
        int nRowTotal, int nCol, const std::vector<int>& _nRow) :
        math::IMatrix<double>(nRowTotal, nCol), stack(_stack), nRow(_nRow)
    {
        assert(stack.size() == nRow.size());
        for(size_t s=0; s<stack.size(); s++)
            dataTypes.push_back(PyArray_TYPE((PyArrayObject*)stack[s]));
    }
    
    virtual size_t size() const { return rows() * cols(); }
    
    virtual double at(size_t row, size_t col) const
    {
        if(row >= rows() || col >= cols())
            throw std::out_of_range("index out of range");
        unsigned int indMatrix = 0;
        while(indMatrix < stack.size() && (int)row >= nRow[indMatrix]) {
            row -= nRow[indMatrix];
            indMatrix++;
        }
        assert(indMatrix < stack.size());
        if(dataTypes[indMatrix] == NPY_FLOAT)
            return pyArrayElem<float>(stack[indMatrix], row, col);
        else if(dataTypes[indMatrix] == NPY_DOUBLE)
            return pyArrayElem<double>(stack[indMatrix], row, col);
        else
            throw std::runtime_error("unknown data type in matrix");
    }
    
    virtual double elem(size_t index, size_t &row, size_t &col) const
    {
        row = index / cols();
        col = index % cols();
        return at(row, col);
    }
};

static const char* docstringOptsolve =
    "Solve a linear or quadratic optimization problem.\n"
    "Find a vector x that solves a system of linear equations  A x = rhs,  "
    "subject to elementwise inequalities  xmin <= x <= xmax, "
    "while minimizing the cost function  F(x) = L^T x + x^T Q x + P(A x - rhs), where "
    "L and Q are penalties for the solution vector, and P(y) is the penalty for violating "
    "the RHS constraints, consisting of two parts: linear penalty rL^T |y| and quadratic penalty "
    "|y|^T diag(rQ) |y|  (both rL and rQ are nonnegative vectors of the same length as rhs).\n"
    "Arguments:\n"
    "  matrix:  2d matrix A of size RxC, or a tuple of several matrices that would be vertically "
    "stacked (they all must have the same number of columns C, and number of rows R1,R2,...). "
    "Providing a list of matrices does not incur copying, unlike the numpy.vstack() function.\n"
    "  rhs:     1d vector of length R, or a tuple of the same number of vectors as the number of "
    "matrices, with sizes R1,R2,...\n"
    "  xpenl:   1d vector of length C - linear penalties for the solution x "
    "(optional - zero if not provided).\n"
    "  xpenq:   1d vector of length C - diagonal of the matrix Q of quadratic "
    "penalties for the solution x (optional).\n"
    "  rpenl:   1d vector of length R, or a tuple of vectors R1,R2,... - "
    "linear penalties for violating the RHS constraints (optional).\n"
    "  rpenq:   same for the quadratic penalties (optional - if neither linear nor quadratic "
    "penalties for RHS violation were provided, it means that RHS must be satisfied exactly).\n"
    "  xmin:    1d vector of length C - minimum allowed values for the solution x (optional - "
    "if not provided, it implies a vector of zeros, i.e. the solution must be nonnegative).\n"
    "  xmax:    1d vector of length C - maximum allowed values for the solution x (optional - "
    "if not provided, it implies no upper limit).\n"
    "Returns:\n"
    "  the vector x solving the above system; if it cannot be solved exactly and no penalties "
    "for constraint violation were provided, then raise an exception.";
///
PyObject* optsolve(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] =
        {"matrix", "rhs", "xpenl", "xpenq", "rpenl", "rpenq", "xmin", "xmax", NULL};
    PyObject *matrix_obj = NULL, *rhs_obj = NULL, *xpenl_obj = NULL, *xpenq_obj = NULL,
        *rpenl_obj = NULL, *rpenq_obj = NULL, *xmin_obj = NULL, *xmax_obj = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "OO|OOOOOO", const_cast<char**>(keywords),
        &matrix_obj, &rhs_obj, &xpenl_obj, &xpenq_obj, &rpenl_obj, &rpenq_obj, &xmin_obj, &xmax_obj))
    {
        //PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to optsolve()");
        return NULL;
    }

    // check that the matrix or a tuple of matrices were provided
    std::vector<PyObject*> matrixStack  = toPyObjectArray(matrix_obj);
    int nCol = 0, nRowTotal = 0, nStack = matrixStack.size();
    std::vector<int> nRow(nStack);
    for(int s=0; s<nStack; s++) {
        PyArrayObject* mat = (PyArrayObject*)matrixStack[s];
        if(!PyArray_Check(mat) ||      // it must be an array in the first place,
            PyArray_NDIM(mat) != 2 ||  // a 2d array, to be specific,
            // for the first item we memorize the # of columns, and check that other items have the same
            !((s == 0 && (nCol = PyArray_DIM(mat, 1)) > 0) || (s > 0 && PyArray_DIM(mat, 1) == nCol)) ||
            !(PyArray_TYPE(mat) == NPY_FLOAT || PyArray_TYPE(mat) == NPY_DOUBLE) )  // check data type
        {
            PyErr_SetString(PyExc_ValueError, "Argument 'matrix' must be a 2d array "
                "or a tuple of such arrays with the same number of columns");
            return NULL;
        }
        nRow[s] = PyArray_DIM(mat, 0);
        nRowTotal += nRow[s];
    }

    // check and stack other input vectors
    std::vector<double> rhs, xpenl, xpenq, rpenl, rpenq, xmin, xmax, result;
    if(!stackVectors(rhs_obj, nRow, rhs) || rhs.empty()) {
        PyErr_SetString(PyExc_ValueError, "Argument 'rhs' must be a 1d array "
            "or a tuple of such arrays matching the number of rows in 'matrix'");
        return NULL;
    }
    if(!stackVectors(rpenl_obj, nRow, rpenl)) {
        PyErr_SetString(PyExc_ValueError, "Argument 'rpenl', if provided, must be a 1d array "
            "or a tuple of such arrays matching the number of rows in 'matrix'");
        return NULL;
    }
    if(!stackVectors(rpenq_obj, nRow, rpenq)) {
        PyErr_SetString(PyExc_ValueError, "Argument 'rpenq', if provided, must be a 1d array "
            "or a tuple of such arrays matching the number of rows in 'matrix'");
        return NULL;
    }
    xpenl = toDoubleArray(xpenl_obj);
    if(!xpenl.empty() && (int)xpenl.size() != nCol) {
        PyErr_SetString(PyExc_ValueError, "Argument 'xpenl', if provided, must be a 1d array "
            "with length matching the number of columns in 'matrix'");
        return NULL;
    }
    xpenq = toDoubleArray(xpenq_obj);
    if(!xpenq.empty() && (int)xpenq.size() != nCol) {
        PyErr_SetString(PyExc_ValueError, "Argument 'xpenq', if provided, must be a 1d array "
            "with length matching the number of columns in 'matrix'");
        return NULL;
    }
    xmin = toDoubleArray(xmin_obj);
    if(!xmin.empty() && (int)xmin.size() != nCol) {
        PyErr_SetString(PyExc_ValueError, "Argument 'xmin', if provided, must be a 1d array "
            "with length matching the number of columns in 'matrix'");
        return NULL;
    }
    xmax = toDoubleArray(xmax_obj);
    if(!xmax.empty() && (int)xmax.size() != nCol) {
        PyErr_SetString(PyExc_ValueError, "Argument 'xmax', if provided, must be a 1d array "
            "with length matching the number of columns in 'matrix'");
        return NULL;
    }

    // construct an interface layer for matrix stacking
    StackedMatrix matrix(matrixStack, nRowTotal, nCol, nRow);

    // call the appropriate solver
    try {
        if(rpenq.empty() && xpenq.empty()) {
            if(rpenl.empty()) {
                if(xpenq.empty())
                    result = math::linearOptimizationSolve(
                        matrix, rhs, xpenl, xmin, xmax);
                else
                    result = math::quadraticOptimizationSolve(
                        matrix, rhs, xpenl, math::BandMatrix<double>(xpenq), xmin, xmax);
            } else
                result = math::linearOptimizationSolveApprox(
                    matrix, rhs, xpenl, rpenl, xmin, xmax);
        } else {
            result = math::quadraticOptimizationSolveApprox(
                matrix, rhs, xpenl, math::BandMatrix<double>(xpenq), rpenl, rpenq, xmin, xmax);
        }
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    return toPyArray(result);
}


///@}
//  -----------------------------
/// \name  Various math routines
//  -----------------------------
///@{

/// description of grid creation function
static const char* docstringNonuniformGrid =
    "Create a grid with unequally spaced nodes:\n"
    "x[k] = (exp(Z k) - 1)/(exp(Z) - 1), i.e., coordinates of nodes increase "
    "nearly linearly at the beginning and then nearly exponentially towards the end; "
    "the value of Z is computed so the the 1st element is at xmin and last at xmax "
    "(0th element is always placed at 0).\n"
    "Arguments: \n"
    "  nnodes   the total number of grid points (>=2)\n"
    "  xmin     the location of the innermost nonzero node (>0);\n"
    "  xmax     the location of the last node (should be >=nnodes*xmin);\n"
    "Returns:   the array of grid nodes.";

/// creation of a non-uniform grid, a complement to linspace and logspace routines
PyObject* nonuniformGrid(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"nnodes", "xmin", "xmax", NULL};
    int nnodes=-1;
    double xmin=-1, xmax=-1;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "idd", const_cast<char**>(keywords),
        &nnodes, &xmin, &xmax))
        return NULL;
    if(nnodes<2 || xmin<=0 || xmax<=xmin) {
        PyErr_SetString(PyExc_ValueError, "Incorrect arguments for nonuniformGrid");
        return NULL;
    }
    return toPyArray(math::createNonuniformGrid(nnodes, xmin, xmax, true));
}

/// wrapper for user-provided Python functions into the C++ compatible form
class FncWrapper: public math::IFunctionNdim {
    OmpDisabler ompDisabler;  // prevent parallel execution by setting OpenMP # of threads to 1
    const unsigned int nvars;
    PyObject* fnc;
public:
    FncWrapper(unsigned int _nvars, PyObject* _fnc): nvars(_nvars), fnc(_fnc) {}
    virtual void eval(const double vars[], double values[]) const {
        npy_intp dims[]  = {1, nvars};
        PyObject* args   = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, const_cast<double*>(vars));
        PyObject* result = PyObject_CallFunctionObjArgs(fnc, args, NULL);
        Py_DECREF(args);
        if(result == NULL) {
            PyErr_Print();
            throw std::runtime_error("Exception occurred inside integrand");
        }
        if(PyArray_Check(result))
            values[0] = pyArrayElem<double>(result, 0);  // TODO: ensure that it's a float array!
        else if(PyNumber_Check(result))
            values[0] = PyFloat_AsDouble(result);
        else {
            Py_DECREF(result);
            throw std::runtime_error("Invalid data type returned from user-defined function");
        }
        Py_DECREF(result);
    }
    virtual unsigned int numVars()   const { return nvars; }
    virtual unsigned int numValues() const { return 1; }
};

/// parse the arguments of integrateNdim and sampleNdim functions
bool parseLowerUpperBounds(PyObject* lower_obj, PyObject* upper_obj,
    std::vector<double> &xlow, std::vector<double> &xupp)
{
    if(!lower_obj) {   // this should always be provided - either # of dimensions, or lower boundary
        PyErr_SetString(PyExc_ValueError,
            "Either integration region or number of dimensions must be provided");
        return false;
    }
    int ndim = -1;
    if(PyInt_Check(lower_obj)) {
        ndim = PyInt_AsLong(lower_obj);
        if(ndim<1) {
            PyErr_SetString(PyExc_ValueError, "Number of dimensions is invalid");
            return false;
        }
        if(upper_obj) {
            PyErr_Format(PyExc_ValueError,
                "May not provide 'upper' argument if 'lower' specifies the number of dimensions (%i)", ndim);
            return false;
        }
        xlow.assign(ndim, 0.);  // default integration region
        xupp.assign(ndim, 1.);
        return true;
    }
    // if the first parameter is not the number of dimensions, then it must be the lower boundary,
    // and the second one must be the upper boundary
    PyArrayObject *lower_arr = (PyArrayObject*) PyArray_FROM_OTF(lower_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(lower_arr == NULL || PyArray_NDIM(lower_arr) != 1) {
        Py_XDECREF(lower_arr);
        PyErr_SetString(PyExc_ValueError,
            "Argument 'lower' does not contain a valid array");
        return false;
    }
    ndim = PyArray_DIM(lower_arr, 0);
    if(!upper_obj) {
        PyErr_SetString(PyExc_ValueError, "Must provide both 'lower' and 'upper' arguments if both are arrays");
        return false;
    }
    PyArrayObject *upper_arr = (PyArrayObject*) PyArray_FROM_OTF(upper_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(upper_arr == NULL || PyArray_NDIM(upper_arr) != 1 || PyArray_DIM(upper_arr, 0) != ndim) {
        Py_XDECREF(upper_arr);
        PyErr_Format(PyExc_ValueError,
            "Argument 'upper' does not contain a valid array of length %i", ndim);
        return false;
    }
    xlow.resize(ndim);
    xupp.resize(ndim);
    for(int d=0; d<ndim; d++) {
        xlow[d] = pyArrayElem<double>(lower_arr, d);
        xupp[d] = pyArrayElem<double>(upper_arr, d);
    }
    Py_DECREF(lower_arr);
    Py_DECREF(upper_arr);
    return true;
}

/// description of integration function
static const char* docstringIntegrateNdim =
    "Integrate an N-dimensional function\n"
    "Arguments:\n"
    "  fnc - a callable object that must accept a single argument "
    "(a 2d array MxN array of coordinates, where N is the dimension of the integration space, "
    "and M>=1 is the number of points where the integrand should be evaluated simultaneously -- "
    "this improves performance), and return a 1d array of length M with function values;\n"
    "  lower, upper - two arrays of the same length N (equal to the number of dimensions) "
    "that specify the lower and upper boundaries of integration hypercube; "
    "alternatively, a single value - the number of dimensions - may be passed instead of 'lower', "
    "in which case the default interval [0:1] is used for each dimension;\n"
    "  toler - relative error tolerance (default is 1e-4);\n"
    "  maxeval - maximum number of function evaluations (will not exceed it even if "
    "the required tolerance cannot be reached, default is 1e5).\n"
    "Returns: a tuple consisting of integral value, error estimate, "
    "and the actual number of function evaluations performed.\n\n"
    "Examples:\n"
    ">>> integrateNdim(fnc, [0,-1,0], [3.14,1,100])   "
    ">>> # three-dimensional integral over the region [0:pi] x [-1:1] x [0:100]\n"
    ">>> integrateNdim(fnc, 2)   # two-dimensional integral over default region [0:1] x [0:1]\n"
    ">>> integrateNdim(fnc, 4, toler=1e-3, maxeval=1e6)   "
    ">>> # non-default values for tolerance and number of evaluations must be passed as named arguments\n";

/// N-dimensional integration
PyObject* integrateNdim(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"fnc", "lower", "upper", "toler", "maxeval", NULL};
    double eps=1e-4;
    int maxNumEval=100000, numEval=-1;
    PyObject *callback=NULL, *lower_obj=NULL, *upper_obj=NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "O|OOdi", const_cast<char**>(keywords),
        &callback, &lower_obj, &upper_obj, &eps, &maxNumEval) ||
        !PyCallable_Check(callback) || eps<=0 || maxNumEval<=0)
    {
        PyErr_SetString(PyExc_ValueError, "Incorrect arguments for integrateNdim");
        return NULL;
    }
    std::vector<double> xlow, xupp;
    if(!parseLowerUpperBounds(lower_obj, upper_obj, xlow, xupp))
        return NULL;
    double result, error;
    try{
        FncWrapper fnc(xlow.size(), callback);
        math::integrateNdim(fnc, &xlow.front(), &xupp.front(), eps, maxNumEval, &result, &error, &numEval);
    }
    catch(std::exception& e) {
        if(!PyErr_Occurred())    // set our own error string if it hadn't been set by Python
            PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    return Py_BuildValue("ddi", result, error, numEval);
}

/// description of sampling function
static const char* docstringSampleNdim =
    "Sample from a non-negative N-dimensional function.\n"
    "Draw a requested number of points from the hypercube in such a way that "
    "the density of points at any location is proportional to the value of function.\n"
    "Arguments:\n"
    "  fnc - a callable object that must accept a single argument "
    "(a 2d array MxN array of coordinates, where N is the dimension of the hypercube, "
    "and M>=1 is the number of points where the function should be evaluated simultaneously -- "
    "this improves performance), and return a 1d array of M non-negative values "
    "(one for each point), interpreted as the probability density;\n"
    "  nsamples - the required number of samples drawn from this function;\n"
    "  lower, upper - two arrays of the same length (equal to the number of dimensions) "
    "that specify the lower and upper boundaries of the region (hypercube) to be sampled; "
    "alternatively, a single value - the number of dimensions - may be passed instead of 'lower', "
    "in which case the default interval [0:1] is used for each dimension;\n"
    "Returns: a tuple consisting of the array of samples with shape (nsamples,ndim), "
    "the integral of the function over the given region estimated in a Monte Carlo way from the samples, "
    "error estimate of the integral, and the actual number of function evaluations performed "
    "(which is typically a factor of few larger than the number of output samples).\n\n"
    "Example:\n"
    ">>> samples,integr,error,_ = sampleNdim(fnc, 10000, [0,-1,0], [10,1,3.14])\n";

/// N-dimensional sampling
PyObject* sampleNdim(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"fnc", "nsamples", "lower", "upper", NULL};
    int numSamples=-1;
    PyObject *callback=NULL, *lower_obj=NULL, *upper_obj=NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "Oi|OO", const_cast<char**>(keywords),
        &callback, &numSamples, &lower_obj, &upper_obj) ||
        !PyCallable_Check(callback) || numSamples<=0)
    {
        PyErr_SetString(PyExc_ValueError, "Incorrect arguments for sampleNdim");
        return NULL;
    }
    std::vector<double> xlow, xupp;
    if(!parseLowerUpperBounds(lower_obj, upper_obj, xlow, xupp))
        return NULL;
    double result, error;
    math::Matrix<double> samples;
    try{
        FncWrapper fnc(xlow.size(), callback);
        size_t numEval=0;
        math::sampleNdim(fnc, &xlow[0], &xupp[0], numSamples, samples, &numEval, &result, &error);
        npy_intp dim[] = {numSamples, static_cast<npy_intp>(xlow.size())};
        PyObject* arr  = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, const_cast<double*>(samples.data()));
        return Py_BuildValue("Nddi", arr, result, error, numEval);
    }
    catch(std::exception& e) {
        if(!PyErr_Occurred())    // set our own error string if it hadn't been set by Python
            PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
}

///@}

/// list of standalone functions exported by the module
static PyMethodDef module_methods[] = {
    { "setUnits",               (PyCFunction)setUnits,
      METH_VARARGS | METH_KEYWORDS, docstringSetUnits },
    { "resetUnits",                          resetUnits,
      METH_NOARGS,                  docstringResetUnits },
    { "nonuniformGrid",         (PyCFunction)nonuniformGrid,
      METH_VARARGS | METH_KEYWORDS, docstringNonuniformGrid },
    { "splineApprox",           (PyCFunction)splineApprox,
      METH_VARARGS | METH_KEYWORDS, docstringSplineApprox },
    { "splineLogDensity",       (PyCFunction)splineLogDensity,
      METH_VARARGS | METH_KEYWORDS, docstringSplineLogDensity },
    { "orbit",                  (PyCFunction)orbit,
      METH_VARARGS | METH_KEYWORDS, docstringOrbit },
    { "sampleOrbitLibrary",     (PyCFunction)sampleOrbitLibrary,
      METH_VARARGS | METH_KEYWORDS, docstringSampleOrbitLibrary },
    { "optsolve",               (PyCFunction)optsolve,
      METH_VARARGS | METH_KEYWORDS, docstringOptsolve },
    { "actions",                (PyCFunction)actions,
      METH_VARARGS | METH_KEYWORDS, docstringActions },
    { "integrateNdim",          (PyCFunction)integrateNdim,
      METH_VARARGS | METH_KEYWORDS, docstringIntegrateNdim },
    { "sampleNdim",             (PyCFunction)sampleNdim,
      METH_VARARGS | METH_KEYWORDS, docstringSampleNdim },
    { NULL }
};


// an annoying feature in Python C API is the use of different types to refer to the same object,
// which triggers a warning about breaking strict aliasing rules, unless compiled
// with -fno-strict-aliasing. To avoid this, we use a dirty typecast.
void* forget_about_type(void* x) { return x; }
#define Py_INCREFx(x) Py_INCREF(forget_about_type(x))

} // end internal namespace
using namespace pywrapper;

// the module initialization function must be outside the internal namespace
PyMODINIT_FUNC
initagama(void)
{
    PyObject* mod = Py_InitModule("agama", module_methods);
    if(!mod) return;
    conv.reset(new units::ExternalUnits());

    DensityType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&DensityType) < 0) return;
    Py_INCREFx(&DensityType);
    PyModule_AddObject(mod, "Density", (PyObject*)&DensityType);

    PotentialType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&PotentialType) < 0) return;
    Py_INCREFx(&PotentialType);
    PyModule_AddObject(mod, "Potential", (PyObject*)&PotentialType);
    PotentialTypePtr = &PotentialType;

    ActionFinderType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&ActionFinderType) < 0) return;
    Py_INCREFx(&ActionFinderType);
    PyModule_AddObject(mod, "ActionFinder", (PyObject*)&ActionFinderType);

    DistributionFunctionType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&DistributionFunctionType) < 0) return;
    Py_INCREFx(&DistributionFunctionType);
    PyModule_AddObject(mod, "DistributionFunction", (PyObject*)&DistributionFunctionType);
    DistributionFunctionTypePtr = &DistributionFunctionType;

    GalaxyModelType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&GalaxyModelType) < 0) return;
    Py_INCREFx(&GalaxyModelType);
    PyModule_AddObject(mod, "GalaxyModel", (PyObject*)&GalaxyModelType);

    ComponentType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&ComponentType) < 0) return;
    Py_INCREFx(&ComponentType);
    PyModule_AddObject(mod, "Component", (PyObject*)&ComponentType);

    SelfConsistentModelType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&SelfConsistentModelType) < 0) return;
    Py_INCREFx(&SelfConsistentModelType);
    PyModule_AddObject(mod, "SelfConsistentModel", (PyObject*)&SelfConsistentModelType);
    
    TargetType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&TargetType) < 0) return;
    Py_INCREFx(&TargetType);
    PyModule_AddObject(mod, "Target", (PyObject*)&TargetType);
    
    CubicSplineType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&CubicSplineType) < 0) return;
    Py_INCREFx(&CubicSplineType);
    PyModule_AddObject(mod, "CubicSpline", (PyObject*)&CubicSplineType);

    import_array();  // needed for NumPy to work properly
}
// ifdef HAVE_PYTHON
#endif
