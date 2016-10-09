/** \file   py_wrapper.cpp
    \brief  Python wrapper for the Agama library
    \author Eugene Vasiliev
    \date   2014-2016

    This is a Python extension module that provides the interface to
    some of the classes and functions from the C++ library.
    It needs to be compiled into a dynamic library and placed in a folder
    that Python is aware of (e.g., through the PYTHONPATH= environment variable).

    Currently this module provides access to potential classes, orbit integration
    routine, action finders, distribution functions, N-dimensional integration
    and sampling routines, and smoothing splines.
    Unit conversion is also part of the calling convention: the quantities 
    received from Python are assumed to be in some physical units and converted
    into internal units inside this module, and the output from the library 
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
#ifdef _OPENMP
#include "omp.h"
#endif
#include "units.h"
#include "potential_factory.h"
#include "potential_composite.h"
#include "actions_spherical.h"
#include "actions_staeckel.h"
#include "df_factory.h"
#include "df_interpolated.h"
#include "galaxymodel.h"
#include "galaxymodel_selfconsistent.h"
#include "orbit.h"
#include "math_core.h"
#include "math_sample.h"
#include "math_spline.h"
#include "utils.h"
#include "utils_config.h"

namespace{  // private namespace

/// \name  ----- Helper class to manage the OpenMP behaviour -----
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
#endif
};

///@}
/// \name  ----- Helper routines for type conversions and argument checking -----
///@{

/// return a string representation of a Python object
static std::string toString(PyObject* obj)
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
static int toInt(PyObject* obj, int defaultValue=-1)
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
static double toDouble(PyObject* obj, double defaultValue=NAN)
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

/// convert a Python array of floats to std::vector, or return empty vector in case of error
static std::vector<double> toFloatArray(PyObject* obj)
{
    if(obj==NULL)
        return std::vector<double>();
    PyArrayObject *arr = (PyArrayObject*)PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(!arr || PyArray_NDIM(arr) != 1) {
        Py_XDECREF(arr);
        return std::vector<double>();
    }
    std::vector<double> vec(
        (double*)PyArray_GETPTR1(arr, 0),
        (double*)PyArray_GETPTR1(arr, PyArray_DIM(arr, 0)) );
    Py_DECREF(arr);
    return vec;
}

/// convert a Python dictionary to its C++ analog
static utils::KeyValueMap convertPyDictToKeyValueMap(PyObject* dict)
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
static bool onlyNamedArgs(PyObject* args, PyObject* namedArgs)
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
static PyObject* getItemFromPyDict(PyObject* dict, const char* itemkey)
{
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(dict, &pos, &key, &value))
        if(utils::stringsEqual(toString(key), itemkey))
            return value;
    return NULL;
}

///@}
/// \name  ------- Unit handling routines --------
///@{

/// internal working units (arbitrary!)
static const units::InternalUnits unit(2.7183 * units::Kpc, 3.1416 * units::Myr);

/// external units that are used in the calling code, set by the user,
/// (or remaining at default values (no conversion) if not set explicitly
static shared_ptr<const units::ExternalUnits> conv;

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
static PyObject* setUnits(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"mass", "length", "velocity", "time", NULL};
    double mass = 0, length = 0, velocity = 0, time = 0;
    if(!onlyNamedArgs(args, namedArgs))
        return NULL;
    if(!PyArg_ParseTupleAndKeywords(
        args, namedArgs, "|dddd", const_cast<char**>(keywords),
        &mass, &length, &velocity, &time) ||
        mass<0 || length<0 || velocity<0 || time<0)
    {
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
    const units::ExternalUnits* newConv = NULL;
    if(length>0 && time>0)
        newConv = new units::ExternalUnits(unit,
            length*units::Kpc, length/time * units::Kpc/units::Myr, mass*units::Msun);
    else if(length>0 && velocity>0)
        newConv = new units::ExternalUnits(unit,
            length*units::Kpc, velocity*units::kms, mass*units::Msun);
    else if(time>0 && velocity>0)
        newConv = new units::ExternalUnits(unit,
            velocity*time * units::kms*units::Myr, velocity*units::kms, mass*units::Msun);
    else {
        PyErr_SetString(PyExc_ValueError,
            "You must specify exactly two out of three units: length, time and velocity");
        return NULL;
    }
    conv.reset(newConv);
    Py_INCREF(Py_None);
    return Py_None;
}

/// description of resetUnits function
static const char* docstringResetUnits = 
    "Reset the unit conversion system to a trivial one "
    "(i.e., no conversion involved and all quantities are assumed to be in N-body units, "
    "with the gravitational constant equal to 1.\n"
    "Note that this is NOT equivalent to setUnits(mass=1, length=1, velocity=1).\n";

/// reset the unit conversion
static PyObject* resetUnits(PyObject* /*self*/, PyObject* /*args*/)
{
    conv.reset(new units::ExternalUnits());
    Py_INCREF(Py_None);
    return Py_None;
}

/// helper function for converting position to internal units
static inline coord::PosCar convertPos(const double input[]) {
    return coord::PosCar(
        input[0] * conv->lengthUnit, 
        input[1] * conv->lengthUnit, 
        input[2] * conv->lengthUnit);
}

/// helper function for converting position/velocity to internal units
static inline coord::PosVelCar convertPosVel(const double input[]) {
    return coord::PosVelCar(
        input[0] * conv->lengthUnit,
        input[1] * conv->lengthUnit,
        input[2] * conv->lengthUnit,
        input[3] * conv->velocityUnit,
        input[4] * conv->velocityUnit,
        input[5] * conv->velocityUnit);
}

/// helper function for converting actions to internal units
static inline actions::Actions convertActions(const double input[]) {
    return actions::Actions(
        input[0] * conv->lengthUnit * conv->velocityUnit, 
        input[1] * conv->lengthUnit * conv->velocityUnit, 
        input[2] * conv->lengthUnit * conv->velocityUnit);
}

/// helper function to convert position from internal units back to user units
static inline void unconvertPos(const coord::PosCar& point, double dest[])
{
    dest[0] = point.x / conv->lengthUnit;
    dest[1] = point.y / conv->lengthUnit;
    dest[2] = point.z / conv->lengthUnit;
}

/// helper function to convert position/velocity from internal units back to user units
static inline void unconvertPosVel(const coord::PosVelCar& point, double dest[])
{
    dest[0] = point.x / conv->lengthUnit;
    dest[1] = point.y / conv->lengthUnit;
    dest[2] = point.z / conv->lengthUnit;
    dest[3] = point.vx / conv->velocityUnit;
    dest[4] = point.vy / conv->velocityUnit;
    dest[5] = point.vz / conv->velocityUnit;
}

/// helper function to convert actions from internal units back to user units
static inline void unconvertActions(const actions::Actions& act, double dest[])
{
    dest[0] = act.Jr   / (conv->lengthUnit * conv->velocityUnit);
    dest[1] = act.Jz   / (conv->lengthUnit * conv->velocityUnit);
    dest[2] = act.Jphi / (conv->lengthUnit * conv->velocityUnit);
}

///@}
/// \name ----- a truly general interface for evaluating some function
///             for some input data and storing its output somewhere -----
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
    OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET = 136 ///< all wonders at once
};

/// size of input array for a single point
template<int numArgs>
static size_t inputLength();

/// parse a list of numArgs floating-point arguments for a Python function, 
/// and store them in inputArray[]; return 1 on success, 0 on failure 
template<int numArgs>
int parseTuple(PyObject* args, double inputArray[]);

/// check that the input array is of right dimensions, and return its length
template<int numArgs>
int parseArray(PyArrayObject* arr)
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
static size_t outputLength();

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

template<> inline size_t inputLength<INPUT_VALUE_SINGLE>()  {return 1;}
template<> inline size_t inputLength<INPUT_VALUE_TRIPLET>() {return 3;}
template<> inline size_t inputLength<INPUT_VALUE_SEXTET>()  {return 6;}

template<> int parseTuple<INPUT_VALUE_SINGLE>(PyObject* args, double input[]) {
    input[0] = PyFloat_AsDouble(args);
    return PyErr_Occurred() ? 0 : 1;
}
template<> int parseTuple<INPUT_VALUE_TRIPLET>(PyObject* args, double input[]) {
    return PyArg_ParseTuple(args, "ddd", &input[0], &input[1], &input[2]);
}
template<> int parseTuple<INPUT_VALUE_SEXTET>(PyObject* args, double input[]) {
    return PyArg_ParseTuple(args, "dddddd",
        &input[0], &input[1], &input[2], &input[3], &input[4], &input[5]);
}

template<>
int parseArray<INPUT_VALUE_SINGLE>(PyArrayObject* arr)
{
    if(PyArray_NDIM(arr) == 1)
        return PyArray_DIM(arr, 0);
    else
        return 0;
}

template<> const char* errStrInvalidArrayDim<INPUT_VALUE_SINGLE>() {
    return "Input does not contain a valid one-dimensional array";
}
template<> const char* errStrInvalidArrayDim<INPUT_VALUE_TRIPLET>() {
    return "Input does not contain a valid Nx3 array";
}
template<> const char* errStrInvalidArrayDim<INPUT_VALUE_SEXTET>() {
    return "Input does not contain a valid Nx6 array";
}

template<> const char* errStrInvalidInput<INPUT_VALUE_SINGLE>() {
    return "Input does not contain valid data (either a single number or a one-dimensional array)";
}
template<> const char* errStrInvalidInput<INPUT_VALUE_TRIPLET>() {
    return "Input does not contain valid data (either 3 numbers for a single point or a Nx3 array)";
}
template<> const char* errStrInvalidInput<INPUT_VALUE_SEXTET>() {
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
template<> inline size_t outputLength<OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET>() {return 10;}

template<> PyObject* formatTuple<OUTPUT_VALUE_SINGLE>(const double result[]) {
    return Py_BuildValue("d", result[0]);
}
template<> PyObject* formatTuple<OUTPUT_VALUE_TRIPLET>(const double result[]) {
    return Py_BuildValue("ddd", result[0], result[1], result[2]);
}
template<> PyObject* formatTuple<OUTPUT_VALUE_SEXTET>(const double result[]) {
    return Py_BuildValue("dddddd",
        result[0], result[1], result[2], result[3], result[4], result[5]);
}
template<> PyObject* formatTuple<OUTPUT_VALUE_SINGLE_AND_SINGLE>(const double result[]) {
    return Py_BuildValue("(dd)", result[0], result[1]);
}
template<> PyObject* formatTuple<OUTPUT_VALUE_SINGLE_AND_TRIPLET>(const double result[]) {
    return Py_BuildValue("d(ddd)", result[0], result[1], result[2], result[3]);
}
template<> PyObject* formatTuple<OUTPUT_VALUE_SINGLE_AND_SEXTET>(const double result[]) {
    return Py_BuildValue("d(dddddd)", result[0],
        result[1], result[2], result[3], result[4], result[5], result[6]);
}
template<> PyObject* formatTuple<OUTPUT_VALUE_TRIPLET_AND_TRIPLET>(const double result[]) {
    return Py_BuildValue("(ddd)(ddd)", result[0], result[1], result[2],
        result[3], result[4], result[5]);
}
template<> PyObject* formatTuple<OUTPUT_VALUE_TRIPLET_AND_SEXTET>(const double result[]) {
    return Py_BuildValue("(ddd)(dddddd)", result[0], result[1], result[2],
        result[3], result[4], result[5], result[6], result[7], result[8]);
}
template<> PyObject* formatTuple<OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET>(const double result[]) {
    return Py_BuildValue("d(ddd)(dddddd)", result[0], result[1], result[2], result[3],
        result[4], result[5], result[6], result[7], result[8], result[9]);
}

template<> PyObject* allocOutputArr<OUTPUT_VALUE_SINGLE>(int size) {
    npy_intp dims[] = {size};
    return PyArray_SimpleNew(1, dims, NPY_DOUBLE);
}
template<> PyObject* allocOutputArr<OUTPUT_VALUE_TRIPLET>(int size) {
    npy_intp dims[] = {size, 3};
    return PyArray_SimpleNew(2, dims, NPY_DOUBLE);
}
template<> PyObject* allocOutputArr<OUTPUT_VALUE_SEXTET>(int size) {
    npy_intp dims[] = {size, 6};
    return PyArray_SimpleNew(2, dims, NPY_DOUBLE);
}
template<> PyObject* allocOutputArr<OUTPUT_VALUE_SINGLE_AND_SINGLE>(int size) {
    npy_intp dims[] = {size};
    PyObject* arr1 = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    return Py_BuildValue("NN", arr1, arr2);
}
template<> PyObject* allocOutputArr<OUTPUT_VALUE_SINGLE_AND_TRIPLET>(int size) {
    npy_intp dims1[] = {size};
    npy_intp dims2[] = {size, 3};
    PyObject* arr1 = PyArray_SimpleNew(1, dims1, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(2, dims2, NPY_DOUBLE);
    return Py_BuildValue("NN", arr1, arr2);
}
template<> PyObject* allocOutputArr<OUTPUT_VALUE_SINGLE_AND_SEXTET>(int size) {
    npy_intp dims1[] = {size};
    npy_intp dims2[] = {size, 6};
    PyObject* arr1 = PyArray_SimpleNew(1, dims1, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(2, dims2, NPY_DOUBLE);
    return Py_BuildValue("NN", arr1, arr2);
}
template<> PyObject* allocOutputArr<OUTPUT_VALUE_TRIPLET_AND_TRIPLET>(int size) {
    npy_intp dims[] = {size, 3};
    PyObject* arr1 = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    return Py_BuildValue("NN", arr1, arr2);
}
template<> PyObject* allocOutputArr<OUTPUT_VALUE_TRIPLET_AND_SEXTET>(int size) {
    npy_intp dims1[] = {size, 3};
    npy_intp dims2[] = {size, 6};
    PyObject* arr1 = PyArray_SimpleNew(2, dims1, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(2, dims2, NPY_DOUBLE);
    return Py_BuildValue("NN", arr1, arr2);
}
template<> PyObject* allocOutputArr<OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET>(int size) {
    npy_intp dims1[] = {size};
    npy_intp dims2[] = {size, 3};
    npy_intp dims3[] = {size, 6};
    PyObject* arr1 = PyArray_SimpleNew(1, dims1, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(2, dims2, NPY_DOUBLE);
    PyObject* arr3 = PyArray_SimpleNew(2, dims3, NPY_DOUBLE);
    return Py_BuildValue("NNN", arr1, arr2, arr3);
}

template<> void formatOutputArr<OUTPUT_VALUE_SINGLE>(
    const double result[], const int index, PyObject* resultObj) 
{
    ((double*)PyArray_DATA((PyArrayObject*)resultObj))[index] = result[0];
}
template<> void formatOutputArr<OUTPUT_VALUE_TRIPLET>(
    const double result[], const int index, PyObject* resultObj) 
{
    for(int d=0; d<3; d++)
        ((double*)PyArray_DATA((PyArrayObject*)resultObj))[index*3+d] = result[d];
}
template<> void formatOutputArr<OUTPUT_VALUE_SEXTET>(
    const double result[], const int index, PyObject* resultObj) 
{
    for(int d=0; d<6; d++)
        ((double*)PyArray_DATA((PyArrayObject*)resultObj))[index*6+d] = result[d];
}
template<> void formatOutputArr<OUTPUT_VALUE_SINGLE_AND_SINGLE>(
    const double result[], const int index, PyObject* resultObj) 
{
    PyArrayObject* arr1 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 0);
    PyArrayObject* arr2 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 1);
    ((double*)PyArray_DATA(arr1))[index] = result[0];
    ((double*)PyArray_DATA(arr2))[index] = result[1];
}
template<> void formatOutputArr<OUTPUT_VALUE_SINGLE_AND_TRIPLET>(
    const double result[], const int index, PyObject* resultObj) 
{
    PyArrayObject* arr1 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 0);
    PyArrayObject* arr2 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 1);
    ((double*)PyArray_DATA(arr1))[index] = result[0];
    for(int d=0; d<3; d++)
        ((double*)PyArray_DATA(arr2))[index*3+d] = result[d+1];
}
template<> void formatOutputArr<OUTPUT_VALUE_SINGLE_AND_SEXTET>(
    const double result[], const int index, PyObject* resultObj) 
{
    PyArrayObject* arr1 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 0);
    PyArrayObject* arr2 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 1);
    ((double*)PyArray_DATA(arr1))[index] = result[0];
    for(int d=0; d<6; d++)
        ((double*)PyArray_DATA(arr2))[index*6+d] = result[d+1];
}
template<> void formatOutputArr<OUTPUT_VALUE_TRIPLET_AND_TRIPLET>(
    const double result[], const int index, PyObject* resultObj) 
{
    PyArrayObject* arr1 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 0);
    PyArrayObject* arr2 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 1);
    for(int d=0; d<3; d++) {
        ((double*)PyArray_DATA(arr1))[index*3+d] = result[d];
        ((double*)PyArray_DATA(arr2))[index*3+d] = result[d+3];
    }
}
template<> void formatOutputArr<OUTPUT_VALUE_TRIPLET_AND_SEXTET>(
    const double result[], const int index, PyObject* resultObj) 
{
    PyArrayObject* arr1 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 0);
    PyArrayObject* arr2 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 1);
    for(int d=0; d<3; d++)
        ((double*)PyArray_DATA(arr1))[index*3+d] = result[d];
    for(int d=0; d<6; d++)
        ((double*)PyArray_DATA(arr2))[index*6+d] = result[d+3];
}
template<> void formatOutputArr<OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET>(
    const double result[], const int index, PyObject* resultObj) 
{
    PyArrayObject* arr1 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 0);
    PyArrayObject* arr2 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 1);
    PyArrayObject* arr3 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 2);
    ((double*)PyArray_DATA(arr1))[index] = result[0];
    for(int d=0; d<3; d++)
        ((double*)PyArray_DATA(arr2))[index*3+d] = result[d+1];
    for(int d=0; d<6; d++)
        ((double*)PyArray_DATA(arr3))[index*6+d] = result[d+4];
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
static PyObject* callAnyFunctionOnArray(void* params, PyObject* args, anyFunction fnc)
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
        if(PyArray_Check(args))
            obj = args;
        else if(PyTuple_Check(args) && PyTuple_Size(args)==1)
            obj = PyTuple_GET_ITEM(args, 0);
        if(obj) {
            PyArrayObject *arr = (PyArrayObject*) PyArray_FROM_OTF(obj,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            if(arr == NULL) {
                PyErr_SetString(PyExc_ValueError, "Input does not contain a valid array");
                return NULL;
            }
            int numpt = 0;
            if(PyArray_NDIM(arr) == 1 && PyArray_DIM(arr, 0) == numArgs) 
            {   // 1d array of length numArgs - a single point
                fnc(params, static_cast<double*>(PyArray_GETPTR1(arr, 0)), output);
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
            // allocate an appropriate output object
            PyObject* outputObj = allocOutputArr<numOutput>(numpt);
            // loop over input array
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i=0; i<numpt; i++) {
                double local_output[outputLength<numOutput>()];  // separate variable in each thread
                fnc(params, static_cast<double*>(PyArray_GETPTR2(arr, i, 0)), local_output);
                formatOutputArr<numOutput>(local_output, i, outputObj);
            }
            Py_DECREF(arr);
            return outputObj;
        }
        PyErr_SetString(PyExc_ValueError, errStrInvalidInput<numArgs>());
        return NULL;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, (std::string("Exception occurred: ")+e.what()).c_str());
        return NULL;
    }
}

///@}
/// \name  ---------- Density class ------------
///@{

/// \cond INTERNAL_DOCS
/// Python type corresponding to Density class
typedef struct {
    PyObject_HEAD
    potential::PtrDensity dens;
} DensityObject;
/// \endcond

static void Density_dealloc(DensityObject* self)
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
    "Dehnen, MiyamotoNagai and DiskAnsatz).\n" \
    "  axisRatio=...   axis ratio z/R for SpheroidDensity density profiles.\n" \
    "  p=...   axis ratio y/x, i.e., intermediate to long axis (applicable to triaxial " \
    "potential models such as Dehnen and Ferrers).\n" \
    "  q=...   axis ratio z/x, i.e., short to long axis (if applicable, same as axisRatio).\n" \
    "  gamma=...   central cusp slope (applicable for Dehnen and SpheroidDensity).\n" \
    "  beta=...   outer density slope (SpheroidDensity).\n" \
    "  innerCutoffRadius=...   radius of inner hole (DiskAnsatz).\n" \
    "  outerCutoffRadius=...   radius of outer exponential cutoff (SpheroidDensity).\n" \
    "  surfaceDensity=...   central surface density (or its value if no inner cutoff exists), " \
    "for DiskAnsatz.\n" \
    "  densityNorm=...   normalization of density profile for SpheroidDensity (the value " \
    "at scaleRadius).\n"

/// description of Density class
static const char* docstringDensity =
    "Density is a class representing a variety of density profiles "
    "that do not necessarily have a corresponding potential defined.\n"
    "An instance of Density class is constructed using the following keyword arguments:\n"
    "  type='...' or density='...'   the name of density profile (required), can be one of the following:\n"
    "    Denhen, Plummer, OblatePerfectEllipsoid, Ferrers, MiyamotoNagai, "
    "NFW, DiskDensity, SpheroidDensity.\n"
    DOCSTRING_DENSITY_PARAMS
    "Most of these parameters have reasonable default values.\n\n"
    "An instance of Potential class may be used in all contexts when a Density object is required;\n"
    "moreover, an arbitrary Python object with a method 'density(x,y,z)' that returns a single value "
    "may also be used in these contexts (i.e., an object presenting a Density interface).";

/// extract a pointer to C++ Density class from a Python object,
/// or return an empty pointer on error.
/// Declared here, implemented after the PotentialObject definition becomes available.
static potential::PtrDensity getDensity(PyObject* dens_obj, coord::SymmetryType sym=coord::ST_TRIAXIAL);

/// attempt to construct a composite density from a tuple of Density objects
static potential::PtrDensity Density_initFromTuple(PyObject* tuple)
{
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

/// attempt to construct a density from key=value parameters
static potential::PtrDensity Density_initFromDict(PyObject* namedArgs)
{
    utils::KeyValueMap params = convertPyDictToKeyValueMap(namedArgs);
    // for convenience, may specify the type of density model in type="..." argument
    if(params.contains("type") && !params.contains("density"))
        params.set("density", params.getString("type"));
    if(!params.contains("density"))
        throw std::invalid_argument("Should provide the name of density model "
            "in type='...' or density='...' argument");
    return potential::createDensity(params, *conv);
}

/// constructor of Density class
static int Density_init(DensityObject* self, PyObject* args, PyObject* namedArgs)
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

static void fncDensity_density(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    result[0] = ((DensityObject*)obj)->dens->density(point)
        / (conv->massUnit / pow_3(conv->lengthUnit));  // unit of density is M/L^3
}
static PyObject* Density_density(PyObject* self, PyObject* args) {
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE>
        (self, args, fncDensity_density);
}

static PyObject* Density_name(PyObject* self)
{
    return Py_BuildValue("s", ((DensityObject*)self)->dens->name());
}

static PyObject* Density_totalMass(PyObject* self)
{
    return Py_BuildValue("d", ((DensityObject*)self)->dens->totalMass() / conv->massUnit);
}

static PyObject* Density_export(PyObject* self, PyObject* args)
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
static PyObject* sampleDensity(const potential::BaseDensity& dens, PyObject* args)
{
    int numPoints=0;
    if(!PyArg_ParseTuple(args, "i", &numPoints) || numPoints<=0)
    {
        PyErr_SetString(PyExc_ValueError,
            "sample() takes one integer argument - the number of particles");
        return NULL;
    }
    try{
        // do the sampling
        particles::ParticleArray<coord::PosCyl> points =
            galaxymodel::generateDensitySamples(dens, numPoints);

        // convert output to NumPy array
        numPoints = points.size();
        npy_intp dims[] = {numPoints, 3};
        PyArrayObject* pos_arr  = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        PyArrayObject* mass_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        for(int i=0; i<numPoints; i++) {
            unconvertPos(coord::toPosCar(points.point(i)), ((double*)PyArray_DATA(pos_arr))+i*3);
            ((double*)PyArray_DATA(mass_arr))[i] = points.mass(i) / conv->massUnit;
        }
        return Py_BuildValue("NN", pos_arr, mass_arr);
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, 
            (std::string("Error in sample(): ")+e.what()).c_str());
        return NULL;
    }
}

static PyObject* Density_sample(PyObject* self, PyObject* args)
{
    return sampleDensity(*((DensityObject*)self)->dens, args);
}

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
    { "sample", Density_sample, METH_VARARGS, 
      "Sample the density profile with N point masses\n"
      "Arguments: the number of particles\n"
      "Returns: a tuple of two arrays: "
      "2d Nx3 array of point cartesian coordinates and 1d array of N point masses" },
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
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Density_name, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringDensity,
    0, 0, 0, 0, 0, 0, Density_methods, 0, 0, 0, 0, 0, 0, 0,
    (initproc)Density_init
};

/// create a Python Density object and initialize it with an existing instance of C++ density class
static PyObject* createDensityObject(const potential::PtrDensity& dens)
{
    DensityObject* dens_obj = PyObject_New(DensityObject, &DensityType);
    if(!dens_obj)
        return NULL;

    // this is a DIRTY HACK!!! we have allocated a new instance of Python class,
    // but have not initialized its extra fields in any way, so they contain garbage.
    // We can't simply assign a new value to its 'pot' member variable,
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
            value = static_cast<double*>(PyArray_GETPTR1((PyArrayObject*)result, 0))[0];
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
/// \name  ---------- Potential class and related data ------------
///@{

/// \cond INTERNAL_DOCS
/// Python type corresponding to Potential class
typedef struct {
    PyObject_HEAD
    potential::PtrPotential pot;
} PotentialObject;
/// \endcond

static void Potential_dealloc(PotentialObject* self)
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
static potential::PtrPotential Potential_initFromParticles(
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
            "(the first one must be 2d array of shape Nx3 and the second one must be 1d array of length N)");
    }
    particles::ParticleArray<coord::PosCar> pointArray;
    pointArray.data.reserve(numpt);
    for(int i=0; i<numpt; i++) {
        pointArray.add(convertPos((double*)PyArray_GETPTR2(pointCoordArr, i, 0)), 
            *((double*)PyArray_GETPTR1(pointMassArr, i)) * conv->massUnit);
    }
    Py_DECREF(pointCoordArr);
    Py_DECREF(pointMassArr);
    return potential::createPotential(params, pointArray, *conv);
}

/// attempt to construct an elementary potential from the parameters provided in dictionary
static potential::PtrPotential Potential_initFromDict(PyObject* args)
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
static potential::PtrPotential Potential_initFromTuple(PyObject* tuple)
{
    if(PyTuple_Size(tuple) == 1 && PyString_Check(PyTuple_GET_ITEM(tuple, 0)))
    {   // assuming that we have one parameter which is the INI file name
        return potential::createPotential(PyString_AsString(PyTuple_GET_ITEM(tuple, 0)), *conv);
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
static int Potential_init(PotentialObject* self, PyObject* args, PyObject* namedArgs)
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

static bool Potential_isCorrect(PyObject* self)
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

// function that do actually compute something from the potential object,
// applying appropriate unit conversions

static void fncPotential_potential(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    result[0] = ((PotentialObject*)obj)->pot->value(point)
        / pow_2(conv->velocityUnit);   // unit of potential is V^2
}
static PyObject* Potential_potential(PyObject* self, PyObject* args, PyObject* /*namedArgs*/) {
    if(!Potential_isCorrect(self))
        return NULL;
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE>
        (self, args, fncPotential_potential);
}

static void fncPotential_density(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    result[0] = ((PotentialObject*)obj)->pot->density(point)
        / (conv->massUnit / pow_3(conv->lengthUnit));  // unit of density is M/L^3
}
static PyObject* Potential_density(PyObject* self, PyObject* args) {
    if(!Potential_isCorrect(self))
        return NULL;
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE>
        (self, args, fncPotential_density);
}

static void fncPotential_force(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    coord::GradCar grad;
    ((PotentialObject*)obj)->pot->eval(point, NULL, &grad);
    // unit of force per unit mass is V/T
    const double convF = 1 / (conv->velocityUnit/conv->timeUnit);
    result[0] = -grad.dx * convF;
    result[1] = -grad.dy * convF;
    result[2] = -grad.dz * convF;
}
static PyObject* Potential_force(PyObject* self, PyObject* args) {
    if(!Potential_isCorrect(self))
        return NULL;
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_TRIPLET>
        (self, args, fncPotential_force);
}

static void fncPotential_forceDeriv(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    coord::GradCar grad;
    coord::HessCar hess;
    ((PotentialObject*)obj)->pot->eval(point, NULL, &grad, &hess);
    // unit of force per unit mass is V/T
    const double convF = 1 / (conv->velocityUnit/conv->timeUnit);
    // unit of force deriv per unit mass is V/T^2
    const double convD = 1 / (conv->velocityUnit/pow_2(conv->timeUnit));
    result[0] = -grad.dx * convF;
    result[1] = -grad.dy * convF;
    result[2] = -grad.dz * convF;
    result[3] = -hess.dx2  * convD;
    result[4] = -hess.dy2  * convD;
    result[5] = -hess.dz2  * convD;
    result[6] = -hess.dxdy * convD;
    result[7] = -hess.dydz * convD;
    result[8] = -hess.dxdz * convD;
}
static PyObject* Potential_forceDeriv(PyObject* self, PyObject* args) {
    if(!Potential_isCorrect(self))
        return NULL;
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_TRIPLET_AND_SEXTET>
        (self, args, fncPotential_forceDeriv);
}

static PyObject* Potential_name(PyObject* self)
{
    if(!Potential_isCorrect(self))
        return NULL;
    return Py_BuildValue("s", ((PotentialObject*)self)->pot->name());
}

static PyObject* Potential_export(PyObject* self, PyObject* args)
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

static PyObject* Potential_totalMass(PyObject* self)
{
    if(!Potential_isCorrect(self))
        return NULL;
    return Py_BuildValue("d", ((PotentialObject*)self)->pot->totalMass() / conv->massUnit);
}

static PyObject* Potential_sample(PyObject* self, PyObject* args)
{
    if(!Potential_isCorrect(self))
        return NULL;
    return sampleDensity(*((PotentialObject*)self)->pot, args);
}

static PyMethodDef Potential_methods[] = {
    { "name", (PyCFunction)Potential_name, METH_NOARGS,
      "Return the name of the potential\n"
      "No arguments\n"
      "Returns: string" },
    { "potential", (PyCFunction)Potential_potential, METH_VARARGS,
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
    { "sample", Potential_sample, METH_VARARGS, 
      "Sample the density profile with N point masses\n"
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
    { NULL }
};

static PyTypeObject PotentialType = {
    PyObject_HEAD_INIT(NULL)
    0, "agama.Potential",
    sizeof(PotentialObject), 0, (destructor)Potential_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Potential_name, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringPotential,
    0, 0, 0, 0, 0, 0, Potential_methods, 0, 0, 0, 0, 0, 0, 0,
    (initproc)Potential_init
};

/// create a Python Potential object and initialize it with an existing instance of C++ potential class
static PyObject* createPotentialObject(const potential::PtrPotential& pot)
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

/// extract a pointer to C++ Potential class from a Python object,
/// or return an empty pointer on error
static potential::PtrPotential getPotential(PyObject* pot_obj)
{
    if(pot_obj == NULL || !PyObject_TypeCheck(pot_obj, &PotentialType) ||
        !((PotentialObject*)pot_obj)->pot)
        return potential::PtrPotential();    // empty pointer
    return ((PotentialObject*)pot_obj)->pot; // pointer to an existing instance of C++ Potential class
}

/// extract a pointer to C++ Density class from a Python object
static potential::PtrDensity getDensity(PyObject* dens_obj, coord::SymmetryType sym)
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
/// \name  ---------- ActionFinder class and related data ------------
///@{

/// create a spherical or non-spherical action finder
static actions::PtrActionFinder createActionFinder(const potential::PtrPotential& pot)
{
    assert(pot);
    actions::PtrActionFinder af = isSpherical(*pot) ?
        actions::PtrActionFinder(new actions::ActionFinderSpherical(*pot)) :
        actions::PtrActionFinder(new actions::ActionFinderAxisymFudge(pot));
    utils::msg(utils::VL_VERBOSE, "Agama",
        "Created "+std::string(isSpherical(*pot) ? "Spherical" : "Fudge")+
        " action finder for "+pot->name()+" potential at "+utils::toString(af.get()));
    return af;
}

/// \cond INTERNAL_DOCS
/// Python type corresponding to ActionFinder class
typedef struct {
    PyObject_HEAD
    actions::PtrActionFinder af;  // C++ object for action finder
} ActionFinderObject;
/// \endcond

static void ActionFinder_dealloc(ActionFinderObject* self)
{
    utils::msg(utils::VL_VERBOSE, "Agama", "Deleted an action finder at "+
        utils::toString(self->af.get()));
    self->af.reset();
    self->ob_type->tp_free(self);
}

static const char* docstringActionFinder =
    "ActionFinder object is created for a given potential, and its () operator "
    "computes actions for a given position/velocity point, or array of points\n"
    "Arguments: a sextet of floats (x,y,z,vx,vy,vz) or array of such sextets\n"
    "Returns: float or array of floats (for each point: Jr, Jz, Jphi)";

static int ActionFinder_init(PyObject* self, PyObject* args, PyObject* /*namedArgs*/)
{
    PyObject* pot_obj=NULL;
    if(!PyArg_ParseTuple(args, "O", &pot_obj)) {
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
        ((ActionFinderObject*)self)->af = createActionFinder(pot);
        return 0;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, 
            (std::string("Error in ActionFinder initialization: ")+e.what()).c_str());
        return -1;
    }
}

static void fncActions(void* obj, const double input[], double *result) {
    try{
        const coord::PosVelCyl point = coord::toPosVelCyl(convertPosVel(input));
        actions::Actions acts = ((ActionFinderObject*)obj)->af->actions(point);
        // unit of action is V*L
        const double convA = 1 / (conv->velocityUnit * conv->lengthUnit);
        result[0] = acts.Jr   * convA;
        result[1] = acts.Jz   * convA;
        result[2] = acts.Jphi * convA;
    }
    catch(std::exception& ) {  // indicates an error, e.g., positive value of energy
        result[0] = result[1] = result[2] = NAN;
    }
}
static PyObject* ActionFinder_value(PyObject* self, PyObject* args, PyObject* /*namedArgs*/)
{
    if(!((ActionFinderObject*)self)->af) {
        PyErr_SetString(PyExc_ValueError, "ActionFinder object is not properly initialized");
        return NULL;
    }
    return callAnyFunctionOnArray<INPUT_VALUE_SEXTET, OUTPUT_VALUE_TRIPLET>
        (self, args, fncActions);
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
static PyObject* createActionFinderObject(actions::PtrActionFinder af)
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

static void fncActionsStandalone(void* obj, const double input[], double *result) {
    try{
        const coord::PosVelCyl point = coord::toPosVelCyl(convertPosVel(input));
        const ActionFinderParams* params = static_cast<const ActionFinderParams*>(obj);
        double ifd = params->ifd * conv->lengthUnit;
        actions::Actions acts = isSpherical(*params->pot) ?
            actions::actionsSpherical  (*params->pot, point) :
            actions::actionsAxisymFudge(*params->pot, point, ifd);
        // unit of action is V*L
        const double convA = 1 / (conv->velocityUnit * conv->lengthUnit);
        result[0] = acts.Jr   * convA;
        result[1] = acts.Jz   * convA;
        result[2] = acts.Jphi * convA;
    }
    catch(std::exception& ) {  // indicates an error, e.g., positive value of energy
        result[0] = result[1] = result[2] = NAN;
    }
}

static const char* docstringActions = 
    "Compute actions for a given position/velocity point, or array of points\n"
    "Arguments: \n"
    "    point - a sextet of floats (x,y,z,vx,vy,vz) or array of such sextets;\n"
    "    pot - Potential object that defines the gravitational potential;\n"
    "    ifd (float) - interfocal distance for the prolate spheroidal coordinate system "
    "(not necessary if the potential is spherical).\n"
    "Returns: float or array of floats (for each point: Jr, Jz, Jphi)";
static PyObject* find_actions(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"point", "pot", "ifd", NULL};
    double ifd = 0;
    PyObject *points_obj = NULL, *pot_obj = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "|OOd", const_cast<char**>(keywords),
        &points_obj, &pot_obj, &ifd) || ifd<0)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to actions()");
        return NULL;
    }
    ActionFinderParams params;
    params.pot = getPotential(pot_obj);
    params.ifd = ifd;
    if(!params.pot) {
        PyErr_SetString(PyExc_TypeError, "Argument 'pot' must be a valid instance of Potential class");
        return NULL;
    }
    return callAnyFunctionOnArray<INPUT_VALUE_SEXTET, OUTPUT_VALUE_TRIPLET>
        (&params, points_obj, fncActionsStandalone);
}

///@}
/// \name  --------- DistributionFunction class -----------
///@{

/// \cond INTERNAL_DOCS
/// Python type corresponding to DistributionFunction class
typedef struct {
    PyObject_HEAD
    df::PtrDistributionFunction df;
} DistributionFunctionObject;
/// \endcond

static void DistributionFunction_dealloc(DistributionFunctionObject* self)
{
    utils::msg(utils::VL_VERBOSE, "Agama", "Deleted a distribution function at "+
        utils::toString(self->df.get()));
    self->df.reset();
    self->ob_type->tp_free(self);
}

// pointer to the DistributionFunctionType object (will be initialized below)
static PyTypeObject* DistributionFunctionTypePtr;
// forward declaration
static df::PtrDistributionFunction getDistributionFunction(PyObject* df_obj);

static const char* docstringDistributionFunction =
    "DistributionFunction class represents an action-based distribution function.\n\n"
    "The constructor accepts several key=value arguments that describe the parameters "
    "of distribution function.\n"
    "Required parameter is type='...', specifying the type of DF: currently available types are "
    "'DoublePowerLaw', (for the halo), 'PseudoIsothermal' (for the disk component), "
    "'Interp1', 'Interp3' (for interpolated DFs).\n"
    "For some of them, one also needs to provide the potential to initialize the table of "
    "epicyclic frequencies (pot=... argument).\n"
    "Other parameters are specific to each DF type.\n"
    "Alternatively, a composite DF may be created from an array of previously constructed DFs:\n"
    ">>> df = DistributionFunction(df1, df2, df3)\n\n"
    "The () operator computes the value of distribution function for the given triplet of actions.\n"
    "The totalMass() function computes the total mass in the entire phase space.\n\n"
    "A user-defined Python function that takes three arguments(Jr, Jz, Jphi) may be provided "
    "in all contexts where a DistributionFunction object is required.";

/// attempt to construct an interpolated distribution function from the parameters provided in dictionary
template<int N>
static df::PtrDistributionFunction DistributionFunction_initInterpolated(PyObject* namedArgs)
{
    PyObject *u_obj = getItemFromPyDict(namedArgs, "u");  // borrowed reference or NULL
    PyObject *v_obj = getItemFromPyDict(namedArgs, "v");
    PyObject *w_obj = getItemFromPyDict(namedArgs, "w");
    PyObject *ampl_obj = getItemFromPyDict(namedArgs, "ampl");
    if(!u_obj || !v_obj || !w_obj || !ampl_obj)
        throw std::invalid_argument("Interpolated DF requires 4 array arguments: u, v, w, ampl");
    std::vector<double>
        ampl (toFloatArray(ampl_obj)),
        gridU(toFloatArray(u_obj)),
        gridV(toFloatArray(v_obj)),
        gridW(toFloatArray(w_obj));
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
static df::PtrDistributionFunction DistributionFunction_initFromDict(PyObject* namedArgs)
{
    PyObject *pot_obj = PyDict_GetItemString(namedArgs, "pot");  // borrowed reference or NULL
    potential::PtrPotential pot;
    if(pot_obj!=NULL) {
        pot = getPotential(pot_obj);
        if(!pot)
            throw std::invalid_argument("Argument 'pot' must be a valid instance of Potential class");
        PyDict_DelItemString(namedArgs, "pot");
    }
    utils::KeyValueMap params = convertPyDictToKeyValueMap(namedArgs);
    if(!params.contains("type"))
        throw std::invalid_argument("Should provide the type='...' argument");
    std::string type = params.getString("type");
    if(utils::stringsEqual(type, "Interp1"))
        return DistributionFunction_initInterpolated<1>(namedArgs);
    else if(utils::stringsEqual(type, "Interp3"))
        return DistributionFunction_initInterpolated<3>(namedArgs);
    return df::createDistributionFunction(params, pot.get(), *conv);  // any other DF type
}

/// attempt to construct a composite distribution function from a tuple of DistributionFunction objects
static df::PtrDistributionFunction DistributionFunction_initFromTuple(PyObject* tuple)
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
static int DistributionFunction_init(DistributionFunctionObject* self, PyObject* args, PyObject* namedArgs)
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

static void fncDistributionFunction(void* obj, const double input[], double *result)
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

static PyObject* DistributionFunction_value(PyObject* self, PyObject* args, PyObject* /*namedArgs*/)
{
    if(((DistributionFunctionObject*)self)->df==NULL) {
        PyErr_SetString(PyExc_ValueError, "DistributionFunction object is not properly initialized");
        return NULL;
    }
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE>
        (self, args, fncDistributionFunction);
}

static PyObject* DistributionFunction_totalMass(PyObject* self)
{
    if(((DistributionFunctionObject*)self)->df==NULL) {
        PyErr_SetString(PyExc_ValueError, "DistributionFunction object is not properly initialized");
        return NULL;
    }
    double err;
    double val = ((DistributionFunctionObject*)self)->df->totalMass(1e-6, 1e6, &err);
    if(err>1e-6*val)
        utils::msg(utils::VL_WARNING, "Agama", "can't reach tolerance in df->totalMass: "
        "rel.err="+utils::toString(err/val));
    return Py_BuildValue("d", val / conv->massUnit);
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
            value = static_cast<double*>(PyArray_GETPTR1((PyArrayObject*)result, 0))[0];
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
static df::PtrDistributionFunction getDistributionFunction(PyObject* df_obj)
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
static PyObject* createDistributionFunctionObject(df::PtrDistributionFunction df)
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
/// \name  ----- GalaxyModel class -----
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

static void GalaxyModel_dealloc(GalaxyModelObject* self)
{
    Py_XDECREF(self->pot_obj);
    Py_XDECREF(self->df_obj);
    Py_XDECREF(self->af_obj);
    self->ob_type->tp_free((PyObject*)self);
}

static bool GalaxyModel_isCorrect(GalaxyModelObject* self)
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
    "  pot - a Potential object;\n"
    "  df  - a DistributionFunction object;\n"
    "  af (optional) - an ActionFinder object; "
    "if not provided then the action finder is created internally.\n";

static int GalaxyModel_init(GalaxyModelObject* self, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"pot", "df", "af", NULL};
    PyObject *pot_obj = NULL, *df_obj = NULL, *af_obj = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "OO|O", const_cast<char**>(keywords),
        &pot_obj, &df_obj, &af_obj))
    {
        PyErr_SetString(PyExc_ValueError,
            "GalaxyModel constructor takes two or three arguments: pot, df, [af]");
        return -1;
    }

    // check and store the potential
    if(!getPotential(pot_obj)) {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'pot' must be a valid instance of Potential class");
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
static PyObject* GalaxyModel_sample_posvel(GalaxyModelObject* self, PyObject* args)
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
        PyArrayObject* posvel_arr = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        PyArrayObject* mass_arr   = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        for(int i=0; i<numPoints; i++) {
            unconvertPosVel(coord::toPosVelCar(points.point(i)), ((double*)PyArray_DATA(posvel_arr))+i*6);
            ((double*)PyArray_DATA(mass_arr))[i] = points.mass(i) / conv->massUnit;
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
    double accuracy;
    int maxNumEval;
    double vz_error;
    GalaxyModelParams(
        const potential::BasePotential& pot,
        const actions::BaseActionFinder& af,
        const df::BaseDistributionFunction& df) :
        model(pot, af, df) {};
};
/// \endcond

static void fncGalaxyModelMoments(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    GalaxyModelParams* params = static_cast<GalaxyModelParams*>(obj);
    double dens;
    coord::VelCyl vel;
    coord::Vel2Cyl vel2;
    computeMoments(params->model, coord::toPosCyl(point), params->accuracy, params->maxNumEval,
        params->needDens ? &dens : NULL,
        params->needVel  ? &vel  : NULL,
        params->needVel2 ? &vel2 : NULL, NULL, NULL, NULL);
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
static PyObject* GalaxyModel_moments(GalaxyModelObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!GalaxyModel_isCorrect(self))
        return NULL;
    static const char* keywords[] = {"point","dens", "vel", "vel2", NULL};
    PyObject *points_obj = NULL, *dens_flag = NULL, *vel_flag = NULL, *vel2_flag = NULL;
    if(!PyArg_ParseTupleAndKeywords(
        args, namedArgs, "O|OOO", const_cast<char**>(keywords),
        &points_obj, &dens_flag, &vel_flag, &vel2_flag))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to moments()");
        return NULL;
    }
    try{
        GalaxyModelParams params(*self->pot_obj->pot, *self->af_obj->af, *self->df_obj->df);
        params.accuracy = 1e-3;
        params.maxNumEval = 1e5;
        params.needDens = dens_flag==NULL || PyObject_IsTrue(dens_flag);
        params.needVel  = vel_flag !=NULL && PyObject_IsTrue(vel_flag);
        params.needVel2 = vel2_flag==NULL || PyObject_IsTrue(vel2_flag);
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

static void fncGalaxyModelProjectedMoments(void* obj, const double input[], double *result) {
    GalaxyModelParams* params = static_cast<GalaxyModelParams*>(obj);
    try{
        double surfaceDensity, losvdisp;
        computeProjectedMoments(params->model, input[0] * conv->lengthUnit,
            params->accuracy, params->maxNumEval, surfaceDensity, losvdisp);
        result[0] = surfaceDensity * pow_2(conv->lengthUnit) / conv->massUnit;
        result[1] = losvdisp / pow_2(conv->velocityUnit);
    }
    catch(std::exception& ) {
        result[0] = NAN;
        result[1] = NAN;
    }
}

/// compute projected moments of distribution function
static PyObject* GalaxyModel_projectedMoments(GalaxyModelObject* self, PyObject* args)
{
    if(!GalaxyModel_isCorrect(self))
        return NULL;
    PyObject *points_obj = NULL;
    if(!PyArg_ParseTuple(args, "O", &points_obj))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to projected_moments()");
        return NULL;
    }
    try{
        GalaxyModelParams params(*self->pot_obj->pot, *self->af_obj->af, *self->df_obj->df);
        params.accuracy = 1e-3;
        params.maxNumEval = 1e5;
        return callAnyFunctionOnArray<INPUT_VALUE_SINGLE, OUTPUT_VALUE_SINGLE_AND_SINGLE>
            (&params, points_obj, fncGalaxyModelProjectedMoments);
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError,
            (std::string("Error in projectedMoments(): ")+e.what()).c_str());
        return NULL;
    }
}

static void fncGalaxyModelProjectedDF(void* obj, const double input[], double *result) {
    const double R = sqrt(pow_2(input[0]) + pow_2(input[1])) * conv->lengthUnit;
    const double vz = input[2] * conv->velocityUnit;
    // dimension of projected distribution function is M L^-2 V^-1
    const double dim = conv->velocityUnit * pow_2(conv->lengthUnit) / conv->massUnit;
    GalaxyModelParams* params = static_cast<GalaxyModelParams*>(obj);
    try{
        result[0] = computeProjectedDF(params->model, R, vz, params->vz_error,
            params->accuracy, params->maxNumEval) * dim;
    }
    catch(std::exception& ) {
        result[0] = NAN;
    }
}

/// compute projected distribution function
static PyObject* GalaxyModel_projectedDF(GalaxyModelObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!GalaxyModel_isCorrect(self))
        return NULL;
    static const char* keywords[] = {"point","vz_error", NULL};
    PyObject *points_obj = NULL;
    double vz_error = 0;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "O|d", const_cast<char**>(keywords),
        &points_obj, &vz_error))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to projected_df()");
        return NULL;
    }
    try{
        GalaxyModelParams params(*self->pot_obj->pot, *self->af_obj->af, *self->df_obj->df);
        params.accuracy = 1e-4;
        params.maxNumEval = 1e5;
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

static PyMemberDef GalaxyModel_members[] = {
    { const_cast<char*>("pot"), T_OBJECT_EX, offsetof(GalaxyModelObject, pot_obj), READONLY,
      const_cast<char*>("Potential (read-only)") },
    { const_cast<char*>("af"),  T_OBJECT_EX, offsetof(GalaxyModelObject, af_obj ), READONLY,
      const_cast<char*>("Action finder (read-only)") },
    { const_cast<char*>("df"),  T_OBJECT_EX, offsetof(GalaxyModelObject, df_obj ), READONLY,
      const_cast<char*>("Distribution function (read-only)") },
    { NULL }
};

static PyMethodDef GalaxyModel_methods[] = {
    { "sample", (PyCFunction)GalaxyModel_sample_posvel, METH_VARARGS,
      "Sample distribution function in the given potential by N particles\n"
      "Arguments:\n"
      "  Number of particles to sample.\n"
      "Returns:\n"
      "  A tuple of two arrays: position/velocity (2d array of size Nx6) and mass (1d array of length N)." },
    { "moments", (PyCFunction)GalaxyModel_moments, METH_VARARGS | METH_KEYWORDS,
      "Compute moments of distribution function in the given potential\n"
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
      "Compute projected moments of distribution function in the given potential\n"
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
/// \name  ----- Component class for SelfConsistentModel -----
///@{

/// \cond INTERNAL_DOCS
/// Python type corresponding to Component class
typedef struct {
    PyObject_HEAD
    galaxymodel::PtrComponent comp;
    const char* name;
} ComponentObject;
/// \endcond

static void Component_dealloc(ComponentObject* self)
{
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
    "  pot --  an instance of Potential class for a static component with a known potential;\n"
    "  it is mutually exclusive with the 'df' argument.\n"
    "  dens --  an object providing a Density interface (e.g., an instance of "
    "Density or Potential class) that specifies the initial guess for the density profile "
    "for DF-based components (needed to compute the potential on the first iteration), "
    "or a fixed density profile for a static component (optional, and may be combined with "
    "the 'pot' argument).\n"
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

static int Component_init(ComponentObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!onlyNamedArgs(args, namedArgs))
        return -1;
    // check if a potential object was provided
    PyObject* pot_obj = getItemFromPyDict(namedArgs, "pot");
    potential::PtrPotential pot = getPotential(pot_obj);
    if(pot_obj!=NULL && !pot) {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'pot' must be a valid instance of Potential class");
        return -1;
    }
    // check if a density object was provided
    PyObject* dens_obj = getItemFromPyDict(namedArgs, "dens");
    potential::PtrDensity dens = getDensity(dens_obj);
    if(dens_obj!=NULL && !dens) {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'dens' must be a valid Density instance");
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
            "Should provide either a 'pot' and/or 'dens' argument for a static component, "
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
            return 0;
        }
        catch(std::exception& e) {
            PyErr_SetString(PyExc_ValueError,
                (std::string("Error in creating a spheroidal component: ")+e.what()).c_str());
            return -1;
        }
    } else {   // disk-like component
        std::vector<double> gridR(toFloatArray(getItemFromPyDict(namedArgs, "gridR")));
        std::vector<double> gridz(toFloatArray(getItemFromPyDict(namedArgs, "gridz")));
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

static PyObject* Component_name(PyObject* self)
{
    return Py_BuildValue("s", ((ComponentObject*)self)->name);
}

static PyObject* Component_getPotential(ComponentObject* self)
{
    potential::PtrPotential pot = self->comp->getPotential();
    if(pot)
        return createPotentialObject(pot);
    // otherwise no potential is available (e.g. for a df-based component)
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* Component_getDensity(ComponentObject* self)
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
/// \name  --------- SelfConsistentModel class -----------
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

static void SelfConsistentModel_dealloc(SelfConsistentModelObject* self)
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

static int SelfConsistentModel_init(SelfConsistentModelObject* self, PyObject* args, PyObject* namedArgs)
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

static PyObject* SelfConsistentModel_iterate(SelfConsistentModelObject* self)
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
    { const_cast<char*>("pot"), T_OBJECT, offsetof(SelfConsistentModelObject, pot), READONLY,
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
/// \name  --------- SplineApprox class -----------
///@{

/// \cond INTERNAL_DOCS
/// Python type corresponding to SplineApprox class
typedef struct {
    PyObject_HEAD
    math::CubicSpline spl;
} SplineApproxObject;
/// \endcond

static void SplineApprox_dealloc(SplineApproxObject* self)
{
    self->ob_type->tp_free((PyObject*)self);
}

static const char* docstringSplineApprox = 
    "SplineApprox is a class that deals with smoothing splines.\n"
    "It approximates a large set of (x,y) points by a smooth curve with "
    "a rather small number of knots, which should encompass the entire range "
    "of input x values, but preferrably in such a way that each interval "
    "between knots contains at least one x-value from the set of input points.\n"
    "The smoothness of the approximating spline is adjusted by an optional "
    "input parameter `smooth`, which determines the tradeoff between smoothness "
    "and approximation error; zero means no additional smoothing (beyond the one "
    "resulting from discreteness of the spacing of knots), and values around "
    "unity usually yield a reasonable smoothing of noise without sacrificing "
    "too much of accuracy.\n"
    "Values of the spline and up to its second derivative are computed using "
    "the () operator with the first argument being a single x-point or an array "
    "of points, and optional second argument being the derivative index (0, 1, or 2).";

static int SplineApprox_init(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    // "dirty hack" (see above) to construct a C++ object in an already allocated chunk of memory
    new (&(((SplineApproxObject*)self)->spl)) math::CubicSpline;
    static const char* keywords[] = {"knots","x","y","smooth",NULL};
    PyObject* k_obj=NULL;
    PyObject* x_obj=NULL;
    PyObject* y_obj=NULL;
    double smoothfactor=0;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "OOO|d", const_cast<char **>(keywords),
        &k_obj, &x_obj, &y_obj, &smoothfactor)) {
        PyErr_SetString(PyExc_ValueError, "Incorrect parameters passed to the SplineApprox constructor: "
            "must provide an array of grid nodes and two arrays of equal length (input x and y points), "
            "and optionally a float (smooth factor)");
        return -1;
    }
    std::vector<double>
        knots  (toFloatArray(k_obj)),
        xvalues(toFloatArray(x_obj)),
        yvalues(toFloatArray(y_obj));
    if(xvalues.empty() || yvalues.empty() || knots.empty()) {
        PyErr_SetString(PyExc_ValueError, "Input does not contain valid arrays");
        return -1;
    }
    if(knots.size() < 2|| xvalues.size() != yvalues.size()) {
        PyErr_SetString(PyExc_ValueError,
            "Arguments must be an array of grid nodes (at least 2) and two arrays of equal length (x and y)");
        return -1;
    }
    try{
        math::SplineApprox spl(knots, xvalues);
        std::vector<double> amplitudes;
        if(smoothfactor>0)
            amplitudes = spl.fitOversmooth(yvalues, smoothfactor);
        else
            amplitudes = spl.fit(yvalues, -smoothfactor);
        ((SplineApproxObject*)self)->spl = math::CubicSpline(knots, amplitudes);
        return 0;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError,
            (std::string("Error in SplineApprox initialization: ")+e.what()).c_str());
        return -1;
    }
}

static double spl_eval(const math::CubicSpline& spl, double x, int der=0)
{
    double result;
    switch(der) {
        case 0: return spl.value(x);
        case 1: spl.evalDeriv(x, NULL, &result); return result;
        case 2: spl.evalDeriv(x, NULL, NULL, &result); return result;
        default: return NAN;
    }
}

static PyObject* SplineApprox_value(PyObject* self, PyObject* args, PyObject* /*kw*/)
{
    PyObject* ptx=NULL;
    int der=0;
    if(self==NULL || ((SplineApproxObject*)self)->spl.isEmpty()) {
        PyErr_SetString(PyExc_ValueError, "SplineApprox object is not properly initialized");
        return NULL;
    }
    if(!PyArg_ParseTuple(args, "O|i", &ptx, &der)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments");
        return NULL;
    }
    if(der>2) {
        PyErr_SetString(PyExc_ValueError, "Can only compute derivatives up to 2nd");
        return NULL;
    }
    if(PyFloat_Check(ptx))  // one value
        return Py_BuildValue("d", spl_eval(((SplineApproxObject*)self)->spl, PyFloat_AsDouble(ptx), der) );
    // else an array of values
    PyArrayObject *arr = (PyArrayObject*)
        PyArray_FROM_OTF(ptx, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY | NPY_ARRAY_ENSURECOPY);
    if(arr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Argument must be either float, list or numpy array");
        return NULL;
    }
    // replace elements of the copy of input array with computed values
    for(int i=0; i<PyArray_SIZE(arr); i++)
        ((double*)PyArray_DATA(arr))[i] = 
            spl_eval(((SplineApproxObject*)self)->spl, ((double*)PyArray_DATA(arr))[i], der);
    return PyArray_Return(arr);
}

static PyMethodDef SplineApprox_methods[] = {
    { NULL, NULL, 0, NULL }  // no named methods
};

static PyTypeObject SplineApproxType = {
    PyObject_HEAD_INIT(NULL)
    0, "agama.SplineApprox",
    sizeof(SplineApproxObject), 0, (destructor)SplineApprox_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, SplineApprox_value, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringSplineApprox, 
    0, 0, 0, 0, 0, 0, SplineApprox_methods, 0, 0, 0, 0, 0, 0, 0,
    SplineApprox_init
};


///@}
/// \name  ----- Orbit integration -----
///@{

/// description of orbit function
static const char* docstringOrbit = 
    "Compute an orbit starting from the given initial conditions in the given potential\n"
    "Arguments:\n"
    "    ic=float[6] : initial conditions - an array of 6 numbers "
    "(3 positions and 3 velocities in Cartesian coordinates);\n"
    "    pot=Potential object that defines the gravitational potential;\n"
    "    time=float : total integration time;\n"
    "    step=float : output timestep (does not affect the integration accuracy);\n"
    "    acc=float, optional : relative accuracy parameter (default 1e-10).\n"
    "Returns: an array of Nx6 numbers, where N=time/step is the number of output points "
    "in the trajectory, and each point consists of position and velocity in Cartesian coordinates.";

/// orbit integration
static PyObject* integrate_orbit(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"ic", "pot", "time", "step", "acc", NULL};
    double time = 0, step = 0, acc = 1e-10;
    PyObject *ic_obj = NULL, *pot_obj = NULL;
    if(!PyArg_ParseTupleAndKeywords(
        args, namedArgs, "|OOddd", const_cast<char**>(keywords),
        &ic_obj, &pot_obj, &time, &step, &acc) ||
        time<=0 || step<=0 || acc<=0)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to orbit()");
        return NULL;
    }
    if(!PyObject_TypeCheck(pot_obj, &PotentialType) || 
        ((PotentialObject*)pot_obj)->pot==NULL ) {
        PyErr_SetString(PyExc_TypeError, "Argument 'pot' must be a valid instance of Potential class");
        return NULL;
    }
    PyArrayObject *ic_arr  = (PyArrayObject*) PyArray_FROM_OTF(ic_obj,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(ic_arr == NULL || PyArray_NDIM(ic_arr) != 1 || PyArray_DIM(ic_arr, 0) != 6) {
        Py_XDECREF(ic_arr);
        PyErr_SetString(PyExc_ValueError, "Argument 'ic' does not contain a valid array of length 6");
        return NULL;
    }
    // initialize
    const coord::PosVelCar ic_point(
        ((double*)PyArray_DATA(ic_arr))[0] * conv->lengthUnit, 
        ((double*)PyArray_DATA(ic_arr))[1] * conv->lengthUnit, 
        ((double*)PyArray_DATA(ic_arr))[2] * conv->lengthUnit,
        ((double*)PyArray_DATA(ic_arr))[3] * conv->velocityUnit, 
        ((double*)PyArray_DATA(ic_arr))[4] * conv->velocityUnit, 
        ((double*)PyArray_DATA(ic_arr))[5] * conv->velocityUnit);
    std::vector<coord::PosVelCar> traj;
    Py_DECREF(ic_arr);
    // integrate
    try{
        orbit::integrate( *((PotentialObject*)pot_obj)->pot, ic_point, 
            time * conv->timeUnit, step * conv->timeUnit, traj, acc);
        // build an appropriate output array
        const unsigned int size = traj.size();
        npy_intp dims[] = {size, 6};
        PyArrayObject* result = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if(!result)
            return NULL;
        for(unsigned int index=0; index<size; index++) {
            ((double*)PyArray_DATA(result))[index*6  ] = traj[index].x  / conv->lengthUnit;
            ((double*)PyArray_DATA(result))[index*6+1] = traj[index].y  / conv->lengthUnit;
            ((double*)PyArray_DATA(result))[index*6+2] = traj[index].z  / conv->lengthUnit;
            ((double*)PyArray_DATA(result))[index*6+3] = traj[index].vx / conv->velocityUnit;
            ((double*)PyArray_DATA(result))[index*6+4] = traj[index].vy / conv->velocityUnit;
            ((double*)PyArray_DATA(result))[index*6+5] = traj[index].vz / conv->velocityUnit;
        }
        return (PyObject*)result;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, 
            (std::string("Error in orbit computation: ")+e.what()).c_str());
        return NULL;
    }
}
///@}
/// \name  ----- Math routines -----
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
static PyObject* nonuniformGrid(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"nnodes", "xmin", "xmax", NULL};
    int nnodes=-1;
    double xmin=-1, xmax=-1;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "idd", const_cast<char**>(keywords),
        &nnodes, &xmin, &xmax) || nnodes<2 || xmin<=0 || xmax<=xmin)
    {
        PyErr_SetString(PyExc_ValueError, "Incorrect arguments for nonuniformGrid");
        return NULL;
    }
    std::vector<double> grid = math::createNonuniformGrid(nnodes, xmin, xmax, true);
    npy_intp size = grid.size();
    PyArrayObject* result = (PyArrayObject*)PyArray_SimpleNew(1, &size, NPY_DOUBLE);
    if(!result)
        return NULL;
    for(npy_intp index=0; index<size; index++)
        ((double*)PyArray_DATA(result))[index] = grid[index];
    return (PyObject*)result;
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
            values[0] = static_cast<double*>(PyArray_GETPTR1((PyArrayObject*)result, 0))[0];
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
static bool parseLowerUpperBounds(PyObject* lower_obj, PyObject* upper_obj,
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
        xlow[d] = static_cast<double*>(PyArray_DATA(lower_arr))[d];
        xupp[d] = static_cast<double*>(PyArray_DATA(upper_arr))[d];
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
static PyObject* integrateNdim(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
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
static PyObject* sampleNdim(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"fnc", "nsamples", "lower", "upper", NULL};
    int numSamples=-1, numEval=-1;
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
        math::sampleNdim(fnc, &xlow[0], &xupp[0], numSamples, samples, &numEval, &result, &error);
        npy_intp dim[] = {numSamples, xlow.size()};
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

static PyMethodDef module_methods[] = {
    { "setUnits", (PyCFunction)setUnits, METH_VARARGS | METH_KEYWORDS, docstringSetUnits },
    { "resetUnits", resetUnits, METH_NOARGS, docstringResetUnits },
    { "nonuniformGrid", (PyCFunction)nonuniformGrid, METH_VARARGS | METH_KEYWORDS, docstringNonuniformGrid },
    { "orbit", (PyCFunction)integrate_orbit, METH_VARARGS | METH_KEYWORDS, docstringOrbit },
    { "actions", (PyCFunction)find_actions, METH_VARARGS | METH_KEYWORDS, docstringActions },
    { "integrateNdim", (PyCFunction)integrateNdim, METH_VARARGS | METH_KEYWORDS, docstringIntegrateNdim },
    { "sampleNdim", (PyCFunction)sampleNdim, METH_VARARGS | METH_KEYWORDS, docstringSampleNdim },
    { NULL }
};

} // end internal namespace

PyMODINIT_FUNC
initagama(void)
{
    PyObject* mod = Py_InitModule("agama", module_methods);
    if(!mod) return;
    conv.reset(new units::ExternalUnits());

    DensityType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&DensityType) < 0) return;
    Py_INCREF(&DensityType);
    PyModule_AddObject(mod, "Density", (PyObject*)&DensityType);

    PotentialType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&PotentialType) < 0) return;
    Py_INCREF(&PotentialType);
    PyModule_AddObject(mod, "Potential", (PyObject*)&PotentialType);
    PotentialTypePtr = &PotentialType;

    ActionFinderType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&ActionFinderType) < 0) return;
    Py_INCREF(&ActionFinderType);
    PyModule_AddObject(mod, "ActionFinder", (PyObject*)&ActionFinderType);

    DistributionFunctionType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&DistributionFunctionType) < 0) return;
    Py_INCREF(&DistributionFunctionType);
    PyModule_AddObject(mod, "DistributionFunction", (PyObject*)&DistributionFunctionType);
    DistributionFunctionTypePtr = &DistributionFunctionType;

    GalaxyModelType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&GalaxyModelType) < 0) return;
    Py_INCREF(&GalaxyModelType);
    PyModule_AddObject(mod, "GalaxyModel", (PyObject*)&GalaxyModelType);

    ComponentType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&ComponentType) < 0) return;
    Py_INCREF(&ComponentType);
    PyModule_AddObject(mod, "Component", (PyObject*)&ComponentType);

    SelfConsistentModelType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&SelfConsistentModelType) < 0) return;
    Py_INCREF(&SelfConsistentModelType);
    PyModule_AddObject(mod, "SelfConsistentModel", (PyObject*)&SelfConsistentModelType);

    SplineApproxType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&SplineApproxType) < 0) return;
    Py_INCREF(&SplineApproxType);
    PyModule_AddObject(mod, "SplineApprox", (PyObject*)&SplineApproxType);

    import_array();  // needed for NumPy to work properly
}
// ifdef HAVE_PYTHON
#endif