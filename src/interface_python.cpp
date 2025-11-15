/** \file   interface_python.cpp
    \brief  Python interface for the Agama library
    \author Eugene Vasiliev
    \date   2014-2025

    This is a Python extension module that provides the interface to
    some of the classes and functions from the Agama C++ library.
    It needs to be compiled into a dynamic library and placed in a folder
    that Python is aware of (e.g., through the PYTHONPATH= environment variable).

    Currently this module provides access to potential classes, orbit integration
    routine, action finders and mappers, distribution functions, self-consistent
    and Schwarzachild orbit-superposition models, N-dimensional integration and
    sampling routines, spline-related tools, linear and quadratic optimization,
    N-body snapshot handling, and a few helper routines.
    Unit conversion is also part of the calling convention: the quantities
    received from Python are assumed to be in some physical units and converted
    into internal units inside this module, and the output from the Agama library
    routines is converted back to physical units. The physical units are assigned
    by calling `setUnits` at the beginning of the script; one may also choose to
    work in N-body units (G=1) and skip the unit conversion entirely.

    Type `dir(agama)` in Python to get a list of exported routines and classes,
    and `help(agama.whatever)` to get the usage syntax for each of them.

    In addition to the C++ extension module provided by the shared library agama.so,
    there is also a native Python part of the agama Python module, provided by
    py/pygama.py; the routines from this submodule are imported into the main
    namespace on importing agama.
*/
#ifdef HAVE_PYTHON
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <structmember.h>
#include <stdexcept>
#include <complex>
#include <algorithm>
#ifdef _OPENMP
#include "omp.h"
#endif
// include almost everything from Agama!
#include "actions_factory.h"
#include "df_factory.h"
#include "galaxymodel_base.h"
#include "galaxymodel_densitygrid.h"
#include "galaxymodel_losvd.h"
#include "galaxymodel_selfconsistent.h"
#include "galaxymodel_velocitysampler.h"
#include "math_core.h"
#include "math_gausshermite.h"
#include "math_optimization.h"
#include "math_random.h"
#include "math_sample.h"
#include "math_spline.h"
#include "particles_io.h"
#include "potential_analytic.h"
#include "potential_composite.h"
#include "potential_factory.h"
#include "potential_multipole.h"
#include "potential_utils.h"
#include "orbit.h"
#include "orbit_variational.h"
#include "units.h"
#include "utils.h"
#include "utils_config.h"
// text string embedded into the python module as the __version__ attribute (including Github commit number)
#define AGAMA_VERSION "1.0.156 compiled on " __DATE__

// older versions of numpy have different macro names
// (will need to expand this list if other similar macros are used in the code)
#ifndef NPY_ARRAY_IN_ARRAY
#define NPY_ARRAY_IN_ARRAY   NPY_IN_ARRAY
#define NPY_ARRAY_OUT_ARRAY  NPY_OUT_ARRAY
#define NPY_ARRAY_FORCECAST  NPY_FORCECAST
#define NPY_ARRAY_ENSURECOPY NPY_ENSURECOPY
#endif

// compatibility with Python 3
#if PY_MAJOR_VERSION >= 3
#define PyString_Check PyUnicode_Check
#define PyString_AsString PyUnicode_AsUTF8
#define PyString_FromString PyUnicode_FromString
#define PyInt_Check PyLong_Check
#define PyInt_AsLong PyLong_AsLong
#define PyInt_FromLong PyLong_FromLong
#else
#define Py_hash_t long
#endif
#if (PY_MAJOR_VERSION < 3 || PY_MINOR_VERSION < 13)
#define Py_HashPointer _Py_HashPointer
#endif

/// utility snippet for allocating temporary storage either on stack (if small) or on heap otherwise
#define ALLOC(NPOINTS, TYPE, NAME) \
    std::vector< TYPE > tmparray; \
    TYPE* NAME; \
    if(NPOINTS * sizeof(TYPE) > 65536) { \
        tmparray.resize(NPOINTS); \
        NAME = &tmparray[0]; \
    } else \
    NAME = static_cast<TYPE*>(alloca(NPOINTS * sizeof(TYPE)));

/// classes and routines for the Python interface
namespace pygama {  // internal namespace

PyObject* thismodule;  // PyObject corresponding to this extension module

// some forward declarations:
// pointers to several Python type descriptors, which will be initialized at module startup
static PyTypeObject
    *DensityTypePtr,
    *PotentialTypePtr,
    *ActionFinderTypePtr,
    *ActionMapperTypePtr,
    *DistributionFunctionTypePtr,
    *SelectionFunctionTypePtr,
    *TargetTypePtr,
    *SplineTypePtr,
    *OrbitTypePtr;

// forward declaration for a routine that constructs a Python cubic spline object
PyObject* createCubicSpline(const std::vector<double>& x, const std::vector<double>& y);

// an annoying feature in Python C API is the use of different types to refer to the same object,
// which triggers a warning about breaking strict aliasing rules, unless compiled
// with -fno-strict-aliasing. To avoid this, we use a dirty typecast.
void* forget_about_type(void* x) { return x; }
#define Py_INCREFx(x) Py_INCREF(forget_about_type(x))


//  -------------------------------
/// \name  Multi-threading support
//  -------------------------------
///@{

/// Class and context manager for setting the number of OpenMP threads;
/// this used to be a standalone function, but it could not be used as a context manager,
/// so is now a class with all work done in the constructor (the name still begins with lowercase)
static const char* docstringSetNumThreads =
    "Set the number of OpenMP threads (if the module was compiled with OpenMP support).\n"
    "A single argument specifies the number of threads in parallellized operations; "
    "0 means system default (typically equal to the number of processor cores, "
    "unless explicitly changed, e.g., by the environment variable OMP_NUM_THREADS).\n"
    "One can use a context manager to temporarily change the number of threads in a code block, "
    "for instance, setting it to 1 when calling some Agama routines with user-defined Python "
    "callback functions, which effectively disable parallelization anyway - it makes sense "
    "to do it explicitly to avoid thread-switching overheads:\n"
    "    with setNumThreads(1):\n"
    "        # call some Agama routines\n";

typedef struct {
    PyObject_HEAD
    int prevNumThreads;
    int currNumThreads;
} setNumThreadsObject;

int setNumThreads_init(setNumThreadsObject* self, PyObject* arg, PyObject* /*namedArgs*/)
{
    if(!self)
        return -1;
    if(!PyArg_ParseTuple(arg, "i", &self->currNumThreads)) {
        PyErr_SetString(PyExc_TypeError,
            "Expected one int argument -- the number of threads (0 means system default)");
        return -1;
    }
    if(self->currNumThreads < 0) {
        PyErr_SetString(PyExc_ValueError,
            "Number of threads must be non-negative (0 means system default)");
        return -1;
    }
#ifdef _OPENMP
    self->prevNumThreads = omp_get_max_threads();
    static int maxThreads = -1;
    if(maxThreads == -1)  // first time this routine is called, remember the actual max # of threads
        maxThreads = self->prevNumThreads;
    if(self->currNumThreads == 0)
        self->currNumThreads = maxThreads;  // restore the original (system-default) max # of threads
    omp_set_num_threads(self->currNumThreads);
#else
    PyErr_WarnEx(NULL, "OpenMP not available\n", 1);
#endif
    return 0;
}

PyObject* setNumThreads_enter(PyObject* self)
{
    // nothing to do - all work done in the init function
    Py_INCREF(self);
    return self;
}

PyObject* setNumThreads_exit(setNumThreadsObject* self, PyObject* /*arg*/)
{
#ifdef _OPENMP
    // restore the previous number of threads saved by the init function
    omp_set_num_threads(self->prevNumThreads);
#else
    (void)self;
#endif
    Py_INCREFx(Py_False);
    return Py_False;
}

PyObject* setNumThreads_repr(setNumThreadsObject* self)
{
    return Py_BuildValue("s", utils::toString(self->currNumThreads).c_str());
}

static PyMethodDef setNumThreads_methods[] = {
  { "__enter__", (PyCFunction)setNumThreads_enter, METH_NOARGS, "enter context manager" },
  { "__exit__",  (PyCFunction)setNumThreads_exit,  METH_VARARGS, "exit context manager" },
  { NULL }
};

static PyMemberDef setNumThreads_members[] = {
  { const_cast<char*>("prevNumThreads"), T_INT, offsetof(setNumThreadsObject, prevNumThreads), READONLY,
    const_cast<char*>("previous number of OpenMP threads (before calling setNumThreads)") },
  { const_cast<char*>("currNumThreads"), T_INT, offsetof(setNumThreadsObject, currNumThreads), READONLY,
    const_cast<char*>("current number of OpenMP threads (after calling setNumThreads)") },
  { NULL }
};

static PyTypeObject setNumThreadsType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "agama.setNumThreads",
    sizeof(setNumThreadsObject), 0, 0,
    0, 0, 0, 0, (reprfunc)setNumThreads_repr, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringSetNumThreads,
    0, 0, 0, 0, 0, 0, setNumThreads_methods, setNumThreads_members, 0, 0, 0, 0, 0, 0,
    (initproc)setNumThreads_init
};


#ifdef _OPENMP
/** Lock-type class for temporarily releasing Python's GIL.
    An instance of this class releases GIL in the constructor and reacquires it in the destructor,
    following the RAII idiom:
    \code
    {   // open a block
        PyReleaseGIL unlock;
        ..do something..
    }   // destructor is called on exiting the block
    \endcode

    This procedure is recommended but _optional_ when the "do something" code performs some
    lengthy computation or IO operation without calling any Python C API function.
    In the case that Python C API is needed inside such a block, the GIL may be reacquired
    with the help of the PyAcquireGIL class. Note that the latter procedure can be performed
    even without releasing GIL (in which case it is a no-op).
    It becomes _necessary_ when the code contains OpenMP-parallelized fragments, which themselves
    contain Python C API calls. In this case these calls still need to be protected by PyAcquireGIL,
    but a prior GIL release is required, since the GIL may need to be acquired in a different thread.

    We need to use this class whenever we call pure C++ routines that may involve OpenMP-parallelized
    loops and that receive pointers to Density/Potential/DistributionFunction/SelectionFunction
    objects that may possibly contain Python callback functions.
    The BatchFunction::run() method covers most of these cases, but we also need to take care of
    a few more situations when a call to some constructor (e.g. ActionFinder) or a method may
    receive a pointer to a Python callback function.
    Finding all such cases is a rather delicate task!
*/
class PyReleaseGIL {
    PyThreadState* const state;
public:
    PyReleaseGIL() : state(PyEval_SaveThread()) {}
    ~PyReleaseGIL() { PyEval_RestoreThread(state); }
};

/** A counterpart to the PyReleaseGIL class, whose purpose is the opposite
    (see the description above).
    The usage pattern follows the same RAII idiom, creating an instance of PyAcquireGIL class
    in a block that contains calls to Python C API.
    Note that it is allowed to create multiple instances of this class in nested code blocks,
    regardless of whether the enclosing scope is run under PyReleaseGIL or not -
    if the current thread already holds GIL, it is a no-op.
*/
class PyAcquireGIL {
    const PyGILState_STATE state;
public:
    PyAcquireGIL() : state(PyGILState_Ensure()) {}
    ~PyAcquireGIL() { PyGILState_Release(state); }
};
#else
// no-ops
struct PyReleaseGIL {
    PyReleaseGIL() {}
};
struct PyAcquireGIL {
    PyAcquireGIL() {}
};
#endif

///@}
//  ---------------------------
/// \name  Progress indication
//  ---------------------------
///@{

/// global pointer to the tqdm python class that displays a fancy progress bar (if available)
PyObject* tqdmClass = NULL;

/** Helper class for displaying progress indication during long computations,
    using a fancy tdqm progress bar if this module is available, otherwise a simple percentage */
class ProgressBar {
public:
    PyObject* tqdmInstance;      ///< tdqm progress bar instance, if needed
    bool initialized;            ///< whether the progress indication has begun
    Py_ssize_t numTotal;         ///< total number of operations
    Py_ssize_t prevNumCompleted; ///< number of completed operations on previous report
    double prevDeltaSeconds;     ///< time since the beginning of computations on previous report
    const char* unitStr;         ///< units of operations (points, orbits, etc.)
    utils::Timer timer;          ///< timer started at the beginning of computations
    const double minTotalTime;   ///< threshold for creating the progress bar (seconds)
    const double minUpdateTime;  ///< threshold for updating the progress bar (seconds)
    long numUpdateCalls;         ///< keeps track of the number of calls to update()
    int minNumUpdateCalls;       ///< minimum number of update calls before the progress bar is shown

    ProgressBar(Py_ssize_t total, const char* unit, double _minTotalTime=5.0, double _minUpdateTime=1.0) :
        tqdmInstance(NULL), initialized(false), numTotal(total), prevNumCompleted(0), prevDeltaSeconds(0),
        unitStr(unit), minTotalTime(_minTotalTime), minUpdateTime(_minUpdateTime), numUpdateCalls(0)
    {
#ifdef _OPENMP
        // when running an OpenMP-parallelized loop, wait until each thread finishes its first batch of
        // operations before showing the progress bar, to get a more reliable estimate of the total time
        minNumUpdateCalls = omp_get_max_threads();
#else
        minNumUpdateCalls = 1;
#endif
    }

    ~ProgressBar() {
        clear();
    }

    /** Update the progress bar with the current number of completed operations.
        Initially, the progress bar is not created: this happens only after some time (e.g. 2 seconds)
        has elapsed and the remaining time is expected to be greater than some threshold (e.g. 5 seconds).
        The update only happens once per second, no matter how often this method is called.
        Can be invoked from multiple threads, but the updating code is protected by a critical section.
    */
    void update(Py_ssize_t numCompleted)
    {
        if(utils::verbosityLevel <= utils::VL_DISABLE)
            return;  // suppress the progress indication altogether
#ifdef _OPENMP
#pragma omp atomic
#endif
        ++numUpdateCalls;
        double deltaSeconds = timer.deltaSeconds();
        if(deltaSeconds - prevDeltaSeconds < (initialized ? minUpdateTime : 2*minUpdateTime) ||
            numUpdateCalls < minNumUpdateCalls)
            return;  // do not update more often than once per second
        prevDeltaSeconds = deltaSeconds;
        if(numCompleted >= numTotal)
            return;  // skip any updates when already finished
        // create the progress bar only when the estimated completion time is long enough
        if(!initialized && numTotal * deltaSeconds < numCompleted * minTotalTime)
            return;
        PyAcquireGIL lock;
        // In a free-threaded Python, the absence of GIL lock does not prevent multiple threads
        // from reaching past this point simultaneously, so an OpenMP critical section is also needed
#ifdef Py_GIL_DISABLED
#ifdef _OPENMP
#pragma omp critical
#endif
#endif
        {
            if(!initialized && tqdmClass) {
                // when first reached here, try to create an instance of a fancy progress bar.
                PyObject
                    *args  = PyTuple_New(0),
                    *kwargs= PyDict_New(),
                    *count = PyInt_FromLong(numTotal),
                    *value = PyInt_FromLong(numCompleted),
                    *unit  = PyString_FromString(unitStr);
                PyDict_SetItemString(kwargs, "total", count);
                PyDict_SetItemString(kwargs, "initial", value);
                PyDict_SetItemString(kwargs, "unit", unit);
                PyDict_SetItemString(kwargs, "leave", Py_False);
                tqdmInstance = PyObject_Call(tqdmClass, args, kwargs);
                Py_DECREF(args);
                Py_DECREF(kwargs);
                Py_DECREF(count);
                Py_DECREF(value);
                Py_DECREF(unit);
                if(!tqdmInstance)
                    // failure to create an instance of progress bar is not critical, suppress the error
                    PyErr_Clear();
                prevNumCompleted = numCompleted;
            }
            initialized = true;  // mark tqdm initialization as complete, even if it failed
            PyObject* result = NULL;
            if(tqdmInstance) {
                result = PyObject_CallMethod(tqdmInstance, const_cast<char*>("update"),
                    const_cast<char*>("i"), numCompleted-prevNumCompleted);
                prevNumCompleted = numCompleted;
            }
            if(result)  // successfully shown a progress bar using tqdm
                Py_DECREF(result);
            else {  // otherwise print a simple progress indicator to python stderr
                PySys_WriteStderr("%li/%li %ss complete\r",
                    (long int)numCompleted, (long int)numTotal, unitStr);
            }
        }
    }

    /** Clean up the progress bar after the end of computations (call only from the main thread) */
    void clear()
    {
        if(!initialized)
            return;
        if(tqdmInstance) {  // clean up the progress bar
            PyObject* result = PyObject_CallMethod(tqdmInstance, const_cast<char*>("close"), NULL);
            Py_XDECREF(result);
            Py_DECREF(tqdmInstance);
        } else {  // overwrite the previously printed text (numCompleted/numTotal) with spaces
            int numSpaces = 2 * (log10(numTotal)+1) + 12 + strlen(unitStr);
            std::string spaces(numSpaces, ' ');
            PySys_WriteStderr("%s\r", spaces.c_str());
        }
        initialized = false;
    }
};

///@}
//  ------------------------------------------------------------------
/// \name  Helper routines for type conversions and argument checking
//  ------------------------------------------------------------------
///@{

/// check that a Python object is a callable function that accepts a 2d NxD array and returns
/// a 1d array of length N with a valid numerical type, but is *not* any of the agama classes
bool checkCallable(PyObject* fnc, int dim)
{
    if(PyCallable_Check(fnc) &&
        // make sure it's not one of the other classes in this module which provide a call interface
        !PyObject_TypeCheck(fnc, ActionFinderTypePtr) &&
        !PyObject_TypeCheck(fnc, ActionMapperTypePtr) &&
        !PyObject_TypeCheck(fnc, DistributionFunctionTypePtr) &&
        !PyObject_TypeCheck(fnc, SelectionFunctionTypePtr) &&
        !PyObject_TypeCheck(fnc, TargetTypePtr) )
    {
        const int Npoints= 2;  // number of input points
        npy_intp dims[]  = {Npoints, dim};
        PyObject* args   = PyArray_ZEROS(2, dims, NPY_DOUBLE, /*fortran order*/0);
        PyObject* result = PyObject_CallFunctionObjArgs(fnc, args, NULL);
        Py_DECREF(args);
        if(result == NULL) {
            return false;  // callable but incorrect; keep the error message set by the above call
        }
        bool success = false;
        if(PyArray_Check(result)) {
            int type = PyArray_TYPE((PyArrayObject*) result);
            success  = PyArray_NDIM((PyArrayObject*) result)==1 &&
                PyArray_DIM((PyArrayObject*) result, 0) == Npoints &&
                (type==NPY_FLOAT || type==NPY_DOUBLE || type==NPY_BOOL);
            if(!success)  // callable but incorrect result - report this
                PyErr_SetString(PyExc_TypeError, "Invalid array returned by the user-provided function");
        } else
            PyErr_SetString(PyExc_TypeError, "User-provided function should return an array");
        Py_DECREF(result);
        return success;
    }
    else
        return false;  // not a callable, no error is set
}

/// return a string representation of a Python object
std::string toString(PyObject* obj)
{
    if(obj==NULL)
        return "";
    if(PyString_Check(obj))
        return std::string(PyString_AsString(obj));
#if PY_MAJOR_VERSION==2
    if(PyUnicode_Check(obj)) {
        PyObject* str = PyUnicode_AsUTF8String(obj);
        std::string result(PyString_AsString(str));
        Py_DECREF(str);
        return result;
    }
#endif
    if(PyNumber_Check(obj)) {
        double value = PyFloat_AsDouble(obj);
        if(!PyErr_Occurred())
            return utils::toString(value, 18);  // keep full precision in the string
        PyErr_Clear();   // otherwise something went wrong in conversion, carry on..
    }
    // if the input is a tuple/list/array, recursively serialize its elements into
    // a comma-separated string surrounded by brackets;
    // multidimensional arrays appear as nested sequences of brackets, e.g. [[1,2],[3,4]]
    if(PySequence_Check(obj)) {   // sequence but not an array
        Py_ssize_t size = PySequence_Size(obj);
        std::string str = "[";
        for(Py_ssize_t i=0; i<size; i++) {
            if(i>0) str += ",";
            PyObject* elem = PySequence_GetItem(obj, i);
            str += toString(elem);
            Py_XDECREF(elem);
        }
        str += "]";
        return str;
    }
    PyObject* s = PyObject_Str(obj);
    if(!s)
        return "";
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
        if(PyErr_Occurred()) {
            PyErr_Clear();
            return defaultValue;
        }
        return value;
    }
    // it wasn't a number, but may be it can be converted to a number
    PyObject* l = PyNumber_Long(obj);
    if(l) {
        int value = PyInt_AsLong(l);
        Py_DECREF(l);
        return value;
    }
    if(PyErr_Occurred())
        PyErr_Clear();
    return defaultValue;
}

/// return a float representation of a Python object, or a default value in case of error
double toDouble(PyObject* obj, double defaultValue=NAN)
{
    if(obj==NULL)
        return defaultValue;
    if(PyNumber_Check(obj)) {
        double value = PyFloat_AsDouble(obj);
        if(PyErr_Occurred()) {
            PyErr_Clear();
            return defaultValue;
        }
        return value;
    }
    PyObject* d = PyNumber_Float(obj);
    if(d) {
        double value = PyFloat_AsDouble(d);
        Py_DECREF(d);
        return value;
    }
    if(PyErr_Occurred())
        PyErr_Clear();
    return defaultValue;
}

/// return a boolean of a Python object (e.g. 0 if the object is False or is a string "False")
/// the type is int rather than bool, so defaultValue=-1 may be used as an error indicator
int toBool(PyObject* obj, int defaultValue)
{
    if(obj==NULL)
        return defaultValue;
    if(PyString_Check(obj)) {
        try {
            return utils::toBool(PyString_AsString(obj));
        }
        catch(std::exception&) {  // conversion error
            return defaultValue;
        }
    }
    return PyObject_IsTrue(obj);
}

/// a convenience function for accessing an element of a PyArrayObject with the given data type
/// \return  a reference to the element, allowing one to modify it in-place
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

/// same as above, but for a 3d array
template<typename DataType>
inline DataType& pyArrayElem(void* arr, npy_intp ind1, npy_intp ind2, npy_intp ind3)
{
    return *static_cast<DataType*>(PyArray_GETPTR3(static_cast<PyArrayObject*>(arr), ind1, ind2, ind3));
}

/// convert a Python array of floats to std::vector, or return an empty vector in case of error;
/// if the argument is a string instead of a proper array (e.g. if it comes from an ini file),
/// it will be parsed as if it were a python expression, like "numpy.linspace(0.,1.,21)"
std::vector<double> toDoubleArray(PyObject* obj)
{
    if(!obj)
        return std::vector<double>();
    if(PyString_Check(obj)) {  // replace the string with the result of eval("...")
        obj = PyRun_String(PyString_AsString(obj), Py_eval_input, PyEval_GetGlobals(), PyEval_GetLocals());
        if(!obj) {  // exception occurred when parsing and executing the string
            PyErr_Print();
            return std::vector<double>();
        }
    }
    PyObject *arr = PyArray_FROM_OTF(obj, NPY_DOUBLE, 0/*no special requirements*/);
    if(!arr || PyArray_NDIM((PyArrayObject*)arr) != 1) {
        Py_XDECREF(arr);
        if(PyErr_Occurred())
            PyErr_Clear();
        return std::vector<double>();
    }
    int size = PyArray_DIM((PyArrayObject*)arr, 0);
    std::vector<double> vec(size);
    for(int i=0; i<size; i++)
        vec[i] = pyArrayElem<double>(arr, i);
    Py_DECREF(arr);
    return vec;
}

/// convert a C++ vector of double into a NumPy array
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

/// convert a C++ vector of float into a NumPy array
PyObject* toPyArray(const std::vector<float>& vec)
{
    npy_intp size = vec.size();
    PyObject* arr = PyArray_SimpleNew(1, &size, NPY_FLOAT);
    if(!arr)
        return arr;
    for(npy_intp i=0; i<size; i++)
        pyArrayElem<float>(arr, i) = vec[i];
    return arr;
}

/// convert a C++ matrix into a NumPy 2d array
PyObject* toPyArray(const math::IMatrix<double>& mat)
{
    npy_intp size[] = { static_cast<npy_intp>(mat.rows()), static_cast<npy_intp>(mat.cols()) };
    PyObject* arr = PyArray_SimpleNew(2, size, NPY_DOUBLE);
    if(!arr)
        return arr;
    for(size_t i=0; i<mat.rows(); i++)
        for(size_t j=0; j<mat.cols(); j++)
            pyArrayElem<double>(arr, i, j) = mat.at(i, j);
    return arr;
}

/// convert a Python tuple or list into an array of borrowed PyObject* references
std::vector<PyObject*> toPyObjectArray(PyObject* obj)
{
    std::vector<PyObject*> result;
    if(!obj) return result;
    if(PyTuple_Check(obj)) {
        for(Py_ssize_t i=0, size=PyTuple_Size(obj); i<size; i++)
            result.push_back(PyTuple_GET_ITEM(obj, i));
    } else
    if(PyList_Check(obj)) {
        for(Py_ssize_t i=0, size=PyList_Size(obj); i<size; i++)
            result.push_back(PyList_GET_ITEM(obj, i));
    } else
    if(PyArray_Check(obj) && PyArray_TYPE((PyArrayObject*)obj) == NPY_OBJECT &&
        PyArray_NDIM((PyArrayObject*)obj) == 1) {
        for(npy_intp i=0, size=PyArray_DIM((PyArrayObject*)obj, 0); i<size; i++)
            result.push_back(pyArrayElem<PyObject*>(obj, i));
    }
    else
        result.push_back(obj);  // return an array consisting of a single object
    return result;
}

/// check that the list of arguments provided to a Python function
/// contains only named args and no positional args
inline bool onlyNamedArgs(PyObject* args, PyObject* namedArgs)
{
    if((args!=NULL && PyTuple_Check(args) && PyTuple_Size(args)>0) ||
        namedArgs==NULL || !PyDict_Check(namedArgs))
    {
        PyErr_SetString(PyExc_TypeError, "function takes only keyword (not positional) arguments");
        return false;
    }
    return true;
}

/// check that the list of arguments provided to a Python function contains *no* named args
inline bool noNamedArgs(PyObject* namedArgs)
{
    if(namedArgs!=NULL && PyDict_Check(namedArgs) && PyDict_Size(namedArgs)>0)
    {
        PyErr_SetString(PyExc_TypeError, "function takes no keyword arguments");
        return false;
    }
    return true;
}

/** Helper class for parsing named arguments in a case-insensitive way.
    It is created from a Python dictionary, and converts its keys into C++ strings,
    while keeping values as PyObject* pointers (borrowed references), creating a list of arguments.
    One can retrieve ("get") its elements with specific keys (a non-existent argument does not
    raise an error), or "pop" elements, removing them from the list of arguments.
    Once all known named arguments have been retrieved, one can convert the remaining ones
    into KeyValueMap (i.e. convert all values into strings), which clears this list (it remains
    the job of downstream code to verify that this KeyValueMap contains only known keys).
    When the object is destroyed while retaining any argument that have not been parsed,
    this raises a Python exception with the list of unknown named arguments.
*/
class NamedArgs {
    std::vector<std::pair<std::string, PyObject*> > args;
public:
    /// Create the list of elements, checking for duplicate keys
    NamedArgs(PyObject* namedArgs)
    {
        if(!namedArgs)
            return;
        assert(PyDict_Check(namedArgs));
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while(PyDict_Next(namedArgs, &pos, &key, &value)) {
            std::string keystr = toString(key);
            // check that the key is not duplicated (using case-insensitive comparison)
            for(size_t i=0; i<args.size(); i++) 
                if(utils::stringsEqual(args[i].first, keystr)) {
                    PyErr_SetString(PyExc_TypeError,
                        ("Duplicate parameter " + args[i].first + " and " + keystr).c_str());
                    return;
                }
            args.push_back(std::pair<std::string, PyObject*>(keystr, value));
        }
    }

    /// If the object is destroyed while retaining any elements in the list,
    /// this raises a Python TypeError exception, unless there is another exception already raised.
    ~NamedArgs()
    {
        if(PyErr_Occurred())  // preserve any other exception that might have occurred earlier
            return;
        size_t size = args.size();
        if(size==0)
            return;
        std::string msg = size>1 ? "Unknown parameters " : "Unknown parameter ";
        for(size_t i=0; i<size; i++) {
            if(i>0)
                msg += ',';
            msg += args[i].first;
        }
        PyErr_SetString(PyExc_TypeError, msg.c_str());
    }

    bool empty() const { return args.empty(); }

    /// Convert the object into a KeyValueMap and clear all remaining elements from the list
    operator utils::KeyValueMap()
    {
        utils::KeyValueMap params;
        for(size_t i=0; i<args.size(); i++)
            params.set(args[i].first, toString(args[i].second));
        args.clear();
        return params;
    }

    /// Retrieve an argument without removing it from the list
    PyObject* get(const char* key) const
    {
        for(size_t i=0; i<args.size(); i++)
            if(utils::stringsEqual(args[i].first, key))
                return args[i].second;
        return NULL;
    }

    /// Retrieve an argument and remove it from the list
    PyObject* pop(const char* key)
    {
        for(size_t i=0; i<args.size(); i++)
            if(utils::stringsEqual(args[i].first, key)) {
                PyObject* value = args[i].second;
                args.erase(args.begin()+i);
                return value;
            }
        return NULL;
    }

    /// Parse an optional argument that may be a single number or an array of the given length.
    /// If the argument is not found in the list, return an array with a single default value,
    /// whereas if the argument does not follow the expected format, set a Python exception
    /// and return an empty array. The argument is removed from the list if found.
    std::vector<double> popArray(const char* key, npy_intp numPoints, double defaultValue=0)
    {
        PyObject* obj = pop(key);
        std::vector<double> result;
        if(!obj) {
            result.push_back(defaultValue);
            return result;
        }
        // it may be an array or something that could be converted to an array
        if(PySequence_Check(obj) && !PyString_Check(obj)) {
            PyObject *arr = PyArray_FROM_OTF(obj, NPY_DOUBLE, 0/*no special requirements*/);
            if(arr && PyArray_NDIM((PyArrayObject*)arr) == 1) {
                npy_intp size = PyArray_DIM((PyArrayObject*)arr, 0);
                if(size==1 || size==numPoints) {
                    result.resize(size);
                    for(npy_intp i=0; i<size; i++)
                        result[i] = pyArrayElem<double>(arr, i);
                }
            }
            Py_XDECREF(arr);
        } else if(PyNumber_Check(obj)) {
            // it may be a single number
            double value = PyFloat_AsDouble(obj);
            if(!PyErr_Occurred())
                result.push_back(value);
        }
        if(result.empty() && !PyErr_Occurred()) {
            // issue an error message but only if no previous errors already being reported
            PyErr_SetString(PyExc_TypeError,
                ("Argument '" + std::string(key) + "', if provided, "
                "must be a single number or an array of the same length as points").c_str());
        }
        return result;
    }
};

/// convert a C++ exception message to a Python exception
/// (either a KeyboardInterrupt or a generic RuntimeError)
void raisePythonException(const char* message, const char* prefix=NULL)
{
    if(PyErr_Occurred())  // if a Python exception has already been raised, keep it
        return;
    PyErr_SetString(
        message == utils::CtrlBreakHandler::message() ? PyExc_KeyboardInterrupt : PyExc_RuntimeError,
        prefix ? (std::string(prefix) + message).c_str() : message);
}

/// convenience overloads
void raisePythonException(const std::string& message, const char* prefix=NULL)
{
    raisePythonException(message.c_str(), prefix);
}
void raisePythonException(const std::exception& ex, const char* prefix=NULL)
{
    raisePythonException(ex.what(), prefix);
}

/// NumPy data type corresponding to the storage container type of additive models
static const int STORAGE_NUM_T =
    sizeof(galaxymodel::StorageNumT) == sizeof(float)  ? NPY_FLOAT  :
    sizeof(galaxymodel::StorageNumT) == sizeof(double) ? NPY_DOUBLE : NPY_NOTYPE /*shouldn't occur*/;

///@}
//  ------------------------------
/// \name  Unit handling routines
//  ------------------------------
///@{

/// internal working units (arbitrary!)
static const units::InternalUnits unit(2.7183 * units::Kpc, 3.1416 * units::Myr);

/// external units that are used in the calling code, set by the user,
/// (or remaining at default values (no conversion) if not set explicitly
static shared_ptr<const units::ExternalUnits> conv;

/// safeguarding variable: it is set to True upon creation of any non-trivial class,
/// and subsequent calls to setUnits will raise a warning about the dire consequences of
/// changing the unit conversion on the fly
static bool unitsWarning = false;

/// description of setUnits function
static const char* docstringSetUnits =
    "Inform the library about the physical units that are used in Python code.\n"
    "Normally this function should be called only once (if at all) at the beginning of a script, "
    "since changing the unit conversion after creation of instances of Potential, "
    "DistributionFunction and other classes invalidates their input/output.\n"
    "Arguments should be any three independent physical quantities that define "
    "'mass', 'length', 'velocity' or 'time' scales "
    "(note that the latter three are not all independent).\n"
    "Their values specify the units in terms of "
    "'Solar mass', 'Kiloparsec', 'km/s' and 'Megayear', correspondingly.\n"
    "The numerical value of the gravitational constant in these units is stored as agama.G\n"
    "Example: standard GADGET units are defined as\n"
    "    setUnits(mass=1e10, length=1, velocity=1)\n"
    "Calling this function with an empty argument list resets the unit conversion system to "
    "a trivial one (i.e., no conversion involved and all quantities are assumed to be in "
    "N-body units, with the gravitational constant equal to 1).\n"
    "Note that this is NOT equivalent to setUnits(mass=1, length=1, velocity=1).\n";

/// define the unit conversion
PyObject* setUnits(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"mass", "length", "velocity", "time", NULL};
    double mass = 0, length = 0, velocity = 0, time = 0;
    bool reset = true;   // if called without any arguments, this means reset units
    if((args!=NULL && PyTuple_Check(args) && PyTuple_Size(args)>0)) {
        PyErr_SetString(PyExc_TypeError, "setUnits() accepts only named arguments");
        return NULL;
    }
    if(namedArgs!=NULL && PyDict_Check(namedArgs) && PyDict_Size(namedArgs)>0)
        reset = false;
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
    if(mass==0 && !reset) {
        PyErr_SetString(PyExc_TypeError, "You must specify mass unit");
        return NULL;
    }
    const units::ExternalUnits oldconv(*conv);
    if(length>0 && time>0)
        conv.reset(new units::ExternalUnits(unit,
            length*units::Kpc, length/time * units::Kpc/units::Myr, mass*units::Msun));
    else if(length>0 && velocity>0)
        conv.reset(new units::ExternalUnits(unit,
            length*units::Kpc, velocity*units::kms, mass*units::Msun));
    else if(time>0 && velocity>0)
        conv.reset(new units::ExternalUnits(unit,
            velocity*time * units::kms*units::Myr, velocity*units::kms, mass*units::Msun));
    else if(reset)
        conv.reset(new units::ExternalUnits());
    else {
        PyErr_SetString(PyExc_TypeError,
            "You must specify exactly two out of three units: length, time and velocity");
        return NULL;
    }
    if(unitsWarning && ( // check if units have changed
        math::fcmp(oldconv.  lengthUnit, conv->  lengthUnit)!=0 ||
        math::fcmp(oldconv.velocityUnit, conv->velocityUnit)!=0 ||
        math::fcmp(oldconv.    massUnit, conv->    massUnit)!=0 ||
        math::fcmp(oldconv.    timeUnit, conv->    timeUnit)!=0) )
        PyErr_WarnEx(NULL, "setUnits() called after creating instances of Potential and other "
            "classes may lead to incorrect scaling of input/output data in their methods", 1);
    double G = reset ? 1.0 : units::Grav *
        (conv->massUnit * unit.to_Msun * units::Msun) /
        pow_2(conv->velocityUnit * unit.to_kms * units::kms) /
        (conv->lengthUnit * unit.to_Kpc * units::Kpc);
    // store the numerical value of G as a module attribute (update the existing constant)
    PyObject* PyG = PyDict_GetItemString(PyModule_GetDict(thismodule), "G");
    if(PyG && PyFloat_CheckExact(PyG))
        // hack: directly manipulate the member variable in the internal representation
        // of the Python float object. This goes against the idea of immutability of numbers,
        // but ensures that the content of this variable is updated synchronously everywhere.
        // The alternative is to create a _new_ module-level variable with the same name and
        // a new value, which replaces the old one. Unfortunately, this only works if agama.so
        // is imported directly, not when its classes, methods and variables are imported into
        // another module (as happens in the __init__.py file): in the latter case,
        // the previously imported variable and the newly created one become different entities
        // (get out of sync), and the first, user-facing one is not updated.
        ((PyFloatObject*)PyG)->ob_fval = G;
    else
        PyErr_WarnEx(NULL, "agama.G has wrong type and its value cannot be updated", 1);
    FILTERMSG(utils::VL_DEBUG, "Agama",   // internal unit conversion factors not for public eye
        "length unit: "  +utils::toString(conv->lengthUnit)+", "
        "velocity unit: "+utils::toString(conv->velocityUnit)+", "
        "time unit: "    +utils::toString(conv->timeUnit)+", "
        "mass unit: "    +utils::toString(conv->massUnit));
    Py_INCREF(Py_None);
    return Py_None;
}

/// description of getUnits function
static const char* docstringGetUnits =
    "Retrieve the current unit conversion settings initialized by setUnits(...).\n"
    "No arguments.\n"
    "Returns a dictionary with four elements, listing the current dimensional units: "
    "length (in kpc), velocity (in km/s), time (in Myr) and mass (in Msun).\n"
    "If the unit conversion was not initialized or was reset to a trivial one by calling "
    "setUnits() without arguments, this function return an empty dictionary.\n";

/// retrieve the current unit conversion settings
PyObject* getUnits(PyObject* /*self*/, PyObject* /*args*/)
{
    PyObject* result = PyDict_New();
    if(conv->lengthUnit == 1 && conv->velocityUnit == 1 && conv->timeUnit == 1 && conv->massUnit == 1)
        return result;   // no unit conversion was set up
    // otherwise populate the dictionary with elements
    PyObject* length   = PyFloat_FromDouble(conv->lengthUnit   * unit.to_Kpc);
    PyObject* velocity = PyFloat_FromDouble(conv->velocityUnit * unit.to_kms);
    PyObject* time     = PyFloat_FromDouble(conv->timeUnit     * unit.to_Myr);
    PyObject* mass     = PyFloat_FromDouble(conv->massUnit     * unit.to_Msun);
    PyDict_SetItemString(result, "length",   length);
    PyDict_SetItemString(result, "velocity", velocity);
    PyDict_SetItemString(result, "time",     time);
    PyDict_SetItemString(result, "mass",     mass);
    Py_DECREF(length);
    Py_DECREF(velocity);
    Py_DECREF(time);
    Py_DECREF(mass);
    return result;
}

/// helper function for converting position to internal units
inline coord::PosCar convertPos(const double input[]) {
    return coord::PosCar(
        input[0] * conv->lengthUnit,
        input[1] * conv->lengthUnit,
        input[2] * conv->lengthUnit);
}

/// helper function for converting velocity to internal units
inline coord::VelCar convertVel(const double input[]) {
    return coord::VelCar(
        input[0] * conv->velocityUnit,
        input[1] * conv->velocityUnit,
        input[2] * conv->velocityUnit);
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

void convertParticlesStep1(PyObject* particles_obj,
    /*output - create new arrays*/ PyArrayObject* &coord_arr, PyArrayObject* &mass_arr)
{
    // parse the input arrays
    static const char* errorstr = "'particles' must be a tuple with two arrays - "
        "coordinates[+velocities] and mass, where the first one is a two-dimensional "
        "Nx3 or Nx6 array and the second one is a one-dimensional array of length N";
    PyObject *coord_obj, *mass_obj;
    if(!PyArg_ParseTuple(particles_obj, "OO", &coord_obj, &mass_obj))
        throw std::invalid_argument(errorstr);
    coord_arr = (PyArrayObject*)PyArray_FROM_OTF(coord_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    mass_arr  = (PyArrayObject*)PyArray_FROM_OTF(mass_obj,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    npy_intp nbody = 0;
    if( coord_arr == NULL || mass_arr == NULL ||      // input should contain valid arrays
        PyArray_NDIM(mass_arr) != 1 ||                // the second one should be 1d array
        (nbody = PyArray_DIM(mass_arr, 0)) <= 0 ||    // of length nbody > 0
        PyArray_NDIM(coord_arr) != 2 ||               // the first one should be a 2d array
        PyArray_DIM(coord_arr, 0) != nbody ||         // with nbody rows
       (PyArray_DIM(coord_arr, 1) != 3 && PyArray_DIM(coord_arr, 1) != 6))  // and 3 or 6 columns
    {
        Py_XDECREF(coord_arr);
        Py_XDECREF(mass_arr);
        throw std::invalid_argument(errorstr);
    }
}

template<typename ParticleT>
particles::ParticleArray<ParticleT> convertParticlesStep2(
    /*input arrays, which will be disposed*/ PyArrayObject* coord_arr, PyArrayObject* mass_arr)
{
    npy_intp nbody = PyArray_DIM(coord_arr, 0);
    bool haveVel   = PyArray_DIM(coord_arr, 1) == 6;  // whether we have velocity data
    particles::ParticleArray<ParticleT> result;
    result.data.reserve(nbody);
    for(npy_intp i=0; i<nbody; i++) {
        const double *xv = &pyArrayElem<double>(coord_arr, i, 0);
        result.add(coord::PosVelCar(
            xv[0] * conv->lengthUnit, xv[1] * conv->lengthUnit, xv[2] * conv->lengthUnit,
            haveVel? xv[3] * conv->velocityUnit : 0.,
            haveVel? xv[4] * conv->velocityUnit : 0.,
            haveVel? xv[5] * conv->velocityUnit : 0.),
            pyArrayElem<double>(mass_arr, i) * conv->massUnit);
    }
    Py_DECREF(coord_arr);
    Py_DECREF(mass_arr);
    return result;
}

/// convert a tuple of two arrays (particle coordinates and possibly velocities, and particle masses)
/// into an equivalent C++ object with appropriate units
template<typename ParticleT>
particles::ParticleArray<ParticleT> convertParticles(PyObject* particles_obj)
{
    PyArrayObject *coord_arr, *mass_arr;
    convertParticlesStep1(particles_obj,   /*create arrays*/ coord_arr, mass_arr);
    return convertParticlesStep2<ParticleT>(/*consume them*/ coord_arr, mass_arr);
}

///@}
//  --------------------------------------------------------------
/// \name  A truly general interface for evaluating some function
///        for some input data and storing its output somewhere
//  --------------------------------------------------------------
///@{

/** Interface for a general function that computes something for one or many input points.
    Each input point is either one number or an array of M numbers, and the number of points N
    may be one or more; hence the input object may be scalar, 1d or 2d array.
    Output consists of one or more elements, with each element itself being a scalar,
    1d, 2d or 3d array. In case of one element, it is returned directly, otherwise the output
    is a tuple of several elements. If the input is an array of points rather than a single point,
    the topmost dimension of output arrays is the number of points (it may even be 1, but an extra
    dimension is still added).

    Example 1: computation of force and its derivative.
    Input is 3 numbers (coordinates), output is 3 force components and 6 force derivatives.
    If the input point is an array of length 3 (i.e. a single point), the output is a tuple of
    two arrays (length 3 and length 6); if the input is an array of shape Nx3 (with N>=1),
    the output is a tuple of two arrays of shape (Nx3) and (Nx6).

    Example 2: computation of DF moments.
    Input is again 2 or 3 numbers, output is one, two or three arrays (depending on input flags).
    The density array has 1 value per input point, the array of second moments of velocity -
    6 values per point.
    Moreover, if the DF has C>=1 components and the flag 'separate' is set to True, each DF component
    produces a separate output value; hence for N input points, the output density array will have
    shape (NxC), and the array of second moments of velocity - shape (NxCx6).

    All these situations are handled by classes derived from this interface.
    The constructor of the base class BatchFunction takes care of analyzing the input, determining
    whether this is one or several points, and ensures  that each point has a correct length M.
    Constructors of derived classes, in turn, allocate the output objects or a tuple of objects 
    of appropriate shape. They typically invoke auxiliary routines allocateOutput<> defined below.
    The run() method of the base class loops over input points and calls a virtual method
    processPoint() for each of them, which must be implemented in derived classes: it carries out
    the actual computations, and stores the results in appropriate places in the output array(s).
    This loop is either performed serially or OpenMP-parallelized if the number of input points
    is above the given threshold.
    The usage scenario is quite rigid: construct an instance of a class derived from BatchFunction,
    and then call its run() method, which returns a Python object containing the results.
    In case of errors during construction (parsing the input or allocating the output),
    no work is done and run() returns NULL, setting a Python exception.
*/
class BatchFunction {
protected:
    double inputPointScalar;    // a single scalar input point or
    PyArrayObject* inputArray;  // a temp.array for all other cases (only one of these two is used)
    const double* inputBuffer;  // pointer to the temp.variable or to the temp.array
    int inputLength;            // dimension of a single input point
    npy_intp numPoints;         // number of input points: -1 means a single point, -2 is error
    PyObject* outputObject;     // the Python object returned by the run() method;
                                // it must be initialized by constructors of derived classes
    volatile npy_intp numCompleted;  // the number of already completed points
    ProgressBar progressBar;         // helper object for displaying progress indication
public:
    /** Constructor of the base class only analyzes the input object, determines the number
        of input points and ensures that the length of each point equals inputLength.
        \param[in]  inputObject  is the Python object (scalar, tuple, list or array)
        containing one or more points.
        \param[in]  inputLength1  is the required length of each input point (M).
        \param[in]  inputLength2  is the alternative required length, in cases that the function
        can operate in two different regimes depending on the point size;
        if not set, assuming that only one value of length is acceptable.
        \param[in]  errorMessage  is an optional customized error message shown in case of
        invalid input dimensions; if not provided, a standard message is shown.
        When the inputObject is parsed successfully, the member variable inputBuffer points to the
        raw buffer containing the single scalar point or the first element of the input array;
        numPoints contains either the number of input points (N), or -1 indicating a single point
        (this is different from an input array containing one point, because the dimensionality
        of output arrays is increased for an input array).
        Otherwise (when the inputObject could not be parsed) a Python exception is raised,
        and inputBuffer contains NULL.
        Constructors of derived classes must additionally allocate the outputObject member variable
        (unless numPoints<-1, indicating an error in parsing the input).
    */
    BatchFunction(PyObject* inputObject, int inputLength1, int inputLength2=0,
        const char* errorMessage=NULL)
    :
        inputPointScalar(NAN),
        inputArray(NULL),
        inputBuffer(NULL),
        inputLength(inputLength1),
        numPoints(-2),
        outputObject(NULL),
        numCompleted(0),
        progressBar(0, "point")
    {
        if(inputObject == NULL) {
            PyErr_SetString(PyExc_TypeError, "No input data provided");
            return;
        }
        if(inputLength2 == 0)
            inputLength2 = inputLength1;

        if(PyTuple_Check(inputObject)) {
            switch(PyTuple_Size(inputObject)) {
                case 1:
                    // inputObject is usually (but not always) the entire tuple of function arguments;
                    // if it only has one element, use this element and not the whole tuple
                    inputObject = PyTuple_GET_ITEM(inputObject, 0);
                    break;
                case 0:
                    // this happens when the function is called without arguments (not allowed);
                    // however, an input array of zero length is allowed and produces zero-length output
                    PyErr_SetString(PyExc_TypeError, "No input data provided");
                    return;
                default: ;
            }
        }

        // check if the input is a single number, if yest then just take it literally
        if(std::min(inputLength1, inputLength2) == 1 && PyNumber_Check(inputObject)) {
            inputPointScalar = PyFloat_AsDouble(inputObject);
            if(!PyErr_Occurred()) {  // successful
                inputLength = 1;
                inputBuffer = &inputPointScalar;
                numPoints = -1;   // means a single input point and not an array of length 1
                return;
            }
            PyErr_Clear();   // clear any conversion errors and try the more general approach
        }

        // otherwise convert the input to an array and analyze its shape
        inputArray = (PyArrayObject*) PyArray_FROM_OTF(inputObject,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if(inputArray == NULL) {
            PyErr_SetString(PyExc_TypeError, "Input does not contain a valid array");
            return;
        }

        if(PyArray_NDIM(inputArray) == 1 &&
           (PyArray_DIM(inputArray, 0) == inputLength1 ||
            PyArray_DIM(inputArray, 0) == inputLength2 ))
        {   // 1d array of size inputLength means a single point and is indicated by numPoints=-1,
            // to distinguish it from a 2d array of size 1*inputLength
            numPoints   = -1;
            inputLength = PyArray_DIM(inputArray, 0);
        } else if(PyArray_NDIM(inputArray) == 1 && std::min(inputLength1, inputLength2) == 1)
        {   // 1d array of points with inputLength=1
            numPoints   = PyArray_DIM(inputArray, 0);
            inputLength = 1;
        } else if(PyArray_NDIM(inputArray) == 2 &&
             (PyArray_DIM(inputArray, 1) == inputLength1 ||
              PyArray_DIM(inputArray, 1) == inputLength2 ))
        {   // 2d array of input points, shape numPoints * inputLength
            numPoints   = PyArray_DIM(inputArray, 0);
            inputLength = PyArray_DIM(inputArray, 1);
        } else {
            // invalid size of input arrays: show a standard or a customized error message
            if(errorMessage) {
                PyErr_SetString(PyExc_TypeError, errorMessage);
            } else {
                std::string error;
                if(inputLength1 == 1 && inputLength2 == 1) {
                    error = "Input does not contain valid data "
                        "(either a single number or a one-dimensional array)";
                } else {
                    std::string inp = utils::toString(inputLength1);
                    if(inputLength1 != inputLength2)
                        inp += "/" + utils::toString(inputLength2);
                    error = "Input does not contain valid data (either " + inp +
                        " numbers for a single point or a Nx" + inp + " array)";
                }
                PyErr_SetString(PyExc_TypeError, error.c_str());
            }
            return;
        }
        // reassign the raw input data buffer to the temporary array
        inputBuffer = static_cast<double*>(PyArray_DATA(inputArray));

        // set the total number of points for the progress bar (once it became known)
        progressBar.numTotal = numPoints;
    }

    // deallocate the temporary buffer containing the input array
    virtual ~BatchFunction()
    {
        Py_XDECREF(inputArray);
    }

    /** The routine performing actual work for each input point, must be implemented in derived
        classes. It may be called in parallel from multiple threads.
        \param[in]  pointIndex  is the index of input point (>=0);
        the actual input data are M floating-point values located in the raw input buffer,
        starting at inputBuffer[pointIndex*M].
        The function must store the results somewhere in outputObject, and may not throw exceptions.
    */
    virtual void processPoint(npy_intp pointIndex) = 0;

    /** Vectorized form of this function, which is the one actually used by the run() method.
        The default implementation simply loops over points one by one,
        but derived classes may reimplement it more efficiently 
        by delegating the vectorized computation to the underlying code.
    */
    virtual void processManyPoints(npy_intp beginIndex, npy_intp endIndex)
    {
        for(npy_intp pointIndex=beginIndex; pointIndex<endIndex; pointIndex++)
            processPoint(pointIndex);
    }

    /** The driver routine that loops over the input array and calls processPoint() for each item.
        \param[in] chunk  determines the OpenMP parallelization strategy:
        If chunk==0 or the number of points N is less than |chunk|, no threads are created at all;
        otherwise OpenMP-parallelize the loop, either with a static or dynamic scheduling.
        If chunk<0, use static workload distribution, meaning that each thread gets an equal number of
        points (N/numthreads) -- this is suitable when the cost of each point is approximately equal.
        If chunk>0, use dynamic scheduling: each thread gets a batch of 'chunk' points at a time, and
        when finished crunching the current batch, it retrieves the next one from the queue --
        this is more effective when the computational cost may vary widely between points, and hence
        some threads may finish their work earlier and start the next chunk immediately.
        However, it involves a larger scheduling overhead, so the chunks shouldn't be too small
        (depending on the cost of processing a single point: if it is high, chunk=1 is best choice).
        \return  the Python object (outputObject) containing the results, or NULL in case of errors
        during initialization (indicated by outputObject==NULL).
        If the user triggers a keyboard interrupt during computations, the loop is terminated and
        NULL is returned as well.
        \note OpenMP-parallelized loop is executed after releasing GIL, and the processPoint()
        methods of derived classes need to acquire GIL if necessary (even if the loop is not
        parallelized, this does not hurt).
    */
    PyObject* run(int chunk)
    {
        if(outputObject == NULL) {  // indicates an error during construction:
            // a Python exception should have been set, but in case it wasn't, make sure we set up one
            if(!PyErr_Occurred())
                PyErr_SetString(PyExc_RuntimeError, "Failed to create output array");
            return NULL;
        }

        if(numPoints == 0) {
            // nothing to do
        }
        else if(numPoints == 1 || numPoints == -1) {
            // fast-track for a single input point (numPoints==-1) or an array of length one
            try{
                processPoint(0);
            }
            catch(std::exception& ex) {
                Py_DECREF(outputObject);
                outputObject = NULL;
                raisePythonException(ex);
            }
        } else {
            std::string errorMessage;      // store the exception that may occur in processPoint()
            utils::CtrlBreakHandler cbrk;  // catch Ctrl-Break keypress
#ifdef _OPENMP
            if(chunk==0 || numPoints <= abs(chunk))
#else
            (void)chunk;  // remove warning about unused parameter
            if(true)
#endif
            {
                try{
                    // no parallelization if the number of points is too small (e.g., just one)
                    processManyPoints(0, numPoints);
                }
                catch(std::exception& ex)
                {
                    errorMessage = ex.what();
                }
            }
#ifdef _OPENMP
            else {
                // OpenMP-parallelized loops need to run GIL-free;
                // any code in the processPoint() method that uses Python C API
                // will need to re-acquire GIL for its own thread (via PyAcquireGIL)
                PyReleaseGIL unlock;
                bool stop = false;   // the loop is terminated once this flag is raised
                npy_intp blockSize = abs(chunk), numBlocks = (numPoints-1) / blockSize + 1;

                // parallel loop over input array
                if(chunk < 0) {
                    // pre-determined split of work across threads
#pragma omp parallel for schedule(static)
                    for(npy_intp indBlock=0; indBlock<numBlocks; indBlock++) {
                        if(cbrk.triggered() || stop) continue;
                        try{
                            npy_intp start = indBlock * blockSize;
                            npy_intp count = std::min(blockSize, numPoints - start);
                            processManyPoints(start, start + count);
#pragma omp atomic
                            numCompleted += count;
                            progressBar.update(numCompleted);
                        }
                        catch(std::exception& ex)
                        {
                            errorMessage = ex.what();
                            stop = true;
                        }
                    }
                } else /*chunk > 0*/ {
                    // dynamical load balancing - the code below is identical but the pragma is different
#pragma omp parallel for schedule(dynamic, 1)
                    for(npy_intp indBlock=0; indBlock<numBlocks; indBlock++) {
                        if(cbrk.triggered() || stop) continue;
                        try{
                            npy_intp start = indBlock * blockSize;
                            npy_intp count = std::min(blockSize, numPoints - start);
                            processManyPoints(start, start + count);
#pragma omp atomic
                            numCompleted += count;
                            progressBar.update(numCompleted);
                        }
                        catch(std::exception& ex)
                        {
                            errorMessage = ex.what();
                            stop = true;
                        }
                    }
                }
            }
#endif
            progressBar.clear();
            // check for any exceptional circumstances
            if(cbrk.triggered())
                errorMessage = cbrk.message();
            if(!errorMessage.empty()) {
                Py_DECREF(outputObject);
                outputObject = NULL;
                raisePythonException(errorMessage.c_str());
            }
        }
        return outputObject;
    }
};

/** Vectorized form of the above class, whose descendants natively implement
    the vectorized method processManyPoints().
*/
class BatchFunctionVectorized: public BatchFunction {
public:
    BatchFunctionVectorized(PyObject* inputObject, int inputLength1, int inputLength2=0) :
        BatchFunction(inputObject, inputLength1, inputLength2) {}

    // implement single-point call via vectorized call
    virtual void processPoint(npy_intp pointIndex) {
        processManyPoints(pointIndex, pointIndex+1);
    }

    // this one needs to be actually implemented in derived classes
    virtual void processManyPoints(npy_intp beginIndex, npy_intp endIndex) = 0;
};

/** Helper routine for allocating an output array for a class derived from BatchFunction.
    \tparam  DIM>=1  is the length of each element of the output array:
    if DIM>1, the last dimension of the output array will have length DIM,
    otherwise the array has one dimension fewer.
    \param[in]  numPoints is the first dimension of the output array (the number of input points N);
    if numPoints==-1, this means a single input point, and the output array has one dimension fewer.
    \param[out]  buffer  is the pointer to a pointer (double*), which will contain the raw buffer
    of the output object (a Python float or a NumPy array); the data should be stored in this buffer.
    \param[in]  C (optional) length of the intermediate dimension of the output array
    (if 0, this intermediate dimension is not created).
    \return  the Python object where the data will be stored (using the raw buffer),
    or NULL if failed to allocate an array.
    Examples:
    DIM=1, N=0, C=0  =>  output is a scalar (Python float);
    DIM=1, N>0, C=0  =>  output is a 1d array of length N;
    DIM>1, N=0, C=0  =>  output is a 1d array of length DIM;
    DIM>1, N>0, C=0  =>  output is a 2d array of shape (N,DIM);
    DIM>1, N>0, C>0  =>  output is a 3d array of shape (N,C,DIM).
*/
template<int DIM>
PyObject* allocateOutput(npy_intp numPoints, double* buffer[1]=NULL, int C=0)
{
    if(numPoints < -1)
        return NULL;
    PyObject* output = NULL;
    if(C==0) {
        if(numPoints == -1 && DIM == 1) {
            // output is a single float
            output = PyFloat_FromDouble(NAN);   // allocate a python float with some initial value
            if(buffer)  // get the pointer to the raw float value, which will be modified later
                buffer[0] = &((PyFloatObject*)output)->ob_fval;
            return output;
        }
        // otherwise output is a 1d or a 2d array
        npy_intp dims[] = {numPoints, DIM};
        if(numPoints == -1)  // output is a single row
            output = PyArray_SimpleNew(1, &dims[1], NPY_DOUBLE);
        else
            output = PyArray_SimpleNew(DIM == 1 ? 1 : 2, dims, NPY_DOUBLE);
    } else {  // add an intermediate dimension C
        npy_intp dims[] = {numPoints, C, DIM};
        if(numPoints == -1)
            output = PyArray_SimpleNew(DIM == 1 ? 1 : 2, &dims[1], NPY_DOUBLE);
        else
            output = PyArray_SimpleNew(DIM == 1 ? 2 : 3, &dims[0], NPY_DOUBLE);
    }
    if(output && buffer)  // output might be NULL if failed to allocate
        buffer[0] = static_cast<double*>(PyArray_DATA((PyArrayObject*)output));  // raw buffer
    return output;
}

/** Helper routine for allocating a tuple of two output arrays for a class derived from BatchFunction.
    \tparam DIM1 - last dimension of the first output array.
    \tparam DIM2 - last dimension of the second output array.
    Other parameters have the same meaning as for the previous function,
    except that buffer[] should be a pointer to an array of two pointers (double*), 
    which will contain raw buffers of both elements of the tuple.
*/
template<int DIM1, int DIM2>
PyObject* allocateOutput(npy_intp numPoints, double* buffer[2]=NULL, int C=0)
{
    PyObject *elem1 = allocateOutput<DIM1>(numPoints, buffer, C);
    PyObject *elem2 = allocateOutput<DIM2>(numPoints, buffer? &buffer[1] : NULL, C);
    if(elem1 && elem2)
        return Py_BuildValue("NN", elem1, elem2);
    else {
        Py_XDECREF(elem1);
        Py_XDECREF(elem2);
        return NULL;
    }
}

/** Helper routine for allocating a tuple of three output arrays for a class derived from BatchFunction.
    \tparam DIM1 - last dimension of the first output array;
    \tparam DIM2 - last dimension of the second output array.
    \tparam DIM3 - last dimension of the third output array.
    Other parameters have the same meaning as for the previous function,
    except that buffer should have 3 elements.
*/
template<int DIM1, int DIM2, int DIM3>
PyObject* allocateOutput(npy_intp numPoints, double* buffer[3]=NULL, int C=0)
{
    PyObject *elem1 = allocateOutput<DIM1>(numPoints, buffer, C);
    PyObject *elem2 = allocateOutput<DIM2>(numPoints, buffer? &buffer[1] : NULL, C);
    PyObject *elem3 = allocateOutput<DIM3>(numPoints, buffer? &buffer[2] : NULL, C);
    if(elem1 && elem2 && elem3)
        return Py_BuildValue("NNN", elem1, elem2, elem3);
    else {
        Py_XDECREF(elem1);
        Py_XDECREF(elem2);
        Py_XDECREF(elem3);
        return NULL;
    }
}


///@}
//  ---------------------
/// \name  Density class
//  ---------------------
///@{

/// common fragment of docstring for Density and Potential classes
#define DOCSTRING_DENSITY_PARAMS \
    "  mass=...   total mass of the model, if applicable.\n" \
    "  scaleRadius=...   scale radius of the model (if applicable).\n" \
    "  scaleHeight=...   scale height of the model (MiyamotoNagai, LongMurali and Disk models).\n" \
    "  p=...   or  axisRatioY=...   axis ratio y/x, i.e., intermediate to long axis " \
    "(applicable to triaxial potential models such as Dehnen and Ferrers, " \
    "and to Spheroid, Nuker or Sersic density models).\n" \
    "  q=...   or  axisRatioZ=...   short to long axis (z/x) (applicable to the same model types " \
    "as above plus the axisymmetric PerfectEllipsoid).\n" \
    "  gamma=...  central cusp slope (applicable for Dehnen, Spheroid or Nuker).\n" \
    "  beta=...   outer density slope (Spheroid or Nuker).\n" \
    "  alpha=...  strength of transition from the inner to the outer slopes (Spheroid or Nuker).\n" \
    "  sersicIndex=...   profile shape parameter 'n' (Sersic or Disk).\n" \
    "  barLength=...  length of the bar (LongMurali).\n" \
    "  innerCutoffRadius=...   radius of inner hole (Disk).\n" \
    "  outerCutoffRadius=...   radius of outer exponential cutoff (Spheroid).\n" \
    "  cutoffStrength=...   strength of outer exponential cutoff  (Spheroid).\n" \
    "  surfaceDensity=...   surface density normalization " \
    "(Disk or Sersic - in the center, Nuker - at scaleRadius).\n" \
    "  densityNorm=...   normalization of density profile (Spheroid).\n" \
    "  W0=...  dimensionless central potential in King models.\n" \
    "  trunc=...  truncation strength in King models.\n" \
    "  center=...  offset of the potential from origin, can be either " \
    "a triplet of numbers, or an array of time-dependent offsets " \
    "(t,x,y,z, and optionally vx,vy,vz) provided directly or as a file name.\n" \
    "  orientation=...  orientation of the principal axes of the model w.r.t. " \
    "the external coordinate system, specified as a triplet of Euler angles.\n" \
    "  rotation=...  angle of rotation of the model about its z axis, " \
    "can be a single number or an array / file with a time-dependent angle.\n" \
    "  scale=...  modification of mass and size scales of the model, " \
    "given either as two numbers or an array / file with time-dependent scaling factors.\n"

/// description of Density class
static const char* docstringDensity =
    "Density is a class representing a variety of density profiles "
    "that do not necessarily have a corresponding potential defined.\n"
    "An instance of Density class is constructed using the following keyword arguments:\n"
    "  type='...' or density='...'   the name of density profile (required), "
    "can be one of the following:\n"
    "    Denhen, Plummer, PerfectEllipsoid, Ferrers, MiyamotoNagai, LongMurali, NFW, "
    "Disk, Spheroid, Nuker, Sersic, King.\n"
    DOCSTRING_DENSITY_PARAMS
    "Most of these parameters have reasonable default values.\n"
    "Alternatively, one may construct a spherically-symmetric density model from a cumulative "
    "mass profile by providing a single argument\n"
    "  cumulmass=...  which should contain a table with two columns: radius and enclosed mass, "
    "both strictly positive and monotonically increasing.\n"
    "One may also load density expansion coefficients that were previously written to a text file "
    "using the `export()` method, by providing the file name as a single unnamed argument. "
    "Such a file may contain any number of density components, each one in a separate "
    "INI sections whose name starts with Density, e.g., [Density], [Density1], ...\n"
    "Finally, one may create a composite density from several Density objects by providing them as "
    "unnamed arguments to the constructor:  densum = Density(den1, den2, den3)\n\n"
    "An instance of Potential class may be used in all contexts when a Density object is required.\n"
    "One may create a modified version of an existing Density object, by passing it as a `density` "
    "argument together with one or more modifier parameters (center, orientation, rotation and scale).\n"
    "One may also construct a Density wrapper around a user-defined Python function that takes "
    "an array of Nx3 points in cartesian coordinates and returns the density values at these N points. "
    "Such a function can be provided as a single positional argument or as a `density` argument, "
    "followed by `symmetry` (if the latter is not given, the symmetry remains unknown "
    "and some methods will not work).\n"
    "If this user-defined function is expected to be used in heavy computations, "
    "it may be more efficient to approximate it by a native density interpolator class: "
    "DensitySphericalHarmonic (for spheroidal profiles that are not too flattened) or "
    "DensityAzimuthalHarmonic (for possibly strongly flattened models with a finite central density). "
    "In this case, the `type` argument must specify either of these two expansion types, "
    "the `density` argument should contain the user-defined function, "
    "the `symmetry` argument describes its degree of symmetry, "
    "and optional additional arguments may specify the grid parameters "
    "(rmin, rmax, gridSizeR; for azimuthal-harmonic expansion - additionally zmin, zmax, gridSizeZ) "
    "and angular expansion orders (lmax, mmax for spherical, or mmax for azimuthal harmonics).\n"
    "Examples:\n\n"
    ">>> dens_disk = Density(type='Disk', scaleRadius=3, scaleHeight=-0.3, sersicIndex=1.5)\n"
    ">>> dens_param = dict(type='Spheroid', scaleradius=2, axisratioZ=0.5, gamma=0, alpha=2)\n"
    ">>> dens_composide = Density(dens_disk, dens_param)\n"
    ">>> dens_shifted = Density(density=dens_composite, center=[1,2,3])\n"
    ">>> dens_func = lambda xyz: 1/(numpy.sum(xyz**4, axis=1) + 1)\n"
    ">>> dens_user = Density(dens_func, symmetry='t')\n"
    ">>> dens_appr = Density(type='DensitySphericalHarmonic', density=dens_func, symmetry='t')\n";

/// \cond INTERNAL_DOCS
/// Python type corresponding to Density class
typedef struct {
    PyObject_HEAD
    potential::PtrDensity dens;
} DensityObject;

/// Python type corresponding to Potential class, which is inherited from Density
typedef struct {
    PyObject_HEAD
    // note that the memory layout of both Density and Potential python classes are the same,
    // but the sole member variable (a smart pointer) has different (but compatible) base types,
    // allowing it to be used in class methods shared by both classes
    potential::PtrPotential pot;
} PotentialObject;
/// \endcond

/// Helper class for providing a BaseDensity interface
/// to a Python function that returns density at one or several point
class DensityWrapper: public potential::BaseDensity{
    PyObject* fnc;
    const coord::SymmetryType sym;
    const std::string fncname;
public:
    DensityWrapper(PyObject* _fnc, coord::SymmetryType _sym):
        fnc(_fnc), sym(_sym), fncname(toString(fnc))
    {
        Py_INCREF(fnc);
        FILTERMSG(utils::VL_DEBUG, "Agama",
            "Created a C++ density wrapper for Python function " + fncname +
            " (symmetry: " + potential::getSymmetryNameByType(sym) + ")");
        if(isUnknown(sym))
            PyErr_WarnEx(NULL, "symmetry is not provided, some methods will not be available", 1);
    }
    ~DensityWrapper()
    {
        FILTERMSG(utils::VL_DEBUG, "Agama",
            "Deleted a C++ density wrapper for Python function " + fncname);
        Py_DECREF(fnc);
    }
    virtual coord::SymmetryType symmetry() const { return sym; }
    virtual std::string name() const { return fncname; };
    // first come the evaluation functions for a single input point in all coordinate systems
    virtual double densityCyl(const coord::PosCyl &pos, double time) const {
        return densityCar(toPosCar(pos), time); }
    virtual double densitySph(const coord::PosSph &pos, double time) const {
        return densityCar(toPosCar(pos), time); }
    virtual double densityCar(const coord::PosCar &pos, double time) const {
        double result;
        evalmanyDensityCar(1, &pos, &result, time);  // call the vectorized function for one input point
        return result;
    }
    // next come vectorized evaluation functions in the 'non-native' coordinate systems
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time) const
    {
        if(npoints==1) {  // fast track
            coord::PosCar poscar = toPosCar(pos[0]);
            evalmanyDensityCar(1, &poscar, values, time);
        } else {
            ALLOC(npoints, coord::PosCar, poscar)
            for(size_t i=0; i<npoints; i++)
                poscar[i] = toPosCar(pos[i]);
            evalmanyDensityCar(npoints, &poscar[0], values, time);  // the actual evaluation function
        }
    }
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time) const
    {
        if(npoints==1) {  // fast track
            coord::PosCar poscar = toPosCar(pos[0]);
            evalmanyDensityCar(1, &poscar, values, time);
        } else {
            ALLOC(npoints, coord::PosCar, poscar)
            for(size_t i=0; i<npoints; i++)
                poscar[i] = toPosCar(pos[i]);
            evalmanyDensityCar(npoints, &poscar[0], values, time);  // the actual evaluation function
        }
    }
    // and finally here is the actual vectorized evaluation function in cartesian coordinates
    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double /*time*/) const
    {
        ALLOC(3*npoints, double, xyz)
        for(size_t p=0; p<npoints; p++)
            unconvertPos(pos[p], xyz + p*3);
        double mult = conv->massUnit / pow_3(conv->lengthUnit);
        PyAcquireGIL lock;
        bool typeerror   = false;
        npy_intp dims[]  = { (npy_intp)npoints, 3};
        PyObject* args   = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, xyz);
        PyObject* result = PyObject_CallFunctionObjArgs(fnc, args, NULL);
        Py_DECREF(args);
        if(result == NULL) {
            PyErr_Print();
        } else if(PyArray_Check(result) &&
            PyArray_NDIM((PyArrayObject*)result) == 1 &&
            PyArray_DIM ((PyArrayObject*)result, 0) == (npy_intp)npoints)
        {
            int type = PyArray_TYPE((PyArrayObject*) result);
            for(size_t p=0; p<npoints; p++) {
                switch(type) {
                    case NPY_DOUBLE: values[p] = pyArrayElem<double>(result, p) * mult; break;
                    case NPY_FLOAT:  values[p] = pyArrayElem<float >(result, p) * mult; break;
                    default: values[p] = NAN; typeerror = true;
                }
            }
        }
        else if(npoints==1 && PyNumber_Check(result)) {
            // in case of a single input point, the user function might return a single number
            values[0] = PyFloat_AsDouble(result) * mult;
        }
        else {
            typeerror = true;
        }
        Py_XDECREF(result);
        if(result == NULL)
            throw std::runtime_error("Call to user-defined density function failed");
        else if(typeerror)
            throw std::runtime_error("Invalid data type returned by user-defined density function");
    }
};

/// destructor of the Density class
void Density_dealloc(DensityObject* self)
{
    if(self->dens)
        FILTERMSG(utils::VL_DEBUG, "Agama", "Deleted " + std::string(self->dens->name()) +
            " density at " + utils::toString(self->dens.get()));
    else
        FILTERMSG(utils::VL_DEBUG, "Agama", "Deleted an empty density");
    self->dens.reset();
    Py_TYPE(self)->tp_free(self);
}

/// extract a pointer to C++ Density class from a Python object, or return an empty pointer on error
// (forward declaration, the function will be defined later)
potential::PtrDensity getDensity(PyObject* dens_obj, utils::KeyValueMap* params = NULL);

// extract a pointer to C++ Potential class from a Python object, or return an empty pointer on error
// (forward declaration, the function will be defined later)
potential::PtrPotential getPotential(PyObject* pot_obj, utils::KeyValueMap* params = NULL);

// create a Python Potential object and initialize it with an existing instance of C++ potential class
// (forward declaration, the function will come later)
PyObject* createPotentialObject(const potential::PtrPotential& pot);

/// create a Python Density object and initialize it with an existing instance of C++ density class
PyObject* createDensityObject(const potential::PtrDensity& dens)
{
    if(!dens) {  // empty density, likely by mistake
        Py_INCREF(Py_None);
        return Py_None;
    }
    DensityObject* dens_obj = PyObject_New(DensityObject, DensityTypePtr);
    if(!dens_obj)
        return NULL;

    // this is a DIRTY HACK!!! we have allocated a new instance of Python class object,
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
    FILTERMSG(utils::VL_DEBUG, "Agama", "Created a Python wrapper for " +
        dens->name() + " density at " + utils::toString(dens.get()));
    return (PyObject*)dens_obj;
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
    std::vector<double> dens = potential::densityFromCumulativeMass(radius, mass);
    return potential::PtrDensity(new potential::DensitySphericalHarmonic(
        radius, std::vector<std::vector<double> >(1, dens)));
}

/// attempt to construct a density from key=value parameters
potential::PtrDensity Density_initFromDict(PyObject* namedArgs)
{
    NamedArgs nargs(namedArgs);
    // check if the input is a cumulative mass profile
    PyObject* cumulmass = nargs.pop("cumulmass");
    if(cumulmass)
        return Density_initFromCumulMass(cumulmass);
    PyObject* dens_obj = nargs.get("density");
    utils::KeyValueMap params(nargs);
    if(dens_obj && !PyString_Check(dens_obj)) {
        // if it's not a name, it must be a Density instance or a user-defined Python function
        potential::PtrDensity dens = getDensity(dens_obj, &params);
        if(!dens)
            throw std::invalid_argument("Argument 'density' must be either a string, "
                "or an instance of Density class, or a function providing that interface");
        // create a density expansion (if params['type'] is provided) or use the input density directly,
        // and add any relevant modifiers on top of it (if corresponding params are provided).
        // possibly passing a user-defined function to the C++ code, so need to release GIL
        PyReleaseGIL unlock;
        return potential::createDensity(params, dens, *conv);
    }
    if(!params.contains("type") && !params.contains("density") && !params.contains("file"))
        throw std::invalid_argument("Should provide the name of density model "
            "in type='...' or density='...', or the file name to load in file='...' arguments");
    return potential::createDensity(params, *conv);
}

/// attempt to construct a composite density from a tuple of Density-like objects
/// or dictionariess with density parameters
potential::PtrDensity Density_initFromTuple(PyObject* tuple)
{
    // if we have one string parameter, it could be the file name
    if(PyTuple_Size(tuple) == 1 && PyString_Check(PyTuple_GET_ITEM(tuple, 0)))
        return potential::readDensity(PyString_AsString(PyTuple_GET_ITEM(tuple, 0)), *conv);
    std::vector<potential::PtrDensity> components;
    for(Py_ssize_t i=0; i<PyTuple_Size(tuple); i++) {
        PyObject* item = PyTuple_GET_ITEM(tuple, i);
        potential::PtrDensity comp = PyDict_Check(item) ?
            Density_initFromDict(item) :
            getDensity(item);   // Density or Potential or a user-defined function
        if(!comp)
            throw std::invalid_argument("Unnamed arguments should contain only valid Density objects, "
                "or functions providing that interface, or dictionaries with density parameters");
        // if it is a composite Density or Potential, unpack it into components
        const potential::CompositeDensity
            *cden = dynamic_cast<const potential::CompositeDensity* >(comp.get());
        const potential::Composite
            *cpot = dynamic_cast<const potential::Composite* >(comp.get());
        if(cden) {
            for(size_t c=0; c<cden->size(); c++)
                components.push_back(cden->component(c));
        } else if(cpot) {
            for(size_t c=0; c<cpot->size(); c++)
                components.push_back(cpot->component(c));
        } else  // not a composite object
            components.push_back(comp);
    }
    // the constructor of composite density does not involve OpenMP, so we may skip releasing GIL
    return components.size()==1 ? components[0] :
        potential::PtrDensity(new potential::CompositeDensity(components));
}

/// generic constructor of Density class
int Density_init(DensityObject* self, PyObject* args, PyObject* namedArgs)
{
    if(self->dens) {
        PyErr_SetString(PyExc_RuntimeError, "Density object cannot be reinitialized");
        return -1;
    }
    try{
        Py_ssize_t numargs = args!=NULL && PyTuple_Check(args) ? PyTuple_Size(args) : 0;
        Py_ssize_t numnamed= namedArgs!=NULL && PyDict_Check(namedArgs) ? PyDict_Size(namedArgs) : 0;
        if(numargs>0 && numnamed==0)
            self->dens = Density_initFromTuple(args);
        else if(numargs==0 && numnamed>0)
            self->dens = Density_initFromDict(namedArgs);
        else if(numargs==0)
            PyErr_SetString(PyExc_TypeError,
                "Argument list cannot be empty, type help(Density) for details");
        else if(numargs==1 && numnamed==1) {
            // an exception to the rule below: may provide one positional argument (a user-defined
            // function, but *not* an instance of Density) and one required named argument "symmetry"
            PyObject* arg = PyTuple_GET_ITEM(args, 0);
            PyObject* namedArg = NamedArgs(namedArgs).pop("symmetry");
            if( namedArg &&
                !PyObject_TypeCheck(arg, DensityTypePtr) &&
                checkCallable(arg, /*dimension of input*/ 3))
            {   // then create a C++ wrapper for this Python function with the prescribed symmetry
                self->dens = potential::PtrDensity(new DensityWrapper(arg,
                    potential::getSymmetryTypeByName(toString(namedArg))));
            }
        } else
            PyErr_SetString(PyExc_TypeError, "Invalid arguments passed to the constructor "
                "(cannot mix positional and named arguments), type help(Density) for details");
        if(PyErr_Occurred())
            return -1;
        else if(!self->dens) {
            PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to the constructor");
            return -1;
        }
        FILTERMSG(utils::VL_DEBUG, "Agama", "Created "+std::string(self->dens->name())+
            " density at "+utils::toString(self->dens.get()));
        unitsWarning = true;  // any subsequent call to setUnits() will raise a warning
        return 0;
    }
    catch(std::exception& ex) {
        raisePythonException(ex, "Error in creating density: ");
        return -1;
    }
}

/// extract a pointer to C++ Density class from a Python object,
/// or create a wrapper around a user-defined Python function that provides the density.
/// If the input object is neither of these, return an empty pointer.
/// The optional second argument is modified upon success, removing 'density' and
/// possibly 'symmetry' keys from the KeyValueMap.
potential::PtrDensity getDensity(PyObject* dens_obj, utils::KeyValueMap* params /* NULL by default */)
{
    if(dens_obj == NULL)
        return potential::PtrDensity();

    // check if this is a Python wrapper class for a C++ Density object (DensityType)
    // or a Python class PotentiaType, which is a subclass of DensityType
    if(PyObject_TypeCheck(dens_obj, DensityTypePtr) && ((DensityObject*)dens_obj)->dens) {
        if(params)
            params->unset("density");
        return ((DensityObject*)dens_obj)->dens;
    }

    // otherwise this could be an arbitrary Python function, in which case
    // create a C++ wrapper for it with a (possibly) prescribed symmetry
    if(checkCallable(dens_obj, /*dimension of input*/ 3)) {
        coord::SymmetryType sym = params?
            potential::getSymmetryTypeByName(params->getString("symmetry")) :
            coord::ST_UNKNOWN;
        if(params) {
            params->unset("density");
            params->unset("symmetry");
        }
        return potential::PtrDensity(new DensityWrapper(dens_obj, sym));
    }

    // none of the above succeeded -- return an empty pointer
    return potential::PtrDensity();
}

/// compute the density at one or more points
class FncDensityDensity: public BatchFunctionVectorized {
    const potential::BaseDensity& dens;
    const std::vector<double> time;
    double* outputBuffer;
public:
    FncDensityDensity(PyObject* input, PyObject* namedArgs, const potential::BaseDensity& _dens) :
        BatchFunctionVectorized(input, /*input length*/ 3),
        dens(_dens),
        time(NamedArgs(namedArgs).popArray("t", numPoints))
    {
        if(!PyErr_Occurred()) {
            outputObject = allocateOutput<1>(numPoints, &outputBuffer);
            assert(!time.empty());
        }
    }
    virtual void processManyPoints(npy_intp indexStart, npy_intp indexEnd)
    {
        npy_intp npoints = indexEnd - indexStart;
        if(time.size() == 1) {
            // a single value of time - may call the vectorized form of density method
            ALLOC(npoints, coord::PosCar, points)
            for(npy_intp i=0; i<npoints; i++)
                points[i] = convertPos(&inputBuffer[(i + indexStart) * 3]);
            dens.evalmanyDensityCar(npoints, points, &outputBuffer[indexStart],
                time[0] * conv->timeUnit);
            for(npy_intp indexPoint=indexStart; indexPoint<indexEnd; indexPoint++)
                outputBuffer[indexPoint] /= conv->massUnit / pow_3(conv->lengthUnit);
        } else {
            // different values of time for each point - have to loop manually
            for(npy_intp indexPoint=indexStart; indexPoint<indexEnd; indexPoint++)
                outputBuffer[indexPoint] =
                    dens.density(coord::PosCar(convertPos(&inputBuffer[indexPoint*3])),
                        time[indexPoint % time.size()] * conv->timeUnit) /
                    (conv->massUnit / pow_3(conv->lengthUnit));
        }
    }
};

PyObject* Density_density(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    return FncDensityDensity(args, namedArgs, *((DensityObject*)self)->dens).run(/*chunk*/1024);
}

/// compute the projected (surface) density for an array of points
class FncDensityProjectedDensity: public BatchFunction {
    const potential::BaseDensity& dens;
    std::vector<double> alpha, beta, gamma, time;
    double* outputBuffer;
public:
    FncDensityProjectedDensity(PyObject* input, PyObject* namedArgs,
        const potential::BaseDensity& _dens) :
        BatchFunction(input, /*input length*/ 2),
        dens(_dens)
    {
        NamedArgs nargs(namedArgs);
        alpha = nargs.popArray("alpha", numPoints);
        beta  = nargs.popArray("beta" , numPoints);
        gamma = nargs.popArray("gamma", numPoints);
        time  = nargs.popArray("t"    , numPoints);
        if(!PyErr_Occurred() && nargs.empty()) {
            outputObject = allocateOutput<1>(numPoints, &outputBuffer);
            assert(!alpha.empty() && !beta.empty() && !gamma.empty() && !time.empty());
        }
    }
    virtual void processPoint(npy_intp indexPoint)
    {
        outputBuffer[indexPoint] =
            projectedDensity(dens, coord::PosProj(
                /*X*/ inputBuffer[indexPoint*2  ] * conv->lengthUnit,
                /*Y*/ inputBuffer[indexPoint*2+1] * conv->lengthUnit),
                coord::Orientation(
                    alpha[indexPoint % alpha.size()],
                    beta [indexPoint % beta .size()],
                    gamma[indexPoint % gamma.size()]),
                /*optional time argument*/ time[indexPoint % time.size()] * conv->timeUnit) /
            (conv->massUnit / pow_2(conv->lengthUnit));
    }
};

PyObject* Density_projectedDensity(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    // args may be just two numbers (a single position X,Y), or a Nx2 array of several positions;
    // namedArgs may be empty or contain up to three rotation angles and time
    return FncDensityProjectedDensity(args, namedArgs, *((DensityObject*)self)->dens).run(/*chunk*/64);
}

/// compute the enclosed mass at one or more values of radius
class FncDensityEnclosedMass: public BatchFunction {
    const potential::BaseDensity& dens;
    double* outputBuffer;
public:
    FncDensityEnclosedMass(PyObject* input, const potential::BaseDensity& _dens) :
        BatchFunction(input, /*input length*/ 1), dens(_dens)
    {
        outputObject = allocateOutput<1>(numPoints, &outputBuffer);
    }
    virtual void processPoint(npy_intp indexPoint)
    {
        try{
            outputBuffer[indexPoint] =
                dens.enclosedMass(inputBuffer[indexPoint] * conv->lengthUnit) / conv->massUnit;
        }
        catch(std::exception& ex) {
            PyAcquireGIL lock;  // need to get hold of GIL to issue the following warning
            PyErr_WarnEx(NULL, ("Error in enclosedMass(r=" +
                utils::toString(inputBuffer[indexPoint]) + "): " + ex.what()).c_str(), 1);
            outputBuffer[indexPoint] = NAN;
        }
    }
};

PyObject* Density_enclosedMass(PyObject* self, PyObject* args)
{
    return FncDensityEnclosedMass(args, *((DensityObject*)self)->dens).run(/*chunk*/1);
}

/// compute the total mass
PyObject* Density_totalMass(PyObject* self)
{
    try{
        return PyFloat_FromDouble(((DensityObject*)self)->dens->totalMass() / conv->massUnit);
    }
    catch(std::exception& ex) {
        raisePythonException(ex, "Error in Density.totalMass(): ");
        return NULL;
    }
}

/// compute the principal axis ratio and orientation at one or more values of radius
class FncDensityPrincipalAxes: public BatchFunction {
    const potential::BaseDensity& dens;
    double* outputBuffer[2];
public:
    FncDensityPrincipalAxes(PyObject* input, const potential::BaseDensity& _dens) :
        BatchFunction(input, /*input length*/ 1), dens(_dens)
    {
        outputObject = allocateOutput<3,3>(numPoints, outputBuffer);
    }
    virtual void processPoint(npy_intp indexPoint)
    {
        try{
            principalAxes(dens, inputBuffer[indexPoint] * conv->lengthUnit,
                &outputBuffer[0][indexPoint*3], &outputBuffer[1][indexPoint*3]);
        }
        catch(std::exception& ex) {
            PyAcquireGIL lock;  // need to get hold of GIL to issue the following warning
            PyErr_WarnEx(NULL, ("Error in principalAxes(r=" +
                utils::toString(inputBuffer[indexPoint]) + "): " + ex.what()).c_str(), 1);
            for(int i=0; i<3; i++)
                outputBuffer[0][indexPoint*3+i] = outputBuffer[1][indexPoint*3+i] = NAN;
        }
    }
};

PyObject* Density_principalAxes(PyObject* self, PyObject* args)
{
    if(args!=NULL && PyTuple_Check(args) && PyTuple_Size(args)==0) {
        // call with no arguments is equivalent to INFINITY as an argument
        args = PyFloat_FromDouble(INFINITY);
        PyObject* result = FncDensityPrincipalAxes(args, *((DensityObject*)self)->dens).run(1);
        Py_DECREF(args);
        return result;
    }
    return FncDensityPrincipalAxes(args, *((DensityObject*)self)->dens).run(/*chunk*/1);
}

/// export density/potential to a text (ini) file
PyObject* Density_export(PyObject* self, PyObject* args)
{
    const char* filename=NULL;
    if(!PyArg_ParseTuple(args, "s", &filename))
        return NULL;
    try{
        if(writeDensity(filename, *((DensityObject*)self)->dens, *conv)) {  // this can also export a potential
            Py_INCREF(Py_None);
            return Py_None;
        }
        PyErr_SetString(PyExc_RuntimeError, "Error writing file");
        return NULL;
    }
    catch(std::exception& ex) {
        raisePythonException(ex, "Error writing file: ");
        return NULL;
    }
}

/// sample the density profile with points
PyObject* Density_sample(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"n", "potential", "beta", "kappa", "method", NULL};
    Py_ssize_t numPoints=0;
    PyObject* pot_obj=NULL;
    double beta=NAN, kappa=NAN;  // undefined by default, if no argument is provided
    math::SampleMethod method = math::SM_DEFAULT;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "n|Oddb", const_cast<char**>(keywords),
        &numPoints, &pot_obj, &beta, &kappa, &method))
        return NULL;
    if(numPoints<=0) {
        PyErr_SetString(PyExc_ValueError, "number of sampling points 'n' must be positive");
        return NULL;
    }
    potential::PtrDensity dens  = ((DensityObject*)self)->dens;
    potential::PtrPotential pot = getPotential(pot_obj);  // if not NULL, will assign velocity as well
    if(pot_obj!=NULL && !pot) {
        PyErr_SetString(PyExc_TypeError,
            "'potential' must be a valid Potential object");
        return NULL;
    }
    try{
        particles::ParticleArray<coord::PosCyl> points;
        particles::ParticleArrayCar pointsvel;
        {   // no-GIL block: both sampleDensity and assignVelocity contain OpenMP-parallelized code
            PyReleaseGIL unlock;

            // do the sampling of the density profile
            points = galaxymodel::sampleDensity(*dens, numPoints, method);

            // assign the velocities if needed
            if(pot) {
                pointsvel = galaxymodel::assignVelocity(points, *dens, *pot, beta, kappa);
                /*PyErr_WarnEx(PyExc_FutureWarning, "assigning velocity in sample(..., potential=...) "
                    "is deprecated and may be removed in the future; the recommended way is "
                    "to create a DistributionFunction(type='QuasiSpherical', ...) for the given "
                    "density and potential, then use GalaxyModel(potential, df).sample(...)", 1);*/
            }
        }

        // convert output to NumPy array
        assert(numPoints == (Py_ssize_t)points.size());
        npy_intp dims[] = {numPoints, pot? 6 : 3};  // either position or position+velocity
        PyArrayObject* point_arr = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        PyArrayObject* mass_arr  = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        for(Py_ssize_t i=0; i<numPoints; i++) {
            if(pot)
                unconvertPosVel(pointsvel.point(i), &pyArrayElem<double>(point_arr, i, 0));
            else
                unconvertPos(coord::toPosCar(points.point(i)), &pyArrayElem<double>(point_arr, i, 0));
            pyArrayElem<double>(mass_arr, i) = points.mass(i) / conv->massUnit;
        }
        return Py_BuildValue("NN", point_arr, mass_arr);
    }
    catch(std::exception& ex) {
        raisePythonException(ex, "Error in sample(): ");
        return NULL;
    }
}

/// return the density or potential name and symmetry
PyObject* Density_name(PyObject* self)
{
    if(!((DensityObject*)self)->dens) {
        PyErr_SetString(PyExc_RuntimeError, "Density is not initialized properly");
        return NULL;
    }
    return Py_BuildValue("s", (((DensityObject*)self)->dens->name() + " (symmetry: " +
        potential::getSymmetryNameByType(((DensityObject*)self)->dens->symmetry()) + ")").c_str());
}

/// return the element of a multi-component or otherwise modified density or potential model
PyObject* Density_elem(PyObject* self, Py_ssize_t index)
{
    potential::PtrDensity dens = ((DensityObject*)self)->dens;

    const potential::BaseComposite<potential::BaseDensity>* cd =
        dynamic_cast<const potential::BaseComposite<potential::BaseDensity>* >(dens.get());
    if(cd) {
        if(index<0 || index >= (Py_ssize_t)cd->size()) {
            PyErr_SetString(PyExc_IndexError, "Density component index out of range");
            return NULL;
        }
        return createDensityObject(cd->component(index));
    }

    const potential::BaseComposite<potential::BasePotential>* cp =
        dynamic_cast<const potential::BaseComposite<potential::BasePotential>* >(dens.get());
    if(cp) {
        if(index<0 || index >= (Py_ssize_t)cp->size()) {
            PyErr_SetString(PyExc_IndexError, "Potential component index out of range");
            return NULL;
        }
        return createPotentialObject(cp->component(index));
    }

    // otherwise it's not a composite entity
    PyErr_SetString(PyExc_IndexError, "Not a composite object");
    return NULL;
}

/// return the length of a multi-component density/potential model
Py_ssize_t Density_len(PyObject* self)
{
    potential::PtrDensity dens = ((DensityObject*)self)->dens;

    const potential::BaseComposite<potential::BaseDensity>* cd =
        dynamic_cast<const potential::BaseComposite<potential::BaseDensity>* >(dens.get());
    if(cd)
        return cd->size();

    const potential::BaseComposite<potential::BasePotential>* cp =
        dynamic_cast<const potential::BaseComposite<potential::BasePotential>* >(dens.get());
    if(cp)
        return cp->size();

    // otherwise it's not a composite thing (but we can't throw an exception, so just return 0)
    return 0;
}

/// compare two Python Density objects "by value" (check if they represent the same C++ object)
PyObject* Density_compare(PyObject* self, PyObject* other, int op)
{
    bool equal = ((DensityObject*)self)->dens == ((DensityObject*)other)->dens;
    switch (op) {
        case Py_EQ:
            return PyBool_FromLong(equal);
        case Py_NE:
            return PyBool_FromLong(!equal);
        default:
            PyErr_SetString(PyExc_TypeError, "Invalid comparison");
            return NULL;
    }
}

Py_hash_t Density_hash(PyObject *self)
{
    // use the smart pointer to the underlying C++ object, not the Python object itself,
    // to establish identity between two Python objects containing the same C++ class instance
    return Py_HashPointer(const_cast<void*>(static_cast<const void*>(((DensityObject*)self)->dens.get())));
}

/// syntactic sugar: construct a composite density object by adding two density objects
PyObject* Density_add(PyObject* arg1, PyObject* arg2)
{
    potential::PtrPotential pot1, pot2;
    potential::PtrDensity dens1, dens2;
    if(PyObject_TypeCheck(arg1, PotentialTypePtr))
        pot1 = ((PotentialObject*)arg1)->pot;
    if(PyObject_TypeCheck(arg2, PotentialTypePtr))
        pot2 = ((PotentialObject*)arg2)->pot;
    if(PyObject_TypeCheck(arg1, DensityTypePtr))  // also true if arg1 is a Potential object
        dens1 = ((DensityObject*)arg1)->dens;
    if(PyObject_TypeCheck(arg2, DensityTypePtr))
        dens2 = ((DensityObject*)arg2)->dens;
    // several possibilities: adding two potentials, potential+density, density+density,
    // or something else (unsupported)
    if(dens1 && dens2) {  // true in the first three cases
        bool allpot = pot1 && pot2;  // true in the first case only; the result type is Potential
        std::vector<potential::PtrPotential> componentsPot; // used in the first case only
        std::vector<potential::PtrDensity> componentsDens;  // used in the second and third cases
        // if either operand is a composite density or potential, "unpack" it into components,
        // (to facilitate chained summation operations), otherwise add to the final list directly.
        // the pointers below are initialized to NULL if the dynamic cast fails
        const potential::CompositeDensity
            *cdens1 = dynamic_cast<const potential::CompositeDensity* >(dens1.get()),
            *cdens2 = dynamic_cast<const potential::CompositeDensity* >(dens2.get());
        const potential::Composite
            *cpot1 = dynamic_cast<const potential::Composite* >(dens1.get()),
            *cpot2 = dynamic_cast<const potential::Composite* >(dens2.get());
        if(cpot1) {
            for(unsigned int index=0, size=cpot1->size(); index<size; index++) {
                if(allpot)
                    componentsPot.push_back(cpot1->component(index));
                else
                    componentsDens.push_back(cpot1->component(index));
            }
        } else if(cdens1) {  // allpot==false in this case
            for(unsigned int index=0, size=cdens1->size(); index<size; index++)
                componentsDens.push_back(cdens1->component(index));
        } else if(allpot) {
            componentsPot.push_back(pot1);
        } else {
            componentsDens.push_back(dens1);
        }
        if(cpot2) {
            for(unsigned int index=0, size=cpot2->size(); index<size; index++) {
                if(allpot)
                    componentsPot.push_back(cpot2->component(index));
                else
                    componentsDens.push_back(cpot2->component(index));
            }
        } else if(cdens2) {  // allpot==false in this case
            for(unsigned int index=0, size=cdens2->size(); index<size; index++)
                componentsDens.push_back(cdens2->component(index));
        } else if(allpot) {
            componentsPot.push_back(pot2);
        } else {
            componentsDens.push_back(dens2);
        }
        return allpot ? 
            createPotentialObject(potential::PtrPotential(new potential::Composite(componentsPot))) :
            createDensityObject(potential::PtrDensity(new potential::CompositeDensity(componentsDens)));
    } else {  // trying to add something other than Density or Potential is not supported
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
}

// arithmetic operations - only addition of two density objects, or density and potential
static PyNumberMethods Density_number_methods = {
    Density_add,
};

// indexing scheme is shared by both Density and Potential python classes
static PySequenceMethods Density_sequence_methods = {
    Density_len, 0, 0, Density_elem,
};

static PyMethodDef Density_methods[] = {
    { "density", (PyCFunction)Density_density, METH_VARARGS | METH_KEYWORDS,
      "Compute density at a given point or array of points\n"
      "Arguments: a triplet of floats (x,y,z) or a 2d Nx3 array; optionally t=... (time)\n"
      "Returns: float or array of floats" },
    { "projectedDensity", (PyCFunction)Density_projectedDensity, METH_VARARGS | METH_KEYWORDS,
      "Compute surface density at a given point or array of points\n"
      "Positional arguments:\n"
      "  a pair of floats (X,Y) or a 2d Nx2 array: coordinates in the image plane.\n"
      "Keyword arguments:\n"
      "  alpha, beta, gamma (optional, default 0): three angles specifying the orientation "
      "of the image plane in the intrinsic coordinate system of the model; "
      "in particular, beta is the inclination angle. "
      "Angles can be single numbers or arrays of the same length as the number of points.\n"
      "  t   (optional, default 0): time at which to compute these quantities; "
      "can be a single number or an array of length N (separate values for each input point).\n"
      "Returns: float or array of floats - the density integrated along the line of sight Z "
      "perpendicular to the image plane."},
    { "export", Density_export, METH_VARARGS,
      "Export density or potential expansion coefficients to a text file\n"
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
      "Arguments:\n"
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
      "  kappa - the degree of net rotation in the axisymmetric Jeans model "
      "(controls the decomposition of <v_phi^2> - total second moment of azimuthal velocity - "
      "into the mean streaming velocity <v_phi> and the velocity dispersion sigma_phi). "
      "kappa=0 means no net rotation, kappa=1 corresponds to maximum rotation"
      "(sigma_phi = 1/2 kappa_epi / Omega_epi * sigma_R, where kappa_epi and Omega_epi are "
      "radial and azimuthal epicyclic frequencies).\n"
      "If this argument is provided, this triggers the use of the axisymmetric Jeans method.\n"
      "  method (optional, default 0) - choice of method (currently applies only to the coordinates "
      "sampling), see the docstring of sampleNdim for description.\n"
      "Returns: a tuple of two arrays: "
      "a 2d array of size Nx3 (in case of positions only) or Nx6 (in case of velocity assignment), "
      "and a 1d array of N point masses." },
    { "totalMass", (PyCFunction)Density_totalMass, METH_NOARGS,
      "Return the total mass of the density model.\n"
      "Returns: float number" },
    { "enclosedMass", Density_enclosedMass, METH_VARARGS,
      "Return the mass enclosed within a given radius or a list of radii.\n"
      "Returns: a single float number or an array of numbers" },
    { "principalAxes", Density_principalAxes, METH_VARARGS,
      "Determine the length and orientation of principal axes of the density profile "
      "within a given radius, using the ellipsoidally-weighted moment of inertia.\n"
      "Arguments: a single value of radius (r) or a list of radii; "
      "calling with no arguments is equivalent to r=infinity.\n"
      "Returns: a tuple of two arrays of length 3 (for one input point) or shape (N,3) for N points;\n"
      "the first one contains axes stretching coefficients (k_X, k_Y, k_Z), sorted in "
      "decreasing order and normalized so that their product is unity;\n"
      "the second one contains the Euler angles specifying the orientation of these principal axes "
      "of the moment of inertia (X,Y,Z) in the original Cartesian system (x,y,z), "
      "see the definition of these angles in the appendix of the Agama reference documentation.\n"
      "The axes are determined iteratively, starting from a spherical region of radius r "
      "and deforming it into an ellipsoidal region with the same volume, so that the axis ratios "
      "of this region coincide with the axis ratios of the moment of inertia of the density profile "
      "within the same ellipsoidal region."},
    { NULL }
};

static PyTypeObject DensityType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "agama.Density",
    sizeof(DensityObject), 0, (destructor)Density_dealloc, 0, 0, 0, 0, Density_name,
    &Density_number_methods, &Density_sequence_methods, 0, Density_hash, 0, 0, 0, 0, 0,
#if PY_MAJOR_VERSION==2
    Py_TPFLAGS_CHECKTYPES | /* allow arithmetic operations on different types (i.e. Density+Potential) */
#endif
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE /*allow it to be subclassed*/,
    docstringDensity, 0, 0, Density_compare, 0, 0, 0, Density_methods, 0, 0, 0, 0, 0, 0, 0,
    (initproc)Density_init
};


///@}
//  -----------------------
/// \name  Potential class
//  -----------------------
///@{

/// description of Potential class
static const char* docstringPotential =
    "Potential is a class that represents a wide range of gravitational potentials.\n"
    "There are several ways of initializing the potential instance:\n"
    "  - from a list of key=value arguments that specify an elementary potential class;\n"
    "  - from a tuple of dictionary objects that contain the same list of possible "
    "key/value pairs for each component of a composite potential;\n"
    "  - from an INI file with these parameters for one or several components "
    "and/or with potential expansion coefficients previously stored by the export() method;\n"
    "  - from a tuple of existing Potential objects created previously "
    "(in this case a composite potential is created from these components).\n"
    "Note that all keywords and their values are not case-sensitive.\n\n"
    "List of possible keywords for a single component:\n"
    "  type='...'   the type of potential, can be one of the following 'basic' types:\n"
    "    Harmonic, Logarithmic, Plummer, MiyamotoNagai, LongMurali, NFW, Ferrers, Dehnen, "
    "PerfectEllipsoid, Disk, Spheroid, Nuker, Sersic, King, KeplerBinary, UniformAcceleration;\n"
    "    or one of the expansion types:  BasisSet, Multipole, CylSpline - "
    "in these cases, one should provide either a density model, file name, "
    "or an array of particles.\n"
    DOCSTRING_DENSITY_PARAMS
    "Parameters for potential expansions:\n"
    "  density=...   the density model for a potential expansion.\n"
    "  It may be a string with the name of density profile (most of the elementary potentials "
    "listed above can be used as density models, except those with infinite mass; "
    "in addition, there are other density models without a corresponding potential).\n"
    "  Alternatively, it may be an object providing an appropriate interface -- "
    "either an instance of Density or Potential class, or a user-defined function "
    "`my_density(xyz)` returning the value of density computed simultaneously at N points, "
    "where xyz is a Nx3 array of points in cartesian coordinates (even if N=1, it's a 2d array).\n"
    "  potential=...   instead of density, one may provide a potential source for the expansion. "
    "This argument shoud be either an instance of Potential class, or a user-defined function "
    "`my_potential(xyz)` returning the value of potential at N point, where xyz is a Nx3 array of "
    "points in cartesian coordinates. \n"
    "  file='...'   the name of another INI file with potential parameters and/or "
    "coefficients of a Multipole/CylSpline potential expansion, or an N-body snapshot file "
    "that will be used to compute the coefficients of such expansion.\n"
    "  particles=(coords, mass)   array of point masses to be used in construction of a "
    "potential expansion (an alternative to density=..., potential=... or file='...' options): "
    "should be a tuple with two arrays - coordinates and mass, where the first one is "
    "a two-dimensional Nx3 array and the second one is a one-dimensional array of length N.\n"
    "  symmetry='...'   assumed symmetry for potential expansion constructed from an N-body "
    "snapshot or from a user-defined density or potential function (required in these cases). "
    "Possible options, in order of decreasing symmetry: "
    "'Spherical', 'Axisymmetric', 'Triaxial', 'Bisymmetric', 'Reflection', 'None', "
    "or a numerical code; only the case-insensitive first letter matters).\n"
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
    "  smoothing=...   amount of smoothing in Multipole initialized from an N-body snapshot.\n"
    "  nmax=...   order of radial expansion in BasisSet (the number of basis functions is nmax+1).\n"
    "  eta=...    shape parameter of basis functions in BasisSet (default is 1.0, corresponding "
    "to the Hernquist-Ostriker basis set, but values up to 2.0 typically provide better accuracy "
    "for cuspy density profiles; the minimum value is 0.5, corresponding to the Clutton-Brock "
    "basis set.\n"
    "  r0=...     scale radius of basis functions in BasisSet; if not provided, will be assigned "
    "automatically to the half-mass radius, unless the model has infinite mass.\n\n"
    "Most of these parameters have reasonable default values; the only necessary ones are "
    "`type`, and for a potential expansion, `density` or `file` or `particles`.\n"
    "If the parameters of the potential (including the coefficients of a potential expansion) are"
    "loaded from a file, then the `type` argument should not be provided, and the argument name "
    "`file=` may be omitted (i.e., may provide only the filename as an unnamed string argument).\n"
    "One may create a modified version of an existing Potential object, by passing it as "
    "a `potential` argument together with one or more modifier parameters "
    "(center, orientation, rotation and scale); in this case, `type` should be empty.\n"
    "Examples:\n\n"
    ">>> pot_halo = Potential(type='Dehnen', mass=1e12, gamma=1, scaleRadius=100, p=0.8, q=0.6)\n"
    ">>> pot_disk = Potential(type='MiyamotoNagai', mass=5e10, scaleRadius=5, scaleHeight=0.5)\n"
    ">>> pot_composite = Potential(pot_halo, pot_disk)\n"
    ">>> pot_from_ini = Potential('my_potential.ini')\n"
    ">>> pot_from_snapshot = Potential(type='Multipole', file='snapshot.dat')\n"
    ">>> pot_from_particles = Potential(type='Multipole', particles=(coords, masses), symmetry='t')\n"
    ">>> pot_user = Potential(lambda x: -(numpy.sum(x**2, axis=1) + 1)**-0.5, symmetry='s')\n"
    ">>> pot_shifted = Potential(potential=pot_composite, center=[1.0,2.0,3.0]\n"
    ">>> dens_func = lambda xyz: 1e8 / (numpy.sum((xyz/10.)**4, axis=1) + 1)\n"
    ">>> pot_exp = Potential(type='Multipole', density=dens_func, symmetry='t', "
    "gridSizeR=20, Rmin=1, Rmax=500, lmax=4)\n"
    ">>> disk_par = dict(type='Disk', surfaceDensity=1e9, scaleRadius=3, scaleHeight=0.4)\n"
    ">>> halo_par = dict(type='Spheroid', densityNorm=2e7, scaleRadius=15, gamma=1, beta=3, "
    "outerCutoffRadius=150, axisRatioZ=0.8)\n"
    ">>> pot_galpot = Potential(disk_par, halo_par)\n\n"
    "The latter example illustrates the use of GalPot components (exponential disks and spheroids) "
    "from Dehnen&Binney 1998; these are internally implemented using a Multipole potential expansion "
    "and a special variant of disk potential, but may also be combined with any other components "
    "if needed.\n"
    "The numerical values in the above examples are given in solar masses and kiloparsecs; "
    "a call to `setUnits(length=1, mass=1, velocity=1)` should precede the construction "
    "of potentials in this approach. "
    "Alternatively, one may provide no units at all, and use the `N-body` convention G=1 "
    "(this is the default regime and is restored by calling `setUnits()` without arguments).\n";

/// Helper class for providing a BasePotential interface
/// to a Python function that returns the value of a potential at one or several point
/// (with 1st and 2nd derivatives estimated by finite differences).
class PotentialWrapper: public potential::BasePotentialCar{
    PyObject* fnc;
    const coord::SymmetryType sym;
    const std::string fncname;
public:
    PotentialWrapper(PyObject* _fnc, coord::SymmetryType _sym):
        fnc(_fnc), sym(_sym), fncname(toString(fnc))
    {
        Py_INCREF(fnc);
        FILTERMSG(utils::VL_DEBUG, "Agama",
            "Created a C++ potential wrapper for Python function " + fncname +
            " (symmetry: " + potential::getSymmetryNameByType(sym) + ")");
        if(isUnknown(sym))
            PyErr_WarnEx(NULL, "symmetry is not provided, some methods will not be available", 1);
    }
    ~PotentialWrapper()
    {
        FILTERMSG(utils::VL_DEBUG, "Agama",
            "Deleted a C++ potential wrapper for Python function " + fncname);
        Py_DECREF(fnc);
    }
    virtual coord::SymmetryType symmetry() const { return sym; }
    virtual std::string name() const { return fncname; };
    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double /*time*/) const
    {
        const int NPOINTS = 21;   // number of points in the 3d finite-difference stencil
        const double OFFSETS[NPOINTS][3] = {  // offsets in units of stepsize h
            { 0, 0, 0},   // function value at the point
            {-2, 0, 0},   // 4-point stencil in x  for df/dx, d2f/dx2
            {-1, 0, 0},
            { 1, 0, 0},
            { 2, 0, 0},
            { 0,-2, 0},   // 4-point in y
            { 0,-1, 0},
            { 0, 1, 0},
            { 0, 2, 0},
            { 0, 0,-2},   // 4-point in z
            { 0, 0,-1},
            { 0, 0, 1},
            { 0, 0, 2},
            {-1,-1,-1},   // 8 corners of unit cube for mixed second derivs
            {-1, 1,-1},
            { 1,-1,-1},
            { 1, 1,-1},
            {-1,-1, 1},
            {-1, 1, 1},
            { 1,-1, 1},
            { 1, 1, 1} };
        double xyz[3*NPOINTS], val[NPOINTS];
        unconvertPos(pos, xyz);
        // if 1st derivatives are needed, they will be estimated by finite differencing with this stepsize
        double eps = fmax(sqrt(pow_2(xyz[0])+pow_2(xyz[1])+pow_2(xyz[2])) * 5e-4 /* ~DBLEPS^(1/5) */,
            SQRT_DBL_EPSILON);
        int npoints = deriv2 ? NPOINTS : deriv ? 13 : 1;
        for(int d=1; d<npoints; d++) {
            xyz[d*3+0] = xyz[0] + OFFSETS[d][0] * eps;
            xyz[d*3+1] = xyz[1] + OFFSETS[d][1] * eps;
            xyz[d*3+2] = xyz[2] + OFFSETS[d][2] * eps;
        }
        npy_intp dims[]  = {npoints, 3};
        PyObject *result = NULL;
        bool typeerror   = false;
        // open a critical section for accessing Python C API
        {
            PyAcquireGIL lock;
            PyObject* args = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, xyz);
            result = PyObject_CallFunctionObjArgs(fnc, args, NULL);
            Py_DECREF(args);
            if(result == NULL) {
                PyErr_Print();
            } else if(PyArray_Check(result) && PyArray_NDIM((PyArrayObject*)result)==1 &&
                PyArray_DIM((PyArrayObject*)result, 0)==dims[0])
            {
                for(int i=0; i<dims[0]; i++) {
                    switch(PyArray_TYPE((PyArrayObject*) result)) {
                        case NPY_DOUBLE: val[i] = pyArrayElem<double>(result, i); break;
                        case NPY_FLOAT:  val[i] = pyArrayElem<float >(result, i); break;
                        default: typeerror = true;
                    }
                }
            }
            else if(PyNumber_Check(result) && dims[0]==1)
                val[0] = PyFloat_AsDouble(result);
            else
                typeerror = true;
            Py_XDECREF(result);
        }
        if(result == NULL)
            throw std::runtime_error("Call to user-defined potential function failed");
        else if(typeerror)
            throw std::runtime_error("Invalid data type returned by user-defined potential function");
        if(potential)
            *potential = val[0] * pow_2(conv->velocityUnit);
        if(deriv) {  // 4-point rule for 1st derivs, accuracy O(h^4)
            double mul = 1./12 / eps * pow_2(conv->velocityUnit) / conv->lengthUnit;
            deriv->dx = (val[1] - val[4] - 8*val[2] + 8*val[3]) * mul;
            deriv->dy = (val[5] - val[8] - 8*val[6] + 8*val[7]) * mul;
            deriv->dz = (val[9] - val[12]- 8*val[10]+ 8*val[11])* mul;
        }
        if(deriv2) {  // 5-point rule for d2f/dx_i^2 (4th order), and 8-point for mixed derivs (2nd order)
            double mul = 1./12 / pow_2(eps) * pow_2(conv->velocityUnit / conv->lengthUnit);
            deriv2->dx2 = (-val[1] + 16*val[2] - 30*val[0] + 16*val[3] - val[4]) * mul;
            deriv2->dy2 = (-val[5] + 16*val[6] - 30*val[0] + 16*val[7] - val[8]) * mul;
            deriv2->dz2 = (-val[9] + 16*val[10]- 30*val[0] + 16*val[11]- val[12])* mul;
            deriv2->dxdy= (val[13]-val[14]-val[15]+val[16]+val[17]-val[18]-val[19]+val[20]) * mul*1.5;
            deriv2->dxdz= (val[13]-val[15]-val[17]+val[19]+val[14]-val[16]-val[18]+val[20]) * mul*1.5;
            deriv2->dydz= (val[13]-val[17]-val[14]+val[18]+val[15]-val[19]-val[16]+val[20]) * mul*1.5;
        }
    }
};

/// destructor of the Potential class
void Potential_dealloc(PotentialObject* self)
{
    if(self->pot)
        FILTERMSG(utils::VL_DEBUG, "Agama", "Deleted " + std::string(self->pot->name()) +
        " potential at " + utils::toString(self->pot.get()));
    else
        FILTERMSG(utils::VL_DEBUG, "Agama", "Deleted an empty potential");
    self->pot.reset();
    Py_TYPE(self)->tp_free(self);
}

/// create a Python Potential object and initialize it with an existing instance of C++ potential class
PyObject* createPotentialObject(const potential::PtrPotential& pot)
{
    if(!pot) {  // empty potential, likely by mistake
        Py_INCREF(Py_None);
        return Py_None;
    }
    PotentialObject* pot_obj = PyObject_New(PotentialObject, PotentialTypePtr);
    if(!pot_obj)
        return NULL;
    // same hack as in 'createDensityObject()'
    new (&(pot_obj->pot)) potential::PtrPotential;
    pot_obj->pot = pot;
    FILTERMSG(utils::VL_DEBUG, "Agama", "Created a Python wrapper for " +
        pot->name() + " potential at " + utils::toString(pot.get()));
    return (PyObject*)pot_obj;
}

/// attempt to construct an elementary potential from the parameters provided in dictionary
potential::PtrPotential Potential_initFromDict(PyObject* namedArgs)
{
    NamedArgs nargs(namedArgs);
    // check if the arguments contain an array of particles, and if so,
    // convert it to an equivalent C++ class and remove from dict to avoid putting it into KeyValueMap
    PyObject* particles_obj = nargs.pop("particles");
    // also obtain (but not remove) the next two parameters as Python objects (or possibly strings)
    PyObject* dens_obj = nargs.get("density");
    PyObject* pot_obj  = nargs.get("potential");
    // convert the remaining items in the Python dictionary into a KeyValueMap
    utils::KeyValueMap params(nargs);
    if(particles_obj) {
        if(params.contains("file"))
            throw std::invalid_argument("Cannot provide both 'particles' and 'file' arguments");
        if(params.contains("density"))
            throw std::invalid_argument("Cannot provide both 'particles' and 'density' arguments");
        if(!params.contains("type"))
            throw std::invalid_argument("Must provide 'type=\"...\"' argument");
        return potential::createPotential(params, convertParticles<coord::PosCar>(particles_obj), *conv);
    }
    // check if the list of arguments contains a density object
    // or a string specifying the name of density model
    if(int(params.contains("file")) + int(dens_obj!=NULL) + int(pot_obj!=NULL) > 1)
        throw std::invalid_argument("Arguments 'file', 'density', 'potential' are mutually exclusive");
    if(dens_obj) {
        // it must be a string, a Density instance, or a user-defined Python function
        potential::PtrDensity dens = getDensity(dens_obj, &params);
        if(dens) {
            // attempt to construct a potential expansion from the provided density model
            if(params.getString("type").empty())
                throw std::invalid_argument("'type' argument must be provided");
            // the constructor of a potential expansion may be evaluating the input density
            // in an OpenMP-parallelized loop, so we need to release GIL beforehand
            PyReleaseGIL unlock;
            return potential::createPotential(params, dens, *conv);
        } else if(!PyString_Check(dens_obj)) {
            throw std::invalid_argument(
                "'density' argument should be the name of density profile "
                "or an object that provides an appropriate interface "
                "(e.g., an instance of Density or Potential class, or a user-defined function "
                "that takes an array of Nx3 coordinates as a single argument)");
        }
    }
    // check if the list of parameters contains a Potential instance or a user-defined function
    if(pot_obj) {
        potential::PtrPotential pot = getPotential(pot_obj, &params);
        if(pot) {
            // attempt to construct a potential expansion from a user-provided potential model
            // or use the input potential itself, and possibly add modifiers on top of the result.
            // possibly passing a user-defined python function to the potential construction routine
            // that may be evaluating it in an OpenMP-parallelized loop, so need to release GIL
            PyReleaseGIL unlock;
            return potential::createPotential(params, pot, *conv);
        } else {
            throw std::invalid_argument(
                "'potential' argument should be an object that provides an appropriate interface "
                "(e.g., an instance of Potential class, or a user-defined function "
                "that takes an array of Nx3 coordinates as a single argument)");
        }
    }
    return potential::createPotential(params, *conv);
}

/// attempt to construct a composite potential from a tuple of Potential-like objects
/// or dictionaries with potential parameters
potential::PtrPotential Potential_initFromTuple(PyObject* tuple)
{
    // if we have one string parameter, it could be the name of an INI file
    if(PyTuple_Size(tuple) == 1 && PyString_Check(PyTuple_GET_ITEM(tuple, 0)))
        return potential::readPotential(PyString_AsString(PyTuple_GET_ITEM(tuple, 0)), *conv);
    std::vector<potential::PtrPotential> components;
    std::vector<utils::KeyValueMap> paramsArr;
    // first check the types of tuple elements
    for(Py_ssize_t i=0; i<PyTuple_Size(tuple); i++) {
        PyObject* item = PyTuple_GET_ITEM(tuple, i);
        if(PyDict_Check(item))  // a dictionary with param=value pairs
            paramsArr.push_back(NamedArgs(item));
        else {
            // could be an existing Potential object or a compatible user-defined function
            potential::PtrPotential pot = getPotential(item);
            if(pot) {
                // if it is a composite Potential, unpack it into components
                const potential::Composite* cpot = dynamic_cast<const potential::Composite*>(pot.get());
                if(cpot) {
                    for(size_t c=0; c<cpot->size(); c++)
                        components.push_back(cpot->component(c));
                } else  // not a composite
                    components.push_back(pot);
            } else
                throw std::invalid_argument("Unnamed arguments should contain instances of Potential, "
                    "user-defined functions, or dictionaries with potential parameters");
        }
    }
    if(!paramsArr.empty()) {
        potential::PtrPotential pot = potential::createPotential(paramsArr, *conv);
        const potential::Composite* cpot = dynamic_cast<const potential::Composite*>(pot.get());
        if(cpot) {
            for(size_t c=0; c<cpot->size(); c++)
                components.push_back(cpot->component(c));
        } else  // not a composite
            components.push_back(pot);
    }
    if(components.size() == 1)
        return components[0];
    else
        // possibly passing a user-defined Python function to the constructor of Composite potential,
        // but it does not invoke any OpenMP code, so we do not need to release GIL
        return potential::PtrPotential(new potential::Composite(components));
}

/// the generic constructor of Potential object
int Potential_init(PotentialObject* self, PyObject* args, PyObject* namedArgs)
{
    if(self->pot) {
        PyErr_SetString(PyExc_RuntimeError, "Potential object cannot be reinitialized");
        return -1;
    }
    try{
        Py_ssize_t numargs = args!=NULL && PyTuple_Check(args) ? PyTuple_Size(args) : 0;
        Py_ssize_t numnamed= namedArgs!=NULL && PyDict_Check(namedArgs) ? PyDict_Size(namedArgs) : 0;
        if(numargs>0 && numnamed==0)
            self->pot = Potential_initFromTuple(args);
        else if(numargs==0 && numnamed>0)
            self->pot = Potential_initFromDict(namedArgs);
        else if(numargs==0)
            PyErr_SetString(PyExc_TypeError,
                "Argument list cannot be empty, type help(Potential) for details");
        else if(numargs==1 && numnamed==1) {
            // an exception to the rule below: may provide one positional argument (a user-defined
            // function, but *not* an instance of Potential) and one required named argument "symmetry"
            PyObject* arg = PyTuple_GET_ITEM(args, 0);
            PyObject* namedArg = NamedArgs(namedArgs).pop("symmetry");
            if( namedArg &&
                !PyObject_TypeCheck(arg, PotentialTypePtr) &&
                checkCallable(arg, /*dimension of input*/ 3))
            {   // then create a C++ wrapper for this Python function with the prescribed symmetry
                self->pot = potential::PtrPotential(new PotentialWrapper(arg,
                    potential::getSymmetryTypeByName(toString(namedArg))));
            }
        } else
            PyErr_SetString(PyExc_TypeError, "Invalid arguments passed to the constructor "
                "(cannot mix positional and named arguments), type help(Potential) for details");
        if(PyErr_Occurred())
            return -1;
        else if(!self->pot) {
            PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to the constructor");
            return -1;
        }
        FILTERMSG(utils::VL_DEBUG, "Agama", "Created "+std::string(self->pot->name())+
            " potential at "+utils::toString(self->pot.get()));
        unitsWarning = true;  // any subsequent call to setUnits() will raise a warning
        return 0;
    }
    catch(std::exception& ex) {
        raisePythonException(ex, "Error in creating potential: ");
        return -1;
    }
}

/// extract a pointer to C++ Potential class from a Python object,
/// or create a wrapper around a user-defined Python function that provides the potential.
/// If the input object is neither of these, return an empty pointer.
/// The optional second argument is modified upon success, removing 'potential' and
/// possibly 'symmetry' keys from the KeyValueMap.
potential::PtrPotential getPotential(PyObject* pot_obj, utils::KeyValueMap* params /* default NULL */)
{
    if(pot_obj == NULL)
        return potential::PtrPotential();

    // check if this is a Python wrapper class for a C++ Potential object (PotentialType)
    if(PyObject_TypeCheck(pot_obj, PotentialTypePtr) && ((PotentialObject*)pot_obj)->pot) {
        if(params)
            params->unset("potential");
        return ((PotentialObject*)pot_obj)->pot;
    }

    // otherwise this could be an arbitrary Python function, in which case
    // create a C++ wrapper for it with a (possibly) prescribed symmetry
    if(checkCallable(pot_obj, /*dimension of input*/ 3)) {
        coord::SymmetryType sym = params?
            potential::getSymmetryTypeByName(params->getString("symmetry")) :
            coord::ST_UNKNOWN;
        if(params) {
            params->unset("potential");
            params->unset("symmetry");
        }
        return potential::PtrPotential(new PotentialWrapper(pot_obj, sym));
    }

    // none of the above succeeded -- return an empty pointer
    return potential::PtrPotential();
}

// this check seems to be unnecessary, but let it remain for historical reasons
bool Potential_isCorrect(PyObject* self)
{
    if(self==NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Should be called as method of Potential object");
        return false;
    }
    if(!((PotentialObject*)self)->pot) {
        PyErr_SetString(PyExc_RuntimeError, "Potential is not initialized properly");
        return false;
    }
    return true;
}

// functions that do actually compute something from the potential object,
// applying appropriate unit conversions

/// compute the potential at one or more points
class FncPotentialPotential: public BatchFunction {
    const potential::BasePotential& pot;
    const std::vector<double> time;
    double* outputBuffer;
public:
    FncPotentialPotential(PyObject* input, PyObject* namedArgs, const potential::BasePotential& _pot) :
        BatchFunction(input, /*input length*/ 3),
        pot(_pot),
        time(NamedArgs(namedArgs).popArray("t", numPoints))
    {
        if(!PyErr_Occurred()) {
            outputObject = allocateOutput<1>(numPoints, &outputBuffer);
            assert(!time.empty());
        }
    }
    virtual void processPoint(npy_intp indexPoint)
    {
        outputBuffer[indexPoint] =
            pot.value(coord::PosCar(convertPos(&inputBuffer[indexPoint*3])),
                time[indexPoint % time.size()] * conv->timeUnit) /
            pow_2(conv->velocityUnit);
    }
};

PyObject* Potential_potential(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!Potential_isCorrect(self))
        return NULL;
    return FncPotentialPotential(args, namedArgs, *((PotentialObject*)self)->pot).run(/*chunk*/1024);
}

/// compute the force and optionally its derivatives
template<bool DERIV>
class FncPotentialForce: public BatchFunction {
    const potential::BasePotential& pot;
    const std::vector<double> time;
    double* outputBuffers[2];
public:
    FncPotentialForce(PyObject* input, PyObject* namedArgs, const potential::BasePotential& _pot) :
        BatchFunction(input, /*input length*/ 3),
        pot(_pot),
        time(NamedArgs(namedArgs).popArray("t", numPoints))
    {
        if(!PyErr_Occurred()) {
            outputObject = DERIV ?
                allocateOutput<3, 6>(numPoints, outputBuffers) :
                allocateOutput<3   >(numPoints, outputBuffers);
            assert(!time.empty());
        }
    }
    virtual void processPoint(npy_intp ip /*point index*/)
    {
        const coord::PosCar point = convertPos(&inputBuffer[ip*3]);
        coord::GradCar grad;
        coord::HessCar hess;
        pot.eval(point, NULL, &grad, DERIV ? &hess : NULL, time[ip % time.size()] * conv->timeUnit);
        // unit of force per unit mass is V/T
        const double convF = 1 / (conv->velocityUnit / conv->timeUnit);
        outputBuffers[0][ip*3 + 0] = -grad.dx   * convF;
        outputBuffers[0][ip*3 + 1] = -grad.dy   * convF;
        outputBuffers[0][ip*3 + 2] = -grad.dz   * convF;
        if(!DERIV) return;
        // unit of force deriv per unit mass is V/T/L
        const double convD = 1 / (conv->velocityUnit / conv->timeUnit / conv->lengthUnit);
        outputBuffers[1][ip*6 + 0] = -hess.dx2  * convD;
        outputBuffers[1][ip*6 + 1] = -hess.dy2  * convD;
        outputBuffers[1][ip*6 + 2] = -hess.dz2  * convD;
        outputBuffers[1][ip*6 + 3] = -hess.dxdy * convD;
        outputBuffers[1][ip*6 + 4] = -hess.dydz * convD;
        outputBuffers[1][ip*6 + 5] = -hess.dxdz * convD;
    }
};

PyObject* Potential_force(PyObject* self, PyObject* args, PyObject* namedArgs) {
    if(!Potential_isCorrect(self))
        return NULL;
    return FncPotentialForce<false>(args, namedArgs, *((PotentialObject*)self)->pot).run(/*chunk*/1024);
}

PyObject* Potential_forceDeriv(PyObject* self, PyObject* args, PyObject* namedArgs) {
    if(!Potential_isCorrect(self))
        return NULL;
    PyErr_WarnEx(PyExc_FutureWarning,
        "forceDeriv is deprecated, use Potential.eval(..., der=True) instead", 1);
    return FncPotentialForce<true>(args, namedArgs, *((PotentialObject*)self)->pot).run(/*chunk*/1024);
}

/// compute the potential, acceleration and its derivatives for an array of points
class FncPotentialEval: public BatchFunction {
    const potential::BasePotential& pot;
    std::vector<double> time;
    double *outputPot, *outputAcc, *outputDer;    // raw buffers for output quantities
public:
    FncPotentialEval(PyObject* input, PyObject* namedArgs, const potential::BasePotential& _pot) :
        BatchFunction(input, /*input length*/ 3),
        pot(_pot),
        outputPot(NULL), outputAcc(NULL), outputDer(NULL)
    {
        NamedArgs nargs(namedArgs);
        time = nargs.popArray("t", numPoints);
        bool needPot = toBool(nargs.pop("pot"), false);
        bool needAcc = toBool(nargs.pop("acc"), false);
        bool needDer = toBool(nargs.pop("der"), false);
        if(PyErr_Occurred() || !nargs.empty())
            return;
        assert(!time.empty());
        double* outputBuffers[3];
        if(needPot) {
            if(needAcc) {
                if(needDer) {
                    outputObject = allocateOutput<1, 3, 6>(numPoints, outputBuffers);
                    outputDer = outputBuffers[2];
                } else {
                    outputObject = allocateOutput<1, 3   >(numPoints, outputBuffers);
                }
                outputAcc = outputBuffers[1];
            } else {  // no needAcc
                if(needDer) {
                    outputObject = allocateOutput<1,    6>(numPoints, outputBuffers);
                    outputDer = outputBuffers[1];
                } else {
                    outputObject = allocateOutput<1      >(numPoints, outputBuffers);
                }
            }
            outputPot = outputBuffers[0];
        } else {  // no needPot
            if(needAcc) {
                if(needDer) {
                    outputObject = allocateOutput<   3, 6>(numPoints, outputBuffers);
                    outputDer = outputBuffers[1];
                } else {
                    outputObject = allocateOutput<   3   >(numPoints, outputBuffers);
                }
                outputAcc = outputBuffers[0];
            } else {  // no needAcc
                if(needDer) {
                    outputObject = allocateOutput<      6>(numPoints, outputBuffers);
                    outputDer = outputBuffers[0];
                } else {
                    PyErr_SetString(PyExc_RuntimeError, "Nothing to compute!");
                }
            }
        }
    }

    virtual void processPoint(npy_intp indexPoint)
    {
        double Phi;
        coord::GradCar grad;
        coord::HessCar hess;
        pot.eval(convertPos(&inputBuffer[indexPoint*3]),
            outputPot ? &Phi  : NULL,
            outputAcc ? &grad : NULL,
            outputDer ? &hess : NULL,
            time[indexPoint % time.size()] * conv->timeUnit);
        if(outputPot)
            outputPot[indexPoint] = Phi / pow_2(conv->velocityUnit);
        if(outputAcc) {
            const double convF = 1 / (conv->velocityUnit / conv->timeUnit);
            outputAcc[indexPoint*3 + 0] = -grad.dx * convF;
            outputAcc[indexPoint*3 + 1] = -grad.dy * convF;
            outputAcc[indexPoint*3 + 2] = -grad.dz * convF;
        }
        if(outputDer) {
            const double convD = 1 / (conv->velocityUnit / conv->timeUnit / conv->lengthUnit);
            outputDer[indexPoint*6 + 0] = -hess.dx2  * convD;
            outputDer[indexPoint*6 + 1] = -hess.dy2  * convD;
            outputDer[indexPoint*6 + 2] = -hess.dz2  * convD;
            outputDer[indexPoint*6 + 3] = -hess.dxdy * convD;
            outputDer[indexPoint*6 + 4] = -hess.dydz * convD;
            outputDer[indexPoint*6 + 5] = -hess.dxdz * convD;
        }
    }
};

PyObject* Potential_eval(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    // args may be just two numbers (a single position X,Y), or a Nx3 array of several positions;
    // namedArgs specify which quantities to compute and may also contain the time specification
    return FncPotentialEval(args, namedArgs, *((PotentialObject*)self)->pot).run(/*chunk*/1024);
}

/// compute the projected potential, acceleration and its derivatives for an array of points
class FncPotentialProjectedEval: public BatchFunction {
    const potential::BasePotential& pot;
    std::vector<double> alpha, beta, gamma;     // conversion between observed and intrinsic coords
    std::vector<double> time;                   // time(s) at which to evaluate the potential
    double *outputPot, *outputAcc, *outputDer;  // raw buffers for output quantities
public:
    FncPotentialProjectedEval(PyObject* input, PyObject* namedArgs,
        const potential::BasePotential& _pot) :
        BatchFunction(input, /*input length*/ 2),
        pot(_pot),
        outputPot(NULL), outputAcc(NULL), outputDer(NULL)
    {
        NamedArgs nargs(namedArgs);
        alpha = nargs.popArray("alpha", numPoints);
        beta  = nargs.popArray("beta" , numPoints);
        gamma = nargs.popArray("gamma", numPoints);
        time  = nargs.popArray("t"    , numPoints);
        bool needPot = toBool(nargs.pop("pot"), false);
        bool needAcc = toBool(nargs.pop("acc"), false);
        bool needDer = toBool(nargs.pop("der"), false);
        if(PyErr_Occurred() || !nargs.empty())
            return;  // e.g., an error in parsing the angles - keep outputObject=NULL
        assert(!alpha.empty() && !beta.empty() && !gamma.empty() && !time.empty());
        double* outputBuffers[3];
        if(needPot) {
            if(needAcc) {
                if(needDer) {
                    outputObject = allocateOutput<1, 2, 3>(numPoints, outputBuffers);
                    outputDer = outputBuffers[2];
                } else {
                    outputObject = allocateOutput<1, 2   >(numPoints, outputBuffers);
                }
                outputAcc = outputBuffers[1];
            } else {  // no needAcc
                if(needDer) {
                    outputObject = allocateOutput<1,    3>(numPoints, outputBuffers);
                    outputDer = outputBuffers[1];
                } else {
                    outputObject = allocateOutput<1      >(numPoints, outputBuffers);
                }
            }
            outputPot = outputBuffers[0];
        } else {  // no needPot
            if(needAcc) {
                if(needDer) {
                    outputObject = allocateOutput<   2, 3>(numPoints, outputBuffers);
                    outputDer = outputBuffers[1];
                } else {
                    outputObject = allocateOutput<   2   >(numPoints, outputBuffers);
                }
                outputAcc = outputBuffers[0];
            } else {  // no needAcc
                if(needDer) {
                    outputObject = allocateOutput<      3>(numPoints, outputBuffers);
                    outputDer = outputBuffers[0];
                } else {
                    PyErr_SetString(PyExc_RuntimeError, "Nothing to compute!");
                }
            }
        }
    }

    virtual void processPoint(npy_intp indexPoint)
    {
        double Phi;
        coord::GradCar grad;
        coord::HessCar hess;
        projectedEval(pot, coord::PosProj(
            /*X*/ inputBuffer[indexPoint*2  ] * conv->lengthUnit,
            /*Y*/ inputBuffer[indexPoint*2+1] * conv->lengthUnit),
            coord::Orientation(
                alpha[indexPoint % alpha.size()],
                beta [indexPoint % beta .size()],
                gamma[indexPoint % gamma.size()]),
            /*output*/
            outputPot ? &Phi  : NULL,
            outputAcc ? &grad : NULL,
            outputDer ? &hess : NULL,
            /*optional time argument*/ time[indexPoint % time.size()] * conv->timeUnit);
        if(outputPot)
            outputPot[indexPoint] = Phi / (pow_2(conv->velocityUnit) * conv->lengthUnit);
        if(outputAcc) {
            outputAcc[indexPoint*2+0] = -grad.dx / pow_2(conv->velocityUnit);
            outputAcc[indexPoint*2+1] = -grad.dy / pow_2(conv->velocityUnit);
        }
        if(outputDer) {
            outputDer[indexPoint*3+0] = -hess.dx2  / (pow_2(conv->velocityUnit) / conv->lengthUnit);
            outputDer[indexPoint*3+1] = -hess.dy2  / (pow_2(conv->velocityUnit) / conv->lengthUnit);
            outputDer[indexPoint*3+2] = -hess.dxdy / (pow_2(conv->velocityUnit) / conv->lengthUnit);
        }
    }
};

PyObject* Potential_projectedEval(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    // args may be just two numbers (a single position X,Y), or a Nx2 array of several positions;
    // namedArgs specify which quantities to compute and may contain up to three rotation angles and time
    return FncPotentialProjectedEval(args, namedArgs, *((PotentialObject*)self)->pot).run(/*chunk*/64);
}

/// compute the radius of a circular orbit as a function of energy or Lz;
/// optimization: if the input array is greater than a certain threshold,
/// construct a potential interpolator for faster root-finding, augmenting it with a polishing step
template<bool INPUTLZ>
class FncPotentialRcirc: public BatchFunction {
    const potential::BasePotential& pot;
    const potential::Axisymmetrized<potential::BasePotential> axipot;
    double* outputBuffer;
public:
    shared_ptr<const potential::Interpolator> interp;
    FncPotentialRcirc(PyObject* input, const potential::BasePotential& _pot) :
        BatchFunction(input, /*input length*/ 1), pot(_pot), axipot(pot),
        interp(numPoints >= 256 ? new potential::Interpolator(axipot) : NULL)
    {
        outputObject = allocateOutput<1>(numPoints, &outputBuffer);
    }
    virtual void processPoint(npy_intp indexPoint)
    {
        if(interp) {
            coord::GradCyl grad;
            coord::HessCyl hess;
            double R, dR;
            if(INPUTLZ) {
                double Lz = inputBuffer[indexPoint] * conv->lengthUnit * conv->velocityUnit;
                R = interp->R_from_Lz(Lz);
                axipot.eval(coord::PosCyl(R, 0, 0), NULL, &grad, &hess);
                dR = -(R * grad.dR - pow_2(Lz/R)) / (3 * grad.dR + R * hess.dR2);
            } else {
                double E = inputBuffer[indexPoint] * pow_2(conv->velocityUnit), Phi;
                R = interp->R_circ(E);
                axipot.eval(coord::PosCyl(R, 0, 0), &Phi, &grad, &hess);
                dR = -(2 * (Phi - E) + R * grad.dR) / (3 * grad.dR + R * hess.dR2);
            }
            if(fabs(dR / R) < 0.01)  // safety measure against NANs and other corner cases
                R += dR;
            outputBuffer[indexPoint] = R / conv->lengthUnit;
        } else
            outputBuffer[indexPoint] = (INPUTLZ ?
                R_from_Lz(pot, /*Lz*/ inputBuffer[indexPoint] * conv->lengthUnit * conv->velocityUnit) :
                R_circ   (pot, /*E */ inputBuffer[indexPoint] * pow_2(conv->velocityUnit))
            ) / conv->lengthUnit;
    }
};

PyObject* Potential_Rcirc(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!Potential_isCorrect(self))
        return NULL;
    static const char* keywords[] = {"L", "E", NULL};
    PyObject *L_obj=NULL, *E_obj=NULL;
    if( onlyNamedArgs(args, namedArgs) &&
        PyArg_ParseTupleAndKeywords(args, namedArgs, "|OO", const_cast<char**>(keywords),
        &L_obj, &E_obj) &&
        ((L_obj!=NULL) ^ (E_obj!=NULL) /*exactly one of them should be non-NULL*/) )
    {
        if(L_obj) {
            FncPotentialRcirc<true > fnc(L_obj, *((PotentialObject*)self)->pot);
            return fnc.run(/*chunk*/fnc.interp ? 1024 : 64);
         } else {
            FncPotentialRcirc<false> fnc(E_obj, *((PotentialObject*)self)->pot);
            return fnc.run(/*chunk*/fnc.interp ? 1024 : 64);
         }
    } else {
        PyErr_SetString(PyExc_TypeError, "Rcirc() takes exactly one argument (either L or E)");
        return NULL;
    }
}

/// compute the period of a circular orbit as a function of energy or x,v;
/// same optimization as above, use an interpolator if savings exceed its cost of construction
class FncPotentialTcirc: public BatchFunction {
    const potential::BasePotential& pot;
    const potential::Axisymmetrized<potential::BasePotential> axipot;
    double* outputBuffer;
public:
    shared_ptr<const potential::Interpolator> interp;
    FncPotentialTcirc(PyObject* input, const potential::BasePotential& _pot) :
        BatchFunction(input, /*input length - two choices*/ 1, 6, /*custom error message*/
            "Input must be a 1d array of energy values or a 2d Nx6 array of position/velocity values"),
        pot(_pot), axipot(pot), interp(numPoints >= 256 ? new potential::Interpolator(axipot) : NULL)
    {
        outputObject = allocateOutput<1>(numPoints, &outputBuffer);
    }
    virtual void processPoint(npy_intp indexPoint)
    {
        double E = inputLength==6 ?
            totalEnergy(pot, convertPosVel(&inputBuffer[indexPoint*6])) :  // input is 6 phase-space point
            inputBuffer[indexPoint] * pow_2(conv->velocityUnit);           // input is one value of energy
        if(interp) {
            coord::GradCyl grad;
            coord::HessCyl hess;
            double Phi, R = interp->R_circ(E);
            axipot.eval(coord::PosCyl(R, 0, 0), &Phi, &grad, &hess);
            double T = sqrt(R / grad.dR) * 2*M_PI;
            double dR = -(2 * (Phi - E) + R * grad.dR) / (3 * grad.dR + R * hess.dR2);
            double dT = M_PI / sqrt(R / grad.dR) * dR * (grad.dR - R * hess.dR2) / pow_2(grad.dR);
            if(fabs(dT / T) < 0.01)  // safety measure against NANs and other corner cases
                T += dT;
            outputBuffer[indexPoint] = T / conv->timeUnit;
        } else
            outputBuffer[indexPoint] = T_circ(pot, E) / conv->timeUnit;
    }
};

PyObject* Potential_Tcirc(PyObject* self, PyObject* args)
{
    if(!Potential_isCorrect(self))
        return NULL;
    FncPotentialTcirc fnc(args, *((PotentialObject*)self)->pot);
    return fnc.run(/*chunk*/fnc.interp ? 1024 : 64);

}

/// compute the maximum radius that can be reached with a given energy, same optimization as above
class FncPotentialRmax: public BatchFunction {
    const potential::BasePotential& pot;
    const potential::Axisymmetrized<potential::BasePotential> axipot;
    double* outputBuffer;
public:
    shared_ptr<const potential::Interpolator> interp;
    FncPotentialRmax(PyObject* input, const potential::BasePotential& _pot) :
        BatchFunction(input, /*input length*/ 1), pot(_pot), axipot(pot),
        interp(numPoints >= 256 ? new potential::Interpolator(axipot) : NULL)
    {
        outputObject = allocateOutput<1>(numPoints, &outputBuffer);
    }
    virtual void processPoint(npy_intp indexPoint)
    {
        double E = inputBuffer[indexPoint] * pow_2(conv->velocityUnit);
        if(interp) {
            coord::GradCyl grad;
            double Phi, R = interp->R_max(E);
            axipot.eval(coord::PosCyl(R, 0, 0), &Phi, &grad);
            double dR = (E - Phi) / grad.dR;
            if(fabs(dR / R) < 0.01)  // safety measure against NANs and other corner cases
                R += dR;
            outputBuffer[indexPoint] = R / conv->lengthUnit;
        } else
            outputBuffer[indexPoint] = R_max(pot, E) / conv->lengthUnit;
    }
};

PyObject* Potential_Rmax(PyObject* self, PyObject* args) {
    if(!Potential_isCorrect(self))
        return NULL;
    FncPotentialRmax fnc(args, *((PotentialObject*)self)->pot);
    return fnc.run(/*chunk*/fnc.interp ? 1024 : 64);
}

/// compute the peri- and apocenter radii of an orbit in the x,y plane with the given E, Lz or x,v
class FncPotentialRperiapo: public BatchFunction {
    const potential::BasePotential& pot;
    double* outputBuffer;
public:
    FncPotentialRperiapo(PyObject* input, const potential::BasePotential& _pot) :
        BatchFunction(input, /*input length - two choices*/ 2, 6,
            /*custom error message*/ "Input must be a pair of values (E,L), "
            "or an Nx2 array of E,L values, or a Nx6 array of position/velocity values"),
        pot(_pot)
    {
        outputObject = allocateOutput<2>(numPoints, &outputBuffer);
    }
    virtual void processPoint(npy_intp indexPoint)
    {
        double E, L, R1, R2;
        if(inputLength == 6) {  // input is 6 phase-space coords
            coord::PosVelCar point = convertPosVel(&inputBuffer[indexPoint*6]);
            E = totalEnergy(pot, point);
            L = Ltotal(point);
        } else {   // input is a pair of values E, Lz
            E = inputBuffer[indexPoint*2+0] * pow_2(conv->velocityUnit);
            L = inputBuffer[indexPoint*2+1] * conv->velocityUnit * conv->lengthUnit;
        }
        findPlanarOrbitExtent(pot, E, L, /*output*/ R1, R2);
        outputBuffer[indexPoint*2+0] = R1 / conv->lengthUnit;
        outputBuffer[indexPoint*2+1] = R2 / conv->lengthUnit;
    }
};

PyObject* Potential_Rperiapo(PyObject* self, PyObject* args)
{
    if(!Potential_isCorrect(self) || args==NULL || !PyTuple_Check(args))
        return NULL;
    return FncPotentialRperiapo(args, *((PotentialObject*)self)->pot).run(/*chunk*/64);
}

static PyMethodDef Potential_methods[] = {
    { "potential", (PyCFunction)Potential_potential, METH_VARARGS | METH_KEYWORDS,
      "Compute potential at a given point or array of points\n"
      "Arguments: a triplet of floats (x,y,z) or array of such triplets; optionally t=... (time)\n"
      "Returns: float or array of floats" },
    { "force", (PyCFunction)Potential_force, METH_VARARGS | METH_KEYWORDS,
      "Compute force per unit mass (i.e. acceleration, -dPhi/dx) "
      "at a given point or array of points\n"
      "Arguments: a triplet of floats (x,y,z) or array of such triplets; optionally t=... (time)\n"
      "Returns: float[3] - x,y,z components of force, or array of such triplets" },
    { "forceDeriv", (PyCFunction)Potential_forceDeriv, METH_VARARGS | METH_KEYWORDS,
      "Compute force per unit mass and its derivatives at a given point or array of points.\n"
      "Deprecated - use the more general method Potential.eval(..., acc=True, der=True).\n"
      "Arguments: a triplet of floats (x,y,z) or array of such triplets; optionally t=... (time)\n"
      "Returns: (float[3],float[6]) - x,y,z components of force, "
      "and the matrix of force derivatives stored as dFx/dx,dFy/dy,dFz/dz,dFx/dy,dFy/dz,dFz/dx; "
      "or if the input was an array of N points, then both items in the tuple are 2d arrays "
      "with sizes Nx3 and Nx6, respectively"},
    { "eval", (PyCFunction)Potential_eval, METH_VARARGS | METH_KEYWORDS,
      "Compute any combination of potential, acceleration, and its derivatives "
      "at a given point or array of points.\n"
      "Positional arguments: \n"
      "  a triplet of floats (x,y,z) or a 2d Nx3 array of such triplets.\n"
      "Keyword arguments:\n"
      "  pot (optional, default False): whether to compute the potential.\n"
      "  acc (optional, default False): whether to compute the acceleration.\n"
      "  der (optional, default False): whether to compute the acceleration derivatives.\n"
      "  t   (optional, default 0): time at which to compute these quantities; "
      "can be a single number or an array of length N (separate values for each input point).\n"
      "Returns: one, two or three arrays, one for each requested quantity.\n"
      "Potential (Phi) - a single number or an array of the same length as the number of input points.\n"
      "Acceleration (-d Phi / d x_i) - three numbers or an array of shape (N,3).\n"
      "Derivatives (-d2 Phi / d x_i d x_j ) - six numbers stored as (i,j) = xx, yy, zz, xy, yz, zx, "
      "or an array of shape (N,6).\n"},
    { "projectedEval", (PyCFunction)Potential_projectedEval, METH_VARARGS | METH_KEYWORDS,
      "Compute any combination of projected potential, acceleration, and its derivatives "
      "(integral of the corresponding quantities along the line of sight Z, computed "
      "analogously to surface density) at a given point X,Y or array of points.\n"
      "Positional arguments: \n"
      "  a pair of floats (X,Y) or a 2d Nx2 array: coordinates in the image plane.\n"
      "Keyword arguments:\n"
      "  pot (optional, default False): whether to compute the projected potential.\n"
      "  acc (optional, default False): whether to compute the projected acceleration.\n"
      "  der (optional, default False): whether to compute the projected acceleration derivatives.\n"
      "  alpha, beta, gamma (optional, default 0): three angles specifying the orientation "
      "of the image plane in the intrinsic coordinate system of the model; "
      "in particular, beta is the inclination angle. "
      "Angles can be single numbers or arrays of the same length as the number of points.\n"
      "  t   (optional, default 0): time at which to compute these quantities; "
      "can be a single number or an array of length N (separate values for each input point).\n"
      "Returns: one, two or three arrays, one for each requested quantity.\n"
      "Potential is a single number or an array of the same length as the number of input points. "
      "Since the integral of Phi(X,Y,Z) dZ diverges logarithmically at large |Z|, "
      "the actual integration is carried out for the potential difference Phi(X,Y,Z)-Phi(0,0,Z), "
      "and is currently not implemented for potentials that are singular at origin.\n"
      "Acceleration is two numbers or an array of shape (N,2) containing the X and Y "
      "components of projected acceleration (partial derivatives of projected potential "
      "with the minus sign); the Z component is identically zero and is not computed.\n"
      "Derivatives of acceleration are three numbers (integrals of d2Phi/dX2, d2Phi/dY2, d2Phi/dXdY; "
      "other components are identically zero and are not computed), "
      "or an array of shape (N,3).\n"},
    { "Rcirc", (PyCFunction)Potential_Rcirc, METH_VARARGS | METH_KEYWORDS,
      "Find the radius of a circular orbit corresponding to either the given z-component "
      "of angular momentum L or energy E; the potential is averaged over the azimuthal angle (phi) "
      "if not already axisymmetric (all quantities are evaluated in the equatorial plane)\n"
      "Arguments (need to be named, and only one of them can be provided):\n"
      "  L=... (a single number or an array of numbers) - the values of angular momentum, or\n"
      "  E=... (same) - the values of energy.\n"
      "Returns: a single number or an array of numbers - the radii of corresponding orbits\n" },
    { "Tcirc", Potential_Tcirc, METH_VARARGS,
      "Compute the period of a circular orbit for the given energy (a) or the (x,v) point (b); "
      "in either case, the orbit lies in the equatorial plane, and the potential is averaged over "
      "the azimuthal angle (phi) if not already axisymmetric\n"
      "Arguments:\n"
      "  (a) a single value of energy or an array of N such values, or\n"
      "  (b) a single point (6 numbers - position and velocity) or a Nx6 array of points\n"
      "Returns: a single value or N values of orbital periods\n" },
    { "Rmax", Potential_Rmax, METH_VARARGS,
      "Find the maximum radius accessible to the given energy (i.e. the root of Phi(Rmax,0,0)=E); "
      "the potential is averaged over the azimuthal angle (phi) if not already axisymmetric\n"
      "Arguments: a single number or an array of numbers - the values of energy\n"
      "Returns: corresponding values of radii\n" },
    { "Rperiapo", Potential_Rperiapo, METH_VARARGS,
      "Compute the peri/apocenter radii of a planar orbit with the given energy E and "
      "angular momentum L_z. The orbit lies in the equatorial plane, and the potential is averaged "
      "over the azimuthal angle (phi) if not already axisymmetric.\n"
      "If E is outside the valid range, both radii are NAN, and if L is incompatible with E "
      "(exceeds the value of circular angular momentum), both radii are set to Rcirc(E).\n"
      "Arguments:\n"
      "  (a) two values (E,L) or a Nx2 array of such values, or\n"
      "  (b) a single point (6 numbers - position and velocity) or a Nx6 array of points; "
      "in the latter case the total L is used, not just L_z, which produces expected results for "
      "spherically-symmetric potentials (the minimum/maximum spherical radius that an orbit can "
      "attain), but only approximate values for out-of-plane orbits in non-spherical potentials.\n"
      "Returns: a pair of values (Rperi,Rapo) or a Nx2 array of these values for each input point\n" },
    { NULL }
};

static PyTypeObject PotentialType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "agama.Potential",
    sizeof(PotentialObject), 0, (destructor)Potential_dealloc, 0, 0, 0, 0, /*sic!*/ Density_name,
    /*sic!*/ &Density_number_methods, /*sic!*/ &Density_sequence_methods, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE /*allow it to be subclassed*/, docstringPotential,
    0, 0, 0, 0, 0, 0, Potential_methods, 0, 0, /*parent class*/ &DensityType, 0, 0, 0, 0,
    (initproc)Potential_init
};


///@}
//  --------------------------
/// \name  ActionFinder class
//  --------------------------
///@{

/// common fragment of docstring for the ActionFinder class and the standalone actions routine
#define DOCSTRING_ACTIONS \
    "  actions (bool, default True) - whether to compute actions.\n" \
    "  angles (bool, default False) - whether to compute angles (extra work).\n" \
    "  frequencies (bool, default is taken from the \"angles\" argument) - " \
    "whether to compute frequencies (extra work).\n" \
    "Returns:\n" \
    "  each requested quantity (actions, angles, frequencies) is a triplet of floats " \
    "when the input is a single point, otherwise an array of Nx3 floats; the order is " \
    "Jr, Jz, Jphi for actions and similarly for other quantities (thetas and Omegas).\n" \
    "  If only one quantity is requested (e.g., just actions), it is returned directly, " \
    "otherwise a tuple of several arrays is returned (e.g., actions and angles)."

static const char* docstringActionFinder =
    "ActionFinder object is created for a given potential (provided as the first argument "
    "to the constructor); if the potential is axisymmetric, there is a further option to use "
    "interpolation tables for actions (optional second argument 'interp=...', False by default), "
    "which speeds up computation of actions (but not frequencies and angles) at the expense of "
    "a somewhat lower accuracy.\n"
    "The () operator computes any combination of actions, angles and frequencies "
    "for a given position/velocity point or an array of points.\n"
    "Arguments:\n"
    "  point - a sextet of floats (x,y,z,vx,vy,vz) or an Nx6 array of N such sextets.\n"
    DOCSTRING_ACTIONS;

/// \cond INTERNAL_DOCS
/// Python type corresponding to ActionFinder class
typedef struct {
    PyObject_HEAD
    actions::PtrActionFinder af;  // C++ object for action finder
} ActionFinderObject;
/// \endcond

/// destructor of ActionFinder class
void ActionFinder_dealloc(ActionFinderObject* self)
{
    FILTERMSG(utils::VL_DEBUG, "Agama", "Deleted an action finder at " +
        utils::toString(self->af.get()));
    self->af.reset();
    Py_TYPE(self)->tp_free(self);
}

/// create a Python ActionFinder object and initialize it
/// with an existing instance of C++ action finder class
PyObject* createActionFinderObject(actions::PtrActionFinder af)
{
    ActionFinderObject* af_obj = PyObject_New(ActionFinderObject, ActionFinderTypePtr);
    if(!af_obj)
        return NULL;
    // same trickery as in 'createDensityObject()'
    new (&(af_obj->af)) actions::PtrActionFinder;
    af_obj->af = af;
    FILTERMSG(utils::VL_DEBUG, "Agama", "Created a Python wrapper for action finder at "+
        utils::toString(af.get()));
    return (PyObject*)af_obj;
}

/// constructor of ActionFinder class
int ActionFinder_init(ActionFinderObject* self, PyObject* args, PyObject* namedArgs)
{
    if(self->af) {
        PyErr_SetString(PyExc_RuntimeError, "ActionFinder object cannot be reinitialized");
        return -1;
    }
    static const char* keywords[] = {"potential", "interp", NULL};
    PyObject* pot_obj=NULL, *interp_flag=NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "O|O", const_cast<char**>(keywords),
        &pot_obj, &interp_flag))
    {
        return -1;
    }
    potential::PtrPotential pot = getPotential(pot_obj);
    if(!pot) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a valid Potential object");
        return -1;
    }
    try{
        // ActionFinder constructors have OpenMP-parallelized loops, which might call back a Python
        // function if the potential contains one, so we need to release GIL beforehand
        PyReleaseGIL unlock;
        self->af = actions::createActionFinder(pot, toBool(interp_flag, false));
        FILTERMSG(utils::VL_DEBUG, "Agama", "Created " + self->af->name() + " action finder at " +
            utils::toString(self->af.get()));
        return 0;
    }
    catch(std::exception& ex) {
        raisePythonException(ex, "Error in ActionFinder initialization: ");
        return -1;
    }
}

/// batch function for computing any combination of actions, angles and frequencies;
/// this base class performs allocation, unit conversion and storage, while the two derived classes
/// implement the actual computation for the cases of action finder or standalone routines.
class FncActions: public BatchFunction {
    const bool needAct, needAng, needFreq;
    double* outputBuffers[3];
public:
    FncActions(PyObject* input, bool _needAct, bool _needAng, bool _needFreq) :
        BatchFunction(input, /*input length*/ 6),
        needAct(_needAct), needAng(_needAng), needFreq(_needFreq)
    {
        int numOutputs = int(needAct) + int(needAng) + int(needFreq);
        switch(numOutputs) {
            case 3:  outputObject = allocateOutput<3, 3, 3>(numPoints, outputBuffers); break;
            case 2:  outputObject = allocateOutput<3, 3   >(numPoints, outputBuffers); break;
            case 1:  outputObject = allocateOutput<3      >(numPoints, outputBuffers); break;
            default: outputObject = Py_None; Py_INCREF(Py_None);
        }
    }

    // actual implementation uses either a standalone action function or an action finder
    virtual void eval(const coord::PosVelCyl& point,
        actions::Actions* act, actions::Angles* ang, actions::Frequencies* freq) const = 0;

    virtual void processPoint(npy_intp indexPoint)
    {
        actions::Actions act;
        actions::Angles ang;
        actions::Frequencies freq;
        eval(coord::toPosVelCyl(convertPosVel(&inputBuffer[indexPoint*6])),
            needAct? &act : NULL, needAng? &ang : NULL, needFreq? &freq : NULL);
        // unit-convert and store output values
        int numOut = 0;
        if(needAct) {
            // unit of action is V*L
            const double convA = 1 / (conv->velocityUnit * conv->lengthUnit);
            outputBuffers[numOut][indexPoint*3 + 0] = act.Jr   * convA;
            outputBuffers[numOut][indexPoint*3 + 1] = act.Jz   * convA;
            outputBuffers[numOut][indexPoint*3 + 2] = act.Jphi * convA;
            numOut++;
        }
        if(needAng) {
            outputBuffers[numOut][indexPoint*3 + 0] = ang.thetar;
            outputBuffers[numOut][indexPoint*3 + 1] = ang.thetaz;
            outputBuffers[numOut][indexPoint*3 + 2] = ang.thetaphi;
            numOut++;
        }
        if(needFreq) {
            // unit of frequency is V/L
            const double convF = conv->lengthUnit / conv->velocityUnit;
            outputBuffers[numOut][indexPoint*3 + 0] = freq.Omegar   * convF;
            outputBuffers[numOut][indexPoint*3 + 1] = freq.Omegaz   * convF;
            outputBuffers[numOut][indexPoint*3 + 2] = freq.Omegaphi * convF;
            numOut++;
        }
    }
};

/// specialization for the case of action finder
class FncActionsFinder: public FncActions {
    const actions::BaseActionFinder& af;
public:
    FncActionsFinder(PyObject* input, bool needAct, bool needAng, bool needFreq,
        const actions::BaseActionFinder& _af) :
    FncActions(input, needAct, needAng, needFreq), af(_af) {}

    virtual void eval(const coord::PosVelCyl& point,
        actions::Actions* act, actions::Angles* ang, actions::Frequencies* freq) const
    {
        af.eval(point, act, ang, freq);
    }
};

/// specialization for the standalone action routine
class FncActionsStandalone: public FncActions {
    const potential::BasePotential& pot;
    double fd;   // focal distance
public:
    FncActionsStandalone(PyObject* input, bool needAct, bool needAng, bool needFreq,
        const potential::BasePotential& _pot, double _fd) :
    FncActions(input, needAct, needAng, needFreq), pot(_pot), fd(_fd * conv->lengthUnit) {}

    virtual void eval(const coord::PosVelCyl& point,
        actions::Actions* act, actions::Angles* ang, actions::Frequencies* freq) const
    {
        actions::eval(pot, point, act, ang, freq, fd);
    }
};

PyObject* ActionFinder_value(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!((ActionFinderObject*)self)->af) {
        PyErr_SetString(PyExc_RuntimeError, "ActionFinder object is not properly initialized");
        return NULL;
    }
    static const char* keywords[] = {"point", "actions", "angles", "frequencies", NULL};
    PyObject *points_obj = NULL, *needAct_flag = NULL, *needAng_flag = NULL, *needFreq_flag = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "O|OOO", const_cast<char**>(keywords),
        &points_obj, &needAct_flag, &needAng_flag, &needFreq_flag))
    {
        PyErr_SetString(PyExc_TypeError,
            "Must provide an array of points and optionally boolean flags specifying "
            "which quantities to compute (actions, angles and/or frequencies)");
        return NULL;
    }
    bool needAct  = toBool(needAct_flag, true);
    bool needAng  = toBool(needAng_flag, false);
    bool needFreq = toBool(needFreq_flag, needAng);
    return FncActionsFinder(points_obj, needAct, needAng, needFreq, *((ActionFinderObject*)self)->af) .
        run(/*chunk*/64);
}

PyObject* ActionFinder_name(PyObject* self)
{
    if(!((ActionFinderObject*)self)->af) {
        PyErr_SetString(PyExc_RuntimeError, "ActionFinder object is not properly initialized");
        return NULL;
    }
    return Py_BuildValue("s", ((ActionFinderObject*)self)->af->name().c_str());
}

static PyTypeObject ActionFinderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "agama.ActionFinder",
    sizeof(ActionFinderObject), 0, (destructor)ActionFinder_dealloc,
    0, 0, 0, 0, ActionFinder_name, 0, 0, 0, 0, ActionFinder_value, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringActionFinder,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    (initproc)ActionFinder_init
};


static const char* docstringActions =
    "Compute actions for a given position/velocity point, or array of points.\n"
    "Arguments:\n"
    "  potential - Potential object that defines the gravitational potential.\n"
    "  point - a sextet of floats (x,y,z,vx,vy,vz) or an Nx6 array of N such sextets.\n"
    "  fd (float, default 0) - focal distance for the prolate spheroidal coordinate system "
    "(not necessary if the potential is spherical).\n"
    DOCSTRING_ACTIONS;

PyObject* actions(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"potential", "point", "fd", "actions", "angles", "frequencies",
        NULL};
    double fd = 0;
    PyObject *pot_obj = NULL, *points_obj = NULL,
        *needAct_flag = NULL, *needAng_flag = NULL, *needFreq_flag = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "|OOdOOO", const_cast<char**>(keywords),
        &pot_obj, &points_obj, &fd, &needAct_flag, &needAng_flag, &needFreq_flag))
    {
        return NULL;
    }
    potential::PtrPotential pot = getPotential(pot_obj);
    if(!pot) {
        PyErr_SetString(PyExc_TypeError, "Argument 'potential' must be a valid Potential object");
        return NULL;
    }
    bool needAct  = toBool(needAct_flag, true);
    bool needAng  = toBool(needAng_flag, false);
    bool needFreq = toBool(needFreq_flag, needAng);
    return FncActionsStandalone(points_obj, needAct, needAng, needFreq, *pot, fd) .
        run(/*chunk*/64);
}


///@}
//  --------------------------
/// \name  ActionMapper class
//  --------------------------
///@{

static const char* docstringActionMapper =
    "ActionMapper performs an inverse operation to ActionFinder, namely, compute the position "
    "and velocity (and optionally frequencies) from actions and angles.\n"
    "It performs exact mapping for spherical potentials and approximate mapping "
    "(using the Torus Machine) for axisymmetric potentials.\n"
    "The object is created for a given potential; an optional parameter 'tol' specifies "
    "the accuracy of torus construction.\n"
    "The () operator computes the positions and velocities for one or more combinations of "
    "actions and angles (Jr, Jz, Jphi, theta_r, theta_z, theta_phi), "
    "returning a sextet of floats (x,y,z,vx,vy,vz) for a single input point, "
    "or an Nx6 array of such sextets in case the input is a Nx6 array of action/angle points.\n"
    "If an optional argument 'frequencies' is True, it additionally returns a second array "
    "containing a triplet of frequencies (Omega_r, Omega_z, Omega_phi) for a single input point, "
    "or an Nx3 array of such triplets when the input contains more than one point.\n"
    "Example:\n"
    "    am = agama.ActionMapper(pot)   # create an action mapper\n"
    "    af = agama.ActionFinder(pot)   # create an inverse action finder\n"
    "    aa = [ [1., 2., 3., 4., 5., 6], [6., 5., 4., 3., 2., 1.] ]   # two action/angle points\n"
    "    xv, Om = am(aa)   # map these to two position/velocity points and two frequency triplets\n"
    "    J,theta,Omega = af(xv, angles=True, frequencies=True)   # convert from x,v to act,ang,freq\n"
    "    print(Om, Omega)  # frequencies of forward and inverse mappings should be roughly equal\n"
    "    print(J, aa[:,0:3])   # computed actions should also approximately match the input values\n"
    "    print(theta, aa[:,3:6])   # and same for angles\n";

/// \cond INTERNAL_DOCS
/// Python type corresponding to ActionMapper class
typedef struct {
    PyObject_HEAD
    actions::PtrActionMapper am;  // C++ object for action mapper
    bool useTorus;  // whether the mapper uses Torus (not thread-safe) or another implementation
} ActionMapperObject;
/// \endcond

void ActionMapper_dealloc(ActionMapperObject* self)
{
    FILTERMSG(utils::VL_DEBUG, "Agama", "Deleted an action mapper at " +
        utils::toString(self->am.get()));
    self->am.reset();
    Py_TYPE(self)->tp_free(self);
}

int ActionMapper_init(ActionMapperObject* self, PyObject* args, PyObject* namedArgs)
{
    if(self->am) {
        PyErr_SetString(PyExc_RuntimeError, "ActionMapper object cannot be reinitialized");
        return -1;
    }
    static const char* keywords[] = {"potential", "tol", NULL};
    PyObject* pot_obj=NULL;
    double tol=NAN;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "O|d", const_cast<char**>(keywords),
        &pot_obj, &tol))
    {
        return -1;
    }
    potential::PtrPotential pot = getPotential(pot_obj);
    if(!pot) {
        PyErr_SetString(PyExc_TypeError, "First argument must be a valid Potential object");
        return -1;
    }
    try{
        PyReleaseGIL unlock;  // the action mapper constructor may have an OpenMP-parallelized loop
        self->am = tol == tol ?
            actions::createActionMapper(pot, tol) :  // use the provided value of tol
            actions::createActionMapper(pot);        // use the default value
        // if the potential is non-spherical, the underlying implementation uses Torus
        self->useTorus = !isSpherical(*pot);
        FILTERMSG(utils::VL_DEBUG, "Agama", "Created " + self->am->name() + " action mapper at " +
            utils::toString(self->am.get()));
        return 0;
    }
    catch(std::exception& ex) {
        raisePythonException(ex, "Error in ActionMapper initialization: ");
        return -1;
    }
}

/// compute the position/velocity from angles
class FncActionMapper: public BatchFunction {
    const actions::BaseActionMapper& am;
    bool needFreq;
    double* outputBuffers[2];
public:
    FncActionMapper(PyObject* input, const actions::BaseActionMapper& _am, bool _needFreq) :
        BatchFunction(input, /*input length*/ 6),
        am(_am), needFreq(_needFreq)
    {
        outputObject = needFreq ?
            allocateOutput<6, 3>(numPoints, outputBuffers) :
            allocateOutput<6   >(numPoints, outputBuffers);
    }
    virtual void processPoint(npy_intp ip /*point index*/)
    {
        try{
            actions::Frequencies freq;
            // unit of action is V*L
            const double convA = conv->velocityUnit * conv->lengthUnit;
            coord::PosVelCyl point = am.map(actions::ActionAngles(
                actions::Actions(
                    inputBuffer[ip*6 + 0] * convA,
                    inputBuffer[ip*6 + 1] * convA,
                    inputBuffer[ip*6 + 2] * convA),
                actions::Angles(
                    inputBuffer[ip*6 + 3],
                    inputBuffer[ip*6 + 4],
                    inputBuffer[ip*6 + 5]) ),
                needFreq? &freq : NULL);
            unconvertPosVel(toPosVelCar(point), &outputBuffers[0][ip*6]);
            if(needFreq) {
                // unit of frequency is V/L
                const double convF = conv->lengthUnit / conv->velocityUnit;
                outputBuffers[1][ip*3 + 0] = freq.Omegar   * convF;
                outputBuffers[1][ip*3 + 1] = freq.Omegaz   * convF;
                outputBuffers[1][ip*3 + 2] = freq.Omegaphi * convF;
            }
        }
        catch(std::exception& ex) {
            FILTERMSG(utils::VL_DEBUG, "Agama", ex.what());
            std::fill(&outputBuffers[0][ip*6], &outputBuffers[0][ip*6+6], NAN);
            if(needFreq)
                std::fill(&outputBuffers[1][ip*3], &outputBuffers[1][ip*3+3], NAN);
        }
    }
};

PyObject* ActionMapper_value(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    if(((ActionMapperObject*)self)->am==NULL) {
        PyErr_SetString(PyExc_RuntimeError, "ActionMapper object is not properly initialized");
        return NULL;
    }
    static const char* keywords[] = {"point", "frequencies", NULL};
    PyObject *points_obj = NULL, *needFreq_flag = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "O|O", const_cast<char**>(keywords),
        &points_obj, &needFreq_flag))
    {
        return NULL;
    }
    bool needFreq = toBool(needFreq_flag, false);
    return FncActionMapper(points_obj, *((ActionMapperObject*)self)->am, needFreq) .
        run(/*chunk*/ ((ActionMapperObject*)self)->useTorus ?
            0  /* disable parallelization: Torus is not thread-safe */ :
            64 /* otherwise parallelize normally */);
}

PyObject* ActionMapper_name(PyObject* self)
{
    if(!((ActionMapperObject*)self)->am) {
        PyErr_SetString(PyExc_RuntimeError, "ActionMapper object is not properly initialized");
        return NULL;
    }
    return Py_BuildValue("s", ((ActionMapperObject*)self)->am->name().c_str());
}

static PyTypeObject ActionMapperType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "agama.ActionMapper",
    sizeof(ActionMapperObject), 0, (destructor)ActionMapper_dealloc,
    0, 0, 0, 0, ActionMapper_name, 0, 0, 0, 0, (PyCFunctionWithKeywords)ActionMapper_value, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringActionMapper,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    (initproc)ActionMapper_init
};


///@}
//  ----------------------------------
/// \name  DistributionFunction class
//  ----------------------------------
///@{

static const char* docstringDistributionFunction =
    "DistributionFunction class represents an action-based distribution function.\n\n"
    "The constructor accepts several key=value arguments that describe the parameters "
    "of distribution function.\n"
    "Required parameter is type='...', specifying the type of DF. Currently available types are:\n"
    "'DoublePowerLaw' (for the halo);\n"
    "'QuasiIsothermal' and 'Exponential' (for the disk component);\n"
    "'QuasiSpherical' (for the isotropic or anisotropic DF of the Cuddeford-Osipkov-Merritt type "
    "corresponding to a given density profile - by default it is the isotropic DF produced by "
    "the Eddington inversion formula).\n"
    //"'Interp1', 'Interp3' (for interpolated DFs - currently under construction).\n"
    "For some of them, one also needs to provide the potential to initialize the table of epicyclic "
    "frequencies (potential=... argument). For the QuasiSpherical DF one needs to provide "
    "an instance of density profile (density=...) and the potential (if they are the same, then only "
    "potential=... is needed), and optionally the central value of anisotropy coefficient `beta0` "
    "(by default 0) and the anisotropy radius `r_a` (by default infinity).\n"
    "Other parameters are specific to each DF type.\n"
    "Alternatively, a composite DF may be created from an array of previously constructed DFs:\n"
    ">>> df = DistributionFunction(df1, df2, df3)\n\n"
    "The () operator computes the value of distribution function for the given triplet of actions, "
    "or N such values if the input is a 2d array of shape Nx3. When called with an optional argument "
    "der=True, it returns a 2-tuple with the DF values (array of length N) and its derivatives w.r.t. "
    "actions (array of shape Nx3).\n"
    "The totalMass() function computes the total mass in the entire phase space.\n\n"
    "One may provide a user-defined DF function in all contexts where a DistributionFunction object "
    "is required. This function should take a single positional argument - Nx3 array of actions "
    "(with columns representing Jr, Jz, Jphi at N>=1 points) and returns an array of length N. "
    "This function may optionally provide derivatives when called with a named argument der=True, "
    "and in this case should return a 2-tuple with DF values (array of length N) and derivatives "
    "(array of shape Nx3).";

/// \cond INTERNAL_DOCS
/// Python type corresponding to DistributionFunction class
typedef struct {
    PyObject_HEAD
    df::PtrDistributionFunction df;
} DistributionFunctionObject;
/// \endcond

/// destructor of DistributionFunction class
void DistributionFunction_dealloc(DistributionFunctionObject* self)
{
    FILTERMSG(utils::VL_DEBUG, "Agama", "Deleted a distribution function at " +
        utils::toString(self->df.get()));
    self->df.reset();
    Py_TYPE(self)->tp_free(self);
}

/// Helper class for providing a BaseDistributionFunction interface
/// to a Python function that returns the value of DF at a point in action space
class DistributionFunctionWrapper: public df::BaseDistributionFunction{
    PyObject* fnc;   ///< Python object providing the __call__ interface to evaluate the DF
public:
    DistributionFunctionWrapper(PyObject* _fnc): fnc(_fnc)
    {
        Py_INCREF(fnc);
        FILTERMSG(utils::VL_DEBUG, "Agama",
            "Created a C++ df wrapper for Python function " + toString(fnc));
    }
    ~DistributionFunctionWrapper()
    {
        FILTERMSG(utils::VL_DEBUG, "Agama",
            "Deleted a C++ df wrapper for Python function " + toString(fnc));
        Py_DECREF(fnc);
    }
    // non-vectorized form
    virtual void evalDeriv(const actions::Actions &J,
        double *val, df::DerivByActions *der=NULL) const
    {
        evalmany(1, &J, /*separate*/ false, val, der);
    }
    // vectorized form is the one that actually does the work
    virtual void evalmany(const size_t npoints, const actions::Actions J[], bool,
        double values[], df::DerivByActions *deriv=NULL) const
    {
        ALLOC(3*npoints, double, act)
        for(size_t p=0; p<npoints; p++)
            unconvertActions(J[p], act + p*3);
        double mult = conv->massUnit / pow_3(conv->velocityUnit * conv->lengthUnit);
        double mult_der = mult / (conv->velocityUnit * conv->lengthUnit);
        PyAcquireGIL lock;
        bool typeerror  = false;
        npy_intp dims[] = { (npy_intp)npoints, 3};
        PyObject *args  = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, act);
        PyObject *result = NULL, *result_der = NULL;
        if(deriv) {
            args = Py_BuildValue("(N)", args);
            PyObject* kw = PyDict_New();
            PyDict_SetItemString(kw, "der", Py_True);
            PyObject* tup = PyObject_Call(fnc, args, kw);
            Py_DECREF(kw);
            if(tup == NULL) {
                // do nothing here; the error will be reported further down
            } else if(PyTuple_Check(tup) && PyTuple_Size(tup) == 2) {
                result     = PyTuple_GetItem(tup, 0);
                result_der = PyTuple_GetItem(tup, 1);
                Py_INCREF(result);
                Py_INCREF(result_der);
                Py_DECREF(tup);
            } else {
                Py_XDECREF(tup);
                Py_DECREF(args);
                throw std::runtime_error("User-defined distribution function should return a tuple "
                    "of two arrays (DF values and derivatives) when called with 'der=True'");
            }
        } else
            result = PyObject_CallFunctionObjArgs(fnc, args, NULL);
        Py_DECREF(args);

        // parse and unit-convert the returned array of DF values
        if(result == NULL) {
            PyErr_Print();
        } else if(PyArray_Check(result) &&
            PyArray_NDIM((PyArrayObject*)result) == 1 &&
            PyArray_DIM ((PyArrayObject*)result, 0) == (npy_intp)npoints)
        {
            int type = PyArray_TYPE((PyArrayObject*)result);
            for(size_t p=0; p<npoints; p++) {
                switch(type) {
                    case NPY_DOUBLE: values[p] = pyArrayElem<double>(result, p) * mult; break;
                    case NPY_FLOAT:  values[p] = pyArrayElem<float >(result, p) * mult; break;
                    default: values[p] = NAN; typeerror = true;
                }
            }
        }
        else if(npoints==1 && PyNumber_Check(result)) {
            // in case of a single input point, the user function might return a single number
            values[0] = PyFloat_AsDouble(result) * mult;
        }
        else {
            typeerror = true;
        }
        Py_XDECREF(result);

        // if the DF derivatives were requested and returned by the user function, convert them too
        if(deriv) {
            if(result_der && PyArray_Check(result_der) &&
                PyArray_NDIM((PyArrayObject*)result_der) == 2 &&
                PyArray_DIM ((PyArrayObject*)result_der, 0) == (npy_intp)npoints &&
                PyArray_DIM ((PyArrayObject*)result_der, 1) == 3)
            {
                int type = PyArray_TYPE((PyArrayObject*)result_der);
                for(size_t p=0; p<npoints; p++) {
                    switch(type) {
                        case NPY_DOUBLE:
                            deriv[p].dbyJr   = pyArrayElem<double>(result_der, p, 0) * mult_der;
                            deriv[p].dbyJz   = pyArrayElem<double>(result_der, p, 1) * mult_der;
                            deriv[p].dbyJphi = pyArrayElem<double>(result_der, p, 2) * mult_der;
                            break;
                        case NPY_FLOAT:
                            deriv[p].dbyJr   = pyArrayElem<float> (result_der, p, 0) * mult_der;
                            deriv[p].dbyJz   = pyArrayElem<float> (result_der, p, 1) * mult_der;
                            deriv[p].dbyJphi = pyArrayElem<float> (result_der, p, 2) * mult_der;
                            break;
                        default:
                            typeerror = true;
                    }
                }
            } else
                typeerror = true;
        }
        if(result == NULL)
            throw std::runtime_error("Call to user-defined distribution function failed");
        else if(typeerror)
            throw std::runtime_error(
                "Invalid data type returned from user-defined distribution function");
    }
};

/// extract a pointer to C++ DistributionFunction class from a Python object,
/// or return an empty pointer on error
df::PtrDistributionFunction getDistributionFunction(PyObject* df_obj);

/// create a Python DistributionFunction object from an existing instance of C++ DF class
PyObject* createDistributionFunctionObject(df::PtrDistributionFunction df)
{
    DistributionFunctionObject* df_obj =
        PyObject_New(DistributionFunctionObject, DistributionFunctionTypePtr);
    if(!df_obj)
        return NULL;
    // same hack as in 'createDensityObject()'
    new (&(df_obj->df)) df::PtrDistributionFunction;
    df_obj->df = df;
    FILTERMSG(utils::VL_DEBUG, "Agama",
        "Created a Python wrapper for distribution function at " + utils::toString(df.get()));
    return (PyObject*)df_obj;
}

/// attempt to construct an elementary distribution function from the parameters provided in dictionary
df::PtrDistributionFunction DistributionFunction_initFromDict(PyObject* namedArgs)
{
    NamedArgs nargs(namedArgs);
    PyObject* pot_obj  = nargs.pop("potential");  // borrowed reference or NULL
    PyObject* dens_obj = nargs.pop("density");
    // convert other parameters into a KeyValueMap
    utils::KeyValueMap params(nargs);

    // density and potential are needed for some types of DF, but are otherwise unnecessary
    potential::PtrPotential pot;
    if(pot_obj!=NULL) {
        pot = getPotential(pot_obj, &params);
        if(!pot)
            throw std::invalid_argument("Argument 'potential' must be a valid Potential object");
    }
    potential::PtrDensity dens;
    if(dens_obj!=NULL) {
        dens = getDensity(dens_obj, &params);
        if(!dens)
            throw std::invalid_argument("Argument 'density' must be a valid Density object");
    }

    if(!params.contains("type"))
        // this is not necessary, as the DF constructor performs this check anyway,
        // but we can display a more meaningful message here
        throw std::invalid_argument("Should provide the type='...' argument");

    // DF constructor may be calling user-defined Python potential & density functions
    // from multiple threads, so we need to release GIL beforehand
    PyReleaseGIL unlock;
    return df::createDistributionFunction(params, pot.get(), dens.get(), *conv);  // any other DF type
}

/// attempt to construct a composite distribution function from a tuple of DistributionFunction objects
df::PtrDistributionFunction DistributionFunction_initFromTuple(PyObject* tuple)
{
    std::vector<df::PtrDistributionFunction> components;
    for(Py_ssize_t i=0; i<PyTuple_Size(tuple); i++) {
        df::PtrDistributionFunction comp = getDistributionFunction(PyTuple_GET_ITEM(tuple, i));
        if(!comp)
            throw std::invalid_argument("Unnamed arguments should contain "
                "only valid DistributionFunction objects or functions providing that interface");
        components.push_back(comp);
    }
    return components.size()==1 ? components[0] :
        df::PtrDistributionFunction(new df::CompositeDF(components));
}

/// the generic constructor of DistributionFunction object
int DistributionFunction_init(DistributionFunctionObject* self, PyObject* args, PyObject* namedArgs)
{
    if(self->df) {
        PyErr_SetString(PyExc_RuntimeError, "DistributionFunction object cannot be reinitialized");
        return -1;
    }
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
            PyErr_SetString(PyExc_TypeError,
                "Should provide either a list of key=value arguments to create an elementary DF, "
                "or a tuple of existing DistributionFunction objects to create a composite DF");
        }
        if(PyErr_Occurred())
            return -1;
        else if(!self->df) {
            PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to the constructor");
            return -1;
        }
        FILTERMSG(utils::VL_DEBUG, "Agama", "Created a distribution function at "+
            utils::toString(self->df.get()));
        unitsWarning = true;  // any subsequent call to setUnits() will raise a warning
        return 0;
    }
    catch(std::exception& ex) {
        raisePythonException(ex, "Error in creating distribution function: ");
        return -1;
    }
}

/// extract a pointer to C++ DistributionFunction class from a Python object,
/// or return an empty pointer on error
df::PtrDistributionFunction getDistributionFunction(PyObject* df_obj)
{
    if(df_obj == NULL)
        return df::PtrDistributionFunction();

    // check if this is a Python wrapper for a genuine C++ DF object
    if(PyObject_TypeCheck(df_obj, DistributionFunctionTypePtr) && ((DistributionFunctionObject*)df_obj)->df)
        return ((DistributionFunctionObject*)df_obj)->df;

    // otherwise this could be an arbitrary callable Python object
    if(checkCallable(df_obj, /*dimensions of input*/ 3)) {
        // then create a C++ wrapper for this Python function
        return df::PtrDistributionFunction(new DistributionFunctionWrapper(df_obj));
    }

    // otherwise this may be a tuple, list or another sequence of DistributionFunction-like objects
    if(PySequence_Check(df_obj) && !PyString_Check(df_obj)) {
        PyObject* tuple = PySequence_Tuple(df_obj);
        if(tuple) {
            try{
                df::PtrDistributionFunction result = DistributionFunction_initFromTuple(tuple);
                Py_DECREF(tuple);
                return result;
            }
            catch(std::exception &ex) {
                Py_DECREF(tuple);
                FILTERMSG(utils::VL_WARNING, "Agama", ex.what());
            }
        } else {
            PyErr_Clear();
        }
    }

    // none succeeded - return an empty pointer
    return df::PtrDistributionFunction();
}

/// compute the distribution function at one or more points in action space
class FncDistributionFunction: public BatchFunctionVectorized {
    const bool der;
    const df::BaseDistributionFunction& df;
    double* outputBuffer[2];
public:
    FncDistributionFunction(PyObject* input, bool _der, const df::BaseDistributionFunction& _df) :
        BatchFunctionVectorized(input, /*input length*/ 3), der(_der), df(_df)
    {
        outputObject = der?
            allocateOutput<1,3>(numPoints, outputBuffer) :
            allocateOutput<1>  (numPoints, outputBuffer);
    }
    virtual void processManyPoints(npy_intp indexStart, npy_intp indexEnd)
    {
        npy_intp npoints = indexEnd - indexStart;
        ALLOC(npoints, actions::Actions, act)
        for(npy_intp i=0; i<npoints; i++)
            act[i] = convertActions(&inputBuffer[(i + indexStart) * 3]);
        df.evalmany(npoints, act, /*separate*/false, &outputBuffer[0][indexStart],
            der ? (df::DerivByActions*)(&outputBuffer[1][indexStart]) : NULL);
        for(npy_intp indexPoint=indexStart; indexPoint<indexEnd; indexPoint++)
            outputBuffer[0][indexPoint] /=  // DF dimension: M L^-3 V^-3
                conv->massUnit / pow_3(conv->velocityUnit * conv->lengthUnit);
        if(der) {
            for(npy_intp indexPoint=indexStart*3; indexPoint<indexEnd*3; indexPoint++)
                outputBuffer[1][indexPoint] /=  // DF deriv dimension: M L^-4 V^-4
                conv->massUnit / pow_2(pow_2(conv->velocityUnit * conv->lengthUnit));
        }
    }
};

PyObject* DistributionFunction_value(DistributionFunctionObject* self, PyObject* args, PyObject* namedArgs)
{
    if(self->df==NULL) {
        PyErr_SetString(PyExc_RuntimeError, "DistributionFunction object is not properly initialized");
        return NULL;
    }
    PyObject* der_obj = NULL;
    if(namedArgs && (PyDict_Size(namedArgs) != 1 ||
        ((der_obj = PyDict_GetItemString(namedArgs, "der")) == NULL)))
    {
        PyErr_SetString(PyExc_RuntimeError,
            "Distribution function must be called either without named arguments, or with der=True");
        return NULL;
    }
    bool der = der_obj ? PyObject_IsTrue(der_obj) : false;
    return FncDistributionFunction(args, der, *self->df).run(/*chunk*/1024);
}

PyObject* DistributionFunction_totalMass(PyObject* self)
{
    if(((DistributionFunctionObject*)self)->df==NULL) {
        PyErr_SetString(PyExc_RuntimeError, "DistributionFunction object is not properly initialized");
        return NULL;
    }
    try{
        double val = ((DistributionFunctionObject*)self)->df->totalMass();
        return Py_BuildValue("d", val / conv->massUnit);
    }
    catch(std::exception& ex) {
        raisePythonException(ex, "Error in DistributionFunction.totalMass(): ");
        return NULL;
    }
}

PyObject* DistributionFunction_totalEntropy(PyObject* self)
{
    if(((DistributionFunctionObject*)self)->df==NULL) {
        PyErr_SetString(PyExc_RuntimeError, "DistributionFunction object is not properly initialized");
        return NULL;
    }
    try{
        double val = totalEntropy(*((DistributionFunctionObject*)self)->df);
        return Py_BuildValue("d", val / conv->massUnit);
    }
    catch(std::exception& ex) {
        raisePythonException(ex, "Error in DistributionFunction.totalEntropy(): ");
        return NULL;
    }
}

PyObject* DistributionFunction_elem(PyObject* self, Py_ssize_t index)
{
    if(((DistributionFunctionObject*)self)->df==NULL) {
        PyErr_SetString(PyExc_RuntimeError, "DistributionFunction object is not properly initialized");
        return NULL;
    }
    const df::CompositeDF* df_comp =
        dynamic_cast<const df::CompositeDF*>(((DistributionFunctionObject*)self)->df.get());
    if(!df_comp) {
        PyErr_SetString(PyExc_TypeError, "DistributionFunction is not a composite object");
        return NULL;
    }
    if(index<0 || index >= (Py_ssize_t)df_comp->numValues()) {
        PyErr_SetString(PyExc_IndexError, "DistributionFunction component index out of range");
        return NULL;
    }
    return createDistributionFunctionObject(df_comp->component(index));
}

Py_ssize_t DistributionFunction_len(PyObject* self)
{
    const df::CompositeDF* df_comp =
        dynamic_cast<const df::CompositeDF*>(((DistributionFunctionObject*)self)->df.get());
    if(!df_comp)
        return 0;
    return ((DistributionFunctionObject*)self)->df->numValues();
}

/// compare two Python DistributionFunction objects "by value" (check if they represent the same C++ object)
PyObject* DistributionFunction_compare(PyObject* self, PyObject* other, int op)
{
    bool equal = ((DistributionFunctionObject*)self)->df == ((DistributionFunctionObject*)other)->df;
    switch (op) {
        case Py_EQ:
            return PyBool_FromLong(equal);
        case Py_NE:
            return PyBool_FromLong(!equal);
        default:
            PyErr_SetString(PyExc_TypeError, "Invalid comparison");
            return NULL;
    }
}

Py_hash_t DistributionFunction_hash(PyObject *self)
{
    // use the smart pointer to the underlying C++ object, not the Python object itself,
    // to establish identity between two Python objects containing the same C++ class instance
    return Py_HashPointer(const_cast<void*>(static_cast<const void*>
        (((DistributionFunctionObject*)self)->df.get())));
}

static PySequenceMethods DistributionFunction_sequence_methods = {
    DistributionFunction_len, 0, 0, DistributionFunction_elem,
};

static PyMethodDef DistributionFunction_methods[] = {
    { "totalMass", (PyCFunction)DistributionFunction_totalMass, METH_NOARGS,
      "Return the total mass of the model (integral of the distribution function "
      "over the entire phase space of actions)\n"
      "No arguments\n"
      "Returns: float number" },
    { "totalEntropy", (PyCFunction)DistributionFunction_totalEntropy, METH_NOARGS,
      "Return the total entropy of the model (integral of -f*ln(f) "
      "over the entire phase space of actions)\n"
      "No arguments\n"
      "Returns: float number" },
    { NULL }
};

static PyTypeObject DistributionFunctionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "agama.DistributionFunction",
    sizeof(DistributionFunctionObject), 0, (destructor)DistributionFunction_dealloc,
    0, 0, 0, 0, 0, 0, &DistributionFunction_sequence_methods, 0, DistributionFunction_hash,
    (PyCFunctionWithKeywords)DistributionFunction_value, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringDistributionFunction,
    0, 0, DistributionFunction_compare, 0, 0, 0, DistributionFunction_methods, 0, 0, 0, 0, 0, 0, 0,
    (initproc)DistributionFunction_init
};


///@}
//  ----------------------------------
/// \name  SelectionFunction class
//  ----------------------------------
///@{

static const char* docstringSelectionFunction =
    "SelectionFunction class represents an arbitrary function of 6 Cartesian phase-space coordinates "
    "S(x,v) that can be passed to the GalaxyModel class and provides a multiplicative factor "
    "in various integrals computed by its methods.\n"
    "The only currently available variant is a function that depends exponentially on the distance "
    "from a given point x0, normalized by a cutoff radius R0 with a steepness parameter xi: \n"
    "  S(x) = exp[ -(|x-x0| / R0)^xi ]\n"
    "Arguments for the constructor:\n"
    "  point  -- an array of 3 Cartesian coordinates of the point x0;\n"
    "  radius -- cutoff radius R0 (must be positive; infinity means no cutoff - S=1 everywhere;\n"
    "  steepness -- cutoff steepness xi, ranges from 0 (no cutoff) to infinity (default value, "
    "means a sharp transition from S=1 below the cutoff to S=0 above it).\n";

/// \cond INTERNAL_DOCS
typedef shared_ptr<const galaxymodel::BaseSelectionFunction> PtrSelectionFunction;

/// Python type corresponding to SelectionFunction class
typedef struct {
    PyObject_HEAD
    PtrSelectionFunction sf;
} SelectionFunctionObject;
/// \endcond

/// destructor of SelectionFunction class
void SelectionFunction_dealloc(SelectionFunctionObject* self)
{
    FILTERMSG(utils::VL_DEBUG, "Agama", "Deleted a selection function at " +
        utils::toString(self->sf.get()));
    self->sf.reset();
    Py_TYPE(self)->tp_free(self);
}

/// constructor of SelectionFunction class
int SelectionFunction_init(SelectionFunctionObject* self, PyObject* args, PyObject* namedArgs)
{
    if(self->sf) {
        PyErr_SetString(PyExc_RuntimeError, "SelectionFunction object cannot be reinitialized");
        return -1;
    }
    PyObject* point_obj = NULL;
    double radius = NAN, steepness = INFINITY;
    static const char* keywords[] = {"point", "radius", "steepness", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "Od|d", const_cast<char **>(keywords),
        &point_obj, &radius, &steepness))
    {
        return -1;
    }
    std::vector<double> point(toDoubleArray(point_obj));
    if(point.size() != 3) {
        PyErr_SetString(PyExc_RuntimeError, "Error in creating SelectionFunction: "
            "'point' must be an array of 3 numbers");
        return -1;
    }
    try{
        self->sf.reset(new galaxymodel::SelectionFunctionDistance(
            convertPos(&point[0]), radius * conv->lengthUnit, steepness));
        assert(self->sf);
        FILTERMSG(utils::VL_DEBUG, "Agama",
            "Created a SelectionFunction at "+utils::toString(self->sf.get()));
        unitsWarning = true;  // any subsequent call to setUnits() will raise a warning
        return 0;
    }
    catch(std::exception& ex) {
        raisePythonException(ex, "Error in creating SelectionFunction: ");
        return -1;
    }
}

/// compute the selection function at one or more points in 6d phase space (x,v)
class FncSelectionFunction: public BatchFunctionVectorized {
    const galaxymodel::BaseSelectionFunction& sf;
    double* outputBuffer;
public:
    FncSelectionFunction(PyObject* input, const galaxymodel::BaseSelectionFunction& _sf) :
        BatchFunctionVectorized(input, /*input length*/ 6), sf(_sf)
    {
        outputObject = allocateOutput<1>(numPoints, &outputBuffer);
    }
    virtual void processManyPoints(npy_intp indexStart, npy_intp indexEnd)
    {
        npy_intp npoints = indexEnd - indexStart;
        ALLOC(npoints, coord::PosVelCar, points)
        for(npy_intp i=0; i<npoints; i++)
            points[i] = convertPosVel(&inputBuffer[(i + indexStart) * 6]);
        sf.evalmany(npoints, points, &outputBuffer[indexStart]);
    }
};

PyObject* SelectionFunction_value(SelectionFunctionObject* self, PyObject* args, PyObject* namedArgs)
{
    if(self->sf==NULL) {
        PyErr_SetString(PyExc_RuntimeError, "SelectionFunction object is not properly initialized");
        return NULL;
    }
    if(!noNamedArgs(namedArgs))
        return NULL;
    return FncSelectionFunction(args, *self->sf).run(/*chunk*/1024);
}

/// Helper class for providing a BaseSelectionFunction interface to a user-defined Python function
/// that returns the value of a selection function at one or more points in x,v space
class SelectionFunctionWrapper: public galaxymodel::BaseSelectionFunction{
    PyObject* fnc;    ///< Python object providing the selection function
public:
    SelectionFunctionWrapper(PyObject* _fnc): fnc(_fnc)
    {
        Py_INCREF(fnc);
        FILTERMSG(utils::VL_DEBUG, "Agama",
            "Created a C++ selection function wrapper for Python function " + toString(fnc));
    }
    ~SelectionFunctionWrapper()
    {
        FILTERMSG(utils::VL_DEBUG, "Agama",
            "Deleted a C++ selection function wrapper for Python function " + toString(fnc));
        Py_DECREF(fnc);
    }
    virtual double value(const coord::PosVelCar& pv) const
    {
        double val;
        evalmany(1, &pv, &val);
        return val;
    }
    virtual void evalmany(const size_t npoints, const coord::PosVelCar points[], double values[]) const
    {
        ALLOC(6*npoints, double, posvel)
        for(size_t p=0; p<npoints; p++)
            unconvertPosVel(points[p], posvel + p*6);
        PyAcquireGIL lock;
        bool typeerror   = false;
        npy_intp dims[]  = { (npy_intp)npoints, 6};
        PyObject* args   = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, posvel);
        PyObject* result = PyObject_CallFunctionObjArgs(fnc, args, NULL);
        Py_DECREF(args);
        if(result == NULL) {
            PyErr_Print();
        } else if(PyArray_Check(result) &&
            PyArray_NDIM((PyArrayObject*)result) == 1 &&
            PyArray_DIM ((PyArrayObject*)result, 0) == (npy_intp)npoints)
        {
            int type = PyArray_TYPE((PyArrayObject*) result);
            for(size_t p=0; p<npoints; p++) {
                switch(type) {
                    case NPY_DOUBLE: values[p] = pyArrayElem<double>(result, p); break;
                    case NPY_FLOAT:  values[p] = pyArrayElem<float >(result, p); break;
                    case NPY_BOOL:   values[p] = pyArrayElem<bool  >(result, p); break;
                    default: values[p] = NAN; typeerror = true;
                }
            }
        }
        else if(npoints==1 && PyNumber_Check(result)) {
            // in case of a single input point, the user function might return a single number
            values[0] = PyFloat_AsDouble(result);
        }
        else {
            typeerror = true;
        }
        Py_XDECREF(result);
        if(result == NULL)
            throw std::runtime_error("Call to user-defined selection function failed");
        else if(typeerror)
            throw std::runtime_error("Invalid data type returned from user-defined selection function");
        // otherwise return the result in values[]
    }
};

// helper function for creating an instance of C++ BaseSelectionFunction class from a Python object
PtrSelectionFunction getSelectionFunction(PyObject* sf_obj)
{
    if(sf_obj == Py_None || sf_obj == NULL)
        return PtrSelectionFunction(new galaxymodel::SelectionFunctionTrivial());
    else if(PyObject_TypeCheck(sf_obj, SelectionFunctionTypePtr) && ((SelectionFunctionObject*)sf_obj)->sf)
        return ((SelectionFunctionObject*)sf_obj)->sf;
    else // otherwise it must be a callable Python function accessed through a wrapper class
        return PtrSelectionFunction(new SelectionFunctionWrapper(sf_obj));
}

static PyMethodDef SelectionFunction_methods[] = {
    { NULL }  // no named methods
};

static PyTypeObject SelectionFunctionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "agama.SelectionFunction",
    sizeof(SelectionFunctionObject), 0, (destructor)SelectionFunction_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    (PyCFunctionWithKeywords)SelectionFunction_value, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringSelectionFunction,
    0, 0, 0, 0, 0, 0, SelectionFunction_methods, 0, 0, 0, 0, 0, 0, 0,
    (initproc)SelectionFunction_init
};


///@}
//  -------------------------
/// \name  GalaxyModel class
//  -------------------------
///@{

static const char* docstringGalaxyModel =
    "GalaxyModel is a class that takes together a Potential, "
    "a DistributionFunction, and an ActionFinder objects, "
    "and provides methods to compute moments and projections of the distribution function "
    "at a given point in the ordinary phase space (coordinate/velocity), as well as "
    "methods for drawing samples from the distribution function in the given potential.\n"
    "The constructor takes the following arguments:\n"
    "  potential - a Potential object.\n"
    "  df - a DistributionFunction object.\n"
    "  af (optional) - an ActionFinder object - must be constructed for the same potential; "
    "if not provided, then the action finder is created internally.\n"
    "  sf (optional) - a SelectionFunction object or a user-defined callable function "
    "that takes a 2d Nx6 array of phase-space points (x,v in cartesian coordinates) as input, "
    "and returns a 1d array of N values between 0 and 1, which will be multiplied by the values "
    "of the DF at corresponding points; if not provided, assumed identically unity.\n"
    "In case of a multicomponent DF, one may compute the moments and projections for each "
    "component separately by providing an optional flag 'separate=True' to the corresponding "
    "methods. This is more efficient than constructing a separate GalaxyModel instance for "
    "each DF component and computing its moments, because the most expensive operation - "
    "conversion between position/velocity and action space - is performed once for all components. "
    "If this flag is not set (default), all components are summed up.";

/// \cond INTERNAL_DOCS
/// Python type corresponding to GalaxyModel class
typedef struct {
    PyObject_HEAD
    PotentialObject* pot_obj;
    DistributionFunctionObject* df_obj;
    ActionFinderObject* af_obj;
    PyObject* sf_obj;
} GalaxyModelObject;
/// \endcond

void GalaxyModel_dealloc(GalaxyModelObject* self)
{
    Py_XDECREF(self->pot_obj);
    Py_XDECREF(self->df_obj);
    Py_XDECREF(self->af_obj);
    Py_XDECREF(self->sf_obj);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

bool GalaxyModel_isCorrect(GalaxyModelObject* self)
{
    if(self==NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Should be called as method of GalaxyModel object");
        return false;
    }
    if( !self->pot_obj|| !self->pot_obj->pot ||
        !self->af_obj || !self->af_obj->af ||
        !self->df_obj || !self->df_obj->df ||
        !self->sf_obj)
    {
        PyErr_SetString(PyExc_RuntimeError, "GalaxyModel is not properly initialized");
        return false;
    }
    return true;
}

int GalaxyModel_init(GalaxyModelObject* self, PyObject* args, PyObject* namedArgs)
{
    if(self->pot_obj || self->df_obj) {
        PyErr_SetString(PyExc_RuntimeError, "GalaxyModel object cannot be reinitialized");
        return -1;
    }
    static const char* keywords[] = {"potential", "df", "af", "sf", NULL};
    PyObject *pot_obj = NULL, *df_obj = NULL, *af_obj = NULL, *sf_obj = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "OO|OO", const_cast<char**>(keywords),
        &pot_obj, &df_obj, &af_obj, &sf_obj))
    {
        return -1;
    }

    // check and store the potential
    if(!getPotential(pot_obj)) {
        PyErr_SetString(PyExc_TypeError, "Argument 'potential' must be a valid Potential object");
        return -1;
    }
    Py_INCREF(pot_obj);
    self->pot_obj = (PotentialObject*)pot_obj;

    // check and store the DF
    df::PtrDistributionFunction df = getDistributionFunction(df_obj);
    if(!df) {
        PyErr_SetString(PyExc_TypeError, "Argument 'df' must be a valid DistributionFunction object");
        return -1;
    }
    if(PyObject_TypeCheck(df_obj, DistributionFunctionTypePtr))
    {   // it is a true Python DF object
        Py_INCREF(df_obj);
        self->df_obj = (DistributionFunctionObject*)df_obj;
    } else {
        // it is a Python function that was wrapped in a C++ class,
        // which now in turn will be wrapped in a new Python DF object (isn't that beautiful?)
        self->df_obj = (DistributionFunctionObject*)createDistributionFunctionObject(df);
    }

    // af_obj might be NULL (then create a new one); if not NULL then check its validity
    // (however there is no way to ensure that the action finder corresponds to the potential!)
    if(af_obj!=NULL && (!PyObject_TypeCheck(af_obj, ActionFinderTypePtr) ||
       ((ActionFinderObject*)af_obj)->af==NULL))
    {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'af' must be a valid instance of ActionFinder class "
            "corresponding to the given potential");
        return -1;
    }
    if(af_obj==NULL) {  // no action finder provided - create one internally
        PyObject *args = Py_BuildValue("(O)", pot_obj);
        self->af_obj = (ActionFinderObject*)PyObject_CallObject((PyObject*)ActionFinderTypePtr, args);
        Py_DECREF(args);
        if(!self->af_obj)
            return -1;
    } else {  // use an existing action finder and increase its refcount
        Py_INCREF(af_obj);
        self->af_obj = (ActionFinderObject*)af_obj;
    }

    // sf_obj, if provided, must be a callable object OR an instance of a SelectionFunction class
    if(sf_obj) {
        if( !PyObject_TypeCheck(sf_obj, SelectionFunctionTypePtr) &&
            !checkCallable(sf_obj, /*input dim*/ 6))
        {
            PyErr_SetString(PyExc_TypeError,
                "Argument 'sf' must be either an instance of SelectionFunction class, or "
                "a callable function that takes an Nx6 array as input and returns an array of N numbers");
            return -1;
        }
    } else {
        // if not provided, replace by None
        sf_obj = Py_None;
    }
    Py_INCREF(sf_obj);  // increase refcount regardless of whether it's a user-provided function or None
    self->sf_obj = sf_obj;

    assert(GalaxyModel_isCorrect(self));
    return 0;
}

/// compute the total mass within the selection region
PyObject* GalaxyModel_totalMass(GalaxyModelObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!GalaxyModel_isCorrect(self))
        return NULL;
    static const char* keywords[] = {"separate", NULL};
    PyObject *separate_flag = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "|O", const_cast<char**>(keywords),
        &separate_flag))
        return NULL;
    bool separate = toBool(separate_flag, false);
    PtrSelectionFunction selFunc(getSelectionFunction(self->sf_obj));
    const galaxymodel::GalaxyModel model(
        *self->pot_obj->pot, *self->af_obj->af, *self->df_obj->df, *selFunc);
    int numVal = separate? model.distrFunc.numValues() : 1;
    std::vector<double> val(numVal);
    try{
        // this routine receives pointers to potential, DF and SF possibly containing user-defined
        // Python functions, and although it is not OpenMP-parallelized, we release GIL just in case..
        {
            PyReleaseGIL unlock;
            computeTotalMass(model, &val[0], separate);
            math::blas_dmul(1./conv->massUnit, val);
        }
        return separate ? toPyArray(val) : Py_BuildValue("d", val[0]);
    }
    catch(std::exception& ex) {
        raisePythonException(ex, "Error in totalMass(): ");
        return NULL;
    }
}

/// generate samples in position/velocity space
PyObject* GalaxyModel_sample_posvel(GalaxyModelObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!GalaxyModel_isCorrect(self))
        return NULL;
    static const char* keywords[] = {"n", "method", NULL};
    Py_ssize_t numPoints = 0;
    math::SampleMethod method = math::SM_DEFAULT;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "n|b", const_cast<char**>(keywords),
        &numPoints, &method) || numPoints<=0)
        return NULL;
    try{
        particles::ParticleArrayCar points;
        // temporary wrapper object for the selection function (or a trivial one if not provided)
        PtrSelectionFunction selfnc = getSelectionFunction(self->sf_obj);

        // do the sampling while releasing GIL, since the sampling routine is OpenMP-parallelized
        // and may call back user-defined Python functions from multiple threads simultaneously,
        // which would then re-acquire GIL in their own respective threads
        {
            PyReleaseGIL unlock;
            points = galaxymodel::samplePosVel(galaxymodel::GalaxyModel(
                *self->pot_obj->pot, *self->af_obj->af, *self->df_obj->df, *selfnc),
                numPoints, method);
        }

        // remaining operations use Python C API and thus should be performed under GIL.
        // convert output to NumPy array; its size may be different from numPoints, depending on method
        npy_intp dims[] = {static_cast<npy_intp>(points.size()), 6};
        PyObject* posvel_arr = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        PyObject* mass_arr   = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        for(npy_intp i=0; i<dims[0]; i++) {
            unconvertPosVel(points.point(i), &pyArrayElem<double>(posvel_arr, i, 0));
            pyArrayElem<double>(mass_arr, i) = points.mass(i) / conv->massUnit;
        }
        return Py_BuildValue("NN", posvel_arr, mass_arr);
    }
    catch(std::exception& ex) {
        raisePythonException(ex, "Error in sample(): ");
        return NULL;
    }
}

/// compute moments of DF at a given 2d or 3d point
class FncGalaxyModelMoments: public BatchFunction {
    PtrSelectionFunction selFunc;                 // selection function
    const galaxymodel::GalaxyModel model;         // potential + df + action finder + sel.fnc.
    bool separate;                                // whether to consider each DF component separately
    unsigned int numComponents;                   // df.numValues() if separate, otherwise 1
    std::vector<double> alpha, beta, gamma;       // conversion between observed and intrinsic coords
    double *outputDens, *outputVel, *outputVel2;  // raw buffers for output values
public:
    FncGalaxyModelMoments(PyObject* input, PyObject* namedArgs, GalaxyModelObject* model_obj) :
        BatchFunction(input, /*inputLength - two possible choices*/ 2, 3,
            /*custom error message*/ "Input should be a 2d/3d point or an array of points"),
        selFunc(getSelectionFunction(model_obj->sf_obj)),
        model(*model_obj->pot_obj->pot, *model_obj->af_obj->af, *model_obj->df_obj->df, *selFunc),
        outputDens(NULL), outputVel(NULL), outputVel2(NULL)
    {
        NamedArgs nargs(namedArgs);
        bool needDens = toBool(nargs.pop("dens"), true);
        bool needVel  = toBool(nargs.pop("vel" ), false);
        bool needVel2 = toBool(nargs.pop("vel2"), true);
        separate      = toBool(nargs.pop("separate"), false);
        numComponents = separate ? model.distrFunc.numValues() : 1;
        alpha = nargs.popArray("alpha", numPoints);
        beta  = nargs.popArray("beta" , numPoints);
        gamma = nargs.popArray("gamma", numPoints);
        if(PyErr_Occurred() || !nargs.empty())
            return;  // e.g., an error in parsing the angles - keep outputObject=NULL
        assert(!alpha.empty() && !beta.empty() && !gamma.empty());
        double* outputBuffers[3];
        int numVal = separate? model.distrFunc.numValues() : 0;
        if(needDens) {
            if(needVel) {
                if(needVel2) {
                    outputObject = allocateOutput<1, 3, 6>(numPoints, outputBuffers, numVal);
                    outputVel2   = outputBuffers[2];
                } else {
                    outputObject = allocateOutput<1, 3   >(numPoints, outputBuffers, numVal);
                }
                outputVel = outputBuffers[1];
            } else {  // no needVel
                if(needVel2) {
                    outputObject = allocateOutput<1,    6>(numPoints, outputBuffers, numVal);
                    outputVel2   = outputBuffers[1];
                } else {
                    outputObject = allocateOutput<1      >(numPoints, outputBuffers, numVal);
                }
            }
            outputDens = outputBuffers[0];
        } else {  // no needDens
            if(needVel) {
                if(needVel2) {
                    outputObject = allocateOutput<   3, 6>(numPoints, outputBuffers, numVal);
                    outputVel2   = outputBuffers[1];
                } else {
                    outputObject = allocateOutput<   3   >(numPoints, outputBuffers, numVal);
                }
                outputVel = outputBuffers[0];
            } else {  // no needVel
                if(needVel2) {
                    outputObject = allocateOutput<      6>(numPoints, outputBuffers, numVal);
                    outputVel2   = outputBuffers[0];
                } else {
                    PyErr_SetString(PyExc_RuntimeError, "Nothing to compute!");
                }
            }
        }
    }

    virtual void processPoint(npy_intp ip /*point index*/)
    {
        double *dens = outputDens ? &outputDens[ip * numComponents    ] : NULL;
        double *vel  = outputVel  ? &outputVel [ip * numComponents * 3] : NULL;
        double *vel2 = outputVel2 ? &outputVel2[ip * numComponents * 6] : NULL;
        coord::Orientation orientation(
            alpha[ip % alpha.size()],
            beta [ip % beta .size()],
            gamma[ip % gamma.size()]);
        try{
            if(inputLength==2)  // projected
                computeMoments(model, coord::PosProj(
                    inputBuffer[ip * inputLength    ] * conv->lengthUnit,
                    inputBuffer[ip * inputLength + 1] * conv->lengthUnit),
                    dens, (coord::VelCar*)vel, (coord::Vel2Car*)vel2,
                    separate, orientation);
            else // inputLength==3
                computeMoments(model, convertPos(&inputBuffer[ip * inputLength]),
                    dens, (coord::VelCar*)vel, (coord::Vel2Car*)vel2,
                    separate, orientation);
            // convert units in the output arrays
            for(unsigned int ic=0; ic<numComponents; ic++) {
                if(dens)
                    dens[ic] /= conv->massUnit / math::pow(conv->lengthUnit, inputLength);
                if(vel)
                    for(int d=0; d<3; d++)
                        vel[ic * 3 + d] /= conv->velocityUnit;
                if(vel2)
                    for(int d=0; d<6; d++)
                        vel2[ic * 6 + d] /= pow_2(conv->velocityUnit);
            }
        }
        catch(std::exception& ex) {
            if(dens)
                std::fill(dens, dens + numComponents, NAN);
            if(vel)
                std::fill(vel,  vel  + numComponents * 3, NAN);
            if(vel2)
                std::fill(vel2, vel2 + numComponents * 6, NAN);
            PyAcquireGIL lock;  // need to get hold of GIL to issue the following warning
            PyErr_WarnEx(NULL, (std::string("GalaxyModel.moments: ")+ex.what()).c_str(), 1);
        }
    }
};

PyObject* GalaxyModel_moments(GalaxyModelObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!GalaxyModel_isCorrect(self))
        return NULL;
    return FncGalaxyModelMoments(args, namedArgs, self).run(/*chunk*/1);
}

/// compute projected distribution function
class FncGalaxyModelProjectedDF: public BatchFunction {
    PtrSelectionFunction selFunc;            // selection function
    const galaxymodel::GalaxyModel model;    // potential + df + action finder + sel.fnc.
    bool separate;                           // whether to consider each DF component separately
    unsigned int numComponents;              // df.numValues() if separate, otherwise 1
    std::vector<double> alpha, beta, gamma;  // conversion between observed and intrinsic coords
    double* outputBuffer;                    // raw buffer for output values
public:
    FncGalaxyModelProjectedDF(PyObject* input, PyObject* namedArgs, GalaxyModelObject* model_obj) :
        BatchFunction(input, /*inputLength*/ 8, 8,
            /*custom error message*/ "Input should be a point or an array of points "
            "with 8 numbers per point: X, Y, vX, vY, vZ, vX_error, vY_error, vZ_error"),
        selFunc(getSelectionFunction(model_obj->sf_obj)),
        model(*model_obj->pot_obj->pot, *model_obj->af_obj->af, *model_obj->df_obj->df, *selFunc)
    {
        NamedArgs nargs(namedArgs);
        separate      = toBool(nargs.pop("separate"), false);
        numComponents = separate ? model.distrFunc.numValues() : 1;
        alpha = nargs.popArray("alpha", numPoints);
        beta  = nargs.popArray("beta" , numPoints);
        gamma = nargs.popArray("gamma", numPoints);
        if(!PyErr_Occurred() && nargs.empty()) {
            outputObject = allocateOutput<1>(
                numPoints, &outputBuffer, separate? model.distrFunc.numValues() : 0);
            assert(!alpha.empty() && !beta.empty() && !gamma.empty());
        }
    }

    virtual void processPoint(npy_intp ip /*point index*/)
    {
        try{
            computeProjectedDF(model, coord::PosProj(
                inputBuffer[ip*8  ] * conv->lengthUnit,
                inputBuffer[ip*8+1] * conv->lengthUnit),
                /*vel*/    convertVel(&inputBuffer[ip*8+2]),
                /*velerr*/ convertVel(&inputBuffer[ip*8+5]),
                /*output*/ &outputBuffer[ip * numComponents],
                separate,
                coord::Orientation(
                    alpha[ip % alpha.size()],
                    beta [ip % beta .size()],
                    gamma[ip % gamma.size()]));
            // convert units in the output array:
            // the dimension of the result is surface density divided by velocity to the power K,
            // where K is the number of velocity components with finite uncertainty
            double dim = conv->massUnit / pow_2(conv->lengthUnit);
            for(unsigned int d=0; d<3; d++)
                if(isFinite(inputBuffer[ip*8+5+d]))
                    dim /= conv->velocityUnit;
            for(unsigned int ic=0; ic<numComponents; ic++)
                outputBuffer[ip * numComponents + ic] /= dim;
        }
        catch(std::exception& ex) {
            for(unsigned int ic=0; ic<numComponents; ic++)
                outputBuffer[ip * numComponents + ic] = NAN;
            PyAcquireGIL lock;  // need to get hold of GIL to issue the following warning
            PyErr_WarnEx(NULL, (std::string("GalaxyModel.projectedDF: ")+ex.what()).c_str(), 1);
        }
    }
};

PyObject* GalaxyModel_projectedDF(GalaxyModelObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!GalaxyModel_isCorrect(self))
        return NULL;
    return FncGalaxyModelProjectedDF(args, namedArgs, self).run(/*chunk*/1);
}

/// compute velocity distribution functions at point(s)
class FncGalaxyModelVDF: public BatchFunction {
    static const int SIZEGRIDV = 50;      // default size of the velocity grid
    PtrSelectionFunction selFunc;         // selection function
    const galaxymodel::GalaxyModel model; // potential + df + action finder + sel.fnc.
    bool separate;                        // whether to consider each DF component separately
    unsigned int numComponents;           // df.numValues() if separate, otherwise 1
    int sizegridv;                        // default or user-provided size of velocity grid
    std::vector<double> usergridv;        // user-provided velocity grid (overrides sizegridv if set)
    std::vector<double> alpha, beta, gamma; // conversion between observed and intrinsic coords
    PyObject **splvX, **splvY, **splvZ;   // raw buffers of the output arrays to store spline objects
    double* outputDensity;                // raw buffer of the output array to store density (if needed)
public:
    FncGalaxyModelVDF(PyObject* input, PyObject* namedArgs, GalaxyModelObject* model_obj) :
        BatchFunction(input, /*inputLength - two possible choices*/ 2, 3,
            /*custom error message*/ "Input should be a 2d/3d point or an array of points"),
        selFunc(getSelectionFunction(model_obj->sf_obj)),
        model(*model_obj->pot_obj->pot, *model_obj->af_obj->af, *model_obj->df_obj->df, *selFunc),
        sizegridv(SIZEGRIDV),
        splvX(NULL), splvY(NULL), splvZ(NULL), outputDensity(NULL)
    {
        NamedArgs nargs(namedArgs);
        PyObject* gridv_obj =  nargs.pop("gridv");
        bool needDens = toBool(nargs.pop("dens"), false);
        separate      = toBool(nargs.pop("separate"), false);
        numComponents = separate ? model.distrFunc.numValues() : 1;
        alpha = nargs.popArray("alpha", numPoints);
        beta  = nargs.popArray("beta" , numPoints);
        gamma = nargs.popArray("gamma", numPoints);
        if(PyErr_Occurred() || !nargs.empty())
            return;  // e.g., an error in parsing the angles - keep outputObject=NULL
        assert(!alpha.empty() && !beta.empty() && !gamma.empty());
        // the "gridv" argument, if provided, may be either the number of nodes in the array
        // or the array itself
        if(gridv_obj) {
            sizegridv = toInt(gridv_obj, SIZEGRIDV);
            usergridv = toDoubleArray(gridv_obj);
            if(sizegridv<2 || (!usergridv.empty() && usergridv.size()<2)) {
                PyErr_SetString(PyExc_TypeError, "argument 'gridv', if provided, must be either "
                    "a number >= 2 or an array of at least that length");
                return;
            }
            math::blas_dmul(conv->velocityUnit, usergridv);  // does nothing if gridv was not provided
        }

        // output is a tuple of 3 or 4 (if needDens==true) Python objects:
        // either individual spline objects, or 1d/2d arrays of such objects.
        if(numPoints==-1 && !separate) {
            // one input point, no separate output for DF components:
            // temporarily initialize the tuple with 3(4) Nones, will be replaced in processPoint()
            outputObject = needDens ?
                Py_BuildValue("OOOd", Py_None, Py_None, Py_None, NAN) :
                Py_BuildValue("OOO",  Py_None, Py_None, Py_None);
            // HACK: assign the pointers to output arrays to the elements of the tuple;
            // in processPoint(), these pointers (currently assigned to Py_None)
            // will be replaced with newly created Python spline objects
            splvX = (PyObject**)&((PyTupleObject*)outputObject)->ob_item;  // 0th element
            splvY = splvX+1;  // next (1st) element
            splvZ = splvY+1;  // next (2nd) element
            if(needDens)
                // get the address of the floating-point value in the last (3rd) tuple element
                outputDensity = &((PyFloatObject*)*(splvZ+1))->ob_fval;
        } else {
            npy_intp ndims, dims[2];
            if(numPoints==-1 && separate) {
                ndims = 1;
                dims[0] = model.distrFunc.numValues();
            } else if(numPoints>=0 && !separate) {
                ndims = 1;
                dims[0] = numPoints;
            } else if(numPoints>=0 &&  separate) {
                ndims = 2;
                dims[0] = numPoints;
                dims[1] = model.distrFunc.numValues();
            } else {
                // the only remaining possibility is numPoints<-1, indicating an error in parsing input
                return;
            }
            // create the 1d or 2d arrays of would-be spline objects and a float array of density
            PyObject* arrvX = PyArray_SimpleNew(ndims, dims, NPY_OBJECT);
            PyObject* arrvY = PyArray_SimpleNew(ndims, dims, NPY_OBJECT);
            PyObject* arrvZ = PyArray_SimpleNew(ndims, dims, NPY_OBJECT);
            PyObject* arrdens = needDens? PyArray_SimpleNew(ndims, dims, NPY_DOUBLE) : NULL;
            if(!arrvX || !arrvY || !arrvZ || (needDens && !arrdens)) {
                Py_XDECREF(arrvX);
                Py_XDECREF(arrvY);
                Py_XDECREF(arrvZ);
                Py_XDECREF(arrdens);
                return;
            }
            // the returned value will be a tuple of 3 or 4 arrays
            outputObject = needDens ?
                Py_BuildValue("NNNN", arrvX, arrvY, arrvZ, arrdens) :
                Py_BuildValue("NNN",  arrvX, arrvY, arrvZ);
            // obtain raw buffers for the arrays of objects
            splvX = static_cast<PyObject**>(PyArray_DATA((PyArrayObject*)arrvX));
            splvY = static_cast<PyObject**>(PyArray_DATA((PyArrayObject*)arrvY));
            splvZ = static_cast<PyObject**>(PyArray_DATA((PyArrayObject*)arrvZ));
            if(needDens)
                outputDensity = static_cast<double*>(PyArray_DATA((PyArrayObject*)arrdens));
            // initialize the arrays with Nones, will be replaced in processPoint()
            npy_intp size = (numPoints==-1 ? 1 : numPoints) * numComponents;
            for(npy_intp ind=0; ind<size; ind++) {
                splvX[ind] = Py_None;  Py_INCREF(Py_None);
                splvY[ind] = Py_None;  Py_INCREF(Py_None);
                splvZ[ind] = Py_None;  Py_INCREF(Py_None);
            }
        }
    }

    virtual void processPoint(npy_intp ip /*point index*/)
    {
        // output storage
        std::vector<double> density(numComponents);
        std::vector< std::vector<double> >
            amplvX(numComponents), amplvY(numComponents), amplvZ(numComponents);
        coord::Orientation orientation(
            alpha[ip % alpha.size()],
            beta [ip % beta .size()],
            gamma[ip % gamma.size()]);
        try{
            // compute the distributions
            const int ORDER = 3;   // degree of VDF (cubic spline)
            std::vector<double> gridv(usergridv);
            if(gridv.empty()) {
                // create a default grid in velocity space (if not provided by the user)
                if(inputLength==2) // projected
                    galaxymodel::computeVelocityDistribution<ORDER>(model, coord::PosProj(
                        inputBuffer[ip * inputLength    ] * conv->lengthUnit,
                        inputBuffer[ip * inputLength + 1] * conv->lengthUnit),
                        sizegridv, /*output*/ gridv,
                        /*output*/ &density[0], &amplvX[0], &amplvY[0], &amplvZ[0],
                        /*other params*/ separate, orientation);
                else  // inputLength==3
                    galaxymodel::computeVelocityDistribution<ORDER>(model,
                        convertPos(&inputBuffer[ip * inputLength]),
                        sizegridv, /*output*/ gridv,
                        /*output*/ &density[0], &amplvX[0], &amplvY[0], &amplvZ[0],
                        /*other params*/ separate, orientation);
            } else { // use the provided velocity grid
                if(inputLength==2)
                    galaxymodel::computeVelocityDistribution<ORDER>(model, coord::PosProj(
                        inputBuffer[ip * inputLength    ] * conv->lengthUnit,
                        inputBuffer[ip * inputLength + 1] * conv->lengthUnit),
                        gridv, gridv, gridv,
                        /*output*/ &density[0], &amplvX[0], &amplvY[0], &amplvZ[0],
                        /*other params*/ separate, orientation);
                else  // inputLength==3
                    galaxymodel::computeVelocityDistribution<ORDER>(model,
                        convertPos(&inputBuffer[ip * inputLength]),
                        gridv, gridv, gridv,
                        /*output*/ &density[0], &amplvX[0], &amplvY[0], &amplvZ[0],
                        /*other params*/ separate, orientation);
            }

            // convert the units for the abscissae (velocity)
            math::blas_dmul(1/conv->velocityUnit, gridv);

            // store the density (if required) while converting its units
            if(outputDensity) {
                math::blas_dmul(math::pow(conv->lengthUnit, inputLength) / conv->massUnit, density);
                std::copy(density.begin(), density.end(), &outputDensity[ip * numComponents]);
            }

            // create and store Python spline objects in the output arrays
            // (protect from concurrent access to Python API from multiple threads)
            {
                PyAcquireGIL lock;
                for(unsigned int ic=0; ic<numComponents; ic++) {
                    // convert the units for the ordinates (f(v) ~ 1/velocity)
                    math::blas_dmul(conv->velocityUnit, amplvX[ic]);
                    math::blas_dmul(conv->velocityUnit, amplvY[ic]);
                    math::blas_dmul(conv->velocityUnit, amplvZ[ic]);
                    // release the elements of output arrays (Py_None objects)
                    Py_XDECREF(splvX[ip * numComponents + ic]);
                    Py_XDECREF(splvY[ip * numComponents + ic]);
                    Py_XDECREF(splvZ[ip * numComponents + ic]);
                    // and replace them with newly created spline objects
                    splvX[ip * numComponents + ic] = createCubicSpline(gridv, amplvX[ic]);
                    splvY[ip * numComponents + ic] = createCubicSpline(gridv, amplvY[ic]);
                    splvZ[ip * numComponents + ic] = createCubicSpline(gridv, amplvZ[ic]);
                }
            }
        }
        catch(std::exception& ex) {
            // leave PyNone as the elements of output arrays and issue a warning
            PyAcquireGIL lock;  // need to get hold of GIL before calling this function
            PyErr_WarnEx(NULL, (std::string("GalaxyModel.vdf: ")+ex.what()).c_str(), 1);
        }
    }
};

PyObject* GalaxyModel_vdf(GalaxyModelObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!GalaxyModel_isCorrect(self))
        return NULL;
    return FncGalaxyModelVDF(args, namedArgs, self).run(/*chunk*/1);
}

static PyMemberDef GalaxyModel_members[] = {
    { const_cast<char*>("potential"), T_OBJECT_EX, offsetof(GalaxyModelObject, pot_obj), READONLY,
      const_cast<char*>("Potential (read-only)") },
    { const_cast<char*>("af"),  T_OBJECT_EX, offsetof(GalaxyModelObject, af_obj ), READONLY,
      const_cast<char*>("Action finder (read-only)") },
    { const_cast<char*>("df"),  T_OBJECT_EX, offsetof(GalaxyModelObject, df_obj ), READONLY,
      const_cast<char*>("Distribution function (read-only)") },
    { const_cast<char*>("sf"),  T_OBJECT_EX, offsetof(GalaxyModelObject, sf_obj ), READONLY,
      const_cast<char*>("Selection function (read-only)") },
    { NULL }
};

#define DOCSTRING_SEPARATE \
    "  separate (boolean, default False) -- " \
    "whether to treat each element of a multicomponent DF separately: if set, the output arrays " \
    "will have one extra dimension of size equal to the number of DF components Ncomp (possibly 1).\n"
#define DOCSTRING_ANGLES \
    "  alpha, beta, gamma -- three Euler angles specifying the orientation of the 'observed' XYZ " \
    "cordinate system with respect to the 'intrinsic' model coordinates xyz " \
    "(by default they are all zero, meaning that the two systems coincide); " \
    "see the illustration in the appendix of the Agama reference documentation; " \
    "in particular, beta is the inclination angle. " \
    "Angles can be single numbers or arrays of the same length as the number of points.\n"

static PyMethodDef GalaxyModel_methods[] = {
    { "totalMass", (PyCFunction)GalaxyModel_totalMass, METH_VARARGS | METH_KEYWORDS,
      "Compute the total mass of the distribution function.\n"
      "Arguments:\n"
      DOCSTRING_SEPARATE
      "Returns:\n"
      "  Integral of the DF multiplied by selection function over the entire space "
      "allowed by the latter; if the selection function is trivial (identically 1), "
      "the result should match df.totalMass() up to integration errors, but is much more "
      "expensive to compute, since the integration is carried over the 6d position/velocity "
      "space rather than the 3d action space. The sum of masses returned by the sample() method "
      "should also be equal to totalMass up to integration errors. "
      "If separate is True, the return value is an array of length Ncomp." },
    { "sample", (PyCFunction)GalaxyModel_sample_posvel, METH_VARARGS | METH_KEYWORDS,
      "Sample distribution function in the given potential by N particles.\n"
      "Arguments:\n"
      "  n -- number of particles to sample.\n"
      "  method (optional, default 0) -- choice of method; "
      "see the docstring of sampleNdim for description.\n"
      "Returns:\n"
      "  A tuple of two arrays: position/velocity (2d array of size Nx6) "
      "and mass (1d array of length N)." },
    { "moments", (PyCFunction)GalaxyModel_moments, METH_VARARGS | METH_KEYWORDS,
      "Compute moments or projected moments of distribution function in the given potential.\n"
      "Positional argument:\n"
      "  point -- a single point (X,Y,Z) in case of intrinsic moments or (X,Y) in case of projected "
      "moments or an array of shape (Npoints, 2 or 3) with the Cartesian coordinates "
      "in the 'observed' coordinate system XYZ, which may be arbitrarily oriented with respect to "
      "the 'intrinsic' coordinate system xyz of the model. "
      "The projected moments are additionally integrated along the line of sight Z.\n"
      "Keyword arguments:\n"
      "  dens (boolean, default True)  -- flag telling whether the density (0th moment) "
      "needs to be computed.\n"
      "  vel  (boolean, default False) -- same for streaming velocity (1st moment).\n"
      "  vel2 (boolean, default True)  -- same for 2nd moment of velocity.\n"
      DOCSTRING_SEPARATE
      DOCSTRING_ANGLES
      "Returns:\n"
      "  For each input point, return the requested moments (one value for density, three for "
      "mean velocity, and 6 components of the 2nd moment tensor: XX, YY, ZZ, XY, XZ, YZ). "
      "The shapes of output arrays are { Npoints, (Npoints, 3), (Npoints, 6) } if separate==False, "
      "or { (Npoints, Ncomp), (Npoints, Ncomp, 3), (Npoints, Ncomp, 6) } if separate==True.\n" },
    { "projectedDF", (PyCFunction)GalaxyModel_projectedDF, METH_VARARGS | METH_KEYWORDS,
      "Compute the projected distribution function "
      "(integrated over the Z coordinate and optionally some velocity components).\n"
      "The input point is given by X,Y coordinates in the 'observed' coordinate system, "
      "whose orientation with respect to the 'intrinsic' coordinate system of the model "
      "is specified by three Euler rotation angles.\n"
      "The three velocity components in the observed system - vX,vY,vZ - may be known precisely, "
      "or with some uncertainty, which may be even infinite. In the case of nonzero uncertainty, "
      "the DF is integrated over the corresponding velocity component, weighted with a Gaussian "
      "function centered on the given value and with the given width. When the uncertainty is "
      "infinite, the integration is carried over the entire available velocity range without "
      "any weight function (and in this case, the provided value of the velocity is ignored).\n"
      "The case of a finite but very large uncertainty is almost equivalent to the infinite "
      "uncertainty, but differs in normalization: if vZerr >> vZ, \n"
      "projectedDF([..., vZ, ..., vZerr]) = \n"
      "projectedDF([...,  0, ...,  inf ]) * exp( -0.5 * (vZ/vZerr)**2 ) / sqrt(2*pi) / vZerr .\n"
      "The uncertainties may be specified independently for each component, with the only "
      "restriction that the vX,vY uncertainties may be either both finite or both infinite.\n"
      "When all three uncertainties are infinite, the result is equivalent to projected density.\n"
      "Positional argument:\n"
      "  point -- a single point (8 numbers) or an array of shape (Npoints, 8) containing "
      "the X,Y components of position in the 'observed' cartesian coordinate system, "
      "vX,vY,vZ components of velocity, and the corresponding velocity uncertainties, "
      "which may range from zero to infinity including boundaries.\n"
      "Keyword arguments:\n"
      DOCSTRING_SEPARATE
      DOCSTRING_ANGLES
      "Returns:\n"
      "  The value of projected DF at each point; "
      "if separate is True, an array of shape (Npoints, Ncomp)." },
    { "vdf", (PyCFunction)GalaxyModel_vdf, METH_VARARGS | METH_KEYWORDS,
      "Compute the velocity distribution functions in three directions at one or several "
      "points in 3d (X,Y,Z), or projected velocity distributions at the given 2d points (X,Y), "
      "integrated over Z, where the 'observed' coordinate system XYZ may be arbitrarily oriented "
      "relative to the 'intrinsic' coordinate system xyz of the model.\n"
      "Positional argument:\n"
      "  point -- a single point (X,Y,Z) in case of intrinsic VDF or (X,Y) in case of projected VDF, "
      "or an array of shape (Npoints, 2 or 3) with the positions in cartesian coordinates.\n"
      "Keyword arguments:\n"
      "  gridv -- (optional, default 50) the size of the grid in the velocity space, or an array "
      "specifying the grid itself (should be monotonically increasing and have at least 2 elements). "
      "If given as a number or left as default, the grid will span the range +- escape velocity, "
      "computed separately for each point, otherwise a single user-provided grid will be used for "
      "all points. All three velocity components use the same grid.\n"
      "  dens -- (optional, default False) if this flag is set, the output will also contain "
      "the density at each input point, which comes for free during computations.\n"
      DOCSTRING_SEPARATE
      DOCSTRING_ANGLES
      "Returns:\n"
      "  A tuple of length 3 (if dens==False) or 4 otherwise. \n"
      "The first three elements are functions (in case of one input point and separate==False), "
      "or arrays of functions with length Npoints or shape (Npoints, Ncomp), which represent "
      "spline-interpolated VDFs f(vX), f(vY), f(vZ) at each input point and each DF component. "
      "Keep in mind that the interpolated values may be negative, especially at the wings of "
      "distribution, and that by default the spline is linearly extrapolated beyond its domain; "
      "to extrapolate as zero use `f(v, ext=False)` when evaluating the spline function.\n"
      "The VDFs are normalized such that the integral of f(v_k) d v_k  over the interval "
      "(-v_escape, v_escape) is unity for each component v_k. \n"
      "If dens==True, the last element is the value of density or an array of such values for "
      "each input point and DF component.\n" },
    { NULL }
};

static PyTypeObject GalaxyModelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "agama.GalaxyModel",
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
        FILTERMSG(utils::VL_DEBUG, "Agama", "Deleted " + std::string(self->name) + " at " +
            utils::toString(self->comp.get()));
    else
        FILTERMSG(utils::VL_DEBUG, "Agama", "Deleted an empty component");
    self->comp.reset();
    // self->name is either NULL or points to a constant string that does not require deallocation
    Py_TYPE(self)->tp_free(self);
}

int Component_init(ComponentObject* self, PyObject* args, PyObject* namedArgs)
{
    if(self->comp) {
        PyErr_SetString(PyExc_RuntimeError, "Component object cannot be reinitialized");
        return -1;
    }
    if(!onlyNamedArgs(args, namedArgs))
        return -1;
    NamedArgs nargs(namedArgs);

    // check if a potential object was provided
    PyObject* pot_obj = nargs.pop("potential");
    potential::PtrPotential pot = getPotential(pot_obj);
    if(pot_obj!=NULL && !pot) {
        PyErr_SetString(PyExc_TypeError, "Argument 'potential' must be a valid Potential object");
        return -1;
    }

    // check if a density object was provided
    PyObject* dens_obj = nargs.pop("density");
    potential::PtrDensity dens = getDensity(dens_obj);
    if(dens_obj!=NULL && !dens) {
        PyErr_SetString(PyExc_TypeError, "Argument 'density' must be a valid Density object");
        return -1;
    }

    // check if a df object was provided
    PyObject* df_obj = nargs.pop("df");
    df::PtrDistributionFunction df = getDistributionFunction(df_obj);
    if(df_obj!=NULL && !df) {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'df' must be a valid DistributionFunction object");
        return -1;
    }

    // check if a 'disklike' flag was provided
    int disklike = toBool(nargs.pop("disklike"), -1);

    // choose the variant of component: static or DF-based
    if((pot_obj!=NULL && df_obj!=NULL) || (pot_obj==NULL && df_obj==NULL && dens_obj==NULL)) {
        PyErr_SetString(PyExc_TypeError,
            "Should provide either a 'potential' and/or 'density' argument for a static component, "
            "or a 'df' argument for a component specified by a distribution function");
        return -1;
    }
    // if density and/or DF is provided, it must be tagged to be either disk-like or spheroidal
    if((dens!=NULL || df!=NULL) && disklike == -1) {
        PyErr_SetString(PyExc_TypeError, "Should provide a boolean argument 'disklike'");
        return -1;
    }
    if(!df_obj) {   // static component with potential and optionally density
        if(PyErr_Occurred() || !nargs.empty())  // not all arguments were used, but no more are expected
            return -1;
        try {
            if(!dens) {  // only potential
                self->comp.reset(new galaxymodel::ComponentStatic(pot));
                self->name = "Static potential component";
            } else {     // both potential and density
                self->comp.reset(new galaxymodel::ComponentStatic(dens, disklike, pot));
                self->name = disklike ? "Static disklike component" : "Static spheroidal component";
            }
            FILTERMSG(utils::VL_DEBUG, "Agama", "Created a " + std::string(self->name) + " at "+
                utils::toString(self->comp.get()));
            return 0;
        }
        catch(std::exception& ex) {
            raisePythonException(ex, "Error in creating a static component: ");
            return -1;
        }
    } else if(disklike == 0) {   // spheroidal component
        double rmin  = toDouble(nargs.pop("rminSph"), NAN) * conv->lengthUnit;
        double rmax  = toDouble(nargs.pop("rmaxSph"), NAN) * conv->lengthUnit;
        int gridSize = toInt(nargs.pop("sizeRadialSph"), -1);
        int lmax     = toInt(nargs.pop("lmaxAngularSph"), 0);
        int mmax     = toInt(nargs.pop("mmaxAngularSph"), 0);
        if(PyErr_Occurred() || !nargs.empty())  // not all arguments were used
            return -1;
        if(rmin<=0 || rmax<=rmin || gridSize<2 || lmax<0 || mmax<0 || mmax>lmax) {
            PyErr_SetString(PyExc_ValueError,
                "For spheroidal components, should provide valid values for the following arguments: "
                "rminSph, rmaxSph, sizeRadialSph, lmaxAngularSph[=0], mmaxAngularSph[=0]");
            return -1;
        }
        try {
            self->comp.reset(new galaxymodel::ComponentWithSpheroidalDF(
                df, dens, lmax, mmax, gridSize, rmin, rmax));
            self->name = "Spheroidal component";
            FILTERMSG(utils::VL_DEBUG, "Agama", "Created a " + std::string(self->name) + " at "+
                utils::toString(self->comp.get()));
            return 0;
        }
        catch(std::exception& ex) {
            raisePythonException(ex, "Error in creating a spheroidal component: ");
            return -1;
        }
    } else {   // disk-like component
        double Rmin  = toDouble(nargs.pop("RminCyl"), NAN) * conv->lengthUnit;
        double Rmax  = toDouble(nargs.pop("RmaxCyl"), NAN) * conv->lengthUnit;
        double zmin  = toDouble(nargs.pop("zminCyl"), NAN) * conv->lengthUnit;
        double zmax  = toDouble(nargs.pop("zmaxCyl"), NAN) * conv->lengthUnit;
        int gridSizeR= toInt(nargs.pop("sizeRadialCyl"),   -1);
        int gridSizez= toInt(nargs.pop("sizeVerticalCyl"), -1);
        int mmax     = toInt(nargs.pop("mmaxAngularCyl"),   0);
        if(PyErr_Occurred() || !nargs.empty())  // not all arguments were used
            return -1;
        if(Rmin<=0 || Rmax<=Rmin || gridSizeR<2 || zmin<=0 || zmax<=zmin || gridSizez<2 || mmax<0) {
            PyErr_SetString(PyExc_ValueError,
                "For disk-like components, should provide valid values for the following arguments: "
                "RminCyl, RmaxCyl, sizeRadialCyl, zminCyl, zmaxCyl, sizeVerticalCyl, mmaxAngularCyl[=0]");
            return -1;
        }
        try {
            self->comp.reset(new galaxymodel::ComponentWithDisklikeDF(
                df, dens, mmax, gridSizeR, Rmin, Rmax, gridSizez, zmin, zmax));
            self->name = "Disklike component";
            FILTERMSG(utils::VL_DEBUG, "Agama", "Created a " + std::string(self->name) + " at "+
                utils::toString(self->comp.get()));
            return 0;
        }
        catch(std::exception& ex) {
            raisePythonException(ex, "Error in creating a disklike component: ");
            return -1;
        }
    }
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
    // otherwise no density is available (e.g. for a static component specified by the potential)
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* Component_getDF(ComponentObject* self)
{
    df::PtrDistributionFunction df = self->comp->getDF();
    if(df)
        return createDistributionFunctionObject(df);
    // otherwise no df is available (e.g. for a static component)
    Py_INCREF(Py_None);
    return Py_None;
}

static PyGetSetDef Component_properties[] = {
    { const_cast<char*>("potential"), (getter)Component_getPotential, NULL,
      const_cast<char*>("Potential associated with a static component, or None"), NULL },
    { const_cast<char*>("density"), (getter)Component_getDensity, NULL,
      const_cast<char*>("Density object representing the fixed density profile "
      "for a static component (or None if it has only a potential profile), "
      "or the density profile from the previous iteration of the self-consistent "
      "modelling procedure for a DF-based component"), NULL },
    { const_cast<char*>("df"), (getter)Component_getDF, NULL,
      const_cast<char*>("DistributionFunction object associated with a DF-based component, "
      "or None for a static component"), NULL },
    { NULL }
};

static PyTypeObject ComponentType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "agama.Component",
    sizeof(ComponentObject), 0, (destructor)Component_dealloc,
    0, 0, 0, 0, Component_name, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringComponent,
    0, 0, 0, 0, 0, 0, 0, 0, Component_properties, 0, 0, 0, 0, 0,
    (initproc)Component_init
};


///@}
//  ---------------------------------
/// \name  SelfConsistentModel class
//  ---------------------------------
///@{

static const char* docstringSelfConsistentModel =
    "A class for performing iterative self-consistent modelling procedure.\n"
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

/// \cond INTERNAL_DOCS
/// Python type corresponding to SelfConsistentModel class
typedef struct {
    PyObject_HEAD
    PyObject* components;         ///< Python list of Component objects
    /// members of galaxymodel::SelfConsistentModel structure listed here
    potential::PtrPotential pot;  ///< current potential (initially may be empty)
    actions::PtrActionFinder af;  ///< corresponding action finder (may be empty initially)
    bool useActionInterpolation;  ///< whether to use the interpolated action finder
    bool verbose;                 ///< whether to print out progress report messages
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
    self->pot.reset();
    self->af.reset();
    Py_TYPE(self)->tp_free(self);
}

int SelfConsistentModel_init(SelfConsistentModelObject* self, PyObject* args, PyObject* namedArgs)
{
    if(self->components) {
        PyErr_SetString(PyExc_RuntimeError, "SelfConsistentModel object cannot be reinitialized");
        return -1;
    }
    if(!onlyNamedArgs(args, namedArgs))
        return -1;
    NamedArgs nargs(namedArgs);

    // allocate a new empty list of components or take it from the input argument
    PyObject* comp_obj = nargs.pop("components");
    if(comp_obj!=NULL) {
        if(PyList_Check(comp_obj)) {
            self->components = comp_obj;
            Py_INCREF(comp_obj);
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument 'components' must be a list");
            return -1;
        }
    } else {
        self->components  = PyList_New(0);
    }

    // check if a potential object was provided
    PyObject* pot_obj = nargs.pop("potential");
    self->pot = getPotential(pot_obj);
    if(pot_obj!=NULL && !self->pot) {
        PyErr_SetString(PyExc_TypeError, "Argument 'potential' must be a valid Potential object");
        return -1;
    }

    // parse remaining parameters and verify that no unknown arguments were provided
    self->useActionInterpolation = toBool(nargs.pop("useActionInterpolation"), false);
    self->verbose       = toBool(nargs.pop("verbose"), true);
    // default values for the grid parameters are invalid, forcing the user to set them explicitly
    self->rminSph     = toDouble(nargs.pop("rminSph"), NAN);
    self->rmaxSph     = toDouble(nargs.pop("rmaxSph"), NAN);
    self->sizeRadialSph  = toInt(nargs.pop("sizeRadialSph" ), -1);
    self->lmaxAngularSph = toInt(nargs.pop("lmaxAngularSph"), -1);
    self->RminCyl     = toDouble(nargs.pop("RminCyl"), NAN);
    self->RmaxCyl     = toDouble(nargs.pop("RmaxCyl"), NAN);
    self->zminCyl     = toDouble(nargs.pop("zminCyl"), NAN);
    self->zmaxCyl     = toDouble(nargs.pop("zmaxCyl"), NAN);
    self->sizeRadialCyl  = toInt(nargs.pop("sizeRadialCyl"  ), -1);
    self->sizeVerticalCyl= toInt(nargs.pop("sizeVerticalCyl"), -1);
    if(PyErr_Occurred() || !nargs.empty())  // not all arguments were used
        return -1;
    return 0;
}

PyObject* SelfConsistentModel_iterate(SelfConsistentModelObject* self)
{
    galaxymodel::SelfConsistentModel model;
    // parse the Python list of components
    if(self->components==NULL || !PyList_Check(self->components) || PyList_Size(self->components)==0)
    {
        PyErr_SetString(PyExc_TypeError,
            "SelfConsistentModel.components should be a non-empty list of Component objects");
        return NULL;
    }
    int numComp = PyList_Size(self->components);
    for(int i=0; i<numComp; i++)
    {
        PyObject* elem = PyList_GetItem(self->components, i);
        if(!PyObject_TypeCheck(elem, &ComponentType)) {
            PyErr_SetString(PyExc_TypeError,
                "SelfConsistentModel.components should contain only Component objects");
            return NULL;
        }
        model.components.push_back(((ComponentObject*)elem)->comp);
    }
    model.useActionInterpolation = self->useActionInterpolation;
    model.verbose = self->verbose;
    model.rminSph = self->rminSph * conv->lengthUnit;
    model.rmaxSph = self->rmaxSph * conv->lengthUnit;
    model.sizeRadialSph = self->sizeRadialSph;
    model.lmaxAngularSph = self->lmaxAngularSph;
    model.RminCyl = self->RminCyl * conv->lengthUnit;
    model.RmaxCyl = self->RmaxCyl * conv->lengthUnit;
    model.zminCyl = self->zminCyl * conv->lengthUnit;
    model.zmaxCyl = self->zmaxCyl * conv->lengthUnit;
    model.sizeRadialCyl   = self->sizeRadialCyl;
    model.sizeVerticalCyl = self->sizeVerticalCyl;
    model.totalPotential  = self->pot;   // may be empty - will be automatically initialized if so
    model.actionFinder    = self->af;    // same
    try {
        {   // this operation contains OpenMP-parallelized loops, which may be calling back
            // user-defined Python potentials or DFs, so we need to release GIL
            PyReleaseGIL unlock;
            doIteration(model);
        }
        // retrieve the updated potential and action finder
        self->pot = model.totalPotential;
        self->af  = model.actionFinder;
        Py_INCREF(Py_None);
        return Py_None;
    }
    catch(std::exception& ex) {
        raisePythonException(ex, "Error in SelfConsistentModel.iterate(): ");
        return NULL;
    }
}

PyObject* SelfConsistentModel_getComponents(SelfConsistentModelObject* self, void*)
{
    Py_INCREF(self->components);
    return self->components;
}

int SelfConsistentModel_setComponents(SelfConsistentModelObject* self, PyObject* comp_obj, void*)
{
    if(!PyList_Check(comp_obj)) {
        PyErr_SetString(PyExc_TypeError, "SelfConsistentModel.components must be a list");
        return -1;
    }
    self->components = comp_obj;
    Py_INCREF(self->components);
    return 0;
}

PyObject* SelfConsistentModel_getPotential(SelfConsistentModelObject* self, void*)
{
    if(self->pot)
        return createPotentialObject(self->pot);
    Py_INCREF(Py_None);  // otherwise the potential is not yet initialized
    return Py_None;
}

int SelfConsistentModel_setPotential(SelfConsistentModelObject* self, PyObject* pot_obj, void*)
{
    potential::PtrPotential pot = getPotential(pot_obj);
    if(!pot) {
        PyErr_SetString(PyExc_TypeError,
            "SelfConsistentModel.potential must be a valid Potential object");
        return -1;
    }
    if(self->pot != pot)
        self->af.reset();  // invalidate the action finder because the potential has changed
    self->pot = pot;
    return 0;
}

PyObject* SelfConsistentModel_getActionFinder(SelfConsistentModelObject* self, void*)
{
    if(self->af)
        return createActionFinderObject(self->af);
    Py_INCREF(Py_None);  // otherwise the action finder is not yet initialized or invalidated
    return Py_None;
}

static PyMemberDef SelfConsistentModel_members[] = {
    { const_cast<char*>("useActionInterpolation"), T_BOOL,
      offsetof(SelfConsistentModelObject, useActionInterpolation), 0,
      const_cast<char*>("Whether to use interpolated action finder (faster but less accurate)") },
    { const_cast<char*>("verbose"), T_BOOL, offsetof(SelfConsistentModelObject, verbose), 0,
      const_cast<char*>("Whether to print out progress report messages") },
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

static PyGetSetDef SelfConsistentModel_properties[] = {
    { const_cast<char*>("components"),
      (getter)SelfConsistentModel_getComponents, (setter)SelfConsistentModel_setComponents,
      const_cast<char*>("List of Component objects (may be modified by the user, but should be non-empty "
      "and contain only instances of Component class upon a call to 'iterate()' method)"), NULL },
    { const_cast<char*>("potential"),
      (getter)SelfConsistentModel_getPotential, (setter)SelfConsistentModel_setPotential,
      const_cast<char*>("Total potential of the model, or None if not yet initialized"), NULL },
    { const_cast<char*>("af"), (getter)SelfConsistentModel_getActionFinder, NULL,
      const_cast<char*>("Action finder associated with the total potential (read-only)"), NULL },
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
    PyVarObject_HEAD_INIT(NULL, 0)
    "agama.SelfConsistentModel",
    sizeof(SelfConsistentModelObject), 0, (destructor)SelfConsistentModel_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringSelfConsistentModel, 0, 0, 0, 0, 0, 0,
    SelfConsistentModel_methods, SelfConsistentModel_members, SelfConsistentModel_properties,
    0, 0, 0, 0, 0, (initproc)SelfConsistentModel_init
};


///@}
//  -----------------------------------------
/// \name  Spline class and related routines
//  -----------------------------------------
///@{

static const char* docstringSpline =
    "A common interface to cubic or quintic splines, and B-splines of degree 0 to 3.\n"
    "Cubic and quintic splines are constructed from the coordinates of grid points (x) and "
    "the spline values at grid points (y), optionally with first derivatives at one or two endpoints "
    "(left/right) or at all grid points (der); they have 1-2 (cubic) or 3-4 (quintic) continuous "
    "derivatives, depending on the setup.\n"
    "B-splines of degree D=0..3 are constructed from the coordinates of grid points "
    "(x, array of length N) and the amplitudes of basis functions (ampl, array of length N+D-1), "
    "and have D-1 continuous derivatives.\n"
    "Arguments:\n"
    "    x (array of floats) -- grid nodes in x, must be sorted in increasing order.\n"
    "    y (array of floats, optional) -- values of the spline at all grid nodes, same length as x "
    "(required only for cubic/quintic splines).\n"
    "    der (array of floats, optional) -- first derivatives at all grid nodes, same length as x "
    "(may be provided only for cubic/quintic splines, mutually exclusive with `ampl`).\n"
    "    left (float, optional) -- first derivative at the leftmost endpoint for clamped cubic or "
    "quintic splines; a default value (NaN) means a natural boundary condition (i.e., zero second "
    "derivative at the endpoint).\n"
    "    right (float, optional) -- same for the rightmost endpoint.\n"
    "    reg (boolean, optional, default False) -- relevant only for natural/clamped cubic splines, "
    "applies a regularization filter to preserve monotonic trends and reduce overshooting "
    "in the case of sharp jumps in the input data and; if set to True, the provided left/right "
    "endpoint derivatives may not be respected, and the second derivative may become discontinuous.\n"
    "    quintic (boolean, optional) -- disambiguates between cubic and quintic splines. "
    "The default behaviour, which makes sense in most cases, is to construct a natural or clamped "
    "cubic spline when only the values `y` are provided, or a 'standard' quintic spline when both "
    "values `y` and first derivatives `der` are provided. To override this behaviour, one should "
    "explicitly set `quintic`=False to create a Hermite cubic interpolator from values and derivatives, "
    "or `quintic`=True to create a natural/clamped quintic spline from values alone.\n"
    "    ampl (array of floats, optional) -- if provided _instead of_ `y`, this produces a B-spline "
    "of degree D=0..3, which is determined automatically from len(ampl) = len(x) + D - 1. "
    "This argument is mutually exclusive with all other optional arguments.\n\n"
    "The following table summarizes the various kinds of interpolators, their required and optional "
    "input arguments, and their smoothness properties.\n"
    "The required input arguments are marked by `+`, optional -- by default values in brackets, "
    "and `-` indicates that the argument is not acceptable for this configuration.\n"
    "Derivatives that are continuous are marked by `c`, discontinuous -- by `d`, and zero -- by `0`; "
    "note that the call operator can return derivatives up to and including third.\n"
    "type               x   ampl  y    y'   y''  y''' y''''  left/right  reg    quintic\n"
    "natural* cubic     +    -    +    c    c/d  d    0      (NaN)     (False)  (False)\n"
    "hermite  cubic     +    -    +    +    d    d    0        -          -     +False\n"
    "natural* quintic   +    -    +    c    c    c    c      (NaN)        -     +True\n"
    "standard quintic   +    -    +    +    c    c    d        -          -     (True)\n"
    "degree-0 B-spline  +    +    d    0    0    0    0        -          -        -\n"
    "degree-1 B-spline  +    +    c    d    0    0    0        -          -        -\n"
    "degree-2 B-spline  +    +    c    c    d    0    0        -          -        -\n"
    "degree-3 B-spline  +    +    c    c    c    d    0        -          -        -\n"
    "The asterisk after `natural` indicates that one can replace the natural boundary condition "
    "(zero second derivative at endpoint) with a clamped one (explicitly providing the first "
    "derivative `left`/`right`) without affecting the smoothness properties of the spline.\n"
    "The `c/d` in the second derivative of the natural/clamped cubic spline indicates that "
    "it is normally continuous, except when a regularization filter is applied (`reg`=True), "
    "in which case it may become discontinuous.\n"
    "The boolean argument `quintic` needs to be specified explicitly in the two non-default cases "
    "marked by `+`, otherwise it takes the default value shown in brackets, or is not acceptable "
    "when constructing a B-spline.\n\n"
    "Values of the spline and up to its third derivative are computed using the () "
    "operator with the first argument being a single x-point or an array of points of any shape, "
    "the optional second argument (der=...) is the derivative index (0, 1, 2 or 3), "
    "and the optional third argument (ext=...) specifies the value returned for points "
    "outside the definition region; if the latter is not provided, cubic and quintic splines "
    "are linearly extrapolated outside its definition region (unless the spline is empty, "
    "in which case the result is always NaN), while B-splines return NaN.\n"
    "If an extra argument conv=... is provided to the () operator, the result is an analytic "
    "convolution of the spline with a given kernel. This could be another instance of Spline "
    "or a Gaussian kernel specified by its width; in the latter case the argument can be "
    "a single number or an array of numbers with the same shape as `x`. "
    "In computing the convolution, the spline is set to zero outside its definition region, "
    "unlike the extrapolation performed for its value or derivative.\n"
    "The return value of the () operator is a single number if the input `x` is a single number, "
    "or has the same shape as `x` if the latter is an array.\n"
    "A Spline object has a length equal to the number of nodes in `x`, and its [] operator "
    "returns the value of `x` at the given node.\n";

/// \cond INTERNAL_DOCS
/// Python type corresponding to Spline class
typedef struct {
    PyObject_HEAD
    math::PtrInterpolator1d spl;
    std::string name;
} SplineObject;
/// \endcond

void Spline_dealloc(SplineObject* self)
{
    FILTERMSG(utils::VL_DEBUG, "Agama",
        self->spl.get() ? "Deleted a " + self->name + " at " + utils::toString(self->spl.get()) :
        "Deleted an empty spline");
    self->spl.reset();
    Py_TYPE(self)->tp_free(self);
}

int Spline_init(SplineObject* self, PyObject* args, PyObject* namedArgs)
{
    if(self->spl) {
        PyErr_SetString(PyExc_RuntimeError, "Spline object cannot be reinitialized");
        return -1;
    }
    PyObject* x_obj=NULL;
    PyObject* y_obj=NULL;
    PyObject* d_obj=NULL;
    PyObject* a_obj=NULL;
    PyObject* q_obj=NULL;
    double derivLeft=NAN, derivRight=NAN;  // undefined by default
    int regularize=0;
    static const char* keywords[] = {"x", "y", "der", "ampl", "left", "right", "reg", "quintic", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "O|OOOddiO", const_cast<char **>(keywords),
        &x_obj, &y_obj, &d_obj, &a_obj, &derivLeft, &derivRight, &regularize, &q_obj) ||
        !((y_obj==NULL) ^ (a_obj==NULL)))
    {
        PyErr_SetString(PyExc_TypeError, "Spline: "
            "must provide the array `x` of grid nodes, and either the array of spline values `y` "
            "or the array of B-spline amplitudes `ampl`");
        return -1;
    }
    std::vector<double>
        xvalues(toDoubleArray(x_obj)),
        yvalues(toDoubleArray(y_obj)),
        dvalues(toDoubleArray(d_obj)),
        avalues(toDoubleArray(a_obj));
    size_t size = xvalues.size();
    if(y_obj && (yvalues.size() != size || (d_obj && dvalues.size() != size))) {
        PyErr_SetString(PyExc_TypeError, "Spline: input does not contain valid arrays");
        return -1;
    }
    if(a_obj && (d_obj || q_obj || derivLeft==derivLeft || derivRight==derivRight || regularize)) {
        PyErr_SetString(PyExc_TypeError,
            "Spline: argument 'ampl' cannot be used together with any other optional arguments");
        return -1;
    }
    if(d_obj && (derivLeft==derivLeft || derivRight==derivRight || regularize)) {
        PyErr_SetString(PyExc_TypeError,
            "Spline: argument 'der' cannot be used together with 'left', 'right' or 'reg'");
        return -1;
    }
    // if not specified explicitly, create a quintic spline only when derivatives are provided
    bool quintic = q_obj ? (bool)PyObject_IsTrue(q_obj) : (bool)d_obj;
    if(quintic && regularize) {
        PyErr_SetString(PyExc_TypeError, "Spline: argument 'reg' cannot be used with quintic splines");
        return -1;
    }
    try {
        new (&(self->name)) std::string;  // initialize with an empty string
        if(a_obj) {
            int D = (int)(avalues.size()) + 1 - size;
            switch(D) {
                case 0: self->spl.reset(new math::BsplineWrapper<0>(xvalues, avalues)); break;
                case 1: self->spl.reset(new math::BsplineWrapper<1>(xvalues, avalues)); break;
                case 2: self->spl.reset(new math::BsplineWrapper<2>(xvalues, avalues)); break;
                case 3: self->spl.reset(new math::BsplineWrapper<3>(xvalues, avalues)); break;
                default:
                    PyErr_SetString(PyExc_TypeError,
                        "Spline: incorrect size of the array of B-spline amplitudes");
                    return -1;
            }
            self->name = "degree " + utils::toString(D) + " B-spline";
        } else if(!quintic) {
            if(d_obj) {
                self->spl.reset(new math::CubicSpline(xvalues, yvalues, dvalues));
                self->name += "hermite ";
            } else {
                self->spl.reset(new math::CubicSpline(xvalues, yvalues,
                    regularize, derivLeft, derivRight));
                if(derivLeft == derivLeft || derivRight == derivRight)
                    self->name += "clamped ";
                if(regularize)
                    self->name += "regularized ";
            }
            self->name += "cubic spline";
        } else /*quintic*/ {
            if(d_obj) {
                self->spl.reset(new math::QuinticSpline(xvalues, yvalues, dvalues));
            } else {
                self->spl.reset(new math::QuinticSpline(xvalues, yvalues, derivLeft, derivRight));
                if(derivLeft == derivLeft || derivRight == derivRight)
                    self->name += "clamped ";
                else
                    self->name += "natural ";
            }
            self->name += "quintic spline";
        }
        self->name[0] -= 32;  // capitalize the first letter of the name
        FILTERMSG(utils::VL_DEBUG, "Agama",
            "Created a "+self->name+" at "+utils::toString(self->spl.get()));
        return 0;
    }
    catch(std::exception& ex) {
        raisePythonException(ex);
        return -1;
    }
}

inline double splEval(const math::BaseInterpolator1d& spl, double x, int der)
{
    double result;
    switch(der) {
        case 0: return spl.value(x);
        case 1: spl.evalDeriv(x, NULL, &result); return result;
        case 2: spl.evalDeriv(x, NULL, NULL, &result); return result;
        case 3: spl.evalDeriv(x, NULL, NULL, NULL, &result); return result;
        default: return NAN;  // shouldn't occur
    }
}

PyObject* Spline_value(SplineObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!self->spl) {  // shouldn't happen
        PyErr_SetString(PyExc_RuntimeError, "Spline object is not properly initialized");
        return NULL;
    }
    static const char* keywords[] = {"x", "der", "ext", "conv", NULL};
    PyObject* ptx=NULL;
    int der=0;
    PyObject* extrapolate_obj=NULL;
    PyObject* conv_obj=NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "O|iOO", const_cast<char **>(keywords),
        &ptx, &der, &extrapolate_obj, &conv_obj))
        return NULL;
    if(der<0 || der>3) {
        PyErr_SetString(PyExc_ValueError, "Can only compute derivatives up to 3rd");
        return NULL;
    }
    if(conv_obj && (der!=0 || extrapolate_obj)) {
        PyErr_SetString(PyExc_ValueError, "`conv` cannot be used together with `der` or `ext`");
        return NULL;
    }

    // `conv` may be another instance of Spline
    math::PtrInterpolator1d conv;
    if(conv_obj && PyObject_TypeCheck(conv_obj, SplineTypePtr) && ((SplineObject*)conv_obj)->spl)
        conv = ((SplineObject*)conv_obj)->spl;

    // check if we should extrapolate the spline (default behaviour),
    // or replace the output with the given value if it's out of range (if ext=... argument was given)
    double extrapolate_val = extrapolate_obj == NULL ? 0 : toDouble(extrapolate_obj);
    double xmin = self->spl->xmin(), xmax = self->spl->xmax();

    // if the input is a single value, just do it
    if(PyFloat_Check(ptx) || PyInt_Check(ptx) || PyLong_Check(ptx)) {
        double x = PyFloat_AsDouble(ptx);
        if(PyErr_Occurred())
            return NULL;
        if(conv_obj) {
            if(conv)
                return PyFloat_FromDouble(self->spl->convolve(x, *conv));
            double sigma = PyFloat_AsDouble(conv_obj);
            if(PyErr_Occurred())
                return NULL;
            return PyFloat_FromDouble(self->spl->convolve(x, math::Gaussian(sigma)));
        }
        if(extrapolate_obj!=NULL && (x<xmin || x>xmax)) {
            Py_INCREF(extrapolate_obj);
            return extrapolate_obj;
        } else {
            return PyFloat_FromDouble(splEval(*self->spl, x, der));
        }
    }

    // otherwise the input should be an array, and the output will be an array of the same shape
    PyArrayObject *arr = (PyArrayObject*)
        PyArray_FROM_OTF(ptx, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY | NPY_ARRAY_ENSURECOPY);
    if(arr == NULL) {
        PyErr_SetString(PyExc_TypeError, "Argument must be either float, list or numpy array");
        return NULL;
    }
    npy_intp size = PyArray_SIZE(arr);

    // if `conv` is provided, it must be a single number or an array of the same shape as input `x`,
    // or another instance of Spline (in this case, the variable 'conv' is already initialized)
    PyArrayObject *sigma_arr = NULL;
    npy_intp sigma_size = 0;
    if(conv_obj && !conv) {
        sigma_arr = (PyArrayObject*)PyArray_FROM_OTF(conv_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        bool shape_ok = sigma_arr!=NULL;
        if(shape_ok && ((sigma_size = PyArray_SIZE(sigma_arr)) != 1)) {
            shape_ok &= sigma_size == size && PyArray_NDIM(arr) == PyArray_NDIM(sigma_arr);
            for(int d=0; shape_ok && d<PyArray_NDIM(arr); d++)
                shape_ok &= PyArray_DIMS(arr)[d] == PyArray_DIMS(sigma_arr)[d];
        }
        if(!shape_ok) {
            Py_XDECREF(sigma_arr);
            Py_DECREF(arr);
            PyErr_SetString(PyExc_TypeError, "Argument `sigma`, if provided, "
                "must be a single number or an array with the same shape as `x`");
            return NULL;
        }
    }

    // replace elements of the copy of input array with computed values
    for(int i=0; i<size; i++) {
        // reference to the array element to be replaced
        double& x = static_cast<double*>(PyArray_DATA(arr))[i];
        if(conv) {
            x = self->spl->convolve(x, *conv);
        }
        else if(sigma_arr) {
            double sigma = static_cast<double*>(PyArray_DATA(sigma_arr))[sigma_size==1 ? 0 : i];
            x = sigma==0 ?
                (x>=xmin && x<=xmax ? self->spl->value(x) : 0) :
                self->spl->convolve(x, math::Gaussian(sigma));
        }
        else if(extrapolate_obj!=NULL && (x<xmin || x>xmax))
            x = extrapolate_val;
        else
            x = splEval(*self->spl, x, der);
    }
    Py_XDECREF(sigma_arr);
    return PyArray_Return(arr);
}

PyObject* Spline_integrate(SplineObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!self->spl) {
        PyErr_SetString(PyExc_RuntimeError, "Spline object is not properly initialized");
        return NULL;
    }
    static const char* keywords[] = {"x1", "x2", "n", NULL};
    double x1, x2;
    PyObject* n_obj = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "dd|O", const_cast<char **>(keywords),
        &x1, &x2, &n_obj))
        return NULL;
    if(!n_obj)
        return PyFloat_FromDouble(self->spl->integrate(x1, x2));
    // `n` may be a nonnegative integer or another instance of Spline
    if(PyInt_Check(n_obj)) {
        long n = PyInt_AsLong(n_obj);
        if(n>=0 && n<0x7fff)
            return PyFloat_FromDouble(self->spl->integrate(x1, x2, n));
    }
    else if(PyObject_TypeCheck(n_obj, SplineTypePtr) && ((SplineObject*)n_obj)->spl) {
        return PyFloat_FromDouble(self->spl->integrate(x1, x2, *((SplineObject*)n_obj)->spl));
    }
    PyErr_SetString(PyExc_RuntimeError,
        "Argument 'n', if present, should be a nonnegative integer or another instance of Spline");
    return NULL;
}

PyObject* Spline_roots(SplineObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!self->spl) {
        PyErr_SetString(PyExc_RuntimeError, "Spline object is not properly initialized");
        return NULL;
    }
    static const char* keywords[] = {"y", "x1", "x2", NULL};
    double y = 0, x1 = NAN, x2 = NAN;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "|ddd", const_cast<char **>(keywords),
        &y, &x1, &x2))
        return NULL;
    return toPyArray(self->spl->roots(y, x1, x2));
}

PyObject* Spline_extrema(SplineObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!self->spl) {
        PyErr_SetString(PyExc_RuntimeError, "Spline object is not properly initialized");
        return NULL;
    }
    static const char* keywords[] = {"x1", "x2", NULL};
    double x1 = NAN, x2 = NAN;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "|dd", const_cast<char **>(keywords), &x1, &x2))
        return NULL;
    return toPyArray(self->spl->extrema(x1, x2));
}

PyObject* Spline_name(SplineObject* self)
{
    if(!self->spl) {
        PyErr_SetString(PyExc_RuntimeError, "Spline object is not properly initialized");
        return NULL;
    }
    size_t size = self->spl->xvalues().size();
    std::string result = self->name;
    if(result.empty())  // shouldn't happen; all spline creation methods also initialize the name
        result = "Unknown spline";
    result += " with " + utils::toString(size) + " nodes";
    if(size)
        result += " on [" + utils::toString(self->spl->xmin(), 16) +
            ":" + utils::toString(self->spl->xmax(), 16) + "]";
    return Py_BuildValue("s", result.c_str());
}

PyObject* Spline_elem(SplineObject* self, Py_ssize_t index)
{
    if(!self->spl) {
        PyErr_SetString(PyExc_RuntimeError, "Spline object is not properly initialized");
        return NULL;
    }
    if(index >= 0 && index < (Py_ssize_t)self->spl->xvalues().size())
        return Py_BuildValue("d", self->spl->xvalues()[index]);
    else {
        PyErr_SetString(PyExc_IndexError, "Spline node index out of range");
        return NULL;
    }
}

Py_ssize_t Spline_len(SplineObject* self)
{
    return self->spl ? self->spl->xvalues().size() : -1;
}

static PySequenceMethods Spline_sequence_methods = {
    (lenfunc)Spline_len, 0, 0, (ssizeargfunc)Spline_elem,
};

static PyMethodDef Spline_methods[] = {
    { "integrate", (PyCFunction)Spline_integrate, METH_VARARGS | METH_KEYWORDS,
      "Compute the integral of the spline, optionally multiplied by x^n or by another spline, "
      "over the given interval x1..x2; spline is set to zero outside its grid.\n"
      "Arguments:\n"
      "  x1 - left boundary of the interval;\n"
      "  x2 - right boundary of the interval;\n"
      "  n (default: 0) - additional weight function in integration; can be a positive integer "
      "(meaning multiplying the spline by x^n) or another instance of Spline class.\n"
      "Returns: the value of the integral.\n" },
    { "roots", (PyCFunction)Spline_roots, METH_VARARGS | METH_KEYWORDS,
      "Return all points on the given interval at which the spline attains the given value y.\n"
      "Arguments:\n"
      "  y - the required value (default: 0);\n"
      "  x1 - left boundary of the interval (default: leftmost point of the spline grid);\n"
      "  x2 - right boundary of the interval (default: rightmost point of the spline grid).\n"
      "Returns: an array (possibly empty) of all roots in order of increase." },
    { "extrema", (PyCFunction)Spline_extrema, METH_VARARGS | METH_KEYWORDS,
      "Return all points on the given interval at which the spline has a local minimum or maximum "
      "(i.e. its derivative crosses zero).\n"
      "Arguments:\n"
      "  x1 - left boundary of the interval (default: leftmost point of the spline grid);\n"
      "  x2 - right boundary of the interval (default: rightmost point of the spline grid).\n"
      "Returns: an array of all extrema in order of increase, including the two endpoints." },
    { NULL }
};

static PyTypeObject SplineType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "agama.Spline",
    sizeof(SplineObject), 0, (destructor)Spline_dealloc,
    0, 0, 0, 0, (reprfunc)Spline_name, 0, &Spline_sequence_methods, 0, 0,
    (PyCFunctionWithKeywords)Spline_value, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringSpline,
    0, 0, 0, 0, 0, 0, Spline_methods, 0, 0, 0, 0, 0, 0, 0,
    (initproc)Spline_init
};

/// Construct a Python cubic spline object from the provided x and y arrays
PyObject* createCubicSpline(const std::vector<double>& x, const std::vector<double>& y)
{
    // allocate a new Python Spline object
    SplineObject* spl_obj = PyObject_New(SplineObject, &SplineType);
    if(!spl_obj)
        return NULL;
    // initialize the smart pointer with zero before (re-)assigning a value to it
    new (&(spl_obj->spl)) math::PtrFunction;
    spl_obj->spl.reset(new math::CubicSpline(x, y));
    new (&(spl_obj->name)) std::string;  // initialize with an empty string
    spl_obj->name = "Cubic spline";
    FILTERMSG(utils::VL_DEBUG, "Agama", "Constructed a cubic spline of size "+
        utils::toString(x.size())+" at "+utils::toString(spl_obj->spl.get()));
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
    "    smooth -- (float or None) is the parameter controlling the tradeoff "
    "between smoothness and approximation error; None means no additional smoothing "
    "(beyond the one resulting from discreteness of the spacing of knots), "
    "zero (default, recommended) means optimal smoothing, and any value larger than zero "
    "results in oversmoothing; values around unity usually yield a reasonable extra suppression "
    "of noise without significantly increasing the rms error in the approximation.\n"
    "Returns: a Spline object.\n";

PyObject* splineApprox(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"knots","x","y","w","smooth",NULL};
    PyObject* k_obj=NULL;
    PyObject* x_obj=NULL;
    PyObject* y_obj=NULL;
    PyObject* w_obj=NULL;
    PyObject* s_obj=NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "OOO|OO", const_cast<char **>(keywords),
        &k_obj, &x_obj, &y_obj, &w_obj, &s_obj))
    {
        PyErr_SetString(PyExc_TypeError, "splineApprox: "
            "must provide an array of grid nodes and two arrays of equal length (input x and y points), "
            "optionally an array of weights and a smoothing factor(float or None)");
        return NULL;
    }
    std::vector<double>
        knots  (toDoubleArray(k_obj)),
        xvalues(toDoubleArray(x_obj)),
        yvalues(toDoubleArray(y_obj)),
        weights(toDoubleArray(w_obj));
    if(xvalues.empty() || yvalues.empty() || knots.empty()) {
        PyErr_SetString(PyExc_TypeError, "Input does not contain valid arrays");
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
    double smoothfactor;
    if(s_obj == NULL)
        smoothfactor = 0.;
    else if(s_obj == Py_None)
        smoothfactor = NAN;
    else if(PyNumber_Check(s_obj))
        smoothfactor = PyFloat_AsDouble(s_obj);
    else {
        PyErr_SetString(PyExc_TypeError, "Argument 'smooth' must be a float or None");
        return NULL;
    }
    try{
        math::SplineApprox spl(knots, xvalues, weights);
        std::vector<double> amplitudes;
        if(smoothfactor >= 0)
            amplitudes = spl.fitOversmooth(yvalues, smoothfactor);
        else if(smoothfactor < 0)  // undocumented: fit with EDF = -smoothfactor
            amplitudes = spl.fit(yvalues, -smoothfactor);
        else  // if smooth=None, fit without any smoothing (EDF = numKnots)
            amplitudes = spl.fit(yvalues, knots.size());
        return createCubicSpline(knots, amplitudes);
    }
    catch(std::exception& ex) {
        raisePythonException(ex);
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
    "    der3 (boolean, default False) determines how the roughness penalty is computed: "
    "using 2nd derivative (False) or 3rd derivative (True). This choice determines the class "
    "of functions that have zero penalty and are the limiting cases for infinitely large smoothing: "
    "the latter choice implies a pure Gaussian, and the former - an exponential function, which is, "
    "however, only attainable on (semi-)finite intervals, when there is no extrapolation.\n"
    "    smooth (float, default 0) -- optional extra smoothing.\n"
    "Returns: a Spline object representing log(rho(x)).\n";

PyObject* splineLogDensity(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"knots","x","w","infLeft","infRight","der3","smooth",NULL};
    PyObject* k_obj=NULL;
    PyObject* x_obj=NULL;
    PyObject* w_obj=NULL;
    int infLeft=0, infRight=0, der3=0;  // should rather be bool, but python doesn't handle it in arg list
    double smooth=0;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "OO|Oiiid", const_cast<char **>(keywords),
        &k_obj, &x_obj, &w_obj, &infLeft, &infRight, &der3, &smooth))
    {
        PyErr_SetString(PyExc_TypeError, "splineLogDensity: "
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
        PyErr_SetString(PyExc_TypeError, "Input does not contain valid arrays");
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
            (infRight? math::FO_INFINITE_RIGHT: 0) |
            (der3    ? math::FO_PENALTY_3RD_DERIV: 0)),
            smooth );
        return createCubicSpline(knots, amplitudes);
    }
    catch(std::exception& ex) {
        raisePythonException(ex);
        return NULL;
    }
}


///@}
//  ----------------------------------------------------------
/// \name  Container class for storing the interpolated orbit
//  ----------------------------------------------------------
///@{

/// docstring of the helper class Orbit
static const char* docstringOrbitClass =
    "Result of orbit integration represented as three spline interpolators.\n"
    "This class cannot be instantiated directly, but can be returned by the agama.orbit() routine "
    "when the requested output dtype is 'object'.\n"
    "The () operator returns an array of 6 phase-space coordinates of the orbit at a given time, "
    "or a 2d array of shape Nx6 if the input is an array of times. "
    "If the time is outside the range spanned by the orbit, the result is filled with NaN.\n"
    "An Orbit object has a length equal to the number of points in the originally recorded "
    "trajectory, and its [] operator returns the value of time at the given point.\n";

/// \cond INTERNAL_DOCS
/// Python type corresponding to Orbit class
typedef struct {
    PyObject_HEAD
    PyObject *x, *y, *z;  ///< quintic spline interpolators
    double Omega;         ///< angular frequency of the rotating frame (needed for correct velocities)
    bool reversed;        ///< whether the orbit is reversed in time (needed for correct indexing)
} OrbitObject;
/// \endcond

void Orbit_dealloc(OrbitObject* self)
{
    Py_XDECREF(self->x);
    Py_XDECREF(self->y);
    Py_XDECREF(self->z);
    Py_TYPE(self)->tp_free(self);
}

int Orbit_init(OrbitObject* /*self*/, PyObject* /*args*/, PyObject* /*namedArgs*/)
{
    PyErr_SetString(PyExc_TypeError, "Instances of Orbit class cannot be created from Python");
    return -1;
}

PyObject* Orbit_value(OrbitObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!noNamedArgs(namedArgs))
        return NULL;
    if( !self->x || !PyObject_TypeCheck(self->x, SplineTypePtr) || !((SplineObject*)(self->x))->spl ||
        !self->y || !PyObject_TypeCheck(self->y, SplineTypePtr) || !((SplineObject*)(self->y))->spl ||
        !self->z || !PyObject_TypeCheck(self->z, SplineTypePtr) || !((SplineObject*)(self->z))->spl )
    {
        PyErr_SetString(PyExc_RuntimeError, "Orbit is not properly initialized");
        return NULL;
    }
    const math::BaseInterpolator1d
        &splx = *((SplineObject*)(self->x))->spl,
        &sply = *((SplineObject*)(self->y))->spl,
        &splz = *((SplineObject*)(self->z))->spl;
    if(PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "Function takes a single argument");
        return NULL;
    }
    PyObject* t_obj = PyTuple_GET_ITEM(args, 0);
    PyArrayObject *t_arr = NULL, *result = NULL;
    npy_intp size = 1;
    double t = NAN;
    if(PyFloat_Check(t_obj) || PyInt_Check(t_obj) || PyLong_Check(t_obj)) {
        t = PyFloat_AsDouble(t_obj);
        if(PyErr_Occurred())
            return NULL;
        npy_intp dims[1] = {6};
        result = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    } else {
        t_arr = (PyArrayObject*)PyArray_FROM_OTF(t_obj, NPY_DOUBLE, 0);
        if(!t_arr || PyArray_NDIM(t_arr) != 1) {
            PyErr_SetString(PyExc_TypeError, "Input should be a single number or a 1d array");
            Py_XDECREF(t_arr);
            return NULL;
        }
        size = PyArray_DIM(t_arr, 0);
        npy_intp dims[2] = {size, 6};
        result = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    }
    if(!result) {
        Py_XDECREF(t_arr);
        return NULL;
    }

    double* data = static_cast<double*>(PyArray_DATA(result));
    double tmin = splx.xmin(), tmax = splx.xmax();  // NaN if the spline is empty
    for(int i=0; i<size; i++) {
        if(t_arr)  // pick up the next element of the input array of times
            t = pyArrayElem<double>(t_arr, i);
        else {}    // otherwise it was a single number and is already assigned
        if(t >= tmin && t <= tmax) {
            // splines provide the trajectory in the inertial frame,
            // need to transform the output to the rotated (but not *rotating*) frame
            double ca=1, sa=0, xi, vxi, yi, vyi;
            splx.evalDeriv(t, &xi, &vxi);
            sply.evalDeriv(t, &yi, &vyi);
            splz.evalDeriv(t, data + i*6 + 2, data + i*6 + 5);
            if(self->Omega != 0)
                math::sincos(self->Omega * t, sa, ca);
            data[i*6 + 0] =  xi * ca +  yi * sa;
            data[i*6 + 1] =  yi * ca -  xi * sa;
            data[i*6 + 3] = vxi * ca + vyi * sa;
            data[i*6 + 4] = vyi * ca - vxi * sa;
        } else {  // no extrapolation outside the time interval spanned by the orbit
            for(int c=0; c<6; c++)
                data[i*6+c] = NAN;
        }
    }
    Py_XDECREF(t_arr);
    return PyArray_Return(result);
}

PyObject* Orbit_name(OrbitObject* self)
{
    if(!self->x || !PyObject_TypeCheck(self->x, SplineTypePtr) || !((SplineObject*)(self->x))->spl) {
        PyErr_SetString(PyExc_RuntimeError, "Orbit is not properly initialized");
        return NULL;
    }
    const std::vector<double> &xvalues = ((SplineObject*)(self->x))->spl->xvalues();
    std::string result = "Orbit with " + utils::toString(xvalues.size()) + " nodes";
    if(!xvalues.empty())
        result += " on t=[" + utils::toString(xvalues.front(), 16) +
            ":" + utils::toString(xvalues.back(), 16) + "]";
    return Py_BuildValue("s", result.c_str());
}

PyObject* Orbit_elem(OrbitObject* self, Py_ssize_t index)
{
    if(!self->x || !PyObject_TypeCheck(self->x, SplineTypePtr) || !((SplineObject*)(self->x))->spl) {
        PyErr_SetString(PyExc_RuntimeError, "Orbit is not properly initialized");
        return NULL;
    }

    Py_ssize_t size = ((SplineObject*)(self->x))->spl->xvalues().size();
    if(index >= 0 && index < size)
        return Py_BuildValue("d",
            ((SplineObject*)(self->x))->spl->xvalues()[self->reversed ? size-1-index : index]);
    else {
        PyErr_SetString(PyExc_IndexError, "Orbit timestamp index out of range");
        return NULL;
    }
}

Py_ssize_t Orbit_len(OrbitObject* self)
{
    if(!self->x || !PyObject_TypeCheck(self->x, SplineTypePtr) || !((SplineObject*)(self->x))->spl)
        return -1;
    return ((SplineObject*)(self->x))->spl->xvalues().size();
}

static PySequenceMethods Orbit_sequence_methods = {
    (lenfunc)Orbit_len, 0, 0, (ssizeargfunc)Orbit_elem,
};

static PyMethodDef Orbit_methods[] = {
    { NULL }  // no named methods
};

static PyMemberDef Orbit_members[] = {
    { const_cast<char*>("x"), T_OBJECT_EX, offsetof(OrbitObject, x), READONLY,
      const_cast<char*>("interpolator for the x coordinate in the inertial frame") },
    { const_cast<char*>("y"), T_OBJECT_EX, offsetof(OrbitObject, y), READONLY,
      const_cast<char*>("interpolator for the y coordinate in the inertial frame") },
    { const_cast<char*>("z"), T_OBJECT_EX, offsetof(OrbitObject, z), READONLY,
      const_cast<char*>("interpolator for the z coordinate in the inertial frame") },
    { const_cast<char*>("Omega"), T_DOUBLE, offsetof(OrbitObject, Omega), READONLY,
      const_cast<char*>("angular frequency of the rotating frame") },
    { const_cast<char*>("reversed"), T_BOOL, offsetof(OrbitObject, reversed), READONLY,
      const_cast<char*>("whether the orbit was integrated backward in time") },
    { NULL }
};

static PyTypeObject OrbitType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "agama.Orbit",
    sizeof(OrbitObject), 0, (destructor)Orbit_dealloc,
    0, 0, 0, 0, (reprfunc)Orbit_name, 0, &Orbit_sequence_methods, 0, 0,
    (PyCFunctionWithKeywords)Orbit_value, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringOrbitClass,
    0, 0, 0, 0, 0, 0, Orbit_methods, Orbit_members, 0, 0, 0, 0, 0, 0,
    (initproc)Orbit_init
};

/// Construct an array of timestamps and trajectory points or a Python Orbit object.
/// \param[in]  dtype  is NPY_FLOAT, NPY_DOUBLE, NPY_CFLOAT, NPY_CDOUBLE or NPY_OBJECT;
/// \param[in]  traj   is the orbit (array of times and phase-space coordinates);
/// \param[in]  Omega  is the angular frequency for integration in the rotating frame
/// (needed because in this case v != dx/dt, so the derivatives for the splines need adjustment);
/// \param[in]  outputTime  is a flag indicating whether to create both time and trajectory arrays
/// (when true) or just the trajectory (when false, this is used for deviation vectors);
/// applies only to dtype!=NPY_OBJECT.
/// \param[in]  normFactor  is the additional multiplicative factor in unit conversion for dev.vectors
/// \param[in/out]  output  is the pointer to the first element of the output storage.
/// If dtype==NPY_OBJECT, the output array contains one PyObject* element per orbit,
/// which will be filled with a newly created instance of Orbit class encapsulating
/// the interpolators for the trajectory (unit-converted).
/// In other cases, the output array will be populated with one (if outputTime==false) or two
/// new PyObject* elements per orbit: the numpy array of timestamps (only if outputTime==true)
/// and another numpy array of phase-space points.
/// \return  true on success, false if failed to allocate arrays
/// (in this case, PyErr_SetString is called to report the error)
bool createOrbit(int dtype, const orbit::Trajectory &traj,
    double Omega, bool outputTime, double normFactor, /*output*/ PyObject* output[])
{
    npy_intp size = traj.size(), sizespl = 0;
    if(dtype == NPY_OBJECT) {
        // create an Orbit object containing three spline interpolators
        OrbitObject* orbit = NULL;
        {   // avoid concurrent non-readonly access to Python C API
            PyAcquireGIL lock;
            orbit = (OrbitObject*)PyObject_New(PyObject, &OrbitType);
            if(orbit) {
                // allocate three new Python Spline objects
                orbit->x = PyObject_New(PyObject, &SplineType);
                orbit->y = PyObject_New(PyObject, &SplineType);
                orbit->z = PyObject_New(PyObject, &SplineType);
                // make sure the smart pointers are initialized with zero before re-assigning them
                if(orbit->x) {
                    new (&(((SplineObject*)orbit->x)->spl)) math::PtrFunction;
                    new (&(((SplineObject*)orbit->x)->name)) std::string;
                }
                if(orbit->y) {
                    new (&(((SplineObject*)orbit->y)->spl)) math::PtrFunction;
                    new (&(((SplineObject*)orbit->y)->name)) std::string;
                }
                if(orbit->z) {
                    new (&(((SplineObject*)orbit->z)->spl)) math::PtrFunction;
                    new (&(((SplineObject*)orbit->z)->name)) std::string;
                }
                if(!orbit->x || !orbit->y || !orbit->z) {
                    Py_DECREF(orbit);  // this also XDECREFs and deletes all three Spline objects
                    orbit = NULL;
                }
            }
        }  // end critical session, the remaining operations are thread-safe
        if(!orbit)
            return false;
        std::vector<double> t(size), x(size), vx(size), y(size), vy(size), z(size), vz(size);
        orbit->Omega = Omega * conv->timeUnit;  // frequency is in user units
        // if the orbit is integrated backward in time, we need to reverse the order of points
        orbit->reversed = size>1 && traj[1].second < traj[0].second;
        for(npy_intp index=0; index<size; index++) {
            double point[6];
            unconvertPosVel(traj[orbit->reversed ? size-1-index : index].first, point);
            double ti = traj[orbit->reversed ? size-1-index : index].second / conv->timeUnit;
            // skip points with identical timestamps, since otherwise spline construction would fail
            if(sizespl > 0 && ti == t[sizespl-1])
                continue;
            t [sizespl] = ti;
            // if the orbit was provided in the rotating frame, transform it to a non-rotating one
            // to construct the interpolators, and transform their output back to rotated frame
            double ca, sa;
            math::sincos(orbit->Omega * ti, sa, ca);
            x [sizespl] = normFactor * (point[0] * ca - point[1] * sa);
            y [sizespl] = normFactor * (point[1] * ca + point[0] * sa);
            z [sizespl] = normFactor *  point[2];
            vx[sizespl] = normFactor * (point[3] * ca - point[4] * sa);
            vy[sizespl] = normFactor * (point[4] * ca + point[3] * sa);
            vz[sizespl] = normFactor *  point[5];
            sizespl++;
        }
        if(sizespl != size) {
            t .resize(sizespl);
            x .resize(sizespl);
            y .resize(sizespl);
            z .resize(sizespl);
            vx.resize(sizespl);
            vy.resize(sizespl);
            vz.resize(sizespl);
        }
        try{
            ((SplineObject*)orbit->x)->spl.reset(new math::QuinticSpline(t, x, vx));
            ((SplineObject*)orbit->y)->spl.reset(new math::QuinticSpline(t, y, vy));
            ((SplineObject*)orbit->z)->spl.reset(new math::QuinticSpline(t, z, vz));
            ((SplineObject*)orbit->x)->name = "Quintic spline";
            ((SplineObject*)orbit->y)->name = "Quintic spline";
            ((SplineObject*)orbit->z)->name = "Quintic spline";
            FILTERMSG(utils::VL_DEBUG, "Agama", "Created three quintic splines of size " +
                utils::toString(x.size()) +
                " at "+utils::toString(((SplineObject*)orbit->x)->spl.get()) +
                ", "  +utils::toString(((SplineObject*)orbit->y)->spl.get()) +
                ", "  +utils::toString(((SplineObject*)orbit->z)->spl.get()));
        }
        catch(std::exception& ex) {
            PyAcquireGIL lock;
            Py_DECREF(orbit);
            raisePythonException(ex);
            return false;
        }
        *output = (PyObject*)orbit;   // place the pointer for the newly created Orbit
    } else {   // dtype is anything but NPY_OBJECT
        npy_intp dims[] = {size, dtype==NPY_CFLOAT || dtype==NPY_CDOUBLE ? 3 : 6};
        PyObject *time_obj = NULL, *traj_obj = NULL;
        {   // access to Python C API for one thread at a time
            PyAcquireGIL lock;
            if(outputTime)
                time_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
            traj_obj = PyArray_SimpleNew(2, dims, dtype);
            if((outputTime && !time_obj) || !traj_obj) {
                Py_XDECREF(time_obj);
                Py_XDECREF(traj_obj);
                return false;
            }
        }  // end critical section

        for(npy_intp index=0; index<size; index++) {
            if(outputTime) {
                // convert the time array
                pyArrayElem<double>(time_obj, index) = traj[index].second / conv->timeUnit;
            }
            // convert the units and numerical type of the trajectory array
            double point[6];
            unconvertPosVel(traj[index].first, point);
            switch(dtype) {
                case NPY_DOUBLE:
                    for(int c=0; c<6; c++)
                        pyArrayElem<double>(traj_obj, index, c) = normFactor * point[c];
                    break;
                case NPY_FLOAT:
                    for(int c=0; c<6; c++)
                        pyArrayElem<float>(traj_obj, index, c) =
                            static_cast<float>(normFactor * point[c]);
                    break;
                case NPY_CDOUBLE:
                    for(int c=0; c<3; c++)
                        pyArrayElem<std::complex<double> >(traj_obj, index, c) =
                            std::complex<double>(normFactor * point[c+0], normFactor * point[c+3]);
                    break;
                case NPY_CFLOAT:
                    for(int c=0; c<3; c++) {
                        pyArrayElem<std::complex<float> >(traj_obj, index, c) =
                            std::complex<float>(normFactor * point[c+0], normFactor * point[c+3]);
                    }
                    break;
                default:
                    assert(!"incorrect dtype");  // shouldn't happen, we checked dtype beforehand
            }
        }
        // in this case, the output array has one or two elements per orbit
        if(outputTime) {
            output[0] = time_obj;
            output[1] = traj_obj;
        } else {
            output[0] = traj_obj;
        }
    }
    return true;
}

/// Apply a Target object to a stored orbit represented by interpolating splines:
/// the procedure is identical to the one used during orbit integration, namely the addPoint
/// method of the Target object is called on NUM_SAMPLES_PER_STEP equally-spaced points
/// for each timestep, but these points are retrieved from the interpolating spline
/// rather than the ODE integrator. The result should be identical up to interpolation errors.
void applyTargetToOrbit(const galaxymodel::BaseTarget& target, OrbitObject* orbit, double normFactor,
    /*output*/ galaxymodel::StorageNumT *output)
{
    const math::BaseInterpolator1d
        &splx = *((SplineObject*)(orbit->x))->spl,
        &sply = *((SplineObject*)(orbit->y))->spl,
        &splz = *((SplineObject*)(orbit->z))->spl;
    const std::vector<double>& times = splx.xvalues();
    math::Matrix<double> datacube = target.newDatacube();
    double *dataptr = datacube.data();
    const int NUM_SAMPLES_PER_STEP = galaxymodel::RuntimeFncTarget::NUM_SAMPLES_PER_STEP;
    for(size_t i=1; i<times.size(); i++) {
        double substep = (times[i] - times[i-1]) / NUM_SAMPLES_PER_STEP;
        for(int s=0; s<NUM_SAMPLES_PER_STEP; s++) {
            // equally-spaced samples in time, offsets from the beginning of the current timestep
            double t = times[i-1] + (s+0.5) * substep;
            double point[6];  // position and velocity in cartesian coordinates at the current sub-step
            double ca=1, sa=0, xi, vxi, yi, vyi;
            splx.evalDeriv(t, &xi, &vxi);
            sply.evalDeriv(t, &yi, &vyi);
            splz.evalDeriv(t, &point[2], &point[5]);
            if(orbit->Omega != 0)
                math::sincos(orbit->Omega * t, sa, ca);
            point[0] =  xi * ca +  yi * sa;
            point[1] =  yi * ca -  xi * sa;
            point[3] = vxi * ca + vyi * sa;
            point[4] = vyi * ca - vxi * sa;
            // spline-interpolated orbit is stored in physical units, so we need to perform
            // unit conversion before passing the point to the Target object
            convertPosVel(point).unpack_to(point);  // in-place conversion
            target.addPoint(point, substep, dataptr);
        }
    }
    target.finalizeDatacube(datacube, output);
    const double mult = normFactor / (times.back() - times.front());
    for(npy_intp i=0, size=target.numCoefs(); i<size; i++)
        output[i] *= mult;
}


///@}
//  ----------------------------------------
/// \name  Target class for additive models
//  ----------------------------------------
///@{

static const char* docstringTarget =
    "Target objects represent various targets that need to be satisfied by an additive model.\n"
    "The type of target is specified by the  type='...' argument, and other available arguments "
    "depend on it. "
    "Below follows the list of possible target types and their parameters (all case-insensitive).\n\n"
    "Density discretization models:\n\n"
    "  DensityClassicTopHat\n"
    "Classical approach for spheroidal Schwarzschild models: "
    "radial shells divided into three panes, each pane - into several strips, "
    "and the density inside each resulting cell is approximated as a constant.\n\n"
    "  DensityClassicLinear\n"
    "Same as classical, but the density is specified at the grid nodes and interpolated "
    "tri-linearly within each cell.\n"
    "Parameters for both of these models:\n"
    "gridr - array of radial grid nodes;\n"
    "stripsPerPane - number of strips in each of three panes in each shell "
    "(hence the number of cells in each pane is stripsPerPane^2);\n"
    "axisRatioY, axisRatioZ - (optional) coefficients for squeezing the grid in each dimension.\n\n"
    "  DensitySphHarm\n"
    "Radial grid is the same as in the classic approach, but the angular dependence "
    "of the density is represented in terms of spherical-harmonic expansion, "
    "and the radial dependence of each term is a linearly-interpolated function.\n"
    "Parameters:\n"
    "gridr - array of radial grid nodes;\n"
    "lmax, mmax - order of angular expansion in theta and phi, respectively.\n\n"
    "  DensityCylindricalTopHat\n"
    "A grid in meridional plane (R,z) aligned with cylindrical coordinates, "
    "with the azimuthal dependence of the density represented by a Fourier expansion; "
    "the density is attributed to each cell (implicitly assumed to be constant within a cell).\n\n"
    "  DensityCylindricalLinear\n"
    "Same as the previous one, but each azimuthal Fourier term in the meridional plane (R,z) "
    "is bi-linearly interpolated within each cell.\n"
    "Parameters for both of these models:\n"
    "gridr, gridz - arrays of grid nodes in cylindrical radius and vertical coordinate "
    "(should cover only the positive quadrant);\n"
    "mmax - order of azmuthal Fourier expansion.\n\n"
    "Kinematic constraints:\n\n"
    "  KinemShell\n"
    "This target object records the density-weighted radial and tangential velocity dispersion "
    "in spherical coordinates, represented as projections onto basis elements of a B-spline grid "
    "in radius.\n"
    "Parameters:\n"
    "degree (integer from 0 to 3) - degree of B-splines;\n"
    "gridr - array of radial grid nodes.\n"
    "These recorded projections may be used to constrain the velocity anisotropy of the model: "
    "after choosing the desired value of coefficient beta = 1 - sigma_t^2 / (2 sigma_r^2), "
    "one demands that\n2 (1-beta) * rho * sigma_r^2 - rho * sigma_t^2 = 0\nfor all radial points.\n\n"
    "  LOSVD\n"
    "This target object is intended for recording line-of-sight velocity distributions in a 2d image "
    "plane; the latter is denoted by coordinates X, Y, with Y pointing up/north and X - left/east, "
    "and the complementary Z axis points along the line of sight away from the observer.\n"
    "The orientation of the image plane w.r.t. the intrinsic coordinate system x,y,z of the galaxy "
    "is described by three Euler angles alpha, beta and gamma. "
    "The first two angles define the line of sight: alpha is the angle between the x axis and "
    "the line of nodes (the intersection between the equatorial (xy) plane of the galaxy and "
    "the image plane); "
    "beta is the inclination angle (beta=0 is face-on, with these two planes coinciding, beta=pi/2 "
    "is edge-on, and beta=pi is again face-on but with z axis pointing towards the observer, "
    "opposite to Z). "
    "The third angle gamma defines additional rotation of the image plane: it is the position angle "
    "(PA) of the projection of z axis onto the image plane, measured counter-clockwise from the Y "
    "(up/north) axis towards the X (left/east) axis. The PA of the line of nodes is gamma+pi/2. "
    "For instance, the projections along each of the principal axes are given by the following "
    "triplets of angles: \n"
    "(0, 0, 0) - face-on orientation, so that the image plane (X,Y) coincides with the (x,y) plane, "
    "and the Z axis (line of sight) coincides with the z axis.\n"
    "(0, pi/2, 0) - view down the y axis, image plane (X,Y) coincides with the (x,z) plane;\n"
    "(pi/2, pi/2, 0) - view down the y axis, image plane coincides with the (y,z) plane.\n"
    "LOSVDs are one-dimensional functions that describe the distribution of matter moving with "
    "the given velocity v_Z in the given spatial region (aperture) in the (X,Y) plane. "
    "Apertures are specified by arbitrary simple polygons (without self-intersection, but not "
    "necessarily convex); different apertures may overlap in the same area. Any collection of slits "
    "at various angles, regular 2d IFU spaxels, or Voronoi bins can be represented in this way. "
    "In each aperture, the LOSVD is represented in terms of a B-spline expansion: "
    "the degree of B-splines and the grid nodes in the velocity axis that together determine "
    "the shape of basis functions are the same for all apertures, and the amplitudes of each basis "
    "function are the free parameters that describe each LOSVD. For instance, the commonly used "
    "approach to represent a LOSVD as a histogram with regularly-spaced bins is equivalent to "
    "a 0th degree B-spline over a uniform grid in v; however, this is certainly not the most "
    "efficient usage scenario. Higher-degree B-spines (2 or 3) result in smoother LOSVDs and need "
    "substantially fewer grid points to achieve the same velocity resolution; in addition, the grid "
    "needs not be uniformly-spaced.\n"
    "The amplitudes of B-spline expansion in each aperture for each orbit or N-body snapshot "
    "are computed by first constructing a datacube - the projection of said orbit onto "
    "an auxiliary grid in the 2+1-dimensional space (X,Y and v_LOS). The grid in velocity space "
    "is the same as used in the B-spline representation, but the grid in the image plane is "
    "somewhat arbitrary (separable in X,Y, but not necessarily uniform); the only requirement "
    "is for it to cover all apertures, and have a sufficient spatial resolution - typically "
    "comparable to the PSF width if 2nd or 3rd-degree B-splines are used. "
    "Then the datacube is convolved with spatial and velocity-space PSFs and re-binned into "
    "the apertures (all done internally by the Target object); the final LOSVD has "
    "numApertures * numBasisFnc elements, where the latter is len(gridv) + degree - 1.\n"
    "Parameters for this target:\n"
    "alpha,beta,gamma - angles specifying the orientation of the image plane w.r.t. model coordinates;\n"
    "degree (integer from 0 to 3) - degree of B-splines;\n"
    "gridv - array of grid nodes in velocity space (typically should be symmetric about origin);\n"
    "gridx - nodes of the auxiliary (internal) grid in the X coordinate of the image plane;\n"
    "gridy (optional - default is the same as gridx) - nodes of the internal grid in Y; "
    "the spatial region covered by this 2d grid should encompass all apertures, and the grid spacing "
    "should be comparable to either the PSF width or the typical aperture size, but not necessarily "
    "uniform (i.e., it may be constructed with the routines 'nonuniformGrid' or 'symmetricGrid');\n"
    "psf - description of spatial point-spread function: it may be either a single number, "
    "interpreted as the width of the Gaussian PSF, or an Kx2 array describing a composition of K "
    "such Gaussians (the first column is the width of each Gaussian, and the second column is "
    "the relative fraction of this component, which should sum up to unity);\n"
    "velpsf - width of the velocity-space smoothing kernel (a single Gaussian);\n"
    "apertures - array of polygons describing the boundaries of each aperture: "
    "each element of this array is a 2d array with X,Y coordinates of the polygon vertices, "
    "and of course the number of vertices may be different for each polygon (but greater than two).\n"
    "symmetry - a letter encoding the symmetries of the potential and orbit shapes, "
    "determines how the points sampled from the trajectory of each orbit will be treated before "
    "projecting them onto the image plane: 't' (triaxial, default) means that only the fourfold "
    "reflection symmetry  z <-> -z  and  x,y <-> -x,-y  will be enforced,  'a' (axisymmetric) means "
    "that the point will be rotated about the z axis by a random angle, and 's' (spherical) means "
    "that both spherical angles will be randomized.\n\n"
    "The role of a Target object is to collect data during the construction of an orbit library: "
    "several instances of them could be provided as a 'targets=[t1,t2,...]' argument of "
    "the 'orbit()' routine, and each one will produce a matrix with Norbit rows and Ncoef columns, "
    "where the number of coefficients for a target t1 is given by the 'len(t1)' function.\n"
    "A Target instance can also be used to produce the right-hand side of the matrix equation "
    "in a linear superposition model, by applying it to a Density object or an N-body snapshot:\n"
    ">>> den=agama.Density(params)     # create an instance of a density model with some parameters\n"
    ">>> snapshot=den.sample(10000)    # draw 10000 sample points from this model\n"
    ">>> rhs_d=t1(den)                 # apply the target 't1' to the analytic density model\n"
    ">>> rhs_s=t1(snapshot)            # apply it to an array of particles\n"
    "The result depends on the type of the target and the type of the argument (density or array "
    "of particles). For density discretization targets, this produces the array of integrals of "
    "the density profile multiplied by the basis functions over the entire volume, or, if the input "
    "is an array of particles, the sum of particle masses multiplied by basis functions. "
    "The length of the resulting array is equal to len(t1). "
    "For a LOSVD target, applying it to a density object computes the integrals of surface mass "
    "density over each aperture, convolved with the spatial PSF; equivalently, this is the overall "
    "normalization of the LOSVD in each aperture, i.e. the integral of f(v) over all velocities. ";
    //"This number may be used as the normalization factor in computing Gauss-Hermite coefficients "
    //"from LOSVD, as shown below:\n";  // TODO expand

/// \cond INTERNAL_DOCS
/// Python type corresponding to Target class
typedef struct {
    PyObject_HEAD
    galaxymodel::PtrTarget target;
    // dimensional unit conversion factor for applying the Target to a Density object
    double unitDensityProjection;
    // same factor for a GalaxyModel object, an N-body snapshot, or during orbit integration
    double unitDFProjection;
} TargetObject;
/// \endcond

void Target_dealloc(TargetObject* self)
{
    if(self->target)
        FILTERMSG(utils::VL_DEBUG, "Agama", "Deleted " + std::string(self->target->name()) +
            " target at " + utils::toString(self->target.get()));
    else
        FILTERMSG(utils::VL_DEBUG, "Agama", "Deleted an empty target");
    self->target.reset();
    Py_TYPE(self)->tp_free(self);
}

int Target_init(TargetObject* self, PyObject* args, PyObject* namedArgs)
{
    if(self->target) {
        PyErr_SetString(PyExc_RuntimeError, "Target object cannot be reinitialized");
        return -1;
    }
    if(!onlyNamedArgs(args, namedArgs))
        return -1;
    NamedArgs nargs(namedArgs);
    std::string type_str = toString(nargs.pop("type"));
    if(type_str.empty()) {
        PyErr_SetString(PyExc_TypeError, "Must provide a type='...' argument");
        return -1;
    }
    try{
        if(utils::stringsEqual(type_str.substr(0, 7), "Density")) {
            // spatial grids
            std::vector<double> gridr = toDoubleArray(nargs.pop("gridr"));
            std::vector<double> gridz = toDoubleArray(nargs.pop("gridz"));
            if(gridr.size()<2)
                throw std::invalid_argument("gridr must be an array with >=2 elements");
            if(gridz.size()<2 && utils::stringsEqual(type_str.substr(0, 18), "DensityCylindrical"))
                throw std::invalid_argument("gridz must be an array with >=2 elements");
            math::blas_dmul(conv->lengthUnit, gridr);
            math::blas_dmul(conv->lengthUnit, gridz);
            // orders of angular expansion or number of lines partitioning a spherical shell into cells
            int lmax = toInt(nargs.pop("lmax"), 0),
                mmax = toInt(nargs.pop("mmax"), 0),
                stripsPerPane = toInt(nargs.pop("stripsPerPane"), 2);
            // flattening of the spheroidal grid
            double
                axisRatioY = toDouble(nargs.pop("axisRatioY"), 1.0),
                axisRatioZ = toDouble(nargs.pop("axisRatioZ"), 1.0);
            if(utils::stringsEqual(type_str, "DensityClassicTopHat"))
                self->target.reset(new galaxymodel::TargetDensityClassic<0>(
                    stripsPerPane, gridr, axisRatioY, axisRatioZ));
            else if(utils::stringsEqual(type_str, "DensityClassicLinear"))
                self->target.reset(new galaxymodel::TargetDensityClassic<1>(
                    stripsPerPane, gridr, axisRatioY, axisRatioZ));
            else if(utils::stringsEqual(type_str, "DensitySphHarm"))
                self->target.reset(new galaxymodel::TargetDensitySphHarm(lmax, mmax, gridr));
            else if(utils::stringsEqual(type_str, "DensityCylindricalTopHat"))
                self->target.reset(new galaxymodel::TargetDensityCylindrical<0>(mmax, gridr, gridz));
            else if(utils::stringsEqual(type_str, "DensityCylindricalLinear"))
                self->target.reset(new galaxymodel::TargetDensityCylindrical<1>(mmax, gridr, gridz));
            else
                throw std::invalid_argument("Unknown type='...' argument");
            self->unitDensityProjection = conv->massUnit;
            self->unitDFProjection = conv->massUnit;
        }

        // check if a KinemShell is being requested
        if(utils::stringsEqual(type_str, "KinemShell")) {
            int degree = toInt(nargs.pop("degree"), -1);
            std::vector<double> gridr = toDoubleArray(nargs.pop("gridr"));
            if(gridr.size()<2)
                throw std::invalid_argument("gridr must be an array with >=2 elements");
            math::blas_dmul(conv->lengthUnit, gridr);
            switch(degree) {
                case 0: self->target.reset(new galaxymodel::TargetKinemShell<0>(gridr)); break;
                case 1: self->target.reset(new galaxymodel::TargetKinemShell<1>(gridr)); break;
                case 2: self->target.reset(new galaxymodel::TargetKinemShell<2>(gridr)); break;
                case 3: self->target.reset(new galaxymodel::TargetKinemShell<3>(gridr)); break;
                default:
                    throw std::invalid_argument(
                        "KinemShell: degree of interpolation should be between 0 and 3");
            }
            self->unitDensityProjection = NAN;
            self->unitDFProjection = conv->massUnit * pow_2(conv->velocityUnit);
        }

        // check if a LOSVD is being requested
        if(utils::stringsEqual(type_str, "LOSVD")) {
            galaxymodel::LOSVDParams params;
            // parameters describing the orientation of the model
            params.alpha = toDouble(nargs.pop("alpha"), params.alpha);
            params.beta  = toDouble(nargs.pop("beta" ), params.beta);
            params.gamma = toDouble(nargs.pop("gamma"), params.gamma);
            // parameters of the internal grids in image plane and line-of-sight velocity
            params.gridx = toDoubleArray(nargs.pop("gridx"));
            params.gridy = toDoubleArray(nargs.pop("gridy"));
            params.gridv = toDoubleArray(nargs.pop("gridv"));
            if(params.gridy.empty())
                params.gridy = params.gridx;
            if(params.gridx.size()<2 || params.gridy.size()<2 || params.gridv.size()<2)
                throw std::invalid_argument("gridx, [gridy, ] gridv must be arrays with >=2 elements");
            math::blas_dmul(conv->lengthUnit, params.gridx);
            math::blas_dmul(conv->lengthUnit, params.gridy);
            math::blas_dmul(conv->velocityUnit, params.gridv);
            // explicitly specified symmetry (triaxial by default)
            params.symmetry = potential::getSymmetryTypeByName(toString(nargs.pop("symmetry")));
            // parameters of the point-spread functions (spatial and velocity)
            PyObject* psf_obj = nargs.pop("psf");
            if(psf_obj) {
                double psf = toDouble(psf_obj, NAN) * conv->lengthUnit;
                if(isFinite(psf))
                    params.spatialPSF.assign(1, galaxymodel::GaussianPSF(psf));
                else {  // may be an array of several PSFs
                    PyArrayObject* psf_arr = (PyArrayObject*)PyArray_FROM_OTF(psf_obj, NPY_DOUBLE, 0);
                    if(psf_arr == NULL || PyArray_NDIM(psf_arr) != 2 || PyArray_DIM(psf_arr, 1) != 2) {
                        Py_XDECREF(psf_arr);
                        throw std::invalid_argument(
                            "Argument 'psf' must be a single number (width of the Gaussian PSF), "
                            "or a Kx2 array of PSF widths and fractional weights");
                    }
                    for(npy_intp k=0; k<PyArray_DIM(psf_arr, 0); k++)
                        params.spatialPSF.push_back(galaxymodel::GaussianPSF(
                            pyArrayElem<double>(psf_arr, k, 0) * conv->lengthUnit,
                            pyArrayElem<double>(psf_arr, k, 1)));
                }
            }  // otherwise no PSF is assigned at all
            params.velocityPSF = toDouble(nargs.pop("velpsf"), 0.) * conv->velocityUnit;
            // apertures in the image plane where LOSVDs are analyzed
            std::vector<PyObject*> apertures = toPyObjectArray(nargs.pop("apertures"));
            if(apertures.empty())
                throw std::invalid_argument("Must provide a list of polygons in 'apertures=...' argument");
            // two possibilities: either apertures is a single 3d array of shape N_apert x N_vert x 2,
            // or a list/tuple/array of individual 2d arrays of shapes N_vert,i x 2
            for(size_t a=0; a<apertures.size(); a++) {
                PyArrayObject* ap_arr =
                    (PyArrayObject*)PyArray_FROM_OTF(apertures[a], NPY_DOUBLE, 0);
                if( ap_arr != NULL &&
                    PyArray_NDIM(ap_arr)  == 2 &&
                    PyArray_DIM(ap_arr, 0) >= 3 &&
                    PyArray_DIM(ap_arr, 1) == 2)
                {   // an element of the list is a 2d array
                    size_t nv = PyArray_DIM(ap_arr, 0);
                    math::Polygon poly(nv);
                    for(size_t v=0; v<nv; v++) {
                        poly[v].x = pyArrayElem<double>(ap_arr, v, 0) * conv->lengthUnit;
                        poly[v].y = pyArrayElem<double>(ap_arr, v, 1) * conv->lengthUnit;
                    }
                    params.apertures.push_back(poly);
                } else
                if( ap_arr != NULL &&
                    apertures.size() == 1 &&
                    PyArray_NDIM(ap_arr)  == 3 &&
                    PyArray_DIM(ap_arr, 1) >= 3 &&
                    PyArray_DIM(ap_arr, 2) == 2)
                {   // the entire input is a single 3d array
                    size_t na = PyArray_DIM(ap_arr, 0), nv = PyArray_DIM(ap_arr, 1);
                    math::Polygon poly(nv);
                    for(size_t aa=0; aa<na; aa++) {
                        for(size_t v=0; v<nv; v++) {
                            poly[v].x = pyArrayElem<double>(ap_arr, aa, v, 0) * conv->lengthUnit;
                            poly[v].y = pyArrayElem<double>(ap_arr, aa, v, 1) * conv->lengthUnit;
                        }
                        params.apertures.push_back(poly);
                    }
                } else {
                    if(PyErr_Occurred())
                        return -1;
                    Py_XDECREF(ap_arr);
                    throw std::invalid_argument(
                        "Each element of the list or tuple provided in the 'apertures=...' argument "
                        "must be a Nx2 array defining a polygon on the sky plane, with N>=3 vertices");
                }
                Py_DECREF(ap_arr);
            }
            // degree of B-splines
            int degree = toInt(nargs.pop("degree"), -1);
            switch(degree) {
                case 0: self->target.reset(new galaxymodel::TargetLOSVD<0>(params)); break;
                case 1: self->target.reset(new galaxymodel::TargetLOSVD<1>(params)); break;
                case 2: self->target.reset(new galaxymodel::TargetLOSVD<2>(params)); break;
                case 3: self->target.reset(new galaxymodel::TargetLOSVD<3>(params)); break;
                default:
                    throw std::invalid_argument(
                        "LOSVD: degree of interpolation should be between 0 and 3");
            }
            self->unitDensityProjection = conv->massUnit;
            self->unitDFProjection = conv->massUnit / conv->velocityUnit;
        }
        if(!self->target)  // none of the above variants worked
            throw std::invalid_argument("Unknown type='...' argument");
    }
    catch(std::exception& ex) {
        raisePythonException(ex, "Error in creating a Target object: ");
        return -1;
    }
    if(!nargs.empty())
        return -1;
    FILTERMSG(utils::VL_DEBUG, "Agama", "Created " + std::string(self->target->name()) +
        " target at " + utils::toString(self->target.get()));
    unitsWarning = true;  // any subsequent call to setUnits() will raise a warning
    return 0;
}

PyObject* Target_value(TargetObject* self, PyObject* args, PyObject* namedArgs)
{
    if(!PyTuple_Check(args) || PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "Expected exactly 1 argument");
        return NULL;
    }
    if(!noNamedArgs(namedArgs))
        return NULL;
    const char* errorstr = "Argument must be an instance of Density, GalaxyModel, "
        "one or several Orbit instances, or an array of particles "
        "(a tuple with two elements - Nx6 position/velocity coordinates and N masses)";

    PyObject* arg = PyTuple_GET_ITEM(args, 0);
    particles::ParticleArrayCar particles;
    npy_intp size = self->target->numCoefs();
    try{

        // check if the input is an instance of Orbit: output is a 1d array of length "size"
        if(PyObject_IsInstance(arg, (PyObject*) OrbitTypePtr)) {
            PyObject* result = PyArray_ZEROS(1, &size, STORAGE_NUM_T, 0);
            if(!result)
                return NULL;
            applyTargetToOrbit(*self->target, (OrbitObject*) arg,
                conv->massUnit / self->unitDFProjection,
                &pyArrayElem<galaxymodel::StorageNumT>(result, 0));
            return result;
        }

        // or it may be an array of Orbits: output is a 2d array of shape "numOrbits * size",
        // and contains nearly the same data as when the Target is used during orbit integration
        if(PyArray_Check(arg)) {
            PyArrayObject* arr = (PyArrayObject*)arg;
            bool valid = PyArray_NDIM(arr) == 1 &&  PyArray_TYPE(arr) == NPY_OBJECT;
            npy_intp numOrbits = valid ? PyArray_DIM(arr, 0) : 0;
            for(npy_intp i=0; valid && i<numOrbits; i++)
                valid &= PyObject_IsInstance(pyArrayElem<PyObject*>(arr, i), (PyObject*) OrbitTypePtr);
            if(valid) {
                npy_intp dims[2] = {numOrbits, size};
                PyObject* result = PyArray_ZEROS(2, dims, STORAGE_NUM_T, 0);
                if(!result)
                    return NULL;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for(int i=0; i<numOrbits; i++) {
                    applyTargetToOrbit(*self->target, pyArrayElem<OrbitObject*>(arr, i),
                        conv->massUnit / self->unitDFProjection,
                        &pyArrayElem<galaxymodel::StorageNumT>(result, i, 0));
                }
                return result;
            }
        }

        // check if we have a density object as input
        potential::PtrDensity dens = getDensity(arg);
        if(dens) {
            std::vector<double> result;
            {
                // this operation contains OpenMP-parallelized loops, so we need to release GIL
                PyReleaseGIL unlock;
                result = self->target->computeDensityProjection(*dens);
            }
            math::blas_dmul(1./self->unitDensityProjection, result);
            return toPyArray(result);
        }

        // otherwise we may have a GalaxyModel object as input
        if(PyObject_IsInstance(arg, (PyObject*) &GalaxyModelType)) {
            std::vector<galaxymodel::StorageNumT> result(self->target->numCoefs());
            {   // same remark here
                PyReleaseGIL unlock;
                self->target->computeDFProjection(galaxymodel::GalaxyModel(
                    *((GalaxyModelObject*)arg)->pot_obj->pot,
                    *((GalaxyModelObject*)arg)->af_obj->af,
                    *((GalaxyModelObject*)arg)->df_obj->df),
                    &result[0]);
            }
            math::blas_dmul(1./self->unitDFProjection, result);
            return toPyArray(result);
        }

        // otherwise this must be a particle object, which will be processed further down
        if(!PyTuple_Check(arg) || PyTuple_Size(arg)!=2) {
            PyErr_SetString(PyExc_TypeError, errorstr);
            return NULL;
        }
        particles = convertParticles<coord::PosVelCar>(arg);
        if(particles.size() == 0) {
            PyErr_SetString(PyExc_TypeError, errorstr);
            return NULL;
        }
    }
    catch(std::exception& ex) {
        raisePythonException(ex);
        return NULL;
    }

    // now work with the input particle array
    PyObject* result = PyArray_ZEROS(1, &size, STORAGE_NUM_T, 0);
    if(!result)
        return NULL;
    std::string errorMessage;
    // the loop below is parallelized, but it does not involve any Python C API functions,
    // so we do not need to release GIL (although we could...)
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        try{
            // define thread-local intermediate matrices
            math::Matrix<double> datacube = self->target->newDatacube();
            std::vector<galaxymodel::StorageNumT> tmpresult(size);
            const double mult = 1./self->unitDFProjection;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for(int i=0; i<(int)particles.size(); i++) {
                double xv[6];
                particles.point(i).unpack_to(xv);
                self->target->addPoint(xv, particles.mass(i), datacube.data());
            }
            self->target->finalizeDatacube(datacube, &tmpresult[0]);
            // now sum up the thread-local intermediate matrices elementwise,
            // but in such a way that the order is always the same, to ensure that
            // the result is also deterministic (floating-point summation is not commutative)
#ifdef _OPENMP
#pragma omp for ordered schedule(static)
            for(int t=0; t<omp_get_num_threads(); t++)
#endif
            {
#ifdef _OPENMP
#pragma omp ordered
#endif
                {
                    for(npy_intp i=0; i<size; i++)
                        pyArrayElem<galaxymodel::StorageNumT>(result, i) += mult * tmpresult[i];
                }
            }
        }
        catch(std::exception& ex) {
            errorMessage = ex.what();
        }
    }

    if(!errorMessage.empty()) {
        raisePythonException(errorMessage);
        Py_DECREF(result);
        return NULL;
    }
    return result;
}

PyObject* Target_name(PyObject* self)
{
    return Py_BuildValue("s", ((TargetObject*)self)->target->name());
}

PyObject* Target_elem(TargetObject* self, Py_ssize_t index)
{
    try{
        return Py_BuildValue("s", self->target->coefName(index).c_str());
    }
    catch(std::exception& ex) {
        raisePythonException(ex);
        return NULL;
    }
}

Py_ssize_t Target_len(TargetObject* self)
{
    return self->target->numCoefs();
}

static PySequenceMethods Target_sequence_methods = {
    (lenfunc)Target_len, 0, 0, (ssizeargfunc)Target_elem,
};
static PyMethodDef Target_methods[] = {
    { NULL }  // no named methods
};

static PyTypeObject TargetType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "agama.Target",
    sizeof(TargetObject), 0, (destructor)Target_dealloc,
    0, 0, 0, 0, Target_name, 0, &Target_sequence_methods, 0, 0,
    (PyCFunctionWithKeywords)Target_value, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringTarget,
    0, 0, 0, 0, 0, 0, Target_methods, 0, 0, 0, 0, 0, 0, 0,
    (initproc)Target_init
};


///@}
//  -------------------------------------
/// \name  Routine for orbit integration
//  -------------------------------------
///@{

/// description of orbit function
static const char* docstringOrbit =
    "Compute a single orbit or a bunch of orbits in the given potential.\n"
    "Named arguments:\n"
    "  ic:  initial conditions - either an array of 6 numbers (3 positions and 3 velocities in "
    "Cartesian coordinates) for a single orbit, or a 2d array of Nx6 numbers for a bunch of orbits.\n"
    "  potential:  a Potential object or a compatible interface.\n"
    "  Omega (optional, default 0):  pattern speed of the rotating frame.\n"
    "  time:  total integration time - may be a single number (if computing a single orbit "
    "or if it is identical for all orbits), or an array of length N (for a bunch of orbits).\n"
    "  timestart (optional, default 0):  initial time for the integration (only matters if "
    "the potential is time-dependent). The final time is thus timestart+time. "
    "May be a single number (for one orbit or if it is identical for all orbits), "
    "or an array of length N (for a bunch of orbits).\n"
    "  targets (optional):  zero or more instances of Target class (a tuple/list if more than one); "
    "each target collects its own data for each orbit.\n"
    "  trajsize (optional):  if given, turns on the recording of trajectory for each orbit "
    "(should be either a single integer or an array of integers with length N). "
    "The trajectory of each orbit is stored either at every timestep of the integrator "
    "(if trajsize=0) or at regular intervals of time (`dt=abs(time)/(trajsize-1)`, "
    "so that the number of points is `trajsize`; the last stored point is always at the end of "
    "integration period, and if trajsize>1, the first point is the initial conditions). "
    "If dtype=object and trajsize is not provided explicitly, this is equivalent to setting trajsize=0. "
    "Both time and trajsize may differ between orbits.\n"
    "  der (optional, default False):  whether to compute the evolution of deviation vectors "
    "(derivatives of the orbit w.r.t. the initial conditions).\n"
    "  lyapunov (optional, default False):  whether to estimate the Lyapunov exponent, which is "
    "a chaos indicator (positive value means that the orbit is chaotic, zero - regular).\n"
    "  accuracy (optional, default 1e-8):  relative accuracy of the ODE integrator.\n"
    "  maxNumSteps (optional, default 1e8):  upper limit on the number of steps in the ODE integrator.\n"
    "  dtype (optional, default 'float32'):  storage data type for trajectories (see below).\n"
    "  method (optional, string):  choice of the ODE integrator, available variants are "
    "'dop853' (default; 8th order Runge-Kutta), 'dprkn8' (8th order Runge-Kutta-Nystrom method, "
    "usually somewhat more efficient than dop853), or 'hermite' (4th order, might be more efficient "
    "in the regime of low accuracy, but only works in static potentials).\n"
    "  verbose (optional, default True):  whether to display progress when integrating multiple orbits.\n"
    "Returns:\n"
    "  depending on the arguments, one or a tuple of several data containers (one for each target, "
    "plus an extra one for trajectories if trajsize>0, plus another one for deviation vectors "
    "if der=True, plus another one for Lyapunov exponents if lyapunov=True). \n"
    "  Each target produces a 2d array of floats with shape NxC, where N is the number of orbits, "
    "and C is the number of constraints in the target (varies between targets); "
    "if there was a single orbit, then this would be a 1d array of length C. "
    "These data storage arrays should be provided to the `solveOpt()` routine. \n"
    "  Turning on the Lyapunov exponent estimation produces two numbers per orbit: "
    "the normalized Lyapunov exponent (lambda * Torb, where the orbital period is taken to be "
    "potential.Tcirc(ic) - values of order unity indicate strongly chaotic orbits, much smaller "
    "than unity - weakly chaotic ones), and the dimensionless timescale for the onset of chaos "
    "(expressed in units of orbital period). In case of N orbits, this array has shape Nx2.\n"
    "  Trajectory output and deviation vectors can be requested in two alternative formats: "
    "arrays or Orbit objects.\n"
    "In the first case, the output of the trajectory is a Nx2 array (or, in case of a single orbit, "
    "a 1d array of length 2), with elements being objects themselves: "
    "each row stands for one orbit, the first element in each row is a 1d array of length `trajsize` "
    "containing the timestamps, and the second is a 2d array of phase-space coordinates "
    "at the corresponding timestamps, in the format depending on dtype:\n"
    "'float' or 'double' means 6 64-bit floats (3 positions and 3 velocities) in each row;\n"
    "'float32' (default) means 6 32-bit floats;\n"
    "'complex' or 'complex128' or 'c16' means 3 128-bit complex values (pairs of 64-bit floats), "
    "with velocity in the imaginary part; and 'complex64' or 'c8' means 3 64-bit complex values.\n"
    "The time array is always 64-bit float. "
    "The choice of dtype only affects trajectories; arrays returned by each target always "
    "contain 32-bit floats.\n"
    "In the second case (dtype=object), the output is a 1d array of length N containing instances "
    "of a special class agama.Orbit, or just an Orbit object for a single orbit. "
    "The agama.Orbit class can only be returned by the orbit() routine and cannot be instantiated "
    "directly. It provides interpolated trajectory at any time within the range spanned by the orbit: "
    "its () operator takes one argument (a single value of time or an array of times), "
    "and returns one or more 6d phase-space coordinates at the requested times. "
    "It also exposes a sequence interface: len(orbit) returns the number of timestamps "
    "in the interpolator, and orbit[i] returns the i-th timestamp, so that the full 6d trajectory "
    "at these timestamps can be reconstructed by applying the () operator to the orbit object itself. "
    "Although such interpolator may be constructed from a regularly-spaced orbit, it makes more "
    "sense to leave trajsize=0 in this case, i.e., record the trajectory at every timestep "
    "of the orbit integrator. This is the default behaviour, and trajsize needs not be specified "
    "explicitly when setting dtype=object.\n"
    "The output for deviation vectors, if they are requested, follows the same format as for "
    "the trajectory, except that there are 6 such vectors for each orbit. "
    "Thus, if dtype=object, each orbit produces an array of 6 agama.Orbit objects, each one "
    "representing a single deviation vector; for other dtypes, the output for one orbit consists of "
    "6 arrays of shape `trajsize`*6 (for dtype='float' or 'double'), or `trajsize`*3 "
    "(for dtype='complex' or 'complex128'), each one representing a single deviation vector "
    "sampled at the same timestamps as the trajectory. "
    "in case of N>1 orbits, the output is an array of shape Nx6 filled with agama.Orbit objects "
    "or 2d arrays of deviation vectors.\n\n"
    "Examples:\n"
    "# compute a single orbit and output the trajectory in a 2d array of size 1001x6:\n"
    ">>> times,traj = agama.orbit(potential=mypot, ic=[x,y,z,vx,vy,vz], time=100, trajsize=1001)\n"
    "# record the same orbit at its 'natural' timestep and represent it as an agama.Orbit object:\n"
    ">>> orbit = agama.orbit(potential=mypot, ic=[x,y,z,vx,vy,vz], time=100, dtype=object)\n"
    ">>> traj_recorded = orbit(orbit)       # produces a 2d array of size len(orbit) x 6\n"
    ">>> traj_interpolated = orbit(times)   # produces an array of size 1001x6 very close to traj\n"
    "# integrate a bunch of orbits with initial conditions taken from a Nx6 array `initcond`, "
    "for a time equivalent to 50 periods for each orbit, collecting the data for two targets "
    "`target1` and `target2` and also storing their trajectories in a Nx2 array of "
    "time and position/velocity arrays:\n"
    ">>> stor1, stor2, trajectories = agama.orbit(potential=mypot, ic=initcond, "
    "time=50*mypot.Tcirc(initcond), trajsize=500, targets=(target1, target2))\n"
    "# compute a single orbit and its deviation vectors v0..v5, storing only the final values "
    "at t=tend, and estimate the Lyapunov exponent (if it is positive, the magnitude of deviation "
    "vectors grows exponentially with time, otherwise grows linearly):\n"
    ">>> (time,endpoint), (v0,v1,v2,v3,v4,v5), lyap = agama.orbit(potential=mypot, "
    "ic=[x,y,z,vx,vy,vz], time=100, trajsize=1, der=True, lyapunov=True)";

/// run a single orbit or the entire orbit library for a Schwarzschild model
PyObject* orbit(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    if(!onlyNamedArgs(args, namedArgs))
        return NULL;

    // parse input arguments
    orbit::OrbitIntParams params;
    double Omega = 0.;
    int haveDer  = 0;
    int haveLyap = 0;
    int verbose  = 1;
    int dtype = NPY_FLOAT;
    PyObject *ic_obj = NULL, *time_obj = NULL, *timestart_obj = NULL, *pot_obj = NULL,
        *targets_obj = NULL, *trajsize_obj = NULL, *dtype_obj = NULL, *method_obj = NULL;
    static const char* keywords[] =
        {"ic", "time", "timestart", "potential", "targets", "trajsize", "der",
        "lyapunov", "Omega", "accuracy", "maxNumSteps", "dtype", "method", "verbose", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "|OOOOOOiiddiOOi", const_cast<char**>(keywords),
        &ic_obj, &time_obj, &timestart_obj, &pot_obj, &targets_obj, &trajsize_obj, &haveDer, &haveLyap,
        &Omega, &params.accuracy, &params.maxNumSteps, &dtype_obj, &method_obj, &verbose))
        return NULL;

    // check if deviation vectors are needed (if yes, the output contains yet another array)
    if(haveDer != 0 && haveDer != 1) {
        PyErr_SetString(PyExc_TypeError, "Argument 'der' must be a boolean or an int 0/1");
        return NULL;
    }

    // check if Lyapunov exponent is needed (if yes, the output contains yet another extra item)
    if(haveLyap != 0 && haveLyap != 1) {
        PyErr_SetString(PyExc_TypeError, "Argument 'lyapunov' must be a boolean");
        return NULL;
    }

    // choice of orbit integrator
    if(method_obj != NULL) {
        std::string method_str = toString(method_obj);
        if(utils::stringsEqual(method_str, "hermite"))
            params.method = orbit::OrbitIntParams::HERMITE;
        else if(utils::stringsEqual(method_str, "dprkn8"))
            params.method = orbit::OrbitIntParams::DPRKN8;
        else if(utils::stringsEqual(method_str, "dop853"))
            params.method = orbit::OrbitIntParams::DOP853;
        else {
            PyErr_SetString(PyExc_ValueError,
                "Unknown ODE integation method (valid values are 'dop853', 'dprkn8' or 'hermite')");
            return NULL;
        }
    }

    // whether to show the progress indicator
    if(verbose != 0 && verbose != 1) {
        PyErr_SetString(PyExc_TypeError, "Argument 'verbose' must be a boolean or an int 0/1");
        return NULL;
    }

    // unit-convert the pattern speed
    Omega /= conv->timeUnit;

    // ensure that a potential object was provided
    potential::PtrPotential pot = getPotential(pot_obj);
    if(!pot) {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'potential' must be a valid Potential object");
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
        PyErr_SetString(PyExc_TypeError, "Argument 'ic' does not contain a valid array of length Nx6");
        return NULL;
    }
    bool singleOrbit = PyArray_NDIM(ic_arr) == 1; // in this case all output arrays have one dimension less

    // unit-convert initial conditions
    npy_intp numOrbits = singleOrbit ? 1 : PyArray_DIM(ic_arr, 0);
    std::vector<coord::PosVelCar> initCond(numOrbits);
    for(npy_intp i=0; i<numOrbits; i++)
        initCond[i] = convertPosVel(PyArray_NDIM(ic_arr) == 1 ?
            &pyArrayElem<double>(ic_arr, 0) : &pyArrayElem<double>(ic_arr, i, 0));
    Py_DECREF(ic_arr);

    // check that integration time(s) were provided
    PyArrayObject *time_arr = time_obj==NULL ? NULL :
        (PyArrayObject*) PyArray_FROM_OTF(time_obj, NPY_DOUBLE, 0);
    if(time_arr == NULL || !( PyArray_NDIM(time_arr) == 0 ||
        (PyArray_NDIM(time_arr) == 1 && (int)PyArray_DIM(time_arr, 0) == numOrbits) ) )
    {
        Py_XDECREF(time_arr);
        PyErr_SetString(PyExc_ValueError,
            "Argument 'time' must either be a scalar or have the same length "
            "as the number of points in the initial conditions");
        return NULL;
    }
    PyArrayObject *timestart_arr = timestart_obj==NULL ? NULL :
        (PyArrayObject*) PyArray_FROM_OTF(timestart_obj, NPY_DOUBLE, 0);
    if(timestart_arr != NULL && !( PyArray_NDIM(timestart_arr) == 0 ||
         (PyArray_NDIM(timestart_arr) == 1 && (int)PyArray_DIM(timestart_arr, 0) == numOrbits) ) )
    {
        Py_DECREF(timestart_arr);
        PyErr_SetString(PyExc_ValueError,
            "Argument 'timestart' must either be a scalar or have the same length "
            "as the number of points in the initial conditions");
        return NULL;
    }

    // unit-convert integration times
    std::vector<double> timestart(numOrbits), timetotal(numOrbits);
    if(PyArray_NDIM(time_arr) == 0)
        timetotal.assign(numOrbits, PyFloat_AsDouble(time_obj) * conv->timeUnit);
    else
        for(npy_intp i=0; i<numOrbits; i++)
            timetotal[i] = pyArrayElem<double>(time_arr, i) * conv->timeUnit;
    if(timestart_arr == NULL)
        timestart.assign(numOrbits, 0);
    else if(PyArray_NDIM(timestart_arr) == 0)
        timestart.assign(numOrbits, PyFloat_AsDouble(timestart_obj) * conv->timeUnit);
    else
        for(npy_intp i=0; i<numOrbits; i++)
            timestart[i] = pyArrayElem<double>(timestart_arr, i) * conv->timeUnit;
    Py_DECREF(time_arr);
    Py_XDECREF(timestart_arr);

    // check that valid targets were provided
    std::vector<PyObject*> targets_vec = toPyObjectArray(targets_obj);
    std::vector<galaxymodel::PtrTarget> targets;
    std::vector<double> unitConversionFactors;
    size_t numTargets = targets_vec.size();
    for(size_t t=0; t<numTargets; t++) {
        if(!PyObject_TypeCheck(targets_vec[t], &TargetType)) {
            PyErr_SetString(PyExc_TypeError, "Argument 'targets' must contain "
                "an instance of Target class or a tuple/list of such instances");
            return NULL;
        }
        targets.push_back(((TargetObject*)targets_vec[t])->target);
        unitConversionFactors.push_back(((TargetObject*)targets_vec[t])->unitDFProjection);
    }

    // determine the output datatype for trajectory
    if(dtype_obj != NULL && dtype_obj != Py_None) {
        PyArray_Descr* dtype_descr = NULL;
        dtype = PyArray_DescrConverter2(dtype_obj, &dtype_descr) ? dtype_descr->type_num : NPY_NOTYPE;
        Py_XDECREF(dtype_descr);
        if( dtype !=  NPY_FLOAT && dtype !=  NPY_DOUBLE &&
            dtype != NPY_CFLOAT && dtype != NPY_CDOUBLE && dtype != NPY_OBJECT )
        {
            PyErr_SetString(PyExc_TypeError,
                "Argument 'dtype' should correspond to 32/64-bit float/complex or object");
            return NULL;
        }
    }

    // check if trajectory needs to be recorded: in this case the output tuple contains one extra item
    std::vector<int> trajSizes;
    bool haveTraj = trajsize_obj != NULL;
    if(haveTraj) {
        PyArrayObject *trajsize_arr =
            (PyArrayObject*) PyArray_FROM_OTF(trajsize_obj, NPY_INT, NPY_ARRAY_FORCECAST);
        if(!trajsize_arr)
            return NULL;
        if(PyArray_NDIM(trajsize_arr) == 0) {
            trajSizes.assign(numOrbits, PyInt_AsLong(trajsize_obj));
        } else if(PyArray_NDIM(trajsize_arr) == 1 && (int)PyArray_DIM(trajsize_arr, 0) == numOrbits) {
            trajSizes.resize(numOrbits);
            for(npy_intp i=0; i<numOrbits; i++)
                trajSizes[i] = pyArrayElem<int>(trajsize_arr, i);
        }
        Py_DECREF(trajsize_arr);
        bool nonneg = true;
        for(npy_intp i=0; i<numOrbits; i++)
            nonneg &= trajSizes[i] >= 0;
        if((npy_intp)trajSizes.size() != numOrbits || !nonneg) {
            PyErr_SetString(PyExc_ValueError,
                "Argument 'trajsize', if provided, must either be a nonnegative integer or an array "
                "of integers with the same length as the number of points in the initial conditions");
            return NULL;
        }
    }
    // for convenience, if dtype is set to 'object' and trajsize is not provided explicitly,
    // this is equivalent to setting trajsize=0, i.e. recording orbits at their 'natural' timestep
    if(dtype == NPY_OBJECT && !haveTraj) {
        trajSizes.assign(numOrbits, 0);
        haveTraj = true;
    }

    // the output is one or more Numpy arrays:
    // each target corresponds to a NumPy array where the collected information for all orbits is stored,
    // plus optionally an array containing the trajectories of all orbits if they are requested,
    // plus optionally an array containing the deviation vectors of all orbits if requested,
    // plus optionally an array of Lyapunov exponents if requested.
    // We first allocate and keep these arrays in a std::vector, then convert it into a tuple
    // at the end, once the data is ready: at this stage, zero-dimension arrays are converted to scalars.
    // The vector result_numCols contains the number of entries per orbit for each output array.
    size_t numOutputs = numTargets + haveTraj + haveLyap + haveDer;
    if(numOutputs == 0) {
        PyErr_SetString(PyExc_RuntimeError, "No output is requested: "
            "if you just need the trajectory, provide trajsize=... for array output "
            "or dtype=object for Orbit object output");
        return NULL;
    }
    if(haveDer && !haveTraj) {
        PyErr_SetString(PyExc_RuntimeError, "Derivatives are requested, "
            "but the trajectory itself is not: provide trajsize=... for array output "
            "or dtype=object for Orbit object output");
        return NULL;
    }
    std::vector<PyArrayObject*> result_arrays(numOutputs);
    std::vector<npy_intp> result_numCols(result_arrays.size());

    // allocate the arrays for storing the information for each target,
    // and optionally for the output trajectory(ies) and/or Lyapunov exponents.
    // the array of trajectories is an array of shape (N,) or (N,2) of Python objects
    volatile bool fail = false;  // error flag (e.g., insufficient memory)
    for(size_t t=0; !fail && t < numOutputs; t++) {
        npy_intp numCols;
        int datatype;
        if(t < numTargets) {                      // ordinary target objects
            numCols  = targets[t]->numCoefs();
            datatype = STORAGE_NUM_T;
        } else if(haveTraj && t == numTargets) {  // trajectory storage
            // one element per orbit when the output dtype is Orbit objects,
            // otherwise two elements per orbit (time and trajectory arrays)
            numCols  = dtype == NPY_OBJECT ? 1 : 2;
            datatype = NPY_OBJECT;
        } else if(haveDer && t == numTargets + haveTraj) {  // deviation vectors storage
            // 6 dev vectors per orbit, stored as arrays or Orbit objects
            numCols  = 6;
            datatype = NPY_OBJECT;
        } else if(haveLyap && t == numTargets + haveTraj + haveDer) {  // Lyapunov exponent and Tchaos
            numCols  = 2;
            datatype = NPY_DOUBLE;
        } else
            assert(!"Counting error in allocating output arrays");  // shouldn't happen
        result_numCols[t] = numCols;
        // if there is only a single orbit, the output array is 1-dimensional,
        // otherwise 2-dimensional (numOrbits rows, numCols columns);
        // but if numCols==1, the dimension of the output array is further reduced by one
        npy_intp size[2] = {numOrbits, numCols};
        result_arrays[t] = (PyArrayObject*)( singleOrbit ?
            PyArray_SimpleNew(1 - (numCols==1), &size[1], datatype) :
            PyArray_SimpleNew(2 - (numCols==1), size, datatype) );
        if(!result_arrays[t])
            fail = true;
    }

    // optionally show a progress bar
    ProgressBar progressBar(numOrbits, "orbit", /*minTotalTime*/ 1.0);

    // set up signal handler to stop the integration on a keyboard interrupt
    utils::CtrlBreakHandler cbrk;

    // finally, run the orbit integration
    volatile npy_intp numCompleted = 0;
    std::string errorMessage;
    if(!fail) {
        // the GIL must be released when running an OpenMP-parallelized loop
        // in which we may call Python C API functions from different threads;
        // these functions will temporarily re-acquire GIL in their respective threads
        PyReleaseGIL unlock;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for(npy_intp orb = 0; orb < numOrbits; orb++) {
            if(fail || cbrk.triggered()) continue;
            // thread-local temporary place for storing the trajectory and deviation vectors
            // before subsequent unit-conversion
            orbit::Trajectory traj, deviationVectors[6];
            try{
                // instance of the orbit integrator for the current orbit
                orbit::OrbitIntegrator<coord::Car> orbint(*pot, Omega, params);

                // construct runtime functions for each target that store the collected data
                // in the respective row of each target's matrix,
                // plus optionally the trajectory and Lyapunov exponent recording functions
                for(size_t t=0; t<numTargets; t++) {
                    galaxymodel::StorageNumT* output = static_cast<galaxymodel::StorageNumT*>(
                        PyArray_DATA(result_arrays[t])) + orb * result_numCols[t];
                    orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(
                        new galaxymodel::RuntimeFncTarget(orbint, *targets[t], output)));
                }
                if(haveTraj) {
                    double trajStep = trajSizes[orb]>1 ?
                        fabs(timetotal[orb]) / (trajSizes[orb]-1) :  // output at regular time intervals
                        trajSizes[orb]==1 ? INFINITY :  // store only the last point
                        /*trajSizes[orb]==0*/ 0;  // store trajectory at every integration timestep
                    orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(
                        new orbit::RuntimeTrajectory(orbint, trajStep, traj)));
                }
                if(haveLyap || haveDer) {
                    double trajStep = 0;
                    if(haveDer) {  // implies trajSizes is not empty
                        trajStep = trajSizes[orb]>1 ?
                        fabs(timetotal[orb]) / (trajSizes[orb]-1) :  // same timestep as for the orbit
                        trajSizes[orb]==1 ? INFINITY :  // store only the last point
                        0;  // store at every integration timestep
                    }
                    double* outputLyap = haveLyap ?
                        static_cast<double*>(
                        PyArray_DATA(result_arrays[numTargets + haveTraj + haveDer])) +
                        orb * result_numCols[numTargets + haveTraj + haveDer] :
                        NULL;  // Lyapunov exponent will not be computed if not requested
                    orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(
                        new orbit::RuntimeVariational(orbint, trajStep,
                            haveDer ? deviationVectors : NULL, outputLyap)));
                }

                // integrate the orbit
                orbint.init(initCond[orb], timestart[orb]);
                orbint.run(timetotal[orb]);
                // finalize the output of runtime functions - they are destroyed along with orbint,
                // performing any necessary procedures before going out of scope
            }
            catch(std::exception& ex) {
                errorMessage = ex.what();
                fail = true;
            }
            // remaining procedures are trivial and should not raise exceptions

            // convert the units for matrices produced by targets
            for(size_t t=0; t<numTargets; t++) {
                galaxymodel::StorageNumT mult = conv->massUnit / unitConversionFactors[t];
                galaxymodel::StorageNumT* output = static_cast<galaxymodel::StorageNumT*>(
                    PyArray_DATA(result_arrays[t])) + orb * result_numCols[t];
                for(npy_intp index=0; index<result_numCols[t]; index++)
                    output[index] *= mult;
            }

            // if the trajectory was recorded, store it in the corresponding item of the output tuple
            if(haveTraj) {
                // pointer to the beginning of storage for the given orbit (one or two PyObject* items)
                PyObject** output = static_cast<PyObject**>(
                    PyArray_DATA(result_arrays[numTargets])) +
                    orb * result_numCols[numTargets];
                // store the orbit in the output array in the form depending on dtype
                if(!createOrbit(dtype, traj, Omega, /*outputTime*/true, /*normFactor*/1, output))
                    fail = true;
            }

            // if deviation vectors were recorded, store them in the corresponding item of the output
            if(haveDer) {
                // pointer to the beginning of storage for the given orbit (six PyObject* items)
                PyObject** output = static_cast<PyObject**>(
                    PyArray_DATA(result_arrays[numTargets + haveTraj])) +
                    orb * result_numCols[numTargets + haveTraj];
                // store the deviation vectors in the output array in the form depending on dtype
                for(int vec=0; vec<6; vec++, output++) {
                    // since the deviation vectors are components of the Jacobian of mapping
                    // between the initial conditions and the current point on the orbit,
                    // their dimensions should be scaled by position (first 3 vectors) or velocity
                    double normFactor = vec<3 ? conv->lengthUnit : conv->velocityUnit;
                    if(!createOrbit(dtype, deviationVectors[vec], Omega,
                        /*outputTime*/false, normFactor, output))
                        fail = true;
                }
            }

            // status update
#ifdef _OPENMP
#pragma omp atomic
#endif
            ++numCompleted;
            if(numOrbits > 1 && verbose)
                progressBar.update(numCompleted);
        }
    }
    if(numOrbits > 1 && verbose) {
        progressBar.clear();
        double orbitsPerSec = numCompleted / progressBar.timer.deltaSeconds();
        PySys_WriteStderr(orbitsPerSec >= 1000 ?
            "%li orbits complete (%.0f orbits/s)\n" :
            "%li orbits complete (%.4g orbits/s)\n",
            (long int)numCompleted, orbitsPerSec);
    }
    if(cbrk.triggered()) {
        fail = true;
        errorMessage = cbrk.message();
    }
    if(fail) {
        for(size_t t=0; t<result_arrays.size(); t++)
            Py_XDECREF(result_arrays[t]);
        if(!PyErr_Occurred())   // set an error message if it hadn't been set previously
            raisePythonException(errorMessage, "Error in orbit(): ");
        return NULL;
    }

    // return a tuple of storage matrices (numpy-compatible) and/or a list of orbit trajectories,
    // but if this tuple only contains one element, return simply this element.
    // Moreover, in case of a single orbit, some of the output arrays may be 0-dimensional,
    // and should be converted to corresponding scalar types
    if(result_arrays.size() == 1) {
        return PyArray_Return(result_arrays[0]);
    } else {
        PyObject* result_tuple = PyTuple_New(result_arrays.size());
        if(!result_tuple)
            return NULL;
        for(size_t t=0; t<result_arrays.size(); t++)
            PyTuple_SET_ITEM(result_tuple, t, PyArray_Return(result_arrays[t]));
        return result_tuple;
    }
}


///@}
//  --------------------------------------------
/// \name  Computation of Gauss-Hermite moments
//  --------------------------------------------
///@{

/// description of ghMoments function
static const char* docstringGhMoments =
    "Compute the coefficients of Gauss-Hermite expansion for line-of-sight velocity "
    "distribution functions represented by a B-spline, as used in the LOSVD Target model.\n"
    "Named arguments:\n"
    "  degree - degree of B-spline expansion (int, 0 to 3).\n"
    "  gridv  - array of grid nodes in velocity that determine the B-spline; "
    "should be the same as used in constructing the Target object.\n"
    "  matrix - a 1d or 2d array with the amplitudes of B-spline expansion of LOSVD. "
    "The number of columns in the matrix is numBasisFnc * numApertures: "
    "the former is the number of amplitudes of B-spline representation of a single LOSVD, "
    "equal to len(gridv)+degree-1; the latter is the number of separate regions in "
    "the image plane, each with its own LOSVD. Note that numApertures is inferred as the number "
    "of columns divided by the number of basis functions (itself known from gridv and degree). "
    "If the matrix is two-dimensional, each row corresponds to a single component "
    "of the model (e.g., an orbit) which has its LOSVD recorded in each aperture. "
    "In the opposite case (one-dimensional array) these could be LOSVDs for the entire model "
    "(e.g., constructed from an N-body snapshot or from observations) in each aperture. "
    "Amplitudes of LOSVD representation for a single aperture are grouped together "
    "(in other words, each component may be viewed as a 2d matrix with numApertures rows "
    "and numBasisFnc columns, reshaped into a 1d array).\n"
    "  ghorder - the order of Gauss-Hermite expansion, should be >=2.\n"
    "  ghbasis (optional) - if provided, should be a 2d array with numApertures rows and 3 columns, "
    "each row containing the parameters of the Gaussian that serves as the basis for expansion: "
    "amplitude, center and width. \n"
    "There are two different scenarios for using this routine. \n"
    "The first is to construct both the expansion parameters (amplitude, center and width) "
    "by finding a best-fit Gaussian for each of the input LOSVDs, and then use these parameters "
    "to compute higher-order GH moments; in this case the input matrix is supposed to represent "
    "the LOSVDs in each aperture for the entire model (i.e., has only one component), "
    "and the argument 'ghbasis' is not provided.\n"
    "The second scenario is to convert the LOSVDs for a multi-component model (e.g., produced by "
    "the Target LOSVD object during orbit integration) into GH moments, reducing the number of "
    "parameters needed to represent each component's LOSVD. In this case all components "
    "naturally should use the same base parameters of the Gaussian (separate for each aperture, "
    "but identical between components), so that a linear superposition of input LOSVDs "
    "corresponds to the same linear superposition of GH moments. Hence the argument 'ghbasis' "
    "should be provided.\n"
    "  Returns: a 1d or 2d array (depending on the number of dimensions of the input matrix), "
    "where each row contains the GH moments for each aperture, and the number of rows is equal "
    "to the number of components (rows of the input matrix).\n"
    "If 'ghbasis' argument was not provided, the output will contain also the parameters of "
    "the best-fit Gaussian serving as the basis for the expansion, i.e. three numbers "
    "(amplitude, center and width), followed by GH moments h_0..h_M, where M is the order "
    "of expansion - in total M+4 numbers for each aperture (grouped together), "
    "of which the first three can be later used as the 'ghbasis' argument for computing the moments "
    "in a multi-component model. In this case, h_0=1, h_1=h_2=0 with high accuracy.\n"
    "In the opposite case when 'ghbasis' is provided, the output for each aperture contains M+1 "
    "moments h_0..h_M.\n";

PyObject* ghMoments(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    if(!onlyNamedArgs(args, namedArgs))
       return NULL;

    int degree = -1, ghorder = -1;
    PyObject *gridv_obj = NULL, *mat_obj = NULL, *gh_obj = NULL;
    static const char* keywords[] = {"degree", "gridv", "matrix", "ghorder", "ghbasis", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "iOOi|O", const_cast<char**>(keywords),
        &degree, &gridv_obj, &mat_obj, &ghorder, &gh_obj))
        return NULL;

    // order of Gauss-Hermite expansion
    if(ghorder<0) {
        PyErr_SetString(PyExc_ValueError, "ghMoments: order of Gauss-Hermite expansion should be >=0");
        return NULL;
    }

    // degree of B-splines
    if(degree<0 || degree>3) {
        PyErr_SetString(PyExc_ValueError, "ghMoments: degree of interpolation may not exceed 3");
        return NULL;
    }

    // grid in velocity space
    std::vector<double> gridv = toDoubleArray(gridv_obj);
    if(gridv.size() < 2) {
        PyErr_SetString(PyExc_ValueError, "ghMoments: gridv must be an array with >= 2 nodes");
        return NULL;
    }
    int numBasisFnc = (npy_intp)gridv.size() + degree - 1;  // number of B-spline basis functions

    // matrix of B-spline amplitudes of LOSVD in each aperture (columns)
    // for each element of the model (e.g. an orbit) (rows)
    PyArrayObject *mat_arr = mat_obj?
        (PyArrayObject*) PyArray_FROM_OTF(mat_obj, STORAGE_NUM_T, NPY_ARRAY_FORCECAST) : NULL;
    npy_intp numApertures = -1;
    if(mat_arr && (PyArray_NDIM(mat_arr) == 1 || PyArray_NDIM(mat_arr) == 2))
        numApertures = PyArray_DIM(mat_arr, PyArray_NDIM(mat_arr)-1) / numBasisFnc;
    if(!mat_arr || numApertures * numBasisFnc != PyArray_DIM(mat_arr, PyArray_NDIM(mat_arr)-1)) {
        Py_XDECREF(mat_arr);
        PyErr_SetString(PyExc_ValueError, ("Argument 'matrix' should be a 1d array "
            "of length numApertures * numBasisFnc (the latter is " + utils::toString(numBasisFnc) +
            " for the provided gridv and degree), or a 2d array with this number of columns").c_str());
        return NULL;
    }
    int ndim = PyArray_NDIM(mat_arr);
    npy_intp numComponents = ndim==1 ? 1 : PyArray_DIM(mat_arr, 0);

    // parameters of Gauss-Hermite expansion(s), if provided
    PyArrayObject *gh_arr = gh_obj? (PyArrayObject*) PyArray_FROM_OTF(gh_obj, NPY_DOUBLE, 0) : NULL;
    if( gh_obj != NULL && (gh_arr == NULL || PyArray_NDIM(gh_arr) != 2 ||
        PyArray_DIM(gh_arr, 0) != numApertures || PyArray_DIM(gh_arr, 1) != 3))
    {
        Py_XDECREF(gh_arr);
        Py_DECREF(mat_arr);
        PyErr_SetString(PyExc_ValueError,
            "Argument 'ghbasis', if provided, should be a 2d array with 3 columns: "
            "amplitude,center,width, and the number of rows equal to the number of apertures");
        return NULL;
    }

    // prepare the output array of Gauss-Hermite moments (and possibly the parameters of GH expansion)
    npy_intp size[2] = {numComponents, numApertures * (gh_arr ? ghorder+1 : ghorder+4)};
    PyObject* output_arr = PyArray_SimpleNew(ndim, &size[2-ndim], STORAGE_NUM_T);
    if(!output_arr) {
        Py_XDECREF(gh_arr);
        Py_DECREF(mat_arr);
        return NULL;
    }

    volatile bool fail = false;
    std::string errorMessage;
    utils::CtrlBreakHandler cbrk;
    // the procedure is different depending on whether the parameters of GH expansion are provided or not
    if(gh_arr) {
        // compute the GH moments for known (provided) parameters of expansion (amplitude,center,width).
        // Note that although this loop is OpenMP-parallelized, it does not involve any calls
        // to Python C API, nor it involves any operations that may invoke Python callback functions,
        // so we do not need to release GIL (same applies to the other loop further down).
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(npy_intp a=0; a<numApertures; a++) {
            if(fail) continue;
            try{
                // obtain the matrix that converts the B-spline amplitudes into Gauss-Hermite moments
                math::Matrix<double> ghmat(math::computeGaussHermiteMatrix(degree, gridv, ghorder,
                    /*ampl  */ pyArrayElem<double>(gh_arr, a, 0),
                    /*center*/ pyArrayElem<double>(gh_arr, a, 1),
                    /*width */ pyArrayElem<double>(gh_arr, a, 2)));
                std::vector<double> srcrow(numBasisFnc), dstrow(ghorder+1);  // temp storage
                // loop over all rows of the input matrix (e.g. orbits)
                for(npy_intp r=0; r<numComponents; r++) {
                    // convert the section of one row of the input array, corresponding to
                    // one aperture and one orbit, from StorageNumT to double
                    for(int b=0; b<numBasisFnc; b++)
                        srcrow[b] = ndim==1 ?
                            pyArrayElem<galaxymodel::StorageNumT>(mat_arr,    a * numBasisFnc + b) :
                            pyArrayElem<galaxymodel::StorageNumT>(mat_arr, r, a * numBasisFnc + b);
                    // multiply the array of amplitudes by the conversion matrix
                    math::blas_dgemv(math::CblasNoTrans, 1., ghmat, srcrow, 0., dstrow);
                    // convert back to StorageNumT and write
                    // to a section of one row of the result array
                    for(int m=0; m<=ghorder; m++)
                        (ndim==1 ?
                        pyArrayElem<galaxymodel::StorageNumT>(output_arr,    a * (ghorder+1) + m) :
                        pyArrayElem<galaxymodel::StorageNumT>(output_arr, r, a * (ghorder+1) + m) ) =
                            static_cast<galaxymodel::StorageNumT>(dstrow[m]);
                }
            }
            catch(std::exception& ex) {
                errorMessage = ex.what();
                fail = true;
            }
            if(cbrk.triggered()) {
                errorMessage = cbrk.message();
                fail = true;
            }
        }
    } else {
        // construct best-fit parameters of GH expansion (find amplitude,center,width)
        // for each aperture and component, and then compute GH moments using these parameters
        const npy_intp count = numApertures * numComponents;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
        for(npy_intp ar=0; ar < count; ar++) {
            if(fail) continue;
            try{
                int r = ar / numApertures, a = ar % numApertures;  // row and aperture indices
                std::vector<double> srcrow(numBasisFnc);
                for(int b=0; b<numBasisFnc; b++)
                    srcrow[b] = ndim == 1 ?
                        pyArrayElem<galaxymodel::StorageNumT>(mat_arr,    a * numBasisFnc + b) :
                        pyArrayElem<galaxymodel::StorageNumT>(mat_arr, r, a * numBasisFnc + b);
                shared_ptr<math::GaussHermiteExpansion> ghexp;
                switch(degree) {
                    case 0: ghexp.reset(new math::GaussHermiteExpansion(
                        math::BsplineWrapper<0>(gridv, srcrow), ghorder));
                        break;
                    case 1: ghexp.reset(new math::GaussHermiteExpansion(
                        math::BsplineWrapper<1>(gridv, srcrow), ghorder));
                        break;
                    case 2: ghexp.reset(new math::GaussHermiteExpansion(
                        math::BsplineWrapper<2>(gridv, srcrow), ghorder));
                        break;
                    case 3: ghexp.reset(new math::GaussHermiteExpansion(
                        math::BsplineWrapper<3>(gridv, srcrow), ghorder));
                        break;
                    default:  // shouldn't occur, we've checked degree beforehand
                        assert(!"Invalid B-spline degree");
                }
                std::vector<double> dstrow(ghorder+4);
                dstrow[0] = ghexp->ampl();    // overall amplitude
                dstrow[1] = ghexp->center();  // center of the expansion
                dstrow[2] = ghexp->width();   // width of the base gaussian
                std::copy(ghexp->coefs().begin(), ghexp->coefs().end(), dstrow.begin()+3);
                for(int m=0; m<=ghorder+3; m++)
                    (ndim==1 ?
                    pyArrayElem<galaxymodel::StorageNumT>(output_arr,    a * (ghorder+4) + m) :
                    pyArrayElem<galaxymodel::StorageNumT>(output_arr, r, a * (ghorder+4) + m) ) =
                        static_cast<galaxymodel::StorageNumT>(dstrow[m]);
            }
            catch(std::exception& ex) {
                errorMessage = ex.what();
                fail = true;
            }
            if(cbrk.triggered()) {
                errorMessage = cbrk.message();
                fail = true;
            }
        }
    }
    Py_XDECREF(gh_arr);
    Py_DECREF(mat_arr);
    if(fail) {
        raisePythonException(errorMessage);
        Py_DECREF(output_arr);
        return NULL;
    }
    return output_arr;
}


///@}
//  ----------------------------------
/// \name  N-body snapshot read/write
//  ----------------------------------
///@{

static const char* docstringReadSnapshot =
    "Read an N-body snapshot from a file.\n"
    "Arguments: file name.\n"
    "File format is determined automatically among the supported ones: "
    "text file with 7 columns (x,y,z,vx,vy,vz,m) is always supported, and NEMO or GADGET formats "
    "can be read if Agama was compiled with UNSIO library."
    "Returns:\n"
    "  a tuple of two arrays:  a 2d Nx6 array of particle coordinates and velocities, "
    "and a 1d array of N masses.";

PyObject* readSnapshot(PyObject* /*self*/, PyObject* arg)
{
    if(!PyString_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Expected one string argument (file name)");
        return NULL;
    }
    try{
        std::string name(PyString_AsString(arg));
        // we do not perform any unit conversion on the particle coordinates/masses:
        // they are read 'as is' from the file, and any such conversion will take place when
        // feeding them to other routines, such as constructing the potential or integrating orbits
        particles::ParticleArrayAux snap;
        {
            PyReleaseGIL unlock;  // temporary release GIL during IO operations as a goodwill measure
            snap = particles::readSnapshot(name);
        }
        npy_intp size[] = { static_cast<npy_intp>(snap.size()), 6 };
        PyObject* posvel_arr = PyArray_SimpleNew(2, size, NPY_DOUBLE);
        PyObject* mass_arr   = PyArray_SimpleNew(1, size, NPY_DOUBLE);
        if(!posvel_arr || !mass_arr) {
            Py_XDECREF(posvel_arr);
            Py_XDECREF(mass_arr);
            return NULL;
        }
        for(size_t i=0; i<snap.size(); i++) {
            snap.point(i).unpack_to(&pyArrayElem<double>(posvel_arr, i, 0));
            pyArrayElem<double>(mass_arr, i) = snap.mass(i);
        }
        return Py_BuildValue("NN", posvel_arr, mass_arr);
    }
    catch(std::exception& ex) {
        raisePythonException(ex);
        return NULL;
    }
}

static const char* docstringWriteSnapshot =
    "Write an N-body snapshot to a file.\n"
    "Arguments: \n"
    "  filename  - a string with file name;\n"
    "  particles  - a tuple of two arrays: a 2d Nx3 or Nx6 array of positions and "
    "optionally velocities, and a 1d array of N masses; \n"
    "  format  - (optional) file format, only the first letter (case-insensitive) matters: "
    "'t' is text (default), 'n' is NEMO, 'g' is GADGET (available if Agama was compiled with "
    "UNSIO library).\n"
    "Returns: none.\n";

PyObject* writeSnapshot(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    PyObject *particles_obj = NULL;
    const char *filename = NULL, *format = NULL;
    static const char* keywords[] = {"filename","particles","format",NULL};
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "sO|s", const_cast<char **>(keywords),
        &filename, &particles_obj, &format))
        return NULL;
    try{
        // determine whether we have only 3 coordinates, or additionally 3 velocities
        PyArrayObject *coord_arr, *mass_arr;
        convertParticlesStep1(particles_obj, /*create arrays*/ coord_arr, mass_arr);
        if(PyArray_DIM(coord_arr, 1) == 6) {
            particles::writeSnapshot(filename,
                convertParticlesStep2<coord::PosVelCar>(coord_arr, mass_arr),  // pos+vel
                format ? format : "text", *conv);
        } else {
            particles::writeSnapshot(filename,
                convertParticlesStep2<coord::PosCar>(coord_arr, mass_arr),  // only pos
                format ? format : "text", *conv);
        }
        Py_INCREF(Py_None);
        return Py_None;
    }
    catch(std::exception& ex) {
        raisePythonException(ex);
        return NULL;
    }
}


///@}
//  ----------------------------
/// \name  Optimization routine
//  ----------------------------
///@{

static const char* docstringSolveOpt =
    "Solve a linear or quadratic optimization problem.\n"
    "Find a vector x that solves a system of linear equations  A x = rhs,  "
    "subject to elementwise inequalities  xmin <= x <= xmax, "
    "while minimizing the cost function  F(x) = XL^T x + (1/2) x^T XQ x + P(A x - rhs), where "
    "XL and XQ are penalties for the solution vector, and P(y) is the penalty for violating "
    "the RHS constraints, consisting of two parts: linear penalty RL^T |y| and quadratic penalty "
    "(1/2) |y|^T diag(RQ) |y|  (both RL and RQ are nonnegative vectors of the same length as rhs).\n"
    "Arguments:\n"
    "  matrix:  2d matrix A of size RxC, or a tuple of several matrices that would be vertically "
    "stacked (they all must have the same number of columns C, and number of rows R1,R2,...). "
    "Providing a list of matrices does not incur copying, unlike the numpy.vstack() function.\n"
    "  rhs:     1d vector of length R, or a tuple of the same number of vectors as the number of "
    "matrices, with sizes R1,R2,...\n"
    "  xpenl:   1d vector of length C - linear penalties XL for the solution x "
    "(optional - zero if not provided).\n"
    "  xpenq:   1d vector of length C - diagonal of the matrix XQ of quadratic "
    "penalties for the solution x (optional).\n"
    "  rpenl:   1d vector of length R, or a tuple of vectors R1,R2,... - "
    "linear penalties RL for violating the RHS constraints (optional).\n"
    "  rpenq:   same for the quadratic penalties RQ (optional - if neither linear nor quadratic "
    "penalties for RHS violation were provided, it means that RHS must be satisfied exactly. "
    "If any of these penalties is set to infinity, it has the same effect, i.e. corresponding "
    "constraint must be satisfied exactly).\n"
    "  xmin:    1d vector of length C - minimum allowed values for the solution x (optional - "
    "if not provided, it implies a vector of zeros, i.e. the solution must be nonnegative).\n"
    "  xmax:    1d vector of length C - maximum allowed values for the solution x (optional - "
    "if not provided, it implies no upper limit).\n"
    "Returns:\n"
    "  the vector x solving the above system; if it cannot be solved exactly and no penalties "
    "for constraint violation were provided, then raise an exception.";

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

PyObject* solveOpt(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] =
        {"matrix", "rhs", "xpenl", "xpenq", "rpenl", "rpenq", "xmin", "xmax", NULL};
    PyObject *matrix_obj = NULL, *rhs_obj = NULL, *xpenl_obj = NULL, *xpenq_obj = NULL,
        *rpenl_obj = NULL, *rpenq_obj = NULL, *xmin_obj = NULL, *xmax_obj = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "OO|OOOOOO", const_cast<char**>(keywords),
        &matrix_obj, &rhs_obj, &xpenl_obj, &xpenq_obj, &rpenl_obj, &rpenq_obj, &xmin_obj, &xmax_obj))
        return NULL;

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
            PyErr_SetString(PyExc_TypeError, "Argument 'matrix' must be a 2d array "
                "or a tuple of such arrays with the same number of columns");
            return NULL;
        }
        nRow[s] = PyArray_DIM(mat, 0);
        nRowTotal += nRow[s];
    }

    // check and stack other input vectors
    std::vector<double> rhs, xpenl, xpenq, rpenl, rpenq, xmin, xmax, result;
    if(!stackVectors(rhs_obj, nRow, rhs) || rhs.empty()) {
        PyErr_SetString(PyExc_TypeError, "Argument 'rhs' must be a 1d array "
            "or a tuple of such arrays matching the number of rows in 'matrix'");
        return NULL;
    }
    if(!stackVectors(rpenl_obj, nRow, rpenl)) {
        PyErr_SetString(PyExc_TypeError, "Argument 'rpenl', if provided, must be a 1d array "
            "or a tuple of such arrays matching the number of rows in 'matrix'");
        return NULL;
    }
    if(!stackVectors(rpenq_obj, nRow, rpenq)) {
        PyErr_SetString(PyExc_TypeError, "Argument 'rpenq', if provided, must be a 1d array "
            "or a tuple of such arrays matching the number of rows in 'matrix'");
        return NULL;
    }
    xpenl = toDoubleArray(xpenl_obj);
    if(xpenl_obj && (int)xpenl.size() != nCol) {
        PyErr_SetString(PyExc_TypeError, "Argument 'xpenl', if provided, must be a 1d array "
            "with length matching the number of columns in 'matrix'");
        return NULL;
    }
    xpenq = toDoubleArray(xpenq_obj);
    if(xpenq_obj && (int)xpenq.size() != nCol) {
        PyErr_SetString(PyExc_TypeError, "Argument 'xpenq', if provided, must be a 1d array "
            "with length matching the number of columns in 'matrix'");
        return NULL;
    }
    xmin = toDoubleArray(xmin_obj);
    if(xmin_obj && (int)xmin.size() != nCol) {
        PyErr_SetString(PyExc_TypeError, "Argument 'xmin', if provided, must be a 1d array "
            "with length matching the number of columns in 'matrix'");
        return NULL;
    }
    xmax = toDoubleArray(xmax_obj);
    if(xmax_obj && (int)xmax.size() != nCol) {
        PyErr_SetString(PyExc_TypeError, "Argument 'xmax', if provided, must be a 1d array "
            "with length matching the number of columns in 'matrix'");
        return NULL;
    }

    // construct an interface layer for matrix stacking
    StackedMatrix matrix(matrixStack, nRowTotal, nCol, nRow);

    // call the appropriate solver
    try {
        if(rpenl.empty() && rpenq.empty()) {
            if(xpenq.empty())
                result = math::linearOptimizationSolve(
                    matrix, rhs, xpenl, xmin, xmax);
            else
                result = math::quadraticOptimizationSolve(
                    matrix, rhs, xpenl, math::BandMatrix<double>(xpenq), xmin, xmax);
        } else {
            if(rpenq.empty() && xpenq.empty())
                result = math::linearOptimizationSolveApprox(
                    matrix, rhs, xpenl, rpenl, xmin, xmax);
            else
                result = math::quadraticOptimizationSolveApprox(
                    matrix, rhs, xpenl, math::BandMatrix<double>(xpenq), rpenl, rpenq, xmin, xmax);
        }
    }
    catch(std::exception& ex) {
        raisePythonException(ex, "Error in solveOpt(): ");
        return NULL;
    }
    return toPyArray(result);
}


///@}
//  -----------------------------
/// \name  Various math routines
//  -----------------------------
///@{


/// wrapper for user-provided Python functions into the C++ compatible form
class FncWrapper: public math::IFunctionNdim {
    const unsigned int nvars;
    PyObject* fnc;
public:
    FncWrapper(unsigned int _nvars, PyObject* _fnc): nvars(_nvars), fnc(_fnc) {}

    /// vectorized evaluation of Python function for several points at once
    /// (making sure it invokes Python callback from a single thread at a time)
    virtual void evalmany(const size_t npoints, const double vars[], double values[]) const
    {
        PyAcquireGIL lock;
        bool typeerror   = false;
        npy_intp dims[]  = { (npy_intp)npoints, nvars};
        PyObject* args   = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, const_cast<double*>(vars));
        PyObject* result = PyObject_CallFunctionObjArgs(fnc, args, NULL);
        Py_DECREF(args);
        if(result == NULL) {
            if(PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_KeyboardInterrupt))
                throw std::runtime_error(utils::CtrlBreakHandler::message());
            else
                throw std::runtime_error("Exception occurred inside integrand");
        } else if(PyArray_Check(result) &&
            PyArray_NDIM((PyArrayObject*)result) == 1 &&
            PyArray_DIM ((PyArrayObject*)result, 0) == (npy_intp)npoints)
        {
            int type = PyArray_TYPE((PyArrayObject*) result);
            for(size_t p=0; p<npoints; p++) {
                switch(type) {
                    case NPY_DOUBLE: values[p] = pyArrayElem<double>(result, p); break;
                    case NPY_FLOAT:  values[p] = pyArrayElem<float >(result, p); break;
                    case NPY_BOOL:   values[p] = pyArrayElem<bool  >(result, p); break;
                    default: values[p] = NAN; typeerror = true;
                }
            }
        } else if(PyNumber_Check(result) && npoints==1)
            // in case of a single input point, may return a single number
            values[0] = PyFloat_AsDouble(result);
        else
            typeerror = true;
        Py_XDECREF(result);
        if(typeerror) {
            PyErr_SetString(PyExc_TypeError, "Invalid data type returned from user-defined function");
            throw std::runtime_error("Invalid data type returned from user-defined function");
        }
    }
    /// same for one point (not used by integration/sampling routines, but required by the interface)
    virtual void eval(const double vars[], double values[]) const {
        evalmany(1, vars, values);
    }
    virtual unsigned int numVars()   const { return nvars; }
    virtual unsigned int numValues() const { return 1; }
};

/// parse the arguments of integrateNdim and sampleNdim functions
bool parseLowerUpperBounds(PyObject* lower_obj, PyObject* upper_obj,
    std::vector<double> &xlow, std::vector<double> &xupp)
{
    if(!lower_obj) {   // this should always be provided - either # of dimensions, or lower boundary
        PyErr_SetString(PyExc_TypeError,
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
            PyErr_Format(PyExc_TypeError,
                "May not provide 'upper' argument if 'lower' specifies the number of dimensions (%i)", ndim);
            return false;
        }
        xlow.assign(ndim, 0.);  // default integration region
        xupp.assign(ndim, 1.);
        return true;
    }
    // if the first parameter is not the number of dimensions, then it must be the lower boundary,
    // and the second one must be the upper boundary
    PyArrayObject *lower_arr = (PyArrayObject*) PyArray_FROM_OTF(lower_obj, NPY_DOUBLE, 0);
    if(lower_arr == NULL || PyArray_NDIM(lower_arr) != 1) {
        Py_XDECREF(lower_arr);
        PyErr_SetString(PyExc_TypeError,
            "Argument 'lower' does not contain a valid array");
        return false;
    }
    ndim = PyArray_DIM(lower_arr, 0);
    if(!upper_obj) {
        PyErr_SetString(PyExc_TypeError, "Must provide both 'lower' and 'upper' arguments if both are arrays");
        return false;
    }
    PyArrayObject *upper_arr = (PyArrayObject*) PyArray_FROM_OTF(upper_obj, NPY_DOUBLE, 0);
    if(upper_arr == NULL || PyArray_NDIM(upper_arr) != 1 || PyArray_DIM(upper_arr, 0) != ndim) {
        Py_XDECREF(upper_arr);
        PyErr_Format(PyExc_TypeError,
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
    "this improves performance when using operations on numpy arrays), "
    "and return a 1d array of length M with function values;\n"
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
    ">>> integrateNdim(fnc, 4, toler=1e-3, maxeval=int(1e6))   "
    ">>> # non-default values for tolerance and number of evaluations must be passed as named arguments\n";

/// N-dimensional integration
PyObject* integrateNdim(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"fnc", "lower", "upper", "toler", "maxeval", NULL};
    double eps=1e-4;
    int maxNumEval=100000, numEval=-1;
    PyObject *callback=NULL, *lower_obj=NULL, *upper_obj=NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "O|OOdi", const_cast<char**>(keywords),
        &callback, &lower_obj, &upper_obj, &eps, &maxNumEval))
        return NULL;
    if(!PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "fnc must be callable");
        return NULL;
    }
    if(eps<=0 || maxNumEval<=0) {
        PyErr_SetString(PyExc_ValueError, "toler and maxeval must be positive");
        return NULL;
    }
    std::vector<double> xlow, xupp;
    if(!parseLowerUpperBounds(lower_obj, upper_obj, xlow, xupp))
        return NULL;
    double result, error;
    try{
        math::integrateNdim(FncWrapper(xlow.size(), callback),
            &xlow.front(), &xupp.front(), eps, maxNumEval, &result, &error, &numEval);
    }
    catch(std::exception& ex) {
        if(!PyErr_Occurred())    // set our own error string if it hadn't been set by Python
            raisePythonException(ex);
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
    "  fnc -- a callable object that must accept a single argument "
    "(a 2d array MxN array of coordinates, where N is the dimension of the hypercube, "
    "and M>=1 is the number of points where the function should be evaluated simultaneously -- "
    "this improves performance), and return a 1d array of M non-negative values "
    "(one for each point), interpreted as the probability density.\n"
    "  n -- the required number of samples drawn from this function.\n"
    "  lower, upper -- two arrays of the same length (equal to the number of dimensions) "
    "that specify the lower and upper boundaries of the region (hypercube) to be sampled; "
    "alternatively, a single value - the number of dimensions - may be passed instead of 'lower', "
    "in which case the default interval [0:1] is used for each dimension.\n"
    "  method (optional) -- choice of sampling method and output:\n"
    "    0 (default): use quasi-random number generator (QRNG, aka low-discrepancy sequences) "
    "for the coordinates of points, perform adaptive refinement of the hypercube, "
    "and return exactly n points (a subset of a larger number of internally collected samples). "
    "The weights of output points are all equal, and their density is proportional to "
    "the function value at each point.\n"
    "    1: use QRNG and refinement, but return all collected samples, rather than "
    "an equally-weighted subset. The sample weights are proportional to the function values "
    "at each point and inversely proportional to the number of samples put into each subregion "
    "during the adaptive refinement procedure. The weights do not exceed the integral of the "
    "function over the entire hypercube divided by n (the originally specified number of samples), "
    "but the length of the resulting arrays most likely will exceed n. "
    "This method may be useful if the samples will be used to estimate integrals of some other "
    "quantities (e.g. moments of the function) using the Monte Carlo approach; the use of QRNG "
    "improves the accuracy of these estimates.\n"
    "    3: use QRNG, but no refinement; in this case exactly n samples are placed uniformly "
    "inside the hypercube, and their weights are proportional to the values of the function; "
    "unless the function is constant, some weights will exceed the integral of the function "
    "over the entire hypercube divided by n. This method may be useful in the same context of "
    "Monte Carlo integration of some derived quantities, but the accuracy will be competitive "
    "with the previous method only if n is a power of two.\n"
    "    4: like 0, but replace QRNG by the more common pseudo-rangom number generator (PRNG). "
    "This method may be useful if one needs the sample points to be truly independent, "
    "but the accuracy of computing the integral of the function over the hypercube "
    "(and thus the error in sample weights) is worse than when using QRNG.\n"
    "Returns: a tuple with the following elements:\n"
    "  - a 2d array of samples with shape (nsamples,ndim), where nsamples=n except when method=1;\n"
    "  - a 1d array of weights of each sample; weights are all equal in methods 0 and 3, and "
    "the sum of all n weights equals the integral I of the function over the entire hypercube "
    "in all methods.\n"
    "  - the estimated error on the integral I;\n"
    "  - the actual number of function evaluations performed during sampling: "
    "it is typically a few times larger than n (except when method=3), and likewise larger than "
    "the length of the returned array of samples (except when method=1 or 3).\n\n"
    "Example:\n"
    ">>> samples,weights,error,_ = sampleNdim(fnc, 10000, [0,-1,0], [10,1,3.14])\n";

/// N-dimensional sampling
PyObject* sampleNdim(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"fnc", "n", "lower", "upper", "method", NULL};
    Py_ssize_t numSamples=-1;
    math::SampleMethod method = math::SM_DEFAULT;
    PyObject *callback=NULL, *lower_obj=NULL, *upper_obj=NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "On|OOb", const_cast<char**>(keywords),
        &callback, &numSamples, &lower_obj, &upper_obj, &method))
        return NULL;
    if(!PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "fnc must be callable");
        return NULL;
    }
    if(numSamples<=0) {
        PyErr_SetString(PyExc_ValueError, "n must be positive");
        return NULL;
    }
    std::vector<double> xlow, xupp;
    if(!parseLowerUpperBounds(lower_obj, upper_obj, xlow, xupp))
        return NULL;
    try{
        math::Matrix<double> samples;
        std::vector<double> weights;
        double error;
        size_t numEval=0;
        {   // run in a no-GIL block since sampleNdim is OpenMP-parallelized
            PyReleaseGIL unlock;
            math::sampleNdim(
                FncWrapper(xlow.size(), callback),
                &xlow[0], &xupp[0], numSamples, method,
                samples, weights, &error, &numEval);
        }
        return Py_BuildValue("NNdn", toPyArray(samples), toPyArray(weights), error, (Py_ssize_t)numEval);
    }
    catch(std::exception& ex) {
        if(!PyErr_Occurred())    // set our own error string if it hadn't been set by Python
            raisePythonException(ex);
        return NULL;
    }
}

/// description of random seed function
static const char* docstringSetRandomSeed =
    "Set the seed for the internal random number generator.\n"
    "This generator is used in various sampling methods, e.g., Density.sample(), GalaxyModel.sample(), "
    "and sampleNdim(). At the start of execution, the seed always has the same initial value; "
    "if one needs to produce different random realizations every time the script is run, "
    "setRandomSeed(0) takes the value from the current time.";
 
PyObject* setRandomSeed(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    int seed = -1;
    if(args != NULL && PyTuple_Check(args) && PyTuple_Size(args) == 1 &&
        ((seed = PyInt_AsLong(PyTuple_GET_ITEM(args, 0))) >= 0) )
    {
        math::randomize(seed);
        Py_INCREF(Py_None);
        return Py_None;
    } else {
        PyErr_SetString(PyExc_TypeError, "setRandomSeed() accepts only one integer argument >= 0");
        return NULL;
    }
}

///@}


static const char* docstringModule =
    "The Python interface for the AGAMA galaxy modelling library";

/// list of standalone functions exported by the module
static PyMethodDef module_methods[] = {
    { "setUnits",               (PyCFunction)setUnits,
      METH_VARARGS | METH_KEYWORDS, docstringSetUnits },
    { "getUnits",                            getUnits,
      METH_NOARGS,                  docstringGetUnits },
    { "splineApprox",           (PyCFunction)splineApprox,
      METH_VARARGS | METH_KEYWORDS, docstringSplineApprox },
    { "splineLogDensity",       (PyCFunction)splineLogDensity,
      METH_VARARGS | METH_KEYWORDS, docstringSplineLogDensity },
    { "orbit",                  (PyCFunction)orbit,
      METH_VARARGS | METH_KEYWORDS, docstringOrbit },
    { "readSnapshot",           (PyCFunction)readSnapshot,
      METH_O,                       docstringReadSnapshot },
    { "writeSnapshot",          (PyCFunction)writeSnapshot,
      METH_VARARGS | METH_KEYWORDS, docstringWriteSnapshot },
    { "ghMoments",              (PyCFunction)ghMoments,
      METH_VARARGS | METH_KEYWORDS, docstringGhMoments },
    { "solveOpt",               (PyCFunction)solveOpt,
      METH_VARARGS | METH_KEYWORDS, docstringSolveOpt },
    { "actions",                (PyCFunction)actions,
      METH_VARARGS | METH_KEYWORDS, docstringActions },
    { "integrateNdim",          (PyCFunction)integrateNdim,
      METH_VARARGS | METH_KEYWORDS, docstringIntegrateNdim },
    { "sampleNdim",             (PyCFunction)sampleNdim,
      METH_VARARGS | METH_KEYWORDS, docstringSampleNdim },
    { "setRandomSeed",          (PyCFunction)setRandomSeed,
      METH_VARARGS,                 docstringSetRandomSeed },
    { NULL }
};


} // end internal namespace
using namespace pygama;

// the module initialization function must be outside the internal namespace,
// and is slightly different in Python 2 and Python 3

#if PY_MAJOR_VERSION < 3
// Python 2.6-2.7
typedef struct PyModuleDef {
    int m_base;
    const char* m_name;
    const char* m_doc;
    Py_ssize_t m_size;
    PyMethodDef *m_methods;
} PyModuleDef;
#define PyModuleDef_HEAD_INIT 0
#define PyModule_Create(def) Py_InitModule3((def)->m_name, (def)->m_methods, (def)->m_doc)
static PyObject* PyInit_agama(void);
PyMODINIT_FUNC initagama(void) { PyInit_agama(); }
static PyObject*
#else
// Python 3
PyMODINIT_FUNC
#endif
PyInit_agama(void)
{
    static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, /* m_base */
        "agama",               /* m_name */
        docstringModule,       /* m_doc  */
        -1,                    /* m_size */
        module_methods,        /* m_methods */
    };
    import_array1(NULL);  // needed for NumPy to work properly

#if PY_VERSION_HEX < 0x03070000
    PyEval_InitThreads();  // this is not needed starting from Python 3.7
#endif
    thismodule = PyModule_Create(&moduledef);
    if(!thismodule)
        return NULL;

#ifdef Py_GIL_DISABLED
    // declare compatibility with free-threading Python >= 3.13
    PyUnstable_Module_SetGIL(thismodule, Py_MOD_GIL_NOT_USED);
#endif

    PyModule_AddStringConstant(thismodule, "__version__", AGAMA_VERSION);
    PyModule_AddObject(thismodule, "G", PyFloat_FromDouble(1.0));
    conv.reset(new units::ExternalUnits());

    setNumThreadsType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&setNumThreadsType) < 0) return NULL;
    Py_INCREFx(&setNumThreadsType);
    PyModule_AddObject(thismodule, "setNumThreads", (PyObject*)&setNumThreadsType);

    DensityType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&DensityType) < 0) return NULL;
    Py_INCREFx(&DensityType);
    PyModule_AddObject(thismodule, "Density", (PyObject*)&DensityType);
    DensityTypePtr = &DensityType;

    PotentialType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&PotentialType) < 0) return NULL;
    Py_INCREFx(&PotentialType);
    PyModule_AddObject(thismodule, "Potential", (PyObject*)&PotentialType);
    PotentialTypePtr = &PotentialType;

    ActionFinderType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&ActionFinderType) < 0) return NULL;
    Py_INCREFx(&ActionFinderType);
    PyModule_AddObject(thismodule, "ActionFinder", (PyObject*)&ActionFinderType);
    ActionFinderTypePtr = &ActionFinderType;

    ActionMapperType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&ActionMapperType) < 0) return NULL;
    Py_INCREFx(&ActionMapperType);
    PyModule_AddObject(thismodule, "ActionMapper", (PyObject*)&ActionMapperType);
    ActionMapperTypePtr = &ActionMapperType;

    DistributionFunctionType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&DistributionFunctionType) < 0) return NULL;
    Py_INCREFx(&DistributionFunctionType);
    PyModule_AddObject(thismodule, "DistributionFunction", (PyObject*)&DistributionFunctionType);
    DistributionFunctionTypePtr = &DistributionFunctionType;

    SelectionFunctionType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&SelectionFunctionType) < 0) return NULL;
    Py_INCREFx(&SelectionFunctionType);
    PyModule_AddObject(thismodule, "SelectionFunction", (PyObject*)&SelectionFunctionType);
    SelectionFunctionTypePtr = &SelectionFunctionType;

    GalaxyModelType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&GalaxyModelType) < 0) return NULL;
    Py_INCREFx(&GalaxyModelType);
    PyModule_AddObject(thismodule, "GalaxyModel", (PyObject*)&GalaxyModelType);

    ComponentType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&ComponentType) < 0) return NULL;
    Py_INCREFx(&ComponentType);
    PyModule_AddObject(thismodule, "Component", (PyObject*)&ComponentType);

    SelfConsistentModelType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&SelfConsistentModelType) < 0) return NULL;
    Py_INCREFx(&SelfConsistentModelType);
    PyModule_AddObject(thismodule, "SelfConsistentModel", (PyObject*)&SelfConsistentModelType);

    TargetType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&TargetType) < 0) return NULL;
    Py_INCREFx(&TargetType);
    PyModule_AddObject(thismodule, "Target", (PyObject*)&TargetType);
    TargetTypePtr = &TargetType;

    SplineType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&SplineType) < 0) return NULL;
    Py_INCREFx(&SplineType);
    PyModule_AddObject(thismodule, "Spline", (PyObject*)&SplineType);
    SplineTypePtr = &SplineType;

    OrbitType.tp_new = PyType_GenericNew;
    if(PyType_Ready(&OrbitType) < 0) return NULL;
    Py_INCREFx(&OrbitType);
    // deliberately not adding this class to the module - it cannot be created from a Python script
    OrbitTypePtr = &OrbitType;

    // if available, use a fancy progress bar for long operations
    PyObject *tqdmModule = PyImport_ImportModule("tqdm");
    tqdmClass = tqdmModule ? PyObject_GetAttrString(tqdmModule, "tqdm") : NULL; // global reference
    if(!tqdmClass)  // failure to import tqdm module is not critical, suppress the error
        PyErr_Clear();
    else
        Py_DECREF(tqdmModule);

    return thismodule;
}
// ifdef HAVE_PYTHON
#endif
