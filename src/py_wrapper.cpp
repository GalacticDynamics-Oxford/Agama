/** \file   py_wrapper.cpp
    \brief  Python wrapper for the library
    \author Eugene Vasiliev
    \date   2015
*/
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#define DEBUGPRINT
#include <numpy/arrayobject.h>
#include "utils_config.h"
#include "potential_factory.h"
#include "math_spline.h"

/// \name  Some general definitions and utility functions
///@{

/// arguments of functions
struct ArgDescription {
    const char* name;  ///< argument name
    char type;         ///< letter that determines argument type ('s','d','i','O')
    const char* descr; ///< textual description
};

/// argument type in human-readable words
static const char* argTypeDescr(char type)
{
    switch(type) {
        case 's': return "string";
        case 'i': return "int";
        case 'd': return "float";
        case 'O': return "object";
        default:  return "unknown type";
    }
}

/// max number of keywords
static const size_t MAX_NUM_KEYWORDS = 64;

/// max size of docstring
static const size_t MAX_LEN_DOCSTRING = 4096;

///@}
/// \name  ---------- Potential class and related data ------------
///@{

/// Python type corresponding to Potential class
typedef struct {
    PyObject_HEAD
    const potential::BasePotential* pot;
} PotentialObject;

/*static PyObject* PotentialObject_new(PyTypeObject *type, PyObject*, PyObject*)
{
    PotentialObject *self = (PotentialObject*)type->tp_alloc(type, 0);
    if(self) self->pot=NULL;
    printf("Allocated a new Potential object\n");
    return (PyObject*)self;
}*/

static void Potential_dealloc(PotentialObject* self)
{
    if(self->pot) {
#ifdef DEBUGPRINT
        printf("Deleted an instance of %s potential\n", self->pot->name());
#endif
        delete self->pot;
    }
#ifdef DEBUGPRINT
    else printf("Deleted empty Potential\n");
#endif
    self->ob_type->tp_free((PyObject*)self);
}

/// list of all possible arguments of Potential class constructor
static const ArgDescription potentialArgs[] = {
    {"file",              's', "the name of ini file, potential coefficients file, or N-body snapshot file"},
    {"type",              's', "potential type, such as 'Plummer', 'Ferrers', or potential expansion type, such as 'SplineExp'"},
    {"density",           's', "density model for potential expansion, like 'Dehnen', 'MGE'"},
    {"symmetry",          's', "assumed symmetry for potential expansion constructed from an N-body snapshot"},
    {"points",            'O', "array of point masses to be used in construction of a potential expansion: "
        "should be a tuple with two arrays - coordinates and mass, where the first one "
        "is a two-dimensional Nx3 array and the second one is a one-dimensional array of length N"},
    {"mass",              'd', "total mass of the model"},
    {"scaleradius",       'd', "scale radius of the model (if applicable)"},
    {"scaleradius2",      'd', "second scale radius of the model (if applicable)"},
    {"q",                 'd', "axis ratio y/x, i.e., intermediate to long axis (if applicable)"},
    {"p",                 'd', "axis ratio z/x, i.e., short to long axis (if applicable)"},
    {"gamma",             'd', "central cusp slope (applicable for Dehnen model)"},
    {"sersicindex",       'd', "Sersic index (applicable for Sersic density model)"},
    {"numcoefsradial",    'i', "number of radial terms in BasisSetExp or grid points in spline potentials"},
    {"numcoefsangular",   'i', "order of spherical-harmonic expansion (max.index of angular harmonic coefficient)"},
    {"numcoefsvertical",  'i', "number of coefficients in z-direction for CylSplineExp potential"},
    {"alpha",             'd', "parameter that determines functional form of BasisSetExp potential"},
    {"splinesmoothfactor",'d', "amount of smoothing in SplineExp initialized from an N-body snapshot"},
    {"splinermin",        'd', "if nonzero, specifies the innermost grid node radius for SplineExp and CylSplineExp"},
    {"splinermax",        'd', "if nonzero, specifies the outermost grid node radius for SplineExp and CylSplineExp"},
    {"splinezmin",        'd', "if nonzero, specifies the z-value of the innermost grid node in CylSplineExp"},
    {"splinezmax",        'd', "if nonzero, specifies the z-value of the outermost grid node in CylSplineExp"},
    {NULL,0,NULL}
};

/// this string will contain full list of parameters and other relevant info to be printed out via `help(Potential)`
static char docstringPotential[MAX_LEN_DOCSTRING];

static const char* keywordsPotential[MAX_NUM_KEYWORDS] = {NULL};

static char keywordTypesPotential[MAX_NUM_KEYWORDS+1] = "|";

/// build the docstring for Potential class
static void buildDocstringPotential()
{
    // header line, store the number of symbols printed
    unsigned int bufIndex = snprintf(docstringPotential, MAX_LEN_DOCSTRING,
        "Potential is a class that represents a wide range of gravitational potentials\n"
        "There are a number of possible named arguments for the constructor:\n\n");
    unsigned int argIndex = 0;
    while(potentialArgs[argIndex].name != NULL && bufIndex < MAX_LEN_DOCSTRING && argIndex < MAX_NUM_KEYWORDS) {
        bufIndex += snprintf(docstringPotential + bufIndex, MAX_LEN_DOCSTRING - bufIndex,
            "    %s (%s) - %s\n", potentialArgs[argIndex].name,
            argTypeDescr(potentialArgs[argIndex].type), potentialArgs[argIndex].descr);
        keywordsPotential[argIndex] = potentialArgs[argIndex].name;
        keywordTypesPotential[argIndex+1] = potentialArgs[argIndex].type;
        argIndex++;
    }
    if(bufIndex < MAX_LEN_DOCSTRING) {
        bufIndex += snprintf(docstringPotential + bufIndex, MAX_LEN_DOCSTRING - bufIndex,
            "\nRequired parameters are either 'type' or 'file' (or both)\n");
    }
    if(bufIndex >= MAX_LEN_DOCSTRING || argIndex >= MAX_NUM_KEYWORDS)
    {   // overflow shouldn't occur, but if it does, issue a warning
        printf("WARNING: Could not properly initialize Potential class docstring\n");
    }
}

static int Potential_init(PyObject* self, PyObject* args, PyObject* namedargs)
{
    potential::ConfigPotential cfg;
    const char* file="";
    const char* type="";
    const char* density="";
    const char* symmetry="";
    PyObject* points=NULL;

    // it is VITALLY IMPORTANT to list all data fields in exactly the same order
    // as appeared in the `potentialArgs` array!!!
    if(!PyArg_ParseTupleAndKeywords(args, namedargs, keywordTypesPotential, const_cast<char**>(keywordsPotential),
        &file, &type, &density, &symmetry,
        &points,
        &(cfg.mass), &(cfg.scaleRadius), &(cfg.scaleRadius2), &(cfg.q), &(cfg.p), &(cfg.gamma), &(cfg.sersicIndex),
        &(cfg.numCoefsRadial), &(cfg.numCoefsAngular), &(cfg.numCoefsVertical), 
        &(cfg.alpha), &(cfg.splineSmoothFactor), &(cfg.splineRMin), &(cfg.splineRMax), &(cfg.splineZMin), &(cfg.splineZMax) ))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to the Potential constructor;\n"
            "type 'help(Potential)' to get the list of possible arguments and their types");
        return -1;
    }
#ifdef DEBUGPRINT
    utils::KeyValueMap kv;
    storeConfigPotential(cfg, kv);
    printf("%s", kv.dump().c_str());
#endif
    cfg.fileName = std::string(file);
    cfg.potentialType = potential::getPotentialTypeByName(type);
    cfg.densityType = potential::getDensityTypeByName(density);
    cfg.symmetryType = potential::getSymmetryTypeByName(symmetry);
    if(cfg.potentialType == potential::PT_UNKNOWN) {
        if(type[0]==0)
            PyErr_SetString(PyExc_ValueError, "Should provide type='...' or file='...' parameter");
        else
            PyErr_SetString(PyExc_ValueError, "Incorrect type='...' parameter");
        return -1;
    }
    try{  // the code below may generate exceptions which shouldn't propagate to Python
        const potential::BasePotential* pot=NULL;
        if(points!=NULL) {  // a list of particles was provided
            if( file[0]!=0 ) {
                PyErr_SetString(PyExc_ValueError, "Cannot provide both points and filename");
                return -1;
            }
            if( cfg.potentialType != potential::PT_BSE &&
                cfg.potentialType != potential::PT_SPLINE &&
                cfg.potentialType != potential::PT_CYLSPLINE ) {
                PyErr_SetString(PyExc_ValueError, "Potential should be of an expansion type");
                return -1;
            }
            PyObject *pointCoordObj, *pointMassObj;
            if(!PyArg_ParseTuple(points, "OO", &pointCoordObj, &pointMassObj)) {
                PyErr_SetString(PyExc_ValueError, "'points' must be a tuple with two arrays - "
                    "coordinates and mass, where the first one is a two-dimensional Nx3 array "
                    "and the second one is a one-dimensional array of length N");
                return -1;
            }
            PyArrayObject *pointCoordArr = (PyArrayObject*)
                PyArray_FROM_OTF(pointCoordObj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            PyArrayObject *pointMassArr  = (PyArrayObject*)
                PyArray_FROM_OTF(pointMassObj,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            if(pointCoordArr == NULL || pointMassArr == NULL) {
                Py_XDECREF(pointCoordArr);
                Py_XDECREF(pointMassArr);
                PyErr_SetString(PyExc_ValueError, "'points' does not contain valid arrays");
                return -1;
            }
            int numpt = 0;
            if(PyArray_NDIM(pointMassArr) == 1)
                numpt = PyArray_DIM(pointMassArr, 0);
            if(numpt == 0 || PyArray_NDIM(pointCoordArr) != 2 || 
                PyArray_DIM(pointCoordArr, 0) != numpt || PyArray_DIM(pointCoordArr, 1) != 3)
            {
                Py_XDECREF(pointCoordArr);
                Py_XDECREF(pointMassArr);
                PyErr_SetString(PyExc_ValueError, "'points' does not contain valid arrays "
                    "(the first one must be 2d array of shape Nx3 and the second one must be 1d array of length N)");
                return -1;
            }
            particles::PointMassArray<coord::PosCar> pointArray;
            pointArray.data.reserve(numpt);
            for(int i=0; i<numpt; i++) {
                double x = *static_cast<double*>(PyArray_GETPTR2(pointCoordArr, i, 0));
                double y = *static_cast<double*>(PyArray_GETPTR2(pointCoordArr, i, 1));
                double z = *static_cast<double*>(PyArray_GETPTR2(pointCoordArr, i, 2));
                double m = *static_cast<double*>(PyArray_GETPTR1(pointMassArr, i));
                pointArray.add(coord::PosCar(x,y,z), m);
            }
            if(PyErr_Occurred()) {  // numerical conversion error
                return -1;
            }
            pot = potential::createPotentialFromPoints(cfg, pointArray);
        } else {
            pot = potential::createPotential(cfg);  // attempt to create potential from configuration parameters
        }
        if(!pot) {  // shouldn't get here -- either everything is ok, or we got a C++ exception already
            PyErr_SetString(PyExc_ValueError, "Incorrect parameters for the potential");
            return -1;
        }
#ifdef DEBUGPRINT
        printf("Created an instance of %s potential\n", pot->name());
#endif
        if(((PotentialObject*)self)->pot) {
#ifdef DEBUGPRINT
            printf("Deleted previous instance of %s potential\n", ((PotentialObject*)self)->pot->name());
#endif
            delete ((PotentialObject*)self)->pot;
        }
        ((PotentialObject*)self)->pot = pot;
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
    if(((PotentialObject*)self)->pot==NULL) {
        PyErr_SetString(PyExc_ValueError, "Potential is not initialized properly");
        return false;
    }
    return true;
}

/// any function that evaluates something for a given potential object
/// and stores one or more values in the `result` array
typedef void (*potentialFunction) 
    (const potential::BasePotential* pot, const coord::PosCar& pos, double *result);

/// potentialFunction output type
enum OUTPUT_VALUE {
    OUTPUT_VALUE_SINGLE,   ///< potential or density (scalar values)
    OUTPUT_VALUE_TRIPLET,  ///< potential derivatives
    OUTPUT_VALUE_TRIPLET_AND_SEXTET  ///< derivatives and second derivatives
};

/// function that computes something from the Potential object for one or many input points
static PyObject* Potential_AnyFunction(PyObject* self, PyObject* args, 
    potentialFunction fnc, OUTPUT_VALUE outputValue)
{
    coord::PosCar pos;
    if(!Potential_isCorrect(self))
        return NULL;
    if(PyArg_ParseTuple(args, "ddd", &pos.x, &pos.y, &pos.z)) {  // one point
        double result[9];
        fnc(((PotentialObject*)self)->pot, pos, result);
        if(outputValue == OUTPUT_VALUE_TRIPLET_AND_SEXTET)
            return Py_BuildValue("(ddd)(dddddd)", result[0], result[1], result[2],
                result[3], result[4], result[5], result[6], result[7], result[8]);
        else if(outputValue == OUTPUT_VALUE_TRIPLET)
            return Py_BuildValue("ddd", result[0], result[1], result[2]);
        else
            return Py_BuildValue("d", result[0]);
    }
    PyErr_Clear();
    PyObject* obj;
    if(PyArg_ParseTuple(args, "O", &obj)) {
        PyArrayObject *arr  = (PyArrayObject*) PyArray_FROM_OTF(obj,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if(arr == NULL) {
            PyErr_SetString(PyExc_ValueError, "Input does not contain valid array");
            return NULL;
        }
        int numpt = 0;
        if(PyArray_NDIM(arr) == 1 && PyArray_DIM(arr, 0) == 3) {  // 1d array of length 3 - a single point
            pos.x = *static_cast<double*>(PyArray_GETPTR1(arr, 0));
            pos.y = *static_cast<double*>(PyArray_GETPTR1(arr, 1));
            pos.z = *static_cast<double*>(PyArray_GETPTR1(arr, 2));
            Py_DECREF(arr);
            double result[9];
            fnc(((PotentialObject*)self)->pot, pos, result);
            if(outputValue == OUTPUT_VALUE_TRIPLET_AND_SEXTET)
                return Py_BuildValue("(ddd)(dddddd)", result[0], result[1], result[2],
                    result[3], result[4], result[5], result[6], result[7], result[8]);
            else if(outputValue == OUTPUT_VALUE_TRIPLET)
                return Py_BuildValue("ddd", result[0], result[1], result[2]);
            else
                return Py_BuildValue("d", result[0]);
        }
        if(PyArray_NDIM(arr) == 2 && PyArray_DIM(arr, 1) == 3)
            numpt = PyArray_DIM(arr, 0);
        else {
            PyErr_SetString(PyExc_ValueError, "Input does not contain valid Nx3 array");
            Py_DECREF(arr);
            return NULL;
        }
        npy_intp dims1[2] = {numpt, 3};  // either a 1d array or a 2d array Nx3
        npy_intp dims2[2] = {numpt, 6};  // 2d array Nx6, used only for second derivatives of potential
        PyArrayObject* out1 = (PyArrayObject*)PyArray_SimpleNew(
            outputValue == OUTPUT_VALUE_SINGLE ? 1 : 2, dims1, NPY_DOUBLE);
        PyArrayObject* out2 = NULL;
        if(outputValue == OUTPUT_VALUE_TRIPLET_AND_SEXTET)
            out2 = (PyArrayObject*)PyArray_SimpleNew(2, dims2, NPY_DOUBLE);
        for(int i=0; i<numpt; i++) {
            pos.x = *static_cast<double*>(PyArray_GETPTR2(arr, i, 0));
            pos.y = *static_cast<double*>(PyArray_GETPTR2(arr, i, 1));
            pos.z = *static_cast<double*>(PyArray_GETPTR2(arr, i, 2));
            double result[9];
            fnc(((PotentialObject*)self)->pot, pos, result);
            if(outputValue == OUTPUT_VALUE_SINGLE)
                ((double*)PyArray_DATA(out1))[i] = result[0];
            else for(int d=0; d<3; d++)
                ((double*)PyArray_DATA(out1))[i*3+d] = result[d];
            if(outputValue == OUTPUT_VALUE_TRIPLET_AND_SEXTET) 
                for(int d=0; d<6; d++)
                    ((double*)PyArray_DATA(out2))[i*6+d] = result[d+3];
        }
        Py_DECREF(arr);
        if(outputValue == OUTPUT_VALUE_TRIPLET_AND_SEXTET)
            return Py_BuildValue("NN", out1, out2);
        else
            return PyArray_Return(out1);        
    }
    PyErr_SetString(PyExc_ValueError, "Input does not contain valid data "
        "(either 3 coordinates of a single point or a Nx3 array)");
    return NULL;
}

void potfuncPotential(const potential::BasePotential* pot, const coord::PosCar& pos, double *result) {
    result[0] = pot->value(pos);
}
static PyObject* Potential_potential(PyObject* self, PyObject* args) {
    return Potential_AnyFunction(self, args, potfuncPotential, OUTPUT_VALUE_SINGLE);
}

void potfuncDensity(const potential::BasePotential* pot, const coord::PosCar& pos, double *result) {
    result[0] = pot->density(pos);
}
static PyObject* Potential_density(PyObject* self, PyObject* args) {
    return Potential_AnyFunction(self, args, potfuncDensity, OUTPUT_VALUE_SINGLE);
}

void potfuncForce(const potential::BasePotential* pot, const coord::PosCar& pos, double *result) {
    coord::GradCar grad;
    pot->eval(pos, NULL, &grad);
    result[0] = -grad.dx;
    result[1] = -grad.dy;
    result[2] = -grad.dz;
}
static PyObject* Potential_force(PyObject* self, PyObject* args) {
    return Potential_AnyFunction(self, args, potfuncForce, OUTPUT_VALUE_TRIPLET);
}

void potfuncForceDeriv(const potential::BasePotential* pot, const coord::PosCar& pos, double *result) {
    coord::GradCar grad;
    coord::HessCar hess;
    pot->eval(pos, NULL, &grad, &hess);
    result[0] = -grad.dx;
    result[1] = -grad.dy;
    result[2] = -grad.dz;
    result[3] = -hess.dx2;
    result[4] = -hess.dy2;
    result[5] = -hess.dz2;
    result[6] = -hess.dxdy;
    result[7] = -hess.dydz;
    result[8] = -hess.dxdz;
}
static PyObject* Potential_force_deriv(PyObject* self, PyObject* args) {
    return Potential_AnyFunction(self, args, potfuncForceDeriv, OUTPUT_VALUE_TRIPLET_AND_SEXTET);
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
    potential::PotentialType t = potential::getPotentialType(*((PotentialObject*)self)->pot);
    if (t != potential::PT_BSE &&
        t != potential::PT_SPLINE &&
        t != potential::PT_CYLSPLINE ) {
        PyErr_SetString(PyExc_ValueError, "Potential is not of an expansion type");
        return NULL;
    }
    try{
        writePotential(filename, *((PotentialObject*)self)->pot);
        Py_INCREF(Py_None);
        return Py_None;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, (std::string("Error writing file: ")+e.what()).c_str());
        return NULL;
    }
}

static PyMethodDef Potential_methods[] = {
    { "name", (PyCFunction)Potential_name, METH_NOARGS, 
      "Return the name of the potential\nNo arguments\nReturns: string" },
    { "potential", Potential_potential, METH_VARARGS, 
      "Compute potential at given point\nArguments: x,y,z (float)\nReturns: float" },
    { "density", Potential_density, METH_VARARGS, 
      "Compute density at given point\nArguments: x,y,z (float)\nReturns: float" },
    { "force", Potential_force, METH_VARARGS, 
      "Compute force at given point\nArguments: x,y,z (float)\nReturns: float[3] - x,y,z components of force" },
    { "force_deriv", Potential_force_deriv, METH_VARARGS, 
      "Compute force and its derivatives at given point\nArguments: x,y,z (float)\nReturns: (float[3],float[6]) - "
      "x,y,z components of force, and the matrix of force derivatives stored as dFx/dx,dFy/dy,dFz/dz,dFx/dy,dFy/dz,dFz/dx" },
    { "export", (PyCFunction)Potential_export, METH_VARARGS, 
      "Export potential expansion coefficients to a text file\nArguments: filename (string)\nReturns: none" },
    { NULL, NULL, 0, NULL }
};

static PyTypeObject PotentialType = {
    PyObject_HEAD_INIT(NULL)
    0, "py_wrapper.Potential",
    sizeof(PotentialObject), 0, (destructor)Potential_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Potential_name, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringPotential, 
    0, 0, 0, 0, 0, 0, Potential_methods, 0, 0, 0, 0, 0, 0, 0,
    Potential_init, 0, /*Potential_new*/
};

///@}
/// \name  --------- SplineApprox class -----------
///@{

typedef struct {
    PyObject_HEAD
    math::CubicSpline* spl;
} SplineApproxObject;

static void SplineApprox_dealloc(SplineApproxObject* self)
{
    if(self->spl)
        delete self->spl;
    self->ob_type->tp_free((PyObject*)self);
}

const char* docstringSplineApprox = 
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

static int SplineApprox_init(PyObject* self, PyObject* args, PyObject* namedargs)
{
    ((SplineApproxObject*)self)->spl=NULL;
    static const char* keywords[] = {"x","y","knots","smooth",NULL};
    PyObject* objx=NULL;
    PyObject* objy=NULL;
    PyObject* objk=NULL;
    double smoothfactor=0;
    if(!PyArg_ParseTupleAndKeywords(args, namedargs, "OOO|d", const_cast<char **>(keywords),
        &objx, &objy, &objk, &smoothfactor)) {
        PyErr_SetString(PyExc_ValueError, "Incorrect parameters passed to the SplineApprox constructor: "
            "must provide two arrays of equal length (input x and y points), "
            "a third array of spline knots, and optionally a float (smooth factor)");
        return -1;
    }
    PyArrayObject *arrx = (PyArrayObject*) PyArray_FROM_OTF(objx, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arry = (PyArrayObject*) PyArray_FROM_OTF(objy, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arrk = (PyArrayObject*) PyArray_FROM_OTF(objk, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(arrx == NULL || arry == NULL || arrk == NULL) {
        Py_XDECREF(arrx);
        Py_XDECREF(arry);
        Py_XDECREF(arrk);
        PyErr_SetString(PyExc_ValueError, "Input does not contain valid arrays");
        return -1;
    }
    int numpt = 0;
    if(PyArray_NDIM(arrx) == 1)
        numpt = PyArray_DIM(arrx, 0);
    int numknots = 0;
    if(PyArray_NDIM(arrk) == 1)
        numknots = PyArray_DIM(arrk, 0);
    if(numpt <= 0 || numknots <= 4|| PyArray_NDIM(arry) != 1 || PyArray_DIM(arry, 0) != numpt) {
        Py_DECREF(arrx);
        Py_DECREF(arry);
        Py_DECREF(arrk);
        PyErr_SetString(PyExc_ValueError, 
            "Arguments must be two arrays of equal length (x and y) and a third array (knots, at least 4)");
        return -1;
    }
    std::vector<double> xvalues((double*)PyArray_DATA(arrx), (double*)PyArray_DATA(arrx) + numpt);
    std::vector<double> yvalues((double*)PyArray_DATA(arry), (double*)PyArray_DATA(arry) + numpt);
    std::vector<double> knots((double*)PyArray_DATA(arrk), (double*)PyArray_DATA(arrk) + numknots);
    try{
        math::SplineApprox spl(xvalues, knots);
        std::vector<double> splinevals;
        double der1, der2;
        if(smoothfactor>0)
            spl.fitDataOversmooth(yvalues, smoothfactor, splinevals, der1, der2);
        else
            spl.fitData(yvalues, -smoothfactor, splinevals, der1, der2);
        ((SplineApproxObject*)self)->spl = new math::CubicSpline(knots, splinevals, der1, der2);
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, (std::string("Error in spline initialization: ")+e.what()).c_str());
        return -1;
    }
    return 0;
}

double spl_eval(const math::CubicSpline* spl, double x, int der=0)
{
    double result;
    switch(der) {
        case 0: return spl->value(x);
        case 1: spl->evalDeriv(x, NULL, &result); return result;
        case 2: spl->evalDeriv(x, NULL, NULL, &result); return result;
        default: return NAN;
    }
}

static PyObject* SplineApprox_value(PyObject* self, PyObject* args, PyObject* /*kw*/)
{
    PyObject* ptx=NULL;
    int der=0;
    if(self==NULL || ((SplineApproxObject*)self)->spl==NULL || !PyArg_ParseTuple(args, "O|i", &ptx, &der))
        return NULL;
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
    0, "py_wrapper.SplineApprox",
    sizeof(SplineApproxObject), 0, (destructor)SplineApprox_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, SplineApprox_value, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringSplineApprox, 
    0, 0, 0, 0, 0, 0, SplineApprox_methods, 0, 0, 0, 0, 0, 0, 0,
    SplineApprox_init, 0, 
};
///@}

static PyMethodDef py_wrapper_methods[] = {
    {NULL}
};

PyMODINIT_FUNC
initpy_wrapper(void)
{
    PyObject* mod = Py_InitModule("py_wrapper", py_wrapper_methods);
    if(!mod) return;

    // Potential class
    buildDocstringPotential();
    PotentialType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PotentialType) < 0) return;
    Py_INCREF(&PotentialType);
    PyModule_AddObject(mod, "Potential", (PyObject *)&PotentialType);

    SplineApproxType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&SplineApproxType) < 0) return;
    Py_INCREF(&SplineApproxType);
    PyModule_AddObject(mod, "SplineApprox", (PyObject *)&SplineApproxType);

    import_array();
}

/// \endcond