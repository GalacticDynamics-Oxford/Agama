/** \file    smart.h
    \brief   Forward declarations of fundamental classes and smart pointers to them
    \author  Eugene Vasiliev
    \date    2015

This file teaches how to be smart and use modern practices in C++ programming.
Smart pointers are a convenient and robust approach for automated memory management.
One does not need to worry about keeping track of dynamically created objects,
does not risk memory leaks or dangling pointers.
In short, instead of using ordinary pointers for handling dynamically allocated objects,
we wrap them into special objects that behave almost like pointers, but are smarter.
The key point is the following C++ feature: when a local variable gets out of scope,
and if it was an instance of some class, its destructor is called automatically,
no matter how the control flow goes (whether we return from anywhere inside a routine,
or even throw an exception which propagates up the call stack).
Ordinary pointers are not objects, but smart pointer wrappers are, and they take care
of deleting the ordinary pointer when they get out of scope or are manually released.

We use only one kind of smart pointers, namely shared_ptr. As implied by its name,
this classl allows shared ownership of an object by several pointers, ensuring that
the object under control stays alive as long as there is at least one smart pointer
that keeps track of it (in other words, it implements reference counting approach).
This is used to pass around wrapped objects between different parts of code that do not
have a predetermined execution order or lifetime. For example, an instance of Potential
can be passed to an ActionFinder that makes a copy of the pointer and keeps it as long
as the action finder itself stays alive, even though the original smart pointer might
have been deleted long ago.

A side advantage of shared smart pointers is that they make possible to share
an object whose exact type is not fully specified (i.e. is a derived class in some
hierarchy, for instance a descendant of BasePotential). In general we cannot copy
these objects directly, because it is not possible to assign an instance of derived
class to a variable of base class; instead, one may copy the pointer, but wrap it in
a shared_ptr object that manages the ownership.
We only use this approach for polymorphic class hierarchies, and prefer to copy simpler
objects by value (e.g. spline classes or other structures that contain splines as member
variables) - as long as this does not occur too often, the overhead is negligible.

A few notes about conventions used to pass arguments to function or class constructors.
First, almost all non-trivial classes are intended to be constant objects: once created,
they never change their internal state and only have const methods.
This greatly simplifies the management of objects -- ensuring that they are not changed
by some other parts of code. If one needs to change an object (for instance, update
the potential used in iterative self-consistent modelling framework), one creates
a new instance of this class, and the old one remains alive as long as it is referenced
by any smart pointers.
We only allow simpler structures to be non-constant -- they are usually declared as
struct, not class, and only have public data fields and no methods; instead, one uses
non-member functions that take these structures as arguments.

When passing an instance of some class as an argument to a function, we follow these
rules:
  - by default, the object is passed as const reference.
  - if the object is allowed to be NULL, then it is passed as const raw pointer
(the only allowed usage of raw pointers).
  - as said above, instances of complex classes cannot be modified, but simpler structs
can be; thus sometimes one may need to pass a non-const reference.
  - non-const pointers are not allowed - after all, if you want to modify data in
in a structure, you want this structure to exist (be not NULL) in the first place!

Somewhat different rules apply to passing arguments to a constructor of some class.
In this case, one may need to take a copy of the input data which would persist
for the entire lifetime of the created object. Consequently, if we pass an argument
which can be copied by value (such as std::vector or anything that contains a vector,
like a spline function), same rules as above do apply. However, if we pass a object
that is a polymorphic pointer to some unspecified derived class, we cannot use
raw pointers because their ownership is not known and the data cannot be copied.
In this case, we pass a shared pointer, which will be internally duplicated in
the object (so that it refers to the same underlying raw data object).
In short: if the constructor of a class takes a shared_ptr (for instance, PtrPotential),
this means that a copy of the object is being kept; if it only takes a reference 
to object of some class (e.g. const BasePotential& ), this means that it will only
be used inside the constructor, but not any longer.
*/

#pragma once
// If compiled in the C++11 mode, use shared_ptr from the standard library,
// otherwise use a custom minimalistic implementation bundled with the code.
#if (__cplusplus >= 201103L) || (defined(_MSC_VER) && (_MSC_VER >= 1600))
// have a C++11 compatible compiler
#include <memory>
using std::shared_ptr;
#else
#include "shared_ptr.h"
#endif

namespace math{

class IFunction;
class IFunctionNdim;
class IFunctionNdimAdd;
class BaseInterpolator1d;
class BaseInterpolator2d;

/// shared pointer to a function class
typedef shared_ptr<const IFunction> PtrFunction;
/// shared pointer to a N-dimensional function
typedef shared_ptr<const IFunctionNdim> PtrFunctionNdim;
/// shared pointer to a N-dimensional accumulator function
typedef shared_ptr<const IFunctionNdimAdd> PtrFunctionNdimAdd;
/// shared pointer to a generic 1d interpolation class
typedef shared_ptr<const BaseInterpolator1d> PtrInterpolator1d;
/// shared pointer to a generic 2d interpolation class
typedef shared_ptr<const BaseInterpolator2d> PtrInterpolator2d;

}  // namespace math


namespace potential{

class BaseDensity;
class BasePotential;
class OblatePerfectEllipsoid;
class PhaseVolume;

/// shared pointer to a density object
typedef shared_ptr<const BaseDensity>   PtrDensity;
/// shared pointer to a potential object
typedef shared_ptr<const BasePotential> PtrPotential;
/// shared pointer to an instance of OblatePerfectEllipsoid potential
typedef shared_ptr<const OblatePerfectEllipsoid> PtrOblatePerfectEllipsoid;
/// shared pointer to a PhaseVolume object
typedef shared_ptr<const PhaseVolume>   PtrPhaseVolume;

}  // namespace potential


namespace actions{

class BaseActionFinder;
class BaseActionMapper;

/// shared pointer to an action finder object
typedef shared_ptr<const BaseActionFinder> PtrActionFinder;
/// shared pointer to an action mapper object
typedef shared_ptr<const BaseActionMapper> PtrActionMapper;

}  // namespace actions


namespace df{

class BaseDistributionFunction;
class BaseActionSpaceScaling;

/// shared pointer to a distribution function object
typedef shared_ptr<const BaseDistributionFunction> PtrDistributionFunction;
/// shared pointer to a scaling transformation of the action space
typedef shared_ptr<const BaseActionSpaceScaling> PtrActionSpaceScaling;

}  // namespace df
