#ifndef COORDSYS_H
#define COORDSYS_H

#include "nr3.h"
#include "utils.hpp"

class OblateSpheroidCoordSys{
	private:
		double Alpha;
		const double Gamma;
	public:
		OblateSpheroidCoordSys(double alpha): Alpha(alpha), Gamma(-1.){}
		inline double alpha(void) const { return Alpha;}
		inline double gamma(void) const { return Gamma;}
		inline void newalpha(double a) {Alpha = a;}
		VecDoub x2tau(VecDoub_I x) const;
		VecDoub tau2x(VecDoub_I x) const;
		VecDoub xv2tau(VecDoub_I x) const;
		VecDoub derivs(VecDoub_I x) const;
		VecDoub tau2p(VecDoub_I tau) const;
};

class ConfocalEllipsoidalCoordSys{
	private:
		double Alpha, Beta, Gamma;
	public:
		ConfocalEllipsoidalCoordSys(double a, double b): Alpha(a), Beta(b), Gamma(-1.){}
		inline double alpha(void) const { return Alpha;}
		inline double beta(void) const { return Beta;}
		inline double gamma(void) const { return Gamma;}
		inline void newalpha(double a) { Alpha = a;}
		inline void newbeta(double b){ Beta = b;}
		VecDoub x2tau(VecDoub_I x) const;
		VecDoub tau2x(VecDoub_I tau) const;
		VecDoub xv2tau(VecDoub_I x) const;
		VecDoub tau2p(VecDoub_I tau) const;
		VecDoub derivs(VecDoub_I x) const;
};

#endif
