// An interface to GSL - hides all the gsl calls behind easier to use functions
// Jason Sanders
#ifndef GSLINTERFACE_H
#define GSLINTERFACE_H

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_sort_double.h>
#include <gsl/gsl_permute.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_ieee_utils.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_monte_vegas.h>
#include <gsl/gsl_siman.h>
#include <vector>

namespace GSLmath{
//=================================================================================================
// RANDOM NUMBERS //
// random number generators - rand_uniform returns random numbers uniformly distributed in the
// interval [0,1], rand_gaussian returns gaussian distributed random numbers with sd = sigma

class rand_base{
	private:
		const gsl_rng_type * TYPE;
		unsigned long int seed;
    public:
       	gsl_rng * r;
     	rand_base(unsigned long int s){
     	// construct random number generator with seed s
     		seed = s;
     		gsl_rng_env_setup();
     	    TYPE = gsl_rng_default;
       		r    = gsl_rng_alloc (TYPE);
       		gsl_rng_set(r, seed);
       		}
       	~rand_base(){gsl_rng_free (r);}
       	void reseed(unsigned long int newseed){
       	// give new seed
       		seed = newseed;
       		gsl_rng_set(r,newseed);
       		}
};

class rand_uniform:public rand_base{
	public:
		rand_uniform(unsigned long int SEED=0):rand_base(SEED){}
		~rand_uniform(){}
		// return uniformly distributed random numbers
		double nextnumber(){return gsl_rng_uniform (r);}
};

class rand_gaussian:public rand_base{
	public:
		double sigma;
		rand_gaussian(double s, unsigned long int SEED=0):rand_base(SEED){sigma = s;}
		~rand_gaussian(){}
		// return gaussian distributed random numbers
		double nextnumber(){return gsl_ran_gaussian (r,sigma);}
		void newsigma(double newsigma){sigma=newsigma;}
};

class rand_exponential:public rand_base{
    public:
        double scale;
        rand_exponential(double scale, unsigned long int SEED=0):rand_base(SEED),scale(scale){}
        ~rand_exponential(){}
        // return exponentially distributed random numbers
        double nextnumber(){return gsl_ran_exponential (r,scale);}
        void new_scale(double newscale){scale=newscale;}
};

//=================================================================================================
// ROOT FINDING	  //
// finds root by Brent's method. Constructor initialises function and tolerances, findroot finds
// root in given interval. Function must be of form double(*func)(double,void*)

class root_find{
	private:
		int status;
       	const gsl_root_fsolver_type *T; gsl_root_fsolver *s;
       	gsl_function F;	double xlo,xhi,eps,root,tol;
       	int iter, max_iter;
    public:
    	root_find(double tol,int max_iter)
    		: tol(tol), max_iter(max_iter){
    		iter=0;eps=1.e-8;
       		T = gsl_root_fsolver_brent;
       		s = gsl_root_fsolver_alloc (T);
		}
		~root_find(){gsl_root_fsolver_free (s);}

    	void bracket(){
    		// crude bracketing routine. Expands interval till root found
			if(xhi<xlo){double xt = xlo; xlo=xhi; xhi=xt;}
			double diff = (xhi-xlo)/2.;
			while(GSL_FN_EVAL(&F,xlo)*GSL_FN_EVAL(&F,xhi)>0.){
				xlo-=diff;xhi+=diff;
			}
		}

     	double findroot(double(*func)(double,void *),double xlo1,double xhi1, void *p=NULL){
     		// finds root in interval [xlo1,xhi1]
    		F.function = func;F.params=p;xhi = xhi1; xlo = xlo1;
     	    if(GSL_FN_EVAL(&F,xlo)*GSL_FN_EVAL(&F,xhi)>0.){
     			bracket();
     		}
       		gsl_root_fsolver_set (s, &F, xlo, xhi);
			do{
				iter++;
				status = gsl_root_fsolver_iterate (s);
				root   = gsl_root_fsolver_root (s);
				xlo    = gsl_root_fsolver_x_lower (s);
				xhi    = gsl_root_fsolver_x_upper (s);
				status = gsl_root_test_interval (xlo, xhi, tol, eps);
			  }
			while (status == GSL_CONTINUE && iter < max_iter);
       		return root;
		}

};

//=================================================================================================
// INTEGRATION //
// Simple 1d numerical integration using adaptive Gauss-Kronrod which can deal with singularities
// constructor takes function and tolerances and integrate integrates over specified region.
// integrand function must be of the form double(*func)(double,void*)
class integrator{
	private:
		gsl_integration_workspace *w;
		void *p;
		double result,err,eps;
		gsl_function F;
		size_t neval;
	public:
		integrator(double eps): eps(eps){
			neval    = 1000;
			w        = gsl_integration_workspace_alloc (neval);
			F.params = &p;
       		}
		~integrator(){gsl_integration_workspace_free (w);}
		double integrate(double(*func)(double,void*),double xa, double xb){
		    F.function = func;
			gsl_integration_qags (&F, xa, xb, 0, eps, neval,w, &result, &err);
			//gsl_integration_qng(&F, xa, xb, 0, eps, &result, &err, &neval);
			return result;
			}
		double error(){return err;}
};

inline double integrate(double(*func)(double,void*),double xa, double xb, double eps){
	double result,err; size_t neval;void *p;gsl_function F;F.function = func;F.params = &p;
	gsl_integration_qng(&F, xa, xb, 0, eps, &result, &err, &neval);
	return result;
}

class MCintegrator{
	private:
		gsl_monte_vegas_state *s;
		const gsl_rng_type *T;
 		gsl_rng *r;
 		size_t dim;
 	public:
 		MCintegrator(size_t Dim){
 			dim=Dim;
 			gsl_rng_env_setup ();
  			T = gsl_rng_default;
  			r = gsl_rng_alloc (T);
 			s = gsl_monte_vegas_alloc(Dim);
 		}
 		~MCintegrator(){
 			gsl_monte_vegas_free(s);
 			gsl_rng_free(r);
 		}
 		double integrate(double(*func)(double*,size_t,void*),double *xlow, double *xhigh,
 		size_t calls, double *err, int burnin=10000){
 			gsl_monte_function G = { func, dim, 0 }; double res;
 			if(burnin)gsl_monte_vegas_integrate(&G,xlow,xhigh,dim,burnin,r,s,&res,err);
 			gsl_monte_vegas_integrate(&G,xlow,xhigh,dim,calls,r,s,&res,err);
 			return res;
 		}
};

//=================================================================================================
// 1D INTERPOLATION //
// Interpolation using cubic splines

class interpolator{
	private:
		gsl_interp_accel *acc;
		gsl_spline *spline;
	public:
		interpolator(double *x, double *y, int n){
			acc = gsl_interp_accel_alloc();
			spline = gsl_spline_alloc(gsl_interp_cspline,n);
			gsl_spline_init (spline, x, y, n);
		}
		~interpolator(){
			gsl_spline_free (spline);
         	gsl_interp_accel_free (acc);
        }
        double interpolate(double xi){
        	return gsl_spline_eval (spline, xi, acc);
        }
        double derivative(double xi){
        	return gsl_spline_eval_deriv(spline, xi, acc);
        }
        void new_arrays(double *x, double *y,int n){
        	spline = gsl_spline_alloc(gsl_interp_cspline,n);
        	gsl_spline_init (spline, x, y, n);
        }
};

//=================================================================================================
// SORTING //
// sorting algorithm
// sort2 sorts first argument and then applies the sorted permutation to second list
class sorter{
	private:
		const gsl_rng_type * T;
       	gsl_rng * r;
    public:
    	sorter(){
    		gsl_rng_env_setup();
            T = gsl_rng_default;
     		r = gsl_rng_alloc (T);
     		}
     	~sorter(){gsl_rng_free (r);}
     	void sort(double *data, int n){
     		gsl_sort(data,1,n);
     	}
     	void sort2(double *data, int n, double *data2){
     		size_t p[n];
     		gsl_sort_index(p,data,1,n);
     		gsl_permute(p,data2,1,n);
     	}

};

//=================================================================================================
// ODE SOLVER //
// Simple ODE integrator using Runge-Kutta Dormand-Prince 8 adaptive stepping
// dy_i/dt = f_i(t) where int (*f)(double t, const double y, double f, void *params)

class ode{
	private:
		const gsl_odeiv2_step_type * T;
		gsl_odeiv2_step * s;
		gsl_odeiv2_control * c;
		gsl_odeiv2_evolve * e;
  		gsl_odeiv2_system sys;
		double h;
		double direction;
	public:
		ode(int (*derivs)(double,const double *,double*,void*),int N,double eps, void *params=NULL){
			sys.function=derivs;sys.jacobian=NULL;sys.dimension=N;sys.params=params;

			T = gsl_odeiv2_step_rk8pd;s = gsl_odeiv2_step_alloc (T, N);
			c = gsl_odeiv2_control_y_new (eps, 0.0);e = gsl_odeiv2_evolve_alloc (N);
			gsl_ieee_env_setup();
		}
		~ode(){
			gsl_odeiv2_evolve_free(e);
  			gsl_odeiv2_control_free(c);
  			gsl_odeiv2_step_free(s);
  		}
		void step(double tstart, double tfinish, double *y, double step){
			h = step;
			direction = (h>0?1.:-1.);
			while ((tfinish-tstart)*direction>0){
      			int status = gsl_odeiv2_evolve_apply (e, c, s, &sys, &tstart, tfinish, &h, y);
      			if (status != GSL_SUCCESS)break;
    		}
		}
};

//=================================================================================================
// MINIMISER //
// finds a minimum of a function of the form double(*func)(const gsl_vector *v, void *params)
// using a downhill simplex algorithm. Setup minimiser with initial guesses and required tolerance
// with constructor and then minimise with minimise().
class minimiser{
	private:
		const gsl_multimin_fminimizer_type *T; ;
		gsl_multimin_fminimizer *s;
		gsl_vector *ss, *x;
		gsl_multimin_function minex_func;
		size_t iter; int status,N_params; double size;
		double eps;
	public:
		minimiser(double(*func)(const gsl_vector *v, void *params),double *parameters,
		 int N, double *sizes, double eps, void *params):N_params(N), eps(eps){

			T = gsl_multimin_fminimizer_nmsimplex2rand;
			ss = gsl_vector_alloc (N_params);x = gsl_vector_alloc (N_params);
			for(int i=0;i<N_params;i++){
				gsl_vector_set (x, i, parameters[i]);gsl_vector_set(ss,i,sizes[i]);}

			minex_func.n = N_params; minex_func.f = func; minex_func.params = params;
			s = gsl_multimin_fminimizer_alloc (T, N_params);
			gsl_multimin_fminimizer_set (s, &minex_func, x, ss);
			status = 0; iter = 0;

		}

		minimiser(double(*func)(const gsl_vector *v, void *params),std::vector<double> parameters,
		 std::vector<double> sizes, double eps,void *params): eps(eps){

			N_params = parameters.size();
			T = gsl_multimin_fminimizer_nmsimplex2rand;
			ss = gsl_vector_alloc (N_params);x = gsl_vector_alloc (N_params);
			for(int i=0;i<N_params;i++){
				gsl_vector_set (x, i, parameters[i]);gsl_vector_set(ss,i,sizes[i]);}

			minex_func.n = N_params; minex_func.f = func; minex_func.params = params;
			s = gsl_multimin_fminimizer_alloc (T, N_params);
			gsl_multimin_fminimizer_set (s, &minex_func, x, ss);
			status = 0; iter = 0;
		}

		~minimiser(){
			gsl_vector_free(x);
			gsl_vector_free(ss);
			gsl_multimin_fminimizer_free (s);
		}

		double minimise(double *results,unsigned int maxiter,bool vocal){
			do
			  {
				iter++; status = gsl_multimin_fminimizer_iterate(s);
				if(status)break;
				size = gsl_multimin_fminimizer_size (s);
				status = gsl_multimin_test_size (size, eps);
//				if(vocal){	std::cout<<iter<<" ";
//							for(int i=0; i<N_params;i++)std::cout<<gsl_vector_get(s->x,i)<<" ";
//							std::cout<<s->fval<<" "<<size<<std::endl;
//							}
			}
			while (status == GSL_CONTINUE && iter < maxiter);
			for(int i=0;i<N_params;i++){results[i] = gsl_vector_get(s->x,i);}
			return s->fval;
		}

		double minimise(std::vector<double> *results,unsigned int maxiter,bool vocal){
			do
			  {
				iter++; status = gsl_multimin_fminimizer_iterate(s);
				if(status)break;
				size = gsl_multimin_fminimizer_size (s);
				status = gsl_multimin_test_size (size, eps);
//				if(vocal){	std::cout<<iter<<" ";
//							for(int i=0; i<N_params;i++)std::cout<<gsl_vector_get(s->x,i)<<" ";
//							std::cout<<s->fval<<" "<<size<<std::endl;
//							}
			}
			while (status == GSL_CONTINUE && iter < maxiter);
			for(int i=0;i<N_params;i++) results->push_back(gsl_vector_get(s->x,i));
			return s->fval;
		}
};

class minimiser1D{
	private:
		const gsl_min_fminimizer_type *T; ;
		gsl_min_fminimizer *s;
		size_t iter; int status;
		double m, a, b, eps;
	public:
		minimiser1D(double(*func)(double, void *params), double m, double a, double b, double eps, void* params)
			:m(m), a(a), b(b), eps(eps){

			gsl_function F;F.function = func;F.params = params;
			T = gsl_min_fminimizer_brent;
			s = gsl_min_fminimizer_alloc (T);
			gsl_min_fminimizer_set (s, &F, m, a, b);
			status = 0; iter = 0;
		}
		~minimiser1D(){
			gsl_min_fminimizer_free (s);
		}
		double minimise(unsigned int maxiter){
			do
			  {
				iter++;
				status = gsl_min_fminimizer_iterate(s);
				m = gsl_min_fminimizer_x_minimum (s);
           		a = gsl_min_fminimizer_x_lower (s);
           		b = gsl_min_fminimizer_x_upper (s);
				status = gsl_min_test_interval (a, b, eps, 0.0);
			}
			while (status == GSL_CONTINUE && iter < maxiter);
			return m;
		}
};
/*
double Distance(void *xp, void *yp){
       double x = *((double *) xp);
       double y = *((double *) yp);
       return fabs(x - y);
}

void Step(const gsl_rng * r, void *xp, double step_size){
    double old_x = *((double *) xp);
    double new_x;

    double u = gsl_rng_uniform(r);
    new_x = u * 2 * step_size - step_size + old_x;

    memcpy(xp, &new_x, sizeof(new_x));
}

void Print(void *xp){
    printf ("%12g", *((double *) xp));
}

class sim_anneal{
	private:
		const gsl_rng_type * T;
    	gsl_rng * r;
    	int N_TRIES, ITER_FIXED_T;
    	double STEP_SIZE, K, T_INITIAL, MU_T, T_MIN;
    	gsl_siman_params_t params;
    public:
    	sim_anneal(int N_TRIES, int ITER_FIXED_T, double STEP_SIZE):
    	N_TRIES(N_TRIES), ITER_FIXED_T(ITER_FIXED_T),STEP_SIZE(STEP_SIZE){
    		gsl_rng_env_setup();
            T = gsl_rng_default;
     	  	r = gsl_rng_alloc(T);
     	  	//params[0]=N_TRIES;params[1]=ITER_FIZED_T;params[2]=STEP_SIZE;
     	  	//K=1.; params[3]=K; T_INITIAL=0.008; params[4]=T_INITIAL;
     	  	//MU_T=1.003; params[5]=MU_T; T_MIN=2.0e-6; params[6]=T_MIN;
     	  	gsl_siman_params_t params
       = {N_TRIES, ITERS_FIXED_T, STEP_SIZE,
          K, T_INITIAL, MU_T, T_MIN};
    	}
    	~sim_anneal(){
    		gsl_rng_free(r);
    	}
    	double minimise(double(*func)(void *xp), double x){
    		double x_initial=x;
    		gsl_siman_solve(r, &x_initial, &func, Step, Distance, Print,
                       		NULL, NULL, NULL,
                       		sizeof(double), params);
    		return x_initial;
    	}
};*/

//=================================================================================================
// SPECIAL FUNCTIONS //
inline double erf(double x){return gsl_sf_erf (x);}
inline double erfc(double x){return gsl_sf_erfc (x);}
inline double besselI(double x, int n){return gsl_sf_bessel_In (n,x);}
inline double besselJ(double x, int n){return gsl_sf_bessel_Jn (n,x);}
inline double gamma(double x){return gsl_sf_gamma (x);}
inline double ellint_first(double phi, double k){ return gsl_sf_ellint_F(phi,k,(gsl_mode_t)1e-15);}
// F(\phi,k) = \int_0^\phi \d t \, \frac{1}{\sqrt{1-k^2\sin^2 t}}
inline double ellint_second(double phi, double k){ return gsl_sf_ellint_E(phi,k,(gsl_mode_t)1e-15);}
// E(\phi,k) = \int_0^\phi \d t \, \sqrt{1-k^2\sin^2 t}
inline double ellint_third(double phi, double k, double n){ return gsl_sf_ellint_P(phi,k,n,(gsl_mode_t)1e-15);}
// \Pi(\phi,k,n) = \int_0^\phi \d t \, \frac{1}{(1+n\sin^2 t)\sqrt{1-k^2\sin^2 t}}

}  // namespace
#endif
