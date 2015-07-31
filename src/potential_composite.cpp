#include "potential_composite.h"

namespace potential{

double CompositeDensity::densityCar(const coord::PosCar &pos) const {
    double sum=0; 
    for(unsigned int i=0; i<components.size(); i++) 
        sum+=components[i]->density(pos); 
    return sum;
}
double CompositeDensity::densityCyl(const coord::PosCyl &pos) const {
    double sum=0; 
    for(unsigned int i=0; i<components.size(); i++) 
        sum+=components[i]->density(pos); 
    return sum;
}
double CompositeDensity::densitySph(const coord::PosSph &pos) const {
    double sum=0; 
    for(unsigned int i=0; i<components.size(); i++) 
        sum+=components[i]->density(pos); 
    return sum;
}

SymmetryType CompositeDensity::symmetry() const {
    int sym=static_cast<int>(ST_SPHERICAL);
    for(unsigned int index=0; index<components.size(); index++)
        sym &= static_cast<int>(components[index]->symmetry());
    return static_cast<SymmetryType>(sym);
};

void CompositeCyl::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const
{
    if(potential) *potential=0;
    if(deriv)  deriv->dR=deriv->dz=deriv->dphi=0;
    if(deriv2) deriv2->dR2=deriv2->dz2=deriv2->dphi2=deriv2->dRdz=deriv2->dRdphi=deriv2->dzdphi=0;
    for(unsigned int i=0; i<components.size(); i++) {
        coord::GradCyl der;
        coord::HessCyl der2;
        double pot;
        components[i]->eval(pos, potential?&pot:0, deriv?&der:0, deriv2?&der2:0);
        if(potential) *potential+=pot;
        if(deriv) { 
            deriv->dR  +=der.dR;
            deriv->dz  +=der.dz;
            deriv->dphi+=der.dphi;
        }
        if(deriv2) {
            deriv2->dR2   +=der2.dR2;
            deriv2->dz2   +=der2.dz2;
            deriv2->dphi2 +=der2.dphi2;
            deriv2->dRdz  +=der2.dRdz;
            deriv2->dRdphi+=der2.dRdphi;
            deriv2->dzdphi+=der2.dzdphi;
        }
    }
}

SymmetryType CompositeCyl::symmetry() const {
    int sym=static_cast<int>(ST_SPHERICAL);
    for(unsigned int index=0; index<components.size(); index++)
        sym &= static_cast<int>(components[index]->symmetry());
    return static_cast<SymmetryType>(sym);
};

} // namespace potential