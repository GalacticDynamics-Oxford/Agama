# AGAMA Examples

This directory contains comprehensive examples demonstrating various features and applications of the AGAMA library.

## Quick Start

All examples can be run from this directory:

```bash
# Basic example
python example_self_consistent_model_simple.py

# Interactive tutorials (requires Jupyter)
jupyter notebook tutorial_potential_orbits.ipynb
```

## Categories

### Basic Examples

**Getting Started:**
- `example_self_consistent_model_simple.py` - Minimal self-consistent galaxy model
- `example_target.py` - Basic target/constraint usage

**Potentials and Orbits:**
- `example_torus.py` - Torus orbital structures
- `example_poincare.py` - Poincaré surfaces of section
- `example_time_dependent_potential.py` - Time-varying potentials
- `example_lyapunov.py` - Lyapunov exponents and orbital chaos

### Self-Consistent Models

**Basic Models:**
- `example_self_consistent_model.py` - Standard self-consistent galaxy model
- `example_self_consistent_model3.py` - Alternative implementation
- `example_self_consistent_model_flattened.py` - Flattened galaxy model

**Advanced Models:**
- `example_self_consistent_model_mw.py` - Milky Way-like galaxy model
- `example_doublepowerlaw.py` - Double power-law models
- `example_adiabatic_contraction.py` - Adiabatic contraction effects

### Milky Way Models

**Realistic MW Models:**
- `example_mw_potential_hunter24.py` - MW potential from Hunter et al. 2024
- `example_mw_bar_potential.py` - Barred MW potential
- `example_mw_nsd.py` - Nuclear stellar disk component
- `example_lmc_mw_interaction.py` - LMC-MW interaction

### Schwarzschild Modeling

**Orbit-Superposition Models:**
- `schwarzschild.py` - General Schwarzschild modeling framework
- `example_schwarzschild_triaxial.py` - Triaxial Schwarzschild model
- `example_schwarzschild_flattened_rotating.py` - Flattened rotating model

### Action-Angle Coordinates

**Stellar Dynamics:**
- `example_actions_nbody.py` - Action-angle coordinates from N-body data
- `example_basis_set.py` - Basis function expansions

### Streams and Tidal Effects

**Tidal Streams:**
- `example_tidal_stream.py` - Tidal stream modeling
- `example_spiral.py` - Spiral structure effects

### Distribution Functions

**DF Fitting:**
- `example_df_fit.py` - Distribution function fitting
- `example_vdf_fit_bspline.py` - Velocity distribution function fitting with B-splines

### Specialized Applications

**Forstand (Schwarzschild Code):**
- `example_forstand.py` - Forstand orbit-superposition modeling

**Shape Measurement:**
- `measureshape.py` - Measure axis ratios of N-body snapshots

**Mathematical Tools:**
- `example_smoothing_spline.py` - Smoothing spline techniques
- `example_deprojection.py` - Deprojection methods

### Integration with Other Codes

**External Software:**
- `example_gala.py` - Integration with gala package
- `example_galpy.py` - Integration with galpy package
- `example_amuse.py` - Integration with AMUSE framework
- `example_amuse_raga.py` - AMUSE with Raga Monte Carlo code

### N-body Simulations

**Simulation Setup:**
- `example_nbody_simulation.py` - N-body simulation preparation
- `example_nbody_simulation_gadget4.param` - Gadget-4 parameter file
- `example_nbody_simulation_gadget4.patch` - Gadget-4 modifications
- `example_nbody_simulation_arepo.param` - Arepo parameter file
- `example_nbody_simulation_arepo.patch` - Arepo modifications
- `example_gizmo_snapshot.py` - GIZMO snapshot analysis

### Gaia Challenge

**Research Applications:**
- `gc_runfit.py` - Main Gaia Challenge fitting routine
- `gc_modelparamsE.py` - Energy-angular momentum models
- `gc_modelparamsJ.py` - Action-based models
- `gc_resample.py` - Missing data handling

## Interactive Tutorials

**Jupyter Notebooks:**
- `tutorial_potential_orbits.ipynb` - Comprehensive potentials and orbits tutorial
- `tutorial_streams.ipynb` - Tidal streams tutorial

These provide step-by-step interactive guides with plots and explanations.

## Running Examples

### Prerequisites

Most examples require:
```bash
pip install agama matplotlib numpy scipy
```

Some examples have additional requirements:
```bash
# For optimization examples
pip install cvxopt

# For Jupyter tutorials
pip install jupyter

# For integration examples
pip install gala galpy  # as needed
```

### Usage Patterns

**Basic Usage:**
```bash
cd examples/
python example_self_consistent_model_simple.py
```

**With Custom Parameters:**
Many examples accept command-line arguments or can be easily modified at the top of the file.

**Data Requirements:**
Some examples require input data files. Check the example header for details on:
- Required data formats
- Sample data locations
- Data generation procedures

## Example Structure

Most examples follow this pattern:

1. **Setup**: Import libraries, set units
2. **Configuration**: Define model parameters
3. **Model Creation**: Build potentials, density profiles, DFs
4. **Computation**: Run simulations, fitting, analysis
5. **Output**: Save results, create plots

## Tips

- **Start Simple**: Begin with `example_self_consistent_model_simple.py`
- **Read Headers**: Each example has detailed comments explaining its purpose
- **Modify Parameters**: Examples are designed to be easily customized
- **Check Requirements**: Some examples need specific data files or external packages
- **Performance**: Large examples may take significant time to run

## Getting Help

- **Documentation**: See main project documentation
- **Comments**: Each example has extensive inline documentation
- **Issues**: Report problems via the project issue tracker
- **Community**: Discuss examples in project forums

## Contributing

When adding new examples:

1. **Naming**: Use descriptive names starting with `example_`
2. **Documentation**: Include header comments explaining purpose and usage
3. **Dependencies**: Clearly state requirements
4. **Testing**: Ensure examples run with current AGAMA version
5. **Categories**: Update this README with new examples

## Performance Notes

**Quick Examples** (< 1 minute):
- `example_torus.py`
- `example_self_consistent_model_simple.py`
- `example_target.py`

**Medium Examples** (1-10 minutes):
- `example_self_consistent_model.py`
- `example_poincare.py`
- `example_df_fit.py`

**Long Examples** (> 10 minutes):
- `example_self_consistent_model_mw.py`
- `gc_runfit.py`
- `example_forstand.py`

Adjust parameters in long examples for faster testing during development. 