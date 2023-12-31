# DGFEM for dynamic viscoelasticity of a power-law model

We solve a dynamic linear viscoelasticity problem of a power-law model.
The governing equation is a vector-valued hyperbolic-type evolution equation involving fractional order time derivative/integral.
In this numerical simulation, we employ the *symmetric interior penalty Galerkin method* (SIPG) and the *Crank-Nicolson scheme* for spatial and temporal discretizations, respectively.
Furthermore, we utilize the *linear interpolation technique* to manage weak singularity in fractional integration.
For more details on the theoretical analysis and numerical experiments, please refer to our manuscript.

Codes for numerical simulations have been implemented with [FEniCS](https://fenicsproject.org) version 2019.1.0 in Python scripts (e.g. Python 3.11.4)

If you have any further questions or need additional information, please do not hesitate to contact us at <yongseok.jang@onera.fr> or <yongseok20007717@gmail.com>.  Thank you for your interest in our research and for considering our work.
