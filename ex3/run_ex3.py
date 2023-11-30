#!/usr/bin/python
from __future__ import print_function
from dolfin import *
import numpy as np
import math
import getopt, sys


# SS added
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True
parameters["ghost_mode"] = "shared_facet"
set_log_active(False)
set_log_level(LogLevel.ERROR) 

tol=1E-16

# Define parameters
gamma0 = 20.0   # penalty parameter
gamma1 = 1.0    # penalty parameter
k = 2           # degree of polynomials

alpha = Constant(0.449)     # fractional order
rho   = Constant(0.920e-3)    # density, scaled by 1e3 
lame1 = Constant(0.456)     # first Lame's parameter
lame2 = Constant(0.228)     # second Lame's parameter

varphi0 = Constant(0.685)   # power law coefficient
varphi1 = Constant(1.37)    # power law coefficient

# problem data
Nx = 60 
Ny = 30  
Nt = 100
dt = 0.001      # time step
T  = Nt*dt     # total simulation time

c_frac=math.gamma(3-alpha) # Gamma(3-alpha) for qn
varphi_a=varphi1*math.gamma(1-alpha) # varphi_alpha

mesh = RectangleMesh(Point(0.0, 0.0), Point(2.0, 1.0), Nx, Ny,"left")

# define coefficients B_{n,i} on the quadrature rule
def Qw(n):
    Bn=[]
    if n==0:
        Bn.append(0.0)
    else:
        Bn.append(n**(1-alpha)*(2-alpha-n)+(n-1.0)**(2-alpha))
        for i in range (1,n):
            Bn.append((n-i-1.0)**(2-alpha)+(n-i+1.0)**(2-alpha)-2.0*(n-i)**(2-alpha))
        Bn.append(1.0)
    return Bn

#-----------------------------------------------------------------------------------------------------------------

ux = Expression(("0.0","0.0"), tn=0, degree=5)

# define traction g
amp = 1.0 # amplitude of traction g
g = Expression(("amp*(tn<dt)","0.0"),amp=amp,dt=dt, degree=5, tn=0) 

def strain(v):
    Dv=grad(v)
    return 0.5*(Dv+Dv.T)

# Stress tensor (elasticity part)
def sigma(v):
    return  lame1*tr(strain(v))*Identity(2) +2*lame2*strain(v)
    
V = VectorFunctionSpace(mesh, 'DG', k)

# Define the boundary partition
domains = MeshFunction("size_t", mesh, mesh.topology().dim())
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundary_parts.set_all(0)

# Mark subdomain 0 for \Gamma_N etc
# non-homogeneous GammaNeumann Neumann BC(left edge)
class GammaNeumann(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],0.0,tol)
Gamma_Neumann = GammaNeumann()
Gamma_Neumann.mark(boundary_parts, 1)

# homogeneous Dirichlet BC (right edge) 
class GammaDirichlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],2.0,tol)
Gamma_Dirichlet = GammaDirichlet()
Gamma_Dirichlet.mark(boundary_parts, 2)

dx = Measure('dx', domain=mesh, subdomain_data=domains)
ds = Measure("ds", domain=mesh, subdomain_data=boundary_parts)

# Define normal vector and mesh size
n = FacetNormal(mesh)
h = FacetArea(mesh)
h_avg = (h('+') + h('-'))/2

u, v = TrialFunction(V), TestFunction(V)

# bilinear form for the solver  
uh = Function(V,name="Displacement")    
wh = Function(V,name="Velocity")
oldu=Function(V)
oldw=Function(V)


# compute Q(wh)+Q(oldw)-wh for r.h.s.
numDof=len(wh.vector().get_local())   #the number of degree of freedoms

def Quad(oldB,newB,n):
    Sq=(newB[0]+oldB[0])*np.array(W[0:numDof])
    for i in range(1,n+1):
        Sq=np.add(Sq,(newB[i]+oldB[i])*np.array(W[numDof*(i):numDof*(i+1)]))
    return Sq

# To evaluate energy quantities

# kinetic energy
def kineticEnergy(v1,v2):
    return 0.5*rho*inner(v1,v2)*dx

# elastic energy
def elasticEnergy(v1,v2):
    return 0.5*varphi0*inner(sigma(v1),strain(v2))*dx

# to define linear systems
mass = rho*inner(u,v)*dx
stiffness = inner(sigma(u),strain(v))*dx- inner(avg(strain(u)), outer(v('+'),n('+'))+outer(v('-'),n('-')))*dS \
            - inner(avg(strain(v)), outer(u('+'),n('+'))+outer(u('-'),n('-')))*dS \
            + gamma0/(h_avg**gamma1)*dot(jump(u), jump(v))*dS \
            - inner(strain(u), outer(v,n))*ds(2) \
            - inner(outer(u,n), strain(v))*ds(2) \
            + gamma0/(h**gamma1)*dot(u,v)*ds(2)
jump_penalty =  gamma0/(h_avg**gamma1)*dot(jump(u), jump(v))*dS  + gamma0/(h**gamma1)*dot(u,v)*ds(2)  


# assemble the system matrix once and for all
M = assemble(mass)
A = assemble(stiffness)
J = assemble(jump_penalty)


oldu = project(ux,V)    # zero initial displacement
oldw = project(ux,V)    # zero initial velocity

W=[]
W.extend(oldw.vector().get_local())

P = (1.0/dt)*M+varphi0*dt/4.0*A+varphi_a*0.5*dt**(1-alpha)/c_frac*A+(1.0/dt)*J

# assemble only once, before the time stepping
b = None 
b2= None
fileResult_u = XDMFFile("visco_vecfield_modi/output_displacement.xdmf")
fileResult_w = XDMFFile("visco_vecfield_modi/output_velocity.xdmf")
fileResult_u.parameters["flush_output"] = True
fileResult_u.parameters["functions_share_mesh"] = True
fileResult_w.parameters["flush_output"] = True
fileResult_w.parameters["functions_share_mesh"] = True

oldB = Qw(0)

for nt in range(0,Nt):
    # update data and solve for tn+k
    g.tn=(nt+0.5)*dt
    newB = Qw(nt+1)    

    # assemble the right hand side
    L=dot(g,v)*ds(1)
    
    b = assemble(L, tensor=b)
    b2=(1.0/dt*M-varphi0*dt/4.0*A+1.0/dt*J)*oldw.vector().get_local()\
        -varphi0*A*oldu.vector().get_local()\
        -varphi_a*0.5*dt**(1-alpha)/c_frac*A*Quad(oldB,newB,nt)
    b.add_local(b2)

    # solve the linear system to get new velocity and new displacement
    solve(P, wh.vector(), b, 'lu')   
    uh.vector()[:]=oldu.vector().get_local()\
                    +dt/2.0*(wh.vector().get_local()+oldw.vector().get_local())
    
    # update old terms
    oldw.assign(wh);oldu.assign(uh);W.extend(wh.vector().get_local())
    oldB = newB

    fileResult_u.write(uh, tn)
    fileResult_w.write(wh, tn)

