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

# SS added
iMin=2; iMax = 7
jMin=0; jMax = 2

# Define parameters
gamma0 = 10.0
gamma1 = 1.0
k=1
c_frac=3.0*sqrt(pi)/4.0 # Gamma(5/2)


def usage():
  print("-h   or --help")
  print("-g g or --gamma g       to specify gamma_0")
  print("-G G or --Gamma G       to specify gamma_1")
  print("-k       to specify k")
  print("-i i or --iMin  i       to specify iMin")
  print("-j j or --jMin  j       to specify jMin")
  print("-I i or --iMax  i       to specify iMax")
  print("-J j or --jMax  j       to specify jMax")
  print(" ")
  os.system('date +%Y_%m_%d_%H-%M-%S')
  print (time.strftime("%d/%m/%Y at %H:%M:%S"))

# parse the command line
try:
  opts, args = getopt.getopt(sys.argv[1:], "hg:G:k:i:I:j:J:",
                   [
                    "help",           # obvious
                    "gamma0=",        # gamma0
                    "gamma1=",        # gamma1
                    "k=",             # degree of polynomials
                    "iMin=",          # iMin
                    "iMax=",          # iMax
                    "jMin=",          # jMin
                    "jMax=",          # jMax
                    ])

except getopt.GetoptError as err:
  # print help information and exit:
  print(err) # will print something like "option -a not recognized"
  usage()
  sys.exit(2)

for o, a in opts:
  if o in ("-h", "--help"):
    usage()
    sys.exit()
  elif o in ("-g", "--gamma"):
    gamma0 = float(a)
    print('setting:  gamma0 = %f;' % gamma0),
  elif o in ("-G", "--Gamma"):
    gamma1 = float(a)
    print('setting:  gamma1 = %f;' % gamma1),    
  elif o in ("-k"):
    k = int(a)
    print('setting:  k = %d;' % k),
  elif o in ("-i", "--iMin"):
    iMin = int(a)
    print('setting:  iMin = %f;' % iMin),
  elif o in ("-I", "--iMax"):
    iMax = int(a)
    print('setting:  iMax = %f;' % iMax),
  elif o in ("-j", "--jMin"):
    jMin = int(a)
    print('setting:  jMin = %f;' % jMin),
  elif o in ("-J", "--jMax"):
    jMax = int(a)
    print('setting:  jMax = %f;' % jMax),
  else:
    assert False, "unhandled option"



# save data for error
L2w_error=np.zeros((iMax-iMin+1,jMax-jMin+1), dtype=np.float64)
L2u_error=np.zeros((iMax-iMin+1,jMax-jMin+1), dtype=np.float64)
H1u_error=np.zeros((iMax-iMin+1,jMax-jMin+1), dtype=np.float64)
H1w_error=np.zeros((iMax-iMin+1,jMax-jMin+1), dtype=np.float64)

# problem data
T = 1.0     # total simulation time


# define coefficients B_{n,i} on the quadrature rule
def Qw(n):
    Bn=[]
    if n==0:
        Bn.append(0.0)
    else:
        Bn.append(n**0.5*(1.5-n)+(n-1.0)**1.5)
        for i in range (1,n):
            Bn.append((n-i-1.0)**1.5+(n-i+1.0)**1.5-2.0*(n-i)**1.5)
        Bn.append(1.0)
    return Bn

#-----------------------------------------------------------------------------------------------------------------

ux=Expression(("utn*sin(pi*x[0])*sin(pi*x[1])","utn*x[0]*x[1]*(1.0-x[0])*(1.0-x[1])"),utn=0.0,degree=5)
wx=Expression(("wtn*sin(pi*x[0])*sin(pi*x[1])","wtn*x[0]*x[1]*(1.0-x[0])*(1.0-x[1])"),wtn=0.0,degree=5)


# define time function values of u w.r.t. time derivative
def uTime(tn):
    # return (tn**2.0 + tn**2.6)
    return (0.5*tn**2.0+0.4*tn**2.5)

def wTime(tn):
    return (tn+tn**1.5)

def fracwTime(tn):
    return (1.0/c_frac*tn**1.5 + math.gamma(2.5)/math.gamma(3)*tn**2)

def dwTime(tn):
    return (1.0+1.5*tn**0.5)

f=Expression(("sin(pi*x[0])*sin(pi*x[1])*dwtn\
                +(utn+fwtn)*(1.5*pi*pi*sin(pi*x[0])*sin(pi*x[1])+0.5*(2.0*x[0]-1.0)*(1.0-2.0*x[1]))",\
              "x[0]*x[1]*(1.0-x[0])*(1.0-x[1])*dwtn\
                +(utn+fwtn)*(-0.5*pi*pi*cos(pi*x[0])*cos(pi*x[1])+2.0*x[0]*(1-x[0])+x[1]*(1.0-x[1]))"),\
              dwtn=2.0,utn=0,fwtn=0,degree=5)
g=Expression(("fwtn*pi*sin(pi*x[1])", "0.5*fwtn*x[1]*(x[1]-1)"),fwtn=0,degree=5)
normal1=Constant((-1.0, 0.))

# Cauchy-infinitestimal strain tensor
def strain(v):
    Dv=grad(v)
    return 0.5*(Dv+Dv.T)
    
#===================================================================================================================
tol=1E-15
for i in range(iMin,iMax):
    Nxy=pow(2,i)
    mesh = UnitSquareMesh(Nxy, Nxy)
    V = VectorFunctionSpace(mesh, 'DG', k)

    # Define the boundary partition
    domains = MeshFunction("size_t", mesh, mesh.topology().dim())
    boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundary_parts.set_all(0)

    # Mark subdomain 0 for \Gamma_N etc
    # GammaNeumann Neumann BC(left edge)
    class GammaNeumann(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],0.0,tol)
    Gamma_Neumann = GammaNeumann()
    Gamma_Neumann.mark(boundary_parts, 1)


    class GammaDirichlet(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (near((1.0-x[0]),0.0,tol) \
                    or near((1.0-x[1]),0.0,tol)\
                    or near(x[1],0.0,tol))

    Gamma_Dirichlet = GammaDirichlet()
    Gamma_Dirichlet.mark(boundary_parts, 2)
    
    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure("ds", domain=mesh, subdomain_data=boundary_parts)

    # Define normal vector and mesh size
    n = FacetNormal(mesh)
    h = FacetArea(mesh)
    h_avg = (h('+') + h('-'))/2
    
    # Initial condition
    ux.utn=0.0; wx.wtn=0.0;

    # bilinear form for the solver
    uh = Function(V)    
    wh = Function(V)
    oldu=Function(V)
    oldw=Function(V)
        
    # compute Q(wh)+Q(oldw)-wh for r.h.s.
    numDof=len(wh.vector().get_local())   #the number of degree of freedoms
  
    def Quad(n):
        Sq=(Qw(n+1)[0]+Qw(n)[0])*np.array(W[0:numDof])
        for i in range(1,n+1):
            Sq=np.add(Sq,(Qw(n+1)[i]+Qw(n)[i])*np.array(W[numDof*(i):numDof*(i+1)]))
        return Sq

    # approximate the exact solution to define surface traction
    U = VectorFunctionSpace(mesh, 'Lagrange', 5)
    ux.utn=1.0
    UX = interpolate(ux, U)

    # to define linear systems
    u, v = TrialFunction(V), TestFunction(V)
    mass = inner(u,v)*dx
    stiffness = inner(strain(u),strain(v))*dx- inner(avg(strain(u)), outer(v('+'),n('+'))+outer(v('-'),n('-')))*dS \
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

    for j in range(jMin,jMax):        
        print('  i = %d (to %d), Nxy = %d; j = %d (to %d)' % (i, iMax-1, Nxy, j, jMax-1))
        Nt=(2**(j))    # number of time steps
        dt = T/Nt      # time step        

        # Initializing
        ux.utn=0.0; wx.wtn=0.0;

        oldu = project(ux,V)    #initial displacement
        oldw = project(wx,V)    #initial velocity
        
        W=[]
        W.extend(oldw.vector().get_local()) 
    
        # assemble the global matrix
        P = (1.0/dt)*M+0.5*sqrt(dt)/c_frac*A+(1.0/dt)*J     

        # assemble only once, before the time stepping
        b = None 
        b2= None
        for nt in range(0,Nt):
            # update data and solve for tn+k
            tn=dt*(nt+1);th=dt*nt;            
            dwtn=0.5*(dwTime(tn)+dwTime(th));
            fwtn=0.5*(fracwTime(tn)+fracwTime(th));
            f.dwtn=dwtn; f.fwtn=fwtn; g.fwtn=fwtn
            
            # assemble the right hand side
            L=inner(f,v)*dx+fwtn*inner(strain(UX),outer(v,n))*ds(1)
            
            b = assemble(L, tensor=b)
            b2=(1.0/dt*M+1.0/dt*J)*oldw.vector().get_local()\
                -0.5*sqrt(dt)/c_frac*A*Quad(nt)
            b.add_local(b2)

            # solve the linear system to get new velocity and new displacement
            solve(P, wh.vector(), b, 'lu')    
            uh.vector()[:]=oldu.vector().get_local()\
                            +dt/2.0*(wh.vector().get_local()+oldw.vector().get_local())
            

            # update old terms
            oldw.assign(wh);oldu.assign(uh);W.extend(wh.vector().get_local())
            
        # compute error at last time step
        ux.utn = uTime(T); wx.wtn = wTime(T); 

        err1 = errornorm(wx,wh,'L2')        
        err2 = errornorm(ux,uh,'L2')
        err3 = errornorm(wx,wh,'H1')        
        err4 = errornorm(ux,uh,'H1')
        # err3 = sqrt(errornorm(wx,wh,'H10')**2+errornorm(wx,wh,'L2')**2)
        # err4 = sqrt(errornorm(ux,uh,'H10')**2+errornorm(ux,uh,'L2')**2)
        
        L2w_error[0,j-jMin+1]=Nt; L2w_error[i-iMin+1,0]=Nxy; L2w_error[i-iMin+1,j-jMin+1]=err1;
        L2u_error[0,j-jMin+1]=Nt; L2u_error[i-iMin+1,0]=Nxy; L2u_error[i-iMin+1,j-jMin+1]=err2;
        H1w_error[0,j-jMin+1]=Nt; H1w_error[i-iMin+1,0]=Nxy; H1w_error[i-iMin+1,j-jMin+1]=err3;
        H1u_error[0,j-jMin+1]=Nt; H1u_error[i-iMin+1,0]=Nxy; H1u_error[i-iMin+1,j-jMin+1]=err4;
        
        

# SS altered the following loop limits
#print(V_error)
print ('L2_error for w')
print('\\begin{tabular}{|c|',end="")
for j in range(jMin,jMax): print('c',end="")
print('|}\\hline')
for j in range(jMin,jMax):
  if j==jMin: print('    ', end="")
  print(' & %10d ' % L2w_error[0,j-jMin+1], end="")
print('  \\\\\\hline')
for i in range(iMin,iMax):
  print('%4d ' % L2w_error[i-iMin+1,0], end="")
  for j in range(jMin,jMax):
      print('& %11.4le ' % L2w_error[i-iMin+1,j-jMin+1], end="")
  print(' \\\\')
print('\\hline\\end{tabular}')

print ('L2_error for u')
print('\\begin{tabular}{|c|',end="")
for j in range(jMin,jMax): print('c',end="")
print('|}\\hline')
for j in range(jMin,jMax):
  if j==jMin: print('    ', end="")
  print(' & %10d ' % L2u_error[0,j-jMin+1], end="")
print('  \\\\\\hline')
for i in range(iMin,iMax):
  print('%4d ' % L2u_error[i-iMin+1,0], end="")
  for j in range(jMin,jMax):
      print('& %11.4le ' % L2u_error[i-iMin+1,j-jMin+1], end="")
  print(' \\\\')
print('\\hline\\end{tabular}')

print ('H1_error for u')
print('\\begin{tabular}{|c|',end="")
for j in range(jMin,jMax): print('c',end="")
print('|}\\hline')
for j in range(jMin,jMax):
  if j==jMin: print('    ', end="")
  print(' & %10d ' % H1u_error[0,j-jMin+1], end="")
print('  \\\\\\hline')
for i in range(iMin,iMax):
  print('%4d ' % H1u_error[i-iMin+1,0], end="")
  for j in range(jMin,jMax):
      print('& %11.4le ' % H1u_error[i-iMin+1,j-jMin+1], end="")
  print(' \\\\')
print('\\hline\\end{tabular}')


print ('H1_error for w')
print('\\begin{tabular}{|c|',end="")
for j in range(jMin,jMax): print('c',end="")
print('|}\\hline')
for j in range(jMin,jMax):
  if j==jMin: print('    ', end="")
  print(' & %10d ' % H1w_error[0,j-jMin+1], end="")
print('  \\\\\\hline')
for i in range(iMin,iMax):
  print('%4d ' % H1w_error[i-iMin+1,0], end="")
  for j in range(jMin,jMax):
      print('& %11.4le ' % H1w_error[i-iMin+1,j-jMin+1], end="")
  print(' \\\\')
print('\\hline\\end{tabular}')

l2wDiag=np.diag(L2w_error)[1:]
l2uDiag=np.diag(L2u_error)[1:]
h1wDiag=np.diag(H1w_error)[1:]
h1uDiag=np.diag(H1u_error)[1:]

m = len(l2wDiag) 
    
if m>2:    
    v1=np.array(h1wDiag)
    t1=np.log(v1[0:m-2]/v1[1:m-1])
    d1=np.mean(t1/np.log(2))

    v2=np.array(h1uDiag)
    t2=np.log(v2[0:m-2]/v2[1:m-1])
    d2=np.mean(t2/np.log(2))

    v3=np.array(l2wDiag)
    t3=np.log(v3[0:m-2]/v3[1:m-1])
    d3=np.mean(t3/np.log(2))
    
    v4=np.array(l2uDiag)
    t4=np.log(v4[0:m-2]/v4[1:m-1])
    d4=np.mean(t4/np.log(2))

    print(t1/np.log(2),t2/np.log(2),t3/np.log(2),t4/np.log(2))
    print('Numeical convergent order when h=dt: H1 error of w = %5.4f,\
    H1 error of u= %5.4f, L2 error of w = %5.4f, L2 error of u = %5.4f'\
    %(d1,d2,d3,d4))   
    if k==1:        
        np.savetxt("L2_error_linear_velo_ex1.txt",l2wDiag,fmt="%2.3e")
        np.savetxt("L2_error_linear_disp_ex1.txt",l2uDiag,fmt="%2.3e")
        np.savetxt("H1_error_linear_velo_ex1.txt",h1wDiag,fmt="%2.3e")
        np.savetxt("H1_error_linear_disp_ex1.txt",h1uDiag,fmt="%2.3e")
        
    elif k==2:
        np.savetxt("L2_error_quad_velo_ex1.txt",l2wDiag,fmt="%2.3e")
        np.savetxt("L2_error_quad_disp_ex1.txt",l2uDiag,fmt="%2.3e")
        np.savetxt("H1_error_quad_velo_ex1.txt",h1wDiag,fmt="%2.3e")
        np.savetxt("H1_error_quad_disp_ex1.txt",h1uDiag,fmt="%2.3e") 

if jMax-jMin==1:    
    v1=L2w_error[1:,1]
    t1=np.log(v1[0:-1]/v1[1:])
    d1=np.mean(t1/np.log(2))
    
    v2=L2u_error[1:,1]
    t2=np.log(v2[0:-1]/v2[1:])
    d2=np.mean(t2/np.log(2))

    v3=H1w_error[1:,1]
    t3=np.log(v3[0:-1]/v3[1:])
    d3=np.mean(t3/np.log(2))
    
    v4=H1u_error[1:,1]
    t4=np.log(v4[0:-1]/v4[1:])
    d4=np.mean(t4/np.log(2))

    
    print('L2 of w orders')
    print(t1/np.log(2))
    print('L2 of u orders')
    print(t2/np.log(2))
    print('H1 of w orders')
    print(t3/np.log(2))
    print('H1 of u orders')
    print(t4/np.log(2))
    print('Numeical convergent order for fixed dt:') 
    print('L2 error of w = %5.4f' % d1) 
    print('L2 error of u = %5.4f' % d2)
    print('H1 error of w = %5.4f' % d3)
    print('H1 error of u = %5.4f' % d4)
    
if iMax-iMin==1:
    
    v1=L2w_error[1,1:]
    t1=np.log(v1[0:-1]/v1[1:])
    d1=np.mean(t1/np.log(2))
    
    v2=L2u_error[1,1:]
    t2=np.log(v2[0:-1]/v2[1:])
    d2=np.mean(t2/np.log(2))

    v3=H1w_error[1,1:]
    t3=np.log(v3[0:-1]/v3[1:])
    d3=np.mean(t3/np.log(2))
    
    v4=H1u_error[1,1:]
    t4=np.log(v4[0:-1]/v4[1:])
    d4=np.mean(t4/np.log(2))

    
    print('L2 of w orders')
    print(t1/np.log(2))
    print('L2 of u orders')
    print(t2/np.log(2))
    print('H1 of w orders')
    print(t3/np.log(2))
    print('H1 of u orders')
    print(t4/np.log(2))
    
    print('Numeical convergent order for fixed h:') 
    print('L2 error of w = %5.4f' % d1) 
    print('L2 error of u = %5.4f' % d2)
    print('H1 error of w = %5.4f' % d3) 
    print('H1 error of u = %5.4f' % d4)