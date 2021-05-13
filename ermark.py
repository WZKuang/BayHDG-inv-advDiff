from ngsolve import *
from netgen.geom2d import SplineGeometry
import numpy as np
import math
import time as timeit

##### Steady state solver
# Solve steady state advection-diffusion eqn
# h0-mesh size, # order-poly order, # wx,wy-wind velocity, # xs,ys-source location
# eps-diffusion constant, # Q-source rate
def ermarkHDGSolver(mesh=None, order = 2, wx=1, wy=-0.0027, xs=0, ys=1, eps=0.2, Q=1, v_depost=0.005,  vis=False):
    if mesh==None:
        print("mesh is needed")
        return
    if vis:
        #import netgen.gui
        from ngsolve.webgui import Draw
    # local space V, global space M for velocity
    V = L2(mesh, order=order)
    # homogeneous DBC on left
    # reasonable considering advection direction
    M = FacetFESpace(mesh, order=order, dirichlet="left|top")
    # compound finite element space
    fes = FESpace([V,M])
    gfu = GridFunction(fes)  # solution

    (u,uhat), (v,vhat) = fes.TnT()  # symbolic object, test and trial ProxyFunctions

    # linear and bilinear forms to be specified
    a = BilinearForm(fes, condense=True)
    f1 = LinearForm(fes)
    # Dirac Delta function approximated
    # by multivariable Gaussian
    f1 += v(xs,ys)


    ######## diffusion operator
    h = specialcf.mesh_size        # the local mesh size coefficient function
    n = specialcf.normal(mesh.dim) # the unit normal direction on edges
    alpha = 4 #stabilization coef

    jmp_u = (u-uhat)
    jmp_v = (v-vhat)

    # volume integration
    a += eps*grad(u)*grad(v)*dx
    # ATTENTION: element-boundary integration
    a += eps*(-grad(u)*n*jmp_v-grad(v)*n*jmp_u+alpha*order**2/h*jmp_u*jmp_v)*dx(element_boundary=True)
    # TODO:: ROBINE BC for bottom.(complete)
    # set settling speed, sign different from taht of wind in y
    v_set = -wy
    a += (v_depost-v_set)*uhat.Trace()*vhat.Trace()*dx(element_boundary=True, definedon=mesh.Boundaries("bottom"))

    ######## convection operator
    # velocity field
    w = CoefficientFunction((wx,wy))

    uhatup = IfPos(w*n, u, uhat) # upwinding flux
    conv = -u*w*grad(v)
    # element-boundary term
    # ATTENTION:: a simple HACK to deactivate vhat on Neumann bdry
    neuflag = GridFunction(FacetFESpace(mesh))
    neuflag.Set(1, definedon=mesh.Boundaries("right|bottom"))
    jmp_v0 = v-vhat*(1-neuflag)
    conv_BND =  uhatup*w*n*jmp_v0

    # convection integration
    a += conv*dx + conv_BND*dx(element_boundary=True)

    with TaskManager():
        t0=timeit.time()
        a.Assemble()
        f1.Assemble()
        rhs = gfu.vec.CreateVector()
        # source term is Q*diracDelta
        rhs.data = Q*f1.vec
        #print("ElAPSED %.2e ASSEMBLE" %(timeit.time()-t0))

        # add Dirichlet BC -- no need here
        #f.vec.data -= a.mat * gfu.vec
        t0=timeit.time()
        rhs.data += a.harmonic_extension_trans * rhs
        inv = a.mat.Inverse(fes.FreeDofs(True), "umfpack")
        gfu.vec.data += inv * rhs
        gfu.vec.data += a.harmonic_extension * gfu.vec
        gfu.vec.data += a.inner_solve * rhs
        #print("ElAPSED %.2e SOLVE" %(timeit.time()-t0))
        #print("Ermark HDG Done. Source rate %.2e, source location: (%.1e,%.1e)\r"%(Q,xs,ys))
        if vis:
            Draw(gfu.components[0], mesh, "uh", min=0, max=1)#, deformation=True)

    return gfu
