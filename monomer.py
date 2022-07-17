#!/usr/bin/env python3

#__________________________________________________________________
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from pylab import interactive
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import fmin as simplex
from numpy import linalg as LA
import math
import random

#__________________________________________________________________
 
''' INTRODUCTION '''
##################################################################
'''
INTRODUCTION: This script was part of the Ph.D. project where we
constructed MOFs consisting of Metallic (Fe, Ru, Cu) ions coordinated 
with bipyridine molecules. This technical construction consists of 
first creating the main unit fully optimized with DFT in the gas phase 
and getting the corresponding NBO or Mulliken charges. The next phase 
is to repeat this unit in all three directions with the same corresponding 
charges of the main unit. Overall, this is the assumption where we have
assumed that the system is homogenous, the electrostatic field is the same 
everywhere. Now, we have to fix the boundary points. This, however, is a 
difficult problem because the truncated charges have to be taken into account. 
This is explained in the thesis. 
@
https://hal.archives-ouvertes.fr/tel-02058650
@

'''
##################################################################
''' MAIN PROGRAM '''
##################################################################
def read_atom_locations(path,file,n0,n1,n2,n3,n4,n5,ax,f1):
    x,y,z = [], [], []
    xi,yi,zi = [], [], []
    label = []
    with open(path+file,"r") as f:
        data = f.readlines()
        for line in data:
            words=line.split()
            label.append(words[0])
            xi.append(words[1])
            yi.append(words[2])
            zi.append(words[3])
    xmi,ymi,zmi = [], [], []
    xm,ym,zm = [], [], []
    labelm = []
    with open(f"{path}methyl.xyz", "r") as f:
        data = f.readlines()
        for line in data:
            words=line.split()
            labelm.append(words[0])
            xmi.append(words[1])
            ymi.append(words[2])
            zmi.append(words[3])
    S0 = define_central_octa(xi,yi,zi,n0,n1,n2,n3,n4,n5)        

    #Rotate into conveniant orientation : (at_3-at_4 axis) = (001)
    ux =  float(S0[3][0]) - float(S0[4][0])
    uy =  float(S0[3][1]) - float(S0[4][1])
    uz =  float(S0[3][2]) - float(S0[4][2])
    norm = np.sqrt(ux*ux + uy *uy +uz*uz)
    ux=ux/norm
    uy=uy/norm
    uz=uz/norm

    G = barycenter(S0)

    xinit = [0,0,0]
    opt = simplex(func3, xinit, args=(0, 0, 1, ux, uy, uz), full_output=0)
    # Rotate & move the atoms of the residu without any terminating CH3 fragments
    for i in range(len(xi)):
        vect1=[[float(xi[i])-float(G[0])],[float(yi[i])-float(G[1])],[float(zi[i])-float(G[2])]]
        vect2 = move3(opt[0],opt[1],opt[2],vect1)
        x.append( vect2[0])
        y.append( vect2[1])
        z.append( vect2[2]) 


    for i in range(54):
        ax.scatter(float(x[i]),float(y[i]),float(z[i]), c='blue', s=30)

    with open("./"+"monomer_0.xyz","w") as f:
        for i in range(len(x)):
            line = f"{label[i]} {float(x[i])} {float(y[i])} {float(z[i])}" + "\n"
            f1.write(line)
            f.write(line)
    # Operate the same displacement for the terminating CH3 fragments
    for i in range(len(xmi)):
        vect1=[[float(xmi[i])-float(G[0])],[float(ymi[i])-float(G[1])],[float(zmi[i])-float(G[2])]]
        vect2 = move3(opt[0],opt[1],opt[2],vect1)
        xm.append( vect2[0])
        ym.append( vect2[1])
        zm.append( vect2[2]) 

    # Do the same for the central octahedra    
    S=[]
    S = define_central_octa(x,y,z,n0,n1,n2,n3,n4,n5)  

    return S, label,x,y,z,labelm,xm,ym,zm

def read_atom_locations2(path,file):
    x,y,z = [], [], []
    label = []
    with open(path+file,"r") as f:
        data = f.readlines()
        for line in data:
            words=line.split()
            label.append(words[0])
            x.append(words[1])
            y.append(words[2])
            z.append(words[3])
    return label,x,y,z


def define_central_octa(x,y,z,n0,n1,n2,n3,n4,n5):    
    A = [float(x[n0])  ,  float(y[n0])   ,   float(z[n0])]
    B = [float(x[n1])  ,  float(y[n1])   ,   float(z[n1])]
    C = [float(x[n2])  ,  float(y[n2])   ,   float(z[n2])]
    D = [float(x[n3])  ,  float(y[n3])   ,   float(z[n3])]
    E = [float(x[n4])  ,  float(y[n4])   ,   float(z[n4])]
    F = [float(x[n5])  ,  float(y[n5])   ,   float(z[n5])]
#    G = barycenter(S)
#    for i in range(6):
#        for j in range(3):
#            S[i][j] = S[i][j]-G[j]
    return [B, A, C, D, E, F]



def change_name(S0,S1_1,S1_2,S1_3,S1_4,S1_5,S1_6,S2_2, \
                S2_3,S2_5,S2_6,S3_2,S3_3,S3_5,S3_6):
    return [
        S0,
        S1_1,
        S1_2,
        S1_3,
        S1_4,
        S1_5,
        S1_6,
        S2_2,
        S2_3,
        S2_5,
        S2_6,
        S3_2,
        S3_3,
        S3_5,
        S3_6,
    ]

def change_name2(S0,S1_2,S1_3,S1_5,S1_6):
    return [S0, S1_2, S1_3, S1_5, S1_6]


def new_monomer(S,n1,n2,R,ax):    
    # PLOT A NEW ONE
    # initializes S2
    S2 = []
    for j in range(len(S)):
        S2.append([])
        for _ in range(3):
            S2[j].append(0.0)
    #print(S2)

    dx = (S[n1][0]-S[n2][0])
    dy = (S[n1][1]-S[n2][1])
    dz = (S[n1][2]-S[n2][2])
    norm = np.sqrt(dx*dx + dy*dy + dz*dz)
    dx = dx/norm
    dy = dy/norm
    dz = dz/norm

    ux =   (S[n1][0]-S[n2][0])  + dx*R
    uy =   (S[n1][1]-S[n2][1])  + dy*R
    uz =   (S[n1][2]-S[n2][2]) +  dz*R    

    for j in range(len(S)):
        S2[j][0] =S[j][0] + ux
        S2[j][1] =S[j][1] + uy
        S2[j][2] =S[j][2] + uz
    #print(S2)    
    a = Arrow3D([S2[n2][0], S[n1][0]],[S2[n2][1], S[n1][1]], [S2[n2][2], S[n1][2]], mutation_scale=20,
            lw=3, arrowstyle="-", color="r")
    ax.add_artist(a)
    return S2


def new_monomer_top(S,n1,n2,R,ax):    
    # PLOT A NEW ONE
    # initializes S2
    S2 = []
    dz = (S[n2][2]-S[n1][2])
    for j in range(len(S)):
        S2.append([])
        for _ in range(3):
            S2[j].append(0.0)
    for j in range(len(S)):
        S2[j][0] =S[j][0] 
        S2[j][1] =S[j][1] 
        S2[j][2] =S[j][2] + R  + dz
    #print(S2)    
    #a = Arrow3D([S2[n2][0], S[n1][0]],[S2[n2][1], S[n1][1]], [S2[n2][2], S[n1][2]], mutation_scale=20,
    #        lw=3, arrowstyle="-", color="r")
    #ax.add_artist(a)    
    return S2


def new_monomer_bottom(S,n1,n2,R,ax):    
    # PLOT A NEW ONE
    # initializes S2
    S2 = []
    dz = (S[n2][2]-S[n1][2])
    for j in range(len(S)):
        S2.append([])
        for _ in range(3):
            S2[j].append(0.0)
    for j in range(len(S)):
        S2[j][0] =S[j][0] 
        S2[j][1] =S[j][1] 
        S2[j][2] =S[j][2] - R  + dz
    #print(S2)    
    #a = Arrow3D([S2[n2][0], S[n1][0]],[S2[n2][1], S[n1][1]], [S2[n2][2], S[n1][2]], mutation_scale=20,
    #       lw=3, arrowstyle="-", color="r")
    #ax.add_artist(a)    
    return S2


def create_first_shell(S,R,ax):
    S2 = new_monomer(S,0,2,R,ax)
    S3 = new_monomer(S,5,1,R,ax)
    S4 = new_monomer(S,1,5,R,ax)
    S5 = new_monomer(S,2,0,R,ax)
    S6 = new_monomer(S,3,4,R,ax)
    S7 = new_monomer(S,4,3,R,ax)
    return S2,S3,S4,S5,S6,S7

def create_on_top_layer(S0,S1,S2,S3,S4,R,ax):
    S0p = new_monomer_top(S0,3,4,R,ax)
    S1p = new_monomer_top(S1,3,4,R,ax)
    S2p = new_monomer_top(S2,3,4,R,ax)
    S3p = new_monomer_top(S3,3,4,R,ax)
    S4p = new_monomer_top(S4,3,4,R,ax)    
    return S0p,S1p,S2p,S3p,S4p

def create_bottom_layer(S0,S1,S2,S3,S4,R,ax):
    S0p = new_monomer_bottom(S0,4,3,R,ax)
    S1p = new_monomer_bottom(S1,4,3,R,ax)
    S2p = new_monomer_bottom(S2,4,3,R,ax)
    S3p = new_monomer_bottom(S3,4,3,R,ax)
    S4p = new_monomer_bottom(S4,4,3,R,ax)    
    return S0p,S1p,S2p,S3p,S4p


def rotation(alpha1,beta1,gamma1):
    alpha1=0.0
    beta1 = 0.0
    #gamma1 = 0.0
    R1x=np.matrix([[1,0,0],[0,np.cos(alpha1),-np.sin(alpha1)],[0,np.sin(alpha1),np.cos(alpha1)]])
    R1y=np.matrix([[np.cos(beta1),0,np.sin(beta1)],[0,1,0],[-np.sin(beta1),0, np.cos(beta1)]])
    R1z=np.matrix([[np.cos(gamma1),-np.sin(gamma1),0],[np.sin(gamma1),np.cos(gamma1),0],[0,0,1]])
    return R1x*R1y*R1z

def rotation2(alpha1,beta1,gamma1):
    R1x=np.matrix([[1,0,0],[0,np.cos(alpha1),-np.sin(alpha1)],[0,np.sin(alpha1),np.cos(alpha1)]])
    R1y=np.matrix([[np.cos(beta1),0,np.sin(beta1)],[0,1,0],[-np.sin(beta1),0, np.cos(beta1)]])
    R1z=np.matrix([[np.cos(gamma1),-np.sin(gamma1),0],[np.sin(gamma1),np.cos(gamma1),0],[0,0,1]])
    return R1x*R1y*R1z

def barycenter(S):
    x = float(S[0][0])+float(S[1][0])+float(S[2][0])+float(S[3][0])+float(S[4][0])+float(S[5][0])
    y = float(S[0][1])+float(S[1][1])+float(S[2][1])+float(S[3][1])+float(S[4][1])+float(S[5][1])
    z = float(S[0][2])+float(S[1][2])+float(S[2][2])+float(S[3][2])+float(S[4][2])+float(S[5][2])
    return np.array([[x/6],[y/6],[z/6]])


def move(Sub,n,m,alpha,beta,gamma):
    G = barycenter(Sub)
    Rot = rotation(alpha,beta,gamma)
    vect1 = np.array([[Sub[n][0]],[Sub[n][1]],[Sub[n][2]]])
    vect2 = np.array([[Sub[m][0]],[Sub[m][1]],[Sub[m][2]]])
    u0=Rot*(vect1-G)
    u=Rot*(vect2-G)
    t1 = vect1 -u0 -G
    u= u  + G + t1
    return u
    

def func(params, Sub1,n1,n2,Sub2,n3,n4,R):
    # extract current values of fit parameters from input array
    chi2 = 0.0
    alpha1= params[0]
    beta1 = params[1]
    gamma1 = params[2]
    alpha2 = params[3]
    beta2 = params[4]
    gamma2 = -params[2]     
    # compute chi-square
    u1 = move(Sub1,n1,n2,alpha1,beta1,gamma1)
    u2 = move(Sub2,n3,n4,alpha2,beta2,gamma2)    
    chi2 = (np.linalg.norm(u2-u1) - R)**2 
    #print(np.linalg.norm(u2-u1))
    return chi2

    
def get_rotation_angles(Sub1,n1,n2,Sub2,n3,n4,R):
    x0 = [0,0,0,0,0,0]
    return simplex(func, x0, args=(Sub1,n1,n2,Sub2,n3,n4,R), full_output=0)

def apply_new_locations(S0,opt,Sub1,n1,n2,Sub2,n3,n4, Natom, label0, x0, y0, z0,xm,ym,zm,ax,f1,icompt):
    for i in range(6):
        u = move(Sub1,n1,i,opt[0],opt[1],opt[2])
        v = move(Sub2,n3,i,opt[3],opt[4],-opt[2])
        if i != n1 :
            for j in range(3):
                Sub1[i][j] = float(u[j])
        if i != n3 :
            for j in range(3):
                Sub2[i][j] = float(v[j])

    u = move(Sub1,n1,n1,opt[0],opt[1],opt[2])
    v = move(Sub2,n3,n3,opt[3],opt[4],-opt[2])
    #print(Sub1[1])
    for j in range(3):
        Sub1[n1][j] = float(u[j])    
        Sub2[n3][j] = float(v[j])
    #print(Sub1[n1])

    #Move the atoms at the same time.
    label,x,y,z,xm,ym,zm = Move_Rotate_octahedra2(S0,Sub1, opt[0],opt[1],opt[2], Natom, label0, x0, y0, z0,xm,ym,zm)
    for i in range(54):
        ax.scatter(float(x[i]),float(y[i]),float(z[i]), c='blue', s=30)
    if icompt ==0 : f=open("./"+"monomer_3.xyz","w")
    if icompt ==1 : f=open("./"+"monomer_4.xyz","w")
    for i in range(len(x)):
        line = f"{label[i]} {float(x[i])} {float(y[i])} {float(z[i])}" + "\n"
        f1.write(line)
        f.write(line)
    f.close()

    label,x,y,z,xm,ym,zm = Move_Rotate_octahedra2(S0,Sub2, opt[3],opt[4],-opt[2], Natom, label0, x0, y0, z0,xm,ym,zm)
    for i in range(54):
        ax.scatter(float(x[i]),float(y[i]),float(z[i]), c='blue', s=30)
    if icompt==0 : f=open("./"+"monomer_1.xyz","w")
    if icompt==1 : f=open("./"+"monomer_2.xyz","w")
    for i in range(len(x)):
        line = f"{label[i]} {float(x[i])} {float(y[i])} {float(z[i])}" + "\n"
        f1.write(line)
        f.write(line)
    f.close()

    return

def Move_Rotate_octahedra2(S0,O, a1,a2,a3,Natom, label0, x0, y0, z0, xmi, ymi, zmi):
    x,y,z = [],[],[]
    xm,ym,zm = [],[],[]
    label = [] 
    G1 = barycenter(O)
    G0 = barycenter(S0)
    xc = float(G0[0])
    yc = float(G0[1])
    zc = float(G0[2])
    for k in range(Natom):
        vect1=[[float(x0[k])-xc],[float(y0[k])-yc],[float(z0[k])-zc]]
        vect2 = move2(a1,a2,a3,vect1)
        label.append(label0[k])
        x.append(float(G1[0]) + vect2[0])
        y.append(float(G1[1]) + vect2[1])
        z.append(float(G1[2]) + vect2[2])

    for k in range(len(xmi)):
        vect1=[[float(xmi[k])-xc],[float(ymi[k])-yc],[float(zmi[k])-zc]]
        vect2 = move2(a1,a2,a3,vect1)
        xm.append(float(G1[0]) + vect2[0])
        ym.append(float(G1[1]) + vect2[1])
        zm.append(float(G1[2]) + vect2[2])     

    return label,x,y,z, xm,ym,zm


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_monomer_top(alf,S,R,ax,i0):
    shift = 1
    x = S[0][0]+S[1][0]+S[2][0]+S[3][0]+S[4][0]+S[5][0]
    y = S[0][1]+S[1][1]+S[2][1]+S[3][1]+S[4][1]+S[5][1]
    z = S[0][2]+S[1][2]+S[2][2]+S[3][2]+S[4][2]+S[5][2]
    G =  [x/6,y/6,z/6]
    label = f"S2_{str(i0)}"
    #ax.text((x/6)+shift,(y/6)+shift,(z/6)+shift, label, size=23, zorder=1,  color='grey') 

    col =  ['red','green','blue','violet','brown','black']
    ax.scatter(G[0], G[1],G[2], c='grey', s=30)
    for i in range(len(S)) :
        ax.scatter(S[i][0],S[i][1],S[i][2], c=col[i], s=30)


    a = Arrow3D([S[0][0], S[1][0]],[S[0][1], S[1][1]], [S[0][2], S[1][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(a)
    b = Arrow3D([S[1][0], S[2][0]],[S[1][1], S[2][1]], [S[1][2], S[2][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(b)
    c = Arrow3D([S[0][0], S[3][0]],[S[0][1], S[3][1]], [S[0][2], S[3][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(c)
    d= Arrow3D([S[0][0], S[5][0]],[S[0][1], S[5][1]], [S[0][2], S[5][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(d)
    e= Arrow3D([S[0][0], S[4][0]],[S[0][1], S[4][1]], [S[0][2], S[4][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(e)
    f= Arrow3D([S[2][0], S[5][0]],[S[2][1], S[5][1]], [S[2][2], S[5][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(f)
    g= Arrow3D([S[2][0], S[4][0]],[S[2][1], S[4][1]], [S[2][2], S[4][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(g)
    h= Arrow3D([S[2][0], S[3][0]],[S[2][1], S[3][1]], [S[2][2], S[3][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(h)
    h1= Arrow3D([S[4][0], S[1][0]],[S[4][1], S[1][1]], [S[4][2], S[1][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(h1)
    h2= Arrow3D([S[3][0], S[1][0]],[S[3][1], S[1][1]], [S[3][2], S[1][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(h2)
    h3= Arrow3D([S[5][0], S[4][0]],[S[5][1], S[4][1]], [S[5][2], S[4][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(h3)
    h4= Arrow3D([S[5][0], S[3][0]],[S[5][1], S[3][1]], [S[5][2], S[3][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(h4)

    return
        
        
def plot_monomer(alf,S,R,ax,i0,i1):
    shift = 1
    x = S[0][0]+S[1][0]+S[2][0]+S[3][0]+S[4][0]+S[5][0]
    y = S[0][1]+S[1][1]+S[2][1]+S[3][1]+S[4][1]+S[5][1]
    z = S[0][2]+S[1][2]+S[2][2]+S[3][2]+S[4][2]+S[5][2]
    G =  [x/6,y/6,z/6]
    if i0 == 0 : label = "S0"
    if i0 !=0 : label = "S1_"+str(i0)
    if i1 ==1 :
        ax.text((x/6)+shift,(y/6)+shift,(z/6)+shift, label, size=20, zorder=1,  color='red') 
    
    col =  ['red','green','blue','violet','brown','black']
    ax.scatter(G[0], G[1],G[2], c='grey', s=30)
    for i in range(len(S)) :
        ax.scatter(S[i][0],S[i][1],S[i][2], c=col[i], s=30)
        if i1 == 1:
            ax.text(S[i][0],S[i][1],S[i][2], str(i), size=20, zorder=1,  color='black') 
    
    a = Arrow3D([S[0][0], S[1][0]],[S[0][1], S[1][1]], [S[0][2], S[1][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(a)
    b = Arrow3D([S[1][0], S[2][0]],[S[1][1], S[2][1]], [S[1][2], S[2][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(b)
    c = Arrow3D([S[0][0], S[3][0]],[S[0][1], S[3][1]], [S[0][2], S[3][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(c)
    d= Arrow3D([S[0][0], S[5][0]],[S[0][1], S[5][1]], [S[0][2], S[5][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(d)
    e= Arrow3D([S[0][0], S[4][0]],[S[0][1], S[4][1]], [S[0][2], S[4][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(e)
    f= Arrow3D([S[2][0], S[5][0]],[S[2][1], S[5][1]], [S[2][2], S[5][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(f)
    g= Arrow3D([S[2][0], S[4][0]],[S[2][1], S[4][1]], [S[2][2], S[4][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(g)
    h= Arrow3D([S[2][0], S[3][0]],[S[2][1], S[3][1]], [S[2][2], S[3][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(h)
    h1= Arrow3D([S[4][0], S[1][0]],[S[4][1], S[1][1]], [S[4][2], S[1][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(h1)
    h2= Arrow3D([S[3][0], S[1][0]],[S[3][1], S[1][1]], [S[3][2], S[1][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(h2)
    h3= Arrow3D([S[5][0], S[4][0]],[S[5][1], S[4][1]], [S[5][2], S[4][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(h3)
    h4= Arrow3D([S[5][0], S[3][0]],[S[5][1], S[3][1]], [S[5][2], S[3][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(h4)

    # draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    for i in range(len(S)):
        x = R*np.cos(u)*np.sin(v)+S[i][0]
        y = R*np.sin(u)*np.sin(v)+S[i][1]
        z = R*np.cos(v)+S[i][2]
        ax.plot_wireframe(x, y, z, color=col[i],alpha = alf)
    return G

def plot_monomer2(alf,S,R,ax,i0,i1,i2):
    shift = 1
    x = S[0][0]+S[1][0]+S[2][0]+S[3][0]+S[4][0]+S[5][0]
    y = S[0][1]+S[1][1]+S[2][1]+S[3][1]+S[4][1]+S[5][1]
    z = S[0][2]+S[1][2]+S[2][2]+S[3][2]+S[4][2]+S[5][2]
    G =  [x/6,y/6,z/6]
    if i0 == 0 : label = "S0"
    if i0 !=0 : label = "O_"+str(i2)
    if i1 ==1 :
        ax.text((x/6)+shift,(y/6)+shift,(z/6)+shift, label, size=20, zorder=1,  color='red') 
    
    col =  ['red','green','blue','violet','brown','black']
    ax.scatter(G[0], G[1],G[2], c='grey', s=30)
    for i in range(len(S)) :
        ax.scatter(S[i][0],S[i][1],S[i][2], c=col[i], s=30)
        if i1 == 1:
            ax.text(S[i][0],S[i][1],S[i][2], str(i), size=20, zorder=1,  color='black') 
    
    a = Arrow3D([S[0][0], S[1][0]],[S[0][1], S[1][1]], [S[0][2], S[1][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(a)
    b = Arrow3D([S[1][0], S[2][0]],[S[1][1], S[2][1]], [S[1][2], S[2][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(b)
    c = Arrow3D([S[0][0], S[3][0]],[S[0][1], S[3][1]], [S[0][2], S[3][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(c)
    d= Arrow3D([S[0][0], S[5][0]],[S[0][1], S[5][1]], [S[0][2], S[5][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(d)
    e= Arrow3D([S[0][0], S[4][0]],[S[0][1], S[4][1]], [S[0][2], S[4][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(e)
    f= Arrow3D([S[2][0], S[5][0]],[S[2][1], S[5][1]], [S[2][2], S[5][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(f)
    g= Arrow3D([S[2][0], S[4][0]],[S[2][1], S[4][1]], [S[2][2], S[4][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(g)
    h= Arrow3D([S[2][0], S[3][0]],[S[2][1], S[3][1]], [S[2][2], S[3][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(h)
    h1= Arrow3D([S[4][0], S[1][0]],[S[4][1], S[1][1]], [S[4][2], S[1][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(h1)
    h2= Arrow3D([S[3][0], S[1][0]],[S[3][1], S[1][1]], [S[3][2], S[1][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(h2)
    h3= Arrow3D([S[5][0], S[4][0]],[S[5][1], S[4][1]], [S[5][2], S[4][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(h3)
    h4= Arrow3D([S[5][0], S[3][0]],[S[5][1], S[3][1]], [S[5][2], S[3][2]], mutation_scale=20,
            lw=1, arrowstyle="-", color="r")
    ax.add_artist(h4)

    # draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    for i in range(len(S)):
        x = R*np.cos(u)*np.sin(v)+S[i][0]
        y = R*np.sin(u)*np.sin(v)+S[i][1]
        z = R*np.cos(v)+S[i][2]
        ax.plot_wireframe(x, y, z, color=col[i],alpha = alf)
    return G


def search_chains_to_be_ploted(N,O,R):
    # calculate all the distances. If one distance is 
    # close to R, spare the pair of atoms.
    error = 0.3
    P=[]
    connect = []
    for k1 in range(N):
        connect.append([])
        for i1 in range(6): 
            connect[k1].append(0)
    
    for k1 in range(N-1):
        for i1 in range(6): 
            for k2 in range(k1+1,N):
                for i2 in range(6):
                    dx2=(O[k1][i1][0]-O[k2][i2][0])**2
                    dy2=(O[k1][i1][1]-O[k2][i2][1])**2
                    dz2=(O[k1][i1][2]-O[k2][i2][2])**2
                    dist = np.sqrt(dx2+dy2+dz2)
                    if dist <= R+error :
                        if dist >= R-error :
                            P.append([k1,i1,k2,i2]) 
                            connect[k1][i1] = 1
                            connect[k2][i2] = 1
                            #print(dist)
    print(P)
    print(connect)
    return P, connect


def plot_chains(N,O,R,ax):
    # plot all the segments corresponding to the pairs of atoms
    # returned by the search routine.
    P, connect = search_chains_to_be_ploted(N,O,R)
    print("pairs of atoms are obtained. Now, have a plot of the segments ...")
    print("Found ",len(P)," segments")
    for k in range(len(P)) :
        a = Arrow3D([O[P[k][0]][P[k][1]][0],O[P[k][2]][P[k][3]][0]], \
                    [O[P[k][0]][P[k][1]][1],O[P[k][2]][P[k][3]][1]], \
                    [O[P[k][0]][P[k][1]][2],O[P[k][2]][P[k][3]][2]], \
                    mutation_scale=20,lw=2, arrowstyle="-", color="b")
        ax.add_artist(a)
    print("segments are about to be plotted ....")
    return P, connect


def move2(alpha,beta,gamma, vect):
    Rot = rotation(alpha,beta,gamma)
    u= Rot* vect
    return u

def move3(alpha,beta,gamma, vect):
    Rot = rotation2(alpha,beta,gamma)
    u= Rot* vect
    return u


def func2(params, S0, O,x0,y0,z0,n1,n2,n3,n1p, n2p, n3p):
    # extract current values of fit parameters from input array
    chi2 = 0.0
    alpha1= params[0]
    beta1 = params[1]
    gamma1 = params[2]
    
    G0 = barycenter(S0)
    xc= float(G0[0])
    yc = float(G0[1])
    zc = float(G0[2])  
    
    vect1=[[float(x0[n1p])-xc],[float(y0[n1p])-yc],[float(z0[n1p])-zc]]
    vect2=[[float(x0[n2p])-xc],[float(y0[n2p])-yc],[float(z0[n2p])-zc]]
    vect3=[[float(x0[n3p])-xc],[float(y0[n3p])-yc],[float(z0[n3p])-zc]]

    # compute chi-square
    u1 = move2(alpha1,beta1,gamma1,vect1)
    u2 = move2(alpha1,beta1,gamma1,vect2)   
    u3 = move2(alpha1,beta1,gamma1,vect3)    
    G2 = barycenter(O)
    
    chi2 = (u1[0]- O[n1][0] + float(G2[0]))**2 +  (u1[1]- O[n1][1]+ float(G2[1]))**2 + \
            (u1[2]- O[n1][2]+ float(G2[2]))**2 + \
           (u2[0]- O[n2][0] + float(G2[0]))**2 +  (u2[1]- O[n2][1]+ float(G2[1]))**2 + \
            (u2[2]- O[n2][2]+ float(G2[2]))**2 + \
           (u3[0]- O[n3][0] + float(G2[0]))**2 +  (u3[1]- O[n3][1]+ float(G2[1]))**2 + \
            (u3[2]- O[n3][2]+ float(G2[2]))**2   
    
    return chi2


def orientate_octahedra(S0, O, x0, y0, z0,n1, n2, n3, n1p, n2p, n3p):
    xinit = [0,0,0,0,0,0]
    opt = simplex(func2, xinit, args=(S0,O,x0,y0,z0,n1,n2,n3,n1p, n2p, n3p), full_output=0)    
    return opt


def Move_Rotate_octahedra(S0, O, Natom, label0, x0, y0, z0, xmi, ymi, zmi, n1, n2, n3, n1p, n2p, n3p):
    x,y,z = [],[],[]
    xm,ym,zm = [],[],[]
    label = [] 
    opt = orientate_octahedra(S0, O, x0, y0, z0, n1, n2, n3, n1p, n2p, n3p)
    G0 = barycenter(S0)
    G1 = barycenter(O)
    xc = float(G0[0])
    yc = float(G0[1])
    zc = float(G0[2])
    for k in range(Natom):
        vect1=[[float(x0[k])-xc],[float(y0[k])-yc],[float(z0[k])-zc]]
        vect2 = move2(opt[0],opt[1],opt[2],vect1)
        label.append(label0[k])
        x.append(float(G1[0]) + vect2[0])
        y.append(float(G1[1]) + vect2[1])
        z.append(float(G1[2]) + vect2[2])
        
    for k in range(len(xmi)):
        vect1=[[float(xmi[k])-xc],[float(ymi[k])-yc],[float(zmi[k])-zc]]
        vect2 = move2(opt[0],opt[1],opt[2],vect1)
        xm.append(float(G1[0]) + vect2[0])
        ym.append(float(G1[1]) + vect2[1])
        zm.append(float(G1[2]) + vect2[2])     
        
    return label,x,y,z, xm,ym,zm


def func3(params, xo, yo, zo, xab, yab, zab):
    # extract current values of fit parameters from input array
    chi2 = 0.0
    alpha1= params[0]
    beta1 = params[1]
    gamma1 = params[2]    
    vect1=[[float(xab)],[float(yab)],[float(zab)]]    
    # compute chi-square
    u1 = move3(alpha1,beta1,gamma1,vect1)   
    chi2 =  (np.abs(float(u1[0])* float(xo)+ float(u1[1])* float(yo)+ float(u1[2])* float(zo))-1.0)**2
    return chi2


def turn_alkyl(xo, yo, zo, xab, yab, zab):
    xinit = [0,0,0]
    opt = simplex(func3, xinit, args=(xo, yo, zo, xab, yab, zab), full_output=0)
    return opt

  
def Move_Rotate_segments(O, P, Nalk, labela, xa, ya, za):
    x,y,z = [],[],[]
    label = [] 
    # FIND THE MIDDLE OF alkyl chain
    xg=float(xa[5]) + (float(xa[0])-float(xa[5]))/2
    yg=float(ya[5]) + (float(ya[0])-float(ya[5]))/2
    zg=float(za[5]) + (float(za[0])-float(za[5]))/2      
    xab = float(xa[5]) -xg
    yab = float(ya[5]) -yg
    zab = float(za[5]) -zg
    print(xa[5],ya[5],za[5])
    print(xg,yg,zg)    
    print(xab,yab,zab)
    norm = np.sqrt(xab**2 + yab**2 + zab**2)
    xab = xab /norm
    yab = yab/norm
    zab = zab/norm
    for k in range(len(P)):
        xo=(O[P[k][0]][P[k][1]][0]-O[P[k][2]][P[k][3]][0])
        yo=(O[P[k][0]][P[k][1]][1]-O[P[k][2]][P[k][3]][1])
        zo=(O[P[k][0]][P[k][1]][2]-O[P[k][2]][P[k][3]][2])
        norm = np.sqrt((xo)**2 + (yo)**2 +(zo)**2)
        xo = (xo)/norm
        yo = (yo)/norm
        zo = (zo)/norm   
        xgo = O[P[k][2]][P[k][3]][0]+float(xo*norm)/2
        ygo = O[P[k][2]][P[k][3]][1]+float(yo*norm)/2
        zgo = O[P[k][2]][P[k][3]][2]+float(zo*norm)/2       
        for i in range(Nalk):
            opt = turn_alkyl(xo, yo, zo, xab, yab, zab)
            vect1=[[float(xa[i])-xg],[float(ya[i])-yg],[float(za[i])-zg]]
            vect2 = move3(opt[0],opt[1],opt[2],vect1)
            label.append(labela[i])
            x.append(vect2[0]+xgo)
            y.append(vect2[1]+ygo)
            z.append(vect2[2]+zgo)           
    return label,x,y,z



def func4(params, ux,uy,uz,x0,y0,z0,xc,yc,zc,lcc):
    # extract current values of fit parameters from input array
    chi2 = 0.0
    alpha1= params[0]
    beta1 = params[1]
    gamma1 = params[2]   
    vect1=[[float(ux)*lcc],[float(uy)*lcc],[float(uz)*lcc]]    
    # compute chi-square
    u1 = move3(alpha1,beta1,gamma1,vect1)   
    chi2 =  (float(u1[0])+xc-x0)**2 + (float(u1[1])+yc-y0)**2 + (float(u1[2])+zc-z0)**2 
    return chi2

def turn_methyl(ux,uy,uz,x0,y0,z0,xc,yc,zc,lcc):
    xinit = [0,0,0]
    opt = simplex(func4, xinit, args=(ux,uy,uz,x0,y0,z0,xc,yc,zc,lcc), full_output=0)
    return opt

def orientate_methyl(x0,y0,z0,xc,yc,zc,xm,ym,zm,i1,lcc):
    xs, ys, zs = [], [], []
    u1x = 1/np.sqrt(3)
    u1y = 1/np.sqrt(3)
    u1z=  1/np.sqrt(3)
    u2x = -1/np.sqrt(3)
    u2y = -1/np.sqrt(3)
    u2z=  1/np.sqrt(3)    
    u3x = 1/np.sqrt(3)
    u3y = -1/np.sqrt(3)
    u3z = -1/np.sqrt(3)
    u4x = -1/np.sqrt(3)
    u4y = 1/np.sqrt(3)
    u4z=  -1/np.sqrt(3)
    opt = turn_methyl(u1x,u1y,u1z,x0,y0,z0,xc,yc,zc,lcc)
    for i in range(3):
        if i==0 :vect1 = [[float(u2x)],[float(u2y)],[float(u2z)]] 
        if i==1 :vect1 = [[float(u3x)],[float(u3y)],[float(u3z)]]
        if i==2 :vect1 = [[float(u4x)],[float(u4y)],[float(u4z)]]
        u1 = move3(opt[0],opt[1], opt[2],vect1)
        xs.append(float(u1[0]))
        ys.append(float(u1[1]))
        zs.append(float(u1[2]))          
    return xs, ys, zs


def add_methyl_fragment2(f1,k,i1,labelm,xm,ym,zm,O,lcc,labelsc,xsc,ysc,zsc):  
    x,y,z = [],[],[]
    label = [] 
    
    G= barycenter(O[k])
    dx2 = (O[k][i1][0]-float(G[0]))**2
    dy2 = (O[k][i1][1]-float(G[1]))**2
    dz2 = (O[k][i1][2]-float(G[2]))**2
    norm = np.sqrt(dx2+dy2+dz2)
    ux = (O[k][i1][0]-float(G[0]))/norm
    uy = (O[k][i1][1]-float(G[1]))/norm
    uz = (O[k][i1][2]-float(G[2]))/norm
    
    xc = O[k][i1][0]+ lcc * ux
    yc = O[k][i1][1]+ lcc * uy
    zc = O[k][i1][2]+ lcc * uz
    
    f1.write(labelm[0]+" "+str(float(xc))+" "+str(float(yc))+" "+str(float(zc))+"\n")
    labelsc.append(labelm[0])
    xsc.append(float(xc))
    ysc.append(float(yc))
    zsc.append(float(zc))
    
    xs,ys,zs = orientate_methyl(O[k][i1][0],O[k][i1][1],O[k][i1][2],xc,yc,zc,xm,ym,zm,i1,lcc) 
    for i in range(3):
        f1.write(labelm[1]+" "+str(float(xs[i])+xc)+" "+str(float(ys[i])+yc)+" "+str(float(zs[i])+zc)+"\n")
        labelsc.append(labelm[1])
        xsc.append(float(xs[i])+xc)
        ysc.append(float(ys[i])+yc)
        zsc.append(float(zs[i])+zc)
    return

def add_methyl_fragment(f2,i2,labelm,xm,ym,zm):    
    if i2 ==0:
        f2.write(labelm[0]+" "+str(float(xm[0]))+" "+str(float(ym[0]))+" "+str(float(zm[0]))+"\n")        
        f2.write(labelm[1]+" "+str(float(xm[1]))+" "+str(float(ym[1]))+" "+str(float(zm[1]))+"\n")
        f2.write(labelm[2]+" "+str(float(xm[2]))+" "+str(float(ym[2]))+" "+str(float(zm[2]))+"\n")
        f2.write(labelm[3]+" "+str(float(xm[3]))+" "+str(float(ym[3]))+" "+str(float(zm[3]))+"\n")
    if i2 ==1:
        f2.write(labelm[4]+" "+str(float(xm[4]))+" "+str(float(ym[4]))+" "+str(float(zm[4]))+"\n")
        f2.write(labelm[5]+" "+str(float(xm[5]))+" "+str(float(ym[5]))+" "+str(float(zm[5]))+"\n")
        f2.write(labelm[6]+" "+str(float(xm[6]))+" "+str(float(ym[6]))+" "+str(float(zm[6]))+"\n")
        f2.write(labelm[7]+" "+str(float(xm[7]))+" "+str(float(ym[7]))+" "+str(float(zm[7]))+"\n")
    if i2 ==2:
        f2.write(labelm[8]+" "+str(float(xm[8]))+" "+str(float(ym[8]))+" "+str(float(zm[8]))+"\n")
        f2.write(labelm[9]+" "+str(float(xm[9]))+" "+str(float(ym[9]))+" "+str(float(zm[9]))+"\n")
        f2.write(labelm[10]+" "+str(float(xm[10]))+" "+str(float(ym[10]))+" "+str(float(zm[10]))+"\n")
        f2.write(labelm[11]+" "+str(float(xm[11]))+" "+str(float(ym[11]))+" "+str(float(zm[11]))+"\n")         
    if i2 ==3:
        f2.write(labelm[12]+" "+str(float(xm[12]))+" "+str(float(ym[12]))+" "+str(float(zm[12]))+"\n")
        f2.write(labelm[13]+" "+str(float(xm[13]))+" "+str(float(ym[13]))+" "+str(float(zm[13]))+"\n")
        f2.write(labelm[14]+" "+str(float(xm[14]))+" "+str(float(ym[14]))+" "+str(float(zm[14]))+"\n")
        f2.write(labelm[15]+" "+str(float(xm[15]))+" "+str(float(ym[15]))+" "+str(float(zm[15]))+"\n")  
    if i2 ==4:
        f2.write(labelm[16]+" "+str(float(xm[16]))+" "+str(float(ym[16]))+" "+str(float(zm[16]))+"\n")
        f2.write(labelm[17]+" "+str(float(xm[17]))+" "+str(float(ym[17]))+" "+str(float(zm[17]))+"\n")
        f2.write(labelm[18]+" "+str(float(xm[18]))+" "+str(float(ym[18]))+" "+str(float(zm[18]))+"\n")
        f2.write(labelm[19]+" "+str(float(xm[19]))+" "+str(float(ym[19]))+" "+str(float(zm[19]))+"\n") 
    if i2 ==5:
        f2.write(labelm[20]+" "+str(float(xm[20]))+" "+str(float(ym[20]))+" "+str(float(zm[20]))+"\n")
        f2.write(labelm[21]+" "+str(float(xm[21]))+" "+str(float(ym[21]))+" "+str(float(zm[21]))+"\n")
        f2.write(labelm[22]+" "+str(float(xm[22]))+" "+str(float(ym[22]))+" "+str(float(zm[22]))+"\n")
        f2.write(labelm[23]+" "+str(float(xm[23]))+" "+str(float(ym[23]))+" "+str(float(zm[23]))+"\n")
    return
 
    
def add_ions(f1,Nci,labelci,xci,yci,zci,xi,yi,zi,label0,x0,y0,z0):
    for i in range(Nci):
        line = labelci[i]+" "+str(float(xci[i])+float(xi))+" "+str(float(yci[i])+float(yi))+" "+str(float(zci[i])+float(zi))+"\n"
        f1.write(line)
        label0.append(labelci[i])
        x0.append(float(xci[i])+float(xi))
        y0.append(float(yci[i])+float(yi))
        z0.append(float(zci[i])+float(zi))
    return 
    
def is_inside(xi,yi,zi,rs,Nat,x,y,z) :
    test = True
    for k in range(Nat) :
        xt, yt, zt = xi-x[k], yi-y[k], zi-z[k]          
        dist = np.sqrt(xt**2 + yt**2 + zt**2)
        if dist <= rs : 
            test = False
            break
    return test
        
def add_electrolyte_ions(f1,Nions,Rc,rs,Nat,label,x,y,z,Nci,labelci,xci,yci,zci) :
    # create a grid of possible locations
    xgrid, ygrid,zgrid = [], [], []
    Nx, Ny, Nz= 25, 25, 25
    for ix in range(-Nx,Nx):
        for iy in range(-Ny,Ny):
            for iz in range(-Nz,Nz):
                dist = np.sqrt((ix*Rc/Nx)**2 + (iy*Rc/Ny)**2 + (iz*Rc/Nz)**2)
                if dist <= Rc :
                    xgrid.append(ix*Rc/Nx)
                    ygrid.append(iy*Rc/Ny)
                    zgrid.append(iz*Rc/Nz)                             
    icompt = 0
    icompt5 = 0
    N = len(xgrid)
    print(N)
    while True :
        icompt5 = icompt5+1
        i = random.randrange(0, N)
        #print(i)
        # Test if one atom is found inside a sphere of radius rs centered on this location
        if is_inside(xgrid[i],ygrid[i],zgrid[i],rs,len(x),x,y,z)== True : 
            add_ions(f1,Nci,labelci,xci,yci,zci,xgrid[i],ygrid[i],zgrid[i],label,x,y,z)
            icompt=icompt+1
        if icompt==Nions:
            break       
    return
    
    
    
    
    
