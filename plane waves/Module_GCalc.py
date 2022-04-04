import meep as mp
import numpy as np
import sys
from joblib import Parallel, delayed

#-----Targets---------------#

# - Deal with placing a block at a pixel
# - Make axis specifications x y z not 0 1 2 

#-----Changes V2 ------#

# - Can now deal with off-center simulation areas, and resolutions not a power of 2
# - G reshaping is now done inside this script
# - 3D get_array_metadata still doesn't work, so it still has to be done manually

# ----Changes V3 ------#

# - Metadata is now found by doing a 1D sim in each direction, this allows uneven/strange resolutions etc to be done

# ----Changes V4 [not yet merged with Coherence optimisation version] ----#

# GMeta and cellinfo updated/simplified to deal with completely independent simbox and obsbox

mp.quiet(quietval=False)

def GAll(geom, src_pos,omega,simbox,boundary_thickness,pad,res,dim):
	GNoCoords =  np.array(Parallel(n_jobs=-1, verbose = 0)(delayed(GRow)(comp = j,
																	   src_pos=src_pos,
																	   omega = omega,
																	   simbox = simbox,
																	   res = res,
																	   dim = dim,
																	   geom = geom,
																	   pad = pad,
																	   boundary_thickness = boundary_thickness,
																	   ) for j in [0,1,2]))

	if dim == 2:
		gridx, gridy = [GMeta(simbox,res,boundary_thickness,pad) for simbox in simbox]
		GWithCoords = [[gridx[i],gridy[j],GNoCoords[:,:,i,j]] for i in range(len(gridx)) for j in range(len(gridy))]
	if dim == 3:
		gridx, gridy, gridz = [GMeta(simbox,res,boundary_thickness,pad) for simbox in simbox]
		GWithCoords = [[gridx[i],gridy[j],gridz[k],GNoCoords[:,:,i,j,k]] for i in range(len(gridx)) for j in range(len(gridy)) for k in range(len(gridz))]
	return np.array(GWithCoords)

def GMeta(simbox_1D,res):
	length = simbox_1D[1]-simbox_1D[0]
	center = (simbox_1D[0]+simbox_1D[1])/2
	sim = mp.Simulation(cell_size=mp.Vector3(length),
	                geometry_center = mp.Vector3(center),
	                sources=[],
	                resolution=res, force_all_components = True, force_complex_fields = True,eps_averaging = False)
	sim.run(until=0)
	sliceVol = mp.Volume(center = mp.Vector3(center), size=mp.Vector3(length))
	metadata = sim.get_array_metadata(vol = sliceVol)[0] # <--- still only works for 2D
	return metadata

def GEps(simbox,boundary_thickness,res,dim,geom,pad):
	cell = cellInfo(simbox,boundary_thickness,dim,pad)
	dpml = boundary_thickness # Thickness of boundary layers which absorb outgoing waves
	resolution = res
	pixelSize = 1/resolution
	cell_size = cell.cell_size
	cell_center = cell.cell_center
	obs_vol = cell.obs_vol
	pml_layers = [mp.PML(dpml)] # Set up the boundary layers in meep code
	sim = mp.Simulation(cell_size=cell_size,
	                geometry=geom,
	                geometry_center = cell_center,
	                resolution=resolution,
	                boundary_layers=pml_layers, force_all_components = True, force_complex_fields = True,eps_averaging = False)

	sliceVol = mp.Volume(center = cell_center, size=obs_vol)
	sim.run(until=0)
	eps_data = sim.get_array(vol = sliceVol,component=mp.Dielectric)
	if dim == 2:
		gridx, gridy = [GMeta(simbox,res,boundary_thickness,pad) for simbox in simbox]
		epsWithCoords = [[gridx[i],gridy[j],eps_data[i,j]] for i in range(len(gridx)) for j in range(len(gridy))]
	if dim == 3:
		gridx, gridy, gridz = [GMeta(simbox,res,boundary_thickness,pad) for simbox in simbox]
		epsWithCoords = [[gridx[i],gridy[j],gridz[k],eps_data[i,j,k]] for i in range(len(gridx)) for j in range(len(gridy)) for k in range(len(gridz))]
	return np.array(epsWithCoords)



#---- !!Different to the GCalc version used in optimisation!! (obs vol and sim vol are specified separately here)---#

class cellInfo:
	def __init__(self,simbox,dpml = 0,dim = 3,pad = 0):
		self.cell_size = mp.Vector3(2*dpml + pad + simbox[0][1]-simbox[0][0],2 * dpml + pad + simbox[1][1]-simbox[1][0],2 * dpml + pad + simbox[2][1]-simbox[2][0]) # Set up the whole volume (including bounding layers) in meep code
		self.cell_center = mp.Vector3((simbox[0][1]+simbox[0][0])/2,(simbox[1][1]+simbox[1][0])/2,(simbox[2][1]+simbox[2][0])/2)

def GRow(comp, src_pos,omega,simbox,boundary_thickness,res,dim,geom,pad):
	cell = cellInfo(simbox,boundary_thickness,dim,pad)
	dpml = boundary_thickness # Thickness of boundary layers which absorb outgoing waves
	resolution = res
	pixelSize = 1/resolution

	cell_size = cell.cell_size
	cell_center = cell.cell_center
	obs_vol = cell.obs_vol

	pml_layers = [mp.PML(dpml)] # Set up the boundary layers in meep code
	pi = np.pi
	fcen = omega/(2*pi) #Central frequency of the guassian source
	dt = 5 # temporal width of the gaussian source


	sourcePos = mp.Vector3(src_pos[0],src_pos[1],src_pos[2])

	#----Calculate the Green's tensor for a source at the given position----#

	src_comp = [mp.Ex,mp.Ey,mp.Ez][comp]
	obs_comp = [mp.Ex,mp.Ey,mp.Ez]

	src =  [mp.Source(mp.GaussianSource(fcen,width=dt,  is_integrated=True), component=	src_comp, center=sourcePos)]

	ft = mp.GaussianSource(fcen,width=dt).fourier_transform(fcen)
	geometry = geom

	sim = mp.Simulation(cell_size=cell_size,
	                geometry=geometry,
	                geometry_center = cell_center,
	                sources=src,
	                resolution=resolution,
	                boundary_layers=pml_layers, force_all_components = True, force_complex_fields = True,eps_averaging = False)

	sliceVol = mp.Volume(center = cell_center, size=obs_vol)

	dft_obj = sim.add_dft_fields(obs_comp, fcen, fcen, 1, where=sliceVol)

	sim.run(until_after_sources=10)
	#metadata = sim.get_array_metadata(vol = sliceVol)[0:3] # <--- still only works for 2D
	#print("metadata gives",metadata[0], "which is length ",len(metadata[0]))
	g_data = np.array([(1/(1j*ft*omega))*sim.get_dft_array(dft_obj,j,0) for j in obs_comp])
	return g_data

def slicer3to2(arr,axis,val):
	GSelectedAxis = arr[:,axis]
	closest = GSelectedAxis[np.argmin(np.abs(GSelectedAxis-val))]
	rightEls = arr[GSelectedAxis == closest]
	return np.array(np.delete(rightEls,axis,axis = 1))


def slicer3to1(arr,axis1,axis2,val1,val2):
	GSelectedAxis1 = arr[:,axis1]
	GSelectedAxis2 = arr[:,axis2]
	closest1 = GSelectedAxis1[np.argmin(np.abs(GSelectedAxis1-val1))]
	closest2 = GSelectedAxis2[np.argmin(np.abs(GSelectedAxis2-val2))]
	bools1 = GSelectedAxis1 == closest1
	bools2 = GSelectedAxis2 == closest2
	bools = [a and b for a, b in zip(bools1,bools2)]
	rightEls = arr[bools]
	return np.array(np.delete(rightEls,[axis1,axis2],axis = 1))

def GOp(G,op):
	dim = np.shape(G)[-1]-1
	if dim==2:
		return np.array([[G[0],G[1],op(G[-1])] for G in G])
	if dim==3:
		return np.array([[G[0],G[1],G[2],op(G[-1])] for G in G])

def pColorMeshPrep(arr,axis = 0,val = 0):
	dim = np.shape(arr)[-1] - 1
	if dim == 2:
		G = arr
		X, Y = [np.unique(G[:,i]) for i in np.arange(dim)]
	if dim == 3:
		G = slicer3to2(arr,axis,val)
		X, Y, Z = [np.unique(G[:,i]) for i in np.arange(dim)]
	return (G[:,-1].reshape(len(X),len(Y))).transpose()



def picker(arr,r):
	coords = np.array(arr[:,0:3])
	arg_of_min = np.argmin([np.linalg.norm(a-r) for a in coords])
	return arr[arg_of_min]

def shadow(arr,axis):
	return np.delete(arr,axis,axis = 1)















