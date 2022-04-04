import meep as mp
import numpy as np
import tensors

SOURCE_POSITION_X = 0
SOURCE_POSITION_Y = 0
SOURCE_POSITION_Z = 0

SIM_BOX_1_X = -0.5
SIM_BOX_1_Y = 0.5

SIM_BOX_2_X = -0.5
SIM_BOX_2_Y = 0.5

SIM_BOX_3_X = -0.5
SIM_BOX_3_Y = 0.5

BLOCK = mp.Vector3(3, 3, 3)
CENTER = mp.Vector3(3, 3, 3)
MATERIAL = mp.Medium(epsilon=1)

sim_box = [[SIM_BOX_1_X, SIM_BOX_1_Y], [SIM_BOX_2_X, SIM_BOX_2_Y], [SIM_BOX_3_X, SIM_BOX_3_Y]]
source_position = [SOURCE_POSITION_X, SOURCE_POSITION_Y, SOURCE_POSITION_Z]

DPML = 2  # thickness of perfectly matched layers (PMLs) around the box
OMEGA = np.pi  # angular frequency of emitter
gmtry = [mp.Block(BLOCK, center=CENTER, material=MATERIAL)]

PADDING = 2  # padding between the simulation box and the PML
RESOLUTION = 4

T = tensors.t_tensor_constructor(gmtry, source_position, OMEGA, sim_box, DPML, PADDING, RESOLUTION, e_field=True)

M, xy = tensors.tensor_comp_extract(x3_static_variable="z", tensor_data=T)
