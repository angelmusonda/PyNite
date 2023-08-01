from PyNite import FEModel3D

# Create a FE Model
frame = FEModel3D()

# Define nodes
# First Level
frame.add_node('N1', 0, 0, 0)
frame.add_node('N2', 4, 0, 0)
frame.add_node('N3', 4, 4, 0)
frame.add_node('N4', 0, 4, 0)

# Second Level
frame.add_node('N5', 0, 0, 3.5)
frame.add_node('N6', 4, 0, 3.5)
frame.add_node('N7', 4, 4, 3.5)
frame.add_node('N8', 0, 4, 3.5)

# Third Level
frame.add_node('N9', 0, 0, 6.5)
frame.add_node('N10', 4, 0, 6.5)
frame.add_node('N11', 4, 4, 6.5)
frame.add_node('N12', 0, 4, 6.5)

# Beam section (300*400) concrete
Iy = 0.4*0.4**3 / 12
Iz = 0.4*0.4**3 / 12
A = 0.4*0.4
J = Iy + Iz

# Material
E = 30e9
nu = 0.3
G = E/(2*(1+nu))
rho = 2400
frame.add_material('Concrete', E, G,nu, rho)


# Add beams
frame.add_member('Beam1', 'N5', 'N6', 'Concrete', Iy, Iz, J, A)
frame.add_member('Beam2', 'N6', 'N7', 'Concrete', Iy, Iz, J, A)
frame.add_member('Beam3', 'N7', 'N8', 'Concrete', Iy, Iz, J, A)
frame.add_member('Beam4', 'N8', 'N5', 'Concrete', Iy, Iz, J, A)


frame.add_member('Beam5', 'N9', 'N10', 'Concrete', Iy, Iz, J, A)
frame.add_member('Beam6', 'N10', 'N11', 'Concrete', Iy, Iz, J, A)
frame.add_member('Beam7', 'N11', 'N12', 'Concrete', Iy, Iz, J, A)
frame.add_member('Beam8', 'N12', 'N9', 'Concrete', Iy, Iz, J, A)

# Column section (300*300) concrete
Iy = 0.3*0.3**3 / 12
Iz = Iy
A = 0.3*0.3
J = Iy + Iz

# Add columns
frame.add_member('Column1', 'N1', 'N5', 'Concrete', Iy, Iz, J, A)
frame.add_member('Column2', 'N2', 'N6', 'Concrete', Iy, Iz, J, A)
frame.add_member('Column3', 'N3', 'N7', 'Concrete', Iy, Iz, J, A)
frame.add_member('Column4', 'N4', 'N8', 'Concrete', Iy, Iz, J, A)

frame.add_member('Column5', 'N5', 'N9', 'Concrete', Iy, Iz, J, A)
frame.add_member('Column6', 'N6', 'N10', 'Concrete', Iy, Iz, J, A)
frame.add_member('Column7', 'N7', 'N11', 'Concrete', Iy, Iz, J, A)
frame.add_member('Column8', 'N8', 'N12', 'Concrete', Iy, Iz, J, A)

# Add Slab
#frame.add_rectangle_mesh('Slab1',0.2,4,4,0.2,'Concrete',origin=[0,0,3.5],element_type='Quad')
#frame.Meshes['Slab1'].generate()

#for element in frame.Quads.values():
#    frame.add_quad_surface_pressure(element.name,-30e3,case = 'P')

# Add supports
frame.def_support('N1', True, True, True, True, True, True)
frame.def_support('N2', True, True, True, True, True, True)
frame.def_support('N3', True, True, True, True, True, True)
frame.def_support('N4', True, True, True, True, True, True)

# Add loads

#frame.add_member_dist_load('Beam1','FZ',-30e3,-30e3)

frame.add_member_dist_load('Beam2','FZ',-20e3,-30e3,case='P')

"""
frame.add_member_dist_load('Beam3','FZ',-10e3,-30e3)
frame.add_member_dist_load('Beam4','FZ',-10e3,0)
frame.add_member_dist_load('Beam5','FZ',-50e3,-30e3)
frame.add_member_dist_load('Beam6','FZ',-30e3,-30e3)
frame.add_member_dist_load('Beam7','FZ',-90e3,-30e3)
frame.add_member_dist_load('Beam8','FZ',-30e3,-1e3)
"""

#Add load combination
#frame.add_load_combo('COMB1',{'Case 1': 1, 'P': 1})

# Analyse Model
#print(frame.Members["Beam1"].m())
frame.analyze_modal(sparse = True, tol= 0.01,log = True,num_modes=48)
#frame.analyze()

# Some Plots
#print(frame.Members['Beam1'].max_deflection('dz',combo_name='COMB1'))
#frame.Members['Beam1'].plot_moment(Direction='My',combo_name='COMB1')
#print(frame.Members['Column1'].D(combo_name = 'COMB1'))


frame.set_active_mode(48)
print(frame.natural_frequency())
from PyNite.Visualization import Renderer
model = Renderer(frame)
model.annotation_size= 0.1
model.render_loads= False
model.combo_name = 'Modal Combo'
#model.combo_name = 'COMB1'
model.deformed_shape = True
model.set_deformed_scale(1)
model.render_model()






