from PyNite import FEModel3D
from numpy import  pi

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

rho = 2500
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
frame.add_rectangle_mesh('Slab1',0.2,4,4,0.2,'Concrete',origin=[0,0,3.5],element_type='Quad')
frame.Meshes['Slab1'].generate()

for element in frame.Plates.values():
    frame.add_plate_surface_pressure(element.name,-300e3,case = 'Pressure')

# Add supports
frame.def_support('N1', True, True, True, True, True, True)
frame.def_support('N2', True, True, True, True, True, True)
frame.def_support('N3', True, True, True, True, True, True)
frame.def_support('N4', True, True, True, True, True, True)

# Add loads

#frame.add_member_dist_load('Beam1','FZ',-30e3,-30e3)

frame.add_member_dist_load('Beam2','FZ',-200e3,-200e3,case='P')
frame.add_member_dist_load('Beam2','FZ',-200e3,-200e3,case='C')
frame.add_member_pt_load('Beam1','FZ',-50e3,2,'case2')
frame.add_member_pt_load('Beam2','FZ',-50,2,'case2')

frame.add_node_load('N11','FX',-500E3,'N')
#frame.set_as_mass_case("Pressure")

#frame.set_as_mass_case('P')
#frame.set_as_mass_case('C')
#frame.set_as_mass_case('N')
#frame.set_as_mass_case('case2',(9.81, 2))

#frame.set_as_mass_case_2('Beam2')

#print(frame.Members['Beam2'].rho_increased(1))
#print(frame.Plates['R3'].M_HRZ())


"""
frame.add_member_dist_load('Beam3','FZ',-10e3,-30e3)
frame.add_member_dist_load('Beam4','FZ',-10e3,0)
frame.add_member_dist_load('Beam5','FZ',-50e3,-30e3)
frame.add_member_dist_load('Beam6','FZ',-30e3,-30e3)
frame.add_member_dist_load('Beam7','FZ',-90e3,-30e3)
frame.add_member_dist_load('Beam8','FZ',-30e3,-1e3)
"""

#Add load combination
frame.add_load_combo('COMB1',{'N': 1})
frame.add_load_combo('static_combo',{'P':1})

# Analyse Model
#print(frame.Members["Beam1"].m())
#frame.analyze_modal(sparse = False, tol= 0.0001,log = False,num_modes=10,type_of_mass_matrix='lumped')
print(frame.analyze_harmonic('COMB1',2,10,20,10,damping_ratios_in_every_mode=0.5,log=False, sparse=True,tol = 0.01))

#print(frame.set_load_frequency_to_query_results_for(2, 'COMB1'))

frame.set_load_frequency_to_query_results_for(harmonic_combo='COMB1',frequency=4.21)
print(frame.Natural_Frequencies)
#frame.set_active_mode(1)
#print(frame.Natural_Frequencies)

"""
import numpy as np
f_list = [2,2.8,3.6,4.21,4.4,5.2,5.7,6,6.8,7.6,8.4,9.2,10]
for f in f_list:
    frame.set_load_frequency_to_query_results_for(f,'COMB1')
    print(round(f,2)," Hz :",round(1000*frame.Nodes['N11'].DX['COMB1'],2))

print(frame.Natural_Frequencies) """
from PyNite.Visualization import render_model

render_model(model=frame,
             deformed_scale=50,
             render_loads=True,
             combo_name='COMB1',
             annotation_size=0.1,
             deformed_shape=True)






