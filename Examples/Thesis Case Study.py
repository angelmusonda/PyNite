import numpy

from PyNite import FEModel3D, Section
from PyNite.Mesh import RectangleMesh
# UNCOMMENT TO EDIT MODEL

# MODELLING
# Instantiate analysis model

model = FEModel3D()

# Material definition
nu = 0.29
rho = 2400
E = 25e9
G = E / (2*(1+nu))
model.add_material('concrete', nu=nu, rho=rho, E=E, G = G)

# Column Section Definition
h = 0.6
w = 0.6
Iy = w * h ** 3 /12
Iz = Iy
A = h * w
J = (h*w**3)*(1/3 - 0.21*w/h * (1-w**4 / (12*h**4)))

column_section = (Iy, Iz, J, A)

# Beam Section Definition
h = 0.6
w = 0.4
Iy = w * h ** 3 /12
Iz = h * w ** 3 /12
A = h * w
J = (h * w**3)*(1/3 - 0.21*w/h * (1-w**4 / (12*h**4)))
beam_section = (Iy, Iz, J, A)

# Mesh size
mesh_size_for_all = 5/3

# Axis definition
x_axis = [0,5,10,15,20,25,30,35]
y_axis = [0,5,10,15,20,25,30,35]
z_axis = [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60]
#z_axis = [0,3,6,9,12,15,18,21]
# Add Control Nodes
ground_floor_column_nodes = []
ground_floor_wall_nodes = []
node_names_for_supports = []


# Add ground nodes
ground_column_node_names = []
rest_of_ground_node_names = []
for x in x_axis:
    for y in y_axis:
        node_name = model.unique_name(dictionary=model.Nodes, prefix='N')
        if y in [0,35]:
            if x in [0,5,30,35]:
                model.add_node(node_name,x,y,0)
                rest_of_ground_node_names.append(node_name)
                continue
        if y in [5,30]:
            if x in [0,35]:
                model.add_node(node_name,x,y,0)
                rest_of_ground_node_names.append(node_name)
                continue
        if y in [15,20]:
            if x in [15,20]:
                model.add_node(node_name,x,y,0)
                rest_of_ground_node_names.append(node_name)
                continue

        model.add_node(node_name,x,y,0)
        ground_column_node_names.append(node_name)




def add_column(model:FEModel3D,name,height,bottom_node):
    top_node = model.unique_name(dictionary=model.Nodes, prefix='N')
    model.add_node(top_node, model.Nodes[bottom_node].X,
                   model.Nodes[bottom_node].Y,
                   model.Nodes[bottom_node].Z + height)
    model.add_member(name,bottom_node,top_node,'concrete',*column_section)
    #model.merge_duplicate_nodes(tolerance=0.05)

def add_y_direction_wall(model:FEModel3D,name,wall_height,wall_width,thickness,origin_x,origin_y,z):
    mesh_size = mesh_size_for_all
    model.add_rectangle_mesh(name=name,
                             mesh_size=mesh_size,
                             thickness=thickness,
                             width=wall_height,
                             height=wall_width,
                             material='concrete',
                             origin=[origin_x,origin_y,z],
                             plane='YZ'
                             )
    model.Meshes[name].generate()
    #model.merge_duplicate_nodes(tolerance=0.01)

def add_x_direction_wall(model:FEModel3D,name,wall_height,wall_width,thickness,origin_x,origin_y,z):
    mesh_size = mesh_size_for_all
    model.add_rectangle_mesh(name=name,
                             mesh_size=mesh_size,
                             thickness=thickness,
                             width=wall_width,
                             height=wall_height,
                             material='concrete',
                             origin=[origin_x,origin_y,z],
                             plane='XZ'
                             )
    model.Meshes[name].generate()
    #model.merge_duplicate_nodes(tolerance=0.01)


def add_floor_beams(model:FEModel3D, z):
    # Y direction beams
    for x in x_axis:
        name = model.unique_name(model.Members, 'B')
        model.add_node(name+'1',x,0,z)
        model.add_node(name+'2',x,35,z)
        model.add_member(name, name + '1', name + '2', 'concrete', *beam_section)


    # X direction beams
    for y in y_axis:
        name = model.unique_name(model.Members, 'B')
        model.add_node(name + '1', 0, y, z)
        model.add_node(name + '2', 35, y, z)
        model.add_member(name, name + '1', name + '2', 'concrete', *beam_section)

num_storey = len(z_axis)
roof_level = z_axis[num_storey-1]

for node in ground_column_node_names:
    column_name = model.unique_name(model.Members, 'C')
    add_column(model, column_name,  max(z_axis), node)

for z in z_axis:

    # Add walls
    add_y_direction_wall(model,'W1_S'+str(z),wall_height=3,wall_width=5,thickness=0.2,
             origin_x=0,origin_y=0,z=z)

    add_y_direction_wall(model,'W2_S'+str(z),wall_height=3,wall_width=5,thickness=0.2,
             origin_x=0,origin_y=30,z=z)

    add_y_direction_wall(model,'W3_S'+str(z),wall_height=3,wall_width=5,thickness=0.2,
             origin_x=35,origin_y=0,z=z)

    add_y_direction_wall(model,'W4_S'+str(z),wall_height=3,wall_width=5,thickness=0.2,
             origin_x=35,origin_y=30,z=z)

    add_y_direction_wall(model,'W5_S'+str(z),wall_height=3,wall_width=5,thickness=0.2,
             origin_x=15,origin_y=15,z=z)

    add_y_direction_wall(model,'W6_S'+str(z),wall_height=3,wall_width=5,thickness=0.2,
             origin_x=20,origin_y=15,z=z)


    # X direction
    add_x_direction_wall(model,'W7_S'+str(z),wall_height=3,wall_width=5,thickness=0.2,
             origin_x=0,origin_y=0,z=z)

    add_x_direction_wall(model,'W8_S'+str(z),wall_height=3,wall_width=5,thickness=0.2,
             origin_x=0,origin_y=35,z=z)

    add_x_direction_wall(model,'W9_S'+str(z),wall_height=3,wall_width=5,thickness=0.2,
             origin_x=30,origin_y=0,z=z)

    add_x_direction_wall(model,'W10_S'+str(z),wall_height=3,wall_width=5,thickness=0.2,
             origin_x=30,origin_y=35,z=z)

    add_x_direction_wall(model,'W11_S'+str(z),wall_height=3,wall_width=5,thickness=0.2,
             origin_x=15,origin_y=20,z=z)


    # Add beams for first floor
    name = model.unique_name(model.Members,'C')
    add_floor_beams(model,z=z+3)


    # Add floor slab
    slab_thickness = 0.2
    model.add_rectangle_mesh(name = 'Floor_'+str(z+3),
                             thickness= slab_thickness,
                             height= 35,
                             width= 35,
                             origin=[0,0,z+3],
                             material='concrete',
                             mesh_size=mesh_size_for_all)
    if z!=roof_level:
        model.Meshes['Floor_'+str(z+3)].add_rect_opening('Open_'+str(z+3),15,15,5,5)
    model.Meshes['Floor_'+str(z+3)].generate()

import time as global_time
start_time = global_time.time()
print('- Building model started')
model.merge_duplicate_nodes(tolerance=0.001)
end_time = global_time.time()
print('- Building model completed - Duration: ', numpy.round((end_time-start_time)/60, 2) , ' mins')


# ADD SUPPORTS
for node in model.Nodes.values():
    if node.Z == 0:
       model.def_support(node.name,1,1,1,1,1,1)


# SAVE MODEL AFTER MODELLING
import pickle
with open('large_model.pickle', 'wb') as file:
    # Serialize and save the object to the file
    pickle.dump(model, file)



# RETRIEVE MODEL
import pickle
with open('large_model.pickle','rb') as file:
    model:FEModel3D = pickle.load(file)


#UNCOMMENT TO RE - ANALYSE MODEL

"""
# ANALYSE MODAL
model.analyze_modal(num_modes=2,log=True,type_of_mass_matrix='lumped',sparse=True,check_stability=False)
print(model.MASS_PARTICIPATION_PERCENTAGES())
print("Natural Frequencies = ",model.NATURAL_FREQUENCIES())
"""
# EARTHQUAKE DATA
from numpy import vstack, array, linspace, column_stack, savetxt
# Specify the path to your text file
file_path = 'D:\MEng Civil Engineering - Angel Musonda\Research\Research Idea 2 - PyNite FEA Structural Dynamics\Ground Motion Data EL Centro\EL Centro.txt'

# Open the file and read its contents into the variable
with open(file_path, 'r') as file:
    file_content = file.read()

# Now, you have the entire content of the file in the variable file_content
# You can process it as needed, including skipping the first 4 lines and splitting into values

# Initialize an empty list to store the values
values = []

# Split the content into lines
lines = file_content.split('\n')

# Skip the first 4 lines
lines = lines[4:]

# Process the remaining lines
for line in lines:
    # Split each line into individual values, remove empty strings, and convert to float
    line_values = [float(val) for val in line.split() if val]
    values.extend(line_values)

num_values = len(values)
time = linspace(start=0,stop = (num_values-1) * 0.01, num=num_values)

# Convert the 'values' list to a numpy array and multiply by gravity
values = 9.81 * array(values)

# Create a 2D array with 'values' in the first row and 'time' in the second row
ground_acceleration = vstack((time, values))

# Create a 2D numpy array with 'time' and 'values'
data = column_stack((time, values))

# Specify the path for the CSV file
csv_file_path = 'D:\MEng Civil Engineering - Angel Musonda\Research\Research Idea 2 - PyNite FEA Structural Dynamics\Ground Motion Data EL Centro\EL Centro Processed.csv'

# Save the data to a CSV file
#savetxt(csv_file_path, data, delimiter=' ', header='time,acceleration', comments='')
print('num values = ', num_values)
# Close the file
file.close()


#TIME HISTORY ANALYSIS

start_time = global_time.time()
damping = dict(r_alpha = 0.01, r_beta = 0.01)
model.analyze_linear_time_history_newmark_beta(
    analysis_method='direct',
    AgY=ground_acceleration,
    step_size=0.01,
    response_duration=10,
    log=True,
    damping_options=damping

)
end_time = global_time.time()
print('- Analysis duration: ', numpy.round((end_time-start_time)/60, 2) , ' mins')




from PyNite.Visualization import Renderer
renderer = Renderer(model)
renderer.annotation_size = 0.1
renderer.window_width = 750
renderer.window_height = 400
renderer.deformed_shape = True
renderer.render_loads = False
renderer.labels = False
renderer.combo_name = model.THA_combo_name
renderer.deformed_scale = 100
renderer.render_model()

#raise ValueError('stopped')

with open('solved_model.pickle', 'wb') as file:
    # Serialize and save the object to the file
    pickle.dump(model, file)


from PyNite.ResultsModelBuilder import ModalResultsModelBuilder, THAResultsModelBuilder
#model_builder = ModalResultsModelBuilder('solved_model.pickle')
model_builder = THAResultsModelBuilder(saved_model_path='solved_model.pickle')
#solved_model = model_builder.get_model(mode = 1)

print('MODEL BUILDER COMPLETE')

from matplotlib import pyplot as plt
d = []

time_for_plot = linspace(0,10,1000)


for t in time_for_plot:
   solved_model = model_builder.get_model(time=t,response_type='D')
   d.append(1000*solved_model.Nodes['N1'].DY['THA combo'])

plt.plot(time_for_plot, d)
#plt.plot(solved_model.TIME_THA(),solved_model.DISPLACEMENT_THA()[solved_model.Nodes['N122'].ID * 6 + 1,:])
plt.show()

data = column_stack((array(time_for_plot), array(d)))

# Specify the path for the CSV file
csv_file_path = 'D:\MEng Civil Engineering - Angel Musonda\Research\Research Idea 2 - PyNite FEA Structural Dynamics\Ground Motion Data EL Centro\disp.txt'

# Save the data to a CSV file
savetxt(csv_file_path, data, delimiter=' ', header='time,acceleration', comments='')

#print(solved_model.NATURAL_FREQUENCIES())
from PyNite.Visualization import render_model

solved_model = model_builder.get_model(time=5.136, response_type='D')
renderer = render_model(solved_model,
                        color_map='Txy',
                        combo_name='THA combo',
                        deformed_shape=True,
                        deformed_scale=100,
                        annotation_size=0.1,
                        render_loads=False)

print(solved_model.LoadCombos)

