from PyNite import FEModel3D

# UNCOMMENT TO EDIT MODEL

# MODELLING
# Instatiate analysis model
model = FEModel3D()

# Material definition
nu = 0.29
rho = 2400
E = 25e9
G = E / (2*(1+nu))
model.add_material('concrete', nu=nu, rho=rho, E=E, G = G)

# Section definition
Iy = 0.4 * 0.4 ** 3 /12
Iz = Iy
A = 0.4*0.4
J = (0.4*0.4**3)*(1/3 - 0.21*0.4/0.4 * (1-0.4**4 / (12*0.4**4)))

# Axis definition
x_axis = [0,4,8,12,16]
y_axis = [0,4,8]
z_axis = [0,3,6,9,12,15,18]

# Add Control Nodes
ground_floor_column_nodes = []
ground_floor_wall_nodes = []
node_names_for_supports = []

num = 0
all_floor_nodes = dict()
for z in z_axis:
    floor_nodes = []
    for y in y_axis:
        for x in x_axis:
            if (x == 0 and y == 4) or (x == 16 and y == 4):
                num += 1
                name = 'N' + str(num)
                model.add_node(name, x, y - 1, z)
                if z == 0:
                    node_names_for_supports.append(name)

                num += 1
                name = 'N' + str(num)
                model.add_node(name, x, y + 1, z)
                if z == 0:
                    node_names_for_supports.append(name)

                num += 1
                name = 'N' + str(num)
                model.add_node(name,x,y,z)
                if z == 0:
                    node_names_for_supports.append(name)

            else:
                num += 1
                name = 'N' + str(num)
                model.add_node(name,x,y,z)
                floor_nodes.append(name)
                if z == 0:
                    node_names_for_supports.append(name)

    all_floor_nodes[str(z)] = floor_nodes


def add_column(model:FEModel3D,name,height,bottom_node):
    top_node = model.unique_name(dictionary=model.Nodes, prefix='N')
    model.add_node(top_node, model.Nodes[bottom_node].X,
                   model.Nodes[bottom_node].Y,
                   model.Nodes[bottom_node].Z + height)
    model.add_member(name,bottom_node,top_node,'concrete',Iy,Iz,J,A)
    model.merge_duplicate_nodes(tolerance=0.05)

def add_wall(model:FEModel3D,name,wall_height,wall_width,thickness,origin_x,origin_y,z):
    mesh_size = 1
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
    model.merge_duplicate_nodes(tolerance=0.01)


def add_floor_beams(model:FEModel3D, z):
    # Y direction beams
    for x in x_axis:
        name = model.unique_name(model.Members, 'B')
        model.add_node(name+'1',x,0,z)
        model.add_node(name+'2', x, 4, z)
        model.add_member(name,name+'1',name+'2','concrete',Iy,Iz,J,A)


        name = model.unique_name(model.Members, 'B')
        model.add_node(name+'1',x,4,z)
        model.add_node(name+'2', x, 8, z)
        model.add_member(name,name+'1',name+'2','concrete',Iy,Iz,J,A)

    # X direction beams
    for y in y_axis:

        name = model.unique_name(model.Members, 'B')
        model.add_node(name+'1', 0, y, z)
        model.add_node(name+'2', 4, y, z)
        model.add_member(name, name+'1', name+'2', 'concrete', Iy, Iz, J, A)


        name = model.unique_name(model.Members, 'B')
        model.add_node(name+'1', 4, y, z)
        model.add_node(name+'2', 8, y, z)
        model.add_member(name, name+'1', name+'2', 'concrete', Iy, Iz, J, A)

        name = model.unique_name(model.Members, 'B')
        model.add_node(name+'1', 8, y, z)
        model.add_node(name+'2', 12, y, z)
        model.add_member(name, name+'1', name+'2', 'concrete', Iy, Iz, J, A)

        name = model.unique_name(model.Members, 'B')
        model.add_node(name+'1', 12, y, z)
        model.add_node(name+'2', 16, y, z)
        model.add_member(name, name+'1', name+'2', 'concrete', Iy, Iz, J, A)

        model.merge_duplicate_nodes()

z_axis = [0,3,6,9,12,15]
for z in z_axis:
    # Add columns for first floor
    for node in all_floor_nodes[str(z)]:
        name = model.unique_name(model.Members,'C')
        add_column(model,name,3,node)

    # Add wall 1
    add_wall(model,'W1_S'+str(z),wall_height=3,wall_width=2,thickness=0.2,
             origin_x=0,origin_y=3,z=z)

    # Add wall 2
    add_wall(model,'W2_S'+str(z),wall_height=3,wall_width=2,thickness=0.2,
             origin_x=16,origin_y=3,z=z)

    # Add beams for first floor
    name = model.unique_name(model.Members,'C')
    add_floor_beams(model,z=z+3)


    # Add floor slab
    slab_thickness = 0.2
    model.add_rectangle_mesh(name = 'Floor_'+str(z+3),
                             thickness= slab_thickness,
                             height= 8,
                             width= 16,
                             origin=[0,0,z+3],
                             material='concrete',
                             mesh_size=1)

    model.Meshes['Floor_'+str(z+3)].generate()
    model.merge_duplicate_nodes()

# ADD SUPPORTS
for node in node_names_for_supports:
    model.def_support(node,1,1,1,1,1,1)

# SAVE MODEL AFTER MODELLING
import pickle
with open('full_model.pickle', 'wb') as file:
    # Serialize and save the object to the file
    pickle.dump(model, file)



# RETRIEVE MODEL
import pickle
with open('full_model.pickle','rb') as file:
    model:FEModel3D = pickle.load(file)


#UNCOMMENT TO RE - ANALYSE MODEL

# ANALYSE MODAL
#model.analyze_modal(num_modes=5,log=True,type_of_mass_matrix='lumped',sparse=False)
#print(model.MASS_PARTICIPATION_PERCENTAGES())


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

"""
model.analyze_linear_time_history_newmark_beta(
    analysis_method='direct',
    AgY=ground_acceleration,
    step_size=0.01,
    response_duration=50,
    log=True

) 

with open('solved_model.pickle', 'wb') as file:
    # Serialize and save the object to the file
    pickle.dump(model, file)

"""
from PyNite.ResultsModelBuilder import ModalResultsModelBuilder, THAResultsModelBuilder
#model_builder = ModalResultsModelBuilder('solved_model.pickle')
model_builder = THAResultsModelBuilder(saved_model='solved_model.pickle')
#solved_model = model_builder.get_model(mode = 1)

print('MODEL BUILDER COMPLETE')

from matplotlib import pyplot as plt
d = []
time_for_plot = linspace(0,49,5000)


for t in time_for_plot:
   solved_model = model_builder.get_model(time=t,response_type='D')
   d.append(solved_model.Nodes['N122'].DY['THA combo'])

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


renderer = render_model(solved_model,
                        combo_name='THA combo',
                        deformed_shape=True,
                        deformed_scale=100,
                        annotation_size=0.1,
                        render_loads=False)
