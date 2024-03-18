import numpy
from PyNite import FEModel3D, Section
import time as global_time

# We get the time at the start of the model generation
start_time = global_time.time()
print("Model preparation started")

# Instantiate the FE 3D Model
model = FEModel3D()

# Define a material
nu = 0.29
rho = 2400  # kg/m^3
E = 25e9  # N/m^3
G = E / (2 * (1 + nu))  # N/m^3
model.add_material('concrete', nu=nu, rho=rho, E=E, G=G)

# Define properties for the column section
h = 0.6
w = 0.6
Iy = w * h ** 3 / 12
Iz = Iy
A = h * w
J = (h * w ** 3) * (1 / 3 - 0.21 * w / h * (1 - w ** 4 / (12 * h ** 4)))

column_section = (Iy, Iz, J, A)

# Define properties for the beam section
h = 0.6  # m
w = 0.4  # m
Iy = w * h ** 3 / 12
Iz = h * w ** 3 / 12
A = h * w
J = (h * w ** 3) * (1 / 3 - 0.21 * w / h * (1 - w ** 4 / (12 * h ** 4)))
beam_section = (Iy, Iz, J, A)

# Define a mesh size
mesh_size_for_all = 5 / 3  # m

# Define some axis lines to aid in the modelling
x_axis = [0, 5, 10, 15, 20, 25, 30, 35]
y_axis = [0, 5, 10, 15, 20, 25, 30, 35]
z_axis = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]

# Add nodes for points where the beams and columns will meet
# ground_floor_column_nodes = []
# ground_floor_wall_nodes = []
# This list will keep the node names for the base floor. We will assign supports
# at those nodes

# This list will contain node names where supports can be added
node_names_for_supports = []

# These two lists will store the node names at the first floor
ground_column_node_names = []
rest_of_ground_node_names = []

for x in x_axis:
    for y in y_axis:
        node_name = model.unique_name(dictionary=model.Nodes, prefix='N')
        if y in [0, 35]:
            if x in [0, 5, 30, 35]:
                model.add_node(node_name, x, y, 0)
                rest_of_ground_node_names.append(node_name)
                continue
        if y in [5, 30]:
            if x in [0, 35]:
                model.add_node(node_name, x, y, 0)
                rest_of_ground_node_names.append(node_name)
                continue
        if y in [15, 20]:
            if x in [15, 20]:
                model.add_node(node_name, x, y, 0)
                rest_of_ground_node_names.append(node_name)
                continue

        model.add_node(node_name, x, y, 0)
        ground_column_node_names.append(node_name)


def add_column(_model, name, height, bottom_node):
    """
    This is a helper function to add a column to the model

    :param _model: The FEModel3D
    :type _model: FEModel3D
    :param name: The name to assign to the column
    :type name: str
    :param height: The height of the column
    :type height: float
    :param bottom_node: The bottom node of the column
    :type bottom_node: str

    Returns
    -------
    None
    """
    # First generate a node name for the top node
    top_node = _model.unique_name(dictionary=_model.Nodes, prefix='N')

    # Add the top node
    _model.add_node(top_node, _model.Nodes[bottom_node].X,
                    _model.Nodes[bottom_node].Y,
                    _model.Nodes[bottom_node].Z + height)

    # Add the column
    _model.add_member(name, bottom_node, top_node, 'concrete', *column_section)


def add_y_direction_wall(_model, name, wall_height, wall_width, thickness, origin):
    """
    This is a helper function to add a wall to the model. This wall's width is along the
    y-axis.
    :param _model: The FEModel3D model
    :type _model: FEModel3D
    :param name: The label to assign to the wall
    :type name: str
    :param wall_height: The height of the wall
    :type wall_height: float
    :param wall_width: The width of the wall
    :type wall_width: float
    :param thickness: The thickness of the wall
    :type thickness: float
    :param origin: The origin coordinates of the wall
    :type origin: list

    Returns
    -------
    None
    """
    mesh_size = mesh_size_for_all
    _model.add_rectangle_mesh(name=name,
                              mesh_size=mesh_size,
                              thickness=thickness,
                              width=wall_height,
                              height=wall_width,
                              material='concrete',
                              origin=origin,
                              plane='YZ'
                              )
    _model.Meshes[name].generate()


def add_x_direction_wall(_model, name, wall_height, wall_width, thickness, origin):
    """
    This is a helper function to add a wall to the model. This wall's width is along the
    x-axis.
    :param _model: The FEModel3D model
    :type _model: FEModel3D
    :param name: The label to assign to the wall
    :type name: str
    :param wall_height: The height of the wall
    :type wall_height: float
    :param wall_width: The width of the wall
    :type wall_width: float
    :param thickness: The thickness of the wall
    :type thickness: float
    :param origin: The origin coordinates of the wall
    :type origin: list

    Returns
    -------
    None
    """
    mesh_size = mesh_size_for_all
    _model.add_rectangle_mesh(name=name,
                              mesh_size=mesh_size,
                              thickness=thickness,
                              width=wall_width,
                              height=wall_height,
                              material='concrete',
                              origin=origin,
                              plane='XZ'
                              )
    _model.Meshes[name].generate()


def add_floor_beams(_model, z):
    """
    This is a helper function to draw all floor beams at any specified level
    :param _model: The FEModel3D model
    :type _model: FEModel3D
    :param z: The level where to draw the beams
    :type z: float

    Returns
    -------
    None
    """
    # Beams going in the y direction
    for x in x_axis:
        name = _model.unique_name(_model.Members, 'B')
        _model.add_node(name + '1', x, 0, z)
        _model.add_node(name + '2', x, 35, z)
        _model.add_member(name, name + '1', name + '2', 'concrete', *beam_section)

    # Beams going in the x direction
    for y in y_axis:
        name = _model.unique_name(_model.Members, 'B')
        _model.add_node(name + '1', 0, y, z)
        _model.add_node(name + '2', 35, y, z)
        _model.add_member(name, name + '1', name + '2', 'concrete', *beam_section)


# Some necessary variables
num_storey = len(z_axis)
roof_level = z_axis[num_storey - 1]

# Add all the columns. The columns are added as one single column from the ground to the roof
# This is not a problem as PyNite will break these up at all nodes, essentially at every floor
# level. This way, the model can be generated much faster.
for node in ground_column_node_names:
    column_name = model.unique_name(model.Members, 'C')
    add_column(model, column_name, max(z_axis), node)

# Now we can add walls, floors, and beams
for z in z_axis:

    # Add walls going the y direction
    add_y_direction_wall(model, 'W1_S' + str(z), wall_height=3, wall_width=5, thickness=0.2,
                         origin = [0,0,z])

    add_y_direction_wall(model, 'W2_S' + str(z), wall_height=3, wall_width=5, thickness=0.2,
                         origin=[0,30,z])

    add_y_direction_wall(model, 'W3_S' + str(z), wall_height=3, wall_width=5, thickness=0.2,
                         origin=[35,0,z])

    add_y_direction_wall(model, 'W4_S' + str(z), wall_height=3, wall_width=5, thickness=0.2,
                         origin=[35,30,z])

    add_y_direction_wall(model, 'W5_S' + str(z), wall_height=3, wall_width=5, thickness=0.2,
                         origin = [15,15,z])

    add_y_direction_wall(model, 'W6_S' + str(z), wall_height=3, wall_width=5, thickness=0.2,
                         origin=[20,15,z])

    # Add walls going the x direction
    add_x_direction_wall(model, 'W7_S' + str(z), wall_height=3, wall_width=5, thickness=0.2,
                         origin=[0,0,z])

    add_x_direction_wall(model, 'W8_S' + str(z), wall_height=3, wall_width=5, thickness=0.2,
                         origin=[0,35,z])

    add_x_direction_wall(model, 'W9_S' + str(z), wall_height=3, wall_width=5, thickness=0.2,
                         origin=[30,0,z])

    add_x_direction_wall(model, 'W10_S' + str(z), wall_height=3, wall_width=5, thickness=0.2,
                         origin=[30,35,z])

    add_x_direction_wall(model, 'W11_S' + str(z), wall_height=3, wall_width=5, thickness=0.2,
                         origin=[15,20,z])

    # Add beams
    add_floor_beams(model, z=z + 3)

    # Add floor
    slab_thickness = 0.2
    model.add_rectangle_mesh(name='Floor_' + str(z + 3),
                             thickness=slab_thickness,
                             height=35,
                             width=35,
                             origin=[0, 0, z + 3],
                             material='concrete',
                             mesh_size=mesh_size_for_all)

    # Add an opening if the floor is not the roof floor
    if z != roof_level:
        model.Meshes['Floor_' + str(z + 3)].add_rect_opening('Open_' + str(z + 3), 15, 15, 5, 5)

    # Mesh the floor
    model.Meshes['Floor_' + str(z + 3)].generate()

# Add the supports
for node in model.Nodes.values():
    if node.Z == 0:
        model.def_support(node.name, True, True,True,True,True,True)


# Merge the duplicate nodes
model.merge_duplicate_nodes(tolerance=0.001)


# In case we do not want to generate the model again, we can save it
import pickle
with open('large_model.pickle', 'wb') as file:
    # Serialize and save the object to the file
    pickle.dump(model, file)

print("Model generation completed")


# Add retrieve it when we want it
import pickle

with open('large_model.pickle', 'rb') as file:
    model: FEModel3D = pickle.load(file)

# For time history analysis, we will use seismic data
# So we start by processing it so that we can convert it into the way PyNite can use it

# Some necessary imports
from numpy import vstack, array, linspace, column_stack, savetxt

# We get the path to the earthquake data
seismic_dat_file_path = 'EL Centro.txt'

# Open the file and read its contents into the variable
with open(seismic_dat_file_path, 'r') as file:
    seismic_file_content = file.read()

# The data only contains acceleration values in units of g at an interval of 0.01s
# Initialize an empty list to store the values
seismic_values = []

# Split the content into lines
lines = seismic_file_content.split('\n')

# Skip the first 4 lines
lines = lines[4:]

# Process the remaining lines
for line in lines:
    # Split each line into individual values, remove empty strings, and convert to float
    line_values = [float(val) for val in line.split() if val]
    seismic_values.extend(line_values)

# Now we can generate a list of times at the interval of 0.01
num_values = len(seismic_values)
time = linspace(start=0, stop=(num_values - 1) * 0.01, num=num_values)

# Convert the 'values' list to a numpy array and multiply by gravity
seismic_values = 9.81 * array(seismic_values)

#Create a 2D numpy array with 'time' and 'values'
processes_seismic_data= column_stack((time, seismic_values))

# We can also export the processed data to csv, so that we don't need to do the above
# all the time
csv_file_path = 'EL Centro Processed.csv'
savetxt(csv_file_path, processes_seismic_data, delimiter=' ', header='time,acceleration', comments='')

# Start the time history analysis

damping = dict(r_alpha = 0.01, r_beta = 0.01)
model.analyze_linear_time_history_newmark_beta(
    analysis_method='direct',
    AgY=processes_seismic_data,
    step_size=0.01,
    response_duration=10,
    log=True,
    damping_options=damping
)

# Print some model information, and time taken for the analysis
end_time = global_time.time()
print('- Analysis duration: ', numpy.round((end_time-start_time)/60, 2) , ' mins')
print('Number of Nodes: ', len(model.Nodes))
print('Number of shell elements :', len(model.Quads))

from PyNite.Visualization import Renderer
renderer = Renderer(model)
renderer.set_deformed_shape(False)
renderer.set_render_loads(False)
renderer.set_annotation_size(0.1)
renderer.set_show_labels(False)
renderer.render_model()
