"""
This Python script aims to tackle a moderately complex structural engineering problem
using PyNite. Before running the script, ensure that the following additional libraries
are installed: Scipy, vtk, and matplotlib.

This example is hypothetical, but closely resembles situations encountered in real-world
design scenarios.
"""

# Additional libraries required: Scipy, vtk, matplotlib
# You can install these libraries using pip:
# pip install scipy vtk matplotlib

from PyNite import FEModel3D
from PyNite.Visualization import Renderer

# Instantiate the analysis model
model = FEModel3D()

# Define a concrete material
nu = 0.29
rho = 2400  # kg/m^3
E = 25e9  # N/m^2
G = E / (2 * (1 + nu))  # N/m^2

model.add_material('concrete', nu=nu, rho=rho, E=E, G=G)

# Define section properties for the columns and beams
b = 0.4
h = 0.4
Iy = b * h ** 3 / 12
Iz = Iy
A = b * h
J = (b * h ** 3) * (1 / 3 - 0.21 * h / b * (1 - b ** 4 / (12 * h ** 4)))

# Define some axis lines to aid in the modelling
x_axis = [0, 4, 8, 12, 16]
y_axis = [0, 4, 8]
z_axis = [0, 3, 6, 9, 12, 15, 18]

# Add nodes for the points where the beams and columns will meet
# ground_floor_column_nodes = []
# ground_floor_wall_nodes = []
# This list will keep the node names for the base floor. We will assign supports
# at those nodes
node_names_for_supports = []

num = 0
# This dictionary will keep track of the nodes at each floor
all_floor_nodes = dict()

# We will loop through all the levels and define nodes
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
                model.add_node(name, x, y, z)
                if z == 0:
                    node_names_for_supports.append(name)

            else:
                num += 1
                name = 'N' + str(num)
                model.add_node(name, x, y, z)
                floor_nodes.append(name)
                if z == 0:
                    node_names_for_supports.append(name)

    all_floor_nodes[str(z)] = floor_nodes
# ------------------------------------------------------------------------------------------------

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
    _model.add_member(name, bottom_node, top_node, 'concrete', Iy, Iz, J, A)

    # Merge any duplicate nodes, which is expected since we already defined control
    # nodes
    _model.merge_duplicate_nodes(tolerance=0.05)
# ------------------------------------------------------------------------------------------------

def add_wall(_model, name, wall_height, wall_width, thickness, origin):
    """
    This is a helper function to add a wall to the model
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
    # Choose a mesh size
    mesh_size = 1

    # Add the wall
    model.add_rectangle_mesh(name=name,
                             mesh_size=mesh_size,
                             thickness=thickness,
                             width=wall_height,
                             height=wall_width,
                             material='concrete',
                             origin=origin,
                             plane='YZ'
                             )
    # Mesh the wall
    _model.Meshes[name].generate()

    # After meshing, there is a possibility of generating duplicate nodes
    # Hence we need to merge the duplicate nodes
    _model.merge_duplicate_nodes(tolerance=0.01)
# ------------------------------------------------------------------------------------------------

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
    # Add beams spanning the Y direction
    for x in x_axis:
        name = _model.unique_name(_model.Members, 'B')
        _model.add_node(name + '1', x, 0, z)
        _model.add_node(name + '2', x, 4, z)
        _model.add_member(name, name + '1', name + '2', 'concrete', Iy, Iz, J, A)

        name = model.unique_name(_model.Members, 'B')
        _model.add_node(name + '1', x, 4, z)
        _model.add_node(name + '2', x, 8, z)
        _model.add_member(name, name + '1', name + '2', 'concrete', Iy, Iz, J, A)

    # Add beams spanning the X direction
    for y in y_axis:
        name = _model.unique_name(_model.Members, 'B')
        _model.add_node(name + '1', 0, y, z)
        _model.add_node(name + '2', 4, y, z)
        _model.add_member(name, name + '1', name + '2', 'concrete', Iy, Iz, J, A)

        name = _model.unique_name(_model.Members, 'B')
        _model.add_node(name + '1', 4, y, z)
        _model.add_node(name + '2', 8, y, z)
        _model.add_member(name, name + '1', name + '2', 'concrete', Iy, Iz, J, A)

        name = _model.unique_name(_model.Members, 'B')
        _model.add_node(name + '1', 8, y, z)
        _model.add_node(name + '2', 12, y, z)
        _model.add_member(name, name + '1', name + '2', 'concrete', Iy, Iz, J, A)

        name = _model.unique_name(_model.Members, 'B')
        _model.add_node(name + '1', 12, y, z)
        _model.add_node(name + '2', 16, y, z)
        _model.add_member(name, name + '1', name + '2', 'concrete', Iy, Iz, J, A)

        # Merge duplicate nodes
        _model.merge_duplicate_nodes()
# ------------------------------------------------------------------------------------------------

# Now we can draw the columns, beams, walls and floor slabs using the helper functions
for z in z_axis[0:len(z_axis) - 1]:

    # Add columns for each of the floor nodes
    for node in all_floor_nodes[str(z)]:
        name = model.unique_name(model.Members, 'C')
        add_column(model, name, 3, node)

    # Add wall 1
    add_wall(model, 'W1_S' + str(z), wall_height=3, wall_width=2, thickness=0.2,
             origin=[0,3,z])

    # Add wall 2
    add_wall(model, 'W2_S' + str(z), wall_height=3, wall_width=2, thickness=0.2,
             origin=[16,3,z])

    # Add floor beams
    name = model.unique_name(model.Members, 'C')
    add_floor_beams(model, z=z + 3)

    # Add floor slab
    slab_thickness = 0.2
    model.add_rectangle_mesh(name='Floor_' + str(z + 3),
                             thickness=slab_thickness,
                             height=8,
                             width=16,
                             origin=[0, 0, z + 3],
                             material='concrete',
                             mesh_size=1)

    model.Meshes['Floor_' + str(z + 3)].generate()

    # Merge duplicates
    model.merge_duplicate_nodes()

# Define the supports
for node in node_names_for_supports:
    model.def_support(node, True, True, True, True, True, True)
# ------------------------------------------------------------------------------------------------

# Visualise the model
renderer = Renderer(model)
renderer.set_render_loads(False)
renderer.set_annotation_size(0.1)
renderer.set_deformed_shape(False)
renderer.render_model()
# ------------------------------------------------------------------------------------------------

# Perform modal analysis
model.analyze_modal(num_modes=10, log=True, type_of_mass_matrix='consistent', sparse=True)
# ------------------------------------------------------------------------------------------------

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

# Now we can run the time history analysis
damping = dict(r_alpha=0.852804, r_beta=0.00106601)
model.analyze_linear_time_history_newmark_beta(
    analysis_method='direct',
    AgY=processes_seismic_data,
    step_size=0.01,
    response_duration=20,
    log=True,
    damping_options=damping,
    recording_frequency=1

)
# ------------------------------------------------------------------------------------------------

# Now that the model is solved, we can save it
import pickle
with open('solved_model.pickle', 'wb') as file:
    pickle.dump(model, file)


# To access modal analysis results, we retrieve the solved model
from PyNite.ResultsModelBuilder import ModalResultsModelBuilder
model_builder = ModalResultsModelBuilder('solved_model.pickle')
model_with_modal_results = model_builder.get_model(mode=1)
print("Natural Frequencies (Hz): ", model_with_modal_results.NATURAL_FREQUENCIES())
# ------------------------------------------------------------------------------------------------

# To access time history analysis results, we do the same
from PyNite.ResultsModelBuilder import THAResultsModelBuilder
model_builder = THAResultsModelBuilder(saved_model_path='solved_model.pickle')

# We want to plot the roof displacement, as well as the direct stress at the corner
# of one of the shear walls. The roof displacement is easy to get, we can directly get
# the displacement of Node "N122". For the direct stress, we need to get the right
# shell element. PyNite can not directly show the labels for shell elements, so we must
# search for it. We know one of its node is N6, from the render. We can loop through the
# shell elements and find the one that contains this node N6

solved_model = model_builder.get_model(time=0)
target_quad = None

for quad_name in solved_model.Meshes['W1_S0'].elements:
    quad = solved_model.Quads[quad_name]
    if quad.i_node.name == 'N6':
        target_quad = quad.name
        break
    elif quad.j_node.name == 'N6':
        target_quad = quad.name
        break
    elif quad.m_node.name == 'N6':
        target_quad = quad.name
        break
    elif quad.n_node.name == 'N6':
        target_quad = quad.name
        break

# Now that we know the shell element of interest, we can extract the direct stress and
# displacement
roof_displacement = []
bottom_wall_direct_stress = []
time_for_plot = linspace(0, 20, 2000)

for t in time_for_plot:
    solved_model = model_builder.get_model(time=t, response_type='D')
    roof_displacement.append(1000 * solved_model.Nodes['N122'].DY['THA combo'])
    stress = -1e-6 * solved_model.Quads[target_quad].membrane(
        s=-1, r=-1, combo_name=solved_model.THA_combo_name)[0]
    bottom_wall_direct_stress.append(stress)
# ------------------------------------------------------------------------------------------------

# Plot the roof displacement
from matplotlib import pyplot as plt
plt.plot(time_for_plot, roof_displacement)
plt.title(label="Roof Displacement")
plt.xlabel('Time (s)')
plt.ylabel('Displacement (mm)')
plt.show()
# ------------------------------------------------------------------------------------------------

# Plot the direct stress
plt.plot(time_for_plot, bottom_wall_direct_stress)
plt.title(label="Direct stress")
plt.xlabel('Time (s)')
plt.ylabel('Stress(MPa)')
plt.show()
