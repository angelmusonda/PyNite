"""
This example is meant for testing the dynamic analysis features of PyNite.
The problem and its solution are provided by REF
"""

import pickle
from numpy import linspace, sqrt
from PyNite import FEModel3D
from PyNite.ResultsModelBuilder import FRAResultsModelBuilder, THAResultsModelBuilder, \
    ModalResultsModelBuilder
from PyNite.Section import Section
from PyNite.Visualization import Renderer

# Instantiate the 3D Finite Element Model
model = FEModel3D()

# Add a steel material
# Density
rho = 7860  # kg/m3

# Modulus of elasticity
E = 200e9  # GPa

# Poisson's ratio
nu = 0.29  # Poisson's ratio

# Shear modulus of elasticity
G = E / (0.2 * (1 + nu))

model.add_material('Steel', E, G, nu, rho)

# Add a section
h = 0.1  # m
b = 0.1  # m
A = b * h

# The second moment of inertia
Iz = b * h ** 3 / 12
Iy = h * b ** 3 / 12

# Torsion constant
J = (h * b ** 3) * (1 / 3 - 0.21 * b / h * (1 - b ** 4 / (12 * h ** 4)))

section = Section(model, '100by100', A, Iy, Iz, J, 'Steel')
model.add_section('100by100', section)

# Add Control Nodes
model.add_node('A', 0, 0, 0)
model.add_node('B', 0, 0, 3)
model.add_node('C', 2, 0, 3)
model.add_node('V', 0, 0, 1.5)
model.add_node('H', 1, 0, 3)

# Add more nodes to discretize the members
z_coordinates = linspace(0.1, 2.9, 29)
for val in z_coordinates:
    model.add_node(f'Nz{val}', 0, 0, val)

x_coordinates = linspace(0.1, 1.9, 19)

for val in x_coordinates:
    model.add_node(f'Ny{val}', val, 0, 3)

# Merge the duplicate nodes, in case there are any
model.merge_duplicate_nodes()

# Add members
model.add_member('Vertical', 'A', 'B', material='Steel', section_name='100by100')
model.add_member('Horizontal', 'B', 'C', material='Steel', section_name='100by100')

# Add supports
model.def_support(
    'A', support_DX=True, support_DY=True, support_DZ=True,
    support_RX=True, support_RY=False, support_RZ=True
)

model.def_support(
    'C', support_DX=False, support_DY=True, support_DZ=True,
    support_RX=True, support_RY=False, support_RZ=True
)

# Visualise the model
renderer = Renderer(model)
renderer.deformed_shape = False
renderer.render_loads = False
renderer.annotation_size = 0.03
renderer.render_model()

# Add a point load at Node C. This load will be used to define the harmonic case
model.add_node_load("C", Direction='FX', P=3000, case='Harmonic')

# PyNite performs analyses on load combinations, so create one, just for this one load
model.add_load_combo(name="Harmonic", factors={"Harmonic": 1})

# Add another point load at Node C. This will be used to define the time history analysis case
model.add_node_load("C", Direction='FX', P=100E3, case='THA')

# To define the load as a dynamic load, define a load profile
# The load profile shows how the load varies with time
model.def_load_profile("THA", time=[0, 0.01], profile=[1, 1])

# Create a load combination just for this case. This is because PyNite performs analyses
# on load combinations, and not individual load cases
model.add_load_combo(name='THA', factors={'THA': 1})

# Perform modal analysis
model.analyze_modal(num_modes=7)

# Perform harmonic analysis (Frequency Response Analysis - FRA)
# Define  constant modal damping of 2%
model_damping = {'constant_modal_damping': 0.02}
model.analyze_harmonic(
    f1=50, f2=160, f_div=120, harmonic_combo="Harmonic",
    damping_options=model_damping
)

# Perform time history analysis
model.analyze_linear_time_history_newmark_beta(
    combo_name='THA',
    analysis_method='direct',
    step_size=0.0001,
    response_duration=0.1
)

# Now that everything is solved, save the solved model
with open('validation.pickle', 'wb') as file:
    # Serialize and save the object to the file
    pickle.dump(model, file)

# All the above code needs to be run only once. For the rest, we can use the saved model
# To see modal analysis results, build an instance of the solved model with the specified mode to
# be visualised
model_builder = ModalResultsModelBuilder(saved_model_path='validation.pickle')
model_with_modal_results = model_builder.get_model(mode=3)

# The mode shape can hence be visualised
# The modal analysis is conducted under a default load combination name "Modal Combo"
# This combo name is also stored in class variable of the FEModel3D class
renderer = Renderer(model_with_modal_results)
renderer.deformed_shape = True
renderer.render_loads = False
renderer.combo_name = model_with_modal_results.Modal_combo_name
renderer.annotation_size = 0.03
renderer.deformed_scale = 3
renderer.labels = False
renderer.render_model()

# The natural frequencies can also be accessed as follows
print("NATURAL FREQUENCIES (Hz): ", model_with_modal_results.NATURAL_FREQUENCIES())

# We now want to plot the x - displacement at the middle of the vertical member
# due to the harmonic load, for the entire load frequency range analysed for

# First we instantiate the harmonic or FRA results model builder class
model_builder = FRAResultsModelBuilder(saved_model_path='validation.pickle')

freq = model.LoadFrequencies
# We need to loop through the load frequencies, and for each frequency, we get the
# solved model, and extract the real and imaginary components of the displacement

# First we declare a variable to hold the displacement.
disp_node_V = []

# Loop through the frequencies. We can use any frequency value within the analysed range.
# The program will perform the necessary interpolations
for f in freq:
    # Get the solved model at the given load frequency value, with the real component of the
    # displacement
    solved_model = model_builder.get_model(freq=f, response_type='DR')

    # Now extract the displacement at Node V
    disp_R = solved_model.Nodes['V'].DX[model.FRA_combo_name] * 1E6

    # Repeat for the imaginary part of the displacement
    solved_model = model_builder.get_model(freq=f, response_type='DI')
    disp_I = solved_model.Nodes['V'].DX[model.FRA_combo_name] * 1E6

    # Now we can find the magnitude of the displacement from the real and imaginary parts
    disp_node_V.append(sqrt(disp_R ** 2 + disp_I ** 2))

# The results can be plotted
from matplotlib import pyplot as plt
plt.plot(freq,disp_node_V)
plt.title(label="Displacement")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Displacement (10^-3 mm)')
plt.show()

# To view the time history results, say the rotation at the middle of the horizontal
# member, we instantiate the time history results builder
model_builder = THAResultsModelBuilder(saved_model_path='validation.pickle')

# We extract the time record. We can also define our own time array, provided it is within the
# duration analysed for
time = model.TIME_THA()

# List variable to store the rotations as we loop through
rotation_node_H = []

for t in time:
    # Get the solved model at the specific time t
    solved_model = model_builder.get_model(time=t, response_type="D")

    # Extract the rotation at Node H
    rotation = -1 * solved_model.Nodes['H'].RY[model.THA_combo_name]
    rotation_node_H.append(rotation)

# Plot the results
from matplotlib import pyplot as plt
plt.plot(time,rotation_node_H)
plt.title(label="Rotation")
plt.xlabel('Times (s)')
plt.ylabel('Rotation (rad)')
plt.show()

# The deformed shape due to the harmonic or time history loads can also be
# visualised for any load frequency and time, respectively
