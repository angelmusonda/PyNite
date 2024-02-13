import pickle

from numpy import linspace, sqrt

from PyNite import FEModel3D
from PyNite.ResultsModelBuilder import ModalResultsModelBuilder, FRAResultsModelBuilder
from PyNite.Section import Section
from PyNite.Visualization import Renderer

# Instantiate the 3D Finite Element Model
model = FEModel3D()

# Add the Steel material
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

# Add more nodes to descritize the members
z_coordinates = linspace(0.1, 2.9, 29)
for val in z_coordinates:
    model.add_node(f'Nz{val}', 0, 0, val)

x_coordinates = linspace(0.1, 1.9, 19)

for val in x_coordinates:
    model.add_node(f'Ny{val}', val, 0, 3)

# Merge the duplicate nodes
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

# Harmonic load
model.add_node_load("C", Direction='FX', P=3000, case='Harmonic')
model.add_load_combo(name="Harmonic", factors={"Harmonic": 1})

# Analyse
model.analyze_modal(num_modes=7)

model_damping = {'constant_modal_damping': 0.02}
model.analyze_harmonic(
    f1=40, f2=160, f_div=120, harmonic_combo="Harmonic",
    damping_options=model_damping
)

with open('validation.pickle', 'wb') as file:
    # Serialize and save the object to the file
    pickle.dump(model, file)

model_builder = FRAResultsModelBuilder(saved_model_path='validation.pickle')

freq = model.LoadFrequencies

disp_node_C = []

for f in freq:
    solved_model = model_builder.get_model(freq=f, response_type='DR')
    disp_R = solved_model.Nodes['V'].DX[model.FRA_combo_name] * 1E6
    solved_model = model_builder.get_model(freq = f, response_type='DI')
    disp_I = solved_model.Nodes['V'].DX[model.FRA_combo_name] * 1E6
    disp_node_C.append(sqrt(disp_R**2+disp_I**2))

# EXPORT csv
import csv

# Zip the lists to create pairs of elements
data = list(zip(freq, disp_node_C))

# Specify the file path for the CSV file
csv_file_path = r'D:\MEng Civil Engineering - Angel Musonda\Research\Research Idea 2 - PyNite FEA Structural Dynamics\Results\Validation\FRA\fra_pynite.csv'

# Write data to the CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write the header (optional)
    csv_writer.writerow(['Freq', 'Disp'])

    # Write the data from the zipped lists
    csv_writer.writerows(data)


print(model.NATURAL_FREQUENCIES())

renderer = Renderer(solved_model)
renderer.render_loads = False
renderer.deformed_shape = True
renderer.deformed_scale = 10
renderer.render_nodes = True

renderer.labels = False
renderer.combo_name = model.FRA_combo_name
renderer.annotation_size = 0.01
renderer.render_model()
