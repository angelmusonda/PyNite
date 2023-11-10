import numpy
import scipy.interpolate
from scipy.interpolate import interp1d

from PyNite import FEModel3D
model = FEModel3D()
node_num = 1;
from numpy import linspace, sqrt, pi, sin


z = linspace(0.3,2.7,8)
for i in z:
    model.add_node("N"+str(node_num),0,0,i)
    node_num+=1
y = linspace(0.2, 1.8, 8)
for i in y:
    model.add_node("N"+str(node_num),0,i,3)
    node_num+=1

# Add nodes
model.add_node("A",0,0,0)
model.add_node("B",0,0,3)
model.add_node("C",0,2,3)

# Add target node
model.add_node("T",0,1,3)
model.add_node('T2',0,0,1.5)

# Add materials
nu = 0.29
rho = 7860
E = 200e9
G = E / (2*(1+nu))
model.add_material('Steel', nu=nu, rho=rho, E=E, G = G)

# Create section
Iy = 0.1 * 0.1**3/12
Iz = 0.1 * 0.1**3/12
Ix = Iy+Iz
A = 0.1 * 0.1
J = (0.1*0.1**3)*(1/3 - 0.21*0.1/0.1 * (1-0.1**4 / (12*0.1**4)))

# Add members
model.add_member('AB','A','B','Steel',Iy,Iz,J,A)
model.add_member('BC','B','C','Steel',Iy,Iz,J,A)

for node in model.Nodes.values():
   model.def_support(node.name,support_DX=True)

# Add supports
model.def_support('A',support_DY=True, support_DZ=True, support_RZ=True,support_DX=True)
model.def_support('C',support_DZ=True, support_RY=True, support_DX=True)

model.add_node_load('C','FY',3000, case='P')
weight = 9.81 * rho * A
model.add_member_dist_load('AB', 'FZ',weight,weight,case='W')
model.add_member_dist_load('AB', 'FZ',weight,weight,case='W')


model.add_load_combo('COMB1',{'P':1})

model.def_load_profile('P', [0,1],[0,1])




# Earthquake data
# Specify the path to your text file
file_path = 'D:\MEng Civil Engineering - Angel Musonda\Research\Research Idea 2 - PyNite FEA Structural Dynamics\sample_earthquake.txt'

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

time = numpy.linspace(start=0,stop=0.005*6001, num=6001)

# Convert the 'values' list to a numpy array and multiply by gravity
values = 9.81 * numpy.array(values)

# Create a 2D array with 'values' in the first row and 'time' in the second row
ground_acceleration = numpy.vstack((time, values))

model.analyze_modal(num_modes=20,sparse=True)

#print(model.MASS_PARTICIPATION_PERCENTAGES())
damping = dict(constant_modal_damping = 0.02)
print(model.analyze_harmonic(harmonic_combo='COMB1',f1 = 50, f2=150,f_div=1000,sparse=True,
                       damping_options=damping))

model.def_node_disp('T2','DY',0.2)
model.def_disp_profile('T2','DY',[0,0.1,1],[1,1,1])

print(model.analyze_linear_time_history_newmark_beta( analysis_method='direct',
                                                      combo_name='COMB1',
                                            step_size=0.5,
                                            response_duration=1,
                                            log=False, sparse=True))

import matplotlib.pyplot as plt

# Sample data
#x = model.TIME_THA()
#dof = model.Nodes['T'].ID * 6 + 3
dof2 = model.Nodes['T2'].ID * 6 + 1
#y = model.DISPLACEMENT_THA()[dof2,:]
# Create a line plot
#plt.plot(x, y)

x2 = model.LoadFrequencies
y2 = sqrt(model.DISPLACEMENT_REAL()[dof2,:]**2 + model.DISPLACEMENT_IMAGINARY()[dof2,:]**2)

plt.plot(x2,y2)
# Add labels and a title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')



#print(model.NATURAL_FREQUENCIES())
# Show the plot
plt.show()


from PyNite.Visualization import render_model
render_model(model,
             deformed_scale=2,
             deformed_shape=True,
             render_loads=False,
             annotation_size=0.05,
             combo_name='COMB1')



