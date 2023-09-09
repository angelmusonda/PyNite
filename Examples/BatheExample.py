from PyNite import FEModel3D
model = FEModel3D()
node_num = 1;
from numpy import linspace
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

#for node in model.Nodes.values():
#   model.def_support(node.name,support_DX=True)

# Add supports
model.def_support('A',support_DY=True, support_DZ=True, support_RZ=True,support_DX=True)
model.def_support('C',support_DZ=True, support_RY=True, support_DX=True)

model.add_node_load('C','FY',100e3, case='Case 1')
model.add_load_combo('COMB1',{'Case 1':1})

model.define_load_profile('Case 1', [0,0.01],[1,1])


#model.analyze_modal(num_modes=20,sparse=True)
model.analyze_linear_time_history_HHT_alpha(  analysis_method='direct',
                                            combo_name='COMB1',
                                            AgX=None,
                                            AgY=None,
                                            AgZ=None,
                                            step_size=0.00001,
                                            response_duration=0.01,
                                            HHT_alpha=-1/3,
                                            log=True)
#print(model.NATURAL_FREQUENCIES())

import matplotlib.pyplot as plt

# Sample data
x = model.TIME_THA()
dof = model.Nodes['T'].ID * 6 + 3

y = model.DISPLACEMENT_THA()[dof,:]
# Create a line plot
plt.plot(x, y)

# Add labels and a title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')

# Show the plot
plt.show()

"""
from PyNite.Visualization import render_model
render_model(model,
             render_loads=True,
             annotation_size=0.05,
             combo_name='COMB1')"""