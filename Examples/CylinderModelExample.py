from PyNite import FEModel3D
from PyNite.Visualization import Renderer

model = FEModel3D()
model.add_node("N1", 0, 0, 0)
model.add_material("Steel", 200e9,300,0.4,3000)
model.add_cylinder_mesh("Cylinder",0.5,3,7,0.2,"Steel")

renderer = Renderer(model)
renderer.annotation_size = 0.02
renderer.labels = True
renderer.scalar_bar = True
renderer.scalar_bar_text_size = 12
renderer.render_model()