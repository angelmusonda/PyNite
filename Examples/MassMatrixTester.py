from PyNite import FEModel3D

single_frame =  FEModel3D()
single_frame.add_node("N1", 0, 0, 0)
single_frame.add_node("N2", 1, 0, 0)

single_frame.add_material("FakeMaterial",1 ,1 ,1 ,1)

single_frame.add_member("Frame","N1","N2","FakeMaterial", 1, 1, 1, 1)
single_frame.def_support("N1")
single_frame.def_support("N2")
single_frame.def_releases("Frame", Ryi=True,Rzi=True, Ryj = True, Rzj = True)
print(single_frame.Members['Frame'].m())

from PyNite.Visualization import Renderer
renderer = Renderer(single_frame)
renderer.annotation_size = 0.02
renderer.labels = True
renderer.scalar_bar = True
renderer.scalar_bar_text_size = 12
#renderer.render_model()

