# %%
from os import rename
import warnings
import copy
from math import isclose

from numpy import array, zeros, matmul, divide, subtract, atleast_2d, nanmax, argsort, ones
from numpy import seterr, real, pi, sqrt, ndarray, interp, linspace
from numpy.linalg import solve
from scipy.sparse.linalg import eigs, eigsh
from scipy.sparse import csr_matrix, lil_matrix
from scipy.interpolate import CubicSpline

from PyNite.Node3D import Node3D
from PyNite.Material import Material
from PyNite.PhysMember import PhysMember
from PyNite.Spring3D import Spring3D
from PyNite.Member3D import Member3D
from PyNite.Quad3D import Quad3D
from PyNite.Plate3D import Plate3D
from PyNite.LoadCombo import LoadCombo
from PyNite.MassCase import MassCase
from PyNite.Mesh import Mesh, RectangleMesh, AnnulusMesh, FrustrumMesh, CylinderMesh
from PyNite import Analysis

from PyNite.PyNiteExceptions import ResultsNotFoundError, InputOutOfRangeError


# %%
class FEModel3D():
    """A 3D finite element model object. This object has methods and dictionaries to create, store,
       and retrieve results from a finite element model.
    """

    def __init__(self):
        """Creates a new 3D finite element model.
        """

        # Initialize the model's various dictionaries. The dictionaries will be prepopulated with
        # the data types they store, and then those types will be removed. This will give us the
        # ability to get type-based hints when using the dictionaries.

        self.Nodes = {str: Node3D}  # A dictionary of the model's nodes
        self.Nodes.pop(str)
        self.AuxNodes = {str: Node3D}  # A dictionary of the model's auxiliary nodes
        self.AuxNodes.pop(str)
        self.Materials = {str: Material}  # A dictionary of the model's materials
        self.Materials.pop(str)
        self.Springs = {str: Spring3D}  # A dictionary of the model's springs
        self.Springs.pop(str)
        self.Members = {str: PhysMember}  # A dictionary of the model's physical members
        self.Members.pop(str)
        self.Quads = {str: Quad3D}  # A dictionary of the model's quadiralterals
        self.Quads.pop(str)
        self.Plates = {str: Plate3D}  # A dictionary of the model's rectangular plates
        self.Plates.pop(str)
        self.Meshes = {str: Mesh}  # A dictionary of the model's meshes
        self.Meshes.pop(str)
        self.LoadCombos = {str: LoadCombo}  # A dictionary of the model's load combinations
        self.LoadCombos.pop(str)
        self._D = {str: []}  # A dictionary of the model's nodal displacements by load combination
        self._D.pop(str)
        self._SHAPE = {int: []}  # A dictionary of the model's mode shape by modes
        self._SHAPE.pop(int)
        self.Active_Mode = 1  # A variable to keep track of the active mode
        self.Natural_Frequencies = []  # A list to store the calculated natural frequencies
        self._Max_D_Harmonic = []  # A dictionary of the models maximum displacements per load frequency
        self.MassCases = {str: []}  # A dictionary of load cases to be considered as mass cases
        self.MassCases.pop(str)

        self.LoadFrequencies = []  # A list to store the calculated load frequencies
        self.solution = None  # Indicates the solution type for the latest run of the model

    @property
    def LoadCases(self):
        """Returns a list of all the load cases in the model (in alphabetical order).
        """

        # Create an empty list of load cases
        cases = []

        # Step through each node
        for node in self.Nodes.values():
            # Step through each nodal load
            for load in node.NodeLoads:
                # Get the load case for each nodal laod
                cases.append(load[2])

        # Step through each member
        for member in self.Members.values():
            # Step through each member point load
            for load in member.PtLoads:
                # Get the load case for each member point load
                cases.append(load[3])
            # Step through each member distributed load
            for load in member.DistLoads:
                # Get the load case for each member distributed load
                cases.append(load[5])

        # Step through each plate/quad
        for plate in list(self.Plates.values()) + list(self.Quads.values()):
            # Step through each surface load
            for load in plate.pressures:
                # Get the load case for each plate/quad pressure
                cases.append(load[1])

        # Remove duplicates and return the list (sorted ascending)
        return sorted(list(dict.fromkeys(cases)))

    def add_node(self, name, X, Y, Z):
        """Adds a new node to the model.

        :param name: A unique user-defined name for the node. If set to None or "" a name will be
                     automatically assigned.
        :type name: str
        :param X: The node's global X-coordinate.
        :type X: number
        :param Y: The node's global Y-coordinate.
        :type Y: number
        :param Z: The node's global Z-coordinate.
        :type Z: number
        :raises NameError: Occurs when the specified name already exists in the model.
        :return: The name of the node added to the model.
        :rtype: str
        """

        # Name the node or check it doesn't already exist
        if name:
            if name in self.Nodes:
                raise NameError(f"Node name '{name}' already exists")
        else:
            # As a guess, start with the length of the dictionary
            name = "N" + str(len(self.Nodes))
            count = 1
            while name in self.Nodes:
                name = "N" + str(len(self.Nodes) + count)
                count += 1

        # Create a new node
        new_node = Node3D(name, X, Y, Z)

        # Add the new node to the list
        self.Nodes[name] = new_node

        # Flag the model as unsolved
        self.solution = None

        # Return the node name
        return name

    def add_auxnode(self, name, X, Y, Z):
        """Adds a new auxiliary node to the model. Together with a member's `i` and `j` nodes, an
        auxiliary node defines the plane in which the member's local z-axis lies, and the side of
        the member the z-axis points toward. If no auxiliary node is specified for a member, PyNite
        uses its own default configuration.

        :param name: A unique user-defined name for the node. If None or "", a name will be
                     automatically assigned.
        :type name: str
        :param X: The global X-coordinate of the node.
        :type X: number
        :param Y: The global Y-coordinate of the node.
        :type Y: number
        :param Z: The global Z-coordinate of the node.
        :type Z: number
        :raises NameError: Occurs when the specified name already exists in the model.
        :return: The name of the auxiliary node that was added to the model.
        :rtype: str
        """

        # Name the node or check it doesn't already exist
        if name:
            if name in self.AuxNodes:
                raise NameError(f"Auxnode name '{name}' already exists")
        else:
            # As a guess, start with the length of the dictionary
            name = "AN" + str(len(self.AuxNodes))
            count = 1
            while name in self.AuxNodes:
                name = "AN" + str(len(self.AuxNodes) + count)
                count += 1

        # Create a new node
        new_node = Node3D(name, X, Y, Z)

        # Add the new node to the list
        self.AuxNodes[name] = new_node

        # Flag the model as unsolved
        self.solution = None

        # Return the node name
        return name

    def add_material(self, name, E, G, nu, rho):
        """Adds a new material to the model.

        :param name: A unique user-defined name for the material.
        :type name: str
        :param E: The modulus of elasticity of the material.
        :type E: number
        :param G: The shear modulus of elasticity of the material.
        :type G: number
        :param nu: Poisson's ratio of the material.
        :type nu: number
        :param rho: The density of the material
        :type rho: number
        :raises NameError: Occurs when the specified name already exists in the model.
        """

        # Name the material or check it doesn't already exist
        if name:
            if name in self.Materials:
                raise NameError(f"Material name '{name}' already exists")
        else:
            # As a guess, start with the length of the dictionary
            name = "M" + str(len(self.Materials))
            count = 1
            while name in self.Materials:
                name = "M" + str(len(self.Materials) + count)
                count += 1

        # Create a new material
        new_material = Material(name, E, G, nu, rho)

        # Add the new material to the list
        self.Materials[name] = new_material

        # Flag the model as unsolved
        self.solution = None

    def add_spring(self, name, i_node, j_node, ks, tension_only=False, comp_only=False):
        """Adds a new spring to the model.

        :param name: A unique user-defined name for the member. If None or "", a name will be
                    automatically assigned
        :type name: str
        :param i_node: The name of the i-node (start node).
        :type i_node: str
        :param j_node: The name of the j-node (end node).
        :type j_node: str
        :param ks: The spring constant (force/displacement).
        :type ks: number
        :param tension_only: Indicates if the member is tension-only, defaults to False
        :type tension_only: bool, optional
        :param comp_only: Indicates if the member is compression-only, defaults to False
        :type comp_only: bool, optional
        :raises NameError: Occurs when the specified name already exists in the model.
        :return: The name of the spring that was added to the model.
        :rtype: str
        """

        # Name the spring or check it doesn't already exist
        if name:
            if name in self.Springs:
                raise NameError(f"Spring name '{name}' already exists")
        else:
            # As a guess, start with the length of the dictionary
            name = "S" + str(len(self.Springs))
            count = 1
            while name in self.Springs:
                name = "S" + str(len(self.Springs) + count)
                count += 1

        # Create a new spring
        new_spring = Spring3D(name, self.Nodes[i_node], self.Nodes[j_node],
                              ks, self.LoadCombos, tension_only=tension_only,
                              comp_only=comp_only)

        # Add the new spring to the list
        self.Springs[name] = new_spring

        # Flag the model as unsolved
        self.solution = None

        # Return the spring name
        return name

    def add_member(self, name, i_node, j_node, material, Iy, Iz, J, A, auxNode=None,
                   tension_only=False, comp_only=False):
        """Adds a new physical member to the model.

        :param name: A unique user-defined name for the member. If ``None`` or ``""``, a name will be automatically assigned
        :type name: str
        :param i_node: The name of the i-node (start node).
        :type i_node: str
        :param j_node: The name of the j-node (end node).
        :type j_node: str
        :param material: The name of the material of the member.
        :type material: str
        :param Iy: The moment of inertia of the member about its local y-axis.
        :type Iy: number
        :param Iz: The moment of inertia of the member about its local z-axis.
        :type Iz: number
        :param J: The polar moment of inertia of the member.
        :type J: number
        :param A: The cross-sectional area of the member.
        :type A: number
        :param auxNode: The name of the auxiliary node used to define the local z-axis. The default is ``None``, in which case the program defines the axis instead of using an auxiliary node.
        :type auxNode: str, optional
        :param tension_only: Indicates if the member is tension-only, defaults to False
        :type tension_only: bool, optional
        :param comp_only: Indicates if the member is compression-only, defaults to False
        :type comp_only: bool, optional
        :raises NameError: Occurs if the specified name already exists.
        :return: The name of the member added to the model.
        :rtype: str
        """

        # Name the member or check it doesn't already exist
        if name:
            if name in self.Members: raise NameError(f"Member name '{name}' already exists")
        else:
            # As a guess, start with the length of the dictionary
            name = "M" + str(len(self.Members))
            count = 1
            while name in self.Members:
                name = "M" + str(len(self.Members) + count)
                count += 1

        # Create a new member
        if auxNode == None:
            new_member = PhysMember(name, self.Nodes[i_node], self.Nodes[j_node], material, Iy, Iz, J, A, model=self,
                                    tension_only=tension_only, comp_only=comp_only)
        else:
            new_member = PhysMember(name, self.Nodes[i_node], self.Nodes[j_node], material, Iy, Iz, J, A, model=self,
                                    aux_node=self.AuxNodes[auxNode], tension_only=tension_only, comp_only=comp_only)

        # Add the new member to the list
        self.Members[name] = new_member

        # Flag the model as unsolved
        self.solution = None

        # Return the member name
        return name

    def add_plate(self, name, i_node, j_node, m_node, n_node, t, material, kx_mod=1.0, ky_mod=1.0):
        """Adds a new rectangular plate to the model. The plate formulation for in-plane (membrane)
        stiffness is based on an isoparametric formulation. For bending, it is based on a 12-term
        polynomial formulation. This element must be rectangular, and must not be used where a
        thick plate formulation is needed. For a more versatile plate element that can handle
        distortion and thick plate conditions, consider using the `add_quad` method instead.

        :param name: A unique user-defined name for the plate. If None or "", a name will be
                     automatically assigned.
        :type name: str
        :param i_node: The name of the i-node.
        :type i_node: str
        :param j_node: The name of the j-node.
        :type j_node: str
        :param m_node: The name of the m-node.
        :type m_node: str
        :param n_node: The name of the n-node.
        :type n_node: str
        :param t: The thickness of the element.
        :type t: number
        :param material: The name of the material for the element.
        :type material: str
        :param kx_mod: Stiffness modification factor for in-plane stiffness in the element's local
                       x-direction, defaults to 1 (no modification).
        :type kx_mod: number, optional
        :param ky_mod: Stiffness modification factor for in-plane stiffness in the element's local
                       y-direction, defaults to 1 (no modification).
        :type ky_mod: number, optional
        :raises NameError: Occurs when the specified name already exists in the model.
        :return: The name of the element added to the model.
        :rtype: str
        """

        # Name the plate or check it doesn't already exist
        if name:
            if name in self.Plates: raise NameError(f"Plate name '{name}' already exists")
        else:
            # As a guess, start with the length of the dictionary
            name = "P" + str(len(self.Plates))
            count = 1
            while name in self.Plates:
                name = "P" + str(len(self.Plates) + count)
                count += 1

        # Create a new plate
        new_plate = Plate3D(name, self.Nodes[i_node], self.Nodes[j_node], self.Nodes[m_node],
                            self.Nodes[n_node], t, material, self, kx_mod, ky_mod)

        # Add the new plate to the list
        self.Plates[name] = new_plate

        # Flag the model as unsolved
        self.solution = None

        # Return the plate name
        return name

    def add_quad(self, name, i_node, j_node, m_node, n_node, t, material, kx_mod=1.0, ky_mod=1.0):
        """Adds a new quadrilateral to the model. The quad formulation for in-plane (membrane)
        stiffness is based on an isoparametric formulation. For bending, it is based on an MITC4
        formulation. This element handles distortion relatively well, and is appropriate for thick
        and thin plates. One limitation with this element is that it does a poor job of reporting
        corner stresses. Corner forces, however are very accurate. Center stresses are very
        accurate as well. For cases where corner stress results are important, consider using the
        `add_plate` method instead.

        :param name: A unique user-defined name for the quadrilateral. If None or "", a name will
                     be automatically assigned.
        :type name: str
        :param i_node: The name of the i-node.
        :type i_node: str
        :param j_node: The name of the j-node.
        :type j_node: str
        :param m_node: The name of the m-node.
        :type m_node: str
        :param n_node: The name of the n-node.
        :type n_node: str
        :param t: The thickness of the element.
        :type t: number
        :param material: The name of the material for the element.
        :type material: str
        :param kx_mod: Stiffness modification factor for in-plane stiffness in the element's local
            x-direction, defaults to 1 (no modification).
        :type kx_mod: number, optional
        :param ky_mod: Stiffness modification factor for in-plane stiffness in the element's local
            y-direction, defaults to 1 (no modification).
        :type ky_mod: number, optional
        :raises NameError: Occurs when the specified name already exists in the model.
        :return: The name of the element added to the model.
        :rtype: str
        """

        # Name the quad or check it doesn't already exist
        if name:
            if name in self.Quads: raise NameError(f"Quad name '{name}' already exists")
        else:
            # As a guess, start with the length of the dictionary
            name = "Q" + str(len(self.Quads))
            count = 1
            while name in self.Quads:
                name = "Q" + str(len(self.Quads) + count)
                count += 1

        # Create a new member
        new_quad = Quad3D(name, self.Nodes[i_node], self.Nodes[j_node], self.Nodes[m_node],
                          self.Nodes[n_node], t, material, self, kx_mod, ky_mod)

        # Add the new member to the list
        self.Quads[name] = new_quad

        # Flag the model as unsolved
        self.solution = None

        # Return the quad name
        return name

    def add_rectangle_mesh(self, name, mesh_size, width, height, thickness, material, kx_mod=1.0, ky_mod=1.0,
                           origin=[0, 0, 0], plane='XY', x_control=None, y_control=None, start_node=None,
                           start_element=None, element_type='Quad'):
        """Adds a rectangular mesh of elements to the model.

        :param name: A unique name for the mesh.
        :type name: str
        :param mesh_size: The desired mesh size.
        :type mesh_size: number
        :param width: The overall width of the rectangular mesh measured along its local x-axis.
        :type width: number
        :param height: The overall height of the rectangular mesh measured along its local y-axis.
        :type height: number
        :param thickness: The thickness of each element in the mesh.
        :type thickness: number
        :param material: The name of the material for elements in the mesh.
        :type material: str
        :param kx_mod: Stiffness modification factor for in-plane stiffness in the element's local x-direction. Defaults to 1.0 (no modification).
        :type kx_mod: float, optional
        :param ky_mod: Stiffness modification factor for in-plane stiffness in the element's local y-direction. Defaults to 1.0 (no modification).
        :type ky_mod: float, optional
        :param origin: The origin of the regtangular mesh's local coordinate system. Defaults to [0, 0, 0]
        :type origin: list, optional
        :param plane: The plane the mesh will be parallel to. Options are 'XY', 'YZ', and 'XZ'. Defaults to 'XY'.
        :type plane: str, optional
        :param x_control: A list of control points along the mesh's local x-axis to work into the mesh. Defaults to `None`.
        :type x_control: list, optional
        :param y_control: A list of control points along the mesh's local y-axis to work into the mesh. Defaults to None.
        :type y_control: list, optional
        :param start_node: The name of the first node in the mesh. If set to `None` the program will use the next available node name. Default is `None`.
        :type start_node: str, optional
        :param start_element: The name of the first element in the mesh. If set to `None` the program will use the next available element name. Default is `None`.
        :type start_element: str, optional
        :param element_type: They type of element to make the mesh out of. Either 'Quad' or 'Rect'. Defaults to 'Quad'.
        :type element_type: str, optional
        :raises NameError: Occurs when the specified name already exists in the model.
        :return: The name of the mesh added to the model.
        :rtype: str
        """

        # Check if a mesh name has been provided
        if name:
            # Check that the mesh name isn't already being used
            if name in self.Meshes: raise NameError(f"Mesh name '{name}' already exists")
        # Rename the mesh if necessary
        else:
            name = self.unique_name(self.Meshes, 'MSH')

        # Identify the starting node and element
        if start_node is None:
            start_node = self.unique_name(self.Nodes, 'N')
        if element_type == 'Rect' and start_element is None:
            start_element = self.unique_name(self.Plates, 'R')
        elif element_type == 'Quad' and start_element is None:
            start_element = self.unique_name(self.Quads, 'Q')

        # Create the mesh
        new_mesh = RectangleMesh(mesh_size, width, height, thickness, material, self, kx_mod,
                                 ky_mod, origin, plane, x_control, y_control, start_node,
                                 start_element, element_type=element_type)

        # Add the new mesh to the `Meshes` dictionary
        self.Meshes[name] = new_mesh

        # Flag the model as unsolved
        self.solution = None

        # Return the mesh's name
        return name

    def add_annulus_mesh(self, name, mesh_size, outer_radius, inner_radius, thickness, material, kx_mod=1.0,
                         ky_mod=1.0, origin=[0, 0, 0], axis='Y', start_node=None, start_element=None):
        """Adds a mesh of quadrilaterals forming an annulus (a donut).

        :param name: A unique name for the mesh.
        :type name: str
        :param mesh_size: The target mesh size.
        :type mesh_size: float
        :param outer_radius: The radius to the outside of the annulus.
        :type outer_radius: float
        :param inner_radius: The radius to the inside of the annulus.
        :type inner_radius: float
        :param thickness: Element thickness.
        :type thickness: float
        :param material: The name of the element material.
        :type material: str
        :param kx_mod: Stiffness modification factor for radial stiffness in the element's local
                       x-direction. Default is 1.0 (no modification).
        :type kx_mod: float, optional
        :param ky_mod: Stiffness modification factor for meridional stiffness in the element's
                       local y-direction. Default is 1.0 (no modification).
        :type ky_mod: float, optional
        :param origin: The origin of the mesh. The default is [0, 0, 0].
        :type origin: list, optional
        :param axis: The global axis about which the mesh will be generated. The default is 'Y'.
        :type axis: str, optional
        :param start_node: The name of the first node in the mesh. If set to `None` the program
                           will use the next available node name. Default is `None`.
        :type start_node: str, optional
        :param start_element: The name of the first element in the mesh. If set to `None` the
                              program will use the next available element name. Default is `None`.
        :type start_element: str, optional
        :raises NameError: Occurs if the specified name already exists in the model.
        :return: The name of the mesh added to the model.
        :rtype: str
        """

        # Check if a mesh name has been provided
        if name:
            # Check that the mesh name doesn't already exist
            if name in self.Meshes: raise NameError(f"Mesh name '{name}' already exists")
        # Give the mesh a new name if necessary
        else:
            name = self.unique_name(self.Meshes, 'MSH')

        # Identify the starting node and element
        if start_node is None:
            start_node = self.unique_name(self.Nodes, 'N')
        if start_element is None:
            start_element = self.unique_name(self.Quads, 'Q')

        # Create a new mesh
        new_mesh = AnnulusMesh(mesh_size, outer_radius, inner_radius, thickness, material, self,
                               kx_mod, ky_mod, origin, axis, start_node, start_element)

        # Add the new mesh to the `Meshes` dictionary
        self.Meshes[name] = new_mesh

        # Flag the model as unsolved
        self.solution = None

        # Return the mesh's name
        return name

    def add_frustrum_mesh(self, name, mesh_size, large_radius, small_radius, height, thickness, material, kx_mod=1.0,
                          ky_mod=1.0, origin=[0, 0, 0], axis='Y', start_node=None, start_element=None):
        """Adds a mesh of quadrilaterals forming a frustrum (a cone intersected by a horizontal plane).

        :param name: A unique name for the mesh.
        :type name: str
        :param mesh_size: The target mesh size
        :type mesh_size: number
        :param large_radius: The larger of the two end radii.
        :type large_radius: number
        :param small_radius: The smaller of the two end radii.
        :type small_radius: number
        :param height: The height of the frustrum.
        :type height: number
        :param thickness: The thickness of the elements.
        :type thickness: number
        :param material: The name of the element material.
        :type material: str
        :param kx_mod: Stiffness modification factor for radial stiffness in each element's local x-direction, defaults to 1 (no modification).
        :type kx_mod: number, optional
        :param ky_mod: Stiffness modification factor for meridional stiffness in each element's local y-direction, defaults to 1 (no modification).
        :type ky_mod: number, optional
        :param origin: The origin of the mesh, defaults to [0, 0, 0].
        :type origin: list, optional
        :param axis: The global axis about which the mesh will be generated, defaults to 'Y'.
        :type axis: str, optional
        :param start_node: The name of the first node in the mesh. If set to None the program will use the next available node name, defaults to None.
        :type start_node: str, optional
        :param start_element: The name of the first element in the mesh. If set to `None` the
                              program will use the next available element name, defaults to None
        :type start_element: str, optional
        :raises NameError: Occurs if the specified name already exists.
        :return: The name of the mesh added to the model.
        :rtype: str
        """

        # Check if a name has been provided
        if name:
            # Check that the mesh name doesn't already exist
            if name in self.Meshes: raise NameError(f"Mesh name '{name}' already exists")
        # Give the mesh a new name if necessary
        else:
            name = self.unique_name(self.Meshes, 'MSH')

        # Identify the starting node and element
        if start_node is None:
            start_node = self.unique_name(self.Nodes, 'N')
        if start_element is None:
            start_element = self.unique_name(self.Quads, 'Q')

        # Create a new mesh
        new_mesh = FrustrumMesh(mesh_size, large_radius, small_radius, height, thickness, material,
                                self, kx_mod, ky_mod, origin, axis, start_node, start_element)

        # Add the new mesh to the `Meshes` dictionary
        self.Meshes[name] = new_mesh

        # Flag the model as unsolved
        self.solution = None

        # Return the mesh's name
        return name

    def add_cylinder_mesh(self, name, mesh_size, radius, height, thickness, material, kx_mod=1,
                          ky_mod=1, origin=[0, 0, 0], axis='Y', num_elements=None, start_node=None,
                          start_element=None, element_type='Quad'):
        """Adds a mesh of elements forming a cylinder.

        :param name: A unique name for the mesh.
        :type name: str
        :param mesh_size: The target mesh size.
        :type mesh_size: float
        :param radius: The radius of the cylinder.
        :type radius: float
        :param height: The height of the cylinder.
        :type height: float
        :param thickness: Element thickness.
        :type thickness: float
        :param material: The name of the element material.
        :type material: str
        :param kx_mod: Stiffness modification factor for hoop stiffness in each element's local
                       x-direction. Defaults to 1.0 (no modification).
        :type kx_mod: int, optional
        :param ky_mod: Stiffness modification factor for meridional stiffness in each element's
                       local y-direction. Defaults to 1.0 (no modification).
        :type ky_mod: int, optional
        :param origin: The origin [X, Y, Z] of the mesh. Defaults to [0, 0, 0].
        :type origin: list, optional
        :param axis: The global axis about which the mesh will be generated. Defaults to 'Y'.
        :type axis: str, optional
        :param num_elements: The number of elements to use to form each course of elements. This
                             is typically only used if you are trying to match the nodes to another
                             mesh's nodes. If set to `None` the program will automatically
                             calculate the number of elements to use based on the mesh size.
                             Defaults to None.
        :type num_elements: int, optional
        :param start_node: The name of the first node in the mesh. If set to `None` the program
                           will use the next available node name. Defaults to `None`.
        :type start_node: str, optional
        :param start_element: The name of the first element in the mesh. If set to `None` the
                              program will use the next available element name. Defaults to `None`.
        :type start_element: str, optional
        :param element_type: The type of element to make the mesh out of. Either 'Quad' or 'Rect'.
                             Defaults to 'Quad'.
        :type element_type: str, optional
        :raises NameError: Occurs when the specified mesh name is already being used in the model.
        :return: The name of the mesh added to the model
        :rtype: str
        """

        # Check if a name has been provided
        if name:
            # Check that the mesh name doesn't already exist
            if name in self.Meshes: raise NameError(f"Mesh name '{name}' already exists")
        # Give the mesh a new name if necessary
        else:
            name = self.unique_name(self.Meshes, 'MSH')

        # Identify the starting node and element
        if start_node is None:
            start_node = self.unique_name(self.Nodes, 'N')
        if element_type == 'Rect' and start_element is None:
            start_element = self.unique_name(self.Plates, 'R')
        elif element_type == 'Quad' and start_element is None:
            start_element = self.unique_name(self.Quads, 'Q')

        # Create a new mesh
        new_mesh = CylinderMesh(mesh_size, radius, height, thickness, material, self,
                                kx_mod, ky_mod, origin, axis, start_node, start_element,
                                num_elements, element_type)

        # Add the new mesh to the `Meshes` dictionary
        self.Meshes[name] = new_mesh

        # Flag the model as unsolved
        self.solution = None

        # Return the mesh's name
        return name

    def merge_duplicate_nodes(self, tolerance=0.001):
        """Removes duplicate nodes from the model and returns a list of the removed node names.

        :param tolerance: The maximum distance between two nodes in order to consider them
                          duplicates. Defaults to 0.001.
        :type tolerance: float, optional
        """

        # Initialize a dictionary marking where each node is used
        node_lookup = {node_name: [] for node_name in self.Nodes.keys()}
        element_dicts = ('Springs', 'Members', 'Plates', 'Quads')
        node_types = ('i_node', 'j_node', 'm_node', 'n_node')

        # Step through each dictionary of elements in the model (springs, members, plates, quads)
        for element_dict in element_dicts:

            # Step through each element in the dictionary
            for element in getattr(self, element_dict).values():

                # Step through each possible node type in the element (i-node, j-node, m-node, n-node)
                for node_type in node_types:

                    # Get the current element's node having the current type
                    # Return `None` if the element doesn't have this node type
                    node = getattr(element, node_type, None)

                    # Determine if the node exists on the element
                    if node is not None:
                        # Add the element to the list of elements attached to the node
                        node_lookup[node.name].append((element, node_type))

        # Make a list of the names of each node in the model
        node_names = list(self.Nodes.keys())

        # Make a list of nodes to be removed from the model
        remove_list = []

        # Step through each node in the copy of the `Nodes` dictionary
        for i, node_1_name in enumerate(node_names):

            # Skip iteration if `node_1` has already been removed
            if node_lookup[node_1_name] is None:
                continue

            # There is no need to check `node_1` against itself
            for node_2_name in node_names[i + 1:]:

                # Skip iteration if node_2 has already been removed
                if node_lookup[node_2_name] is None:
                    continue

                # Calculate the distance between nodes
                if self.Nodes[node_1_name].distance(self.Nodes[node_2_name]) > tolerance:
                    continue

                # Replace references to `node_2` in each element with references to `node_1`
                for element, node_type in node_lookup[node_2_name]:
                    setattr(element, node_type, self.Nodes[node_1_name])

                # Flag `node_2` as no longer used
                node_lookup[node_2_name] = None

                # Merge any boundary conditions
                support_cond = ('support_DX', 'support_DY', 'support_DZ', 'support_RX', 'support_RY', 'support_RZ')
                for dof in support_cond:
                    if getattr(self.Nodes[node_2_name], dof) == True:
                        setattr(self.Nodes[node_1_name], dof, True)

                # Merge any spring supports
                spring_cond = ('spring_DX', 'spring_DY', 'spring_DZ', 'spring_RX', 'spring_RY', 'spring_RZ')
                for dof in spring_cond:
                    value = getattr(self.Nodes[node_2_name], dof)
                    if value != [None, None, None]:
                        setattr(self.Nodes[node_1_name], dof, value)

                # Fix the mesh labels
                for mesh in self.Meshes.values():

                    # Fix the nodes in the mesh
                    if node_2_name in mesh.nodes.keys():
                        # Attach the correct node to the mesh
                        mesh.nodes[node_2_name] = self.Nodes[node_1_name]

                        # Fix the dictionary key
                        mesh.nodes[node_1_name] = mesh.nodes.pop(node_2_name)

                    # Fix the elements in the mesh
                    for element in mesh.elements.values():
                        if node_2_name == element.i_node.name: element.i_node = self.Nodes[node_1_name]
                        if node_2_name == element.j_node.name: element.j_node = self.Nodes[node_1_name]
                        if node_2_name == element.m_node.name: element.m_node = self.Nodes[node_1_name]
                        if node_2_name == element.n_node.name: element.n_node = self.Nodes[node_1_name]

                # Add the node to the `remove` list
                remove_list.append(node_2_name)

        # Remove `node_2` from the model's `Nodes` dictionary
        for node_name in remove_list:
            self.Nodes.pop(node_name)

        # Flag the model as unsolved
        self.solution = None

    def delete_node(self, node_name):
        """Removes a node from the model. All nodal loads associated with the node and elements attached to the node will also be removed.

        :param node_name: The name of the node to be removed.
        :type node_name: str
        """

        # Remove the node. Nodal loads are stored within the node, so they
        # will be deleted automatically when the node is deleted.
        self.Nodes.pop(node_name)

        # Find any elements attached to the node and remove them
        self.Members = {name: member for name, member in self.Members.items() if
                        member.i_node.name != node_name and member.j_node.name != node_name}
        self.Plates = {name: plate for name, plate in self.Plates.items() if
                       plate.i_node.name != node_name and plate.j_node.name != node_name and plate.m_node.name != node_name and plate.n_node.name != node_name}
        self.Quads = {name: quad for name, quad in self.Quads.items() if
                      quad.i_node.name != node_name and quad.j_node.name != node_name and quad.m_node.name != node_name and quad.n_node.name != node_name}

        # Flag the model as unsolved
        self.solution = None

    def delete_auxnode(self, auxnode_name):
        """Removes an auxiliary node from the model.

        :param auxnode_name: The name of the auxiliary node to be removed.
        :type auxnode_name: str
        """

        # Remove the auxiliary node
        self.AuxNodes.pop(auxnode_name)

        # Remove the auxiliary node from any members that were using it
        for member in self.Members.values():
            if member.auxNode == auxnode_name:
                member.auxNode = None

        # Flag the model as unsolved
        self.solution = None

    def delete_spring(self, spring_name):
        """Removes a spring from the model.

        :param spring_name: The name of the spring to be removed.
        :type spring_name: str
        """

        # Remove the spring
        self.Springs.pop(spring_name)

        # Flag the model as unsolved
        self.solution = None

    def delete_member(self, member_name):
        """Removes a member from the model. All member loads associated with the member will also
           be removed.

        :param member_name: The name of the member to be removed.
        :type member_name: str
        """

        # Remove the member. Member loads are stored within the member, so they
        # will be deleted automatically when the member is deleted.
        self.Members.pop(member_name)

        # Flag the model as unsolved
        self.solution = None

    def def_support(self, node_name, support_DX=False, support_DY=False, support_DZ=False, support_RX=False,
                    support_RY=False, support_RZ=False):
        """Defines the support conditions at a node. Nodes will default to fully unsupported
           unless specified otherwise.

        :param node_name: The name of the node where the support is being defined.
        :type node_name: str
        :param support_DX: Indicates whether the node is supported against translation in the
                           global X-direction. Defaults to False.
        :type support_DX: bool, optional
        :param support_DY: Indicates whether the node is supported against translation in the
                           global Y-direction. Defaults to False.
        :type support_DY: bool, optional
        :param support_DZ: Indicates whether the node is supported against translation in the
                           global Z-direction. Defaults to False.
        :type support_DZ: bool, optional
        :param support_RX: Indicates whether the node is supported against rotation about the
                           global X-axis. Defaults to False.
        :type support_RX: bool, optional
        :param support_RY: Indicates whether the node is supported against rotation about the
                           global Y-axis. Defaults to False.
        :type support_RY: bool, optional
        :param support_RZ: Indicates whether the node is supported against rotation about the
                           global Z-axis. Defaults to False.
        :type support_RZ: bool, optional
        """

        # Get the node to be supported
        node = self.Nodes[node_name]

        # Set the node's support conditions
        node.support_DX = support_DX
        node.support_DY = support_DY
        node.support_DZ = support_DZ
        node.support_RX = support_RX
        node.support_RY = support_RY
        node.support_RZ = support_RZ

        # Flag the model as unsolved
        self.solution = None

    def def_support_spring(self, node_name, dof, stiffness, direction=None):
        """Defines a spring support at a node.

        :param node_name: The name of the node to apply the spring support to.
        :type node_name: str
        :param dof: The degree of freedom to apply the spring support to.
        :type dof: str ('DX', 'DY', 'DZ', 'RX', 'RY', or 'RZ')
        :param stiffness: The translational or rotational stiffness of the spring support.
        :type stiffness: float
        :param direction: The direction in which the spring can act. '+' allows the spring to resist positive displacements. '-' allows the spring to resist negative displacements. None allows the spring to act in both directions. Default is None.
        :type direction: str or None ('+', '-', None), optional
        :raises ValueError: Occurs when an invalid support spring direction has been specified.
        :raises ValueError: Occurs when an invalid support spring degree of freedom has been specified.
        """

        if dof in ('DX', 'DY', 'DZ', 'RX', 'RY', 'RZ'):
            if direction in ('+', '-', None):
                if dof == 'DX':
                    self.Nodes[node_name].spring_DX = [stiffness, direction, True]
                elif dof == 'DY':
                    self.Nodes[node_name].spring_DY = [stiffness, direction, True]
                elif dof == 'DZ':
                    self.Nodes[node_name].spring_DZ = [stiffness, direction, True]
                elif dof == 'RX':
                    self.Nodes[node_name].spring_RX = [stiffness, direction, True]
                elif dof == 'RY':
                    self.Nodes[node_name].spring_RY = [stiffness, direction, True]
                elif dof == 'RZ':
                    self.Nodes[node_name].spring_RZ = [stiffness, direction, True]
            else:
                raise ValueError('Invalid support spring direction. Specify \'+\', \'-\', or None.')
        else:
            raise ValueError(
                'Invalid support spring degree of freedom. Specify \'DX\', \'DY\', \'DZ\', \'RX\', \'RY\', or \'RZ\'')

        # Flag the model as unsolved
        self.solution = None

    def def_node_disp(self, node_name, direction, magnitude):
        """Defines a nodal displacement at a node.

        :param node_name: The name of the node where the nodal displacement is being applied.
        :type node_name: str
        :param direction: The global direction the nodal displacement is being applied in. Displacements are 'DX', 'DY', and 'DZ'. Rotations are 'RX', 'RY', and 'RZ'.
        :type direction: str
        :param magnitude: The magnitude of the displacement.
        :type magnitude: float
        :raises ValueError: _description_
        """

        # Validate the value of Direction
        if direction not in ('DX', 'DY', 'DZ', 'RX', 'RY', 'RZ'):
            raise ValueError(f"Direction must be 'DX', 'DY', 'DZ', 'RX', 'RY', or 'RZ'. {direction} was given.")
        # Get the node
        node = self.Nodes[node_name]

        if direction == 'DX':
            node.EnforcedDX = magnitude
        if direction == 'DY':
            node.EnforcedDY = magnitude
        if direction == 'DZ':
            node.EnforcedDZ = magnitude
        if direction == 'RX':
            node.EnforcedRX = magnitude
        if direction == 'RY':
            node.EnforcedRY = magnitude
        if direction == 'RZ':
            node.EnforcedRZ = magnitude

        # Flag the model as unsolved
        self.solution = None

    def def_releases(self, Member, Dxi=False, Dyi=False, Dzi=False, Rxi=False, Ryi=False, Rzi=False, Dxj=False,
                     Dyj=False, Dzj=False, Rxj=False, Ryj=False, Rzj=False):
        """Defines member end realeses for a member. All member end releases will default to unreleased unless specified otherwise.

        :param Member: The name of the member to have its releases modified.
        :type Member: str
        :param Dxi: Indicates whether the member is released axially at its start. Defaults to False.
        :type Dxi: bool, optional
        :param Dyi: Indicates whether the member is released for shear in the local y-axis at its start. Defaults to False.
        :type Dyi: bool, optional
        :param Dzi: Indicates whether the member is released for shear in the local z-axis at its start. Defaults to False.
        :type Dzi: bool, optional
        :param Rxi: Indicates whether the member is released for torsion at its start. Defaults to False.
        :type Rxi: bool, optional
        :param Ryi: Indicates whether the member is released for moment about the local y-axis at its start. Defaults to False.
        :type Ryi: bool, optional
        :param Rzi: Indicates whether the member is released for moment about the local z-axis at its start. Defaults to False.
        :type Rzi: bool, optional
        :param Dxj: Indicates whether the member is released axially at its end. Defaults to False.
        :type Dxj: bool, optional
        :param Dyj: Indicates whether the member is released for shear in the local y-axis at its end. Defaults to False.
        :type Dyj: bool, optional
        :param Dzj: Indicates whether the member is released for shear in the local z-axis. Defaults to False.
        :type Dzj: bool, optional
        :param Rxj: Indicates whether the member is released for torsion at its end. Defaults to False.
        :type Rxj: bool, optional
        :param Ryj: Indicates whether the member is released for moment about the local y-axis at its end. Defaults to False.
        :type Ryj: bool, optional
        :param Rzj: Indicates whether the member is released for moment about the local z-axis at its end. Defaults to False.
        :type Rzj: bool, optional
        """

        # Apply the end releases to the member
        self.Members[Member].Releases = [Dxi, Dyi, Dzi, Rxi, Ryi, Rzi, Dxj, Dyj, Dzj, Rxj, Ryj, Rzj]

        # Flag the model as unsolved
        self.solution = None

    def add_load_combo(self, name, factors, combo_tags=None):
        """Adds a load combination to the model.

        :param name: A unique name for the load combination (e.g. '1.2D+1.6L+0.5S' or 'Gravity Combo').
        :type name: str
        :param factors: A dictionary containing load cases and their corresponding factors (e.g. {'D':1.2, 'L':1.6, 'S':0.5}).
        :type factors: dict
        :param combo_tags: A list of tags used to categorize load combinations. Default is `None`. This can be useful for filtering results later on, or for limiting analysis to only those combinations with certain tags. This feature is provided for convenience. It is not necessary to use tags.
        :type combo_tags: list, optional
                    
        """


        # Create a new load combination object
        new_combo = LoadCombo(name, combo_tags, factors)

        # Add the load combination to the dictionary of load combinations
        self.LoadCombos[name] = new_combo

        # Flag the model as solved
        self.solution = None

    def set_as_mass_case(self, name, gravity=9.81, factor=1):
        """
        Set a load case as a mass case for analysis.

        This function designates a specific load case to be treated as a mass case during analysis.

        Parameters:
        -----------
        :param name: str
            The name of the load case to be used as a mass case.

        :param gravity: float, optional
            The gravitational acceleration at the location where the structure is analyzed. Default value is 9.81 m/s² for Earth.

        :param factor: float, optional
            A percentage of the load to be used as a mass case. This factor should be specified as a decimal (e.g., 0.5 for 50%).
            Some loads, such as live loads, may only contribute a small percentage to the total mass as specified in the design code.
            Default value is 1, representing the full load.

        """
        self.MassCases[name] = MassCase(name, gravity, factor)

    def add_node_load(self, Node, Direction, P, case='Case 1'):
        """Adds a nodal load to the model.

        :param Node: The name of the node where the load is being applied.
        :type Node: str
        :param Direction: The global direction the load is being applied in. Forces are `'FX'`,
                          `'FY'`, and `'FZ'`. Moments are `'MX'`, `'MY'`, and `'MZ'`.
        :type Direction: str
        :param P: The numeric value (magnitude) of the load.
        :type P: float
        :param case: The name of the load case the load belongs to. Defaults to 'Case 1'.
        :type case: str, optional
        :raises ValueError: Occurs when an invalid load direction was specified.
        """

        # Validate the value of Direction
        if Direction not in ('FX', 'FY', 'FZ', 'MX', 'MY', 'MZ'):
            raise ValueError(f"Direction must be 'FX', 'FY', 'FZ', 'MX', 'MY', or 'MZ'. {Direction} was given.")
        # Add the node load to the model
        self.Nodes[Node].NodeLoads.append((Direction, P, case))

        # Flag the model as unsolved
        self.solution = None

    def add_member_pt_load(self, Member, Direction, P, x, case='Case 1'):
        """Adds a member point load to the model.

        :param Member: The name of the member the load is being applied to.
        :type Member: str
        :param Direction: The direction in which the load is to be applied. Valid values are `'Fx'`,
                          `'Fy'`, `'Fz'`, `'Mx'`, `'My'`, `'Mz'`, `'FX'`, `'FY'`, `'FZ'`, `'MX'`, `'MY'`, or `'MZ'`.
                          Note that lower-case notation indicates use of the beam's local
                          coordinate system, while upper-case indicates use of the model's globl
                          coordinate system.
        :type Direction: str
        :param P: The numeric value (magnitude) of the load.
        :type P: float
        :param x: The load's location along the member's local x-axis.
        :type x: float
        :param case: The load case to categorize the load under. Defaults to 'Case 1'.
        :type case: str, optional
        :raises ValueError: Occurs when an invalid load direction has been specified.
        """

        # Validate the value of Direction
        if Direction not in ('Fx', 'Fy', 'Fz', 'FX', 'FY', 'FZ', 'Mx', 'My', 'Mz', 'MX', 'MY', 'MZ'):
            raise ValueError(
                f"Direction must be 'Fx', 'Fy', 'Fz', 'FX', 'FY', FZ', 'Mx', 'My', 'Mz', 'MX', 'MY', or 'MZ'. {Direction} was given.")

        # Add the point load to the member
        self.Members[Member].PtLoads.append((Direction, P, x, case))

        # Flag the model as unsolved
        self.solution = None

    def add_member_dist_load(self, Member, Direction, w1, w2, x1=None, x2=None, case='Case 1'):
        """Adds a member distributed load to the model.

        :param Member: The name of the member the load is being appied to.
        :type Member: str
        :param Direction: The direction in which the load is to be applied. Valid values are `'Fx'`,
                          `'Fy'`, `'Fz'`, `'FX'`, `'FY'`, or `'FZ'`.
                          Note that lower-case notation indicates use of the beam's local
                          coordinate system, while upper-case indicates use of the model's globl
                          coordinate system.
        :type Direction: str
        :param w1: The starting value (magnitude) of the load.
        :type w1: float
        :param w2: The ending value (magnitude) of the load.
        :type w2: float
        :param x1: The load's start location along the member's local x-axis. If this argument is
                   not specified, the start of the member will be used. Defaults to `None`
        :type x1: float, optional
        :param x2: The load's end location along the member's local x-axis. If this argument is not
                   specified, the end of the member will be used. Defaults to `None`.
        :type x2: float, optional
        :param case: _description_, defaults to 'Case 1'
        :type case: str, optional
        :raises ValueError: Occurs when an invalid load direction has been specified.
        """

        # Validate the value of Direction
        if Direction not in ('Fx', 'Fy', 'Fz', 'FX', 'FY', 'FZ'):
            raise ValueError(f"Direction must be 'Fx', 'Fy', 'Fz', 'FX', 'FY', or 'FZ'. {Direction} was given.")
        # Determine if a starting and ending points for the load have been specified.
        # If not, use the member start and end as defaults
        if x1 == None:
            start = 0
        else:
            start = x1

        if x2 == None:
            end = self.Members[Member].L()
        else:
            end = x2

        # Add the distributed load to the member
        self.Members[Member].DistLoads.append((Direction, w1, w2, start, end, case))

        # Flag the model as unsolved
        self.solution = None

    def add_plate_surface_pressure(self, plate_name, pressure, case='Case 1'):
        """Adds a surface pressure to the rectangular plate element.
        

        :param plate_name: The name for the rectangular plate to add the surface pressure to.
        :type plate_name: str
        :param pressure: The value (magnitude) for the surface pressure.
        :type pressure: float
        :param case: The load case to add the surface pressure to. Defaults to 'Case 1'.
        :type case: str, optional
        :raises Exception: Occurs when an invalid plate name has been specified.
        """

        # Add the surface pressure to the rectangle
        if plate_name in self.Plates.keys():
            self.Plates[plate_name].pressures.append([pressure, case])
        else:
            raise Exception('Invalid plate name specified for plate surface pressure.')

        # Flag the model as unsolved
        self.solution = None

    def add_quad_surface_pressure(self, quad_name, pressure, case='Case 1'):
        """Adds a surface pressure to the quadrilateral element.

        :param quad_name: The name for the quad to add the surface pressure to.
        :type quad_name: str
        :param pressure: The value (magnitude) for the surface pressure.
        :type pressure: float
        :param case: The load case to add the surface pressure to. Defaults to 'Case 1'.
        :type case: str, optional
        :raises Exception: Occurs when an invalid quad name has been specified.
        """

        # Add the surface pressure to the quadrilateral
        if quad_name in self.Quads.keys():
            self.Quads[quad_name].pressures.append([pressure, case])
        else:
            raise Exception('Invalid quad name specified for quad surface pressure.')

        # Flag the model as unsolved
        self.solution = None

    def delete_loads(self):
        """Deletes all loads from the model along with any results based on the loads.
        """

        # Delete the member loads and the calculated internal forces
        for member in self.Members.values():
            member.DistLoads = []
            member.PtLoads = []
            member.SegmentsZ = []
            member.SegmentsY = []
            member.SegmentsX = []

        # Delete the plate loads
        for plate in self.Plates.values():
            plate.pressures = []

        # Delete the quadrilateral loads
        for quad in self.Quads.values():
            quad.pressures = []

        # Delete the nodal loads, calculated displacements, and calculated reactions
        for node in self.Nodes.values():
            node.NodeLoads = []

            node.DX = {}
            node.DY = {}
            node.DZ = {}
            node.RX = {}
            node.RY = {}
            node.RZ = {}

            node.RxnFX = {}
            node.RxnFY = {}
            node.RxnFZ = {}
            node.RxnMX = {}
            node.RxnMY = {}
            node.RxnMZ = {}

        # Flag the model as unsolved
        self.solution = None

    def _aux_list(self):
        """Builds a list with known nodal displacements and with the positions in global stiffness
           matrix of known and unknown nodal displacements

        :return: A list of the global matrix indices for the unknown nodal displacements (D1_indices). A
                 list of the global matrix indices for the known nodal displacements (D2_indices). A list
                 of the known nodal displacements (D2).
        :rtype: list, list, list
        """

        D1_indices = []  # A list of the indices for the unknown nodal displacements
        D2_indices = []  # A list of the indices for the known nodal displacements
        D2 = []  # A list of the values of the known nodal displacements (D != None)

        # Create the auxiliary table
        for node in self.Nodes.values():

            # Unknown displacement DX
            if node.support_DX == False and node.EnforcedDX == None:
                D1_indices.append(node.ID * 6 + 0)
            # Known displacement DX
            elif node.EnforcedDX != None:
                D2_indices.append(node.ID * 6 + 0)
                D2.append(node.EnforcedDX)
            # Support at DX
            else:
                D2_indices.append(node.ID * 6 + 0)
                D2.append(0.0)

            # Unknown displacement DY
            if node.support_DY == False and node.EnforcedDY == None:
                D1_indices.append(node.ID * 6 + 1)
            # Known displacement DY
            elif node.EnforcedDY != None:
                D2_indices.append(node.ID * 6 + 1)
                D2.append(node.EnforcedDY)
            # Support at DY
            else:
                D2_indices.append(node.ID * 6 + 1)
                D2.append(0.0)

            # Unknown displacement DZ
            if node.support_DZ == False and node.EnforcedDZ == None:
                D1_indices.append(node.ID * 6 + 2)
            # Known displacement DZ
            elif node.EnforcedDZ != None:
                D2_indices.append(node.ID * 6 + 2)
                D2.append(node.EnforcedDZ)
            # Support at DZ
            else:
                D2_indices.append(node.ID * 6 + 2)
                D2.append(0.0)

            # Unknown displacement RX
            if node.support_RX == False and node.EnforcedRX == None:
                D1_indices.append(node.ID * 6 + 3)
            # Known displacement RX
            elif node.EnforcedRX != None:
                D2_indices.append(node.ID * 6 + 3)
                D2.append(node.EnforcedRX)
            # Support at RX
            else:
                D2_indices.append(node.ID * 6 + 3)
                D2.append(0.0)

            # Unknown displacement RY
            if node.support_RY == False and node.EnforcedRY == None:
                D1_indices.append(node.ID * 6 + 4)
            # Known displacement RY
            elif node.EnforcedRY != None:
                D2_indices.append(node.ID * 6 + 4)
                D2.append(node.EnforcedRY)
            # Support at RY
            else:
                D2_indices.append(node.ID * 6 + 4)
                D2.append(0.0)

            # Unknown displacement RZ
            if node.support_RZ == False and node.EnforcedRZ == None:
                D1_indices.append(node.ID * 6 + 5)
            # Known displacement RZ
            elif node.EnforcedRZ != None:
                D2_indices.append(node.ID * 6 + 5)
                D2.append(node.EnforcedRZ)
            # Support at RZ
            else:
                D2_indices.append(node.ID * 6 + 5)
                D2.append(0.0)

        # Return the indices and the known displacements
        return D1_indices, D2_indices, D2

    def K(self, combo_name='Combo 1', log=False, check_stability=True, sparse=True):
        """Returns the model's global stiffness matrix. The stiffness matrix will be returned in
           scipy's sparse lil format, which reduces memory usage and can be easily converted to
           other formats.

        :param combo_name: The load combination to get the stiffness matrix for. Defaults to 'Combo 1'.
        :type combo_name: str, optional
        :param log: Prints updates to the console if set to True. Defaults to False.
        :type log: bool, optional
        :param check_stability: Causes Pynite to check for instabilities if set to True. Defaults
                                to True. Set to False if you want the model to run faster.
        :type check_stability: bool, optional
        :param sparse: Returns a sparse matrix if set to True, and a dense matrix otherwise.
                       Defaults to True.
        :type sparse: bool, optional
        :return: The global stiffness matrix for the structure.
        :rtype: ndarray or coo_matrix
        """

        # Determine if a sparse matrix has been requested
        if sparse == True:
            # The stiffness matrix will be stored as a scipy `coo_matrix`. Scipy's
            # documentation states that this type of matrix is ideal for efficient
            # construction of finite element matrices. When converted to another
            # format, the `coo_matrix` sums values at the same (i, j) index. We'll
            # build the matrix from three lists.
            row = []
            col = []
            data = []
        else:
            # Initialize a dense matrix of zeros
            K = zeros((len(self.Nodes) * 6, len(self.Nodes) * 6))

        # Add stiffness terms for each nodal spring in the model
        if log: print('- Adding nodal spring support stiffness terms to global stiffness matrix')
        for node in self.Nodes.values():

            # Determine if the node has any spring supports
            if node.spring_DX[0] != None:

                # Check for an active spring support
                if node.spring_DX[2] == True:
                    m, n = node.ID * 6, node.ID * 6
                    if sparse == True:
                        data.append(float(node.spring_DX[0]))
                        row.append(m)
                        col.append(n)
                    else:
                        K[m, n] += float(node.spring_DX[0])

            if node.spring_DY[0] != None:

                # Check for an active spring support
                if node.spring_DY[2] == True:
                    m, n = node.ID * 6 + 1, node.ID * 6 + 1
                    if sparse == True:
                        data.append(float(node.spring_DY[0]))
                        row.append(m)
                        col.append(n)
                    else:
                        K[m, n] += float(node.spring_DY[0])

            if node.spring_DZ[0] != None:

                # Check for an active spring support
                if node.spring_DZ[2] == True:
                    m, n = node.ID * 6 + 2, node.ID * 6 + 2
                    if sparse == True:
                        data.append(float(node.spring_DZ[0]))
                        row.append(m)
                        col.append(n)
                    else:
                        K[m, n] += float(node.spring_DZ[0])

            if node.spring_RX[0] != None:

                # Check for an active spring support
                if node.spring_RX[2] == True:
                    m, n = node.ID * 6 + 3, node.ID * 6 + 3
                    if sparse == True:
                        data.append(float(node.spring_RX[0]))
                        row.append(m)
                        col.append(n)
                    else:
                        K[m, n] += float(node.spring_RX[0])

            if node.spring_RY[0] != None:

                # Check for an active spring support
                if node.spring_RY[2] == True:
                    m, n = node.ID * 6 + 4, node.ID * 6 + 4
                    if sparse == True:
                        data.append(float(node.spring_RY[0]))
                        row.append(m)
                        col.append(n)
                    else:
                        K[m, n] += float(node.spring_RY[0])

            if node.spring_RZ[0] != None:

                # Check for an active spring support
                if node.spring_RZ[2] == True:
                    m, n = node.ID * 6 + 5, node.ID * 6 + 5
                    if sparse == True:
                        data.append(float(node.spring_RZ[0]))
                        row.append(m)
                        col.append(n)
                    else:
                        K[m, n] += float(node.spring_RZ[0])

        # Add stiffness terms for each spring in the model
        if log: print('- Adding spring stiffness terms to global stiffness matrix')
        for spring in self.Springs.values():

            if spring.active[combo_name] == True:

                # Get the spring's global stiffness matrix
                # Storing it as a local variable eliminates the need to rebuild it every time a term is needed
                spring_K = spring.K()

                # Step through each term in the spring's stiffness matrix
                # 'a' & 'b' below are row/column indices in the spring's stiffness matrix
                # 'm' & 'n' are corresponding row/column indices in the global stiffness matrix
                for a in range(12):

                    # Determine if index 'a' is related to the i-node or j-node
                    if a < 6:
                        # Find the corresponding index 'm' in the global stiffness matrix
                        m = spring.i_node.ID * 6 + a
                    else:
                        # Find the corresponding index 'm' in the global stiffness matrix
                        m = spring.j_node.ID * 6 + (a - 6)

                    for b in range(12):

                        # Determine if index 'b' is related to the i-node or j-node
                        if b < 6:
                            # Find the corresponding index 'n' in the global stiffness matrix
                            n = spring.i_node.ID * 6 + b
                        else:
                            # Find the corresponding index 'n' in the global stiffness matrix
                            n = spring.j_node.ID * 6 + (b - 6)

                        # Now that 'm' and 'n' are known, place the term in the global stiffness matrix
                        if sparse == True:
                            row.append(m)
                            col.append(n)
                            data.append(spring_K[a, b])
                        else:
                            K[m, n] += spring_K[a, b]

        # Add stiffness terms for each physical member in the model
        if log: print('- Adding member stiffness terms to global stiffness matrix')
        for phys_member in self.Members.values():

            # Check to see if the physical member is active for the given load combination
            if phys_member.active[combo_name] == True:

                # Step through each sub-member in the physical member and add terms
                for member in phys_member.sub_members.values():

                    # Get the member's global stiffness matrix
                    # Storing it as a local variable eliminates the need to rebuild it every time a term is needed
                    member_K = member.K()

                    # Step through each term in the member's stiffness matrix
                    # 'a' & 'b' below are row/column indices in the member's stiffness matrix
                    # 'm' & 'n' are corresponding row/column indices in the global stiffness matrix
                    for a in range(12):

                        # Determine if index 'a' is related to the i-node or j-node
                        if a < 6:
                            # Find the corresponding index 'm' in the global stiffness matrix
                            m = member.i_node.ID * 6 + a
                        else:
                            # Find the corresponding index 'm' in the global stiffness matrix
                            m = member.j_node.ID * 6 + (a - 6)

                        for b in range(12):

                            # Determine if index 'b' is related to the i-node or j-node
                            if b < 6:
                                # Find the corresponding index 'n' in the global stiffness matrix
                                n = member.i_node.ID * 6 + b
                            else:
                                # Find the corresponding index 'n' in the global stiffness matrix
                                n = member.j_node.ID * 6 + (b - 6)

                            # Now that 'm' and 'n' are known, place the term in the global stiffness matrix
                            if sparse == True:
                                row.append(m)
                                col.append(n)
                                data.append(member_K[a, b])
                            else:
                                K[m, n] += member_K[a, b]

        # Add stiffness terms for each quadrilateral in the model
        if log: print('- Adding quadrilateral stiffness terms to global stiffness matrix')
        for quad in self.Quads.values():

            # Get the quadrilateral's global stiffness matrix
            # Storing it as a local variable eliminates the need to rebuild it every time a term is needed
            quad_K = quad.K()

            # Step through each term in the quadrilateral's stiffness matrix
            # 'a' & 'b' below are row/column indices in the quadrilateral's stiffness matrix
            # 'm' & 'n' are corresponding row/column indices in the global stiffness matrix
            for a in range(24):

                # Determine which node the index 'a' is related to
                if a < 6:
                    # Find the corresponding index 'm' in the global stiffness matrix
                    m = quad.m_node.ID * 6 + a
                elif a < 12:
                    # Find the corresponding index 'm' in the global stiffness matrix
                    m = quad.n_node.ID * 6 + (a - 6)
                elif a < 18:
                    # Find the corresponding index 'm' in the global stiffness matrix
                    m = quad.i_node.ID * 6 + (a - 12)
                else:
                    # Find the corresponding index 'm' in the global stiffness matrix
                    m = quad.j_node.ID * 6 + (a - 18)

                for b in range(24):

                    # Determine which node the index 'b' is related to
                    if b < 6:
                        # Find the corresponding index 'n' in the global stiffness matrix
                        n = quad.m_node.ID * 6 + b
                    elif b < 12:
                        # Find the corresponding index 'n' in the global stiffness matrix
                        n = quad.n_node.ID * 6 + (b - 6)
                    elif b < 18:
                        # Find the corresponding index 'n' in the global stiffness matrix
                        n = quad.i_node.ID * 6 + (b - 12)
                    else:
                        # Find the corresponding index 'n' in the global stiffness matrix
                        n = quad.j_node.ID * 6 + (b - 18)

                    # Now that 'm' and 'n' are known, place the term in the global stiffness matrix
                    if sparse == True:
                        row.append(m)
                        col.append(n)
                        data.append(quad_K[a, b])
                    else:
                        K[m, n] += quad_K[a, b]

        # Add stiffness terms for each plate in the model
        if log: print('- Adding plate stiffness terms to global stiffness matrix')
        for plate in self.Plates.values():

            # Get the plate's global stiffness matrix
            # Storing it as a local variable eliminates the need to rebuild it every time a term is needed
            plate_K = plate.K()

            # Step through each term in the plate's stiffness matrix
            # 'a' & 'b' below are row/column indices in the plate's stiffness matrix
            # 'm' & 'n' are corresponding row/column indices in the global stiffness matrix
            for a in range(24):

                # Determine which node the index 'a' is related to
                if a < 6:
                    # Find the corresponding index 'm' in the global stiffness matrix
                    m = plate.i_node.ID * 6 + a
                elif a < 12:
                    # Find the corresponding index 'm' in the global stiffness matrix
                    m = plate.j_node.ID * 6 + (a - 6)
                elif a < 18:
                    # Find the corresponding index 'm' in the global stiffness matrix
                    m = plate.m_node.ID * 6 + (a - 12)
                else:
                    # Find the corresponding index 'm' in the global stiffness matrix
                    m = plate.n_node.ID * 6 + (a - 18)

                for b in range(24):

                    # Determine which node the index 'b' is related to
                    if b < 6:
                        # Find the corresponding index 'n' in the global stiffness matrix
                        n = plate.i_node.ID * 6 + b
                    elif b < 12:
                        # Find the corresponding index 'n' in the global stiffness matrix
                        n = plate.j_node.ID * 6 + (b - 6)
                    elif b < 18:
                        # Find the corresponding index 'n' in the global stiffness matrix
                        n = plate.m_node.ID * 6 + (b - 12)
                    else:
                        # Find the corresponding index 'n' in the global stiffness matrix
                        n = plate.n_node.ID * 6 + (b - 18)

                    # Now that 'm' and 'n' are known, place the term in the global stiffness matrix
                    if sparse == True:
                        row.append(m)
                        col.append(n)
                        data.append(plate_K[a, b])
                    else:
                        K[m, n] += plate_K[a, b]

        if sparse:
            # The stiffness matrix will be stored as a scipy `coo_matrix`. Scipy's
            # documentation states that this type of matrix is ideal for efficient
            # construction of finite element matrices. When converted to another
            # format, the `coo_matrix` sums values at the same (i, j) index.
            from scipy.sparse import coo_matrix
            row = array(row)
            col = array(col)
            data = array(data)
            K = coo_matrix((data, (row, col)), shape=(len(self.Nodes) * 6, len(self.Nodes) * 6))

        # Check that there are no nodal instabilities
        if check_stability:
            if log: print('- Checking nodal stability')

            if sparse: Analysis._check_stability(self, K.tocsr())
            else: Analysis._check_stability(self, K)

        # Return the global stiffness matrix
        return K

    def M(self, combo_name='Combo 1', log=False, check_stability=True, sparse=True,type_mass_matrix = 'consistent'):
        """Returns the model's global mass matrix. The mass matrix will be returned in
           scipy's sparse lil format, which reduces memory usage and can be easily converted to
           other formats.

        :param combo_name: The load combination to get the mass matrix for. Defaults to 'Combo 1'.
        :type combo_name: str, optional
        :param log: Prints updates to the console if set to True. Defaults to False.
        :type log: bool, optional
        :param check_stability: Causes Pynite to check for instabilities if set to True. Defaults
                                to True. Set to False if you want the model to run faster.
        :type check_stability: bool, optional
        :param sparse: Returns a sparse matrix if set to True, and a dense matrix otherwise.
                       Defaults to True.
        :type sparse: bool, optional
        :param type_mass_matrix: The type of element mass matrix to use in the analysis
                                 Possible values: consistent, lumped
        :type type_mass_matrix: str, optional

        :return: The global mass matrix for the structure.
        :rtype: ndarray or coo_matrix
        """
        # Convert the type of mass matrix paramater to lowercase, for easy checking
        type_mass_matrix = type_mass_matrix.lower()

        # Determine if a sparse matrix has been requested
        if sparse == True:
            # The mass matrix will be stored as a scipy `coo_matrix`. Scipy's
            # documentation states that this type of matrix is ideal for efficient
            # construction of finite element matrices. When converted to another
            # format, the `coo_matrix` sums values at the same (i, j) index. We'll
            # build the matrix from three lists.
            row = []
            col = []
            data = []
        else:
            # Initialize a dense matrix of zeros
            M = zeros((len(self.Nodes) * 6, len(self.Nodes) * 6))

        # Add mass terms for each physical member in the model
        if log: print('- Adding member mass terms to global mass matrix')
        for phys_member in self.Members.values():

            # Check to see if the physical member is active for the given load combination
            if phys_member.active[combo_name] == True:

                # Step through each sub-member in the physical member and add terms
                for member in phys_member.sub_members.values():

                    # Get the member's global mass matrix
                    # Storing it as a local variable eliminates the need to rebuild it every time a term is needed
                    if type_mass_matrix == 'consistent':
                        member_M = member.M()
                    elif type_mass_matrix == 'lumped':
                        member_M = member.M_HRZ()
                    else:
                        raise ValueError('Provided input '+ str(type_mass_matrix)+ ' is incorrect. Possible values are "consistent" and "lumped"')


                    # Step through each term in the member's mass matrix
                    # 'a' & 'b' below are row/column indices in the member's mass matrix
                    # 'm' & 'n' are corresponding row/column indices in the global mass matrix
                    for a in range(12):

                        # Determine if index 'a' is related to the i-node or j-node
                        if a < 6:
                            # Find the corresponding index 'm' in the global mass matrix
                            m = member.i_node.ID * 6 + a
                        else:
                            # Find the corresponding index 'm' in the global mass matrix
                            m = member.j_node.ID * 6 + (a - 6)

                        for b in range(12):

                            # Determine if index 'b' is related to the i-node or j-node
                            if b < 6:
                                # Find the corresponding index 'n' in the global mass matrix
                                n = member.i_node.ID * 6 + b
                            else:
                                # Find the corresponding index 'n' in the global mass matrix
                                n = member.j_node.ID * 6 + (b - 6)

                            # Now that 'm' and 'n' are known, place the term in the global mass matrix
                            if sparse == True:
                                row.append(m)
                                col.append(n)
                                data.append(member_M[a, b])
                            else:
                                M[m, n] += member_M[a, b]

        # Add mass terms for each quadrilateral in the model
        if log: print('- Adding quadrilateral mass terms to global mass matrix')
        for quad in self.Quads.values():

            # Get the quadrilateral's global mass matrix
            # Storing it as a local variable eliminates the need to rebuild it every time a term is needed
            if type_mass_matrix == 'consistent':
                quad_M = quad.M()
            elif type_mass_matrix == 'lumped':
                quad_M = quad.M_HRZ()
            else:
                raise ValueError('Provided input '+ str(type_mass_matrix)+
                                 ' is incorrect. Possible values are "consistent" and "lumped"')

            # Step through each term in the quadrilateral's mass matrix
            # 'a' & 'b' below are row/column indices in the quadrilateral's mass matrix
            # 'm' & 'n' are corresponding row/column indices in the global mass matrix
            for a in range(24):

                # Determine which node the index 'a' is related to
                if a < 6:
                    # Find the corresponding index 'm' in the global mass matrix
                    m = quad.m_node.ID * 6 + a
                elif a < 12:
                    # Find the corresponding index 'm' in the global mass matrix
                    m = quad.n_node.ID * 6 + (a - 6)
                elif a < 18:
                    # Find the corresponding index 'm' in the global mass matrix
                    m = quad.i_node.ID * 6 + (a - 12)
                else:
                    # Find the corresponding index 'm' in the global mass matrix
                    m = quad.j_node.ID * 6 + (a - 18)

                for b in range(24):

                    # Determine which node the index 'b' is related to
                    if b < 6:
                        # Find the corresponding index 'n' in the global mass matrix
                        n = quad.m_node.ID * 6 + b
                    elif b < 12:
                        # Find the corresponding index 'n' in the global mass matrix
                        n = quad.n_node.ID * 6 + (b - 6)
                    elif b < 18:
                        # Find the corresponding index 'n' in the global mass matrix
                        n = quad.i_node.ID * 6 + (b - 12)
                    else:
                        # Find the corresponding index 'n' in the global mass matrix
                        n = quad.j_node.ID * 6 + (b - 18)

                    # Now that 'm' and 'n' are known, place the term in the global mass matrix
                    if sparse == True:
                        row.append(m)
                        col.append(n)
                        data.append(quad_M[a, b])
                    else:
                        M[m, n] += quad_M[a, b]

        # Add mass terms for each plate in the model
        if log: print('- Adding plate mass terms to global stiffness matrix')
        for plate in self.Plates.values():

            # Get the plate's global mass matrix
            # Storing it as a local variable eliminates the need to rebuild it every time a term is needed
            if type_mass_matrix == 'consistent':
                plate_M = plate.M()
            elif type_mass_matrix == 'lumped':
                plate_M = plate.M_HRZ()
            else:
                raise ValueError('Provided input ' + str(type_mass_matrix)
                                 + ' is incorrect. Possible values are "consistent" and "lumped"')

            # Step through each term in the plate's mass matrix
            # 'a' & 'b' below are row/column indices in the plate's mass matrix
            # 'm' & 'n' are corresponding row/column indices in the global mass matrix
            for a in range(24):

                # Determine which node the index 'a' is related to
                if a < 6:
                    # Find the corresponding index 'm' in the global mass matrix
                    m = plate.i_node.ID * 6 + a
                elif a < 12:
                    # Find the corresponding index 'm' in the global mass matrix
                    m = plate.j_node.ID * 6 + (a - 6)
                elif a < 18:
                    # Find the corresponding index 'm' in the global mass matrix
                    m = plate.m_node.ID * 6 + (a - 12)
                else:
                    # Find the corresponding index 'm' in the global mass matrix
                    m = plate.n_node.ID * 6 + (a - 18)

                for b in range(24):

                    # Determine which node the index 'b' is related to
                    if b < 6:
                        # Find the corresponding index 'n' in the global mass matrix
                        n = plate.i_node.ID * 6 + b
                    elif b < 12:
                        # Find the corresponding index 'n' in the global mass matrix
                        n = plate.j_node.ID * 6 + (b - 6)
                    elif b < 18:
                        # Find the corresponding index 'n' in the global mass matrix
                        n = plate.m_node.ID * 6 + (b - 12)
                    else:
                        # Find the corresponding index 'n' in the global mass matrix
                        n = plate.n_node.ID * 6 + (b - 18)

                    # Now that 'm' and 'n' are known, place the term in the global mass matrix
                    if sparse == True:
                        row.append(m)
                        col.append(n)
                        data.append(plate_M[a, b])
                    else:
                        M[m, n] += plate_M[a, b]

        # Add concentrated masses for point load cases taken as mass cases
        # We will distribute the point mass to the translation degrees of freedom

        for node in self.Nodes.values():

            # Get the node's ID
            ID = node.ID

            # Step through the Mass Cases
            for case in self.MassCases.keys():
                gravity = self.MassCases[case].gravity
                factor = self.MassCases[case].factor

                # Step through the nodal loads
                for load in node.NodeLoads:

                    if load[2] == case:

                        if load[0] == 'FZ' and load[1] <= 0:
                            # Calculate mass
                            mass = factor * abs(load[1])/gravity

                            # Calculate mass per translational dof
                            # mass_per_dof = mass/3    This was the initial idea but abandoned upon further research
                            mass_per_dof = mass

                            # Get the corresponding index of the node in the global matrix
                            m = ID * 6

                            if sparse == True:
                                # Translation in the FX direction
                                row.append(m+0)
                                col.append(m+0)
                                data.append(mass_per_dof)

                                # Translation in the FY direction
                                row.append(m+1)
                                col.append(m+1)
                                data.append(mass_per_dof)

                                # Translation in the FZ direction
                                row.append(m+2)
                                col.append(m+2)
                                data.append(mass_per_dof)
                            else:
                                # Translation in the FX direction
                                M[m+0, m+0] += mass_per_dof

                                # Translation in the FY direction
                                M[m+1, m+1] += mass_per_dof

                                # Translation in the FZ direction
                                M[m+2, m+2] += mass_per_dof

                        else:
                            raise Exception('Direction error: Mass cases should have a direction of "FZ"')

        if sparse:
            # The mass matrix will be stored as a scipy `coo_matrix`. Scipy's
            # documentation states that this type of matrix is ideal for efficient
            # construction of finite element matrices. When converted to another
            # format, the `coo_matrix` sums values at the same (i, j) index.
            from scipy.sparse import coo_matrix
            row = array(row)
            col = array(col)
            data = array(data)
            M = coo_matrix((data, (row, col)), shape=(len(self.Nodes) * 6, len(self.Nodes) * 6))

        # Check that there are no nodal instabilities
        if check_stability:
            if log: print('- Checking nodal stability')
            if sparse:
                Analysis._check_stability(self, M.tocsr())
            else:
                Analysis._check_stability(self, M)

        # Return the global mass matrix
        return M

    def Kg(self, combo_name='Combo 1', log=False, sparse=True):
        """Returns the model's global geometric stiffness matrix. The model must have a static
           solution prior to obtaining the geometric stiffness matrix. Stiffness of plates is not
           included.

        :param combo_name: The name of the load combination to derive the matrix for. Defaults to
                           'Combo 1'.
        :type combo_name: str, optional
        :param log: Prints updates to the console if set to `True`. Defaults to `False`.
        :type log: bool, optional
        :param sparse: Returns a sparse matrix if set to `True`, and a dense matrix otherwise.
                       Defaults to `True`.
        :type sparse: bool, optional
        :return: The global geometric stiffness matrix for the structure.
        :rtype: ndarray or coo_matrix
        """

        if sparse == True:
            # Initialize a zero matrix to hold all the stiffness terms. The matrix will be stored as a
            # scipy sparse `lil_matrix`. This matrix format has several advantages. It uses less memory
            # if the matrix is sparse, supports slicing, and can be converted to other formats (sparse
            # or dense) later on for mathematical operations.
            from scipy.sparse import lil_matrix
            Kg = lil_matrix((len(self.Nodes) * 6, len(self.Nodes) * 6))
        else:
            Kg = zeros(len(self.Nodes) * 6, len(self.Nodes) * 6)

        # Add stiffness terms for each physical member in the model
        if log: print('- Adding member geometric stiffness terms to global geometric stiffness matrix')
        for phys_member in self.Members.values():

            # Check to see if the physical member is active for the given load combination
            if phys_member.active[combo_name] == True:

                # Step through each sub-member in the physical member and add terms
                for member in phys_member.sub_members.values():

                    # Calculate the axial force in the member
                    E = member.E
                    A = member.A
                    L = member.L()
                    d = member.d(combo_name)
                    P = E * A / L * (d[6, 0] - d[0, 0])

                    # Get the member's global stiffness matrix
                    # Storing it as a local variable eliminates the need to rebuild it every time a term is needed
                    member_Kg = member.Kg(P)

                    # Step through each term in the member's stiffness matrix
                    # 'a' & 'b' below are row/column indices in the member's stiffness matrix
                    # 'm' & 'n' are corresponding row/column indices in the global stiffness matrix
                    for a in range(12):

                        # Determine if index 'a' is related to the i-node or j-node
                        if a < 6:
                            # Find the corresponding index 'm' in the global stiffness matrix
                            m = member.i_node.ID * 6 + a
                        else:
                            # Find the corresponding index 'm' in the global stiffness matrix
                            m = member.j_node.ID * 6 + (a - 6)

                        for b in range(12):

                            # Determine if index 'b' is related to the i-node or j-node
                            if b < 6:
                                # Find the corresponding index 'n' in the global stiffness matrix
                                n = member.i_node.ID * 6 + b
                            else:
                                # Find the corresponding index 'n' in the global stiffness matrix
                                n = member.j_node.ID * 6 + (b - 6)

                            # Now that 'm' and 'n' are known, place the term in the global stiffness matrix
                            Kg[m, n] += member_Kg[(a, b)]

        # Return the global geometric stiffness matrix
        return Kg

    def FER(self, combo_name='Combo 1'):
        """Assembles and returns the global fixed end reaction vector for any given load combo.

        :param combo_name: The name of the load combination to get the fixed end reaction vector
                           for. Defaults to 'Combo 1'.
        :type combo_name: str, optional
        :return: The fixed end reaction vector
        :rtype: array
        """

        # Initialize a zero vector to hold all the terms
        FER = zeros((len(self.Nodes) * 6, 1))

        # Step through each physical member in the model
        for phys_member in self.Members.values():

            # Step through each sub-member and add terms
            for member in phys_member.sub_members.values():

                # Get the member's global fixed end reaction vector
                # Storing it as a local variable eliminates the need to rebuild it every time a term is needed
                member_FER = member.FER(combo_name)

                # Step through each term in the member's fixed end reaction vector
                # 'a' below is the row index in the member's fixed end reaction vector
                # 'm' below is the corresponding row index in the global fixed end reaction vector
                for a in range(12):

                    # Determine if index 'a' is related to the i-node or j-node
                    if a < 6:
                        # Find the corresponding index 'm' in the global fixed end reaction vector
                        m = member.i_node.ID * 6 + a
                    else:
                        # Find the corresponding index 'm' in the global fixed end reaction vector
                        m = member.j_node.ID * 6 + (a - 6)

                    # Now that 'm' is known, place the term in the global fixed end reaction vector
                    FER[m, 0] += member_FER[a, 0]

        # Add terms for each rectangle in the model
        for plate in self.Plates.values():

            # Get the quadrilateral's global fixed end reaction vector
            # Storing it as a local variable eliminates the need to rebuild it every time a term is needed
            plate_FER = plate.FER(combo_name)

            # Step through each term in the quadrilateral's fixed end reaction vector
            # 'a' below is the row index in the quadrilateral's fixed end reaction vector
            # 'm' below is the corresponding row index in the global fixed end reaction vector
            for a in range(24):

                # Determine if index 'a' is related to the i-node, j-node, m-node, or n-node
                if a < 6:
                    # Find the corresponding index 'm' in the global fixed end reaction vector
                    m = plate.i_node.ID * 6 + a
                elif a < 12:
                    # Find the corresponding index 'm' in the global fixed end reaction vector
                    m = plate.j_node.ID * 6 + (a - 6)
                elif a < 18:
                    # Find the corresponding index 'm' in the global fixed end reaction vector
                    m = plate.m_node.ID * 6 + (a - 12)
                else:
                    # Find the corresponding index 'm' in the global fixed end reaction vector
                    m = plate.n_node.ID * 6 + (a - 18)

                # Now that 'm' is known, place the term in the global fixed end reaction vector
                FER[m, 0] += plate_FER[a, 0]

        # Add terms for each quadrilateral in the model
        for quad in self.Quads.values():

            # Get the quadrilateral's global fixed end reaction vector
            # Storing it as a local variable eliminates the need to rebuild it every time a term is needed
            quad_FER = quad.FER(combo_name)

            # Step through each term in the quadrilateral's fixed end reaction vector
            # 'a' below is the row index in the quadrilateral's fixed end reaction vector
            # 'm' below is the corresponding row index in the global fixed end reaction vector
            for a in range(24):

                # Determine if index 'a' is related to the i-node, j-node, m-node, or n-node
                if a < 6:
                    # Find the corresponding index 'm' in the global fixed end reaction vector
                    m = quad.m_node.ID * 6 + a
                elif a < 12:
                    # Find the corresponding index 'm' in the global fixed end reaction vector
                    m = quad.n_node.ID * 6 + (a - 6)
                elif a < 18:
                    # Find the corresponding index 'm' in the global fixed end reaction vector
                    m = quad.i_node.ID * 6 + (a - 12)
                else:
                    # Find the corresponding index 'm' in the global fixed end reaction vector
                    m = quad.j_node.ID * 6 + (a - 18)

                # Now that 'm' is known, place the term in the global fixed end reaction vector
                FER[m, 0] += quad_FER[a, 0]

        # Return the global fixed end reaction vector
        return FER

    def P(self, combo_name='Combo 1'):
        """Assembles and returns the global nodal force vector.

        :param combo_name: The name of the load combination to get the force vector for. Defaults
                           to 'Combo 1'.
        :type combo_name: str, optional
        :return: The global nodal force vector.
        :rtype: array
        """

        # Initialize a zero vector to hold all the terms
        P = zeros((len(self.Nodes) * 6, 1))

        # Get the load combination for the given 'combo_name'
        combo = self.LoadCombos[combo_name]

        # Add terms for each node in the model
        for node in self.Nodes.values():

            # Get the node's ID
            ID = node.ID

            # Step through each load factor in the load combination
            for case, factor in combo.factors.items():

                # Add the node's loads to the global nodal load vector
                for load in node.NodeLoads:

                    if load[2] == case:

                        if load[0] == 'FX':
                            P[ID * 6 + 0, 0] += factor * load[1]
                        elif load[0] == 'FY':
                            P[ID * 6 + 1, 0] += factor * load[1]
                        elif load[0] == 'FZ':
                            P[ID * 6 + 2, 0] += factor * load[1]
                        elif load[0] == 'MX':
                            P[ID * 6 + 3, 0] += factor * load[1]
                        elif load[0] == 'MY':
                            P[ID * 6 + 4, 0] += factor * load[1]
                        elif load[0] == 'MZ':
                            P[ID * 6 + 5, 0] += factor * load[1]

        # Return the global nodal force vector
        return P

    def D(self, combo_name='Combo 1'):
        """Returns the global displacement vector for the model.

        :param combo_name: The name of the load combination to get the results for. Defaults to
                           'Combo 1'.
        :type combo_name: str, optional
        :return: The global displacement vector for the model
        :rtype: array
        """

        # Return the global displacement vector
        return self._D[combo_name]

    def _partition(self, unp_matrix, D1_indices, D2_indices):
        """Partitions a matrix (or vector) into submatrices (or subvectors) based on degree of freedom boundary conditions.

        :param unp_matrix: The unpartitioned matrix (or vector) to be partitioned.
        :type unp_matrix: ndarray or lil_matrix
        :param D1_indices: A list of the indices for degrees of freedom that have unknown displacements.
        :type D1_indices: list
        :param D2_indices: A list of the indices for degrees of freedom that have known displacements.
        :type D2_indices: list
        :return: Partitioned submatrices (or subvectors) based on degree of freedom boundary conditions.
        :rtype: array, array, array, array
        """

        # Determine if this is a 1D vector or a 2D matrix

        # 1D vectors
        if unp_matrix.shape[1] == 1:
            # Partition the vector into 2 subvectors
            m1 = unp_matrix[D1_indices, :]
            m2 = unp_matrix[D2_indices, :]
            return m1, m2
        # 2D matrices
        else:
            # Partition the matrix into 4 submatrices
            m11 = unp_matrix[D1_indices, :][:, D1_indices]
            m12 = unp_matrix[D1_indices, :][:, D2_indices]
            m21 = unp_matrix[D2_indices, :][:, D1_indices]
            m22 = unp_matrix[D2_indices, :][:, D2_indices]
            return m11, m12, m21, m22

    def analyze(self, log=False, check_stability=True, check_statics=False, max_iter=30, sparse=True, combo_tags=None):
        """Performs first-order static analysis. Iterations are performed if tension-only members or compression-only members are present.

        :param log: Prints the analysis log to the console if set to True. Default is False.
        :type log: bool, optional
        :param check_stability: When set to `True`, checks for nodal instabilities. This slows down analysis a little. Default is `True`.
        :type check_stability: bool, optional
        :param check_statics: When set to `True`, causes a statics check to be performed
        :type check_statics: bool, optional
        :param max_iter: The maximum number of iterations to try to get convergence for tension/compression-only analysis. Defaults to 30.
        :type max_iter: int, optional
        :param sparse: Indicates whether the sparse matrix solver should be used. A matrix can be considered sparse or dense depening on how many zero terms there are. Structural stiffness matrices often contain many zero terms. The sparse solver can offer faster solutions for such matrices. Using the sparse solver on dense matrices may lead to slower solution times.
        :type sparse: bool, optional
        :raises Exception: _description_
        :raises Exception: _description_
        """

        if log:
            print('+-----------+')
            print('| Analyzing |')
            print('+-----------+')
        
        # Import `scipy` features if the sparse solver is being used
        if sparse == True:
            from scipy.sparse.linalg import spsolve


        # Prepare the model for analysis
        Analysis._prepare_model(self)

"""
        # Ensure there is at least 1 load combination to solve if the user didn't define any
        if self.LoadCombos == {}:
            # Create and add a default load combination to the dictionary of load combinations
            self.LoadCombos['Combo 1'] = LoadCombo('Combo 1', factors={'Case 1': 1.0})

        # Generate all meshes
        for mesh in self.Meshes.values():
            if mesh.is_generated == False:
                mesh.generate()

        # Activate all springs and members for all load combinations
        for spring in self.Springs.values():
            for combo_name in self.LoadCombos.keys():
                spring.active[combo_name] = True

        # Activate all physical members for all load combinations
        for phys_member in self.Members.values():
            for combo_name in self.LoadCombos.keys():
                phys_member.active[combo_name] = True

        # Assign an internal ID to all nodes and elements in the model
        self._renumber() """


        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = self._aux_list()

        # Convert D2 from a list to a vector
        D2 = atleast_2d(D2).T

        # Identify which load combinations to evaluate
        if combo_tags is None:
            combo_list = self.LoadCombos.values()
        else:
            combo_list = []
            for combo in self.LoadCombos.values():
                if any(tag in combo.combo_tags for tag in combo_tags):
                    combo_list.append(combo)

        # Step through each load combination
        for combo in combo_list:

            if log:
                print('')
                print('- Analyzing load combination ' + combo.name)

            # Keep track of the number of iterations
            iter_count = 1
            convergence = False
            divergence = False

            # Iterate until convergence or divergence occurs
            while convergence == False and divergence == False:
                
                # Check for tension/compression-only divergence
                if iter_count > max_iter:
                    divergence = True
                    raise Exception('Model diverged during tension/compression-only analysis')
                
                # Get the partitioned global stiffness matrix K11, K12, K21, K22
                if sparse == True:
                    K11, K12, K21, K22 = self._partition(self.K(combo.name, log, check_stability, sparse).tolil(),
                                                         D1_indices, D2_indices)
                else:
                    K11, K12, K21, K22 = self._partition(self.K(combo.name, log, check_stability, sparse), D1_indices,
                                                         D2_indices)

                # Get the partitioned global fixed end reaction vector
                FER1, FER2 = self._partition(self.FER(combo.name), D1_indices, D2_indices)

                # Get the partitioned global nodal force vector       
                P1, P2 = self._partition(self.P(combo.name), D1_indices, D2_indices)

                # Calculate the global displacement vector
                if log: print('- Calculating global displacement vector')
                if K11.shape == (0, 0):
                    # All displacements are known, so D1 is an empty vector
                    D1 = []
                else:
                    try:
                        # Calculate the unknown displacements D1
                        if sparse == True:
                            # The partitioned stiffness matrix is in `lil` format, which is great
                            # for memory, but slow for mathematical operations. The stiffness
                            # matrix will be converted to `csr` format for mathematical operations.
                            # The `@` operator performs matrix multiplication on sparse matrices.
                            D1 = spsolve(K11.tocsr(), subtract(subtract(P1, FER1), K12.tocsr() @ D2))
                            D1 = D1.reshape(len(D1), 1)
                        else:
                            D1 = solve(K11, subtract(subtract(P1, FER1), matmul(K12, D2)))
                    except:
                        # Return out of the method if 'K' is singular and provide an error message
                        raise Exception(
                            'The stiffness matrix is singular, which implies rigid body motion. The structure is unstable. Aborting analysis.')

                # Form the global displacement vector, D, from D1 and D2
                D = zeros((len(self.Nodes) * 6, 1))

                for node in self.Nodes.values():

                    if D2_indices.count(node.ID * 6 + 0) == 1:
                        D.itemset((node.ID * 6 + 0, 0), D2[D2_indices.index(node.ID * 6 + 0), 0])
                    else:
                        D.itemset((node.ID * 6 + 0, 0), D1[D1_indices.index(node.ID * 6 + 0), 0])

                    if D2_indices.count(node.ID * 6 + 1) == 1:
                        D.itemset((node.ID * 6 + 1, 0), D2[D2_indices.index(node.ID * 6 + 1), 0])
                    else:
                        D.itemset((node.ID * 6 + 1, 0), D1[D1_indices.index(node.ID * 6 + 1), 0])

                    if D2_indices.count(node.ID * 6 + 2) == 1:
                        D.itemset((node.ID * 6 + 2, 0), D2[D2_indices.index(node.ID * 6 + 2), 0])
                    else:
                        D.itemset((node.ID * 6 + 2, 0), D1[D1_indices.index(node.ID * 6 + 2), 0])

                    if D2_indices.count(node.ID * 6 + 3) == 1:
                        D.itemset((node.ID * 6 + 3, 0), D2[D2_indices.index(node.ID * 6 + 3), 0])
                    else:
                        D.itemset((node.ID * 6 + 3, 0), D1[D1_indices.index(node.ID * 6 + 3), 0])

                    if D2_indices.count(node.ID * 6 + 4) == 1:
                        D.itemset((node.ID * 6 + 4, 0), D2[D2_indices.index(node.ID * 6 + 4), 0])
                    else:
                        D.itemset((node.ID * 6 + 4, 0), D1[D1_indices.index(node.ID * 6 + 4), 0])

                    if D2_indices.count(node.ID * 6 + 5) == 1:
                        D.itemset((node.ID * 6 + 5, 0), D2[D2_indices.index(node.ID * 6 + 5), 0])
                    else:
                        D.itemset((node.ID * 6 + 5, 0), D1[D1_indices.index(node.ID * 6 + 5), 0])

                        # Save the global displacement vector
                self._D[combo.name] = D

                # Store the calculated global nodal displacements into each node
                for node in self.Nodes.values():

                    node.DX[combo.name] = D[node.ID*6 + 0, 0]
                    node.DY[combo.name] = D[node.ID*6 + 1, 0]
                    node.DZ[combo.name] = D[node.ID*6 + 2, 0]
                    node.RX[combo.name] = D[node.ID*6 + 3, 0]
                    node.RY[combo.name] = D[node.ID*6 + 4, 0]
                    node.RZ[combo.name] = D[node.ID*6 + 5, 0]
                
                # Check for tension/compression-only convergence
                convergence = Analysis._check_TC_convergence(self, combo.name, log=log)


                if convergence == False:
                    if log: print(
                        '- Tension/compression-only analysis did not converge. Adjusting stiffness matrix and reanalyzing.')
                else:
                    if log: print(
                        '- Tension/compression-only analysis converged after ' + str(iter_count) + ' iteration(s)')

                # Keep track of the number of tension/compression only iterations
                iter_count += 1

        # Calculate reactions
        Analysis._calc_reactions(self, log, combo_tags)

        if log:
            print('')
            print('- Analysis complete')
            print('')

        # Check statics if requested
        if check_statics == True:

            Analysis._check_statics(self, combo_tags)
        

        # Flag the model as solved
        self.solution = 'Linear TC'

    def analyze_linear(self, log=False, check_stability=True, check_statics=False, sparse=True, combo_tags=None):
        """Performs first-order static analysis. This analysis procedure is much faster since it only assembles the global stiffness matrix once, rather than once for each load combination. It is not appropriate when non-linear behavior such as tension/compression only analysis or P-Delta analysis are required.

        :param log: Prints the analysis log to the console if set to True. Default is False.
        :type log: bool, optional
        :param check_stability: When set to True, checks the stiffness matrix for any unstable degrees of freedom and reports them back to the console. This does add to the solution time. Defaults to True.
        :type check_stability: bool, optional
        :param check_statics: When set to True, causes a statics check to be performed. Defaults to False.
        :type check_statics: bool, optional
        :param sparse: Indicates whether the sparse matrix solver should be used. A matrix can be considered sparse or dense depening on how many zero terms there are. Structural stiffness matrices often contain many zero terms. The sparse solver can offer faster solutions for such matrices. Using the sparse solver on dense matrices may lead to slower solution times. Be sure ``scipy`` is installed to use the sparse solver. Default is True.
        :type sparse: bool, optional
        :raises Exception: Occurs when a singular stiffness matrix is found. This indicates an unstable structure has been modeled.
        """

        if log:
            print('+-------------------+')
            print('| Analyzing: Linear |')
            print('+-------------------+')
        
        # Import `scipy` features if the sparse solver is being used
        if sparse == True:
            from scipy.sparse.linalg import spsolve


        # Prepare the model for analysis
        Analysis._prepare_model(self)
"""
        # Ensure there is at least 1 load combination to solve if the user didn't define any
        if self.LoadCombos == {}:
            # Create and add a default load combination to the dictionary of load combinations
            self.LoadCombos['Combo 1'] = LoadCombo('Combo 1', factors={'Case 1': 1.0})

        # Generate all meshes
        for mesh in self.Meshes.values():
            if mesh.is_generated == False:
                mesh.generate()

        # Activate all springs for all load combinations
        for spring in self.Springs.values():
            for combo_name in self.LoadCombos.keys():
                spring.active[combo_name] = True

        # Activate all physical members for all load combinations
        for phys_member in self.Members.values():
            for combo_name in self.LoadCombos.keys():
                phys_member.active[combo_name] = True

        # Assign an internal ID to all nodes and elements in the model
        self._renumber()"""


        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = self._aux_list()

        # Convert D2 from a list to a vector
        D2 = atleast_2d(D2).T

        # Get the partitioned global stiffness matrix K11, K12, K21, K22
        # Note that for linear analysis the stiffness matrix can be obtained for any load combination, as it's the same for all of them
        combo_name = list(self.LoadCombos.keys())[0]
        if sparse == True:
            K11, K12, K21, K22 = self._partition(self.K(combo_name, log, check_stability, sparse).tolil(), D1_indices,
                                                 D2_indices)
        else:
            K11, K12, K21, K22 = self._partition(self.K(combo_name, log, check_stability, sparse), D1_indices,
                                                 D2_indices)

        # Identify which load combinations to evaluate
        if combo_tags is None:
            combo_list = self.LoadCombos.values()
        else:
            combo_list = []
            for combo in self.LoadCombos.values():
                if any(tag in combo.combo_tags for tag in combo_tags):
                    combo_list.append(combo)

        # Step through each load combination
        for combo in combo_list:

            if log:
                print('')
                print('- Analyzing load combination ' + combo.name)

            # Get the partitioned global fixed end reaction vector
            FER1, FER2 = self._partition(self.FER(combo.name), D1_indices, D2_indices)

            # Get the partitioned global nodal force vector       
            P1, P2 = self._partition(self.P(combo.name), D1_indices, D2_indices)

            # Calculate the global displacement vector
            if log: print('- Calculating global displacement vector')
            if K11.shape == (0, 0):
                # All displacements are known, so D1 is an empty vector
                D1 = []
            else:
                try:
                    # Calculate the unknown displacements D1
                    if sparse == True:
                        # The partitioned stiffness matrix is in `lil` format, which is great
                        # for memory, but slow for mathematical operations. The stiffness
                        # matrix will be converted to `csr` format for mathematical operations.
                        # The `@` operator performs matrix multiplication on sparse matrices.
                        D1 = spsolve(K11.tocsr(), subtract(subtract(P1, FER1), K12.tocsr() @ D2))
                        D1 = D1.reshape(len(D1), 1)
                    else:
                        D1 = solve(K11, subtract(subtract(P1, FER1), matmul(K12, D2)))
                except:
                    # Return out of the method if 'K' is singular and provide an error message
                    raise Exception(
                        'The stiffness matrix is singular, which implies rigid body motion. The structure is unstable. Aborting analysis.')

            # Form the global displacement vector, D, from D1 and D2
            D = zeros((len(self.Nodes) * 6, 1))

            for node in self.Nodes.values():

                if D2_indices.count(node.ID * 6 + 0) == 1:
                    D.itemset((node.ID * 6 + 0, 0), D2[D2_indices.index(node.ID * 6 + 0), 0])
                else:
                    D.itemset((node.ID * 6 + 0, 0), D1[D1_indices.index(node.ID * 6 + 0), 0])

                if D2_indices.count(node.ID * 6 + 1) == 1:
                    D.itemset((node.ID * 6 + 1, 0), D2[D2_indices.index(node.ID * 6 + 1), 0])
                else:
                    D.itemset((node.ID * 6 + 1, 0), D1[D1_indices.index(node.ID * 6 + 1), 0])

                if D2_indices.count(node.ID * 6 + 2) == 1:
                    D.itemset((node.ID * 6 + 2, 0), D2[D2_indices.index(node.ID * 6 + 2), 0])
                else:
                    D.itemset((node.ID * 6 + 2, 0), D1[D1_indices.index(node.ID * 6 + 2), 0])

                if D2_indices.count(node.ID * 6 + 3) == 1:
                    D.itemset((node.ID * 6 + 3, 0), D2[D2_indices.index(node.ID * 6 + 3), 0])
                else:
                    D.itemset((node.ID * 6 + 3, 0), D1[D1_indices.index(node.ID * 6 + 3), 0])

                if D2_indices.count(node.ID * 6 + 4) == 1:
                    D.itemset((node.ID * 6 + 4, 0), D2[D2_indices.index(node.ID * 6 + 4), 0])
                else:
                    D.itemset((node.ID * 6 + 4, 0), D1[D1_indices.index(node.ID * 6 + 4), 0])

                if D2_indices.count(node.ID * 6 + 5) == 1:
                    D.itemset((node.ID * 6 + 5, 0), D2[D2_indices.index(node.ID * 6 + 5), 0])
                else:
                    D.itemset((node.ID * 6 + 5, 0), D1[D1_indices.index(node.ID * 6 + 5), 0])

                    # Save the global displacement vector
            self._D[combo.name] = D

            # Store the calculated global nodal displacements into each node
            for node in self.Nodes.values():
                node.DX[combo.name] = D[node.ID * 6 + 0, 0]
                node.DY[combo.name] = D[node.ID * 6 + 1, 0]
                node.DZ[combo.name] = D[node.ID * 6 + 2, 0]
                node.RX[combo.name] = D[node.ID * 6 + 3, 0]
                node.RY[combo.name] = D[node.ID * 6 + 4, 0]
                node.RZ[combo.name] = D[node.ID * 6 + 5, 0]

        # Calculate reactions
        Analysis._calc_reactions(self, log, combo_tags)

        if log:
            print('')
            print('- Analysis complete')
            print('')

        # Check statics if requested
        if check_statics == True:

            Analysis._check_statics(self, combo_tags)


        # Flag the model as solved
        self.solution = 'Linear'

    def analyze_PDelta(self, log=False, check_stability=True, max_iter=30, tol=0.01, sparse=True, combo_tags=None):
        """Performs second order (P-Delta) analysis. This type of analysis is appropriate for most models using beams, columns and braces. Second order analysis is usually required by material specific codes. The analysis is iterative and takes longer to solve. Models with slender members and/or members with combined bending and axial loads will generally have more significant P-Delta effects. P-Delta effects in plates/quads are not considered.

        :param log: Prints updates to the console if set to True. Default is False.
        :type log: bool, optional
        :param check_stability: When set to True, checks the stiffness matrix for any unstable degrees of freedom and reports them back to the console. This does add to the solution time. Defaults to True.
        :type check_stability: bool, optional
        :param max_iter: The maximum number of iterations permitted. If this value is exceeded the program will report divergence. Defaults to 30.
        :type max_iter: int, optional
        :param tol: The deflection tolerance (as a percentage) between iterations that will be used to define whether the model has converged (e.g. 0.01 = deflections must converge within 1% between iterations).
        :type tol: float, optional
        :param sparse: Indicates whether the sparse matrix solver should be used. A matrix can be considered sparse or dense depening on how many zero terms there are. Structural stiffness matrices often contain many zero terms. The sparse solver can offer faster solutions for such matrices. Using the sparse solver on dense matrices may lead to slower solution times. Be sure ``scipy`` is installed to use the sparse solver. Default is True.
        :type sparse: bool, optional
        :raises ValueError: Occurs when there is a singularity in the stiffness matrix, which indicates an unstable structure.
        :raises Exception: Occurs when a model fails to converge.
        """

        if log:
            print('+--------------------+')
            print('| Analyzing: P-Delta |')
            print('+--------------------+')

        # Import `scipy` features if the sparse solver is being used
        if sparse == True:
            from scipy.sparse.linalg import spsolve


        # Prepare the model for analysis
        Analysis._prepare_model(self)
        
 """
        # Ensure there is at least 1 load combination to solve if the user didn't define any
        if self.LoadCombos == {}:
            # Create and add a default load combination to the dictionary of load combinations
            self.LoadCombos['Combo 1'] = LoadCombo('Combo 1', factors={'Case 1': 1.0})

        # Generate all meshes
        for mesh in self.Meshes.values():
            if mesh.is_generated == False:
                mesh.generate()

        # Activate all springs for all load combinations. They can be turned inactive
        # during the course of the tension/compression-only analysis
        for spring in self.Springs.values():
            for combo_name in self.LoadCombos.keys():
                spring.active[combo_name] = True

        # Activate all physical members for all load combinations
        for phys_member in self.Members.values():
            for combo_name in self.LoadCombos.keys():
                phys_member.active[combo_name] = True

        # Assign an internal ID to all nodes and elements in the model
        self._renumber()
"""

        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = self._aux_list()

        # Convert D2 from a list to a matrix
        D2 = array(D2, ndmin=2).T

        # Identify which load combinations to evaluate
        if combo_tags is None:
            combo_list = self.LoadCombos.values()
        else:
            combo_list = []
            for combo in self.LoadCombos.values():
                if any(tag in combo.combo_tags for tag in combo_tags):
                    combo_list.append(combo)

        # Step through each load combination

        for combo in combo_list:

            if log:
                print('')
                print('- Analyzing load combination ' + combo.name)

            iter_count_TC = 1  # Tracks tension/compression-only iterations
            iter_count_PD = 1  # Tracks P-Delta iterations
            prev_results = None  # Used to store results from the previous iteration

            convergence_TC = False  # Tracks tension/compression-only convergence
            convergence_PD = False  # Tracks P-Delta convergence

            divergence_TC = False  # Tracks tension/compression-only divergence
            divergence_PD = False  # Tracks P-Delta divergence

            # Iterate until convergence or divergence occurs
            while ((convergence_TC == False or convergence_PD == False)
                   and (divergence_TC == False and divergence_PD == False)):

                # Inform the user which iteration we're on
                if log:
                    print('- Beginning tension/compression-only iteration #' + str(iter_count_TC))
                    print('- Beginning P-Delta iteration #' + str(iter_count_PD))

                # On the first iteration, get all the partitioned global matrices
                if iter_count_PD == 1:

                    if sparse == True:
                        K11, K12, K21, K22 = self._partition(self.K(combo.name, log, check_stability, sparse).tolil(),
                                                             D1_indices, D2_indices)  # Initial stiffness matrix
                    else:
                        K11, K12, K21, K22 = self._partition(self.K(combo.name, log, check_stability, sparse),
                                                             D1_indices, D2_indices)  # Initial stiffness matrix

                    # Check that the structure is stable
                    if log: print('- Checking stability')
                    Analysis._check_stability(self, K11)

                    # Assemble the force matrices
                    FER1, FER2 = self._partition(self.FER(combo.name), D1_indices, D2_indices)  # Fixed end reactions
                    P1, P2 = self._partition(self.P(combo.name), D1_indices, D2_indices)  # Nodal forces

                # On subsequent iterations, recalculate the stiffness matrix to account for P-Delta
                # effects
                else:

                    # Calculate the partitioned global stiffness matrices
                    if sparse == True:

                        K11, K12, K21, K22 = self._partition(self.K(combo.name, log, check_stability, sparse).tolil(),
                                                             D1_indices, D2_indices)  # Initial stiffness matrix
                        Kg11, Kg12, Kg21, Kg22 = self._partition(self.Kg(combo.name, log, sparse), D1_indices,
                                                                 D2_indices)  # Geometric stiffness matrix

                        # The stiffness matrices are currently `lil` format which is great for
                        # memory, but slow for mathematical operations. They will be converted to
                        # `csr` format. The `+` operator performs matrix addition on `csr`
                        # matrices.
                        K11 = K11.tocsr() + Kg11.tocsr()
                        K12 = K12.tocsr() + Kg12.tocsr()
                        K21 = K21.tocsr() + Kg21.tocsr()
                        K22 = K22.tocsr() + Kg22.tocsr()

                    else:

                        K11, K12, K21, K22 = self._partition(self.K(combo.name, log, check_stability, sparse),
                                                             D1_indices, D2_indices)  # Initial stiffness matrix
                        Kg11, Kg12, Kg21, Kg22 = self._partition(self.Kg(combo.name, log, sparse), D1_indices,
                                                                 D2_indices)  # Geometric stiffness matrix

                        K11 = K11 + Kg11
                        K12 = K12 + Kg12
                        K21 = K21 + Kg21
                        K22 = K22 + Kg22

                # Calculate the global displacement vector
                if log: print('- Calculating the global displacement vector')
                if K11.shape == (0, 0):
                    # All displacements are known, so D1 is an empty vector
                    D1 = []
                else:
                    try:
                        # Calculate the unknown displacements D1
                        if sparse == True:
                            # The partitioned stiffness matrix is already in `csr` format. The `@`
                            # operator performs matrix multiplication on sparse matrices.
                            D1 = spsolve(K11.tocsr(), subtract(subtract(P1, FER1), K12.tocsr() @ D2))
                            D1 = D1.reshape(len(D1), 1)
                        else:
                            # The partitioned stiffness matrix is in `csr` format. It will be
                            # converted to a 2D dense array for mathematical operations.
                            D1 = solve(K11, subtract(subtract(P1, FER1), matmul(K12, D2)))

                    except:
                        # Return out of the method if 'K' is singular and provide an error message
                        raise ValueError(
                            'The stiffness matrix is singular, which implies rigid body motion. The structure is unstable. Aborting analysis.')

                D = zeros((len(self.Nodes) * 6, 1))

                for node in self.Nodes.values():

                    if node.ID * 6 + 0 in D2_indices:
                        D[(node.ID * 6 + 0, 0)] = D2[D2_indices.index(node.ID * 6 + 0), 0]
                    else:
                        D[(node.ID * 6 + 0, 0)] = D1[D1_indices.index(node.ID * 6 + 0), 0]

                    if node.ID * 6 + 1 in D2_indices:
                        D[(node.ID * 6 + 1, 0)] = D2[D2_indices.index(node.ID * 6 + 1), 0]
                    else:
                        D[(node.ID * 6 + 1, 0)] = D1[D1_indices.index(node.ID * 6 + 1), 0]

                    if node.ID * 6 + 2 in D2_indices:
                        D[(node.ID * 6 + 2, 0)] = D2[D2_indices.index(node.ID * 6 + 2), 0]
                    else:
                        D[(node.ID * 6 + 2, 0)] = D1[D1_indices.index(node.ID * 6 + 2), 0]

                    if node.ID * 6 + 3 in D2_indices:
                        D[(node.ID * 6 + 3, 0)] = D2[D2_indices.index(node.ID * 6 + 3), 0]
                    else:
                        D[(node.ID * 6 + 3, 0)] = D1[D1_indices.index(node.ID * 6 + 3), 0]

                    if node.ID * 6 + 4 in D2_indices:
                        D[(node.ID * 6 + 4, 0)] = D2[D2_indices.index(node.ID * 6 + 4), 0]
                    else:
                        D[(node.ID * 6 + 4, 0)] = D1[D1_indices.index(node.ID * 6 + 4), 0]

                    if node.ID * 6 + 5 in D2_indices:
                        D[(node.ID * 6 + 5, 0)] = D2[D2_indices.index(node.ID * 6 + 5), 0]
                    else:
                        D[(node.ID * 6 + 5, 0)] = D1[D1_indices.index(node.ID * 6 + 5), 0]

                # Save the global displacement vector
                self._D[combo.name] = D

                # Store the calculated global nodal displacements into each node
                for node in self.Nodes.values():
                    node.DX[combo.name] = D[node.ID * 6 + 0, 0]
                    node.DY[combo.name] = D[node.ID * 6 + 1, 0]
                    node.DZ[combo.name] = D[node.ID * 6 + 2, 0]
                    node.RX[combo.name] = D[node.ID * 6 + 3, 0]
                    node.RY[combo.name] = D[node.ID * 6 + 4, 0]
                    node.RZ[combo.name] = D[node.ID * 6 + 5, 0]

                # Assume the model has converged (to be checked below)

                convergence_TC = Analysis._check_TC_convergence(self, combo.name, log)
                

                # Report on convergence of tension/compression only analysis
                if convergence_TC == False:

                    if log:
                        print('- Tension/compression-only analysis did not converge on this iteration')
                        print('- Stiffness matrix will be adjusted')
                        print('- P-Delta analysis will be restarted')

                    # Increment the tension/compression-only iteration count
                    iter_count_TC += 1

                    # Reset the P-Delta analysis since the T/C analysis didn't converge
                    convergence_PD = False
                    iter_count_PD = 0

                else:
                    if log: print(
                        '- Tension/compression-only analysis converged after ' + str(iter_count_TC) + ' iteration(s)')

                # Check for divergence in the tension/compression-only analysis
                if iter_count_TC > max_iter:
                    divergence_TC = True
                    raise Exception('- Model diverged during tension/compression-only analysis')

                # Check for P-Delta convergence
                if iter_count_PD > 1:

                    # Print a status update for the user
                    if log: print('- Checking for P-Delta convergence')

                    # Temporarily disable error messages for invalid values.
                    # We'll be dealing with some 'nan' values due to division by zero at supports with zero deflection.
                    seterr(invalid='ignore')

                    # Check for convergence
                    # Note: if the shape of K11 is (0, 0) then all degrees of freedom are fully
                    # restrained, and P-Delta analysis automatically converges
                    if K11.shape == (0, 0) or abs(nanmax(divide(D1, prev_results)) - 1) <= tol:
                        convergence_PD = True
                        if log: print('- P-Delta analysis converged after ' + str(iter_count_PD) + ' iteration(s)')
                    # Check for divergence
                    elif iter_count_PD > max_iter:
                        divergence_PD = True
                        if log: print('- P-Delta analysis failed to converge after ' + str(max_iter) + ' iteration(s)')

                    # Turn invalid value warnings back on
                    seterr(invalid='warn')

                    # Save the results for the next iteration
                prev_results = D1

                # Increment the P-Delta iteration count
                iter_count_PD += 1

        # Calculate reactions
        Analysis._calc_reactions(self, log, combo_tags)

        if log:
            print('')
            print('- Analysis complete')
            print('')

        # Flag the model as solved
        self.solution = 'P-Delta'
   def analyze_modal(self, log=False, check_stability=True, num_modes=1, tol=0.01, sparse=True,
                      type_of_mass_matrix = 'consistent'):
        """Performs modal analysis.

        :param log: Prints the analysis log to the console if set to True. Default is False.
        :type log: bool, optional
        :param check_stability: When set to True, checks the stiffness matrix for any unstable degrees of freedom and reports them back to the console. This does add to the solution time. Defaults to True.
        :type check_stability: bool, optional
        :para num_modes: The number of modes required
        :type num_modes: int, optional
        :para tol: The required accuracy in the results
        :type tol: float, optional
        :param sparse: Indicates whether the sparse matrix solver should be used. A matrix can be considered sparse or dense depening on how many zero terms there are. Structural stiffness matrices often contain many zero terms. The sparse solver can offer faster solutions for such matrices. Using the sparse solver on dense matrices may lead to slower solution times. Be sure ``scipy`` is installed to use the sparse solver. Default is True.
        :type sparse: bool, optional
        :param type_of_mass_matrix: The type of element mass matrix to use in the analysis
        :type type_of_mass_matrix: str, optional
        :raises Exception: Occurs when a singular stiffness matrix is found. This indicates an unstable structure has been modeled.
        """

        if log:
            print('+-------------------+')
            print('| Analyzing: Modal  |')
            print('+-------------------+')

        # Import `scipy` features if the sparse solver is being used
        if sparse == True:
            from scipy.sparse.linalg import spsolve

        # Add a modal load combination if not present
        if 'Modal Combo' not in self.LoadCombos:
            self.LoadCombos['Modal Combo'] = LoadCombo('Modal Combo', factors={'Modal Case': 0})

        # Generate all meshes
        for mesh in self.Meshes.values():
            if mesh.is_generated == False:
                mesh.generate()

        # Activate all springs
        for spring in self.Springs.values():
            spring.active['Modal Combo'] = True

        # Activate all physical members
        for phys_member in self.Members.values():
            phys_member.active['Modal Combo'] = True

        # Assign an internal ID to all nodes and elements in the model
        self._renumber()

        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = self._aux_list()

        # In the context of mode shapes, D2 should just be zeroes
        D2 = zeros((len(D2), 1))
        # Get the partitioned global stiffness and mass matrix
        combo_name = "Modal Combo"
        if sparse == True:
            K11, K12, K21, K22 = self._partition(self.K(combo_name, log, check_stability, sparse).tolil(), D1_indices,
                                                 D2_indices)
            # We will not check for stability of the mass matrix. check_stability will be set to False
            # This is because for the shell elements, the mass matrix has zeroes
            # on the rotation about z-axis DOFs
            # Only the stiffness matrix is modified to account for this 'drilling' effect
            # ref: Boutagouga, D., & Djeghaba, K. (2016). Nonlinear dynamic co-rotational
            # formulation for membrane elements with in-plane drilling rotational degree of freedom. Engineering Computations, 33(3).
            M11, M12, M21, M22 = self._partition(self.M(combo_name, log, False, sparse,type_of_mass_matrix).tolil(), D1_indices, D2_indices)
        else:
            K11, K12, K21, K22 = self._partition(self.K(combo_name, log, check_stability, sparse), D1_indices,
                                                 D2_indices)
            M11, M12, M21, M22 = self._partition(self.M(combo_name, log, False, sparse, type_of_mass_matrix), D1_indices, D2_indices)

        if log:
            print('')
            print('- Calculating modes ')

        eigVal = None  # Vector to store eigenvalues
        eigVec = None  # Matrix to store eigenvectors

        if K11.shape == (0, 0):
            if log: print('The model does not have any degree of freedom')
        elif num_modes < 1:
            raise Exception("The model does not have any degree of freedom")
        else:
            try:
                if sparse == True:
                    # The partitioned matrices are in `lil` format, which is great
                    # for memory, but slow for mathematical operations. The stiffness
                    # matrix will be converted to `csr` format for mathematical operations.
                    if num_modes == K11.shape[0]:
                        # If all mode shapes are required, the matrices are converted to dense
                        # and format in order to use eig(), the structure is probably small.
                        from scipy.linalg import eig
                        eigVal, eigVec = eig(a=K11.tocsr().toarray(), b=M11.tocsr().toarray())

                    else:
                        # Calculate only the first num_modes modes.
                        eigVal, eigVec = eigs(tol=tol, A=K11.tocsr(), k=num_modes, M=M11.tocsr(), sigma=-1)

                else:
                    if num_modes == K11.shape[0]:
                        # If all mode shapes are required, the matrices are converted to dense
                        # and format in order to use eig(), the structure is probably small.
                        from scipy.linalg import eig
                        eigVal, eigVec = eig(a=K11.tocsr().toarray(), b=M11.tocsr().toarray())
                    else:
                        # To calculate only some modes, convert to sparse and use eigs()
                        eigVal, eigVec = eigs(tol=tol, A=csr_matrix(K11), k=num_modes, M=csr_matrix(M11), sigma=-1)
            except:
                raise Exception(
                    'The stiffness matrix is singular, which implies rigid body motion. The structure is unstable. Aborting analysis.')

        # The functions used to calculate the eigenvalues and eigenvectors are iterative
        # Hence they have a tendence to return complex numbers even though we do not expect
        # results of that nature in simple modal analysis
        # The complex parts of the results are very small, so we will only extract the real part

        eigVal = real(eigVal)
        eigVec = real(eigVec)

        # Sort the eigenvalues to start from the lowest
        sort_indices = argsort(eigVal)
        eigVal = eigVal[sort_indices]

        # Use the same order from above to sort the corresponding eigenvectors
        eigVec = eigVec[:, sort_indices]

        # Calculate and store the natural frequencies
        self.Natural_Frequencies = array([sqrt(eig_val) / (2 * pi) for eig_val in eigVal])

        # Store the calculated modal displacements
        self._SHAPE = real(eigVec)

        # Form the global displacement vector, D, from D1 and D2
        D1 = eigVec[:, self.Active_Mode - 1].reshape((-1, 1))
        D = zeros((len(self.Nodes) * 6, 1))

        for node in self.Nodes.values():

            if D2_indices.count(node.ID * 6 + 0) == 1:
                D.itemset((node.ID * 6 + 0, 0), D2[D2_indices.index(node.ID * 6 + 0), 0])
            else:
                D.itemset((node.ID * 6 + 0, 0), D1[D1_indices.index(node.ID * 6 + 0), 0])

            if D2_indices.count(node.ID * 6 + 1) == 1:
                D.itemset((node.ID * 6 + 1, 0), D2[D2_indices.index(node.ID * 6 + 1), 0])
            else:
                D.itemset((node.ID * 6 + 1, 0), D1[D1_indices.index(node.ID * 6 + 1), 0])

            if D2_indices.count(node.ID * 6 + 2) == 1:
                D.itemset((node.ID * 6 + 2, 0), D2[D2_indices.index(node.ID * 6 + 2), 0])
            else:
                D.itemset((node.ID * 6 + 2, 0), D1[D1_indices.index(node.ID * 6 + 2), 0])

            if D2_indices.count(node.ID * 6 + 3) == 1:
                D.itemset((node.ID * 6 + 3, 0), D2[D2_indices.index(node.ID * 6 + 3), 0])
            else:
                D.itemset((node.ID * 6 + 3, 0), D1[D1_indices.index(node.ID * 6 + 3), 0])

            if D2_indices.count(node.ID * 6 + 4) == 1:
                D.itemset((node.ID * 6 + 4, 0), D2[D2_indices.index(node.ID * 6 + 4), 0])
            else:
                D.itemset((node.ID * 6 + 4, 0), D1[D1_indices.index(node.ID * 6 + 4), 0])

            if D2_indices.count(node.ID * 6 + 5) == 1:
                D.itemset((node.ID * 6 + 5, 0), D2[D2_indices.index(node.ID * 6 + 5), 0])
            else:
                D.itemset((node.ID * 6 + 5, 0), D1[D1_indices.index(node.ID * 6 + 5), 0])

        # Store the calculated global nodal modal displacements into each node
        for node in self.Nodes.values():
            node.DX[combo_name] = D[node.ID * 6 + 0, 0]
            node.DY[combo_name] = D[node.ID * 6 + 1, 0]
            node.DZ[combo_name] = D[node.ID * 6 + 2, 0]
            node.RX[combo_name] = D[node.ID * 6 + 3, 0]
            node.RY[combo_name] = D[node.ID * 6 + 4, 0]
            node.RZ[combo_name] = D[node.ID * 6 + 5, 0]

            # return eigVec[:,0].reshape((-1,1))
        if log:
            print('')
            print('- Analysis complete')
            print('')

        # Flag the model as solved
        self.solution = 'Modal'


    def analyze_harmonic(self, harmonic_combo, f1, f2, f_div, num_modes, constant_modal_damping = 0.02,
                         rayleigh_alpha_1 = None, rayleigh_alpha_2 = None, first_mode_damping_ratio = None,
                         highest_mode_damping_ratio = None, damping_ratios_in_every_mode = None,
                         static_combo=None, log=False, check_stability=True, check_statics=False, tol=0.01,
                         sparse=True, type_of_mass_matrix = 'consistent'):
        """Performs harmonic analysis for given harmonic load combination and load frequency. It begins by performing a modal analysis followed by
        a harmonic analysis. If specified, a static linear analysis will also be performed and the results will be superimposed with those from
        harmonic analysis


        :param harmonic_combo: The harmonic load combination
        :type harmonic_combo: LoadCombo
        :param f1: The lowest forcing frequency to consider
        :type f1: float
        :param f2: The highest forcing frequency to consider
        :type f2: float
        :param f_div: The number of frequencies in the range to compute for
        :type f_div: int
        :param num_modes: The number of modes to use in the analysis
        :type num_modes: int
        :param constant_modal_damping: A constant damping ratio to be used in all modes. E.g 0.02 to mean 2%
        :type constant_modal_damping: float, optional
        :param rayleigh_alpha_1: The rayleigh damping coefficient corresponding to mass proportionate damping
        :type rayleigh_alpha_1: float, optional
        :param rayleigh_alpha_2: The rayleigh damping coefficient corresponding to stiffness proportionate damping
        :type rayleigh_alpha_2: float, optional
        :param first_mode_damping_ratio: Damping ratio in the first mode. If provided, it will be used to calculate the
            rayleigh damping coefficients.
        :type first_mode_damping_ratio: float, optional
        :param highest_mode_damping_ratio: Damping ratio in the highest mode. If provided, it will be used to calculate
            the rayleigh damping coefficients
        :type highest_mode_damping_ratio: float, optional
        :param damping_ratios_in_every_mode: A list of damping ratios specified for every mode
        :type damping_ratios_in_every_mode: list, tuple, optional
        :param static_combo: The static load combination. For example self weight
        :type static_combo: LoadCombo
        :param log: Prints the analysis log to the console if set to True. Default is False.
        :type log: bool, optional
        :param check_stability: When set to True, checks the stiffness matrix for any unstable degrees of freedom and reports them back to the console. This does add to the solution time. Defaults to True.
        :type check_stability: bool, optional
        :param check_statics: When set to True, causes a statics check to be performed. Defaults to False.
        :type check_statics: bool, optional
        :para tol: The required accuracy in the results
        :type tol: float, optional
        :param sparse: Indicates whether the sparse matrix solver should be used. A matrix can be considered sparse or dense depening on how many zero terms there are. Structural stiffness matrices often contain many zero terms. The sparse solver can offer faster solutions for such matrices. Using the sparse solver on dense matrices may lead to slower solution times. Be sure ``scipy`` is installed to use the sparse solver. Default is True.
        :type sparse: bool, optional
        :param type_of_mass_matrix: The type of element mass matrix to use in the analysis
        :type type_of_mass_matrix: str, optional
        :raises Exception: Occurs when a singular stiffness matrix is found. This indicates an unstable structure has been modeled.

        """

        # Check frequency
        if f1 < 0 or f2 < 0 or f_div < 0:
            raise ValueError("f1, f2 and f_div must be positive")

        if f2 < f1:
            raise ValueError("f2 must be greater than f1")

        if f_div == 1:
            raise ValueError("f_div must be atleast 2")

        # Perform modal analysis
        self.analyze_modal(log, check_stability, num_modes, tol, sparse, type_of_mass_matrix)

        # Perform static linear analysis if requested for
        if static_combo != None:
            # We do not want to perform static analysis for all the load combinations
            # Hence we will keep the load combinations in a temporary object
            load_combos_temp = copy.deepcopy(self.LoadCombos)

            # Then remove all other load combos except the required load combo
            self.LoadCombos.clear()
            self.LoadCombos = {static_combo: load_combos_temp[static_combo]}

            # Perform the analysis
            self.analyze_linear(log, check_stability, check_statics, sparse)

            # Restore the load combos
            self.LoadCombos = copy.deepcopy(load_combos_temp)

            # Delete the temp dictionary
            del load_combos_temp

        # At this point, we have the mode shapes, natural frequencies, and static displacement results stored in
        # self._SHAPE, self.Natural_Frequencies, and self._D respectively

        # We can now begin the harmonic analysis
        if log:
            print('+--------------------+')
            print('| Analyzing: Harmonic|')
            print('+--------------------+')

        # Import `scipy` features if the sparse solver is being used
        if sparse == True:
            from scipy.sparse.linalg import spsolve

        # Activate all springs for the harmonic load combination

        for spring in self.Springs.values():
            spring.active[harmonic_combo] = True

        # Activate all physical members for the harmonic load combination
        for phys_member in self.Members.values():
            phys_member.active[harmonic_combo] = True

        # Assign an internal ID to all nodes and elements in the model
        self._renumber()

        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = self._aux_list()

        # Convert D2 from a list to a vector
        D2 = atleast_2d(D2).T

        # Get the partitioned global stiffness matrix K11, K12, K21, K22
        if sparse == True:
            K11, K12, K21, K22 = self._partition(self.K(harmonic_combo, log, check_stability, sparse).tolil(),
                                                 D1_indices, D2_indices)
            # We will not check for stability of the mass matrix. check_stability will be set to False
            # This is because for the shell elements, the mass matrix has zeroes
            # on the rotation about z-axis DOFs
            # Only the stiffness matrix is modified to account for this 'drilling' effect
            # ref: Boutagouga, D., & Djeghaba, K. (2016). Nonlinear dynamic co-rotational
            # formulation for membrane elements with in-plane drilling rotational degree of freedom. Engineering Computations, 33(3).
            M11, M12, M21, M22 = self._partition(self.M(harmonic_combo, log, False, sparse,type_of_mass_matrix).tolil(), D1_indices,
                                                 D2_indices)
        else:
            K11, K12, K21, K22 = self._partition(self.K(harmonic_combo, log, check_stability, sparse), D1_indices,
                                                 D2_indices)
            M11, M12, M21, M22 = self._partition(self.M(harmonic_combo, log, False, sparse, type_of_mass_matrix), D1_indices, D2_indices)

        # Get the mass normalised mode shape matrix
        Z = self._mass_normalised_mode_shapes(M11, self._SHAPE)

        # Get the partitioned global fixed end reaction vector
        FER1, FER2 = self._partition(self.FER(harmonic_combo), D1_indices, D2_indices)

        # Get the partitioned global nodal force vector
        P1, P2 = self._partition(self.P(harmonic_combo), D1_indices, D2_indices)

        # Calculate the normalised force vector
        FV_n = Z.T @ subtract(P1,FER1)

        # Initialise vectors to hold the modal displacements coordinates
        Q = zeros((FV_n.shape[0], 1))

        # Calculate the damping coefficients
        w = 2 * pi * self.Natural_Frequencies  # Angular natural frequencies

        # Calculate the damping matrix
        # Initialise it
        C_n = zeros(FV_n.shape[0])
        if damping_ratios_in_every_mode != None:
            # Declare new variable called ratios, easier to work with than the original
            ratios = damping_ratios_in_every_mode
            # Check if it is a list or turple
            if isinstance(ratios, (list, tuple)):
                # Calculate the modal damping coefficient for each mode
                # If too many damping ratios have been provided, only the first entries
                # corresponding to the requested modes will be used
                # If few ratio have been provided, the last ratio will be used for the rest
                # of the modes
                for k in range(len(w)):
                    C_n[k] = 2 * w[k] * ratios[min(k,len(ratios)-1)]
            else:
                # The provided input is perhaps a just a number, not a list
                # That number will be used for all the modes
                for k in range(len(w)):
                   C_n[k] = 2 * damping_ratios_in_every_mode * w[k]
        elif rayleigh_alpha_1 != None or rayleigh_alpha_2 != None:
            # Atleast one rayleigh damping coefficient has been specified
            if rayleigh_alpha_1 == None:
                rayleigh_alpha_1 = 0
            if rayleigh_alpha_2 == None:
                rayleigh_alpha_2 = 0
            for k in range(len(w)):
                C_n[k] = rayleigh_alpha_1 + rayleigh_alpha_2 * w[k] ** 2

        elif first_mode_damping_ratio != None or highest_mode_damping_ratio != None:
            # Rayleigh damping is requested and at-least one damping ratio is given
            # If only one is given, the same will be assumed to the damping ratios
            # in the lowest and heighest modes
            if first_mode_damping_ratio == None:
                first_mode_damping_ratio = highest_mode_damping_ratio
            if highest_mode_damping_ratio == None:
                highest_mode_damping_ratio = first_mode_damping_ratio

            # Calculate the rayleigh damping coefficients
            # Create new shorter variables
            ratio1 = first_mode_damping_ratio
            ratio2 = highest_mode_damping_ratio

            # Extract the first and last angular frequencies
            w1 = w[0] # Angular frequency of first mode
            w2 = w[-1] # Angular frequency of last mode

            # Calculate the rayleigh damping coefficients
            alpha1 = 2 * w1 * w2 * (w2 * ratio1 - w1 * ratio2) / (w2 ** 2 - w1 ** 2)
            alpha2 = 2 * (w2 * ratio2 - w1 * ratio1) / (w2 ** 2 - w1 ** 2)

            # Calculate the modal damping coefficients
            for k in range(len(w)):
                C_n[k] = alpha1 + alpha2 * w[k] ** 2
        else:
            # Use one damping ratio for all modes, default ratio is 0.02 (2%)
            for k in range(len(w)):
                C_n[k] = 2 * w[k] * constant_modal_damping

        # Calculate the forcing frequencies
        freq = linspace(f1,f2, f_div)
        omega_list = 2 * pi * array(freq)  # Angular frequency of load

        self.LoadFrequencies = array(freq)  # Save it

        # Initialise matrix to hold the normal displacements
        D_temp = zeros((len(self.Nodes) * 6, omega_list.shape[0]))

        # Calculate the modal coordinates for each forcing frequency

        try:
            n = 0  # Index for each displacement vector
            for omega in omega_list:
                for j in range(FV_n.shape[0]):
                    Q[j, 0] = FV_n[j, 0] * sqrt(1 / ((w[j] ** 2 - omega ** 2) ** 2 + (omega ** 2) * (C_n[j]) ** 2))

                # Calculate the Physical displacements
                D1 = Z @ Q

                # Form the global displacement vector, D, from D1 and D2
                D = zeros((len(self.Nodes) * 6, 1))

                for node in self.Nodes.values():

                    if D2_indices.count(node.ID * 6 + 0) == 1:
                        D.itemset((node.ID * 6 + 0, 0), D2[D2_indices.index(node.ID * 6 + 0), 0])
                    else:
                        D.itemset((node.ID * 6 + 0, 0), D1[D1_indices.index(node.ID * 6 + 0), 0])

                    if D2_indices.count(node.ID * 6 + 1) == 1:
                        D.itemset((node.ID * 6 + 1, 0), D2[D2_indices.index(node.ID * 6 + 1), 0])
                    else:
                        D.itemset((node.ID * 6 + 1, 0), D1[D1_indices.index(node.ID * 6 + 1), 0])

                    if D2_indices.count(node.ID * 6 + 2) == 1:
                        D.itemset((node.ID * 6 + 2, 0), D2[D2_indices.index(node.ID * 6 + 2), 0])
                    else:
                        D.itemset((node.ID * 6 + 2, 0), D1[D1_indices.index(node.ID * 6 + 2), 0])

                    if D2_indices.count(node.ID * 6 + 3) == 1:
                        D.itemset((node.ID * 6 + 3, 0), D2[D2_indices.index(node.ID * 6 + 3), 0])
                    else:
                        D.itemset((node.ID * 6 + 3, 0), D1[D1_indices.index(node.ID * 6 + 3), 0])

                    if D2_indices.count(node.ID * 6 + 4) == 1:
                        D.itemset((node.ID * 6 + 4, 0), D2[D2_indices.index(node.ID * 6 + 4), 0])
                    else:
                        D.itemset((node.ID * 6 + 4, 0), D1[D1_indices.index(node.ID * 6 + 4), 0])

                    if D2_indices.count(node.ID * 6 + 5) == 1:
                        D.itemset((node.ID * 6 + 5, 0), D2[D2_indices.index(node.ID * 6 + 5), 0])
                    else:
                        D.itemset((node.ID * 6 + 5, 0), D1[D1_indices.index(node.ID * 6 + 5), 0])

                # Save the all the maximum global displacement vectors for each load frequency
                if static_combo == None:
                    D_temp[:, n] = D[:, 0]
                else:
                    D_temp[:, n] = D[:, 0] + self._D[static_combo][:, 0]
                n += 1
            self._Max_D_Harmonic = D_temp




        except:
            raise Exception("'The stiffness matrix is singular, which implies rigid body motion."
                            "The structure is unstable. Aborting analysis.")


        if log:
            print('')
            print('- Analysis complete')
            print('')

        # Check statics if requested
        if check_statics == True:
            Analysis._check_statics(self)

        # Flag the model as solved
        self.solution = 'Harmonic'

        # Select the frequency to show results for
        self.set_load_frequency_to_query_results_for(f1,harmonic_combo)
        #return Z.T @ M11 @ Z


    def _mass_normalised_mode_shapes(self, m, modes):
        """
        Normalises the Mode shapes with respect to the mass matrix
        :param m: Mass matrix
        :type m: ndarray, lil_matrix, csr_matrix
        :param modes: Mode shapes
        :type modes: ndarray or lil_matrix
        """
        if isinstance(m, ndarray):
            Mr = modes.T @ m @ modes
        elif isinstance(m, lil_matrix):
            Mr = modes.T @ m.tocsr() @ modes
        elif isinstance(m, csr_matrix):
            Mr = modes.T @ m @ modes
        else:
            raise ValueError("Invalid input type for 'm'. Expected ndarray, lil_matrix, or csr_matrix.")

        for col in range(modes.shape[1]):
            modes[:, col] = modes[:, col] / sqrt(Mr[col, col])
        return modes

    def set_active_mode(self, active_mode):
        """
        Sets the active mode

        Parameters
        ---------
        active_mode : int
            The mode to set active

        Exceptions
        ----------
        Occurs when the function is called and the modal analysis results are not available
        """

        # Check if modal analysis results are available
        if self.solution == 'Modal' or self.solution == 'Harmonic':
            # Check that the requested mode is among the calculated modes
            calculated_modes = self._SHAPE.shape[0]
            if active_mode > calculated_modes:
                # If a higher mode is selected, set the maximum calculated mode as active
                active_mode = calculated_modes
            elif active_mode < 1:
                # Sets the active mode to 1 if a negative number is entered
                active_mode = 1

            # Set the active mode finally
            self.Active_Mode = active_mode

            # Set the combination name
            combo_name = 'Modal Combo'

            # Form the global displacement vector, D, from D1 and D2
            # Get the free and constrained indices
            D1_indices, D2_indices, D2 = self._aux_list()

            # Set zero modal displacements to the constrained DOFs
            D2 = zeros((len(D2), 1))

            # From the calculated modal displacements, select the required
            D1 = self._SHAPE[:, active_mode - 1].reshape((-1, 1))

            # Initialise the global modal displacements
            D = zeros((len(self.Nodes) * 6, 1))

            # The global modal displacement vector can now be formed
            for node in self.Nodes.values():

                if D2_indices.count(node.ID * 6 + 0) == 1:
                    D.itemset((node.ID * 6 + 0, 0), D2[D2_indices.index(node.ID * 6 + 0), 0])
                else:
                    D.itemset((node.ID * 6 + 0, 0), D1[D1_indices.index(node.ID * 6 + 0), 0])

                if D2_indices.count(node.ID * 6 + 1) == 1:
                    D.itemset((node.ID * 6 + 1, 0), D2[D2_indices.index(node.ID * 6 + 1), 0])
                else:
                    D.itemset((node.ID * 6 + 1, 0), D1[D1_indices.index(node.ID * 6 + 1), 0])

                if D2_indices.count(node.ID * 6 + 2) == 1:
                    D.itemset((node.ID * 6 + 2, 0), D2[D2_indices.index(node.ID * 6 + 2), 0])
                else:
                    D.itemset((node.ID * 6 + 2, 0), D1[D1_indices.index(node.ID * 6 + 2), 0])

                if D2_indices.count(node.ID * 6 + 3) == 1:
                    D.itemset((node.ID * 6 + 3, 0), D2[D2_indices.index(node.ID * 6 + 3), 0])
                else:
                    D.itemset((node.ID * 6 + 3, 0), D1[D1_indices.index(node.ID * 6 + 3), 0])

                if D2_indices.count(node.ID * 6 + 4) == 1:
                    D.itemset((node.ID * 6 + 4, 0), D2[D2_indices.index(node.ID * 6 + 4), 0])
                else:
                    D.itemset((node.ID * 6 + 4, 0), D1[D1_indices.index(node.ID * 6 + 4), 0])

                if D2_indices.count(node.ID * 6 + 5) == 1:
                    D.itemset((node.ID * 6 + 5, 0), D2[D2_indices.index(node.ID * 6 + 5), 0])
                else:
                    D.itemset((node.ID * 6 + 5, 0), D1[D1_indices.index(node.ID * 6 + 5), 0])

            # Store the calculated global nodal modal displacements into each node
            for node in self.Nodes.values():
                node.DX[combo_name] = D[node.ID * 6 + 0, 0]
                node.DY[combo_name] = D[node.ID * 6 + 1, 0]
                node.DZ[combo_name] = D[node.ID * 6 + 2, 0]
                node.RX[combo_name] = D[node.ID * 6 + 3, 0]
                node.RY[combo_name] = D[node.ID * 6 + 4, 0]
                node.RZ[combo_name] = D[node.ID * 6 + 5, 0]

        else:
            raise ResultsNotFoundError

    def natural_frequency(self, mode=None):
        # Check the results availability
        if self.solution == "Modal":
            if mode is None:
                mode = self.Active_Mode
            elif mode > len(self.Natural_Frequencies):
                mode = len(self.Natural_Frequencies)
            elif mode < 1:
                mode = 1
        else:
            raise Exception('Modal analysis results are not available')
        return self.Natural_Frequencies[mode - 1]

    def set_load_frequency_to_query_results_for(self, frequency, harmonic_combo, log = False):
        """
        Sets the frequency to query results for. Only works when harmonic results are available

        Parameters
        ----------
        :param frequency: The frequency for which results are desired
        :type frequency: int

        Raises
        ------
        : ResultsNotFoundError
             Occurs when the function is called and the harmonic analysis results are not available

        : InputOutOfRangeError
             Occurs when the provided frequency is out of range of frequencies analysed for

        """

        # Check if harmonic results are available and raise error if not

        if self.solution != 'Harmonic':
            raise ResultsNotFoundError

        # Get the frequency range analysed for
        f_min = min(self.LoadFrequencies)
        f_max = max(self.LoadFrequencies)

        # Check if the requested for frequency is within the range analysed for
        if frequency < f_min or frequency > f_max:
            raise InputOutOfRangeError

        dof = self._Max_D_Harmonic.shape[0]

        # Number of columns in y_data
        # num_columns = y_data.shape[1]

        # Perform cubic spline interpolation for each dof
        D = zeros((dof, 1))
        for i in range(dof):
            # Linear interpolation
            D[i, 0] = interp(frequency, self.LoadFrequencies, self._Max_D_Harmonic[i, :])

            # The Cubic spline interpolation below can also be used
            # spline_function = CubicSpline(self.LoadFrequencies, self._Max_D_Harmonic[i, :])
            # D[i,0] = spline_function(frequency)

        # Store the calculated global nodal modal displacements into each node
        for node in self.Nodes.values():
            node.DX[harmonic_combo] = D[node.ID * 6 + 0, 0]
            node.DY[harmonic_combo] = D[node.ID * 6 + 1, 0]
            node.DZ[harmonic_combo] = D[node.ID * 6 + 2, 0]
            node.RX[harmonic_combo] = D[node.ID * 6 + 3, 0]
            node.RY[harmonic_combo] = D[node.ID * 6 + 4, 0]
            node.RZ[harmonic_combo] = D[node.ID * 6 + 5, 0]

        # Re-calculate reactions
        # We do not want to calculate reactions for all the load combinations
        # Hence we will keep the load combinations in a temporary object
        load_combos_temp = copy.deepcopy(self.LoadCombos)

        # Then remove all other load combos except the required load combo
        self.LoadCombos.clear()
        self.LoadCombos = {harmonic_combo: load_combos_temp[harmonic_combo]}

        # Calculate the reactions
        Analysis._calc_reactions(self,log)

        # Restore the load combos
        self.LoadCombos = copy.deepcopy(load_combos_temp)

    def _renumber(self):
        """
        Assigns node and element ID numbers to be used internally by the program. Numbers are
        assigned according to the order in which they occur in each dictionary.
        """

        # Number each node in the model
        for id, node in enumerate(self.Nodes.values()):
            node.ID = id

        # Number each spring in the model
        for id, spring in enumerate(self.Springs.values()):
            spring.ID = id

        # Descritize all the physical members and number each member in the model
        id = 0
        for phys_member in self.Members.values():
            phys_member.descritize()
            for member in phys_member.sub_members.values():
                member.ID = id
                id += 1

        # Number each plate in the model
        for id, plate in enumerate(self.Plates.values()):
            plate.ID = id

        # Number each quadrilateral in the model
        for id, quad in enumerate(self.Quads.values()):
            quad.ID = id


    def unique_name(self, dictionary, prefix):
        """Returns the next available unique name for a dictionary of objects.

        :param dictionary: The dictionary to get a unique name for.
        :type dictionary: dict
        :param prefix: The prefix to use for the unique name.
        :type prefix: str
        :return: A unique name for the dictionary.
        :rtype: str
        """

        # Select a trial value for the next available name
        name = prefix + str(len(dictionary) + 1)
        i = 2
        while name in dictionary.keys():
            name = prefix + str(len(dictionary) + i)
            i += 1

        # Return the next available name
        return name

    def rename(self):
        """
        Renames all the nodes and elements in the model.
        """

        # Rename each node in the model
        temp = self.Nodes.copy()
        id = 1
        for old_key in temp.keys():
            new_key = 'N' + str(id)
            self.Nodes[new_key] = self.Nodes.pop(old_key)
            self.Nodes[new_key].name = new_key
            id += 1

        # Rename each spring in the model
        temp = self.Springs.copy()
        id = 1
        for old_key in temp.keys():
            new_key = 'S' + str(id)
            self.Springs[new_key] = self.Springs.pop(old_key)
            self.Springs[new_key].name = new_key
            id += 1

        # Rename each member in the model
        temp = self.Members.copy()
        id = 1
        for old_key in temp.keys():
            new_key = 'M' + str(id)
            self.Members[new_key] = self.Members.pop(old_key)
            self.Members[new_key].name = new_key
            id += 1

        # Rename each plate in the model
        temp = self.Plates.copy()
        id = 1
        for old_key in temp.keys():
            new_key = 'P' + str(id)
            self.Plates[new_key] = self.Plates.pop(old_key)
            self.Plates[new_key].name = new_key
            id += 1

        # Rename each quad in the model
        temp = self.Quads.copy()
        id = 1
        for old_key in temp.keys():
            new_key = 'Q' + str(id)
            self.Quads[new_key] = self.Quads.pop(old_key)
            self.Quads[new_key].name = new_key
            id += 1

    def orphaned_nodes(self):
        """
        Returns a list of the names of nodes that are not attached to any elements.
        """

        # Initialize a list of orphaned nodes
        orphans = []

        # Step through each node in the model
        for node in self.Nodes.values():

            orphaned = False

            # Check to see if the node is attached to any elements
            quads = [quad.name for quad in self.Quads.values() if
                     quad.i_node == node or quad.j_node == node or quad.m_node == node or quad.n_node == node]
            plates = [plate.name for plate in self.Plates.values() if
                      plate.i_node == node or plate.j_node == node or plate.m_node == node or plate.n_node == node]
            members = [member.name for member in self.Members.values() if
                       member.i_node == node or member.j_node == node]
            springs = [spring.name for spring in self.Springs.values() if
                       spring.i_node == node or spring.j_node == node]

            # Determine if the node is orphaned
            if quads == [] and plates == [] and members == [] and springs == []:
                orphaned = True

            # Add the orphaned nodes to the list of orphaned nodes
            if orphaned == True:
                orphans.append(node.name)

        return orphans
