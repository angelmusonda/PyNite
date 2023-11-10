# %%
from os import rename
import warnings
import copy
from math import isclose, ceil


from numpy import array, zeros, matmul, divide, subtract, atleast_2d

from numpy import array, zeros, matmul, divide, subtract, atleast_2d, nanmax, argsort, ones, cos, sin, exp, imag, \
    outer
from numpy import seterr, real, pi, sqrt, ndarray, interp, linspace, sum, append, arctan2

from numpy.linalg import solve
from scipy.linalg import inv
from scipy.sparse.linalg import eigs, eigsh
from scipy.sparse import csr_matrix, lil_matrix
from scipy.interpolate import CubicSpline

from PyNite.DisplacementProfile import DisplacementProfile
from PyNite.Node3D import Node3D
from PyNite.Material import Material
from PyNite.Section import Section
from PyNite.PhysMember import PhysMember
from PyNite.Spring3D import Spring3D
from PyNite.Member3D import Member3D
from PyNite.Quad3D import Quad3D
from PyNite.Plate3D import Plate3D
from PyNite.LoadCombo import LoadCombo
from PyNite.MassCase import MassCase
from PyNite.Mesh import Mesh, RectangleMesh, AnnulusMesh, FrustrumMesh, CylinderMesh
from PyNite import Analysis
from PyNite.LoadProfile import LoadProfile

from PyNite.PyNiteExceptions import ResultsNotFoundError, InputOutOfRangeError, ParameterIncompatibilityError, \
    DefinitionNotFoundError, DampingOptionsKeyWordError, DynamicLoadNotDefinedError


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

        self.Sections = {str:Section}      # A dictonary of the model's cross-sections
        self.Sections.pop(str)
        self.Springs = {str:Spring3D}      # A dictionary of the model's springs

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
        self._MODE_SHAPES = None  # A matrix of the model's mode shapes
        self._eigen_vectors = None # Eigen vectors of the model
        self._mass_participation = None # A dictionary to hold the percentages of mass participation in the x, y and z directions
        self.Active_Mode = 1  # A variable to keep track of the active mode
        self._NATURAL_FREQUENCIES = None  # A list to store the calculated natural frequencies
        self._DISPLACEMENT_REAL = None  # A matrix of the model's real part of the displacement for each load frequency
        self._VELOCITY_REAL = None # A matrix of the model's real part of the velocity for each load frequency
        self._ACCELERATION_REAL = None  # A matrix of the model's real part of the acceleration for each load frequency
        self._DISPLACEMENT_IMAGINARY = None  # A matrix of the model's imaginary part of the displacement for each load frequency
        self._VELOCITY_IMAGINARY = None  # A matrix of the model's imaginary part of the velocity for each load frequency
        self._ACCELERATION_IMAGINARY = None  # A matrix of the model's imaginary part of the acceleration for each load frequency
        self._REACTIONS_REAL = None # A matrix real part of reactions for the model solved using harmonic analysis for all load frequencies
        self._REACTIONS_IMAGINARY = None # A matrix of the imaginary part of reactions for the model solved using harmonic analysis for all load frequencies
        self.MassCases = {str: []}  # A dictionary of load cases to be considered as mass cases
        self.MassCases.pop(str)
        self.LoadProfiles = {str: LoadProfile}  # A dictionary of the model's dynamic load profiles
        self.LoadProfiles.pop(str)
        self.DisplacementProfiles = {str: DisplacementProfile}  # A dictionary of the model's dynamic displacement profiles
        self.DisplacementProfiles.pop(str)
        self._DISPLACEMENT_THA = None  # A matrix to store the global time history displacements
        self._VELOCITY_THA = None  # A matrix to store the global time history velocities
        self._ACCELERATION_THA = None  # A matrix to store the global time history accelerations
        self._TIME_THA = None  # A vector to store the time over which the time history analysis has been conducted
        self.F_TOTAL = None # A matrix to store the global dynamic total force used for calculating reactions
        self._REACTIONS_THA = None  # A matrix to store the reactions of the model for the entire time history
        self.LoadFrequencies = []  # A list to store the calculated load frequencies
        self.solution = None  # Indicates the solution type for the latest run of the model
        self.DynamicSolution = {
            "Harmonic": False,
            "Modal": False,
            "Time History": False
        }

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

        # Return the node name
        return name

    def add_material(self, name, E, G, nu, rho, fy=None):
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

        new_material = Material(name, E, G, nu, rho, fy)
        

        # Add the new material to the list
        self.Materials[name] = new_material

        # Flag the model as unsolved
        self.solution = None
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

    def add_section(self, name, section):
        """Adds a cross-section to the model.

        :param name: A unique name for the cross-section.
        :type name: string
        :param section: A 'PyNite' `Section` object.
        :type section: Section
        """

        # Check if the section name has already been used
        if name not in self.Sections.keys():
            # Store the section in the `Sections` dictionary
            self.Sections[name] = section
        else:
            # Stop execution and notify the user that the section name is already being used
            raise Exception('Cross-section name ' + name + ' already exists in the model.')

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

        # Return the spring name
        return name

    def add_member(self, name, i_node, j_node, material, Iy=None, Iz=None, J=None, A=None, aux_node=None, tension_only=False, comp_only=False, section_name=None):
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
        :type aux_node: str, optional
        :param tension_only: Indicates if the member is tension-only, defaults to False
        :type tension_only: bool, optional
        :param comp_only: Indicates if the member is compression-only, defaults to False
        :type comp_only: bool, optional
        :raises NameError: Occurs if the specified name already exists.
        :param section_name: The name of the cross section to use for section properties. If section is `None` the user must provide `Iy`, `Iz`, `A`, and `J`. If section is not `None` any values of `Iy`, `Iz`, `A`, and `J` will be ignored and the cross-section's values will be used instead.
        :type section: string
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

        if aux_node == None:
            new_member = PhysMember(name, self.Nodes[i_node], self.Nodes[j_node], material, self, Iy, Iz, J, A, tension_only=tension_only, comp_only=comp_only, section_name=section_name)
        else:
            new_member = PhysMember(name, self.Nodes[i_node], self.Nodes[j_node], material, self, Iy, Iz, J, A, aux_node=self.AuxNodes[aux_node], tension_only=tension_only, comp_only=comp_only, section_name=section_name)
        
        # Add the new member to the list
        self.Members[name] = new_member

        # Flag the model as unsolved
        self.solution = None
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

    def delete_spring(self, spring_name):
        """Removes a spring from the model.

        :param spring_name: The name of the spring to be removed.
        :type spring_name: str
        """

        # Remove the spring
        self.Springs.pop(spring_name)

        # Flag the model as unsolved
        self.solution = None
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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

        # Flag the model as unsolved
        self.solution = None
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

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
        for key in self.DynamicSolution.keys():
            self.DynamicSolution[key] = False

    def def_load_profile(self, case, time, profile):
        """
            Define a load profile for a specific load case.

            :param case: The name or identifier of the load case.
            :type case: str
            :param time: A list of time values associated with the load profile.
            :type time: list[float]
            :param profile: A list of load values corresponding to the time values.
            :type profile: list[float]

            :raises ParameterIncompatibilityError: If the lengths of 'time' and 'profile' lists are not the same.

            This method creates a LoadProfile object for the specified load case and associates it with the given time and profile data.
            """
        if len(time) != len(profile):
            raise ParameterIncompatibilityError('Time and Profile lists should have the same length')
        self.LoadProfiles[case] = LoadProfile(load_case_name=case, time=time, profile=profile)

    def def_disp_profile(self, node, direction, time, profile):
        """
            Define a displacement profile for a specific node and direction.

            :param node: The node or point at which the displacement is measured.
            :type node: str
            :param direction: The direction of the displacement ('DX', 'DY', 'DZ', 'RX', 'RY', 'RZ').
            :type direction: str
            :param time: A list of time values associated with the displacement profile.
            :type time: list[float]
            :param profile: A list of displacement values corresponding to the time values.
            :type profile: list[float]

            :raises ParameterIncompatibilityError: If the lengths of 'time' and 'profile' lists are not the same.
            :raises ValueError: If 'direction' is not one of ('DX', 'DY', 'DZ', 'RX', 'RY', 'RZ').

            This method creates a DisplacementProfile object for the specified node and direction and associates it with the given time and profile data.
            """
        if len(time) != len(profile):
            raise ParameterIncompatibilityError('Time and Profile lists should have the same length')
        if direction not in ('DX', 'DY', 'DZ', 'RX', 'RY', 'RZ'):
            raise ValueError(f"Direction must be 'DX', 'DY', 'DZ', 'RX', 'RY', or 'RZ'. {direction} was given.")
        self.DisplacementProfiles[node] = DisplacementProfile(node=node, direction= direction,time=time, profile=profile)
        
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


                    # Calculate the axial force acting on the member
                    if first_step:
                        # For the first load step take P = 0
                        P = 0
                    else:
                        # Calculate the member axial force due to axial strain
                        d = member.d(combo_name)
                        P = E*A/L*(d[6, 0] - d[0, 0])


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

      
    def Km(self, combo_name='Combo 1', push_combo='Pushover', step_num=1, log=False, sparse=True):
        """Calculates the structure's global plastic reduction matrix, which is used for nonlinear inelastic analysis.

        :param combo_name: The name of the load combination to get the plastic reduction matrix for. Defaults to 'Combo 1'.
        :type combo_name: str, optional
        :param push_combo: The name of the load combination that contains the pushover load definition. Defaults to 'Pushover'.
        :type push_combo: str, optional
        :param step_num: The load step used to generate the plastic reduction matrix. Defaults to 1.
        :type step_num: int, optional
        :param log: Determines whether this method writes output to the console as it runs. Defaults to False.
        :type log: bool, optional
        :param sparse: Indicates whether the sparse solver should be used. Defaults to True.
        :type sparse: bool, optional
        :return: The gloabl plastic reduction matrix.
        :rtype: array
        """
       
        # Determine if a sparse matrix has been requested
        if sparse == True:
            # The plastic reduction matrix will be stored as a scipy `coo_matrix`. Scipy's documentation states that this type of matrix is ideal for efficient construction of finite element matrices. When converted to another format, the `coo_matrix` sums values at the same (i, j) index. We'll build the matrix from three lists.
            row = []
            col = []
            data = []
        else:
            # Initialize a dense matrix of zeros
            Km = zeros((len(self.Nodes)*6, len(self.Nodes)*6))

        # Add stiffness terms for each physical member in the model
        if log: print('- Calculating the plastic reduction matrix')
        for phys_member in self.Members.values():
            
            # Check to see if the physical member is active for the given load combination
            if phys_member.active[combo_name] == True:

                # Step through each sub-member in the physical member and add terms
                for member in phys_member.sub_members.values():
                    
                    # Get the member's global plastic reduction matrix
                    # Storing it as a local variable eliminates the need to rebuild it every time a term is needed
                    member_Km = member.Km(combo_name, push_combo, step_num)

                    # Step through each term in the member's plastic reduction matrix
                    # 'a' & 'b' below are row/column indices in the member's matrix
                    # 'm' & 'n' are corresponding row/column indices in the structure's global matrix
                    for a in range(12):
                    
                        # Determine if index 'a' is related to the i-node or j-node
                        if a < 6:
                            # Find the corresponding index 'm' in the global plastic reduction matrix
                            m = member.i_node.ID*6 + a
                        else:
                            # Find the corresponding index 'm' in the global plastic reduction matrix
                            m = member.j_node.ID*6 + (a-6)
                        
                        for b in range(12):
                        
                            # Determine if index 'b' is related to the i-node or j-node
                            if b < 6:
                                # Find the corresponding index 'n' in the global plastic reduction matrix
                                n = member.i_node.ID*6 + b
                            else:
                                # Find the corresponding index 'n' in the global plastic reduction matrix
                                n = member.j_node.ID*6 + (b-6)
                        
                            # Now that 'm' and 'n' are known, place the term in the global plastic reduction matrix
                            if sparse == True:
                                row.append(m)
                                col.append(n)
                                data.append(member_Km[a, b])
                            else:
                                Km[m, n] += member_Km[a, b]

        if sparse:
            # The plastic reduction matrix will be stored as a scipy `coo_matrix`. Scipy's documentation states that this type of matrix is ideal for efficient construction of finite element matrices. When converted to another format, the `coo_matrix` sums values at the same (i, j) index.
            from scipy.sparse import coo_matrix
            row = array(row)
            col = array(col)
            data = array(data)
            Km = coo_matrix((data, (row, col)), shape=(len(self.Nodes)*6, len(self.Nodes)*6))

        # Check that there are no nodal instabilities
        # if check_stability:
        #     if log: print('- Checking nodal stability')
        #     if sparse: Analysis._check_stability(self, Km.tocsr())
        #     else: Analysis._check_stability(self, Km)

        # Return the global plastic reduction matrix
        return Km 


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

        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = Analysis._partition_D(self)

        # Identify which load combinations have the tags the user has given
        combo_list = Analysis._identify_combos(self, combo_tags)

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

                    K11, K12, K21, K22 = Analysis._partition(self, self.K(combo.name, log, check_stability, sparse).tolil(), D1_indices, D2_indices)
                else:
                    K11, K12, K21, K22 = Analysis._partition(self, self.K(combo.name, log, check_stability, sparse), D1_indices, D2_indices)


                # Get the partitioned global fixed end reaction vector
                FER1, FER2 = Analysis._partition(self, self.FER(combo.name), D1_indices, D2_indices)

                # Get the partitioned global nodal force vector       

                P1, P2 = Analysis._partition(self, self.P(combo.name), D1_indices, D2_indices)          


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


                # Store the calculated displacements to the model and the nodes in the model
                Analysis._store_displacements(self, D1, D2, D1_indices, D2_indices, combo)

                
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


        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = Analysis._partition_D(self)

        # Get the partitioned global stiffness matrix K11, K12, K21, K22
        # Note that for linear analysis the stiffness matrix can be obtained for any load combination, as it's the same for all of them
        combo_name = list(self.LoadCombos.keys())[0]
        if sparse == True:

            K11, K12, K21, K22 = Analysis._partition(self, self.K(combo_name, log, check_stability, sparse).tolil(), D1_indices, D2_indices)
        else:
            K11, K12, K21, K22 = Analysis._partition(self, self.K(combo_name, log, check_stability, sparse), D1_indices, D2_indices)


        # Identify which load combinations have the tags the user has given
        combo_list = Analysis._identify_combos(self, combo_tags)

        # Step through each load combination
        for combo in combo_list:

            if log:
                print('')
                print('- Analyzing load combination ' + combo.name)

            # Get the partitioned global fixed end reaction vector
            FER1, FER2 = Analysis._partition(self, self.FER(combo.name), D1_indices, D2_indices)

            # Get the partitioned global nodal force vector       

            P1, P2 = Analysis._partition(self, self.P(combo.name), D1_indices, D2_indices)          


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


            # Store the calculated displacements to the model and the nodes in the model
            Analysis._store_displacements(self, D1, D2, D1_indices, D2_indices, combo)


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

    def analyze_PDelta(self, log=False, check_stability=True, max_iter=30, sparse=True, combo_tags=None):
        """Performs second order (P-Delta) analysis. This type of analysis is appropriate for most models using beams, columns and braces. Second order analysis is usually required by material specific codes. The analysis is iterative and takes longer to solve. Models with slender members and/or members with combined bending and axial loads will generally have more significant P-Delta effects. P-Delta effects in plates/quads are not considered.

        :param log: Prints updates to the console if set to True. Default is False.
        :type log: bool, optional
        :param check_stability: When set to True, checks the stiffness matrix for any unstable degrees of freedom and reports them back to the console. This does add to the solution time. Defaults to True.
        :type check_stability: bool, optional
        :param max_iter: The maximum number of iterations permitted. If this value is exceeded the program will report divergence. Defaults to 30.
        :type max_iter: int, optional
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


        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = Analysis._partition_D(self)

        # Identify which load combinations have the tags the user has given
        combo_list = Analysis._identify_combos(self, combo_tags)

        # Step through each load combination

        for combo in combo_list:


            # Get the partitioned global fixed end reaction vector
            FER1, FER2 = Analysis._partition(self, self.FER(combo.name), D1_indices, D2_indices)

            # Get the partitioned global nodal force vector       
            P1, P2 = Analysis._partition(self, self.P(combo.name), D1_indices, D2_indices)

            # Run the P-Delta analysis for this load combination
            Analysis._PDelta_step(self, combo.name, P1, FER1, D1_indices, D2_indices, D2, log, sparse, check_stability, max_iter, True)
        
        # Calculate reactions
        Analysis._calc_reactions(self, log, combo_tags)


        if log:
            print('')
            print('- Analysis complete')
            print('')
        
        # Flag the model as solved
        self.solution = 'P-Delta'
    
    def _not_ready_yet_analyze_pushover(self, log=False, check_stability=True, push_combo='Combo 1', max_iter=30, tol=0.01, sparse=True, combo_tags=None):

        if log:
            print('+---------------------+')
            print('| Analyzing: Pushover |')
            print('+---------------------+')
        
        # Import `scipy` features if the sparse solver is being used
        if sparse == True:
            from scipy.sparse.linalg import spsolve

        # Prepare the model for analysis
        Analysis._prepare_model(self)
        
        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = Analysis._partition_D(self)


        # Identify and tag the primary load combinations the pushover load will be added to
        for combo in self.LoadCombos.values():

            # No need to tag the pushover combo
            if combo.name != push_combo:

                # Add 'primary' to the combo's tags if it's not already there
                if combo.combo_tags is None:
                    combo.combo_tags = ['primary']
                elif 'primary' not in combo.combo_tags:
                    combo.combo_tags.append('primary')

        # Identify which load combinations have the tags the user has given
        # TODO: Remove the pushover combo istelf from `combo_list`
        combo_list = Analysis._identify_combos(self, combo_tags)
        combo_list = [combo for combo in combo_list if combo.name != push_combo]


        # Step through each load combination
        for combo in combo_list:

            # Skip the pushover combo
            if combo.name == push_combo:
                continue


            if log:
                print('')
                print('- Analyzing load combination ' + combo.name)
            
            # Get the pushover load step and initialize the load factor
            load_step = list(self.LoadCombos[push_combo].factors.values())[0]
            load_factor = load_step
            step_num = 1
            
            # Get the partitioned global fixed end reaction vector
            FER1, FER2 = Analysis._partition(self, self.FER(combo.name), D1_indices, D2_indices)

            # Get the partitioned global nodal force vector       
            P1, P2 = Analysis._partition(self, self.P(combo.name), D1_indices, D2_indices)


            # Get the partitioned global fixed end reaction vector for a pushover load increment
            FER1_push, FER2_push = Analysis._partition(self, self.FER(push_combo), D1_indices, D2_indices)

            # Get the partitioned global nodal force vector for a pushover load increment
            P1_push, P2_push = Analysis._partition(self, self.P(push_combo), D1_indices, D2_indices)


            # Solve the current load combination without the pushover load applied
            Analysis._PDelta_step(self, combo.name, P1, FER1, D1_indices, D2_indices, D2, log, sparse, check_stability, max_iter, tol, first_step=True)


            # Delete this next line used for debugging
            fx, my, mz = self.Members['M1'].f(combo.name)[0, 0], self.Members['M1'].f(combo.name)[4, 0], self.Members['M1'].f(combo.name)[5, 0]
            dummy = 1

            # Apply the pushover load in steps, summing deformations as we go, until the full pushover load has been analyzed
            while load_factor <= 1:
                
                # Inform the user which pushover load step we're on
                if log:
                    print('- Beginning pushover load step #' + str(step_num))


                # Run the next pushover load step
                Analysis._pushover_step(self, combo.name, push_combo, step_num, P1_push, FER1_push, D1_indices, D2_indices, D2, log, sparse, check_stability)

                # Delete this next line used for debugging
                fx, my, mz = self.Members['M1'].f(combo.name)[0, 0], self.Members['M1'].f(combo.name)[4, 0], self.Members['M1'].f(combo.name)[5, 0]


                # Move on to the next load step
                load_factor += load_step
                step_num += 1

        # Calculate reactions for every primary load combination
        Analysis._calc_reactions(self, log, combo_tags=['primary'])


        if log:
            print('')
            print('- Analysis complete')
            print('')

        # Flag the model as solved

        self.solution = 'Pushover'
    

    def analyze_modal(self, log=False, check_stability=True, num_modes=1, tol=0.01, sparse=True,
                      type_of_mass_matrix = 'consistent'):
        """Performs modal analysis. Also calculates the mass participation percentages in each direction

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
        :return: Mass participation percentages
        :rtype: dict
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

        #Prepare model
        Analysis._prepare_model(self)

        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = Analysis._partition_D(self)

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

        # Make sure the required number of modes is less or equal to the total number of modes
        num_modes = min(K11.shape[0],num_modes)

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
        self._NATURAL_FREQUENCIES = sqrt(eigVal)/ (2 * pi)

        # Store the calculated modal displacements
        self._eigen_vectors = real(eigVec)

        # Form the mode shapes
        D = zeros((len(self.Nodes) * 6, 1))
        MODE_SHAPE_TEMP = zeros((len(self.Nodes)*6, len(eigVal)))

        for i in range(len(eigVal)):

            D1 = eigVec[:, i]
            D1 = D1.reshape(len(D1),1)

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

                MODE_SHAPE_TEMP[:, i] = D[:,0]

        # Store the calculated mode shapes
        self._MODE_SHAPES = MODE_SHAPE_TEMP

        # Store the calculated global nodal modal displacements into each node
        D = self._MODE_SHAPES
        for node in self.Nodes.values():
            node.DX[combo_name] = D[node.ID * 6 + 0, 0]
            node.DY[combo_name] = D[node.ID * 6 + 1, 0]
            node.DZ[combo_name] = D[node.ID * 6 + 2, 0]
            node.RX[combo_name] = D[node.ID * 6 + 3, 0]
            node.RY[combo_name] = D[node.ID * 6 + 4, 0]
            node.RZ[combo_name] = D[node.ID * 6 + 5, 0]

        # Calculate the effective modal mass percentage
        # Begin by calculating the total mass of the structure in each direction
        total_dof = len(D1_indices)+len(D2_indices)
        i = 0
        influence_X = zeros((total_dof,1))
        influence_Y = zeros((total_dof,1))
        influence_Z = zeros((total_dof,1))

        while i < total_dof:
            influence_X[i+0,0] = 1
            influence_Y[i+1,0] = 1
            influence_Z[i+2,0] = 1
            i += 6


        # Partition the influence vectors
        influence_X = self._partition(influence_X, D1_indices, D2_indices)[0]
        influence_Y = self._partition(influence_Y, D1_indices, D2_indices)[0]
        influence_Z = self._partition(influence_Z, D1_indices, D2_indices)[0]

        # Calculate total masses in each direction
        if sparse:
            Mass_X = sum(M11.tocsr() @ influence_X)
            Mass_Y = sum(M11.tocsr() @ influence_Y)
            Mass_Z = sum(M11.tocsr() @ influence_Z)

        else:
            Mass_X = sum(M11 @ influence_X)
            Mass_Y = sum(M11 @ influence_Y)
            Mass_Z = sum(M11 @ influence_Z)

        # Calculate the mass normalised mode shapes
        Z = self._mass_normalised_mode_shapes(M11,self._eigen_vectors)

        # Calculate the modal participation factors
        MPF_X = Z.T @ M11 @ influence_X
        MPF_Y = Z.T @ M11 @ influence_Y
        MPF_Z = Z.T @ M11 @ influence_Z

        # Calculate the effective modal masses
        EMM_X = sum(MPF_X**2)
        EMM_Y = sum(MPF_Y**2)
        EMM_Z = sum(MPF_Z**2)

        # Calculate and save the participating mass
        self._mass_participation = {'X':100*EMM_X/Mass_X,
                                    'Y':100*EMM_Y/Mass_Y,
                                    'Z':100*EMM_Z/Mass_Z}

        if log:
            print('')
            print('- Analysis complete')
            print('')

        # Flag the model as solved
        self.DynamicSolution['Modal'] = True

        return self._mass_participation



    def analyze_harmonic(self, f1, f2, f_div,harmonic_combo = None,
                        log=False, sparse=True, type_of_mass_matrix = 'consistent',
                         damping_options = dict()):
        """
          Conducts harmonic analysis for a specified harmonic load combination within a defined load frequency range. This analysis presupposes that a modal analysis has been conducted beforehand.

          In this analysis, prescribed displacements are treated as amplitudes that share the same frequency as the applied forces. Consequently, the analysis can also be executed exclusively for prescribed displacements, effectively using them as the sole source of excitation.
          :param f1: The lowest forcing frequency to consider.
          :type f1: float
          :param f2: The highest forcing frequency to consider.
          :type f2: float
          :param f_div: The number of frequencies in the range to compute for.
          :type f_div: int
          :param harmonic_combo: The harmonic load combination. (Optional)
          :type harmonic_combo: str
          :param log: If True, print analysis log to the console. Default is False.
          :type log: bool, optional
          :param sparse: Indicates whether the sparse matrix solver should be used. Default is True.
                         Sparse solvers can offer faster solutions for matrices with many zero terms.
          :type sparse: bool, optional
          :param type_of_mass_matrix: The type of element mass matrix to use in the analysis. Default is 'consistent'.
          :type type_of_mass_matrix: str, optional
          :param damping_options: Dictionary containing damping options (optional).
          :type damping_options: dict, optional

          :raises Exception: Occurs when a singular stiffness matrix is found, indicating an unstable structure has been modeled.
          :raises ResultsNotFoundError: Raised if modal results are not available, as harmonic analysis requires modal results.
          :raises ValueError: Raised for invalid input values, such as negative frequencies or invalid direction values.
          :raises DynamicLoadNotDefinedError: Raised if no load combination or prescribed displacement is provided.
          """

        # Check if modal results are available
        if self.DynamicSolution['Modal'] == False:
            raise ResultsNotFoundError(message=" Modal results are not available. Harmonic analysis requires modal results")


        # Check frequency
        if f1 < 0 or f2 < 0 or f_div < 0:
            raise ValueError("f1, f2 and f_div must be positive")

        if f2 < f1:
            raise ValueError("f2 must be greater than f1")

        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = Analysis._partition_D(self)

        # Convert D2 from a list to a vector
        D2 = atleast_2d(D2)

        # Check if the load combo name is among the defined load combos
        if harmonic_combo != None and harmonic_combo not in self.LoadCombos.keys():
            raise ValueError("'" + harmonic_combo + "' is not among the defined load combinations")

        # Atleast a load combination should be provided or a harmonic displacement should be prescribed
        if harmonic_combo == None:
            if any(D2) == False:
                raise DynamicLoadNotDefinedError('Provide the name of the dynamic load combination or at least prescribe a displacement at one node')

        # If only prescribed displacement has been provided, add default the load combination 'FRA combo'
        if harmonic_combo==None:
            harmonic_combo = 'FRA combo'
            self.LoadCombos['FRA combo'] = LoadCombo(name='FRA combo', factors={'Case 1':0}, combo_tags='FRA')


        # We can now begin the harmonic analysis
        if log:
            print('+--------------------+')
            print('| Analyzing: Harmonic|')
            print('+--------------------+')


        # Import `scipy` features if the sparse solver is being used
        if sparse == True:
            from scipy.sparse.linalg import spsolve

        # Prepare model
        Analysis._prepare_model(self)


        # Get the partitioned global stiffness matrix K11, K12, K21, K22
        if sparse == True:
            K11, K12, K21, K22 = self._partition(self.K(harmonic_combo, log, False, sparse).tolil(), D1_indices, D2_indices)
            M11, M12, M21, M22 = self._partition(self.M(harmonic_combo, log, False, sparse,type_of_mass_matrix).tolil(), D1_indices,
                                                 D2_indices)
            K_total = self.K(harmonic_combo, log, False, sparse).tolil()
            M_total = self.M(harmonic_combo, log, False, sparse,type_of_mass_matrix).tolil()
        else:
            K11, K12, K21, K22 = self._partition(self.K(harmonic_combo, log, False, sparse).tolil(), D1_indices, D2_indices)
            M11, M12, M21, M22 = self._partition(self.M(harmonic_combo, log, False, sparse, type_of_mass_matrix), D1_indices, D2_indices)
            K_total = self.K(harmonic_combo, log, False, sparse).tolil()
            M_total = self.M(harmonic_combo, log, False, sparse, type_of_mass_matrix)

        # Get the mass normalised mode shape matrix
        Z = self._mass_normalised_mode_shapes(M11, self._eigen_vectors)

        # Get the partitioned global fixed end reaction vector
        FER1, FER2 = self._partition(self.FER(harmonic_combo), D1_indices, D2_indices)

        # Get the partitioned global nodal force vector
        P1, P2 = self._partition(self.P(harmonic_combo), D1_indices, D2_indices)

        # Get the total force vector to be used for calculation of reactions
        F_total = self.P(harmonic_combo) - self.FER(harmonic_combo)

        # Calculate the normalised force vector
        FV_n = Z.T @ (P1-FER1 - K12 @ D2)

        # Calculate the damping coefficients
        w = 2 * pi * self._NATURAL_FREQUENCIES  # Angular natural frequencies

        # Calculate the damping matrix
        # Initialise the damping options
        constant_modal_damping = 0.00
        rayleigh_alpha = None
        rayleigh_beta = None
        first_mode_damping_ratio = None
        highest_mode_damping_ratio = None
        damping_ratios_in_every_mode = None

        # Get the damping options from the dictionary
        if 'constant_modal_damping' in damping_options:
            constant_modal_damping = damping_options['constant_modal_damping']
        if 'r_alpha' in damping_options:
            rayleigh_alpha = damping_options['r_alpha']

        if 'r_beta' in damping_options:
            rayleigh_beta = damping_options['r_beta']

        if 'first_mode_damping' in damping_options:
            first_mode_damping_ratio = damping_options['first_mode_damping']

        if 'highest_mode_damping' in damping_options:
            highest_mode_damping_ratio = ['highest_mode_damping']

        if 'damping_in_every_mode' in damping_options:
            damping_ratios_in_every_mode = damping_options['damping_in_every_mode']

        # Build the modal damping matrix
        # Initialise a one dimensional array
        C_n = zeros(FV_n.shape[0])
        if damping_ratios_in_every_mode != None:
            # Declare new variable called ratios, easier to work with than the original
            ratios = damping_ratios_in_every_mode
            # Check if it is a list or turple
            if isinstance(ratios, (list, tuple)):
                # Calculate the modal damping coefficient for each mode
                # If too many damping ratios have been provided, only the first entries
                # corresponding to the requested modes will be used
                # If fewer ratios have been provided, the last ratio will be used for the rest
                # of the modes
                for k in range(len(w)):
                    C_n[k] = 2 * w[k] * ratios[min(k, len(ratios) - 1)]
            else:
                # The provided input is perhaps a just a number, not a list
                # That number will be used for all the modes
                for k in range(len(w)):
                    C_n[k] = 2 * damping_ratios_in_every_mode * w[k]
        elif rayleigh_alpha != None or rayleigh_beta != None:
            # At-least one rayleigh damping coefficient has been specified
            if rayleigh_alpha == None:
                rayleigh_alpha = 0
            if rayleigh_beta == None:
                rayleigh_beta = 0
            for k in range(len(w)):
                C_n[k] = rayleigh_alpha + rayleigh_beta * w[k] ** 2

        elif first_mode_damping_ratio != None or highest_mode_damping_ratio != None:
            # Rayleigh damping is requested and at-least one damping ratio is given
            # If only one ratio is given, it will be assumed to be the damping
            # in the lowest and highest modes
            if first_mode_damping_ratio == None:
                first_mode_damping_ratio = highest_mode_damping_ratio
            if highest_mode_damping_ratio == None:
                highest_mode_damping_ratio = first_mode_damping_ratio

            # Calculate the rayleigh damping coefficients
            # Create new shorter variables
            ratio1 = first_mode_damping_ratio
            ratio2 = highest_mode_damping_ratio

            # Extract the first and last angular frequencies
            w1 = w[0]  # Angular frequency of first mode
            w2 = w[-1]  # Angular frequency of last mode

            # Calculate the rayleigh damping coefficients
            alpha_r = 2 * w1 * w2 * (w2 * ratio1 - w1 * ratio2) / (w2 ** 2 - w1 ** 2)
            beta_r = 2 * (w2 * ratio2 - w1 * ratio1) / (w2 ** 2 - w1 ** 2)

            # Calculate the modal damping coefficients
            for k in range(len(w)):
                C_n[k] = alpha_r + beta_r * w[k] ** 2
        else:
            # Use one damping ratio for all modes, default ratio is 0.02 (2%)
            for k in range(len(w)):
                C_n[k] = 2 * w[k] * constant_modal_damping

        # Calculate the list of forcing frequencies
        # Distribute the load frequencies around the natural frequencies
        natural_frequencies = self._NATURAL_FREQUENCIES
        freq = list(linspace(f1, f2, f_div))
        for natural_freq in natural_frequencies:
            if(natural_freq>=f1 and natural_freq<=f2):
                freq.extend(
                    [0.9 * natural_freq,
                     0.95 * natural_freq,
                     natural_freq,
                     1.05 * natural_freq,
                     1.1 * natural_freq])

        freq = list(set(freq))
        freq.sort()

        omega_list = 2 * pi * array(freq)  # Angular frequency of load

        self.LoadFrequencies = array(freq)  # Save it

        # Initialise matrices to hold the three responses
        D_temp = zeros((len(self.Nodes) * 6, omega_list.shape[0]),dtype=complex)
        V_temp = zeros((len(self.Nodes) * 6, omega_list.shape[0]),dtype=complex)
        A_temp = zeros((len(self.Nodes) * 6, omega_list.shape[0]),dtype=complex)

        # Calculate the modal coordinates for each forcing frequency
        try:
            k = 0  # Index for each displacement vector
            # Initialise vectors to hold the modal response in complex notation
            Q = zeros(len(FV_n), dtype=complex) # Displacement
            Q_dot = zeros(len(FV_n), dtype=complex)  # Velocity
            Q_dot_dot = zeros(len(FV_n), dtype=complex)  # Acceleration

            # Loop through the load frequencies
            for omega in omega_list:
                # Add the effect of the prescribed acceleration [- omega**2 * Z.T @ M12 @ D2]
                # We should subtract the above from the force vector, hence the addition below
                # Not that the effect of prescribed velocity is not considered since C12 is not known
                FV_temp =  FV_n[:,0].reshape(len(FV_n),1) + omega**2 * Z.T @ M12 @ D2
                for n in range(FV_n.shape[0]):
                    # Calculate the amplitude
                    q = FV_temp[n, 0] * 1 / sqrt((w[n] ** 2 - omega ** 2) ** 2 + (omega ** 2) * (C_n[n]) ** 2)

                    # Calculate the phase
                    phase = arctan2(C_n[n]*omega, w[n]**2-omega**2)

                    # Calculate the response in complex notation
                    Q[n] = q*(cos(phase)+1j*sin(phase))
                    Q_dot[n] = omega * q * (cos(phase-pi/2)+1j*sin(phase-pi/2))
                    Q_dot_dot[n] = -(omega**2) * Q[n]

                # Calculate the physical responses
                D1 =  Z @ Q
                V1 = Z @ Q_dot
                A1 = Z @ Q_dot_dot

                # Make sure the dimensions are correct
                D1 = D1.reshape(len(D1),1)
                V1 = V1.reshape(len(V1), 1)
                A1 = A1.reshape(len(A1), 1)

                V2 = omega * D2 * 1j * sin(-pi/2)
                A2 = -(omega**2) * D2

                # Form the global displacement, velocity and acceleration vectors
                D = zeros((len(self.Nodes) * 6, 1), dtype=complex)
                V = zeros((len(self.Nodes) * 6, 1), dtype=complex)
                A = zeros((len(self.Nodes) * 6, 1), dtype=complex)

                for node in self.Nodes.values():

                    if D2_indices.count(node.ID * 6 + 0) == 1:
                        D.itemset((node.ID * 6 + 0, 0), D2[D2_indices.index(node.ID * 6 + 0), 0])
                        V.itemset((node.ID * 6 + 0, 0), V2[D2_indices.index(node.ID * 6 + 0), 0])
                        A.itemset((node.ID * 6 + 0, 0), A2[D2_indices.index(node.ID * 6 + 0), 0])
                    else:
                        D.itemset((node.ID * 6 + 0, 0), D1[D1_indices.index(node.ID * 6 + 0), 0])
                        V.itemset((node.ID * 6 + 0, 0), V1[D1_indices.index(node.ID * 6 + 0), 0])
                        A.itemset((node.ID * 6 + 0, 0), A1[D1_indices.index(node.ID * 6 + 0), 0])

                    if D2_indices.count(node.ID * 6 + 1) == 1:
                        D.itemset((node.ID * 6 + 1, 0), D2[D2_indices.index(node.ID * 6 + 1), 0])
                        V.itemset((node.ID * 6 + 1, 0), V2[D2_indices.index(node.ID * 6 + 1), 0])
                        A.itemset((node.ID * 6 + 1, 0), A2[D2_indices.index(node.ID * 6 + 1), 0])
                    else:
                        D.itemset((node.ID * 6 + 1, 0), D1[D1_indices.index(node.ID * 6 + 1), 0])
                        V.itemset((node.ID * 6 + 1, 0), V1[D1_indices.index(node.ID * 6 + 1), 0])
                        A.itemset((node.ID * 6 + 1, 0), A1[D1_indices.index(node.ID * 6 + 1), 0])
                    if D2_indices.count(node.ID * 6 + 2) == 1:
                        D.itemset((node.ID * 6 + 2, 0), D2[D2_indices.index(node.ID * 6 + 2), 0])
                        V.itemset((node.ID * 6 + 2, 0), V2[D2_indices.index(node.ID * 6 + 2), 0])
                        A.itemset((node.ID * 6 + 2, 0), A2[D2_indices.index(node.ID * 6 + 2), 0])
                    else:
                        D.itemset((node.ID * 6 + 2, 0), D1[D1_indices.index(node.ID * 6 + 2), 0])
                        V.itemset((node.ID * 6 + 2, 0), V1[D1_indices.index(node.ID * 6 + 2), 0])
                        A.itemset((node.ID * 6 + 2, 0), A1[D1_indices.index(node.ID * 6 + 2), 0])

                    if D2_indices.count(node.ID * 6 + 3) == 1:
                        D.itemset((node.ID * 6 + 3, 0), D2[D2_indices.index(node.ID * 6 + 3), 0])
                        V.itemset((node.ID * 6 + 3, 0), V2[D2_indices.index(node.ID * 6 + 3), 0])
                        A.itemset((node.ID * 6 + 3, 0), A2[D2_indices.index(node.ID * 6 + 3), 0])
                    else:
                        D.itemset((node.ID * 6 + 3, 0), D1[D1_indices.index(node.ID * 6 + 3), 0])
                        V.itemset((node.ID * 6 + 3, 0), V1[D1_indices.index(node.ID * 6 + 3), 0])
                        A.itemset((node.ID * 6 + 3, 0), A1[D1_indices.index(node.ID * 6 + 3), 0])

                    if D2_indices.count(node.ID * 6 + 4) == 1:
                        D.itemset((node.ID * 6 + 4, 0), D2[D2_indices.index(node.ID * 6 + 4), 0])
                        V.itemset((node.ID * 6 + 4, 0), V2[D2_indices.index(node.ID * 6 + 4), 0])
                        A.itemset((node.ID * 6 + 4, 0), A2[D2_indices.index(node.ID * 6 + 4), 0])
                    else:
                        D.itemset((node.ID * 6 + 4, 0), D1[D1_indices.index(node.ID * 6 + 4), 0])
                        V.itemset((node.ID * 6 + 4, 0), V1[D1_indices.index(node.ID * 6 + 4), 0])
                        A.itemset((node.ID * 6 + 4, 0), A1[D1_indices.index(node.ID * 6 + 4), 0])

                    if D2_indices.count(node.ID * 6 + 5) == 1:
                        D.itemset((node.ID * 6 + 5, 0), D2[D2_indices.index(node.ID * 6 + 5), 0])
                        V.itemset((node.ID * 6 + 5, 0), V2[D2_indices.index(node.ID * 6 + 5), 0])
                        A.itemset((node.ID * 6 + 5, 0), A2[D2_indices.index(node.ID * 6 + 5), 0])
                    else:
                        D.itemset((node.ID * 6 + 5, 0), D1[D1_indices.index(node.ID * 6 + 5), 0])
                        V.itemset((node.ID * 6 + 5, 0), V1[D1_indices.index(node.ID * 6 + 5), 0])
                        A.itemset((node.ID * 6 + 5, 0), A1[D1_indices.index(node.ID * 6 + 5), 0])

                # Save the all the maximum global displacement vectors for each load frequency

                D_temp[:, k] = D[:, 0]
                V_temp[:, k] = V[:, 0]
                A_temp[:, k] = A[:, 0]
                k += 1
            self._DISPLACEMENT_REAL = real(D_temp)
            self._DISPLACEMENT_IMAGINARY = imag(D_temp)
            self._VELOCITY_REAL = real(V_temp)
            self._VELOCITY_IMAGINARY = imag(V_temp)
            self._ACCELERATION_REAL = real(A_temp)
            self._ACCELERATION_IMAGINARY = imag(A_temp)

        except:
            raise Exception("'The stiffness matrix is singular, which implies rigid body motion."
                            "The structure is unstable. Aborting analysis.")

        # Put the displacements for the first load frequency into each node
        for node in self.Nodes.values():
            node.DX[harmonic_combo] = real(D_temp[node.ID * 6 + 0, 0])
            node.DY[harmonic_combo] = real(D_temp[node.ID * 6 + 1, 0])
            node.DZ[harmonic_combo] = real(D_temp[node.ID * 6 + 2, 0])
            node.RX[harmonic_combo] = real(D_temp[node.ID * 6 + 3, 0])
            node.RY[harmonic_combo] = real(D_temp[node.ID * 6 + 4, 0])
            node.RZ[harmonic_combo] = real(D_temp[node.ID * 6 + 5, 0])

        # Calculate reactions
        # The damping models "rayleigh" and "modal" do not offer methods for specifying the complete
        # damping matrix. As a result, the reaction forces are solely computed based on elastic and
        # inertial forces. I am still researching on this matter to determine the appropriate
        # approach for addressing this.

        if sparse:
            R =  M_total.tocsr() @ (self._ACCELERATION_REAL + 1j * self._ACCELERATION_IMAGINARY) \
                 + K_total.tocsr() @ (self._DISPLACEMENT_REAL + 1j * self._DISPLACEMENT_IMAGINARY) \
                 - F_total

        else:
            R = M_total @ (self._ACCELERATION_REAL + 1j * self._ACCELERATION_IMAGINARY) \
                + K_total @ (self._DISPLACEMENT_REAL + 1j * self._DISPLACEMENT_IMAGINARY) \
                - F_total

        # Save the reactions
        self._REACTIONS_REAL = real(R)
        self._REACTIONS_IMAGINARY = imag(R)

        # Put the reactions at the last time step into the constrained nodes
        for node in self.Nodes.values():
            if node.support_DX == True:
                node.RxnFX[harmonic_combo] = R[node.ID * 6 + 0, -1]
            else:
                node.RxnFX[harmonic_combo] = 0.0
            if node.support_DY == True:
                node.RxnFY[harmonic_combo] = R[node.ID * 6 + 1, -1]
            else:
                node.RxnFY[harmonic_combo] = 0.0
            if node.support_DZ == True:
                node.RxnFZ[harmonic_combo] = R[node.ID * 6 + 2, -1]
            else:
                node.RxnFZ[harmonic_combo] = 0.0
            if node.support_RX == True:
                node.RxnMX[harmonic_combo] = R[node.ID * 6 + 3, -1]
            else:
                node.RxnMX[harmonic_combo] = 0.0
            if node.support_RY == True:
                node.RxnMY[harmonic_combo] = R[node.ID * 6 + 4, -1]
            else:
                node.RxnMY[harmonic_combo] = 0.0
            if node.support_RZ == True:
                node.RxnMZ[harmonic_combo] = R[node.ID * 6 + 5, -1]
            else:
                node.RxnMZ[harmonic_combo] = 0.0

        if log:
            print('')
            print('- Analysis complete')
            print('')

        # Save the load combination under which the results can be viewed
        self.FRA_combo_name = harmonic_combo

        # Flag the model as solved
        self.DynamicSolution['Harmonic'] = True

    def analyze_linear_time_history_newmark_beta(self, analysis_method = 'direct', combo_name=None, AgX=None,
                                                 AgY=None, AgZ=None, step_size = 0.01, response_duration = 1,
                                                 newmark_gamma = 1/2, newmark_beta = 1/4, sparse=True,
                                                 log = False, d0 = None, v0 = None,
                                                 damping_options = dict(), type_of_mass_matrix='consistent'):

        """
            Perform linear time history analysis using the Newmark-Beta method by either direct or modal decomposition
            methods.
            If modal superposition, the analysis should be preceded by a modal analysis step.

            :param analysis_method: The analysis method to use ('direct' or 'modal'). Default is 'direct'.
            :type analysis_method: str, optional

            :param combo_name: Name of the load combination. Default is None.
            :type combo_name: str, optional

            :param AgX: Seismic ground acceleration data for X-axis as a 2D numpy array with time and acceleration values. Default is None.
            :type AgX: ndarray, optional

            :param AgY: Seismic ground acceleration data for Y-axis as a 2D numpy array with time and acceleration values. Default is None.
            :type AgY: ndarray, optional

            :param AgZ: Seismic ground acceleration data for Z-axis as a 2D numpy array with time and acceleration values. Default is None.
            :type AgZ: ndarray, optional

            :param step_size: Time step size for the analysis. Default is 0.01 seconds.
            :type step_size: float, optional

            :param response_duration: Duration of the analysis in seconds. Default is 1 second.
            :type response_duration: float, optional

            :param newmark_gamma: Newmark-Beta gamma parameter. Default is 1/2.
            :type newmark_gamma: float, optional

            :param newmark_beta: Newmark-Beta beta parameter. Default is 1/4.
            :type newmark_beta: float, optional

            :param sparse: Use sparse matrix representations. Default is True.
            :type sparse: bool, optional

            :param log: Enable verbose logging. Default is False.
            :type log: bool, optional

            :param d0: Initial displacements. Default is None.
            :type d0: ndarray, optional

            :param v0: Initial velocities. Default is None.
            :type v0: ndarray, optional

            :param damping_options: Dictionary of damping options for the analysis. Default is zero damping.
            :type damping_options: dict, optional
                \nAllowed keywords in damping_options:
                - 'constant_modal_damping' (float): Constant modal damping ratio (default: 0.00).
                - 'r_alpha' (float): Rayleigh mass proportional damping coefficient.
                - 'r_beta' (float): Rayleigh stiffness proportional damping coefficient.
                - 'first_mode_damping' (float): Damping ratio for the first mode.
                - 'highest_mode_damping' (float): Damping ratio for the highest mode.
                - 'damping_in_every_mode' (list or tuple): Damping ratio(s) for each mode.

            :param type_of_mass_matrix: Type of mass matrix ('consistent' or 'lumped'). Default is 'consistent'.
            :type type_of_mass_matrix: str, optional

            :raises DampingOptionsKeyWordError: If an incorrect damping keyword is used.
            :raises DynamicLoadNotDefinedError: If no dynamic load combination or ground acceleration data is provided.
            :raises ResultsNotFoundError: If modal results are not available.
            :raises ValueError: If input data is not in the expected format.

            :return: None
         """

        if log:
            print('Checking parameters for time history analysis')

        # Check if enter load combination name exists
        if combo_name!=None and combo_name not in self.LoadCombos.keys():
            raise ValueError("'"+combo_name+"' is not among the defined load combinations")

        # Check if the analysis method name is correct
        if analysis_method!='direct' and analysis_method!='modal':
            raise ValueError("Allowed analysis methods are 'modal' and 'direct'")

        # Check if modal results are available if modal superposition approach is selected
        if self.DynamicSolution['Modal'] == False and analysis_method == 'modal':
            raise ResultsNotFoundError('Modal results are not available. Modal superposition method should be preceded by a modal analysis step')

        # Check if correct keyword in damping options has been used
        if analysis_method == 'direct':
            accepted_damping_key_words = ['r_alpha','r_beta']
            for damping_key_word in damping_options.keys():
                if damping_key_word not in accepted_damping_key_words:
                    raise DampingOptionsKeyWordError("For direct analysis, allowed damping keywords in damping options are 'r_alpha' and 'r_beta'")
        else:
            accepted_damping_key_words = ['constant_modal_damping',
                                          'r_alpha',
                                          'r_beta',
                                          'first_mode_damping',
                                          'highest_mode_damping',
                                          'damping_in_every_mode']
            for damping_key_word in damping_options.keys():
                if damping_key_word not in accepted_damping_key_words:
                    raise DampingOptionsKeyWordError


        # Get the auxiliary list used to determine how the matrices will be partitioned
        Analysis._renumber(self)
        D1_indices, D2_indices, D2_for_check = Analysis._partition_D(self)

        # Check if the dynamic load combination or at least one seismic ground acceleration has been given
        # Or at least one nonzero displacement at a node
        if combo_name == None and AgX is None and AgY is None and AgZ is None:
            if any(D2_for_check) == False:
                raise DynamicLoadNotDefinedError

        # If only a seismic load has been provided, add default load combination 'THA combo'
        # Define its profile too
        combo_exists = True
        if combo_name==None:
            combo_exists = False
            combo_name = 'THA combo'
            self.LoadCombos[combo_name] = LoadCombo(name=combo_name, factors={'Case 1':1}, combo_tags='THA')
            self.LoadProfiles['Case 1'] = LoadProfile(load_case_name='Case 1',time = [0,response_duration],profile=[0,0])

        # Calculate the required number of time history steps
        total_steps = ceil(response_duration/step_size)+1

        # Check if profiles for all load cases in the load combination are defined
        load_profiles = self.LoadProfiles.keys()
        for case in self.LoadCombos[combo_name].factors.keys():
            if case not in load_profiles:
                raise DefinitionNotFoundError('Profile for load case '+str(case)+' has not been defined')

        # Extract the Rayleigh damping coefficients
        if analysis_method=='direct':
            if 'r_alpha' in damping_options:
                r_alpha = damping_options['r_alpha']
            else:
                r_alpha = 0

            if 'r_beta' in damping_options:
                r_beta = damping_options['r_beta']
            else:
                r_beta = 0

        if log:
            print('Preparing parameters for time history analysis')

        # Edit the load profiles so that they are defined for the entire duration of analysis
        for disp_profile in self.DisplacementProfiles.values():
            node = disp_profile.node
            dir = disp_profile.direction
            time : list = disp_profile.time
            profile: list = disp_profile.profile

            # Check if the profile has not been defined for the entire duration of analysis
            if time[-1] < response_duration:

                # We want to bring the displacement profile to zero immediately after the last given value
                # We don't want to define two values at the same instance of time
                # It can cause problems during interpolation
                # So we will begin our definition slightly later
                t = 1.0001 * time[-1]
                if t < response_duration:
                    time.extend([t,response_duration])
                    profile.extend([0,0])

                # Redefine the profile
                self.LoadProfiles[node] = DisplacementProfile(node,dir,time,profile)

        # Edit the displacement profiles so that they are defined for the entire duration of the response
        for load_profile in self.LoadProfiles.values():
            case = load_profile.load_case_name
            time : list = load_profile.time
            profile: list = load_profile.profile

            # Check if the profile has not been defined for the entire duration of analysis
            if time[-1] < response_duration:

                # We want to bring the load profile to zero immediately after the last given value
                # We don't want to define two values at the same instance of time
                # It can cause problems during interpolation
                # So we will begin our definition slightly later
                t = 1.0001 * time[-1]
                if t < response_duration:
                    time.extend([t,response_duration])
                    profile.extend([0,0])

                # Redefine the profile
                self.LoadProfiles[case] = LoadProfile(case,time,profile)

        # Extend the AgX seismic input definition too if it is less than the response duration
        if AgX is not None:
            if isinstance(AgX, ndarray):
                if AgX.shape[0] == 2 or AgX.shape[1] == 2:
                    if AgX.shape[1] == 2:
                        AgX = AgX.T

                    time, a_g = AgX[0,:], AgX[1,:]
                    if time[-1] < response_duration:

                        # We want to bring the load profile to zero immediately after the last given value
                        # We don't want to define two values at the same instance of time
                        # It can cause problems during interpolation
                        # So we will begin our definition slightly later
                        t = 1.0001 * time[-1]
                        if t < response_duration:
                            time = append(time, [t, response_duration])
                            a_g = append(a_g, [0, 0])

                        # Redefine seismic input
                        AgX = array([time,a_g])

                else:
                    raise ValueError ("AgX must have two rows, first one for time and second one for ground acceleration")
            else:
                raise ValueError("AgX must be a numpy array")

        # Extend the AgY seismic input definition too if it is less than the response duration
        if AgY is not None:
            if isinstance(AgY, ndarray):
                if AgY.shape[0] == 2 or AgY.shape[1] == 2:
                    if AgY.shape[1] == 2:
                        AgY = AgY.T

                    time, a_g = AgY[0, :], AgY[1, :]
                    if time[-1] < response_duration:

                        # We want to bring the load profile to zero immediately after the last given value
                        # We don't want to define two values at the same instance of time
                        # It can cause problems during interpolation
                        # So we will begin our definition slightly later
                        t = 1.0001 * time[-1]
                        if t < response_duration:
                            time = append(time, [t, response_duration])
                            a_g = append(a_g, [0, 0])

                        # Redefine seismic input
                        AgY = array([time, a_g])

                else:
                    raise ValueError(
                        " AgY must have two rows, first one for time and second one for ground acceleration")
            else:
                raise ValueError("AgY must be a numpy array")

        # Extend the AgZ seismic input definition too if it is less than the response duration
        if AgZ is not None:
            if isinstance(AgZ, ndarray):
                if AgZ.shape[0] == 2 or AgZ.shape[1] == 2:
                    if AgZ.shape[1] == 2:
                        AgZ = AgZ.T

                    time, a_g = AgZ[0, :], AgZ[1, :]
                    if time[-1] < response_duration:

                        # We want to bring the load profile to zero immediately after the last given value
                        # We don't want to define two values at the same instance of time
                        # It can cause problems during interpolation
                        # So we will begin our definition slightly later
                        t = 1.0001 * time[-1]
                        if t < response_duration:
                            time = append(time, [t, response_duration])
                            a_g = append(a_g, [0, 0])

                        # Redefine seismic input
                        AgZ = array([time, a_g])

                else:
                    raise ValueError(
                        " AgZ must have two rows, first one for time and second one for ground acceleration")
            else:
                raise ValueError("AgZ must be a numpy array")

        # Prepare the model
        Analysis._prepare_model(self)
        D1_indices, D2_indices, D2_for_check = Analysis._partition_D(self)

        # Get the partitioned matrices
        if sparse == True:
            K11, K12, K21, K22 = self._partition(self.K(combo_name, log, False, sparse).tolil(), D1_indices,D2_indices)
            M11, M12, M21, M22 = self._partition(self.M(combo_name, log, False, sparse,type_of_mass_matrix).tolil(),
                                                 D1_indices, D2_indices)
            K_total = self.K(combo_name, log, False, sparse).tolil()
            M_total = self.M(combo_name, log, False, sparse,type_of_mass_matrix).tolil()
        else:
            K11, K12, K21, K22 = self._partition(self.K(combo_name, log, False, sparse), D1_indices, D2_indices)
            M11, M12, M21, M22 = self._partition(self.M(combo_name, log, False, sparse, type_of_mass_matrix),
                                                 D1_indices, D2_indices)
            K_total = self.K(combo_name, log, False, sparse)
            M_total = self.M(combo_name, log, False, sparse, type_of_mass_matrix)


        if log:
            print('- Building the loading time history')

        # Initialise load vector for the entire analysis duration
        F = zeros((len(D1_indices), total_steps))

        # Initialise second load vector. This will store the entire load vector which will be used for
        # calculating the reactions
        total_dof = len(D1_indices) + len(D2_indices)
        F_total = zeros((total_dof, total_steps))

        # Initialise D2, V2 and A2 for the entire analysis duration
        D2 = zeros((len(D2_for_check),total_steps))
        V2 = zeros((len(D2_for_check), total_steps))
        A2 = zeros((len(D2_for_check), total_steps))

        # Build the influence vectors used to define the earthquake force
        # Influence vectors are used to select the degrees of freedom in
        # the model stimulated by the earthquake

        i = 0
        unp_influence_X = zeros((total_dof,1))
        unp_influence_Y = zeros((total_dof,1))
        unp_influence_Z = zeros((total_dof,1))
        while i < total_dof:
            unp_influence_X[i+0,0] = 1
            unp_influence_Y[i+1,0] = 1
            unp_influence_Z[i+2,0] = 1
            i += 6

        # Partition the influence vectors
        influence_X = self._partition(unp_influence_X, D1_indices, D2_indices)[0]
        influence_Y = self._partition(unp_influence_Y, D1_indices, D2_indices)[0]
        influence_Z = self._partition(unp_influence_Z, D1_indices, D2_indices)[0]


        # We want to build the loading history, i.e load for each time step
        # Since each load case has its own loading profile or function, we want to calculate
        # the nodal and fixed end forces separately for each load case
        # We can do this by creating separate load combinations for each case
        # These temporary load combinations only consist of one load case
        # Hence we need to add combinations to add combinations to the existing load combinations dictionary
        # Since we are modifying the models load combinations dictionary, we must make a copy of it
        # This copy will be used to restore the original

        original_load_combo = copy.deepcopy(self.LoadCombos)

        # We initialise a dictionary that will hold the nodal and fixed end forces for each load case
        # We want to compute these vectors once and for all, as opposed to doing it for each time step
        # Doing this for each time step is extremely slow
        # We actually initialise two, one to hold the partitioned and the other to hold the total
        P_and_FER = dict()
        unp_P_and_FER = dict()

        # We compute the nodal and fixed end forces for each load case
        for case_name in self.LoadCombos[combo_name].factors.keys():

            # We create a temporary load combination that consists of only one load case
            temp_combo = LoadCombo(name=combo_name,
                                        factors={case_name: original_load_combo[combo_name].factors[case_name]})

            # We add this temporary load combination to the models load combination dictionary
            self.LoadCombos[case_name] = temp_combo

            # We finally compute the nodal and fixed end forces for this load combination, and, essentially
            # for this load case
            P_and_FER_temp = self.P(case_name)[:,0] - self.FER(case_name)[:,0]

            # Save into the total force dictionary
            unp_P_and_FER[case_name] = P_and_FER_temp

            # Save into the partitioned force dictionary
            P_and_FER[case_name] = self._partition(P_and_FER_temp.reshape(total_dof,1),
                                                   D1_indices,D2_indices)[0]

        # We restore the original load combination
        self.LoadCombos = original_load_combo


        # We create a list of time instances from 0 to the duration of the analysis
        expanded_time = linspace(0,response_duration,total_steps)

        # We calculate the load vectors for each case for each time instance, and sum them up

        # FOR DEBUGGING
        combo_exists = True

        if combo_exists:
            for case_name in self.LoadCombos[combo_name].factors.keys():
                # Extract the time vector from the load profile definition for this load case
                time_list = self.LoadProfiles[case_name].time

                # Extract the load profile for this load case
                profile_list = self.LoadProfiles[case_name].profile

                # Interpolate the load profiles
                interpolated_profile = interp(expanded_time, time_list, profile_list)

                # Multiply the nodal and fixed end forces with the interpolated profile
                # Then add this contribution to the model's load vectors (partitioned and total)
                F_total += outer(unp_P_and_FER[case_name], interpolated_profile)
                F += outer(P_and_FER[case_name], interpolated_profile)

        # If ground acceleration in the X direction has been given, calculate the corresponding forces
        if AgX is not None:
            # Interpolate the ground acceleration
            interpolated_AgX = interp(expanded_time,AgX[0,:],AgX[1,:])
            if sparse:
                AgX_F = -M11.tocsr() @ outer(influence_X ,interpolated_AgX)
                unp_AgX_F = -M_total.tocsr() @ outer(unp_influence_X, interpolated_AgX)
            else:
                AgX_F = -M11 @ influence_X @ outer(influence_X ,interpolated_AgX)
                unp_AgX_F = -M_total @ influence_X @ outer(unp_influence_X, interpolated_AgX)

            # Add to the models partitioned and total force vectors
            F_total += unp_AgX_F
            F += AgX_F

        # If ground acceleration in the X direction has been given, calculate the corresponding forces
        if AgY is not None:
            # Interpolate the ground acceleration
            interpolated_AgY = interp(expanded_time,AgY[0,:],AgY[1,:])
            if sparse:
                AgY_F = -M11.tocsr() @ outer(influence_Y ,interpolated_AgY)
                unp_AgY_F = -M_total.tocsr() @ outer(unp_influence_Y, interpolated_AgY)
            else:
                AgY_F = -M11 @ influence_Y @ outer(influence_Y ,interpolated_AgY)
                unp_AgY_F = -M_total @ influence_Y @ outer(unp_influence_Y, interpolated_AgY)

            # Add to the models partitioned and total force vectors
            F_total += unp_AgY_F
            F += AgY_F

        # If ground acceleration in the X direction has been given, calculate the corresponding forces
        if AgZ is not None:
            # Interpolate the ground acceleration
            interpolated_AgZ = interp(expanded_time,AgZ[0,:],AgZ[1,:])
            if sparse:
                AgZ_F = -M11.tocsr() @ outer(influence_Z ,interpolated_AgZ)
                unp_AgZ_F = -M_total.tocsr() @ outer(unp_influence_Z, interpolated_AgZ)
            else:
                AgZ_F = -M11 @ influence_Z @ outer(influence_Z ,interpolated_AgZ)
                unp_AgZ_F = -M_total @ influence_Z @ outer(unp_influence_Z, interpolated_AgZ)

            # Add to the models partitioned and total force vectors
            F_total += unp_AgZ_F
            F += AgZ_F

        # Save the total global force
        self.F_TOTAL = F_total


        # Get the mode shapes and natural frequencies if modal analysis method is specified
        if analysis_method == 'modal':
            Z = self._mass_normalised_mode_shapes(M11, self._eigen_vectors)

            # Get the natural frequencies
            w = 2 * pi * self.NATURAL_FREQUENCIES()

        # Get the initial values of displacements and velocities
        # If not given, set them to zero
        if d0 == None:
            d0_phy = zeros(len(D1_indices))
        else:
            d0_phy, d02 = self._partition(d0, D1_indices, D2_indices)

        if v0 == None:
            v0_phy = zeros(len(D1_indices))
        else:
            v0_phy, v02 = self._partition(v0, D1_indices, D2_indices)

        # Find the corresponding quantities in the modal coordinate system if modal superposition method is requested
        if analysis_method == 'modal':
            d0_n = solve(Z.T @ Z, Z.T @ d0_phy)
            v0_n = solve(Z.T @ Z, Z.T @ v0_phy)
            F_n = Z.T @ F

        # Run the analysis

        if log:
            print('')
            print('+-------------------------+')
            print('| Analyzing: Time History |')
            print('+-------------------------+')

        try:
            if analysis_method == 'direct':

                TIME, D1, V1, A1 = \
                     Analysis._transient_solver_linear_direct(K=K11, M=M11,d0=d0_phy,v0=v0_phy,
                                                              F0=F[:,0],F = F,step_size=step_size,
                                                              required_duration=response_duration,
                                                              newmark_gamma=newmark_gamma,
                                                              newmark_beta=newmark_beta,
                                                              taylor_alpha=0, wilson_theta=1,
                                                              rayleigh_alpha=r_alpha, rayleigh_beta=r_beta,
                                                              sparse=sparse,log=log)

            else:
                TIME, D1, V1, A1 = \
                    Analysis._transient_solver_linear_modal(d0_n=d0_n, v0_n=v0_n, F0_n=F_n[:, 0],
                                                            F_n=F_n, step_size=step_size,
                                                            required_duration=response_duration,
                                                            mass_normalised_eigen_vectors=Z,
                                                            natural_freq=w, newmark_gamma=newmark_gamma,
                                                            newmark_beta=newmark_beta, taylor_alpha=0,
                                                            wilson_theta=1, damping_options=damping_options,
                                                            log=log)
        except:
            raise Exception("Out of memory")

        # Form the global response vectors D, V and A
        D = zeros((len(self.Nodes) * 6, total_steps))
        V = zeros((len(self.Nodes) * 6, total_steps))
        A = zeros((len(self.Nodes) * 6, total_steps))

        for node in self.Nodes.values():
            if D2_indices.count(node.ID * 6 + 0) == 1:
                D[node.ID * 6 + 0, :] = D2[D2_indices.index(node.ID * 6 + 0), :]
                V[node.ID * 6 + 0, :] = V2[D2_indices.index(node.ID * 6 + 0), :]
                A[node.ID * 6 + 0, :] = A2[D2_indices.index(node.ID * 6 + 0), :]
            else:
                D[node.ID * 6 + 0, :] = D1[D1_indices.index(node.ID * 6 + 0), :]
                V[node.ID * 6 + 0, :] = V1[D1_indices.index(node.ID * 6 + 0), :]
                A[node.ID * 6 + 0, :] = A1[D1_indices.index(node.ID * 6 + 0), :]

            if D2_indices.count(node.ID * 6 + 1) == 1:
                D[node.ID * 6 + 1, :] = D2[D2_indices.index(node.ID * 6 + 1), :]
                V[node.ID * 6 + 1, :] = V2[D2_indices.index(node.ID * 6 + 1), :]
                A[node.ID * 6 + 1, :] = A2[D2_indices.index(node.ID * 6 + 1), :]
            else:
                D[node.ID * 6 + 1, :] = D1[D1_indices.index(node.ID * 6 + 1), :]
                V[node.ID * 6 + 1, :] = V1[D1_indices.index(node.ID * 6 + 1), :]
                A[node.ID * 6 + 1, :] = A1[D1_indices.index(node.ID * 6 + 1), :]

            if D2_indices.count(node.ID * 6 + 2) == 1:
                D[node.ID * 6 + 2, :] = D2[D2_indices.index(node.ID * 6 + 2), :]
                V[node.ID * 6 + 2, :] = V2[D2_indices.index(node.ID * 6 + 2), :]
                A[node.ID * 6 + 2, :] = A2[D2_indices.index(node.ID * 6 + 2), :]
            else:
                D[node.ID * 6 + 2, :] = D1[D1_indices.index(node.ID * 6 + 2), :]
                V[node.ID * 6 + 2, :] = V1[D1_indices.index(node.ID * 6 + 2), :]
                A[node.ID * 6 + 2, :] = A1[D1_indices.index(node.ID * 6 + 2), :]

            if D2_indices.count(node.ID * 6 + 3) == 1:
                D[node.ID * 6 + 3, :] = D2[D2_indices.index(node.ID * 6 + 3), :]
                V[node.ID * 6 + 3, :] = V2[D2_indices.index(node.ID * 6 + 3), :]
                A[node.ID * 6 + 3, :] = A2[D2_indices.index(node.ID * 6 + 3), :]
            else:
                D[node.ID * 6 + 3, :] = D1[D1_indices.index(node.ID * 6 + 3), :]
                V[node.ID * 6 + 3, :] = V1[D1_indices.index(node.ID * 6 + 3), :]
                A[node.ID * 6 + 3, :] = A1[D1_indices.index(node.ID * 6 + 3), :]

            if D2_indices.count(node.ID * 6 + 4) == 1:
                D[node.ID * 6 + 4, :] = D2[D2_indices.index(node.ID * 6 + 4), :]
                V[node.ID * 6 + 4, :] = V2[D2_indices.index(node.ID * 6 + 4), :]
                A[node.ID * 6 + 4, :] = A2[D2_indices.index(node.ID * 6 + 4), :]
            else:
                D[node.ID * 6 + 4, :] = D1[D1_indices.index(node.ID * 6 + 4), :]
                V[node.ID * 6 + 4, :] = V1[D1_indices.index(node.ID * 6 + 4), :]
                A[node.ID * 6 + 4, :] = A1[D1_indices.index(node.ID * 6 + 4), :]

            if D2_indices.count(node.ID * 6 + 5) == 1:
                D[node.ID * 6 + 5, :] = D2[D2_indices.index(node.ID * 6 + 5), :]
                V[node.ID * 6 + 5, :] = V2[D2_indices.index(node.ID * 6 + 5), :]
                A[node.ID * 6 + 5, :] = A2[D2_indices.index(node.ID * 6 + 5), :]
            else:
                D[node.ID * 6 + 5, :] = D1[D1_indices.index(node.ID * 6 + 5), :]
                V[node.ID * 6 + 5, :] = V1[D1_indices.index(node.ID * 6 + 5), :]
                A[node.ID * 6 + 5, :] = A1[D1_indices.index(node.ID * 6 + 5), :]

        # Save the responses
        self._TIME_THA = TIME
        self._DISPLACEMENT_THA = D
        self._VELOCITY_THA = V
        self._ACCELERATION_THA = A

        # Put the displacements at the end of the analysis into each node
        for node in self.Nodes.values():
            node.DX[combo_name] = D[node.ID * 6 + 0, -1]
            node.DY[combo_name] = D[node.ID * 6 + 1, -1]
            node.DZ[combo_name] = D[node.ID * 6 + 2, -1]
            node.RX[combo_name] = D[node.ID * 6 + 3, -1]
            node.RY[combo_name] = D[node.ID * 6 + 4, -1]
            node.RZ[combo_name] = D[node.ID * 6 + 5, -1]

        # Calculate reactions
        # The damping models "rayleigh" and "modal" do not offer methods for specifying the complete
        # damping matrix. As a result, the reaction forces are solely computed based on elastic and
        # inertial forces. I am still researching on this matter to determine the appropriate
        # approach for addressing it.
        if sparse:
            R = M_total.tocsr() @ self._ACCELERATION_THA\
                + K_total.tocsr() @ self._DISPLACEMENT_THA\
                - self.F_TOTAL
        else:
            R = M_total @ self._ACCELERATION_THA \
                + K_total @ self._DISPLACEMENT_THA \
                - self.F_TOTAL

        # Save the reactions
        self._REACTIONS_THA = R

        # Put the reactions at the last time step into the constrained nodes
        for node in self.Nodes.values():
            if node.support_DX == True:
                node.RxnFX[combo_name] = R[node.ID * 6 + 0, -1]
            else:
                node.RxnFX[combo_name] = 0.0
            if node.support_DY == True:
                node.RxnFY[combo_name] = R[node.ID * 6 + 1, -1]
            else:
                node.RxnFY[combo_name] = 0.0
            if node.support_DZ == True:
                node.RxnFZ[combo_name] = R[node.ID * 6 + 2, -1]
            else:
                node.RxnFZ[combo_name] = 0.0
            if node.support_RX == True:
                node.RxnMX[combo_name] = R[node.ID * 6 + 3, -1]
            else:
                node.RxnMX[combo_name] = 0.0
            if node.support_RY == True:
                node.RxnMY[combo_name] = R[node.ID * 6 + 4, -1]
            else:
                node.RxnMY[combo_name] = 0.0
            if node.support_RZ == True:
                node.RxnMZ[combo_name] = R[node.ID * 6 + 5, -1]
            else:
                node.RxnMZ[combo_name] = 0.0

        if log:
            print('')
            print('Analysis Complete')
            print('')

        # Save the load combination name under which the results can be viewed
        self.THA_combo_name = combo_name

        # Flag the model as solved
        self.DynamicSolution['Time History'] = True


    def analyze_linear_time_history_wilson_theta(self, analysis_method = 'direct', combo_name=None, AgX=None, AgY=None,
                                                 AgZ=None,step_size = 0.01, response_duration = 1,
                                                 wilson_theta = 1.420815, sparse=True, log = False, d0 = None, v0 = None,
                                                 damping_options = dict(), type_of_mass_matrix='consistent'):

        """
            Perform linear time history analysis using the Wilson theta method by either direct or modal decomposition
            methods.
            If modal superposition, the analysis should be preceded by a modal analysis step.

            :param analysis_method: The analysis method to use ('direct' or 'modal'). Default is 'direct'.
            :type analysis_method: str, optional

            :param combo_name: Name of the load combination. Default is None.
            :type combo_name: str, optional

            :param AgX: Seismic ground acceleration data for X-axis as a 2D numpy array with time and acceleration values. Default is None.
            :type AgX: ndarray, optional

            :param AgY: Seismic ground acceleration data for Y-axis as a 2D numpy array with time and acceleration values. Default is None.
            :type AgY: ndarray, optional

            :param AgZ: Seismic ground acceleration data for Z-axis as a 2D numpy array with time and acceleration values. Default is None.
            :type AgZ: ndarray, optional

            :param step_size: Time step size for the analysis. Default is 0.01 seconds.
            :type step_size: float, optional

            :param response_duration: Duration of the analysis in seconds. Default is 1 second.
            :type response_duration: float, optional

            :param wilson_theta: Wilson theta parameter. Default is 1.420815.
            :type wilson_theta: float, optional

            :param sparse: Use sparse matrix representations. Default is True.
            :type sparse: bool, optional

            :param log: Enable verbose logging. Default is False.
            :type log: bool, optional

            :param d0: Initial displacements. Default is None.
            :type d0: ndarray, optional

            :param v0: Initial velocities. Default is None.
            :type v0: ndarray, optional

            :param damping_options: Dictionary of damping options for the analysis. Default is zero damping.
            :type damping_options: dict, optional
                \nAllowed keywords in damping_options:
                - 'constant_modal_damping' (float): Constant modal damping ratio (default: 0.00).
                - 'r_alpha' (float): Rayleigh mass proportional damping coefficient.
                - 'r_beta' (float): Rayleigh stiffness proportional damping coefficient.
                - 'first_mode_damping' (float): Damping ratio for the first mode.
                - 'highest_mode_damping' (float): Damping ratio for the highest mode.
                - 'damping_in_every_mode' (list or tuple): Damping ratio(s) for each mode.

            :param type_of_mass_matrix: Type of mass matrix ('consistent' or 'lumped'). Default is 'consistent'.
            :type type_of_mass_matrix: str, optional

            :raises DampingOptionsKeyWordError: If an incorrect damping keyword is used.
            :raises DynamicLoadNotDefinedError: If no dynamic load combination or ground acceleration data is provided.
            :raises ResultsNotFoundError: If modal results are not available.
            :raises ValueError: If input data is not in the expected format.

            :return: None
         """

        if log:
            print('Checking parameters for time history analysis')

        # Check if enter load combination name exists
        if combo_name!=None and combo_name not in self.LoadCombos.keys():
            raise ValueError("'"+combo_name+"' is not among the defined load combinations")

        # Check if the analysis method name is correct
        if analysis_method!='direct' and analysis_method!='modal':
            raise ValueError("Allowed analysis methods are 'modal' and 'direct'")

        # Check if modal results are available if modal superposition approach is selected
        if self.DynamicSolution['Modal'] == False and analysis_method == 'modal':
            raise ResultsNotFoundError('Modal results are not available. Modal superposition method should be preceded by a modal analysis step')

        # Check if correct keyword in damping options has been used
        if analysis_method == 'direct':
            accepted_damping_key_words = ['r_alpha','r_beta']
            for damping_key_word in damping_options.keys():
                if damping_key_word not in accepted_damping_key_words:
                    raise DampingOptionsKeyWordError("For direct analysis, allowed damping keywords in damping options are 'r_alpha' and 'r_beta'")
        else:
            accepted_damping_key_words = ['constant_modal_damping',
                                          'r_alpha',
                                          'r_beta',
                                          'first_mode_damping',
                                          'highest_mode_damping',
                                          'damping_in_every_mode']
            for damping_key_word in damping_options.keys():
                if damping_key_word not in accepted_damping_key_words:
                    raise DampingOptionsKeyWordError

        # Check if the dynamic load combination or at least one seismic ground acceleration has been given
        if combo_name == None and AgX is None and AgY is None and AgZ is None:
            raise DynamicLoadNotDefinedError

        # If only a seismic load has been provided, add default the load combination 'Combo 1'
        if combo_name==None:
            combo_name = 'Combo 1'
            self.LoadCombos['Combo 1'] = LoadCombo(name='Combo 1', factors={'Case 1':1}, combo_tags='THS')

        # Add tags to the load combinations that do not have tags. We will use this to filter results
        for combo in self.LoadCombos.values():
            if combo.combo_tags == None:
                combo.combo_tags = 'unknown_type'


        # Calculate the required number of time history steps
        total_steps = ceil(response_duration/step_size)+1

        # Check if profiles for all load cases in the load combination are defined
        load_profiles = self.LoadProfiles.keys()
        for case in self.LoadCombos[combo_name].factors.keys():
            if case not in load_profiles:
                raise DefinitionNotFoundError('Profile for load case '+str(case)+' has not been defined')

        if log:
            print('Preparing parameters for time history analysis')

        # Edit the load profiles so that they are defined for the entire duration of analysis
        for load_profile in self.LoadProfiles.values():
            case = load_profile.load_case_name
            time : list = load_profile.time
            profile: list = load_profile.profile

            # Check if the profile has not been defined for the entire duration of analysis
            if time[-1] < response_duration:

                # We want to bring the load profile to zero immediately after the last given value
                # We don't want to define two values at the same instance of time
                # It can cause problems during interpolation
                # So we will begin our definition slightly later
                t = 1.0001 * time[-1]
                if t < response_duration:
                    time.extend([t,response_duration])
                    profile.extend([0,0])

                # Redefine the profile
                self.LoadProfiles[case] = LoadProfile(case,time,profile)


        # Extend the AgX seismic input definition too if it is less than the response duration
        if AgX is not None:
            if isinstance(AgX, ndarray):
                if AgX.shape[0] == 2 or AgX.shape[1] == 2:
                    if AgX.shape[1] == 2:
                        AgX = AgX.T

                    time, a_g = AgX[0,:], AgX[1,:]
                    if time[-1] < response_duration:

                        # We want to bring the load profile to zero immediately after the last given value
                        # We don't want to define two values at the same instance of time
                        # It can cause problems during interpolation
                        # So we will begin our definition slightly later
                        t = 1.0001 * time[-1]
                        if t < response_duration:
                            time = append(time, [t, response_duration])
                            a_g = append(a_g, [0, 0])

                        # Redefine seismic input
                        AgX = array([time,a_g])

                else:
                    raise ValueError ("AgX must have two rows, first one for time and second one for ground acceleration")
            else:
                raise ValueError("AgX must be a numpy array")

        # Extend the AgY seismic input definition too if it is less than the response duration
        if AgY is not None:
            if isinstance(AgY, ndarray):
                if AgY.shape[0] == 2 or AgY.shape[1] == 2:
                    if AgY.shape[1] == 2:
                        AgY = AgY.T

                    time, a_g = AgY[0, :], AgY[1, :]
                    if time[-1] < response_duration:

                        # We want to bring the load profile to zero immediately after the last given value
                        # We don't want to define two values at the same instance of time
                        # It can cause problems during interpolation
                        # So we will begin our definition slightly later
                        t = 1.0001 * time[-1]
                        if t < response_duration:
                            time = append(time, [t, response_duration])
                            a_g = append(a_g, [0, 0])

                        # Redefine seismic input
                        AgY = array([time, a_g])

                else:
                    raise ValueError(
                        " AgY must have two rows, first one for time and second one for ground acceleration")
            else:
                raise ValueError("AgY must be a numpy array")

        # Extend the AgZ seismic input definition too if it is less than the response duration
        if AgZ is not None:
            if isinstance(AgZ, ndarray):
                if AgZ.shape[0] == 2 or AgZ.shape[1] == 2:
                    if AgZ.shape[1] == 2:
                        AgZ = AgZ.T

                    time, a_g = AgZ[0, :], AgZ[1, :]
                    if time[-1] < response_duration:

                        # We want to bring the load profile to zero immediately after the last given value
                        # We don't want to define two values at the same instance of time
                        # It can cause problems during interpolation
                        # So we will begin our definition slightly later
                        t = 1.0001 * time[-1]
                        if t < response_duration:
                            time = append(time, [t, response_duration])
                            a_g = append(a_g, [0, 0])

                        # Redefine seismic input
                        AgZ = array([time, a_g])

                else:
                    raise ValueError(
                        " AgZ must have two rows, first one for time and second one for ground acceleration")
            else:
                raise ValueError("AgZ must be a numpy array")

        # Prepare the model
        Analysis._prepare_model(self)

        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = Analysis._partition_D(self)

        # Make sure D2 is a matrix in 2 dimensions
        D2 = D2.reshape(len(D2), 1)

        # Get the partitioned matrices
        if sparse == True:
            K11, K12, K21, K22 = self._partition(self.K(combo_name, log, False, sparse).tolil(), D1_indices,D2_indices)
            M11, M12, M21, M22 = self._partition(self.M(combo_name, log, False, sparse,type_of_mass_matrix).tolil(),
                                                 D1_indices, D2_indices)
        else:
            K11, K12, K21, K22 = self._partition(self.K(combo_name, log, False, sparse), D1_indices, D2_indices)
            M11, M12, M21, M22 = self._partition(self.M(combo_name, log, False, sparse, type_of_mass_matrix),
                                                 D1_indices, D2_indices)

        # Initialise load vector for the entire analysis duration
        F = zeros((len(D1_indices), total_steps))

        # Build the influence vectors used to define the earthquake force
        # Influence vectors are used to select the degrees of freedom in
        # the model stimulated by the earthquake

        total_dof = len(D1_indices)+len(D2_indices)
        i = 0
        influence_X = zeros((total_dof,1))
        influence_Y = zeros((total_dof,1))
        influence_Z = zeros((total_dof,1))
        while i < total_dof:
            influence_X[i+0,0] = 1
            influence_Y[i+1,0] = 1
            influence_Z[i+2,0] = 1
            i += 6

        # Build the load vector step by step
        # We will make modifications to the load combination in order to define the dynamic load
        # Each step uses the original load combination, modifies it according to the given load
        # profiles, and calculates the load vector for that step
        # Hence we must make a copy of the load combination since each step needs the original
        # load combination. Also at the end of the analysis we want to restore the original load
        # combination
        original_load_combo = copy.deepcopy(self.LoadCombos)
        t = 0
        for i in range(total_steps):
            # Check if a load combination has been given
            if combo_name != None:
                # For each load case in the load combination
                for case_name in self.LoadCombos[combo_name].factors.keys():

                    # Extract the time vector from the load profile definition for this load case
                    time_list = self.LoadProfiles[case_name].time

                    # Extract the load profile for this load case
                    profile_list = self.LoadProfiles[case_name].profile

                    # Interpolate the load profile to find the scaling factor for this time t
                    scaler = interp(t, time_list,profile_list)

                    # Get the load factor from the load combination. We will scale this factor
                    # It would be more correct to scale the actual load value and not its factor
                    # But it's hard to do so especially that some load cases are defined by more
                    # than one value, e.g line member loads are defined by two values
                    # Also the same load case can apply to different load types e.g the same
                    # load case can be used for point loads, line loads, pressures, etc.
                    # Hence, the easiest way to accomplish this is by scaling the load factor
                    current_factor = self.LoadCombos[combo_name].factors[case_name]

                    # Edit the load combination using the changed load factor
                    self.LoadCombos[combo_name].factors[case_name] = scaler * current_factor

                # For this time t, the load combination has been edited to reflect the loading situation
                # at this point in time. Hence the fixed end reactions and nodal loads can be calculated
                # for this point in time
                FER1, FER2 = self._partition(self.FER(combo_name), D1_indices, D2_indices)
                P1, P2 = self._partition(self.P(combo_name), D1_indices, D2_indices)

                # Save the calculated nodal loads and fixed end reactions into the load vector
                F[:,i] = P1[:,0] - FER1[:,0]

            # Add the seismic forces as well if the ground acceleration has been given
            # F_s = -[M]{i}a_g where {i} is the influence vector and [M] is the mass matrix
            # and a_g is the ground acceleration
            # Interpolation is used again to find the ground acceleration at time t
            if AgX is not None:
                AgX_F = -M11 @ self._partition(influence_X , D1_indices, D2_indices)[0] \
                        * interp(t, AgX[0, :], AgX[1, :])
                F[:,i] += AgX_F[0,:]

            if AgY is not None:
                AgY_F = -M11 @ self._partition(influence_Y , D1_indices, D2_indices)[0] \
                         * interp(t, AgY[0, :], AgY[1, :])
                F[:,i] += AgY_F[0,:]

            if AgZ is not None:
                AgZ_F = -M11 @ self._partition(influence_Z , D1_indices, D2_indices)[0] \
                         * interp(t, AgZ[0,:], AgZ[1,:])
                F[:,i] += AgZ_F[0,:]

            # Update the time
            t += step_size

            # Restore the time load combination to the original one
            self.LoadCombos = copy.deepcopy(original_load_combo)

        # Update force vector to include the effects of the constrained degrees of freedom
        # In theory we also need to add -M12 @ A2 and -C12 @ V2.
        # However, the constrained displacements don't really change in time for normal
        # structural analysis
        # Hence, the velocity and accelerations will be zero
        F[:,:] -= K12 @ D2

        # Get the mode shapes and natural frequencies if modal analysis method is specified
        if analysis_method == 'modal':
            Z = self._mass_normalised_mode_shapes(M11, self._eigen_vectors)

            # Get the natural frequencies
            w = 2 * pi * self.NATURAL_FREQUENCIES()

        # Get the initial values of displacements and velocities
        # If not given, set them to zero
        if d0 == None:
            d0_phy = zeros(len(D1_indices))
        else:
            d0_phy, d02 = self._partition(d0, D1_indices, D2_indices)

        if v0 == None:
            v0_phy = zeros(len(D1_indices))
        else:
            v0_phy, v02 = self._partition(v0, D1_indices, D2_indices)

        # Find the corresponding quantities the modal coordinate system if modal superposition method is requested
        if analysis_method == 'modal':
            d0_n = solve(Z.T @ Z, Z.T @ d0_phy)
            v0_n = solve(Z.T @ Z, Z.T @ v0_phy)
            F_n = Z.T @ F

        # Run the analysis
        if log:
            print('+-------------------------+')
            print('| Analyzing: Time History |')
            print('+-------------------------+')

        try:
            if analysis_method == 'direct':
                # Extract the Rayleigh damping coefficients
                if 'r_alpha' in damping_options:
                    r_alpha = damping_options['r_alpha']
                else:
                    r_alpha = 0

                if 'r_beta' in damping_options:
                    r_beta = damping_options['r_beta']
                else:
                    r_beta = 0

                TIME, D1, V1, A1 = \
                     Analysis._transient_solver_linear_direct(K=K11, M=M11,d0=d0_phy,v0=v0_phy,
                                                              F0=F[:,0],F = F,step_size=step_size,
                                                              required_duration=response_duration,
                                                              newmark_gamma=1/2, newmark_beta=1/6,
                                                              taylor_alpha=0, wilson_theta=wilson_theta,
                                                              rayleigh_alpha=r_alpha, rayleigh_beta=r_beta,
                                                              sparse=sparse,log=log)

            else:
                TIME, D1, V1, A1 = \
                    Analysis._transient_solver_linear_modal(d0_n=d0_n, v0_n=v0_n, F0_n=F_n[:, 0], F_n=F_n,
                                                            step_size=step_size,
                                                            required_duration=response_duration,
                                                            mass_normalised_eigen_vectors=Z,
                                                            natural_freq=w, newmark_gamma=1/2,
                                                            newmark_beta=1/6, taylor_alpha=0,
                                                            wilson_theta=wilson_theta,
                                                            damping_options=damping_options,
                                                            log=log)
        except:
            raise Exception("Out of memory")

        # Form the global response vectors D, V and A
        D = zeros((len(self.Nodes) * 6, total_steps))
        V = zeros((len(self.Nodes) * 6, total_steps))
        A = zeros((len(self.Nodes) * 6, total_steps))

        # The accelerations and velocities for the constrained degrees of freedom will be zeros
        V2 = zeros(D2.shape)
        A2 = zeros(D2.shape)

        for node in self.Nodes.values():

            if D2_indices.count(node.ID * 6 + 0) == 1:
                D.itemset((node.ID * 6 + 0, 0), D2[D2_indices.index(node.ID * 6 + 0), 0])
                V.itemset((node.ID * 6 + 0, 0), V2[D2_indices.index(node.ID * 6 + 0), 0])
                A.itemset((node.ID * 6 + 0, 0), A2[D2_indices.index(node.ID * 6 + 0), 0])
            else:
                D[node.ID * 6 + 0, :] = D1[D1_indices.index(node.ID * 6 + 0), :]
                V[node.ID * 6 + 0, :] = V1[D1_indices.index(node.ID * 6 + 0), :]
                A[node.ID * 6 + 0, :] = A1[D1_indices.index(node.ID * 6 + 0), :]
            if D2_indices.count(node.ID * 6 + 1) == 1:
                D.itemset((node.ID * 6 + 1, 0), D2[D2_indices.index(node.ID * 6 + 1), 0])
                V.itemset((node.ID * 6 + 1, 0), V2[D2_indices.index(node.ID * 6 + 1), 0])
                A.itemset((node.ID * 6 + 1, 0), A2[D2_indices.index(node.ID * 6 + 1), 0])
            else:
                D[node.ID * 6 + 1, :] = D1[D1_indices.index(node.ID * 6 + 1), :]
                V[node.ID * 6 + 1, :] = V1[D1_indices.index(node.ID * 6 + 1), :]
                A[node.ID * 6 + 1, :] = A1[D1_indices.index(node.ID * 6 + 1), :]

            if D2_indices.count(node.ID * 6 + 2) == 1:
                D.itemset((node.ID * 6 + 2, 0), D2[D2_indices.index(node.ID * 6 + 2), 0])
                V.itemset((node.ID * 6 + 2, 0), V2[D2_indices.index(node.ID * 6 + 2), 0])
                A.itemset((node.ID * 6 + 2, 0), A2[D2_indices.index(node.ID * 6 + 2), 0])
            else:
                D[node.ID * 6 + 2, :] = D1[D1_indices.index(node.ID * 6 + 2), :]
                V[node.ID * 6 + 2, :] = V1[D1_indices.index(node.ID * 6 + 2), :]
                A[node.ID * 6 + 2, :] = A1[D1_indices.index(node.ID * 6 + 2), :]

            if D2_indices.count(node.ID * 6 + 3) == 1:
                D.itemset((node.ID * 6 + 3, 0), D2[D2_indices.index(node.ID * 6 + 3), 0])
                V.itemset((node.ID * 6 + 3, 0), V2[D2_indices.index(node.ID * 6 + 3), 0])
                A.itemset((node.ID * 6 + 3, 0), A2[D2_indices.index(node.ID * 6 + 3), 0])
            else:
                D[node.ID * 6 + 3, :] = D1[D1_indices.index(node.ID * 6 + 3), :]
                V[node.ID * 6 + 3, :] = V1[D1_indices.index(node.ID * 6 + 3), :]
                A[node.ID * 6 + 3, :] = A1[D1_indices.index(node.ID * 6 + 3), :]

            if D2_indices.count(node.ID * 6 + 4) == 1:
                D.itemset((node.ID * 6 + 4, 0), D2[D2_indices.index(node.ID * 6 + 4), 0])
                V.itemset((node.ID * 6 + 4, 0), V2[D2_indices.index(node.ID * 6 + 4), 0])
                A.itemset((node.ID * 6 + 4, 0), A2[D2_indices.index(node.ID * 6 + 4), 0])
            else:
                D[node.ID * 6 + 4, :] = D1[D1_indices.index(node.ID * 6 + 4), :]
                V[node.ID * 6 + 4, :] = V1[D1_indices.index(node.ID * 6 + 4), :]
                A[node.ID * 6 + 4, :] = A1[D1_indices.index(node.ID * 6 + 4), :]

            if D2_indices.count(node.ID * 6 + 5) == 1:
                D.itemset((node.ID * 6 + 5, 0), D2[D2_indices.index(node.ID * 6 + 5), 0])
                V.itemset((node.ID * 6 + 5, 0), V2[D2_indices.index(node.ID * 6 + 5), 0])
                A.itemset((node.ID * 6 + 5, 0), A2[D2_indices.index(node.ID * 6 + 5), 0])
            else:
                D[node.ID * 6 + 5, :] = D1[D1_indices.index(node.ID * 6 + 5), :]
                V[node.ID * 6 + 5, :] = V1[D1_indices.index(node.ID * 6 + 5), :]
                A[node.ID * 6 + 5, :] = A1[D1_indices.index(node.ID * 6 + 5), :]

        # Save the responses
        self._TIME_THA = TIME
        self._DISPLACEMENT_THA = D
        self._VELOCITY_THA = V
        self._ACCELERATION_THA = A

        # Put the displacements at the end of the analysis into each node
        for node in self.Nodes.values():
            node.DX[combo_name] = D[node.ID * 6 + 0, -1]
            node.DY[combo_name] = D[node.ID * 6 + 1, -1]
            node.DZ[combo_name] = D[node.ID * 6 + 2, -1]
            node.RX[combo_name] = D[node.ID * 6 + 3, -1]
            node.RY[combo_name] = D[node.ID * 6 + 4, -1]
            node.RZ[combo_name] = D[node.ID * 6 + 5, -1]

        Analysis._calc_reactions(self, log, combo_tags='THA')

        if log:
            print('')
            print('Analysis Complete')
            print('')

        # Flag the model as solved
        self.DynamicSolution['Time History'] = True


    def analyze_linear_time_history_HHT_alpha(self, analysis_method = 'direct', combo_name=None, AgX=None, AgY=None,
                                              AgZ=None,step_size = 0.01, response_duration = 1,
                                              HHT_alpha = 0,  sparse=True, log = False, d0 = None, v0 = None,
                                              damping_options = dict(), type_of_mass_matrix='consistent'):

        """
            Perform linear time history analysis using the Hilber-Hughes-Taylor method by either direct or modal decomposition
            methods.
            If modal superposition, the analysis should be preceded by a modal analysis step.

            :param analysis_method: The analysis method to use ('direct' or 'modal'). Default is 'direct'.
            :type analysis_method: str, optional

            :param combo_name: Name of the load combination. Default is None.
            :type combo_name: str, optional

            :param AgX: Seismic ground acceleration data for X-axis as a 2D numpy array with time and acceleration values. Default is None.
            :type AgX: ndarray, optional

            :param AgY: Seismic ground acceleration data for Y-axis as a 2D numpy array with time and acceleration values. Default is None.
            :type AgY: ndarray, optional

            :param AgZ: Seismic ground acceleration data for Z-axis as a 2D numpy array with time and acceleration values. Default is None.
            :type AgZ: ndarray, optional

            :param step_size: Time step size for the analysis. Default is 0.01 seconds.
            :type step_size: float, optional

            :param response_duration: Duration of the analysis in seconds. Default is 1 second.
            :type response_duration: float, optional

            :param HHT_alpha: Hilber-Hughes-Taylor parameter. Default is 0.
            :type HHT_alpha: float, optional

            :param sparse: Use sparse matrix representations. Default is True.
            :type sparse: bool, optional

            :param log: Enable verbose logging. Default is False.
            :type log: bool, optional

            :param d0: Initial displacements. Default is None.
            :type d0: ndarray, optional

            :param v0: Initial velocities. Default is None.
            :type v0: ndarray, optional

            :param damping_options: Dictionary of damping options for the analysis. Default is zero damping.
            :type damping_options: dict, optional
                \nAllowed keywords in damping_options:
                - 'constant_modal_damping' (float): Constant modal damping ratio (default: 0.00).
                - 'r_alpha' (float): Rayleigh mass proportional damping coefficient.
                - 'r_beta' (float): Rayleigh stiffness proportional damping coefficient.
                - 'first_mode_damping' (float): Damping ratio for the first mode.
                - 'highest_mode_damping' (float): Damping ratio for the highest mode.
                - 'damping_in_every_mode' (list or tuple): Damping ratio(s) for each mode.

            :param type_of_mass_matrix: Type of mass matrix ('consistent' or 'lumped'). Default is 'consistent'.
            :type type_of_mass_matrix: str, optional

            :raises DampingOptionsKeyWordError: If an incorrect damping keyword is used.
            :raises DynamicLoadNotDefinedError: If no dynamic load combination or ground acceleration data is provided.
            :raises ResultsNotFoundError: If modal results are not available.
            :raises ValueError: If input data is not in the expected format.

            :return: None
         """

        if log:
            print('Checking parameters for time history analysis')

        # Check if enter load combination name exists
        if combo_name!=None and combo_name not in self.LoadCombos.keys():
            raise ValueError("'"+combo_name+"' is not among the defined load combinations")

        # Check if the analysis method name is correct
        if analysis_method!='direct' and analysis_method!='modal':
            raise ValueError("Allowed analysis methods are 'modal' and 'direct'")

        # Check if modal results are available if modal superposition approach is selected
        if self.DynamicSolution['Modal'] == False and analysis_method == 'modal':
            raise ResultsNotFoundError('Modal results are not available. Modal superposition method should be preceded by a modal analysis step')

        # Check if correct keyword in damping options has been used
        if analysis_method == 'direct':
            accepted_damping_key_words = ['r_alpha','r_beta']
            for damping_key_word in damping_options.keys():
                if damping_key_word not in accepted_damping_key_words:
                    raise DampingOptionsKeyWordError("For direct analysis, allowed damping keywords in damping options are 'r_alpha' and 'r_beta'")
        else:
            accepted_damping_key_words = ['constant_modal_damping',
                                          'r_alpha',
                                          'r_beta',
                                          'first_mode_damping',
                                          'highest_mode_damping',
                                          'damping_in_every_mode']
            for damping_key_word in damping_options.keys():
                if damping_key_word not in accepted_damping_key_words:
                    raise DampingOptionsKeyWordError

        # Check if the dynamic load combination or at least one seismic ground acceleration has been given
        if combo_name == None and AgX is None and AgY is None and AgZ is None:
            raise DynamicLoadNotDefinedError

        # If only a seismic load has been provided, add default the load combination 'Combo 1'
        if combo_name==None:
            combo_name = 'Combo 1'
            self.LoadCombos['Combo 1'] = LoadCombo(name='Combo 1', factors={'Case 1':1}, combo_tags='THS')

        # Add tags to the load combinations that do not have tags. We will use this to filter results
        for combo in self.LoadCombos.values():
            if combo.combo_tags == None:
                combo.combo_tags = 'unknown_type'


        # Calculate the required number of time history steps
        total_steps = ceil(response_duration/step_size)+1

        # Check if profiles for all load cases in the load combination are defined
        load_profiles = self.LoadProfiles.keys()
        for case in self.LoadCombos[combo_name].factors.keys():
            if case not in load_profiles:
                raise DefinitionNotFoundError('Profile for load case '+str(case)+' has not been defined')

        if log:
            print('Preparing parameters for time history analysis')

        # Edit the load profiles so that they are defined for the entire duration of analysis
        for load_profile in self.LoadProfiles.values():
            case = load_profile.load_case_name
            time : list = load_profile.time
            profile: list = load_profile.profile

            # Check if the profile has not been defined for the entire duration of analysis
            if time[-1] < response_duration:

                # We want to bring the load profile to zero immediately after the last given value
                # We don't want to define two values at the same instance of time
                # It can cause problems during interpolation
                # So we will begin our definition slightly later
                t = 1.0001 * time[-1]
                if t < response_duration:
                    time.extend([t,response_duration])
                    profile.extend([0,0])

                # Redefine the profile
                self.LoadProfiles[case] = LoadProfile(case,time,profile)


        # Extend the AgX seismic input definition too if it is less than the response duration
        if AgX is not None:
            if isinstance(AgX, ndarray):
                if AgX.shape[0] == 2 or AgX.shape[1] == 2:
                    if AgX.shape[1] == 2:
                        AgX = AgX.T

                    time, a_g = AgX[0,:], AgX[1,:]
                    if time[-1] < response_duration:

                        # We want to bring the load profile to zero immediately after the last given value
                        # We don't want to define two values at the same instance of time
                        # It can cause problems during interpolation
                        # So we will begin our definition slightly later
                        t = 1.0001 * time[-1]
                        if t < response_duration:
                            time = append(time, [t, response_duration])
                            a_g = append(a_g, [0, 0])

                        # Redefine seismic input
                        AgX = array([time,a_g])

                else:
                    raise ValueError ("AgX must have two rows, first one for time and second one for ground acceleration")
            else:
                raise ValueError("AgX must be a numpy array")

        # Extend the AgY seismic input definition too if it is less than the response duration
        if AgY is not None:
            if isinstance(AgY, ndarray):
                if AgY.shape[0] == 2 or AgY.shape[1] == 2:
                    if AgY.shape[1] == 2:
                        AgY = AgY.T

                    time, a_g = AgY[0, :], AgY[1, :]
                    if time[-1] < response_duration:

                        # We want to bring the load profile to zero immediately after the last given value
                        # We don't want to define two values at the same instance of time
                        # It can cause problems during interpolation
                        # So we will begin our definition slightly later
                        t = 1.0001 * time[-1]
                        if t < response_duration:
                            time = append(time, [t, response_duration])
                            a_g = append(a_g, [0, 0])

                        # Redefine seismic input
                        AgY = array([time, a_g])

                else:
                    raise ValueError(
                        " AgY must have two rows, first one for time and second one for ground acceleration")
            else:
                raise ValueError("AgY must be a numpy array")

        # Extend the AgZ seismic input definition too if it is less than the response duration
        if AgZ is not None:
            if isinstance(AgZ, ndarray):
                if AgZ.shape[0] == 2 or AgZ.shape[1] == 2:
                    if AgZ.shape[1] == 2:
                        AgZ = AgZ.T

                    time, a_g = AgZ[0, :], AgZ[1, :]
                    if time[-1] < response_duration:

                        # We want to bring the load profile to zero immediately after the last given value
                        # We don't want to define two values at the same instance of time
                        # It can cause problems during interpolation
                        # So we will begin our definition slightly later
                        t = 1.0001 * time[-1]
                        if t < response_duration:
                            time = append(time, [t, response_duration])
                            a_g = append(a_g, [0, 0])

                        # Redefine seismic input
                        AgZ = array([time, a_g])

                else:
                    raise ValueError(
                        " AgZ must have two rows, first one for time and second one for ground acceleration")
            else:
                raise ValueError("AgZ must be a numpy array")

        # Prepare the model
        Analysis._prepare_model(self)

        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = Analysis._partition_D(self)

        # Make sure D2 is a matrix in 2 dimensions
        D2 = D2.reshape(len(D2), 1)

        # Get the partitioned matrices
        if sparse == True:
            K11, K12, K21, K22 = self._partition(self.K(combo_name, log, False, sparse).tolil(), D1_indices,D2_indices)
            M11, M12, M21, M22 = self._partition(self.M(combo_name, log, False, sparse,type_of_mass_matrix).tolil(),
                                                 D1_indices, D2_indices)
        else:
            K11, K12, K21, K22 = self._partition(self.K(combo_name, log, False, sparse), D1_indices, D2_indices)
            M11, M12, M21, M22 = self._partition(self.M(combo_name, log, False, sparse, type_of_mass_matrix),
                                                 D1_indices, D2_indices)

        # Initialise load vector for the entire analysis duration
        F = zeros((len(D1_indices), total_steps))

        # Build the influence vectors used to define the earthquake force
        # Influence vectors are used to select the degrees of freedom in
        # the model stimulated by the earthquake

        total_dof = len(D1_indices)+len(D2_indices)
        i = 0
        influence_X = zeros((total_dof,1))
        influence_Y = zeros((total_dof,1))
        influence_Z = zeros((total_dof,1))
        while i < total_dof:
            influence_X[i+0,0] = 1
            influence_Y[i+1,0] = 1
            influence_Z[i+2,0] = 1
            i += 6

        # Build the load vector step by step
        # We will make modifications to the load combination in order to define the dynamic load
        # Each step uses the original load combination, modifies it according to the given load
        # profiles, and calculates the load vector for that step
        # Hence we must make a copy of the load combination since each step needs the original
        # load combination. Also at the end of the analysis we want to restore the original load
        # combination
        original_load_combo = copy.deepcopy(self.LoadCombos)
        t = 0
        for i in range(total_steps):
            # Check if a load combination has been given
            if combo_name != None:
                # For each load case in the load combination
                for case_name in self.LoadCombos[combo_name].factors.keys():

                    # Extract the time vector from the load profile definition for this load case
                    time_list = self.LoadProfiles[case_name].time

                    # Extract the load profile for this load case
                    profile_list = self.LoadProfiles[case_name].profile

                    # Interpolate the load profile to find the scaling factor for this time t
                    scaler = interp(t, time_list,profile_list)

                    # Get the load factor from the load combination. We will scale this factor
                    # It would be more correct to scale the actual load value and not its factor
                    # But it's hard to do so especially that some load cases are defined by more
                    # than one value, e.g line member loads are defined by two values
                    # Also the same load case can apply to different load types e.g the same
                    # load case can be used for point loads, line loads, pressures, etc.
                    # Hence, the easiest way to accomplish this is by scaling the load factor
                    current_factor = self.LoadCombos[combo_name].factors[case_name]

                    # Edit the load combination using the changed load factor
                    self.LoadCombos[combo_name].factors[case_name] = scaler * current_factor

                # For this time t, the load combination has been edited to reflect the loading situation
                # at this point in time. Hence the fixed end reactions and nodal loads can be calculated
                # for this point in time
                FER1, FER2 = self._partition(self.FER(combo_name), D1_indices, D2_indices)
                P1, P2 = self._partition(self.P(combo_name), D1_indices, D2_indices)

                # Save the calculated nodal loads and fixed end reactions into the load vector
                F[:,i] = P1[:,0] - FER1[:,0]

            # Add the seismic forces as well if the ground acceleration has been given
            # F_s = -[M]{i}a_g where {i} is the influence vector and [M] is the mass matrix
            # and a_g is the ground acceleration
            # Interpolation is used again to find the ground acceleration at time t
            if AgX is not None:
                AgX_F = -M11 @ self._partition(influence_X , D1_indices, D2_indices)[0] \
                        * interp(t, AgX[0, :], AgX[1, :])
                F[:,i] += AgX_F[0,:]

            if AgY is not None:
                AgY_F = -M11 @ self._partition(influence_Y , D1_indices, D2_indices)[0] \
                         * interp(t, AgY[0, :], AgY[1, :])
                F[:,i] += AgY_F[0,:]

            if AgZ is not None:
                AgZ_F = -M11 @ self._partition(influence_Z , D1_indices, D2_indices)[0] \
                         * interp(t, AgZ[0,:], AgZ[1,:])
                F[:,i] += AgZ_F[0,:]

            # Update the time
            t += step_size

            # Restore the time load combination to the original one
            self.LoadCombos = copy.deepcopy(original_load_combo)

        # Update force vector to include the effects of the constrained degrees of freedom
        # In theory we also need to add -M12 @ A2 and -C12 @ V2.
        # However, the constrained displacements don't really change in time for normal
        # structural analysis
        # Hence, the velocity and accelerations will be zero
        F[:,:] -= K12 @ D2

        # Get the mode shapes and natural frequencies if modal analysis method is specified
        if analysis_method == 'modal':
            Z = self._mass_normalised_mode_shapes(M11, self._eigen_vectors)

            # Get the natural frequencies
            w = 2 * pi * self.NATURAL_FREQUENCIES()

        # Get the initial values of displacements and velocities
        # If not given, set them to zero
        if d0 == None:
            d0_phy = zeros(len(D1_indices))
        else:
            d0_phy, d02 = self._partition(d0, D1_indices, D2_indices)

        if v0 == None:
            v0_phy = zeros(len(D1_indices))
        else:
            v0_phy, v02 = self._partition(v0, D1_indices, D2_indices)

        # Find the corresponding quantities the modal coordinate system if modal superposition method is requested
        if analysis_method == 'modal':
            d0_n = solve(Z.T @ Z, Z.T @ d0_phy)
            v0_n = solve(Z.T @ Z, Z.T @ v0_phy)
            F_n = Z.T @ F

        # Run the analysis
        if log:
            print('+-------------------------+')
            print('| Analyzing: Time History |')
            print('+-------------------------+')

        try:
            if analysis_method == 'direct':
                # Extract the Rayleigh damping coefficients
                if 'r_alpha' in damping_options:
                    r_alpha = damping_options['r_alpha']
                else:
                    r_alpha = 0

                if 'r_beta' in damping_options:
                    r_beta = damping_options['r_beta']
                else:
                    r_beta = 0

                TIME, D1, V1, A1 = \
                     Analysis._transient_solver_linear_direct(K=K11, M=M11,d0=d0_phy,v0=v0_phy,
                                                              F0=F[:,0],F = F,step_size=step_size,
                                                              required_duration=response_duration,
                                                              newmark_gamma=0.5 - HHT_alpha,
                                                              newmark_beta=0.25 * (1-HHT_alpha)**2,
                                                              taylor_alpha=HHT_alpha, wilson_theta=1,
                                                              rayleigh_alpha=r_alpha, rayleigh_beta=r_beta,
                                                              sparse=sparse,log=log)

            else:
                TIME, D1, V1, A1 = \
                    Analysis._transient_solver_linear_modal(d0_n=d0_n, v0_n=v0_n, F0_n=F_n[:, 0],
                                                            F_n=F_n, step_size=step_size,
                                                            required_duration=response_duration,
                                                            mass_normalised_eigen_vectors=Z,
                                                            natural_freq=w, newmark_gamma=0.5-HHT_alpha,
                                                            newmark_beta=0.25*(1-HHT_alpha)**2, taylor_alpha=HHT_alpha,
                                                            wilson_theta=1, damping_options=damping_options,
                                                            log=log)
        except:
            raise Exception("Out of memory")

        # Form the global response vectors D, V and A
        D = zeros((len(self.Nodes) * 6, total_steps))
        V = zeros((len(self.Nodes) * 6, total_steps))
        A = zeros((len(self.Nodes) * 6, total_steps))

        # The accelerations and velocities for the constrained degrees of freedom will be zeros
        V2 = zeros(D2.shape)
        A2 = zeros(D2.shape)

        for node in self.Nodes.values():

            if D2_indices.count(node.ID * 6 + 0) == 1:
                D.itemset((node.ID * 6 + 0, 0), D2[D2_indices.index(node.ID * 6 + 0), 0])
                V.itemset((node.ID * 6 + 0, 0), V2[D2_indices.index(node.ID * 6 + 0), 0])
                A.itemset((node.ID * 6 + 0, 0), A2[D2_indices.index(node.ID * 6 + 0), 0])
            else:
                D[node.ID * 6 + 0, :] = D1[D1_indices.index(node.ID * 6 + 0), :]
                V[node.ID * 6 + 0, :] = V1[D1_indices.index(node.ID * 6 + 0), :]
                A[node.ID * 6 + 0, :] = A1[D1_indices.index(node.ID * 6 + 0), :]
            if D2_indices.count(node.ID * 6 + 1) == 1:
                D.itemset((node.ID * 6 + 1, 0), D2[D2_indices.index(node.ID * 6 + 1), 0])
                V.itemset((node.ID * 6 + 1, 0), V2[D2_indices.index(node.ID * 6 + 1), 0])
                A.itemset((node.ID * 6 + 1, 0), A2[D2_indices.index(node.ID * 6 + 1), 0])
            else:
                D[node.ID * 6 + 1, :] = D1[D1_indices.index(node.ID * 6 + 1), :]
                V[node.ID * 6 + 1, :] = V1[D1_indices.index(node.ID * 6 + 1), :]
                A[node.ID * 6 + 1, :] = A1[D1_indices.index(node.ID * 6 + 1), :]

            if D2_indices.count(node.ID * 6 + 2) == 1:
                D.itemset((node.ID * 6 + 2, 0), D2[D2_indices.index(node.ID * 6 + 2), 0])
                V.itemset((node.ID * 6 + 2, 0), V2[D2_indices.index(node.ID * 6 + 2), 0])
                A.itemset((node.ID * 6 + 2, 0), A2[D2_indices.index(node.ID * 6 + 2), 0])
            else:
                D[node.ID * 6 + 2, :] = D1[D1_indices.index(node.ID * 6 + 2), :]
                V[node.ID * 6 + 2, :] = V1[D1_indices.index(node.ID * 6 + 2), :]
                A[node.ID * 6 + 2, :] = A1[D1_indices.index(node.ID * 6 + 2), :]

            if D2_indices.count(node.ID * 6 + 3) == 1:
                D.itemset((node.ID * 6 + 3, 0), D2[D2_indices.index(node.ID * 6 + 3), 0])
                V.itemset((node.ID * 6 + 3, 0), V2[D2_indices.index(node.ID * 6 + 3), 0])
                A.itemset((node.ID * 6 + 3, 0), A2[D2_indices.index(node.ID * 6 + 3), 0])
            else:
                D[node.ID * 6 + 3, :] = D1[D1_indices.index(node.ID * 6 + 3), :]
                V[node.ID * 6 + 3, :] = V1[D1_indices.index(node.ID * 6 + 3), :]
                A[node.ID * 6 + 3, :] = A1[D1_indices.index(node.ID * 6 + 3), :]

            if D2_indices.count(node.ID * 6 + 4) == 1:
                D.itemset((node.ID * 6 + 4, 0), D2[D2_indices.index(node.ID * 6 + 4), 0])
                V.itemset((node.ID * 6 + 4, 0), V2[D2_indices.index(node.ID * 6 + 4), 0])
                A.itemset((node.ID * 6 + 4, 0), A2[D2_indices.index(node.ID * 6 + 4), 0])
            else:
                D[node.ID * 6 + 4, :] = D1[D1_indices.index(node.ID * 6 + 4), :]
                V[node.ID * 6 + 4, :] = V1[D1_indices.index(node.ID * 6 + 4), :]
                A[node.ID * 6 + 4, :] = A1[D1_indices.index(node.ID * 6 + 4), :]

            if D2_indices.count(node.ID * 6 + 5) == 1:
                D.itemset((node.ID * 6 + 5, 0), D2[D2_indices.index(node.ID * 6 + 5), 0])
                V.itemset((node.ID * 6 + 5, 0), V2[D2_indices.index(node.ID * 6 + 5), 0])
                A.itemset((node.ID * 6 + 5, 0), A2[D2_indices.index(node.ID * 6 + 5), 0])
            else:
                D[node.ID * 6 + 5, :] = D1[D1_indices.index(node.ID * 6 + 5), :]
                V[node.ID * 6 + 5, :] = V1[D1_indices.index(node.ID * 6 + 5), :]
                A[node.ID * 6 + 5, :] = A1[D1_indices.index(node.ID * 6 + 5), :]

        # Save the responses
        self._TIME_THA = TIME
        self._DISPLACEMENT_THA = D
        self._VELOCITY_THA = V
        self._ACCELERATION_THA = A

        # Put the displacements at the end of the analysis into each node
        for node in self.Nodes.values():
            node.DX[combo_name] = D[node.ID * 6 + 0, -1]
            node.DY[combo_name] = D[node.ID * 6 + 1, -1]
            node.DZ[combo_name] = D[node.ID * 6 + 2, -1]
            node.RX[combo_name] = D[node.ID * 6 + 3, -1]
            node.RY[combo_name] = D[node.ID * 6 + 4, -1]
            node.RZ[combo_name] = D[node.ID * 6 + 5, -1]

        Analysis._calc_reactions(self, log, combo_tags='THA')

        if log:
            print('')
            print('Analysis Complete')
            print('')

        # Flag the model as solved
        self.DynamicSolution['Time History'] = True


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

    def DISPLACEMENT_REAL(self):
        """
        Returns the real part of the global nodal displacement of the model solved using harmonic analysis
        """
        return self._DISPLACEMENT_REAL

    def DISPLACEMENT_IMAGINARY(self):
        """
        Returns the imaginary part of the global nodal displacement of the model solved using harmonic analysis
        """
        return self._DISPLACEMENT_IMAGINARY

    def VELOCITY_REAL(self):
        """
        Returns the real part of the global nodal velocity of the model solved using harmonic analysis
        """
        return self._VELOCITY_REAL

    def VELOCITY_IMAGINARY(self):
        """
        Returns the imaginary part of the global nodal velocity of the model solved using harmonic analysis
        """
        return self._VELOCITY_IMAGINARY

    def ACCELERATION_REAL(self):
        """
        Returns the real part of the global nodal acceleration of the model solved using harmonic analysis
        """
        return self._ACCELERATION_REAL

    def ACCELERATION_IMAGINARY(self):
        """
        Returns the real part of the global nodal acceleration of the model solved using harmonic analysis
        """
        return self._ACCELERATION_IMAGINARY

    def REACTIONS_REAL(self):
        """
        Returns the real part of the models reaction forces solved from harmonic analysis
        """
        return self._REACTIONS_REAL

    def REACTIONS_IMAGINARY(self):
        """
        Returns the imaginary part of the models reaction forces solved from harmonic analysis
        """
        return self._REACTIONS_IMAGINARY

    def NATURAL_FREQUENCIES(self):
        """
        Returns the natural frequencies of the model
        """
        return self._NATURAL_FREQUENCIES

    def MODE_SHAPES(self):
        """
        Returns the Mode shapes of the structure
        """
        return self._MODE_SHAPES

    def DISPLACEMENT_THA(self):
        """
        Returns the global time history displacements
        """
        return self._DISPLACEMENT_THA

    def VELOCITY_THA(self):
        """
        Returns the global time history velocities
        """
        return self._VELOCITY_THA

    def ACCELERATION_THA(self):
        """
        Returns the global time history accelerations
        """
        return self._ACCELERATION_THA

    def REACTIONS_THA(self):
        """
        Returns the model's reactions time history
        """
        return self._REACTIONS_THA

    def TIME_THA(self):
        """
        Returns the time vector over which the time history analysis has been conducted
        """
        return self._TIME_THA

    def MASS_PARTICIPATION_PERCENTAGES(self):
        """
        Returns the mass participation of the calculated modes in the x, y and z directions
        """
        return self._mass_participation

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
