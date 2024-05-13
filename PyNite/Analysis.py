import sys

from math import isclose, ceil

from numpy.lib.function_base import interp, gradient

from PyNite import FEModel3D
from PyNite.LoadCombo import LoadCombo

from numpy import subtract, matmul, divide, seterr, nanmax

from numpy import array, atleast_2d, zeros, diag, ones, identity, linspace
from numpy.linalg import solve
import logging


def _prepare_model(model):
    """Prepares a model for analysis by ensuring at least one load combination is defined, generating all meshes that have not already been generated, activating all non-linear members, and internally numbering all nodes and elements.

    :param model: The model being prepared for analysis.
    :type model: FEModel3D
    """
    
    # Reset any nodal displacements
    model._D = {}
    for node in model.Nodes.values():
        node.DX = {}
        node.DY = {}
        node.DZ = {}
        node.RX = {}
        node.RY = {}
        node.RZ = {}

    # Ensure there is at least 1 load combination to solve if the user didn't define any
    if model.LoadCombos == {}:
        # Create and add a default load combination to the dictionary of load combinations
        model.LoadCombos['Combo 1'] = LoadCombo('Combo 1', factors={'Case 1':1.0})
    
    # Generate all meshes
    for mesh in model.Meshes.values():
        if mesh.is_generated == False:
            mesh.generate()

    # Activate all springs and members for all load combinations
    for spring in model.Springs.values():
        for combo_name in model.LoadCombos.keys():
            spring.active[combo_name] = True
    
    # Activate all physical members for all load combinations
    for phys_member in model.Members.values():
        for combo_name in model.LoadCombos.keys():
            phys_member.active[combo_name] = True
    
    # Assign an internal ID to all nodes and elements in the model. This number is different from the name used by the user to identify nodes and elements.
    _renumber(model)

def _identify_combos(model, combo_tags=None):
    """Returns a list of load combinations that are to be run based on tags given by the user.

    :param model: The model being analyzed.
    :type model: FEModel3D
    :param combo_tags: A list of tags used for the load combinations to be evaluated. Defaults to `None` in which case all load combinations will be added to the list of load combinations to be run.
    :type combo_tags: list, optional
    :return: A list containing the load combinations to be analyzed.
    :rtype: list
    """
    
    # Identify which load combinations to evaluate
    if combo_tags is None:
        # Evaluate all load combinations if not tags have been provided
        combo_list = model.LoadCombos.values()
    else:
        # Initialize the list of load combinations to be evaluated
        combo_list = []
        # Step through each load combination in the model
        for combo in model.LoadCombos.values():
            # Check if this load combination is tagged with any of the tags we're looking for
            if combo.combo_tags is not None and any(tag in combo.combo_tags for tag in combo_tags):
                # Add the load combination to the list of load combinations to be evaluated
                combo_list.append(combo)
    
    # Return the list of load combinations to be evaluated
    return combo_list

def _check_stability(model, K):
    """
    Identifies nodal instabilities in a model's stiffness matrix.
    """

    # Initialize the `unstable` flag to `False`
    unstable = False

    # Step through each diagonal term in the stiffness matrix
    for i in range(K.shape[0]):
        
        # Determine which node this term belongs to
        node = [node for node in model.Nodes.values() if node.ID == int(i/6)][0]

        # Determine which degree of freedom this term belongs to
        dof = i%6

        # Check to see if this degree of freedom is supported
        if dof == 0:
            supported = node.support_DX
        elif dof == 1:
            supported = node.support_DY
        elif dof == 2:
            supported = node.support_DZ
        elif dof == 3:
            supported = node.support_RX
        elif dof == 4:
            supported = node.support_RY
        elif dof == 5:
            supported = node.support_RZ

        # Check if the degree of freedom on this diagonal is unstable
        if isclose(K[i, i], 0) and not supported:

            # Flag the model as unstable
            unstable = True

            # Identify which direction this instability effects
            if i%6 == 0: direction = 'for translation in the global X direction.'
            if i%6 == 1: direction = 'for translation in the global Y direction.'
            if i%6 == 2: direction = 'for translation in the global Z direction.'
            if i%6 == 3: direction = 'for rotation about the global X axis.'
            if i%6 == 4: direction = 'for rotation about the global Y axis.'
            if i%6 == 5: direction = 'for rotation about the global Z axis.'

            # Print a message to the console
            print('* Nodal instability detected: node ' + node.name + ' is unstable ' + direction)

    if unstable:
        raise Exception('Unstable node(s). See console output for details.')

    return

def _PDelta_step(model, combo_name, P1, FER1, D1_indices, D2_indices, D2, log=True, sparse=True, check_stability=False, max_iter=30, first_step=True):
    """Performs second order (P-Delta) analysis. This type of analysis is appropriate for most models using beams, columns and braces. Second order analysis is usually required by material-specific codes. The analysis is iterative and takes longer to solve. Models with slender members and/or members with combined bending and axial loads will generally have more significant P-Delta effects. P-Delta effects in plates/quads are not considered.

    :param combo_name: The name of the load combination to evaluate P-Delta effects for.
    :type combo_name: string
    :param log: Prints updates to the console if set to True. Default is False.
    :type log: bool, optional
    :param check_stability: When set to True, checks the stiffness matrix for any unstable degrees of freedom and reports them back to the console. This does add to the solution time. Defaults to True.
    :type check_stability: bool, optional
    :param max_iter: The maximum number of iterations permitted. If this value is exceeded the program will report divergence. Defaults to 30.
    :type max_iter: int, optional
    :param sparse: Indicates whether the sparse matrix solver should be used. A matrix can be considered sparse or dense depening on how many zero terms there are. Structural stiffness matrices often contain many zero terms. The sparse solver can offer faster solutions for such matrices. Using the sparse solver on dense matrices may lead to slower solution times. Be sure ``scipy`` is installed to use the sparse solver. Default is True.
    :type sparse: bool, optional
    :param check_stability: Indicates whether nodal stability should be checked. This slows down the analysis considerably, but can be useful for small models or for debugging. Default is `False`.
    :type check_stability: bool, optional
    :param first_step: Indicates whether this P-Delta analysis is the first load step. Usually this should be set to `True`, unless this is a subsequent step of an analysis using multiple load steps. Default is True.
    :type first_step: bool, optional
    :raises ValueError: Occurs when there is a singularity in the stiffness matrix, which indicates an unstable structure.
    :raises Exception: Occurs when a model fails to converge.
    """

    # Import `scipy` features if the sparse solver is being used
    if sparse == True:
        from scipy.sparse.linalg import spsolve

    iter_count_TC = 1    # Tracks tension/compression-only iterations
    iter_count_PD = 1    # Tracks P-Delta iterations

    convergence_TC = False  # Tracks tension/compression-only convergence
    divergence_TC = False  # Tracks tension/compression-only divergence

    # Iterate until either T/C convergence or divergence occurs. Perform at least 2 iterations for the P-Delta analysis.
    while (convergence_TC == False and divergence_TC == False) or iter_count_PD <= 2:

        # Inform the user which iteration we're on
        if log:
            print('- Beginning tension/compression-only iteration #' + str(iter_count_TC))

        # Calculate the partitioned global stiffness matrices
        if sparse == True:

            # Calculate the initial stiffness matrix
            K11, K12, K21, K22 = _partition(model, model.K(combo_name, log, check_stability, sparse).tolil(), D1_indices, D2_indices)

            # Calculate the geometric stiffness matrix
            if iter_count_PD == 1 and first_step:
                # For the first iteration of the first load step P=0
                Kg11, Kg12, Kg21, Kg22 = _partition(model, model.Kg(combo_name, log, sparse, True), D1_indices, D2_indices)
            else:
                # For subsequent iterations P will be calculated based on member end displacements
                Kg11, Kg12, Kg21, Kg22 = _partition(model, model.Kg(combo_name, log, sparse, False), D1_indices, D2_indices)
            
            # The stiffness matrices are currently `lil` format which is great for
            # memory, but slow for mathematical operations. They will be converted to
            # `csr` format. The `+` operator performs matrix addition on `csr`
            # matrices.
            K11 = K11.tocsr() + Kg11.tocsr()
            K12 = K12.tocsr() + Kg12.tocsr()
            K21 = K21.tocsr() + Kg21.tocsr()
            K22 = K22.tocsr() + Kg22.tocsr()

        else:

            # Initial stiffness matrix
            K11, K12, K21, K22 = _partition(model, model.K(combo_name, log, check_stability, sparse), D1_indices, D2_indices)
            
            # Geometric stiffness matrix
            if iter_count_PD == 1 and first_step:
                # For the first iteration of the first load step P=0
                Kg11, Kg12, Kg21, Kg22 = _partition(model, model.Kg(combo_name, log, sparse, True), D1_indices, D2_indices)
            else:
                # For subsequent iterations P will be calculated based on member end displacements
                Kg11, Kg12, Kg21, Kg22 = _partition(model.Kg(combo_name, log, sparse, False), D1_indices, D2_indices)
            
            K11 = K11 + Kg11
            K12 = K12 + Kg12
            K21 = K21 + Kg21
            K22 = K22 + Kg22

        # Calculate the changes to the global displacement vector
        if log: print('- Calculating changes to the global displacement vector')
        if K11.shape == (0, 0):
            # All displacements are known, so D1 is an empty vector
            Delta_D1 = []
        else:
            try:
                # Calculate the change in the displacements Delta_D1
                if sparse == True:
                    # The partitioned stiffness matrix is already in `csr` format. The `@`
                    # operator performs matrix multiplication on sparse matrices.
                    Delta_D1 = spsolve(K11.tocsr(), subtract(subtract(P1, FER1), K12.tocsr() @ D2))
                    Delta_D1 = Delta_D1.reshape(len(Delta_D1), 1)
                else:
                    # The partitioned stiffness matrix is in `csr` format. It will be
                    # converted to a 2D dense array for mathematical operations.
                    Delta_D1 = solve(K11, subtract(subtract(P1, FER1), matmul(K12, D2)))

            except:
                # Return out of the method if 'K' is singular and provide an error message
                raise ValueError('The stiffness matrix is singular, which indicates that the structure is unstable.')

        # Sum the calculated displacements
        if first_step:
            _store_displacements(model, Delta_D1, D2, D1_indices, D2_indices, model.LoadCombos[combo_name])
        else:
            _sum_displacements(model, Delta_D1, D2, D1_indices, D2_indices, model.LoadCombos[combo_name])
        
        # Check whether the tension/compression-only analysis has converged and deactivate any members that are showing forces they can't hold
        convergence_TC = _check_TC_convergence(model, combo_name, log)
        
        # Report on convergence of tension/compression only analysis
        if convergence_TC == False:
            
            if log:
                print('- Tension/compression-only analysis did not converge on this iteration')
                print('- Stiffness matrix will be adjusted')
                print('- P-Delta analysis will be restarted')
            
            # Increment the tension/compression-only iteration count
            iter_count_TC += 1

            # Undo the last iteration of the analysis since the T/C analysis didn't converge
            _sum_displacements(model, -Delta_D1, D2, D1_indices, D2_indices, model.LoadCombos[combo_name])
            iter_count_PD = 0

        else:
            if log: print('- Tension/compression-only analysis converged after ' + str(iter_count_TC) + ' iteration(s)')
        
        # Check for divergence in the tension/compression-only analysis
        if iter_count_TC > max_iter:
            divergence_TC = True
            raise Exception('- Model diverged during tension/compression-only analysis')

        # Increment the P-Delta iteration count
        iter_count_PD += 1
    
    # Flag the model as solved
    model.solution = 'P-Delta'

def _pushover_step(model, combo_name, push_combo, step_num, P1, FER1, D1_indices, D2_indices, D2, log=True, sparse=True, check_stability=False):

    # Run at least one iteration
    run_step = True

    # Run/rerun the load step until convergence occurs
    while run_step == True:

        # Calculate the partitioned global stiffness matrices
        # Sparse solver
        if sparse == True:

            from scipy.sparse.linalg import spsolve
            
            # Calculate the initial stiffness matrix
            K11, K12, K21, K22 = _partition(model, model.K(combo_name, log, check_stability, sparse).tolil(), D1_indices, D2_indices)

            # Calculate the geometric stiffness matrix
            # The `combo_name` variable in the code below is not the name of the pushover load combination. Rather it is the name of the primary combination that the pushover load will be added to. Axial loads used to develop Kg are calculated from the displacements stored in `combo_name`.
            Kg11, Kg12, Kg21, Kg22 = _partition(model, model.Kg(combo_name, log, sparse, False).tolil(), D1_indices, D2_indices)

            # Calculate the stiffness reduction matrix
            Km11, Km12, Km21, Km22 = _partition(model, model.Km(combo_name, push_combo, step_num, log, sparse).tolil(), D1_indices, D2_indices)
            
            # The stiffness matrices are currently `lil` format which is great for
            # memory, but slow for mathematical operations. They will be converted to
            # `csr` format. The `+` operator performs matrix addition on `csr`
            # matrices.
            K11 = K11.tocsr() + Kg11.tocsr() + Km11.tocsr()
            K12 = K12.tocsr() + Kg12.tocsr() + Km12.tocsr()
            K21 = K21.tocsr() + Kg21.tocsr() + Km21.tocsr()
            K22 = K22.tocsr() + Kg22.tocsr() + Km22.tocsr()

        # Dense solver
        else:

            # Initial stiffness matrix
            K11, K12, K21, K22 = _partition(model, model.K(combo_name, log, check_stability, sparse), D1_indices, D2_indices)
            
            # Geometric stiffness matrix
            # The `combo_name` variable in the code below is not the name of the pushover load combination. Rather it is the name of the primary combination that the pushover load will be added to. Axial loads used to develop Kg are calculated from the displacements stored in `combo_name`.
            Kg11, Kg12, Kg21, Kg22 = _partition(model.Kg(combo_name, log, sparse, False), D1_indices, D2_indices)
            
            # Calculate the stiffness reduction matrix
            Km11, Km12, Km21, Km22 = _partition(model, model.Km(combo_name, push_combo, step_num, log, sparse), D1_indices, D2_indices)
            
            K11 = K11 + Kg11 + Km11
            K12 = K12 + Kg12 + Km12
            K21 = K21 + Kg21 + Km21
            K22 = K22 + Kg22 + Km22
        
        # Calculate the changes to the global displacement vector
        if log: print('- Calculating changes to the global displacement vector')
        if K11.shape == (0, 0):
            # All displacements are known, so D1 is an empty vector
            Delta_D1 = []
        else:
            try:
                # Calculate the change in the displacements Delta_D1
                if sparse == True:
                    # The partitioned stiffness matrix is already in `csr` format. The `@`
                    # operator performs matrix multiplication on sparse matrices.
                    Delta_D1 = spsolve(K11.tocsr(), subtract(subtract(P1, FER1), K12.tocsr() @ D2))
                    Delta_D1 = Delta_D1.reshape(len(Delta_D1), 1)
                else:
                    # The partitioned stiffness matrix is in `csr` format. It will be
                    # converted to a 2D dense array for mathematical operations.
                    Delta_D1 = solve(K11, subtract(subtract(P1, FER1), matmul(K12, D2)))

            except:
                # Return out of the method if 'K' is singular and provide an error message
                raise ValueError('The structure is unstable. Unable to proceed any further with analysis.')

        # Unpartition the displacement results from the analysis step
        Delta_D = _unpartition_disp(model, Delta_D1, D2, D1_indices, D2_indices)

        # Step through each member in the model
        for member in model.Members.values():
                        
            # Check for plastic load reversal at the i-node in this load step
            if member.i_reversal == False and member.lamb(Delta_D, combo_name, push_combo, step_num)[0, 1] < 0:

                # Flag the member as having plastic load reversal at the i-node
                i_reversal = True

                # Flag the load step for reanalysis
                run_step = True

            # Check for plastic load reversal at the j-node in this load step
            if member.j_reversal == False and member.lamb(Delta_D, combo_name, push_combo, step_num)[1, 1] < 0:

                # Flag the member as having plastic load reversal at the j-node
                j_reversal = True

                # Flag the load step for reanalysis
                run_step = True

        # Undo the last loadstep if plastic load reversal was discovered. We'll rerun it with the corresponding gradients set to zero vectors.
        if run_step == True:
            _sum_displacements(model, -Delta_D1, D2, D1_indices, D2_indices, model.LoadCombos[combo_name])
    
    # Sum the calculated displacements
    _sum_displacements(model, Delta_D1, D2, D1_indices, D2_indices, model.LoadCombos[combo_name])

    # Flag the model as solved
    model.solution = 'Pushover'

def _unpartition_disp(model, D1, D2, D1_indices, D2_indices):
    """Unpartitions displacements from the solver and returns them as a global displacement vector

    :param model: The finite element model being evaluated
    :type model: FEModel3D
    :param D1: An array of calculated displacements
    :type D1: array
    :param D2: An array of enforced displacements
    :type D2: array
    :param D1_indices: A list of the degree of freedom indices for each displacement in D1
    :type D1_indices: list
    :param D2_indices: A list of the degree of freedom indices for each displacement in D2
    :type D2_indices: list
    :return: Global displacement matrix
    :rtype: array
    """
    
    D = zeros((len(model.Nodes)*6, 1))

    # Step through each node in the model
    for node in model.Nodes.values():
        
        if node.ID*6 + 0 in D2_indices:
            # Get the enforced displacement
            D[(node.ID*6 + 0, 0)] = D2[D2_indices.index(node.ID*6 + 0), 0]
        else:
            # Get the calculated displacement
            D[(node.ID*6 + 0, 0)] = D1[D1_indices.index(node.ID*6 + 0), 0]

        if node.ID*6 + 1 in D2_indices:
            # Get the enforced displacement
            D[(node.ID*6 + 1, 0)] = D2[D2_indices.index(node.ID*6 + 1), 0]
        else:
            # Get the calculated displacement
            D[(node.ID*6 + 1, 0)] = D1[D1_indices.index(node.ID*6 + 1), 0]

        if node.ID*6 + 2 in D2_indices:
            # Get the enforced displacement
            D[(node.ID*6 + 2, 0)] = D2[D2_indices.index(node.ID*6 + 2), 0]
        else:
            # Get the calculated displacement
            D[(node.ID*6 + 2, 0)] = D1[D1_indices.index(node.ID*6 + 2), 0]

        if node.ID*6 + 3 in D2_indices:
            # Get the enforced rotation
            D[(node.ID*6 + 3, 0)] = D2[D2_indices.index(node.ID*6 + 3), 0]
        else:
            # Get the calculated rotation
            D[(node.ID*6 + 3, 0)] = D1[D1_indices.index(node.ID*6 + 3), 0]

        if node.ID*6 + 4 in D2_indices:
            # Get the enforced rotation
            D[(node.ID*6 + 4, 0)] = D2[D2_indices.index(node.ID*6 + 4), 0]
        else:
            # Get the calculated rotation
            D[(node.ID*6 + 4, 0)] = D1[D1_indices.index(node.ID*6 + 4), 0]

        if node.ID*6 + 5 in D2_indices:
            # Get the enforced rotation
            D[(node.ID*6 + 5, 0)] = D2[D2_indices.index(node.ID*6 + 5), 0]
        else:
            # Get the calculated rotation
            D[(node.ID*6 + 5, 0)] = D1[D1_indices.index(node.ID*6 + 5), 0]
    
    # Return the displacement vector
    return D

def _store_displacements(model, D1, D2, D1_indices, D2_indices, combo):
    """Stores calculated displacements from the solver into the model's displacement vector `_D` and into each node object in the model

    :param model: The finite element model being evaluated.
    :type model: FEModel3D
    :param D1: An array of calculated displacements
    :type D1: array
    :param D2: An array of enforced displacements
    :type D2: array
    :param D1_indices: A list of the degree of freedom indices for each displacement in D1
    :type D1_indices: list
    :param D2_indices: A list of the degree of freedom indices for each displacement in D2
    :type D2_indices: list
    :param combo: The load combination to store the displacements for
    :type combo: LoadCombo
    """
    
    # The raw results from the solver are partitioned. Unpartition them.
    D = _unpartition_disp(model, D1, D2, D1_indices, D2_indices)
    
    # Store the displacements in the model's global displacement vector
    model._D[combo.name] = D

    # Store the calculated global nodal displacements into each node object
    for node in model.Nodes.values():

        node.DX[combo.name] = D[node.ID*6 + 0, 0]
        node.DY[combo.name] = D[node.ID*6 + 1, 0]
        node.DZ[combo.name] = D[node.ID*6 + 2, 0]
        node.RX[combo.name] = D[node.ID*6 + 3, 0]
        node.RY[combo.name] = D[node.ID*6 + 4, 0]
        node.RZ[combo.name] = D[node.ID*6 + 5, 0]

def _sum_displacements(model, Delta_D1, Delta_D2, D1_indices, D2_indices, combo):
    """Sums calculated displacements for a load step from the solver into the model's displacement vector `_D` and into each node object in the model.

    :param model: The finite element model being evaluated.
    :type model: FEModel3D
    :param Delta_D1: An array of calculated displacements for a load step
    :type Delta_D1: array
    :param Delta_D2: An array of enforced displacements for a load step
    :type Delta_D2: array
    :param D1_indices: A list of the degree of freedom indices for each displacement in D1
    :type D1_indices: list
    :param D2_indices: A list of the degree of freedom indices for each displacement in D2
    :type D2_indices: list
    :param combo: The load combination to store the displacements for
    :type combo: LoadCombo
    """
    
    # The raw results from the solver are partitioned. Unpartition them.
    Delta_D = _unpartition_disp(model, Delta_D1, Delta_D2, D1_indices, D2_indices)

    # Sum the load step's global displacement vector with the model's global displacement vector
    model._D[combo.name] += Delta_D

    # Sum the load step's calculated global nodal displacements to each node object's global displacement
    for node in model.Nodes.values():

        node.DX[combo.name] += Delta_D[node.ID*6 + 0, 0]
        node.DY[combo.name] += Delta_D[node.ID*6 + 1, 0]
        node.DZ[combo.name] += Delta_D[node.ID*6 + 2, 0]
        node.RX[combo.name] += Delta_D[node.ID*6 + 3, 0]
        node.RY[combo.name] += Delta_D[node.ID*6 + 4, 0]
        node.RZ[combo.name] += Delta_D[node.ID*6 + 5, 0]

def _check_TC_convergence(model, combo_name='Combo 1', log=True):
    
    # Assume the model has converged until we find out otherwise
    convergence = True
    
    # Provide an update to the console if requested by the user
    if log: print('- Checking for tension/compression-only support spring convergence')
    for node in model.Nodes.values():

        # Check convergence of tension/compression-only spring supports and activate/deactivate them as necessary
        if node.spring_DX[1] is not None:
            if ((node.spring_DX[1] == '-' and node.DX[combo_name] > 0)
            or (node.spring_DX[1] == '+' and node.DX[combo_name] < 0)):
                # Check if the spring is switching from active to inactive
                if node.spring_DX[2] == True: convergence = False
                # Make sure the spring is innactive
                node.spring_DX[2] = False
            elif ((node.spring_DX[1] == '-' and node.DX[combo_name] < 0)
            or (node.spring_DX[1] == '+' and node.DX[combo_name] > 0)):
                # Check if the spring is switching from inactive to active
                if node.spring_DX[2] == False: convergence = False
                # Make sure the spring is active
                node.spring_DX[2] = True
        if node.spring_DY[1] is not None:
            if ((node.spring_DY[1] == '-' and node.DY[combo_name] > 0)
            or (node.spring_DY[1] == '+' and node.DY[combo_name] < 0)):
                # Check if the spring is switching from active to inactive
                if node.spring_DY[2] == True: convergence = False
                # Make sure the spring is innactive
                node.spring_DY[2] = False
            elif ((node.spring_DY[1] == '-' and node.DY[combo_name] < 0)
            or (node.spring_DY[1] == '+' and node.DY[combo_name] > 0)):
                # Check if the spring is switching from inactive to active
                if node.spring_DY[2] == False: convergence = False
                # Make sure the spring is active
                node.spring_DY[2] = True
        if node.spring_DZ[1] is not None:
            if ((node.spring_DZ[1] == '-' and node.DZ[combo_name] > 0)
            or (node.spring_DZ[1] == '+' and node.DZ[combo_name] < 0)):
                # Check if the spring is switching from active to inactive
                if node.spring_DZ[2] == True: convergence = False
                # Make sure the spring is innactive
                node.spring_DZ[2] = False
            elif ((node.spring_DZ[1] == '-' and node.DZ[combo_name] < 0)
            or (node.spring_DZ[1] == '+' and node.DZ[combo_name] > 0)):
                # Check if the spring is switching from inactive to active
                if node.spring_DZ[2] == False: convergence = False
                # Make sure the spring is active
                node.spring_DZ[2] = True
        if node.spring_RX[1] is not None:
            if ((node.spring_RX[1] == '-' and node.RX[combo_name] > 0)
            or (node.spring_RX[1] == '+' and node.RX[combo_name] < 0)):
                # Check if the spring is switching from active to inactive
                if node.spring_RX[2] == True: convergence = False
                # Make sure the spring is innactive
                node.spring_RX[2] = False
            elif ((node.spring_RX[1] == '-' and node.RX[combo_name] < 0)
            or (node.spring_RX[1] == '+' and node.RX[combo_name] > 0)):
                # Check if the spring is switching from inactive to active
                if node.spring_RX[2] == False: convergence = False
                # Make sure the spring is active
                node.spring_RX[2] = True
        if node.spring_RY[1] is not None:
            if ((node.spring_RY[1] == '-' and node.RY[combo_name] > 0)
            or (node.spring_RY[1] == '+' and node.RY[combo_name] < 0)):
                # Check if the spring is switching from active to inactive
                if node.spring_RY[2] == True: convergence = False
                # Make sure the spring is innactive
                node.spring_RY[2] = False
            elif ((node.spring_RY[1] == '-' and node.RY[combo_name] < 0)
            or (node.spring_RY[1] == '+' and node.RY[combo_name] > 0)):
                # Check if the spring is switching from inactive to active
                if node.spring_RY[2] == False: convergence = False
                # Make sure the spring is active
                node.spring_RY[2] = True
        if node.spring_RZ[1] is not None:
            if ((node.spring_RZ[1] == '-' and node.RZ[combo_name] > 0)
            or (node.spring_RZ[1] == '+' and node.RZ[combo_name] < 0)):
                # Check if the spring is switching from active to inactive
                if node.spring_RZ[2] == True: convergence = False
                # Make sure the spring is innactive
                node.spring_RZ[2] = False
            elif ((node.spring_RZ[1] == '-' and node.RZ[combo_name] < 0)
            or (node.spring_RZ[1] == '+' and node.RZ[combo_name] > 0)):
                # Check if the spring is switching from inactive to active
                if node.spring_RZ[2] == False: convergence = False
                # Make sure the spring is active
                node.spring_RZ[2] = True
    
    # TODO: Adjust the code below to allow elements to reactivate on subsequent iterations if deformations at element nodes indicate the member goes back into an active state. This will lead to a less conservative and more realistic analysis. Nodal springs (above) already do this.

    # Check tension/compression-only springs
    if log: print('- Checking for tension/compression-only spring convergence')
    for spring in model.Springs.values():

        if spring.active[combo_name] == True:

            # Check if tension-only conditions exist
            if spring.tension_only == True and spring.axial(combo_name) > 0:
                spring.active[combo_name] = False
                convergence = False
            
            # Check if compression-only conditions exist
            elif spring.comp_only == True and spring.axial(combo_name) < 0:
                spring.active[combo_name] = False
                convergence = False

    # Check tension/compression only members
    if log: print('- Checking for tension/compression-only member convergence')
    for phys_member in model.Members.values():

        # Only run the tension/compression only check if the member is still active
        if phys_member.active[combo_name] == True:

            # Check if tension-only conditions exist
            if phys_member.tension_only == True and phys_member.max_axial(combo_name) > 0:
                phys_member.active[combo_name] = False
                convergence = False

            # Check if compression-only conditions exist
            elif phys_member.comp_only == True and phys_member.min_axial(combo_name) < 0:
                phys_member.active[combo_name] = False
                convergence = False

    # Return whether the TC analysis has converged
    return convergence

def _calc_reactions(model, log=False, combo_tags=None):
    """
    Calculates reactions internally once the model is solved.

    Parameters
    ----------
    log : bool, optional
        Prints updates to the console if set to True. Default is False.
    """

    # Print a status update to the console
    if log: print('- Calculating reactions')

    # Identify which load combinations to evaluate
    combo_list = _identify_combos(model, combo_tags)

    # Calculate the reactions node by node
    for node in model.Nodes.values():
        
        # Step through each load combination
        for combo in combo_list:
            
            # Initialize reactions for this node and load combination
            node.RxnFX[combo.name] = 0.0
            node.RxnFY[combo.name] = 0.0
            node.RxnFZ[combo.name] = 0.0
            node.RxnMX[combo.name] = 0.0
            node.RxnMY[combo.name] = 0.0
            node.RxnMZ[combo.name] = 0.0

            # Determine if the node has any supports
            if (node.support_DX or node.support_DY or node.support_DZ 
            or  node.support_RX or node.support_RY or node.support_RZ):

                # Sum the spring end forces at the node
                for spring in model.Springs.values():

                    if spring.i_node == node and spring.active[combo.name] == True:
                        
                        # Get the spring's global force matrix
                        # Storing it as a local variable eliminates the need to rebuild it every time a term is needed                    
                        spring_F = spring.F(combo.name)

                        if node.support_DX: node.RxnFX[combo.name] += spring_F[0, 0]
                        if node.support_DY: node.RxnFY[combo.name] += spring_F[1, 0]
                        if node.support_DZ: node.RxnFZ[combo.name] += spring_F[2, 0]
                        if node.support_RX: node.RxnMX[combo.name] += spring_F[3, 0]
                        if node.support_RY: node.RxnMY[combo.name] += spring_F[4, 0]
                        if node.support_RZ: node.RxnMZ[combo.name] += spring_F[5, 0]

                    elif spring.j_node == node and spring.active[combo.name] == True:
                    
                        # Get the spring's global force matrix
                        # Storing it as a local variable eliminates the need to rebuild it every time a term is needed                    
                        spring_F = spring.F(combo.name)
                    
                        if node.support_DX: node.RxnFX[combo.name] += spring_F[6, 0]
                        if node.support_DY: node.RxnFY[combo.name] += spring_F[7, 0]
                        if node.support_DZ: node.RxnFZ[combo.name] += spring_F[8, 0]
                        if node.support_RX: node.RxnMX[combo.name] += spring_F[9, 0]
                        if node.support_RY: node.RxnMY[combo.name] += spring_F[10, 0]
                        if node.support_RZ: node.RxnMZ[combo.name] += spring_F[11, 0]

                # Step through each physical member in the model
                for phys_member in model.Members.values():

                    # Sum the sub-member end forces at the node
                    for member in phys_member.sub_members.values():
                        
                        if member.i_node == node and phys_member.active[combo.name] == True:
                        
                            # Get the member's global force matrix
                            # Storing it as a local variable eliminates the need to rebuild it every time a term is needed                    
                            member_F = member.F(combo.name)

                            if node.support_DX: node.RxnFX[combo.name] += member_F[0, 0]
                            if node.support_DY: node.RxnFY[combo.name] += member_F[1, 0]
                            if node.support_DZ: node.RxnFZ[combo.name] += member_F[2, 0]
                            if node.support_RX: node.RxnMX[combo.name] += member_F[3, 0]
                            if node.support_RY: node.RxnMY[combo.name] += member_F[4, 0]
                            if node.support_RZ: node.RxnMZ[combo.name] += member_F[5, 0]

                        elif member.j_node == node and phys_member.active[combo.name] == True:
                        
                            # Get the member's global force matrix
                            # Storing it as a local variable eliminates the need to rebuild it every time a term is needed                    
                            member_F = member.F(combo.name)
                        
                            if node.support_DX: node.RxnFX[combo.name] += member_F[6, 0]
                            if node.support_DY: node.RxnFY[combo.name] += member_F[7, 0]
                            if node.support_DZ: node.RxnFZ[combo.name] += member_F[8, 0]
                            if node.support_RX: node.RxnMX[combo.name] += member_F[9, 0]
                            if node.support_RY: node.RxnMY[combo.name] += member_F[10, 0]
                            if node.support_RZ: node.RxnMZ[combo.name] += member_F[11, 0]

                # Sum the plate forces at the node
                for plate in model.Plates.values():

                    if plate.i_node == node:

                        # Get the plate's global force matrix
                        # Storing it as a local variable eliminates the need to rebuild it every time a term is needed                    
                        plate_F = plate.F(combo.name)
                
                        if node.support_DX: node.RxnFX[combo.name] += plate_F[0, 0]
                        if node.support_DY: node.RxnFY[combo.name] += plate_F[1, 0]
                        if node.support_DZ: node.RxnFZ[combo.name] += plate_F[2, 0]
                        if node.support_RX: node.RxnMX[combo.name] += plate_F[3, 0]
                        if node.support_RY: node.RxnMY[combo.name] += plate_F[4, 0]
                        if node.support_RZ: node.RxnMZ[combo.name] += plate_F[5, 0]

                    elif plate.j_node == node:

                        # Get the plate's global force matrix
                        # Storing it as a local variable eliminates the need to rebuild it every time a term is needed                    
                        plate_F = plate.F(combo.name)
                
                        if node.support_DX: node.RxnFX[combo.name] += plate_F[6, 0]
                        if node.support_DY: node.RxnFY[combo.name] += plate_F[7, 0]
                        if node.support_DZ: node.RxnFZ[combo.name] += plate_F[8, 0]
                        if node.support_RX: node.RxnMX[combo.name] += plate_F[9, 0]
                        if node.support_RY: node.RxnMY[combo.name] += plate_F[10, 0]
                        if node.support_RZ: node.RxnMZ[combo.name] += plate_F[11, 0]

                    elif plate.m_node == node:

                        # Get the plate's global force matrix
                        # Storing it as a local variable eliminates the need to rebuild it every time a term is needed                    
                        plate_F = plate.F(combo.name)
                
                        if node.support_DX: node.RxnFX[combo.name] += plate_F[12, 0]
                        if node.support_DY: node.RxnFY[combo.name] += plate_F[13, 0]
                        if node.support_DZ: node.RxnFZ[combo.name] += plate_F[14, 0]
                        if node.support_RX: node.RxnMX[combo.name] += plate_F[15, 0]
                        if node.support_RY: node.RxnMY[combo.name] += plate_F[16, 0]
                        if node.support_RZ: node.RxnMZ[combo.name] += plate_F[17, 0]

                    elif plate.n_node == node:

                        # Get the plate's global force matrix
                        # Storing it as a local variable eliminates the need to rebuild it every time a term is needed                    
                        plate_F = plate.F(combo.name)
                
                        if node.support_DX: node.RxnFX[combo.name] += plate_F[18, 0]
                        if node.support_DY: node.RxnFY[combo.name] += plate_F[19, 0]
                        if node.support_DZ: node.RxnFZ[combo.name] += plate_F[20, 0]
                        if node.support_RX: node.RxnMX[combo.name] += plate_F[21, 0]
                        if node.support_RY: node.RxnMY[combo.name] += plate_F[22, 0]
                        if node.support_RZ: node.RxnMZ[combo.name] += plate_F[23, 0]

                # Sum the quad forces at the node
                for quad in model.Quads.values():

                    if quad.m_node == node:

                        # Get the quad's global force matrix
                        # Storing it as a local variable eliminates the need to rebuild it every time a term is needed                    
                        quad_F = quad.F(combo.name)

                        if node.support_DX: node.RxnFX[combo.name] += quad_F[0, 0]
                        if node.support_DY: node.RxnFY[combo.name] += quad_F[1, 0]
                        if node.support_DZ: node.RxnFZ[combo.name] += quad_F[2, 0]
                        if node.support_RX: node.RxnMX[combo.name] += quad_F[3, 0]
                        if node.support_RY: node.RxnMY[combo.name] += quad_F[4, 0]
                        if node.support_RZ: node.RxnMZ[combo.name] += quad_F[5, 0]

                    elif quad.n_node == node:

                        # Get the quad's global force matrix
                        # Storing it as a local variable eliminates the need to rebuild it every time a term is needed                    
                        quad_F = quad.F(combo.name)
                
                        if node.support_DX: node.RxnFX[combo.name] += quad_F[6, 0]
                        if node.support_DY: node.RxnFY[combo.name] += quad_F[7, 0]
                        if node.support_DZ: node.RxnFZ[combo.name] += quad_F[8, 0]
                        if node.support_RX: node.RxnMX[combo.name] += quad_F[9, 0]
                        if node.support_RY: node.RxnMY[combo.name] += quad_F[10, 0]
                        if node.support_RZ: node.RxnMZ[combo.name] += quad_F[11, 0]

                    elif quad.i_node == node:

                        # Get the quad's global force matrix
                        # Storing it as a local variable eliminates the need to rebuild it every time a term is needed                    
                        quad_F = quad.F(combo.name)
                
                        if node.support_DX: node.RxnFX[combo.name] += quad_F[12, 0]
                        if node.support_DY: node.RxnFY[combo.name] += quad_F[13, 0]
                        if node.support_DZ: node.RxnFZ[combo.name] += quad_F[14, 0]
                        if node.support_RX: node.RxnMX[combo.name] += quad_F[15, 0]
                        if node.support_RY: node.RxnMY[combo.name] += quad_F[16, 0]
                        if node.support_RZ: node.RxnMZ[combo.name] += quad_F[17, 0]

                    elif quad.j_node == node:

                        # Get the quad's global force matrix
                        # Storing it as a local variable eliminates the need to rebuild it every time a term is needed                    
                        quad_F = quad.F(combo.name)
                
                        if node.support_DX: node.RxnFX[combo.name] += quad_F[18, 0]
                        if node.support_DY: node.RxnFY[combo.name] += quad_F[19, 0]
                        if node.support_DZ: node.RxnFZ[combo.name] += quad_F[20, 0]
                        if node.support_RX: node.RxnMX[combo.name] += quad_F[21, 0]
                        if node.support_RY: node.RxnMY[combo.name] += quad_F[22, 0]
                        if node.support_RZ: node.RxnMZ[combo.name] += quad_F[23, 0]
                
                # Sum the joint loads applied to the node
                for load in node.NodeLoads:

                    for case, factor in combo.factors.items():
                        
                        if load[2] == case:

                            if load[0] == 'FX' and node.support_DX:
                                node.RxnFX[combo.name] -= load[1]*factor
                            elif load[0] == 'FY' and node.support_DY:
                                node.RxnFY[combo.name] -= load[1]*factor
                            elif load[0] == 'FZ' and node.support_DZ:
                                node.RxnFZ[combo.name] -= load[1]*factor
                            elif load[0] == 'MX' and node.support_RX:
                                node.RxnMX[combo.name] -= load[1]*factor
                            elif load[0] == 'MY' and node.support_RY:
                                node.RxnMY[combo.name] -= load[1]*factor
                            elif load[0] == 'MZ' and node.support_RZ:
                                node.RxnMZ[combo.name] -= load[1]*factor
            
            # Calculate any reactions due to active spring supports at the node
            if node.spring_DX[0] != None and node.spring_DX[2] == True:
                sign = node.spring_DX[1]
                k = node.spring_DX[0]
                if sign != None: k = float(sign + str(k))
                DX = node.DX[combo.name]
                node.RxnFX[combo.name] += k*DX
            if node.spring_DY[0] != None and node.spring_DY[2] == True:
                sign = node.spring_DY[1]
                k = node.spring_DY[0]
                if sign != None: k = float(sign + str(k))
                DY = node.DY[combo.name]
                node.RxnFY[combo.name] += k*DY
            if node.spring_DZ[0] != None and node.spring_DZ[2] == True:
                sign = node.spring_DZ[1]
                k = node.spring_DZ[0]
                if sign != None: k = float(sign + str(k))
                DZ = node.DZ[combo.name]
                node.RxnFZ[combo.name] += k*DZ
            if node.spring_RX[0] != None and node.spring_RX[2] == True:
                sign = node.spring_RX[1]
                k = node.spring_RX[0]
                if sign != None: k = float(sign + str(k))
                RX = node.RX[combo.name]
                node.RxnMX[combo.name] += k*RX
            if node.spring_RY[0] != None and node.spring_RY[2] == True:
                sign = node.spring_RY[1]
                k = node.spring_RY[0]
                if sign != None: k = float(sign + str(k))
                RY = node.RY[combo.name]
                node.RxnMY[combo.name] += k*RY
            if node.spring_RZ[0] != None and node.spring_RZ[2] == True:
                sign = node.spring_RZ[1]
                k = node.spring_RZ[0]
                if sign != None: k = float(sign + str(k))
                RZ = node.RZ[combo.name]
                node.RxnMZ[combo.name] += k*RZ

def _check_statics(model, combo_tags=None):
    '''
    Checks static equilibrium and prints results to the console.

    Parameters
    ----------
    precision : number
        The number of decimal places to carry the results to.
    '''

    print('+----------------+')
    print('| Statics Check: |')
    print('+----------------+')
    print('')

    from prettytable import PrettyTable

    # Start a blank table and create a header row
    statics_table = PrettyTable()
    statics_table.field_names = ['Load Combination', 'Sum FX', 'Sum RX', 'Sum FY', 'Sum RY', 'Sum FZ', 'Sum RZ', 'Sum MX', 'Sum RMX', 'Sum MY', 'Sum RMY', 'Sum MZ', 'Sum RMZ']

    # Identify which load combinations to evaluate
    if combo_tags is None:
        combo_list = model.LoadCombos.values()
    else:
        combo_list = []
        for combo in model.LoadCombos.values():
            if any(tag in combo.combo_tags for tag in combo_tags):
                combo_list.append(combo)

    # Step through each load combination
    for combo in combo_list:

        # Initialize force and moment summations to zero
        SumFX, SumFY, SumFZ = 0.0, 0.0, 0.0
        SumMX, SumMY, SumMZ = 0.0, 0.0, 0.0
        SumRFX, SumRFY, SumRFZ = 0.0, 0.0, 0.0
        SumRMX, SumRMY, SumRMZ = 0.0, 0.0, 0.0

        # Get the global force vector and the global fixed end reaction vector
        P = model.P(combo.name)
        FER = model.FER(combo.name)

        # Step through each node and sum its forces
        for node in model.Nodes.values():

            # Get the node's coordinates
            X = node.X
            Y = node.Y
            Z = node.Z

            # Get the nodal forces
            FX = P[node.ID*6+0][0] - FER[node.ID*6+0][0]
            FY = P[node.ID*6+1][0] - FER[node.ID*6+1][0]
            FZ = P[node.ID*6+2][0] - FER[node.ID*6+2][0]
            MX = P[node.ID*6+3][0] - FER[node.ID*6+3][0]
            MY = P[node.ID*6+4][0] - FER[node.ID*6+4][0]
            MZ = P[node.ID*6+5][0] - FER[node.ID*6+5][0]

            # Get the nodal reactions
            RFX = node.RxnFX[combo.name]
            RFY = node.RxnFY[combo.name]
            RFZ = node.RxnFZ[combo.name]
            RMX = node.RxnMX[combo.name]
            RMY = node.RxnMY[combo.name]
            RMZ = node.RxnMZ[combo.name]

            # Sum the global forces
            SumFX += FX
            SumFY += FY
            SumFZ += FZ
            SumMX += MX - FY*Z + FZ*Y
            SumMY += MY + FX*Z - FZ*X
            SumMZ += MZ - FX*Y + FY*X

            # Sum the global reactions
            SumRFX += RFX
            SumRFY += RFY
            SumRFZ += RFZ
            SumRMX += RMX - RFY*Z + RFZ*Y
            SumRMY += RMY + RFX*Z - RFZ*X
            SumRMZ += RMZ - RFX*Y + RFY*X 

        # Add the results to the table
        statics_table.add_row([combo.name, '{:.3g}'.format(SumFX), '{:.3g}'.format(SumRFX),
                                            '{:.3g}'.format(SumFY), '{:.3g}'.format(SumRFY),
                                            '{:.3g}'.format(SumFZ), '{:.3g}'.format(SumRFZ),
                                            '{:.3g}'.format(SumMX), '{:.3g}'.format(SumRMX),
                                            '{:.3g}'.format(SumMY), '{:.3g}'.format(SumRMY),
                                            '{:.3g}'.format(SumMZ), '{:.3g}'.format(SumRMZ)])

    # Print the static check table
    print(statics_table)
    print('')
    
def _partition_D(model):
    """Builds a list with known nodal displacements and with the positions in global stiffness matrix of known and unknown nodal displacements

    :return: A list of the global matrix indices for the unknown nodal displacements (D1_indices). A list of the global matrix indices for the known nodal displacements (D2_indices). A list of the known nodal displacements (D2).
    :rtype: list, list, list
    """

    D1_indices = [] # A list of the indices for the unknown nodal displacements
    D2_indices = [] # A list of the indices for the known nodal displacements
    D2 = []         # A list of the values of the known nodal displacements (D != None)

    # Create the auxiliary table
    for node in model.Nodes.values():
        
        # Unknown displacement DX
        if node.support_DX==False and node.EnforcedDX == None:
            D1_indices.append(node.ID*6 + 0)
        # Known displacement DX
        elif node.EnforcedDX != None:
            D2_indices.append(node.ID*6 + 0)
            D2.append(node.EnforcedDX)
        # Support at DX
        else:
            D2_indices.append(node.ID*6 + 0)
            D2.append(0.0)

        # Unknown displacement DY
        if node.support_DY == False and node.EnforcedDY == None:
            D1_indices.append(node.ID*6 + 1)
        # Known displacement DY
        elif node.EnforcedDY != None:
            D2_indices.append(node.ID*6 + 1)
            D2.append(node.EnforcedDY)
        # Support at DY
        else:
            D2_indices.append(node.ID*6 + 1)
            D2.append(0.0)

        # Unknown displacement DZ
        if node.support_DZ == False and node.EnforcedDZ == None:
            D1_indices.append(node.ID*6 + 2)
        # Known displacement DZ
        elif node.EnforcedDZ != None:
            D2_indices.append(node.ID*6 + 2)
            D2.append(node.EnforcedDZ)
        # Support at DZ
        else:
            D2_indices.append(node.ID*6 + 2)
            D2.append(0.0)

        # Unknown displacement RX
        if node.support_RX == False and node.EnforcedRX == None:
            D1_indices.append(node.ID*6 + 3)
        # Known displacement RX
        elif node.EnforcedRX != None:
            D2_indices.append(node.ID*6 + 3)
            D2.append(node.EnforcedRX)
        # Support at RX
        else:
            D2_indices.append(node.ID*6 + 3)
            D2.append(0.0)

        # Unknown displacement RY
        if node.support_RY == False and node.EnforcedRY == None:
            D1_indices.append(node.ID*6 + 4)
        # Known displacement RY
        elif node.EnforcedRY != None:
            D2_indices.append(node.ID*6 + 4)
            D2.append(node.EnforcedRY)
        # Support at RY
        else:
            D2_indices.append(node.ID*6 + 4)
            D2.append(0.0)

        # Unknown displacement RZ
        if node.support_RZ == False and node.EnforcedRZ == None:
            D1_indices.append(node.ID*6 + 5)
        # Known displacement RZ
        elif node.EnforcedRZ != None:
            D2_indices.append(node.ID*6 + 5)
            D2.append(node.EnforcedRZ)
        # Support at RZ
        else:
            D2_indices.append(node.ID*6 + 5)
            D2.append(0.0)
    
    # Legacy code on the next line. I will leave it here until the line that follows has been proven over time.
    # D2 = atleast_2d(D2)
    
    # Convert D2 from a list to a matrix
    D2 = array(D2, ndmin=2).T

    # Return the indices and the known displacements
    return D1_indices, D2_indices, D2


def _partition(model, unp_matrix, D1_indices, D2_indices):
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


def _D2(model, expanded_time, D2_length):
    """
    Builds the enforced displacements, velocities and accelerations for the entire duration of analysis.

    :param model: The finite element model
    :type model: FEModel3D
    :param expanded_time: An array of time instances for analysis
    :type expanded_time: ndarray
    :param D2_length: Number of constrained degrees of freedom
    :type D2_length: int
    :return: Prescribed displacements, velocities and accelerstions for the entire duration on analysis.
    :rtype: ndarray, ndarray, ndarray, ndarray

    """

    expanded_time_length = len(expanded_time)

    D2 = zeros((D2_length, expanded_time_length))
    V2 = zeros((D2_length, expanded_time_length))
    A2 = zeros((D2_length, expanded_time_length))

    # Default displacement profile, which is just ones
    default_profile = ones((1,expanded_time_length))

    profile_values= model.DisplacementProfiles
    profile_keys = model.DisplacementProfiles.keys()
    # Create the auxiliary table

    # Variable for indexing
    i = 0
    for node in model.Nodes.values():

        # Unknown displacement DX
        if node.support_DX==False and node.EnforcedDX == None:
            # In future the function _partition() and this function can be merged. Some of the code from
            # _partition() can be put here
            pass
        # Known displacement DX
        elif node.EnforcedDX != None:
            if node.name in profile_keys and profile_values[node.name].direction == 'DX':
                node_name = node.name

                # Original time and displacement profile array
                time_array = profile_values[node_name].time
                displacement_profile = profile_values[node_name].profile

                # To get the velocity and acceleration arrays, we need the arrays to be closely spaced
                # inorder to carry out proper numerical differentiation
                # Hence we generate new arrays from the originals by interpolation
                new_time_array = linspace(min(time_array),max(time_array),10*expanded_time_length)
                new_disp_profile = interp(new_time_array,time_array,displacement_profile)

                # The velocity profile is found by differentiating the new displacement profile
                velocity_profile = gradient(new_disp_profile,new_time_array)

                # The acceleration profile is found by differentiating the velocity profile
                acceleration_profile = gradient(velocity_profile,new_time_array)

                # The final displacement profile for the given time array is then calculated
                disp_profile_final = interp(expanded_time,time_array,displacement_profile)

                # The velocity and acceleration profiles are also found by interpolating
                # their respective profiles
                vel_profile_final = interp(expanded_time, new_time_array, velocity_profile)
                acc_profile_final = interp(expanded_time, new_time_array, acceleration_profile)

                # The actual values are then calculated and appended
                D2[i,:] = disp_profile_final*node.EnforcedDX
                V2[i,:] = vel_profile_final*node.EnforcedDX
                A2[i,:] = acc_profile_final*node.EnforcedDX
            else:
                # If there is no profile defined, then the displacement is constant
                # Hence the velocity and acceleration are zeros
                D2[i,:] = node.EnforcedDX * default_profile

            # Increment i
            i += 1

        # Support at DX
        else:
            # D2, V2 and A2 are zeros at the supports, which is what we have initialised the matrices to.
            # Hence, we only need to update our counter
            i+=1

        # Unknown displacement DY
        if node.support_DY==False and node.EnforcedDY == None:
            # In future the function _partition() and this function can be merged. Some of the code from
            # _partition() can be put here
            pass
        # Known displacement DY
        elif node.EnforcedDY != None:
            if node.name in profile_keys and profile_values[node.name].direction == 'DY':
                node_name = node.name

                # Original time and displacement profile array
                time_array = profile_values[node_name].time
                displacement_profile = profile_values[node_name].profile

                # To get the velocity and acceleration arrays, we need the arrays to be closely spaced
                # inorder to carry out proper numerical differentiation
                # Hence we generate new arrays from the originals by interpolation
                new_time_array = linspace(min(time_array),max(time_array),10*expanded_time_length)
                new_disp_profile = interp(new_time_array,time_array,displacement_profile)

                # The velocity profile is found by differentiating the new displacement profile
                velocity_profile = gradient(new_disp_profile,new_time_array)

                # The acceleration profile is found by differentiating the velocity profile
                acceleration_profile = gradient(velocity_profile,new_time_array)

                # The final displacement profile for the given time array is then calculated
                disp_profile_final = interp(expanded_time,time_array,displacement_profile)

                # The velocity and acceleration profiles are also found by interpolating
                # their respective profiles
                vel_profile_final = interp(expanded_time, new_time_array, velocity_profile)
                acc_profile_final = interp(expanded_time, new_time_array, acceleration_profile)

                # The actual values are then calculated and appended
                D2[i,:] = disp_profile_final*node.EnforcedDY
                V2[i,:] = vel_profile_final*node.EnforcedDY
                A2[i,:] = acc_profile_final*node.EnforcedDY
            else:
                # If there is no profile defined, then the displacement is constant
                # Hence the velocity and acceleration are zeros
                D2[i,:] = node.EnforcedDY * default_profile

            # Increment i
            i+=1

        # Support at DY
        else:
            # D2, V2 and A2 are zeros at the supports, which is what we have initialised the matrices to.
            # Hence, we only need to update our counter
            i+=1

        # Unknown displacement DZ
        if node.support_DZ==False and node.EnforcedDZ == None:
            # In future the function _partition() and this function can be merged. Some of the code from
            # _partition() can be put here
            pass
        # Known displacement DZ
        elif node.EnforcedDZ != None:
            if node.name in profile_keys and profile_values[node.name].direction == 'DZ':
                node_name = node.name

                # Original time and displacement profile array
                time_array = profile_values[node_name].time
                displacement_profile = profile_values[node_name].profile

                # To get the velocity and acceleration arrays, we need the arrays to be closely spaced
                # inorder to carry out proper numerical differentiation
                # Hence we generate new arrays from the originals by interpolation
                new_time_array = linspace(min(time_array),max(time_array),10*expanded_time_length)
                new_disp_profile = interp(new_time_array,time_array,displacement_profile)

                # The velocity profile is found by differentiating the new displacement profile
                velocity_profile = gradient(new_disp_profile,new_time_array)

                # The acceleration profile is found by differentiating the velocity profile
                acceleration_profile = gradient(velocity_profile,new_time_array)

                # The final displacement profile for the given time array is then calculated
                disp_profile_final = interp(expanded_time,time_array,displacement_profile)

                # The velocity and acceleration profiles are also found by interpolating
                # their respective profiles
                vel_profile_final = interp(expanded_time, new_time_array, velocity_profile)
                acc_profile_final = interp(expanded_time, new_time_array, acceleration_profile)

                # The actual values are then calculated and appended
                D2[i,:] = disp_profile_final*node.EnforcedDZ
                V2[i,:] = vel_profile_final*node.EnforcedDZ
                A2[i,:] = acc_profile_final*node.EnforcedDZ
            else:
                # If there is no profile defined, then the displacement is constant
                # Hence the velocity and acceleration are zeros
                D2[i,:] = node.EnforcedDZ * default_profile

            # Increment i
            i+=1

        # Support at DZ
        else:
            # D2, V2 and A2 are zeros at the supports, which is what we have initialised the matrices to.
            # Hence, we only need to update our counter
            i+=1

        # Unknown displacement RX
        if node.support_RX==False and node.EnforcedRX == None:
            # In future the function _partition() and this function can be merged. Some of the code from
            # _partition() can be put here
            pass
        # Known displacement RX
        elif node.EnforcedRX != None:
            if node.name in profile_keys and profile_values[node.name].direction == 'RX':
                node_name = node.name

                # Original time and displacement profile array
                time_array = profile_values[node_name].time
                displacement_profile = profile_values[node_name].profile

                # To get the velocity and acceleration arrays, we need the arrays to be closely spaced
                # inorder to carry out proper numerical differentiation
                # Hence we generate new arrays from the originals by interpolation
                new_time_array = linspace(min(time_array),max(time_array),10*expanded_time_length)
                new_disp_profile = interp(new_time_array,time_array,displacement_profile)

                # The velocity profile is found by differentiating the new displacement profile
                velocity_profile = gradient(new_disp_profile,new_time_array)

                # The acceleration profile is found by differentiating the velocity profile
                acceleration_profile = gradient(velocity_profile,new_time_array)

                # The final displacement profile for the given time array is then calculated
                disp_profile_final = interp(expanded_time,time_array,displacement_profile)

                # The velocity and acceleration profiles are also found by interpolating
                # their respective profiles
                vel_profile_final = interp(expanded_time, new_time_array, velocity_profile)
                acc_profile_final = interp(expanded_time, new_time_array, acceleration_profile)

                # The actual values are then calculated and appended
                D2[i,:] = disp_profile_final*node.EnforcedRX
                V2[i,:] = vel_profile_final*node.EnforcedRX
                A2[i,:] = acc_profile_final*node.EnforcedRX
            else:
                # If there is no profile defined, then the displacement is constant
                # Hence the velocity and acceleration are zeros
                D2[i,:] = node.EnforcedRX * default_profile

            # Increment i
            i+=1

        # Support at RX
        else:
            # D2, V2 and A2 are zeros at the supports, which is what we have initialised the matrices to.
            # Hence, we only need to update our counter
            i+=1

        # Unknown displacement RY
        if node.support_RY==False and node.EnforcedRY == None:
            # In future the function _partition() and this function can be merged. Some of the code from
            # _partition() can be put here
            pass
        # Known displacement RY
        elif node.EnforcedRY != None:
            if node.name in profile_keys and profile_values[node.name].direction == 'RY':
                node_name = node.name

                # Original time and displacement profile array
                time_array = profile_values[node_name].time
                displacement_profile = profile_values[node_name].profile

                # To get the velocity and acceleration arrays, we need the arrays to be closely spaced
                # inorder to carry out proper numerical differentiation
                # Hence we generate new arrays from the originals by interpolation
                new_time_array = linspace(min(time_array),max(time_array),10*expanded_time_length)
                new_disp_profile = interp(new_time_array,time_array,displacement_profile)

                # The velocity profile is found by differentiating the new displacement profile
                velocity_profile = gradient(new_disp_profile,new_time_array)

                # The acceleration profile is found by differentiating the velocity profile
                acceleration_profile = gradient(velocity_profile,new_time_array)

                # The final displacement profile for the given time array is then calculated
                disp_profile_final = interp(expanded_time,time_array,displacement_profile)

                # The velocity and acceleration profiles are also found by interpolating
                # their respective profiles
                vel_profile_final = interp(expanded_time, new_time_array, velocity_profile)
                acc_profile_final = interp(expanded_time, new_time_array, acceleration_profile)

                # The actual values are then calculated and appended
                D2[i,:] = disp_profile_final*node.EnforcedRY
                V2[i,:] = vel_profile_final*node.EnforcedRY
                A2[i,:] = acc_profile_final*node.EnforcedRY
            else:
                # If there is no profile defined, then the displacement is constant
                # Hence the velocity and acceleration are zeros
                D2[i,:] = node.EnforcedRY * default_profile

            # Increment i
            i+=1

        # Support at RY
        else:
            # D2, V2 and A2 are zeros at the supports, which is what we have initialised the matrices to.
            # Hence, we only need to update our counter
            i+=1


        # Unknown displacement RZ
        if node.support_RZ==False and node.EnforcedRZ == None:
            # In future the function _partition() and this function can be merged. Some of the code from
            # _partition() can be put here
            pass
        # Known displacement RZ
        elif node.EnforcedRZ != None:
            if node.name in profile_keys and profile_values[node.name].direction == 'RZ':
                node_name = node.name

                # Original time and displacement profile array
                time_array = profile_values[node_name].time
                displacement_profile = profile_values[node_name].profile

                # To get the velocity and acceleration arrays, we need the arrays to be closely spaced
                # inorder to carry out proper numerical differentiation
                # Hence we generate new arrays from the originals by interpolation
                new_time_array = linspace(min(time_array),max(time_array),10*expanded_time_length)
                new_disp_profile = interp(new_time_array,time_array,displacement_profile)

                # The velocity profile is found by differentiating the new displacement profile
                velocity_profile = gradient(new_disp_profile,new_time_array)

                # The acceleration profile is found by differentiating the velocity profile
                acceleration_profile = gradient(velocity_profile,new_time_array)

                # The final displacement profile for the given time array is then calculated
                disp_profile_final = interp(expanded_time,time_array,displacement_profile)

                # The velocity and acceleration profiles are also found by interpolating
                # their respective profiles
                vel_profile_final = interp(expanded_time, new_time_array, velocity_profile)
                acc_profile_final = interp(expanded_time, new_time_array, acceleration_profile)

                # The actual values are then calculated and appended
                D2[i,:] = disp_profile_final*node.EnforcedRZ
                V2[i,:] = vel_profile_final*node.EnforcedRZ
                A2[i,:] = acc_profile_final*node.EnforcedRZ
            else:
                # If there is no profile defined, then the displacement is constant
                # Hence the velocity and acceleration are zeros
                D2[i,:] = node.EnforcedRZ * default_profile

            # Increment i
            i+=1

        # Support at RZ
        else:
            # D2, V2 and A2 are zeros at the supports, which is what we have initialised the matrices to.
            # Hence, we only need to update our counter
            i+=1


    return D2, V2, A2


def _renumber(model):
    """
    Assigns node and element ID numbers to be used internally by the program. Numbers are
    assigned according to the order in which they occur in each dictionary.
    """

    # Number each node in the model
    for id, node in enumerate(model.Nodes.values()):
        node.ID = id
    
    # Number each spring in the model
    for id, spring in enumerate(model.Springs.values()):
        spring.ID = id

    # Descritize all the physical members and number each member in the model
    id = 0
    for phys_member in model.Members.values():
        phys_member.descritize()
        for member in phys_member.sub_members.values():
            member.ID = id
            id += 1
    
    # Number each plate in the model
    for id, plate in enumerate(model.Plates.values()):
        plate.ID = id
    
    # Number each quadrilateral in the model
    for id, quad in enumerate(model.Quads.values()):
        quad.ID = id


def _transient_solver_linear_modal(d0_n, v0_n, F0_n, F_n, step_size, required_duration,
                                   steps_for_results_recording,
                                   mass_normalised_eigen_vectors, natural_freq,
                                   newmark_gamma, newmark_beta,
                                   taylor_alpha, wilson_theta, damping_options = dict(),
                                   log = False):
    """
        General time history solver using modal superposition.

        :param d0_n: Initial displacement in modal coordinates.
        :type d0_n: numpy.ndarray
        :param v0_n: Initial velocity in modal coordinates.
        :type v0_n: numpy.ndarray
        :param F0_n: Initial applied forces in modal coordinates
        :type F0_n: numpy.ndarray
        :param F_n: Time-varying applied forces in modal coordinates.
        :type F_n: numpy.ndarray
        :param step_size: Time step for the analysis.
        :type step_size: float
        :param required_duration: Total duration of the analysis.
        :type required_duration: float
        :param steps_for_results_recording: An array of steps for which results should be recorded
        :type steps_for_results_recording: ndarray
        :param mass_normalised_eigen_vectors: Mass-normalized eigen vectors.
        :type mass_normalised_eigen_vectors: numpy.ndarray
        :param natural_freq: Angular frequencies of the modes.
        :type natural_freq: numpy.ndarray
        :param newmark_gamma: Newmark gamma parameter for time integration.
        :type newmark_gamma: float
        :param newmark_beta: Newmark beta parameter for time integration.
        :type newmark_beta: float
        :param taylor_alpha: Taylor alpha parameter for time integration.
        :type taylor_alpha: float
        :param wilson_theta: Wilson theta parameter for time integration.
        :type wilson_theta: float
        :param damping_options: Dictionary containing damping options (optional).
            \nAllowed keywords in damping_options:
            - 'constant_modal_damping' (float): Constant modal damping ratio (default: 0.00).
            - 'r_alpha' (float): Rayleigh mass proportional damping coefficient.
            - 'r_beta' (float): Rayleigh stiffness proportional damping coefficient.
            - 'first_mode_damping' (float): Damping ratio for the first mode.
            - 'highest_mode_damping' (float): Damping ratio for the highest mode.
            - 'damping_in_every_mode' (list or tuple): Damping ratio(s) for each mode.

        :type damping_options: dict, optional
        :param log: If True, print analysis log to the console. Default is False.
        :type log: bool, optional

        :return: Tuple containing TIME, D (displacement history), V (velocity history),
          A (acceleration history).
        :rtype: tuple of numpy.ndarray
        :raises ValueError: If invalid input is provided.
        """

    # Initialise the damping options
    constant_modal_damping = 0.00
    rayleigh_alpha= None
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


    # Shorten variable names
    w = natural_freq
    gamma = newmark_gamma
    beta = newmark_beta
    alpha = taylor_alpha
    theta = wilson_theta
    step = step_size

    # Get size of matrices
    size = F_n.shape[0]

    # Build the modal stiffness  matrix
    K_n = w**2

    # Build the modal mass matrix
    M_n = ones(size)

    # Build the modal damping matrix
    # Initialise a one dimensional array
    C_n = zeros(F_n.shape[0])
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

    # Compute the adjusted time step, tau
    tau = theta * step

    # Calculate the initial acceleration
    a0_n = F0_n - C_n * v0_n - K_n * d0_n

    # Compute the constant coefficient matrix, CCM
    CCM = M_n + gamma * tau * C_n + (1+alpha) * beta * tau**2 * K_n

    # Invert the CCM
    CCM = 1/CCM

    # Calculate the required number of steps
    total_steps = ceil(required_duration/step) + 1

    # Initialise the matrices to store displacements, velocities and accelerations
    length_of_results = len(steps_for_results_recording)
    D_n = zeros((size, length_of_results))
    V_n = zeros((size, length_of_results))
    A_n = zeros((size, length_of_results))

    # Store the initial values of displacements, velocities and accelerations
    D_n[:,0] = d0_n
    V_n[:,0] = v0_n
    A_n[:,0] = a0_n

    d_prev = d0_n
    v_prev = v0_n
    a_prev = a0_n

    # Index for recording results
    j = 0

    for i in range(total_steps-1):
        if log:
            update_progress(i, total_steps - 2, '- Analysis Progress')

        # Calculate the predictors
        dp = d_prev + tau * v_prev + tau**2 * (0.5-beta) * a_prev
        vp = v_prev + tau * (1-gamma) * a_prev

        # Calculate right hand side, RHS
        RHS = (1 - theta) * F_n[:, i] + theta * F_n[:, i + 1] - C_n * vp - K_n * ((1 + alpha) * dp - alpha * d_prev)

        # Calculate the collocation acceleration
        A_col = CCM * RHS

        # Compute and save the acceleration, velocity and displacement
        a_current = a_prev + (1/theta) * (A_col - a_prev)
        v_current = v_prev + step * (1-gamma) * a_prev + gamma * step * a_current
        d_current = d_prev + step * v_prev + step**2 * (0.5-beta) * a_prev + beta * step**2 * a_current

        if i + 1 in steps_for_results_recording:
            j += 1
            A_n[:, j] = a_current
            V_n[:, j] = v_current
            D_n[:, j] = d_current

        a_prev = a_current
        v_prev = v_current
        d_prev = d_current

    # Calculate the physical coordinates
    A = mass_normalised_eigen_vectors @ A_n
    V = mass_normalised_eigen_vectors @ V_n
    D = mass_normalised_eigen_vectors @ D_n
    import numpy
    print(numpy.linalg.norm(D))
    return D, V, A


def _transient_solver_linear_direct(K, M, d0, v0, F0, F,
                                    step_size, required_duration,
                                    steps_for_results_recording,
                                    newmark_gamma, newmark_beta,
                                    taylor_alpha, wilson_theta,
                                    rayleigh_alpha = 0, rayleigh_beta = 0,

                                    sparse=True, log = False):
    """
       General direct linear time history solver.

       :param K: Stiffness matrix.
       :type K: numpy.ndarray or scipy.sparse.csr_matrix
       :param M: Mass matrix.
       :type M: numpy.ndarray or scipy.sparse.csr_matrix
       :param d0: Initial displacement.
       :type d0: numpy.ndarray
       :param v0: Initial velocity.
       :type v0: numpy.ndarray
       :param F0: Initial applied force.
       :type F0: numpy.ndarray
       :param F: Time-varying applied force.
       :type F: numpy.ndarray
       :param step_size: Time step for the analysis.
       :type step_size: float
       :param required_duration: Total duration of the analysis.
       :type required_duration: float
       :param newmark_gamma: Newmark gamma parameter for time integration.
       :type newmark_gamma: float
       :param newmark_beta: Newmark beta parameter for time integration.
       :type newmark_beta: float
       :param taylor_alpha: Taylor alpha parameter for time integration.
       :type taylor_alpha: float
       :param wilson_theta: Wilson theta parameter for time integration.
       :type wilson_theta: float
       :param rayleigh_alpha: Rayleigh mass proportional damping coefficient (alpha) (default: 0).
       :type rayleigh_alpha: float
       :param rayleigh_beta: Rayleigh stiffness proportional damping coefficient (beta) (default: 0).
       :type rayleigh_beta: float
       :param steps_for_results_recording: An array of steps for which results should be recorded
       :type steps_for_results_recording: ndarray
       :param sparse: Indicates whether matrices are sparse (default: True).
       :type sparse: bool
       :param log: If True, print analysis log to the console. Default is False.
       :type log: bool, optional

       :return: Tuple containing TIME, D (displacement history), V (velocity history),
          A (acceleration history).
       :rtype: tuple of numpy.ndarray
       :raises ValueError: If invalid input is provided.
       """

    # Shorten some variable names
    step = step_size
    gamma = newmark_gamma
    beta = newmark_beta
    alpha = taylor_alpha
    theta = wilson_theta

    # Import sparse solver if matrices are sparse
    if sparse == True:
        M = M.tocsc()
        K = K.tocsc()

    # Get size of matrices
    size = F.shape[0]

    # Build the damping matrix
    C = rayleigh_alpha * M + rayleigh_beta * K

    # Compute the adjusted time step, tau
    tau = theta * step

    # Calculate the initial acceleration
    if sparse:
        from scipy.sparse.linalg import spsolve, splu
        a0 = spsolve(M, F0 - C @ v0 - K @ d0)
    else:
        a0 = solve(M, F0 - C @ v0 - K @ d0)

    # Compute the constant coefficient matrix, CCM
    CCM = M + gamma * tau * C + (1+alpha) * beta * tau**2 * K

    # Decompose the constant coefficient matrix
    lu_sparse_CCM = None
    #lu_CCM, lu_piv_CCM = None, None
    if sparse:
        lu_sparse_CCM = splu (CCM.tocsc(),permc_spec="MMD_ATA")
    else:
        #lu_CCM, lu_piv_CCM = lu_factor (CCM, overwrite_a=True)
        from numpy.linalg import qr
        Q_CCM, R_CCM = qr(CCM)


    # Calculate the required number of steps
    total_steps = ceil(required_duration/step) + 1

    # Initialise the matrices to store displacements, velocities and accelerations
    length_of_results = len(steps_for_results_recording)
    D = zeros((size, length_of_results))
    V = zeros((size, length_of_results))
    A = zeros((size, length_of_results))

    # Store the initial values of displacements, velocities and accelerations
    D[:,0] = d0
    V[:,0] = v0
    A[:,0] = a0

    d_prev = d0
    v_prev = v0
    a_prev = a0

    # Index for recording results
    j = 0

    if sparse:
        for i in range(total_steps-1):
            if log:
                update_progress(i,total_steps-2,'- Analysis Progress')

            # Calculate the predictors
            dp = d_prev + tau * v_prev + tau ** 2 * (0.5 - beta) * a_prev
            vp = v_prev + tau * (1 - gamma) * a_prev

            # Calculate right hand side, RHS
            RHS = (1 - theta) * F[:, i] + theta * F[:, i + 1] - C @ vp - K @ ((1 + alpha) * dp - alpha * d_prev)

            # Calculate the collocation acceleration
            A_col = lu_sparse_CCM.solve(RHS)

            # Compute and save the acceleration, velocity and displacement
            a_current = a_prev + (1 / theta) * (A_col - a_prev)
            v_current = v_prev + step * (1 - gamma) * a_prev + gamma * step * a_current
            d_current = d_prev + step * v_prev + step ** 2 * (0.5 - beta) * a_prev + beta * step ** 2 * a_current

            if i+1 in steps_for_results_recording:
                j+=1
                A[:,j] = a_current
                V[:,j] = v_current
                D[:,j] = d_current

            a_prev = a_current
            v_prev = v_current
            d_prev = d_current

    else:

        for i in range(total_steps-1):
            if log:
                update_progress(i,total_steps-2,'- Analysis Progress')

            # Calculate the predictors
            dp = d_prev + tau * v_prev + tau ** 2 * (0.5 - beta) * a_prev
            vp = v_prev + tau * (1 - gamma) * a_prev

            # Calculate right hand side, RHS
            RHS = (1 - theta) * F[:, i] + theta * F[:, i + 1] - C @ vp - K @ ((1 + alpha) * dp - alpha * d_prev)

            # Calculate the collocation acceleration
            #A_col = lu_solve( (lu_CCM, lu_piv_CCM) ,b = RHS , overwrite_b = True)
            A_col = backward_solve(R_CCM, Q_CCM.T @ RHS)

            # Compute and save the acceleration, velocity and displacement
            a_current = a_prev + (1 / theta) * (A_col - a_prev)
            v_current = v_prev + step * (1 - gamma) * a_prev + gamma * step * a_current
            d_current = d_prev + step * v_prev + step ** 2 * (0.5 - beta) * a_prev + beta * step ** 2 * a_current

            if i+1 in steps_for_results_recording:
                j+=1
                A[:,j] = a_current
                V[:,j] = v_current
                D[:,j] = d_current

            a_prev = a_current
            v_prev = v_current
            d_prev = d_current

    return D, V, A

def update_progress(step, total, process_name):
    progress = int(100 * step / total)
    bar_length = 50
    progress_bar = '=' * int(bar_length * progress / 100)
    sys.stdout.write(f'\r{process_name}: [{progress_bar: <{bar_length}}] {progress:3}%')
    sys.stdout.flush()


def backward_solve(U, b):
    """x = back_sub(U, b) is the solution to U x = b
       U must be an upper-triangular matrix
       b must be a vector of the same leading dimension as U
    """
    n = U.shape[0]
    x = zeros(n)
    for i in range(n-1, -1, -1):
        tmp = b[i]
        for j in range(i+1, n):
            tmp -= U[i,j] * x[j]
        x[i] = tmp / U[i,i]
    return x

