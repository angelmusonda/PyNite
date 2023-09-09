import pickle
from PyNite import FEModel3D, Analysis
from numpy import pi, zeros, interp
from PyNite.PyNiteExceptions import InputOutOfRangeError, ResultsNotFoundError
import copy

class ResultsModelBuilder():
    """
    Represents a parent class for other results model builders to inherit from.
    """
    def __int__(self):
        pass

    def _save_response_into_node(self,model:FEModel3D, response, combo_name):
        """
        Saves the calculated global response into each node object.

        Args:
            model (FEModel3D): The solved finite element model.
            response: The calculated response (displacement, velocity, acceleration)
            combo_name (str): The name of the load combination.

        Returns:
            None
        """

        # Store the calculated global response into each node object
        for node in model.Nodes.values():
            node.DX[combo_name] = response[node.ID * 6 + 0, 0]
            node.DY[combo_name] = response[node.ID * 6 + 1, 0]
            node.DZ[combo_name] = response[node.ID * 6 + 2, 0]
            node.RX[combo_name] = response[node.ID * 6 + 3, 0]
            node.RY[combo_name] = response[node.ID * 6 + 4, 0]
            node.RZ[combo_name] = response[node.ID * 6 + 5, 0]

    def _calculate_reactions(self, model: FEModel3D, combo_name,log=True):
        """
        Calculates the reactions of the model.

        Args:
            model (FEModel3D): The solved finite element model.
            combo_name (str): The name of the load combination.

        Returns:
            None
        """
        # Re-calculate reactions
        # We do not want to calculate reactions for all the load combinations
        # Hence we will keep the load combinations in a temporary object
        load_combos_temp = copy.deepcopy(model.LoadCombos)

        # Then remove all other load combos except the required load combo
        model.LoadCombos.clear()
        model.LoadCombos = {combo_name: load_combos_temp[combo_name]}

        # Calculate the reactions
        Analysis._calc_reactions(model,log)

        # Restore the load combos
        model.LoadCombos = copy.deepcopy(load_combos_temp)


class HarmonicResultsModelBuilder(ResultsModelBuilder):
    """
    Represents a results model builder for Harmonic analysis.
    The class is used to build a model with results at a specified load frequency.

    """

    def __init__(self, saved_model):
        """
        Initialize a HarmonicResultsModelBuilder object.

        Args:
            saved_model (str): The path to the saved finite element model file.

        Returns:
            None
        """
        with open(str(saved_model), 'rb') as file:
            self._solved_model: FEModel3D = pickle.load(file)

        self._total_response = None


    def get_solved_model_at_freq(self, freq, combo_name, response_type = "D", log=False):
        """
        Get the model's response at a specific frequency. This model can be visualised and can be used to extract other
        response quantities, graphs, for every member at the specified frequency of load

        Args:
            freq (float): The frequency of the load at which results are required.
            combo_name (str): The name of the harmonic load combination.
            response_type (str, optional, default="D"): The type of response:
                - "D": Displacement response (default).
                - "V": Velocity response.
                - "A": Acceleration response.

        Returns:
            FEModel3D: The finite element model with Harmonic response stored in the nodes.

        Raises:
            InputOutOfRange Error, ResultsNotFound Error
        """

        model = self._solved_model

        # Check if results are available
        if model.DynamicSolution['Harmonic'] == False:
            raise ResultsNotFoundError

        # Check if the frequency is within the calculated range
        if freq<min(model.LoadFrequencies) or freq>max(model.LoadFrequencies):
            raise InputOutOfRangeError

        # Determine the type of response quantity requested for and get the total response
        if response_type == "D":
            self._total_response = model.DISPLACEMENT_AMPLITUDE()
        elif response_type == "V":
            self._total_response = model.DISPLACEMENT_AMPLITUDE()*(2*pi*freq)
        elif response_type == "A":
            self._total_response = model.DISPLACEMENT_AMPLITUDE()*(2*pi*freq)**2
        else:
            self._total_response = model.DISPLACEMENT_AMPLITUDE()

        # Get the number of degrees of freedom for building the response vector at a specified load frequency
        dof = self._total_response.shape[0]

        # Initialise the response vector at a specified load frequency
        response_at_freq = zeros((dof, 1))

        # Build the response at a specified frequency from the total response through interpolation
        for i in range(dof):
            # Linear interpolation
            response_at_freq[i, 0] = interp(freq, model.LoadFrequencies, self._total_response[i, :])

        # Now that the response is interpolated, put it into respective nodes
        self._save_response_into_node(model=model,response=response_at_freq,combo_name=combo_name)

        # Calculate the reactions
        self._calculate_reactions(model = model, combo_name=combo_name, log=log)
        # Return the model with requested for response quantity and load frequency
        return model

class ModalResultsModelBuilder(ResultsModelBuilder):
    """
    Represents a results model builder for Modal analysis.
    The class is used to build a model with mode shapes of a specified mode.

    """

    def __init__(self, saved_model):
        """
        Initialize a ModalResultsModelBuilder object.

        Args:
            saved_model (str): The path to the saved finite element model file.

        Returns:
            None
        """
        with open(str(saved_model), 'rb') as file:
            self._solved_model: FEModel3D = pickle.load(file)

    def get_solved_model_for_mode(self, mode=1):
        """
        Get the model for a specified mode. This model's modal deformation can be visualised.

        Args:
            mode (int): The required mode

        Returns:
            FEModel3D: The finite element model with Modal deformation stored in the nodes.

        Raises:
            InputOutOfRange, ResultsNotFound Error
        """

        model = self._solved_model
        mode = int(mode)

        # Check if results are available
        if model.DynamicSolution['Modal'] == False:
            raise ResultsNotFoundError

        # Get the Mode Shapes
        Mode_Shapes = model.MODE_SHAPES()
        Natural_Frequencies = model.NATURAL_FREQUENCIES()

        # Check that the requested mode is among the calculated modes
        if mode < 1 or mode > Mode_Shapes.shape[0]:
            raise InputOutOfRangeError

        # Get the required mode shape
        Single_Mode_Shape = (Mode_Shapes[:,mode-1])
        Single_Mode_Shape = Single_Mode_Shape.reshape(len(Single_Mode_Shape),1)
        # Now that the response is interpolated, put it into respective nodes
        self._save_response_into_node(model=model, response=Single_Mode_Shape, combo_name='Modal Combo')

        # Return the model with requested for response quantity and load frequency
        return model

    def get_natural_frequency_for_mode(self, mode):
        """
        Get the natural frequency for specified mode.
        Args:
            mode (int): The mode for which frequency is required
        Returns:
            float: The natural frequency of the specified mode.

        Raises:
            InputOutOfRange, ResultsNotFound Error
        """

        model = self._solved_model
        mode = int(mode)

        # Check if results are available
        if model.DynamicSolution['Modal'] == False:
            raise ResultsNotFoundError

        return model.NATURAL_FREQUENCIES()[mode - 1]

    def get_natural_frequencies(self):
        """
        Get the all the calculated natural frequencies of the model.
        Returns:
            list: The natural frequencies of the model.

        Raises:
             ResultsNotFound Error
        """

        model = self._solved_model

        # Check if results are available
        if model.DynamicSolution['Modal'] == False:
            raise ResultsNotFoundError

        return model.NATURAL_FREQUENCIES()


# TESTING
model_builder = HarmonicResultsModelBuilder("model.pickle")
solved_model =model_builder.get_solved_model_at_freq(freq=3,combo_name='COMB1',log=True )
load_freq = solved_model.LoadFrequencies

for freq in load_freq:
    solved_model = model_builder.get_solved_model_at_freq(freq, combo_name='COMB1',log=False)
    print(round(freq,2) , " : ", round(1000 * solved_model.Nodes['N11'].DX['COMB1']))


"""from PyNite.Visualization import render_model
render_model(model = solved_model,
             render_loads=False,
             annotation_size=0.05,
             deformed_shape=False,
             deformed_scale=30,
             combo_name='COMB1')"""





