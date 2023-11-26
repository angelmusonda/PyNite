import pickle
from PyNite import FEModel3D, Analysis
from numpy import pi, zeros, interp, searchsorted
from PyNite.PyNiteExceptions import InputOutOfRangeError, ResultsNotFoundError
import copy

class ResultsModelBuilder():
    """
    Represents a parent class for other results model builders to inherit from.

    Args:
        saved_model (str): The path to the saved finite element model file.
    """
    def __init__(self,saved_model_path):
        """
        Initializes a ResultsModelBuilder instance by loading the solved finite element model from a saved file.

        Args:
            saved_model_path(str): The path to the solved and saved finite element model file.
        """
        with open(str(saved_model_path), 'rb') as file:
            self._solved_model: FEModel3D = pickle.load(file)


    def _save_response_into_node(self,model:FEModel3D, response, combo_name, reaction = None):
        """
        Saves the calculated global response into each node object.

        Args:
            model (FEModel3D): The solved finite element model.
            response: The calculated response (displacement, velocity, acceleration)
            combo_name (str): The name of the load combination.
            reaction: The calculated reaction, can be real or imaginary in case of harmonic analysis
            combo_name (str): The name of the load combination.

        Returns:
            None
        """

        # Store the calculated global response into each node object
        if reaction is not None:
            for node in model.Nodes.values():

                # Displacement, Velocity or Acceleration
                node.DX[combo_name] = response[node.ID * 6 + 0, 0]
                node.DY[combo_name] = response[node.ID * 6 + 1, 0]
                node.DZ[combo_name] = response[node.ID * 6 + 2, 0]
                node.RX[combo_name] = response[node.ID * 6 + 3, 0]
                node.RY[combo_name] = response[node.ID * 6 + 4, 0]
                node.RZ[combo_name] = response[node.ID * 6 + 5, 0]

                # Reaction
                if node.support_DX == True:
                    node.RxnFX[combo_name] = reaction[node.ID * 6 + 0, -1]
                else:
                    node.RxnFX[combo_name] = 0.0
                if node.support_DY == True:
                    node.RxnFY[combo_name] = reaction[node.ID * 6 + 1, -1]
                else:
                    node.RxnFY[combo_name] = 0.0
                if node.support_DZ == True:
                    node.RxnFZ[combo_name] = reaction[node.ID * 6 + 2, -1]
                else:
                    node.RxnFZ[combo_name] = 0.0
                if node.support_RX == True:
                    node.RxnMX[combo_name] = reaction[node.ID * 6 + 3, -1]
                else:
                    node.RxnMX[combo_name] = 0.0
                if node.support_RY == True:
                    node.RxnMY[combo_name] = reaction[node.ID * 6 + 4, -1]
                else:
                    node.RxnMY[combo_name] = 0.0
                if node.support_RZ == True:
                    node.RxnMZ[combo_name] = reaction[node.ID * 6 + 5, -1]
                else:
                    node.RxnMZ[combo_name] = 0.0

        else:
            # Reactions not available
            for node in model.Nodes.values():
                # Displacement, Velocity or Acceleration
                node.DX[combo_name] = response[node.ID * 6 + 0, 0]
                node.DY[combo_name] = response[node.ID * 6 + 1, 0]
                node.DZ[combo_name] = response[node.ID * 6 + 2, 0]
                node.RX[combo_name] = response[node.ID * 6 + 3, 0]
                node.RY[combo_name] = response[node.ID * 6 + 4, 0]
                node.RZ[combo_name] = response[node.ID * 6 + 5, 0]

    def simple_interpolater(self, target_x, X_vector, Y_matrix):

        """
        Interpolates the Y values corresponding to the target_x using linear interpolation.

        Parameters:
        - target_x (float): The x-value for which interpolation is performed.
        - X_vector (list or numpy array): The vector of x-values in ascending order.
        - Y_matrix (numpy array): The matrix of corresponding y-values, where each column represents a different set of y-values.

        Returns:
        - output_y (numpy array): The interpolated y-values corresponding to the target_x.
        """

        # Function performs linear interpolation based on the given parameters.
        # If target_x is equal to the first or last value in X_vector, the corresponding Y values are directly taken.
        # Otherwise, linear interpolation is applied between the two nearest x-values.

        # Note: Make sure X_vector is in ascending order for accurate results.
        # Note: The dimensions of Y_matrix should match the requirements for numpy array operations.

        if target_x == X_vector[0]:
            output_y = Y_matrix[:, 0]
        elif target_x == X_vector[-1]:
            output_y = Y_matrix[:, -1]
        else:
            # Find the index
            i = searchsorted(X_vector,target_x, side='right')

            a = X_vector[i-1]
            A = Y_matrix[:,i-1]
            b = X_vector[i]
            B = Y_matrix[:,i-1]

            output_y =(target_x-a)/(b-a) * (B-A) + A

        return output_y

class FRAResultsModelBuilder(ResultsModelBuilder):
    """
    Represents a results model builder for Frequency Response Analysis (FRA).

    This class is used to build a results model with data at a specified load frequency in FRA.

    Parameters:
        saved_model_path (str): The path to the saved finite element model file with results.

    Example:
        builder = FRAResultsModelBuilder("model_file.pkl")
        result_model = builder.get_model(freq=10.0, response_type="DR")  # Access the results model with FRA data at frequency 10.0 and real displacement response.
    """

    def __init__(self, saved_model_path):
        super().__init__(saved_model_path)

    def get_model(self,freq, response_type = 'DR'):
        """
        Get the finite element model with results at a specified load frequency in FRA.

        Args:
            freq (float): The load frequency at which results are requested within the FRA.
            response_type (str, optional): The type of response quantity requested ("DR" for real displacement, "VR" for real velocity, "AR" for real acceleration, "DI" for imaginary displacement, "VI" for imaginary velocity, or "AI" for imaginary acceleration). Default is "DR".

        Raises:
            ResultsNotFoundError: If no FRA results are available in the model.
            InputOutOfRangeError: If the requested frequency is outside the calculated frequency range of the FRA analysis.
            ValueError: If an invalid response_type is provided.

        Returns:
            FEModel3D: The finite element model with FRA results at the specified frequency.
        """
        model = self._solved_model

        # Check if results are available
        if model.DynamicSolution['Harmonic'] == False:
            raise ResultsNotFoundError

        # Check if the frequency is within the calculated range
        if freq<min(model.LoadFrequencies) or freq>max(model.LoadFrequencies):
            raise InputOutOfRangeError

        # Determine the type of response quantity requested for and get the total response
        if response_type == "DR":
            self._total_response = model.DISPLACEMENT_REAL()
            self._total_reaction = model.REACTIONS_REAL()
        elif response_type == "VR":
            self._total_response = model.VELOCITY_REAL()
            self._total_reaction = model.REACTIONS_REAL()
        elif response_type == "AR":
            self._total_response = model.ACCELERATION_REAL()
            self._total_reaction = model.REACTIONS_REAL()
        elif response_type == "DI":
            self._total_response = model.DISPLACEMENT_IMAGINARY()
            self._total_reaction = model.REACTIONS_IMAGINARY()
        elif response_type == "VI":
            self._total_response = model.VELOCITY_IMAGINARY()
            self._total_reaction = model.REACTIONS_IMAGINARY()
        elif response_type == "AI":
            self._total_response = model.ACCELERATION_IMAGINARY()
            self._total_reaction = model.REACTIONS_IMAGINARY()
        else:
            raise ValueError('Allowed response types are "DR", "VR", "AR", "D1", "V1", and "AI" but '+response_type+ ' was given.')

        dof = self._total_response.shape[0]
        response_at_freq = self.simple_interpolater(freq, model.LoadFrequencies, self._total_response).reshape(dof,1)
        reactions_at_freq = self.simple_interpolater(freq, model.LoadFrequencies, self._total_reaction).reshape(dof,1)

        # Now that the response is interpolated, put it into respective nodes
        self._save_response_into_node(
            model=model, response=response_at_freq,
            combo_name=model.FRA_combo_name,
            reaction=reactions_at_freq
        )

        # The model now contains results for the specified frequency, return it
        return model

class ModalResultsModelBuilder(ResultsModelBuilder):
    """
    Represents a results model builder for Modal Analysis.

    This class is used to build a results model with mode shapes for a specified mode.

    Args:
        saved_model (str): The path to the saved finite element model file with results.


    Raises:
        ResultsNotFoundError: If no modal analysis results are available in the model.
        InputOutOfRangeError: If the requested mode is outside the calculated modes.

    Example:
        builder = ModalResultsModelBuilder("model_file.pkl")
        result_model = builder.get_model(mode=3)  # Access the results model with mode shape data for mode 3.
    """

    def __init__(self, saved_model):
        super().__init__(saved_model)

    def get_model(self,mode):
        """
        Get the finite element model with mode shapes for a specified mode.

        Args:
            mode (int): The mode number for which mode shapes are requested.

        Raises:
            ResultsNotFoundError: If no modal analysis results are available in the model.
            InputOutOfRangeError: If the requested mode is outside the calculated modes.

        Returns:
            FEModel3D: The finite element model with mode shapes for the specified mode.
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

        # Return the model with requested for mode
        model.Active_Mode = mode
        return model

    def get_natural_frequency_for_mode(self, mode):
        """
        Get the natural frequency for a specified mode.

        Args:
            mode (int): The mode number for which the natural frequency is required.

        Raises:
            ResultsNotFoundError: If no modal analysis results are available in the model.
            InputOutOfRangeError: If the requested mode is outside the calculated modes.

        Returns:
            float: The natural frequency of the specified mode.
        """
        model = self._solved_model
        mode = int(mode)

        # Check if results are available
        if model.DynamicSolution['Modal'] == False:
            raise ResultsNotFoundError

        return model.NATURAL_FREQUENCIES()[mode - 1]

class THAResultsModelBuilder(ResultsModelBuilder):
    """
        Represents a results model builder for Time History Analysis (THA).

        This class is used to build a results model with data at a specified time instance within a time history analysis.

        Args:
            saved_model (str): The path to the saved finite element model file with results.

        Raises:
            ResultsNotFoundError: If no time history results are available in the model.
            InputOutOfRangeError: If the requested time is outside the calculated time range of the time history analysis.
            ValueError: If an invalid response_type is provided.

        Example:
            builder = THAResultsModelBuilder("model_file.pkl")
            result_model = builder.get_model(time=2.0, response_type="D")  # Access the results model with data at time instance 2.0 and displacement response.
        """

    def __init__(self, saved_model):
        super().__init__(saved_model)

    def get_model(self, time, response_type='D'):
        """
        Get the finite element model with data at a specified time instance within a time history analysis.

        Args:
            time (list, ndarray): The time instance for which results are requested within the time history analysis.
            response_type (str, optional): The type of response quantity requested ("D" for displacement, "V" for velocity, or "A" for acceleration). Default is "D".

        Raises:
            ResultsNotFoundError: If no time history results are available in the model.
            InputOutOfRangeError: If the requested time is outside the calculated time range of the time history analysis.
            ValueError: If an invalid response_type is provided.

        Returns:
            FEModel3D: The finite element model with time history analysis results at the specified time instance.
        """
        model = self._solved_model

        # Check if results are available
        if model.DynamicSolution['Time History'] == False:
            raise ResultsNotFoundError

        # Check if the time is within the calculated range
        if time < min(model.TIME_THA()) or time > max(model.TIME_THA()):
            raise InputOutOfRangeError('Time history results are only available for up to t = '+str(max(model.TIME_THA())))

        # Determine the type of response quantity requested for and get the total response
        if response_type == "D":
            self._total_response = model.DISPLACEMENT_THA()
        elif response_type == "V":
            self._total_response = model.VELOCITY_THA()
        elif response_type == "A":
            self._total_response = model.ACCELERATION_THA()
        else:
            raise ValueError(
                'Allowed response types are "D", "V", and "A" but ' + response_type + ' was given.')

        # Get the number of degrees of freedom for building the response vector at a specified time instance
        dof = self._total_response.shape[0]

        # Build the response at a specified time
        response_at_time = self.simple_interpolater(
            target_x=time,
            X_vector=model.TIME_THA(),
            Y_matrix=self._total_response
        ).reshape(dof,1)

        reactions_at_time= self.simple_interpolater(time, model.TIME_THA(), model.REACTIONS_THA()).reshape(dof,1)

        self._save_response_into_node(
            model=model,
            response=response_at_time,
            combo_name=model.THA_combo_name,
            reaction=reactions_at_time
        )

        # The model contains results at the specified time instance, return it
        return model

