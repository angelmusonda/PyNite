class ResultsNotFoundError (Exception):
    def __init__(self, message="Results are not available for this type of analysis"):
        self.message = message
        super().__init__(self.message)

class InputOutOfRangeError (Exception):
    def __init__(self, message="Provided input is out of range"):
        self.message = message
        super().__init__(self.message)

class ParameterIncompatibilityError (Exception):
    def __init__(self, message="Input parameters are incompatible"):
        self.message = message
        super().__init__(self.message)

class DefinitionNotFoundError (Exception):
    def __init__(self, message="Definition not found"):
        self.message = message
        super().__init__(self.message)

class DampingOptionsKeyWordError (Exception):
    def __init__(self, message= "Allowed damping options keywords are: 'constant_modal_damping', 'r_alpha', 'r_beta', 'first_mode_damping', 'highest_mode_damping' and 'damping_in_every_mode'"):
        self.message = message
        super().__init__(self.message)

class DynamicLoadNotDefinedError (Exception):
    def __init__(self, message="Provide the name of the dynamic load combination or at least one seismic ground acceleration"):
        self.message = message
        super().__init__(self.message)