class ResultsNotFoundError (Exception):
    def __init__(self, message="Results are not available for this type of analysis"):
        self.message = message
        super().__init__(self.message)

class InputOutOfRangeError (Exception):
    def __init__(self, message="Provided input is out of range"):
        self.message = message
        super().__init__(self.message)