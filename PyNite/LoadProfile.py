class LoadProfile():
    """
    A class representing a load profile for a specific load case.

    :param load_case_name: The name or identifier of the load case.
    :type load_case_name: str
    :param time: A list of time values associated with the load profile.
    :type time: list[float]
    :param profile: A list of load values corresponding to the time values.
    :type profile: list[float]
    """
    def __init__(self, load_case_name, time, profile):
        """
        Initializes a LoadProfile object with the provided parameters.

        :param load_case_name: The name or identifier of the load case.
        :type load_case_name: str
        :param time: A list of time values associated with the load profile.
        :type time: list[float]
        :param profile: A list of load values corresponding to the time values.
        :type profile: list[float]
        """
        self.load_case_name = load_case_name
        self.time = time
        self.profile = profile