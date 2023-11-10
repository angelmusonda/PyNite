class DisplacementProfile():
    """
    A class representing a displacement profile for a specific node and direction.

    :param node: The node or point at which the displacement is measured.
    :type node: str
    :param direction: The direction of the displacement ('DX', 'DY', 'DZ', 'RX', 'RY', 'RZ').
    :type direction: str
    :param time: A list of time values associated with the displacement profile.
    :type time: list[float]
    :param profile: A list of displacement values corresponding to the time values.
    :type profile: list[float]
    """
    def __init__(self, node,direction,time, profile):
        """
        Initializes a DisplacementProfile object with the provided parameters.

        :param node: The node or point at which the displacement is measured.
        :type node: str
        :param direction: The direction of the displacement ('DX', 'DY', 'DZ', 'RX', 'RY', 'RZ').
        :type direction: str
        :param time: A list of time values associated with the displacement profile.
        :type time: list[float]
        :param profile: A list of displacement values corresponding to the time values.
        :type profile: list[float]
        """
        self.node = node
        self.direction = direction
        self.time = time
        self.profile = profile