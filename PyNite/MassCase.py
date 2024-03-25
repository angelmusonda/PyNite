class MassCase():
    """
    A class that represents a mass case for structural analysis.

    A mass case contains information necessary to define how a specific load case should be treated as a mass for analysis purposes.

    Parameters:
    -----------
    :param name: str
        The name of the load case to use as a mass case.

    :param gravity: float, optional
        The acceleration due to gravity to use in converting the force to mass. Default value is 9.81 m/s² for Earth.

    :param factor: float, optional
        The percentage at which the load contributes to the mass. This factor should be specified as a decimal (e.g., 0.5 for 50%).
        Some loads, such as live loads, may only contribute a small percentage to the total mass as specified in the design code. Default value is 1, representing the full load.

    """
    def __init__(self, name, gravity=9.81, factor=1):
        self.name = name
        self.gravity = gravity
        self.factor = factor
