import numpy as np
from motion import *

class vehicle():
    def __init__(self) -> None:
        self.antenna = 1 # number of antennas
        self.position = []
        self.orientation = []
        self.velocity = []
    