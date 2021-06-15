""" Single source of truth for all sorts of training parameters"""


class TrainConfiguration:
    """ Class for storing training parameters"""
    def __init__(
        self,
        val_split: float = 0.2,
        test_split: float = 0.1
    ):
        self.val_split = val_split
        self.test_split = test_split

    def public_method(self):
        """public method 1"""

    def public_method_2(self):
        """public method 2"""
