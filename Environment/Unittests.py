import unittest
import numpy as np 
from Utility import Utility

class TestClass(unittest.TestCase):

    def test_log2_standardization_factor_1(self):
        board = [ [2,4,8,16], [32,64,128,256],[512,1024,2048,4096],[0,0,0,0]]

        result_board = Utility.process_state_using_log2_and_factor(board, 1)

        solution_board = np.array([ [1,2,3,4],[5,6,7,8],[9,10,11,12],[0,0,0,0]])

        self.assertTrue( np.array_equal(result_board, solution_board) )

    def test_log2_standardization_factor_12(self):
        board = [ [2,4,8,16], [32,64,128,256],[512,1024,2048,4096],[0,0,0,0]]

        result_board = Utility.process_state_using_log2_and_factor(board, 12)

        solution_board = np.array([ [1 / 12, 0.16666667, 0.25, 0.33333333],
                           [0.41666667, 0.5, 0.58333333, 0.66666667],
                           [0.75, 0.83333333, 0.91666667, 1.],
                           [0,0,0,0]], dtype = np.float32)

        self.assertTrue( np.array_equal(result_board, solution_board) )

if __name__ == '__main__':
    unittest.main(verbosity = 2)