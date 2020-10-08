import unittest
import numpy as np
 
from utilities import PreprocessingUtility,DataUtility,RewardUtility
from history import History

class TestClass(unittest.TestCase):

    def test_log2_standardization_factor_1(self):
        board = [ [2,4,8,16], [32,64,128,256],[512,1024,2048,4096],[0,0,0,0]]

        result_board = PreprocessingUtility.process_state_using_log2_and_factor(board, 1)

        solution_board = np.array([ [1,2,3,4],[5,6,7,8],[9,10,11,12],[0,0,0,0]])

        self.assertTrue( np.array_equal(result_board, solution_board) )

    def test_log2_standardization_factor_12(self):
        board = [ [2,4,8,16], [32,64,128,256],[512,1024,2048,4096],[0,0,0,0]]

        result_board = PreprocessingUtility.process_state_using_log2_and_factor(board, 12)

        solution_board = np.array([ [1 / 12, 0.16666667, 0.25, 0.33333333],
                           [0.41666667, 0.5, 0.58333333, 0.66666667],
                           [0.75, 0.83333333, 0.91666667, 1.],
                           [0,0,0,0]], dtype = np.float32)

        self.assertTrue( np.array_equal(result_board, solution_board) )

    def test_min_max_normalization(self):
        board = [ [2,4,8,16], [32,64,128,256],[512,1024,2048,4096],[0,0,0,0]]

        result_board = PreprocessingUtility.min_max_normalize_state(board)

        solution_board = np.array([[4.8828125e-04, 9.7656250e-04, 1.9531250e-03, 3.9062500e-03],
                                   [7.8125000e-03, 1.5625000e-02, 3.1250000e-02, 6.2500000e-02],
                                   [1.2500000e-01, 2.5000000e-01, 5.0000000e-01, 1.0000000e+00],
                                   [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00]],
                                   dtype = np.float32)

        self.assertTrue( np.array_equal(result_board, solution_board) )

    def test_standardization(self):
        board = [ [2,4,8,16], [32,64,128,256],[512,1024,2048,4096],[0,0,0,0]]

        result_board = PreprocessingUtility.standardize_state(board)

        solution_board = np.array([[-4.4879995e-04, -4.4703952e-04, -4.4351866e-04, -4.3647693e-04],
                                    [-4.2239347e-04, -3.9422658e-04, -3.3789276e-04, -2.2522517e-04],
                                    [ 1.1002695e-07,  4.5078044e-04,  1.3521212e-03,  3.1548028e-03],
                                    [-4.5056039e-04, -4.5056039e-04, -4.5056039e-04, -4.5056039e-04]],
                                    dtype = np.float32)

        self.assertTrue(np.array_equal(result_board, solution_board))

    def test_moving_average(self):
        data = [i for i in range(0,1000,10)]
        moving_average = DataUtility.calculate_moving_average_for(data)

        solution = np.array([i for i in range(45,955,10)], dtype = np.float32)
        self.assertTrue(np.array_equal(moving_average, solution))

    def test_cell_distribution(self):
        history = History()
        history.max_cell       = [128, 16, 4, 8, 32, 64, 2]
        history.max_cell_count = [4,   2,  2, 2, 2,  6,  2]

        cell_dictionary = DataUtility.build_and_sort_max_cell_distribution_from_history(history)

        solution = {'2':2, '4':2, '8':2, '16':2, '32':2, '64':6, '128':4}

        self.assertEqual(cell_dictionary, solution)

    def test_get_max_cell_value_and_count_from_board(self):
        board = [ [2,4,4096,16], [32,64,128,256],[512,1024,2048,4096],[4096,512,512,4]]

        max_cell_solution = 4096
        max_cell_count_solution = 3

        max_cell, max_cell_count = DataUtility.get_max_cell_value_and_count_from_board(board)

        self.assertTrue((max_cell == max_cell_solution and max_cell_count == max_cell_count_solution))

    def test_get_reward_from_dictionary(self):
        dictionary = {2:2, 4:2, 8:2, 16:2, 32:2, 64:6, 128:4}

        reward = RewardUtility.get_reward_from_dictionary(dictionary)

        correct_reward = 2 * 2 + 4 * 2 + 8 * 2 + 16 * 2 + 32 * 2 + 64 * 6 + 128 * 4

        self.assertEqual(reward, correct_reward)

    def test_get_reward_by_distance_to_2048(self):
        dictionary = {2:2, 4:2, 8:2, 16:2, 32:2, 64:6, 128:4}

        reward = RewardUtility.get_reward_by_distance_to_2048(dictionary)

        correct_reward = 128 - 2048

        self.assertEqual(reward, correct_reward)

    def test_high_cell_high_reward(self):
        dictionary = {2:2, 4:2, 8:2, 16:2, 32:2, 64:6, 128:4}

        reward = RewardUtility.get_high_cell_high_reward(dictionary)

        correct_reward = 128

        self.assertEqual(reward, correct_reward)

if __name__ == '__main__':
    unittest.main(verbosity = 2)