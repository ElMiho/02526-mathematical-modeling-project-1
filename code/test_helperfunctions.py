import unittest

import numpy as np
import exercise_2

class TestExercise2(unittest.TestCase):
    def __init__(self, args):
        super().__init__(args)
        self.images = np.array([
            [
                [-1, -2, -3], 
                [-4, -5, -6]
            ], 
            [
                [7, 8, 9],
                [10, 11, 12]
            ],
            [
                [13, 14, 15],
                [16, 17, 18]
            ],
            [
                [19, 20, 21],
                [22, 23, 24]
            ]
        ])
        self.flattened_images = np.reshape(self.images, (-1,))

    def test_Vx_calc(self):
        Vx_expected = np.array([
            [
                [-1, -1, 0], 
                [-1, -1, 0]
            ], 
            [
                [1, 1, 0],
                [1, 1, 0]
            ],
            [
                [1, 1, 0],
                [1, 1, 0]
            ],
            [
                [1, 1, 0],
                [1, 1, 0]
            ]
        ])

        depth, height, width = self.images.shape
        # Make sure that flattened image has expected dimensions
        self.assertEqual(len(self.flattened_images), depth*height*width)

        # Perform gradient calculation
        Vx_calculated_LL = exercise_2.calculate_Vx_LL(self.flattened_images, depth*height*width, width)
        Vx_calculated_LL = np.reshape(Vx_calculated_LL, (depth, height, width))

        # Make sure that the result is as expected
        are_equal = np.array_equiv(Vx_expected, Vx_calculated_LL)
        self.assertTrue(are_equal)

    def test_Vy_calc(self):
        Vy_expected = np.array([
            [
                [-3, -3, -3], 
                [0, 0, 0]
            ], 
            [
                [3, 3, 3],
                [0, 0, 0]
            ],
            [
                [3, 3, 3],
                [0, 0, 0]
            ],
            [
                [3, 3, 3],
                [0, 0, 0]
            ]
        ])

        depth, height, width = self.images.shape

        # Make sure that flattened image has expected dimensions
        self.assertEqual(len(self.flattened_images), depth*height*width)

        # Perform gradient calculation
        Vy_calculated = exercise_2.calculate_Vy_LL(self.flattened_images, depth*height*width, width, height)
        Vy_calculated = np.reshape(Vy_calculated, (depth, height, width))

        # Make sure that the result is as expected
        are_equal = np.array_equiv(Vy_expected, Vy_calculated)
        self.assertTrue(are_equal)

    def test_Vt_calc(self):
        Vt_expected = np.array([
            [
                [8, 10, 12], 
                [14, 16, 18]
            ], 
            [
                [6, 6, 6],
                [6, 6, 6]
            ],
            [
                [6, 6, 6],
                [6, 6, 6]
            ],
            [
                [0, 0, 0],
                [0, 0, 0]
            ]
        ])

        depth, height, width = self.images.shape
        # Make sure that flattened image has expected dimensions
        self.assertEqual(len(self.flattened_images), depth*height*width)

        # Perform gradient calculation
        Vt_calculated = exercise_2.calculate_Vt_LL(self.flattened_images, width, height)
        Vt_calculated = np.reshape(Vt_calculated, (depth, height, width))

        # Make sure that the result is as expected
        are_equal = np.array_equiv(Vt_expected, Vt_calculated)
        self.assertTrue(are_equal)






if __name__ == "__main__":
    unittest.main()