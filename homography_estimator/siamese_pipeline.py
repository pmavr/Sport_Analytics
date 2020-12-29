import numpy as np
import matplotlib.pyplot as plt

from homography_estimator.siamese import define_branch


if __name__ == '__main__':
	# model_path = utils.get_homography_estimator_model_path()

	img_shape = (180, 320, 1)

	branch = define_branch(img_shape)

	branch.summary()

