import sys
from tensorflow.keras.optimizers import Adam

from homography_estimator.Siamese import Siamese

if __name__ == '__main__':
	# model_path = utils.get_homography_estimator_model_path()

	img_shape = (180, 320, 1)
	learning_rate = .01
	batch_size = 64
	num_epoch = 10

	optimizer = Adam(lr=learning_rate, weight_decay=0.000001)

	siamese = Siamese(input_shape=img_shape)

	siamese.model.compile(loss='contrastive_loss', optimizer=optimizer, metrics=['accuracy'])



	siamese.model.fit(train_generator,
				   steps_per_epoch=train_steps,
				   epochs=num_epoch,
				   validation_data=test_generator,
				   validation_steps=test_steps
				   )
	sys.exit()
