# Self-driving car behavioural cloning

Additional images from left/right cameras included in training.

Lowered learning rate to get model to train properly.

Prediction of previous model against centre camera gives mean steering angle of 0.012
Against the left camera it gives mean angle of 0.045, right camera -0.033.

Validation loss = 0.0044 after 8 epochs with camera adjustment angle of 0.1.

Drives closer to the centre of the road, but gets stuck on the bridge.