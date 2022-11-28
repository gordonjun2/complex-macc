import numpy as np

real_img = np.load("./icf-jag-10k/jag10K_images.npy")
complex_img = real_img.astype(np.complex64)

np.save("./icf-jag-10k/jag10K_images_complex.npy", complex_img)