import joblib
import pygame
import numpy as np
import time
import tensorflow as tf
import os

from tensorflow import keras

from PIL import Image

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

# Load the saved model generated with the Jupyter notebook (MNIST_DigitClassifier_v2.ipynb)
digit_clf = tf.keras.models.load_model("KNN_MNIST_ImageClassifier_v4")
# print (digit_clf.summary())

# Guess the number given the individual digit data
def guessNum (data):
    prediction = ''
    for x in data:

        # Guess what the number is
        # print (x.shape)
        pred = np.argmax(digit_clf.predict(x))

        # Append the prediction to the string
        prediction+=str(pred)
    return prediction

# Center the individual images (the individual digits), add appropriate amount of pixels to the left and right of the "slice"
# of the image
def centerImage (image):

    # Get the x dimension of the sliced digit image
    size = len(image)

    # To make the image more square which will make it closer to the training data, fill the image to the right and left
    fill = (height - size) // 2

    centerArr = np.zeros([fill, height], dtype = int)

    # Image is now the original slice plus white space to the left and right (to make it square)
    image = np.concatenate([centerArr, image, centerArr], axis = 0).tolist()
    return image

# Since while processing the individual slices/digits, we appended the columns of the image rather than the rows,
# the image is vertically flipped and rotated 90 degrees counter clockwise, so we must unflip and rotate it back.
# Also, we need to resize the image to 28x28 so we can feed it to our model
def processSplit (images):
    imageData = []
    for ix in range (0, len(images)):

        # Center/squarify the image and store it in a numpy array for easy manipulation
        x = np.array(centerImage(images[ix]))

        # Flip the sliced image vertically
        x = np.flipud(x)

        # Rotate the image 90 degrees clockwise
        x = np.rot90(x, 1, (1, 0))

        image = Image.fromarray(x)
        # image.show()

        # Resize the image
        image = image.resize((28,28))

        imageData.append(np.array(image).reshape(-1,28,28,1))
    return imageData

# Finds the sections (by pixel columns) of the screen that contain drawn pixels and splits them into parts.
# I.e. if there is a 3 and a 4 drawn by the user, it will split the parts of the screen containing the 3 and the 4.
def splitDigits (originalImage):
    images = []
    imageSplit = []


    for ix in range (0, originalImage.shape[1] - 1):
        col = originalImage[ :, ix].copy()
        nextCol = originalImage [:, ix + 1].copy()

        # If the column contains non-black pixels, it contains a number, so append it to imageSplit (which represents
        # each individual digit
        if (col.sum() != 0):
            imageSplit.append(col)

            # If the column after the current one is black (contains all black pixels) or we've reached the edge of the screen, 
            # then we've reached the end of the current digit slice, so add it to the images array.
            if (nextCol.sum() == 0 or ix == originalImage.shape[1]-2):
                images.append(np.array(imageSplit))

                # Clear the image slice after appending it to images
                imageSplit.clear()

    # Process these image slices, make them square and resize them to 28x28, which we will feed into our model             
    images = processSplit(images)
    return images

# Main loop for the graphics for the number drawing
def main():
    running = True
    while (running):
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # If the user presses ENTER/RETURN, then they want model to guess the number
            if (keys[pygame.K_RETURN] == True):
                    start_time = time.time()
                    # Save a screenshot of the current screen
                    pygame.image.save(screen, "images/number.png")
                    image = Image.open("images/number.png")
                    # image.save(r"C:\Users\408aa\Desktop\Python\DataScience\MNIST_DigitClassifier\number.png")

                    # Save array representation of whole screen
                    image_arr = np.asarray(image).reshape((height*height), 3)

                    # Image is saved in RGB format, save it to a numpy array representing intensity of each pixel
                    data = np.zeros(shape = ((height*height), 1))
                    for x in range (0, height*height):
                        data[x] = np.mean(image_arr[x])
                    
                    # Reshape the array to 2d array representing the screen
                    data = data.reshape((height,height))

                    # Split the individual digits on the screen
                    digits = splitDigits (data)

                    # Feed this array of individual digits to the model for predictions
                    prediction = guessNum(digits)
                    print ("Predicted Number: " + str(prediction))
                    end_time = time.time()
                    print ("Ran in " + str(end_time-start_time))
            # Draw on the screen if the mouse 1 button (left click) is pressed
            if (pygame.mouse.get_pressed()[0] == True ):
                mouse_pos = pygame.mouse.get_pos()
                pygame.draw.circle(screen, WHITE, mouse_pos, penWidth)
            
            # Erase (make the screen black) if the mouse 2 button (right click) is pressed.
            if (pygame.mouse.get_pressed()[2] == True):
                screen.fill(BLACK)
        pygame.display.update()


WHITE = (255,255,255)
BLACK = (0,0,0)

# Height of the display screen
height = 350
screen = pygame.display.set_mode((height,height))
pygame.display.set_caption("Draw Number")

# Width of the stylist/pencil that the user uses to draw on the screen (tried to make the ratio of screen size
# to pencil/stylist width as close to the original MNIST data as possible to produce the most accurate predictions
# generally, the thinner the pencil width, the worse the predictions.
penRatio = 27
penWidth = height // penRatio
main()
pygame.quit()

