## !/usr/local/bin/python3
#Created and debugged in Python 3.7.6
# using Tensorflow 1.15.2
# Requires: Numpy ,Cython, pyTsetlinMachine ,pyTsetlinMachineParallel
### Global Imports
import random as rand
import numpy as np
import tensorflow as tf
from tensorflow import keras

##Global constants
amount_training_images = 50000
amount_testing_images = 10000
image_resolution = "32x32"
cifar_label_dict = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
 }

storedImageDirectory = "./"

def fixDockerInstanceReset(typeFlag=3):
    import os
    os.system("pip install --upgrade pip")
    if typeFlag == 0:
        os.system("pip install pyTsetlinMachine")
    elif typeFlag == 1:
        print("Paralell")
        os.system("pip install pyTsetlinMachineParallel")
    elif typeFlag == 2:
        print("CUDA")
        os.system("pip install pycuda")
        os.system("pip install pyTsetlinMachineCUDA")
    os.system("pip install opencv-python")

def imageAsPyplot(image, saveFlag=0, imageName=""):
    if saveFlag:
        saveImageAsPyplot(image, imageName)
    else:
        process(image)

def process(image: str=None) -> None:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(image)

def saveImageAsPyplot(image, name):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.savefig(name)

def archtectureInfo():
    print("\t JupyterLab Info")
    print("Keras Version:\t\t", keras.__version__)
    print("Tensorflow Version:\t", tf.__version__)
    
def lossyGaussianFormatting(X_train, X_test):
    import cv2
    for i in range(X_train.shape[0]):
            for j in range(X_train.shape[3]):
                    X_train[i,:,:,j] = cv2.adaptiveThreshold(X_train[i,:,:,j], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) #cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

    for i in range(X_test.shape[0]):
            for j in range(X_test.shape[3]):
                    X_test[i,:,:,j] = cv2.adaptiveThreshold(X_test[i,:,:,j], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)#cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
                    #Changing the last number will change the threshold from mean, you can even set it to minus
    return X_train, X_test

def importCifarDataset():
    from tensorflow.keras.datasets import cifar10 #If you are using tensorflow
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data() # Load all data
    print("Example Color:",type(X_train[0]), X_train[0][0][0],X_train[0][0][1],X_train[0][0][2])
    return X_train, Y_train, X_test, Y_test


def showLossyGaussianTechniques(imageSet, X_train, X_test=[], saveFlag=0):
    X_train, X_test = lossyGaussianFormatting(X_train, X_test) #Using known working method, with 50 images to test
    print("Example Binary:",type(X_train[0]), X_train[0][0][0],X_train[0][0][1],X_train[0][0][2])
    imageAsPyplot(X_train[imageSet], saveFlag, "Gaussian-Composite_"+str(imageSet))
    for xIndex, x in enumerate(splitImageByColour(X_train[imageSet])): imageAsPyplot(x, saveFlag, "Gaussian-%c_" % ("R" if xIndex == 0 else "G" if xIndex == 1 else "B")+str(imageSet))
    
def losslessBinarisation(colorImages, pictureWidth=32, pictureHeight=32, colorChannels=3, colorDepth=8):
    binaryImageList = np.empty([colorChannels, colorDepth ,pictureWidth, pictureHeight]) #np.empty([3, 8, x, 32, 32])
    for rIndex, row in enumerate(colorImages):
        for hIndex, height in enumerate(row): # This is technically also column(s)
            for cIndex, color in enumerate(height):
                for bit in range(colorDepth): #If it is important o have the bits in the right order, change this to "range(colorDepth-1, -1, -1)"
                    if((color & 2**bit) == 2**bit):
                        binaryImageList[cIndex][bit][rIndex][hIndex] = 1
                    else:
                        binaryImageList[cIndex][bit][rIndex][hIndex] = 0
    return binaryImageList #Take note that the binary numbers are reversed, i.e x[0][0] gives the red 2^0 bit while x[1][4] gives the blue 2^4 bit
    
def showLosslessBinarization(image, imageSet, saveFlag=0):
    imageList = losslessBinarisation(image)
    print("Lossless Shape:",imageList.shape)
    for iIndex, i in enumerate(imageList):
        for yIndex, y in enumerate(i):
            imageAsPyplot(y, 
                        saveFlag, 
                        "Lossless-"+("%c"+"2^"+str(yIndex)) % ("R" if iIndex == 0 else "G" if iIndex == 1 else "B")+"_"+str(imageSet))

def splitImageByColour(image):
    imagesSplitByColour = np.empty([len(image[0][0]), len(image), len(image[0])]) #np.empty([3, 32, 32])
    for rIndex, row in enumerate(image):
        for pIndex, pixel in enumerate(row):
            for cIndex, colour in enumerate(pixel):
                imagesSplitByColour[cIndex][rIndex][pIndex] = colour
    return imagesSplitByColour

def testShowMultipleGraphsOrImages():
    '''
    Creates a grid with  cordinates, image/plot and name.
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    width=32
    height=32
    rows = 3
    cols = 2
    axes=[]
    fig=plt.figure()

    for a in range(rows*cols):
        b = np.random.randint(7, size=(height,width))
        axes.append( fig.add_subplot(rows, cols, a+1) )
        subplot_title=("Subplot"+str(a))
        axes[-1].set_title(subplot_title)  
        plt.imshow(b)
    fig.tight_layout()    
    plt.show()

def createMockComparisonImage(imageSet, X_train, X_test, multiColor=1):
    #Variables
    imagePerRow = 4
    spacingBetweenRows = 1 #Currently does not support 0 spacing
    imagePerColumn = 10
    spacingBetweenColumns = 1 #Currently does not support 0 spacing
    imageWidth = 32
    imageHeight = 32
    colours = 3
    spacingColor = [255, 255, 255] #Default = [255, 255, 255] = white

    #baseImage = np.copy(X_train[imageSet])
    
    ''''''#Method for inserting a specific image
    colourComparisonRows = [   [[255, 255 - int(x*(255/32)), 255 - int(x*(255/32))] for x in range(32)],
                    [[255 - int(x*(255/32)), 255, 255 - int(x*(255/32))] for x in range(32)],
                    [[255 - int(x*(255/32)), 255 - int(x*(255/32)), 255] for x in range(32)],
                    [[255 - int(x*(255/32)), 255 - int(x*(255/32)), 255 - int(x*(255/32))] for x in range(32)]
    ]
    baseImage = []
    for x in colourComparisonRows:
        for y in range(8):
            baseImage.append(x)
    X_train[imageSet] = baseImage
    
    groundTruth = []
    for x in splitImageByColour(baseImage):
        groundTruth.append(x)
    groundTruth.append(baseImage)
    #print("GT",type(groundTruth))
    lossyGauss = []
    X_train, X_test = lossyGaussianFormatting(X_train, X_test)
    for x in splitImageByColour(X_train[imageSet]):
        lossyGauss.append(x)
    lossyGauss.append(X_train[imageSet])

    losslessPerBit = np.flip(losslessBinarisation(baseImage, pictureWidth=len(baseImage), pictureHeight=len(baseImage[0])), 1) #(3, 8, 32, 32)
    #print("LL",type(losslessPerBit))
    #print(losslessPerBit[0][0][0])

    heightSpacerHelp = 0
    widthSpacerHelp = 0
    
    colourHelp = 0 #Help variable used to determine color channel to be used for the composite image in pyplot
    if not multiColor: colourHelp+=1
    colourHelp2 = 0 #Help variable used to extract pictures from lossless binarization

    #Constants
    rowLength = imagePerRow * (imageHeight + spacingBetweenRows) + spacingBetweenRows
    columnLength = imagePerColumn * (imageWidth + spacingBetweenColumns) + spacingBetweenColumns
    compositeImage = np.zeros((rowLength, columnLength, colours))
    #print(rowLength,columnLength,np.shape(compositeImage))

    for rIndex, row in enumerate(compositeImage): #Height
        widthSpacerHelp=0
        for wIndex, column in enumerate(row): #Width
            for cIndex, _ in enumerate(column): #Colours
                imageWidthIndex = int((wIndex-widthSpacerHelp) % imageWidth)
                imageHeightIndex = int((rIndex-heightSpacerHelp) % imageHeight)
                if  (wIndex % (imageWidth+spacingBetweenColumns) >= 0 and wIndex % (imageWidth+spacingBetweenColumns) <= spacingBetweenColumns-1) or (rIndex % (imageHeight+spacingBetweenRows) >= 0 and rIndex % (imageHeight+spacingBetweenRows) <= spacingBetweenColumns-1): #Borders
                    compositeImage[rIndex][wIndex][cIndex] = (spacingColor[cIndex] - 0) / (255 - 0)
                elif wIndex < (imageWidth+spacingBetweenColumns): #Groundtruth images
                    #print("Groundtruth")
                    if len(groundTruth) == 1: #Spaghetti solution to account for different dimensions of elements
                        compositeImage[rIndex][wIndex][cIndex] = (groundTruth[0][imageHeightIndex][imageWidthIndex][cIndex] - 0) / (255 - 0)
                    else:
                        compositeImage[rIndex][wIndex][colourHelp] = (groundTruth[0][imageHeightIndex][imageWidthIndex] - 0) / (255 - 0)
                elif wIndex < 2*(imageWidth+spacingBetweenColumns): #Lossy Gauss images
                    #print("Lossy Gauss")
                    if len(lossyGauss) == 1: #Spaghetti solution to account for different dimensions of elements
                        compositeImage[rIndex][wIndex][cIndex] = (lossyGauss[0][imageHeightIndex][imageWidthIndex][cIndex] - 0) / (255 - 0)
                    else:
                        compositeImage[rIndex][wIndex][colourHelp] = (lossyGauss[0][imageHeightIndex][imageWidthIndex] - 0) / (255 - 0)
                elif rIndex > 3*(imageWidth+spacingBetweenColumns): #Empty brackets with no information
                    compositeImage[rIndex][wIndex][cIndex] = (spacingColor[cIndex] - 0) / (255 - 0)
                elif wIndex > 2*(imageWidth+spacingBetweenColumns): #Lossless images
                    #print("Lossless")
                    helpImageIndex = int(widthSpacerHelp/spacingBetweenColumns)-3
                    compositeImage[rIndex][wIndex][colourHelp] = losslessPerBit[colourHelp2][helpImageIndex][imageHeightIndex][imageWidthIndex]
            if wIndex % (imageWidth+spacingBetweenColumns) == 0: #wIndex = 32 * (10+1) -> wIndex % 33 -> widthSpaceHelp<0-10>
                widthSpacerHelp+=spacingBetweenColumns
        if rIndex % (imageHeight+spacingBetweenRows) == 0: #rIndex = 32 * (4+1) -> rIndex % 33 -> heightSpaceHelp<0-4>
            heightSpacerHelp+=spacingBetweenRows
            if heightSpacerHelp > spacingBetweenRows: #When done creating the initial border
                if multiColor: colourHelp+=1
                colourHelp2 += 1
                groundTruth.pop(0)
                lossyGauss.pop(0)
    
    import matplotlib.pyplot as plt
    plt.imshow(compositeImage)
    plt.axis('off')
    plt.show()
    
def showEverything(imageSet, saveFlag, multiColor=1):
    import os
    import random as rand
    archtectureInfo()
    #fixDockerInstanceReset() #Only needs to be run on jupyterlab
    ####Setup with manualally installed Cifar-10
    #fileList = [i for i in os.listdir(cifarPackedLocation) if "data" in i]
    #dataPacket = importCifarDataModule(cifarPackedLocation+fileList[rand.randint(0,len(fileList)-1)])
    #generalInfoOnCifarDataset(dataPacket)
    #exampleImageFromTheDataset(dataPacket)
    ####Setup with Cifar-10 imported from Keras
    
    import os
    #os.chdir(storedImageDirectory)
    #os.mkdir(str(imageSet))
    #os.chdir(storedImageDirectory+"/"+str(imageSet))
    X_train, Y_train, X_test, _ = importCifarDataset() #Returns X_train, Y_train, X_test, Y_test
    print("Groundtruth Label:",Y_train[imageSet][0]," - Human Readable:",cifar_label_dict[Y_train[imageSet][0]]," - Index:",imageSet)
    '''
    carryImage = np.copy(X_train[imageSet]) #As Python is pointer based and we will edit the original, create a copy
    imageAsPyplot(X_train[imageSet], saveFlag, "Groundtruth_"+str(imageSet))
    for xIndex, x in enumerate(splitImageByColour(X_train[imageSet])): imageAsPyplot(x, saveFlag, "Groundtruth-%c_" % ("R" if xIndex == 0 else "G" if xIndex == 1 else "B")+str(imageSet))
    showLossyGaussianTechniques(imageSet, X_train=X_train, X_test=X_test, saveFlag=saveFlag)
    showLosslessBinarization(carryImage, imageSet, saveFlag)
    '''
    createMockComparisonImage(imageSet,X_train, X_test,multiColor=multiColor)
    #import matplotlib.pyplot as plt
    #plt.show()





#showEverything(rand.randint(0, 49999), 1)
#showEverything(23099, 1)
#showEverything(1123, 1)
showEverything(1, 1, 0)