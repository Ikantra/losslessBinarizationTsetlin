#!/usr/local/bin/python3
#Created and debugged in Python 3.7.6
# using Tensorflow 1.15.2
# Requires: numpy ,Cython, pycuda, opencv-python, tqdm, pyTsetlinMachine ,pyTsetlinMachineParallel, pyTsetlinMachineCUDA
### Global Imports
import random as rand
import numpy as np
from tqdm import tqdm
##  AI/ML

##Constants
versionUsed = {0: "gaussian", 1: "lossless"}
tsetlinArrayDimensions = {1: [1, 1], 2: [3, 8]} #Note only the [1, 1] works for the compressed code
#tsetlinTypes = {0: "base", 1: "paralell",2 : "cuda"} #Reversed dictionary order making input easier to write, but harder to human-read
tsetlinTypes = {"base": 0, "paralell" : 1,"cuda" : 2}

##Variables
#Tsetlin Type
tsetlinType = tsetlinTypes["paralell"]
#Setup
#images, clauses, T, s, patch = 500, 800*8, 75, 10.0, [8, 8] #Some default values
imagesToTrainOn = 5000 #Total pre augment for train is 50000, test is 10000, 50000+ gives all
epochs = 300 #There might be a memory leak on ctm.fit, so avoid very high number of epochs (>350)
#Tsetlin Hyper-Parameters
clauses = 800 #Note this should be multiplied by 8 for the new lossless color format when doing comparisons, as it takes in 8 times the information
T = 75
s = 10.0
patch_Dim = (8, 8)
#Misc
showDebugInfo = 0 # Mostly used for debugging of the code, just set it to 1 if you want to see a lot of extra information
inputArrayDimension = tsetlinArrayDimensions[1] #This is only used for the LL4 format, ignore it otherwise
informationProcessingType = 1 # 0 = lossy gaussian // 1 = new lossless

def printTimeStamp():
    '''This takes in the current time, and returns it, used mostly for timing of runs.'''
    import datetime
    from time import time
    timestamp = time() 
    value = datetime.datetime.fromtimestamp(timestamp)
    return(value.strftime('%H:%M:%S - %d.%m.%Y'))
    
def fixDockerInstanceReset(typeFlag):
    '''
    This is mostly used for jupyterlab as the docker instance might reset to a state without certain libraries.
    The typeflag just determines if its 0: Standard, 1: Paralell or 2: CUDA version of Tsetlin.
    '''
    import os
    os.system("pip install --upgrade pip")
    os.system("pip install opencv-python")
    if typeFlag == 0:
        os.system("pip install pyTsetlinMachine")
    elif typeFlag == 1:
        print("Paralell")
        os.system("pip install pyTsetlinMachineParallel")
    elif typeFlag == 2:
        print("CUDA")
        os.system("pip install pycuda")
        os.system("pip install pyTsetlinMachineCUDA")
    
def useTheTsetlinFormatThatWork(imageSet, typeFlag=0, epochs=50, clauses=800, T=75, s=10.0, patch_Dim=(8, 8), debugInfo=0):
    '''To make troubleshooting easier, a known working version of AGauss were provided some modifications needed to be done.
    They are constrained outside the inner function with the exception of the mean of mean change.'''
    fixDockerInstanceReset(typeFlag=typeFlag) #This is a workaround for jupyterLab so you could probably comment out this
    import numpy as np
    from time import time
    
    def lossyGaussianFormatting(X_train, X_test):
        '''
        The known working method of the Gaussian Thresholding.
        '''
        import cv2
        for i in range(X_train.shape[0]):
                for j in range(X_train.shape[3]):
                        X_train[i,:,:,j] = cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) #cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

        for i in range(X_test.shape[0]):
                for j in range(X_test.shape[3]):
                        X_test[i,:,:,j] = cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)#cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
        return X_train, X_test
    
    if typeFlag == 0:
        ##Standard
        from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
        from pyTsetlinMachine.tm import MultiClassTsetlinMachine
    elif typeFlag == 1:
        ##Paralell
        from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D
        from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
    elif typeFlag == 2:
        ##CUDA
        from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
        from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
    if debugInfo:
        print("Tsetlin loading correctly!")
    
    from tensorflow.keras.datasets import cifar10 #If you are using tensorflow
    #from keras.datasets import cifar10 #If you are using keras as a standalone module
    print("Dataset loaded correctly")
    
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data() # Load all data
    Y_train=Y_train.reshape(Y_train.shape[0])[:imageSet]
    Y_test=Y_test.reshape(Y_test.shape[0])[:imageSet]
    X_train, X_test = lossyGaussianFormatting(X_train[:imageSet], X_test[:imageSet]) #Using known working method, with 50 images to test
    print("Type X_train:",type(X_train[0][0][0][0]),"Type Y_train",type(Y_train[0])) #Both are "numpy.uint8"
    print("Post Processing works! New shapes -","X_train:",X_train.shape,"X_test:",X_test.shape,"Y_train:",Y_train.shape,"Y_test",Y_test.shape)
    
    #MultiClassConvolutionalTsetlinMachine2D(self, number_of_clauses, T, s, patch_dim, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, weighted_clauses=False, s_range=False)
    ctm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, patch_Dim) #Clauses, T, s, (mask , mask) ?

    results = np.zeros(0)
    lastEpochResults = np.zeros(0)
    for i in range(epochs): #There might be a memory leak on ctm.fit, so avoid high number of epochs
        start = time()
        ctm.fit(X_train, Y_train, epochs=1, incremental=True)
        stop = time()
        results = np.append(results, 100*(ctm.predict(X_test) == Y_test).mean())
        #results = np.append(results, 100*(ctm.predict(X_test) == Y_test) #This line was changed for the one above, to account for different output formats.
        if i == epochs-1: lastEpochResults = np.append(results, 100*(ctm.predict(X_test) == Y_test))
        print("#%d Mean Accuracy (%%): %.2f; Std.dev.: %.2f; Training Time: %.1f ms/epoch" % (i+1, np.mean(results), np.std(results), (stop-start)))
    print(lastEpochResults)
        
def useNewTsetlinFormat(imageSet, typeFlag=1, epochs=50, clauses=800*8, T=75, s=10.0, patch_Dim=(8, 8), inputArrayDim=[1, 1], debugInfo=1):
    '''This is a function that can perform LL4, LL8 and any variant of it.
    Because of runtime, and a possible memory leak of the CTM it is advised to do pre-processing and training in seperate steps.'''
    def specialImports(typeFlag, debugInfo=1):
        fixDockerInstanceReset(typeFlag) #This is a workaround for jupyterLab or other instances without persistance.
        if debugInfo:
            print("Spaghetti Docker Fix Worked!")
    
    def savePostProcessToDisc(arrayToSave, name):
        '''Saves the post processed data to disc'''
        # open a binary file in write mode
        file = open(name, "wb")
        np.save(file, arrayToSave)
        # close the file
        file.close
    
    def losslessBinarisation(colorImages, mode=1, pictureWidth=32, pictureHeight=32, colorChannels=3, colorDepth=8):
        '''
        Mode==1 is modified to handle different bit lengths, LL4 and LL8 supported.
        Mode==2 can only handle LL8, but is modified for purposes mentioned in future work about per-pixel labeling.
        '''
        binaryImageList = "" #Instanciate the help variable so the compiler sees it for the return
        if mode == 1:
            binaryImageList = np.empty([len(colorImages),pictureWidth,pictureHeight,colorChannels*colorDepth]) #np.empty([x, 32, 32, 24])
            for iIndex, image in tqdm(enumerate(colorImages)):
                for rIndex, row in enumerate(image):
                    for pIndex, pixel in enumerate(row): # This is technically also column(s)
                        for cIndex, color in enumerate(pixel):
                            for bit in range(colorDepth): #If it is important o have the bits in ascending order, change this to "range(colorDepth-1, -1, -1)"
                                if((color & 2**bit) == 2**bit):
                                    binaryImageList[iIndex][rIndex][pIndex][(cIndex*8)+bit] = 1
                                else:
                                    binaryImageList[iIndex][rIndex][pIndex][(cIndex*8)+bit] = 0
        elif mode == 2:
            binaryImageList = np.empty([colorChannels, colorDepth, len(colorImages),pictureWidth,pictureHeight]) #np.empty([3, 8, x, 32, 32])
            for iIndex, image in tqdm(enumerate(colorImages)):
                for rIndex, row in enumerate(image):
                    for pIndex, pixel in enumerate(row): # This is technically also column(s)
                        for cIndex, color in enumerate(pixel):
                            for bit in range(colorDepth): #If it is important o have the bits in ascending order, change this to "range(colorDepth-1, -1, -1)"
                                if((color & 2**bit) == 2**bit):
                                    binaryImageList[cIndex][bit][iIndex][rIndex][pIndex] = 1
                                else:
                                    binaryImageList[cIndex][bit][iIndex][rIndex][pIndex] = 0
        return binaryImageList #Take note that the binary numbers are reversed, i.e x[0][0] gives the red 2^0 bit while x[1][4] gives the blue 2^4 bit
    
    def loadBaseData(typeFlag=0,debugInfo=1):
        '''Importing of the dataset:
        typeFlag = 1 omits importing X_train and X_test as it assumes this is done in the pre-processing step.
        typeflag = 0 imports all data as normal.'''
        from tensorflow.keras.datasets import cifar10 #If you are using tensorflow
        if typeFlag:
            (_, Y_train), (_, Y_test) = cifar10.load_data() # All data
            Y_train=Y_train.reshape(Y_train.shape[0])
            Y_test=Y_test.reshape(Y_test.shape[0])
            return Y_train, Y_test
        else:
            #from keras.datasets import cifar10 #If you are using keras as a standalone module
            (X_train, Y_train), (X_test, Y_test) = cifar10.load_data() # All data
            if debugInfo:
                print("Dataset loaded correctly! Original shapes -","X_train:",X_train.shape,"X_test:",X_test.shape,"Y_train:",Y_train.shape,"Y_test",Y_test.shape)
            return X_train, Y_train, X_test, Y_test
            
    def preProcessData(imageSet,typeFlag,saveFlag,debugInfo=1):
        '''Takes in base dataset, and returns or saves pre-processed dataset.'''
        X_train, Y_train, X_test, Y_test = loadBaseData(0, debugInfo)
        Y_train=Y_train.reshape(Y_train.shape[0])
        Y_test=Y_test.reshape(Y_test.shape[0])
        X_train, X_test = losslessBinarisation(X_train[:imageSet], typeFlag), losslessBinarisation(X_test[:imageSet], typeFlag)
        if saveFlag:
            savePostProcessToDisc(X_train, "trainPP_"+str(X_train.shape))
            savePostProcessToDisc(X_test, "testPP_"+str(X_test.shape))
        else:
            return X_train, Y_train, X_test, Y_test
    
    def preProcess3DImages(imageSet,typeFlag,saveFlag,debugInfo=1):
        '''Technically a redundant function, used mostly for debugging input/output and its shape.'''
        #import numpy as np
        X_train, Y_train, X_test, Y_test = preProcessData(imageSet,typeFlag,saveFlag,debugInfo)
        if debugInfo:
            print("Post Processing works! New shapes -","X_train:",X_train.shape,"X_test:",X_test.shape,"Y_train:",Y_train.shape,"Y_test",Y_test.shape)
        if not saveFlag:
            return X_train, Y_train, X_test, Y_test

    def loadPostProcessFromDisc(nameOfArray,debugInfo=1):
        '''Loads post-processed data from file.'''
        from time import time
        start = time()
        file = open(nameOfArray, "rb")
        #read the file to numpy array
        loadedArray = np.load(file)
        #close the file
        #print(arr1)
        stop = time()
        if debugInfo:
            print("Pre Processed Data",nameOfArray,"loaded correctly! Time spent: %.1f s" % (stop-start))
        # close the file
        file.close
        return loadedArray
    
    def loadFirstLayerInput(imageSet,loadFiles=["trainPP_(50000, 32, 32, 24)","testPP_(10000, 32, 32, 24)"],debugInfo=1):
        '''This function assumes you are doing pre-processing and training in 2 steps, and returns all data pre-processed.'''
        Y_train, Y_test = loadBaseData(1, debugInfo)
        #X_train, X_test = loadPostProcessFromDisc("trainPP_(3, 8, 50000, 32, 32)"), loadPostProcessFromDisc("testPP_(3, 8, 10000, 32, 32)")
        X_train, X_test = loadPostProcessFromDisc(loadFiles[0], debugInfo), loadPostProcessFromDisc(loadFiles[1], debugInfo)
        #print("Just checking to see if imageSet is local to the function here",imageSet)
        X_train, Y_train, X_test, Y_test = X_train[:imageSet], Y_train[:imageSet], X_test[:imageSet], Y_test[:imageSet]
        return  X_train, Y_train, X_test, Y_test
    
    def tsetlinHyperAndSetup(typeFlag=1, clauses=800, T=75, s=10.0, patch_Dim=(8, 8),dimensionArray=[1, 1],debugInfo=1):
        '''Imports the specified module of the CTM, and defines the hyperparameters.
        The default values set here, are the comparison values from the AGauss paper.
        There is also readied a TM, with regards to the per-pixel method.'''
        if typeFlag == 0:
            ##Standard
            from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
            from pyTsetlinMachine.tm import MultiClassTsetlinMachine
        elif typeFlag == 1:
            ##Paralell
            from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D
            from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
        elif typeFlag == 2:
            ##CUDA
            from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
            from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
        if debugInfo:
            print("Tsetlin loading correctly!")
        
        ctm = []
        if dimensionArray[0] == 1 and dimensionArray[1] == 1:
            ctm.append(MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, patch_Dim))
        else:
            for i in range(dimensionArray[0]):
                perColorTCNN = []
                for y in range(dimensionArray[1]):
                    #MultiClassConvolutionalTsetlinMachine2D(self, number_of_clauses, T, s, patch_dim, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, weighted_clauses=False, s_range=False)
                    perColorTCNN.append(MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, patch_Dim))
                ctm.append(perColorTCNN)
        
        #MultiClassTsetlinMachine(self, number_of_clauses, T, s, boost_true_positive_feedback=1, number_of_state_bits=8, indexed=True, append_negated=True, weighted_clauses=False, s_range=False)
        tm = MultiClassTsetlinMachine(1024, 15, 3.9)
        #tm = MultiClassConvolutionalTsetlinMachine2D(800, 75, 10.0, (1, 4)) #Default values
        if debugInfo:
            print("Finished creating Tsetlin Array")
        return ctm, tm
    
    def trainNeuralNetwork(typeFlag, dataSizeArray, dimensionArray, ctm, tm=0, debugInfo=1):
        '''Main training of the CTM, with prediction matrix.'''
        if typeFlag == 0:
            ##Standard
            from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
            from pyTsetlinMachine.tm import MultiClassTsetlinMachine
        elif typeFlag == 1:
            ##Paralell
            from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D
            from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
        elif typeFlag == 2:
            ##CUDA
            from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
            from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
        if debugInfo:
            print("Tsetlin loading correctly!")
        
        #predictionMatrix = np.zeros([imageSet, 24])
        if dimensionArray[0] == 1 and dimensionArray[1] == 1:
            trainTMMatrix = np.zeros([dataSizeArray[0]])
            predictionMatrix = np.zeros([dataSizeArray[1]])
            tCNNResultsArray = np.zeros(0)
        else:
            trainTMMatrix = np.zeros([dimensionArray[0], dimensionArray[1], dataSizeArray[0]])
            predictionMatrix = np.zeros([dimensionArray[0], dimensionArray[1],dataSizeArray[1]])
            tCNNResultsArray = np.zeros([dimensionArray[0]*dimensionArray[1],])
        results = np.zeros(0)
        print("ResultArray before training",tCNNResultsArray)
        from time import time
        for i in range(epochs): #There might be a memory leak on ctm.fit, so avoid high number of epochs
            if dimensionArray[0] == 1 and dimensionArray[1] == 1:
                for tCNN in ctm: #Always 1 because of the new method
                    start = time()
                    tCNN.fit(X_train, Y_train, epochs=1, incremental=True)
                    #tCNNResultsArray = np.append(tCNNResultsArray, 100*(tCNN.predict(X_test) == Y_test).mean())
                    tCNNResultsArray = np.append(tCNNResultsArray, 100*(tCNN.predict(X_test) == Y_test))
                    #results          = np.append(results,          100*(ctm.predict(X_test)  == Y_test))
                    #tCNNResultsArray = tCNNResultsArray, 100*(tCNN.predict(X_test) == Y_test).mean()
                    stop = time()
                    print("| #%d Mean Accuracy (%%): %.2f; Std.dev.: %.2f; Training Time: %.1f s/epoch" % (i+1, np.mean(tCNNResultsArray), np.std(tCNNResultsArray), (stop-start)))
                    #print("| #%d Mean Accuracy (%%): %.2f; Std.dev.: %.2f; Training Time: %.1f s/epoch" % (i+1, np.mean(tCNNResultsArray), np.std(tCNNResultsArray), (stop-start)), "Error check",tCNNResultsArray)
            else:
                print("Something went wrong with dimensions. NOTE: only [1, 1] dimensions work for this setup")
        print("ResultArray after training",tCNNResultsArray)
                
    ### Sub-Functions
    ## Check that libraries are present
    specialImports(typeFlag, debugInfo)
    ## Pre process the data
    #preProcess3DImages(imageSet=imageSet, typeFlag=TypeFlag, saveFlag=1, debugInfo=debugInfo) #Saveflag determines whether to save to disk (1) or do preprocess abd train (0)
    ## Train the model
    X_train, Y_train, X_test, Y_test = loadFirstLayerInput(imageSet,debugInfo=debugInfo)
    ctm, _ = tsetlinHyperAndSetup(typeFlag=typeFlag, clauses=clauses, T=T, s=s, patch_Dim=patch_Dim, dimensionArray=inputArrayDim, debugInfo=debugInfo) #returns : ctm, tm
    trainNeuralNetwork(tsetlinType, [len(X_train), len(X_test)],inputArrayDim, ctm)


## Runtime
print("Program Started -",printTimeStamp())
print("Current Variables - Img:",imagesToTrainOn,",epoch:",epochs,",clauses",clauses,",T:",T,",s:",s,",slidingWindow:",patch_Dim,",type:",versionUsed[informationProcessingType])
if informationProcessingType:
    useNewTsetlinFormat(imageSet=imagesToTrainOn, 
                                typeFlag=tsetlinType, 
                                epochs=epochs, 
                                clauses=clauses, 
                                T=T, 
                                s=s, 
                                patch_Dim=patch_Dim,
                                inputArrayDim=inputArrayDimension, #Can comment out this as the default values is there
                                debugInfo=showDebugInfo)
else:
    useTheTsetlinFormatThatWork(imageSet=imagesToTrainOn, 
                                typeFlag=tsetlinType, 
                                epochs=epochs, 
                                clauses=clauses, 
                                T=T, 
                                s=s, 
                                patch_Dim=patch_Dim, 
                                debugInfo=showDebugInfo)
print("Program Finished -",printTimeStamp())