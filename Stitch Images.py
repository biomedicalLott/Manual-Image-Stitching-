import itertools
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

class ImageObj:

    def __init__(self, img,gray, keypoints, descriptors):
        self.original = img
        self.gray = gray
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.imageSize = np.shape(img)
        self.bestIndices = {}
        self.bestkeypoints = {}
        self.bestdescriptors ={}
        self.bestPoints = {}
    def storeBestValues(self, bestIndices):
        self.bestIndices = bestIndices
        keypoints = np.array(self.keypoints)
        self.bestkeypoints = keypoints[bestIndices]
        descriptors = np.array(self.descriptors)
        self.bestdescriptors = descriptors[bestIndices]
        self.bestPoints = np.array([keypoint.pt for keypoint in self.bestkeypoints])
        return


def stitch(imgmark, N=4, savepath=''):
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for n,ipath in enumerate(imgpath):
        img = cv2.imread(ipath)
        imgs.append(img)
    # First create the sift object

    # siftObj = cv2.SIFT_create(nfeatures=1000,contrastThreshold=0.02, edgeThreshold=20, sigma=3) %works on my weak white walls
    siftObj = cv2.SIFT_create()
    # <editor-fold desc="pre-processing to get overlap array and determine order to take images in">
    # Check if images need to be scaled down if they're too darn big
    scaling = float(950)
    shape = [np.shape(img) for img in imgs]
    shape2 = np.max(np.max(shape))
    if float(shape2) >= scaling:
        imgs = resizeImages(imgs,scaling)
    # imgs2 = []
    # for i, img in enumerate(imgs):
    #     imgs2.append(np.array(smooth(img,51),dtype=np.float32))
    # Now take the images, extract features, and create image objects
    padding = 500
    imgCorners  = []
    imageObjects = np.empty((N,1), ImageObj)
    for i,img in enumerate(imgs):
        paddedImg, corners = padImage(img,padding)
        imgCorners.append(corners)
        imageObjects[i] = feature_extraction(np.array(paddedImg), siftObj)
    # Compute overlap array
    scoreThreshold = 1200
    overlap_arr = getOverlapArray(imageObjects, N,scoreThreshold)
    imageObjects = determineOrderOfOperations(overlap_arr,imageObjects)
    # </editor-fold desc="pre-processing to get overlap array and determine order to take images in">
    # <editor-fold desc="Image Stitching">

    # # Now remove bad images from play so we're not overlapping things that can't overlap
    # matchRow = np.sum(overlap_arr,axis = 0)
    # matchRow = matchRow-1
    # imageObjects = imageObjects[np.nonzero(matchRow)]
    N = len(imageObjects)
    ############ STITCH IMAGES TOGETHER ############
    newImageObject = imageObjects[0][0]
    newCorners = imgCorners[0]
    for index in range(1,int(N)):
        imgObj = imageObjects[index][0]
        # cv2.imshow('original image from index',imgObj.original)
        # Find the best matches, brute force style
        descriptors = FindBestDescriptors(imgObj, newImageObject)
        # cv2.imshow('original image from index B', imgObj.original)
        # Store them in the image object
        imgObj,newImageObject= \
            storeBestValues(descriptors, imgObj, newImageObject)
        # cv2.imshow('original image from index C', imgObj.original)
        # Compute the homography of the object and warp it to fit
        r,g,b = computeHomographyAndWarp(imgObj,newImageObject)
        # Now stitch the one image onto the other image
        # Experimental dualBandBlending - couldn't get it to work, commenting out.
        # holdImage = cv2.merge((b,g,r))
        # dualBandBlending(holdImage,newImageObject.original)
        newImage = mergeImages(r, g, b, newImageObject)
        newImage = unpadImage(np.copy(newImage))
        # If we're at the end, don't continue to pad and extract features
        if N-index > 1:
            newPaddedImg, newCorners = padImage(np.copy(newImage), padding)
            newImageObject = feature_extraction(np.copy(newPaddedImg), siftObj)
    # </editor-fold desc="Image Stitching">

    # finally remove excess padding from the image
    stitchedImage = unpadImage(np.copy(newImage))
    # and show us what i've got
    # cv2.imshow('lazy stitch',stitchedImage)
    cv2.imwrite(savepath, stitchedImage)

    return overlap_arr


#<editor-fold desc="Image Merging Section">
def computeHomographyAndWarp(img1Obj,img2Obj):
    # Take the best points we found
    img1Best = np.copy(img1Obj.bestPoints)
    img2Best = np.copy(img2Obj.bestPoints)
    # Compute the homography needed to move img1 to img2
    hMatrix, mask = cv2.findHomography(img1Best, img2Best,cv2.RANSAC,4.0)
    # Take in the size of the destination image
    destImageSize = [img2Obj.imageSize[1], img2Obj.imageSize[0]]
    # Warp the perspective of the r,g, and b layers
    bTemp = cv2.warpPerspective(np.copy(img1Obj.original[:,:, 0]), hMatrix, destImageSize)
    gTemp = cv2.warpPerspective(np.copy(img1Obj.original[:,:, 1]), hMatrix, destImageSize)
    rTemp = cv2.warpPerspective(np.copy(img1Obj.original[:,:, 2]), hMatrix, destImageSize)
    # Separate images out into layers
    return rTemp, gTemp, bTemp
def mergeImages(rTemp,gTemp,bTemp, img2Obj):
    r1 = np.copy(rTemp);
    g1 = np.copy(gTemp);
    b1 = np.copy(bTemp);

    r2 = np.copy(img2Obj.original[:, :, 2]);
    g2 = np.copy(img2Obj.original[:, :, 1]);
    b2 = np.copy(img2Obj.original[:, :, 0]);
    # now merge the layers over top of one another
    r = cv2.merge((r1, r2));
    g = cv2.merge((g1, g2));
    b = cv2.merge((b1, b2))
    # find the maximum value to make the foreground objects disappear, hopefully.
    r = np.max(r, axis=2);
    g = np.max(g, axis=2);
    b = np.max(b, axis=2)
    merge = cv2.merge((b, g, r))
    # cv2.imshow("final merge ", merge / np.max(merge))
    return merge
#</editor-fold >

#<editor-fold desc="Image feature extraction and placement into object.">
def storeBestValues(bestDescriptors, img1Obj, img2Obj):
    # Take the index for the best descriptors and the corresponding keypoint and store it
    img1Indices = np.array([int(index[1]) for index in bestDescriptors])
    indexCount = np.size(img1Indices)
    img1Indices = img1Indices.reshape(indexCount)
    img2Indices = np.array([int(index[3]) for index in bestDescriptors])
    img2Indices = img2Indices.reshape(indexCount)
    img1Obj.storeBestValues(img1Indices)
    img2Obj.storeBestValues(img2Indices)
    return img1Obj, img2Obj
def FindBestDescriptors(img1Obj, img2Obj):
    # Find idk, 400 best descriptors?
    descriptorSet1 = np.copy(img1Obj.descriptors)
    descriptorSet2 = np.copy(img2Obj.descriptors)
    descriptorCount = np.shape(descriptorSet1)
    bestIndex = np.ones([descriptorCount[0],1])
    scores = np.zeros([descriptorCount[0],1])
    indexRange = list(range(0,descriptorCount[0]))
    tempTuple = []
    matchesCap = 400
    # First, create a tuple list to store all of them
    for i,descriptor in enumerate(descriptorSet1):
        scores[i], bestIndex[i] = ssd(descriptor,descriptorSet2)
        tempTuple.append([descriptor,i,scores[i],bestIndex[i]])
    # Then sort them by the score from SSD
    tempTuple.sort(key = lambda x: x[2])
    # Retain only the best so many matches
    bestDescriptorTuple =tempTuple[0:matchesCap]
    return bestDescriptorTuple

def feature_extraction(originalImg, siftObj):
    gray = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = siftObj.detectAndCompute(gray,None)
    # Create new image using extracted descriptors
    imgObj = ImageObj(originalImg, gray, keypoints, descriptors)
    return imgObj
#</editor-fold>

#<editor-fold desc="Distance Formulas">
def normHamming(descriptor, descriptorset):
    descriptor = np.array(descriptor,dtype=np.uintc)
    descriptorset = np.array(descriptorset,dtype= np.uintc)
    hamminged = np.linalg.norm(descriptor-descriptorset,axis=1)
    # summed = np.sum(hamminged,axis=1)
    # subtractedAndSquared = pow(descriptor - descriptorset, 2)
    # summed = np.sum(subtractedAndSquared, axis=1)
    minIndex = np.argmin(hamminged)
    minValue = hamminged[minIndex]
    return minValue,minIndex

def ssd(descriptor, descriptorset):
    subtractedAndSquared = pow(descriptor - descriptorset,2)
    summed = np.sum(subtractedAndSquared, axis = 1)
    minIndex = np.argmin(summed)
    minValue = summed[minIndex]
    return minValue,minIndex
#</editor-fold>

#<editor-fold desc="Padding Section">

def padImage(img, padding):
    Black = [0, 0, 0]
    imgSize = np.shape(img)
    corners = np.array([[0,0],
                        [imgSize[0],0],
                        [imgSize[0],imgSize[1]],
                        [0, imgSize[1]]])
    pad = padding
    corners = corners + pad
    paddedImg = cv2.copyMakeBorder(img.copy(), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=Black)

    return paddedImg, corners
def unpadImage(img):
    # Remove black padding from the image
    # first store a copy, and find the maximum value across r,g, and b layers
    tempImg = np.max(np.copy(img),axis=2)
    # Then, take the sum across the axis to form rows and columns
    rows = np.sum(tempImg,axis=0)
    columns = np.sum(tempImg,axis=1)
    # Now, look for non-zero values in those rows and columns
    nonZeroRows = np.nonzero(rows)
    nonZeroColumns = np.nonzero(columns)
    # and use the first and last of those values to eliminate the unnecessary border
    img = img[nonZeroColumns[0][0]:nonZeroColumns[0][-1],nonZeroRows[0][0]:nonZeroRows[0][-1],:]
    return img
#</editor-fold desc="Padding Section">

#<editor-fold desc="Pre-processing">

def getOverlapArray(imageObjects, N, scoreThreshold):
    matchArray = np.zeros((N, N))
    # The score threshold is 100000, there's no really great reason for it

    # Compare all the images descriptors to find out what images pair well
    for i, Obj in enumerate(imageObjects):
        imgObj = Obj[0]
        for j in range(0, N):
            if i == j:
                # Skip some processing, store a 1 if it's looking at itself
                matchArray[i, j] = 1
                continue
            #     Find the 400 best descriptors between the images
            descriptors = FindBestDescriptors(imgObj, imageObjects[j][0])
            # Now find the lowest score
            scores = np.array([score[2] for score in descriptors])
            minScore = np.min(scores)
            # If the lowest score is higher than the threshold, it doesn't match up
            if minScore > scoreThreshold:
                matchArray[i, j] = 0
                continue
            matchArray[i, j] = 1
    return matchArray

def determineOrderOfOperations(overlap_arr, imageObjects):
    # Step 1, figure out if there are any images that don't match at all
    matchRow = np.sum(overlap_arr, axis=0)
    matchRow = matchRow - 1
    # How many non zero values gives us a good idea of how well this will go
    matchedIndex = np.nonzero(matchRow)
    matchRow = matchRow[matchedIndex]
    # the image with the fewest connections will be our starting point
    startingPoint = np.argmin(matchRow)
    # 2. Determine Permutations
    # From here, delete the starting point from the matched index
    matchedIndex = np.delete(matchedIndex,startingPoint)
    indexLen = len(matchedIndex)
    # Important tool for determining that, it'll return a goodImage count by goodImage count factorial
    combos = np.array([combo for combo in itertools.permutations(matchedIndex,indexLen)])
    # Now, place the starting point back into the starting position
    arrayOfPossiblePaths = np.ones([len(combos),indexLen+1])
    arrayOfPossiblePaths[:,0] = arrayOfPossiblePaths[:,0] * startingPoint
    # Insert the permutations
    arrayOfPossiblePaths[:,1:] = combos
    # Then remove the places where these points don't even exist
    overlapRow = overlap_arr[startingPoint, :]
    overlapRow[startingPoint] = 0;
    removeTheseOptions = np.where(overlapRow == 0 )
    killArray = []
    # And remove all the bad paths that would stop at the very first step
    for option in removeTheseOptions[0]:
        killArray.append(arrayOfPossiblePaths[:,1] == option)
    killArray2 =  np.nonzero(np.sum(killArray, axis=0) == 0)
    arrayOfPossiblePaths = arrayOfPossiblePaths[killArray2]
    # If only one path remains, we can stop now, there's no point in continuing
    arrayCount = len(arrayOfPossiblePaths)
    if arrayCount == 1:
        imagePath = np.array(arrayOfPossiblePaths[0])
        imageObjects = imageObjects[np.flip(np.array(imagePath,dtype=np.int32))]
        return imageObjects, imagePath
    # Now, finally, pick the best possible path
    for path in arrayOfPossiblePaths:
        badPath = False;
        pathLen = len(path)
        # Iterate along the path checking if it exists in the next array
        for i in range(0, pathLen-1):
            overlapRow = overlap_arr[int(path[i]),:]
            if overlapRow[int(path[i+1])] == 0:
                badPath = True;
                break
        if badPath == False:
            imagePath = np.array(path)
            organizedImages = [imageObjects[int(point)] for point in np.flip(imagePath)]
            return organizedImages
    print("Oh crap. Something went wrong, no good paths were found!")
    imageObjects = [imageObjects[int(point)] for point in np.flip(arrayOfPossiblePaths[0])]
    return imageObjects

def resizeImages(imgs, scaling = 950):
    needsToBeResized = False
    # scaling = np.array(1)
    largestDimensions = np.array([1000, 1000])
    for img in imgs:
        size = np.shape(img)
        size = np.array(size[:2])
        if needsToBeResized == False:
            if size[0] > 1000 or size[1] > 1000:
                needsToBeResized = True
        if needsToBeResized == True:
            x = np.max([largestDimensions[0],size[0]])
            y = np.max([largestDimensions[1],size[1]])
            largestDimensions = [x,y]
    if needsToBeResized == False:
        return imgs
    scaling = scaling / np.max(largestDimensions)
    for i, img in enumerate(imgs):
        size = np.float32(np.shape(img))
        size = np.int16(size*scaling)
        imgs[i] = cv2.resize(img, size[:2], interpolation=cv2.INTER_AREA);
    return imgs
#</editor-fold desc="Pre-processing">

if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='results/task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3',N=4, savepath='results/task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
