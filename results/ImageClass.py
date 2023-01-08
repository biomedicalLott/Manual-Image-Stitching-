import numpy as np
import cv2
class Pixel:
    def __init__(self, red,green,blue, x,y):
        self.red = red
        self.green = green
        self.blue = blue
        self.oldX = x
        self.oldY = y
        self.x = x
        self.y = y
        self.pt = [x,y,1]
    def applyShift(self, x,y):
        self.x = x
        self.y = y
        self.pt = [x, y, 1]


class ImageObj:
    def __init__(self):
        self.name = "no"
    def __init__(self, img,smoothed, gray, keypoints, descriptors):
        self.original = img
        self.smoothed = smoothed
        self.gray = gray
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.imageSize = np.shape(img)
        self.pixelSize = self.imageSize
        self.pixels = np.empty((self.imageSize[0],self.imageSize[1]),Pixel)
        self.bestIndices = {}
        self.bestkeypoints = {}
        self.bestdescriptors ={}
        self.bestPoints = {}
        self.makePixels()

    def makePixels(self):
        img = self.original
        for i in range(0, self.imageSize[0]):
            for j in range(0, self.imageSize[1]):
                blue = img[i,j,0]
                green = img[i,j,1]
                red = img[i,j,2]
                x = i
                y = j
                self.pixels[i,j] = Pixel(red,green, blue, x, y)
    def storeBestValues(self, bestIndices):
        self.bestIndices = bestIndices
        keypoints = np.array(self.keypoints)
        self.bestkeypoints = keypoints[bestIndices]
        descriptors = np.array(self.descriptors)
        self.bestdescriptors = descriptors[bestIndices]
        self.bestPoints = np.array([keypoint.pt for keypoint in self.bestkeypoints])
        return
    def warpImage(self,hMatrix):
        h11 = hMatrix[0, 0];h12 = hMatrix[0, 1];h13 = hMatrix[0, 2]
        h21 = hMatrix[1, 0];h22 = hMatrix[1, 1];h23 = hMatrix[1, 2]
        h31 = hMatrix[2, 0];h32 = hMatrix[2, 1];h33 = hMatrix[2, 2]
        imagesize = np.size(self.gray)
        yOffset = np.zeros((imagesize,1))
        xOffset = np.zeros((imagesize,1))
        count  = 0
        for i in range(self.imageSize[0]):
            for j in range(self.imageSize[1]):
                x = i
                y = j
                x = int(np.round((h11*x + h12*y + h13) / (h31*x + h32*y + h33)))
                y = int(np.round((h21*x + h22*y + h23) / (h31*x + h32*y + h33)))
                self.pixels[i, j].applyShift(x, y)
                yOffset[count] = y
                xOffset[count] = x
                count = count +1
        xOffset = int(np.abs(np.min(xOffset))+10)
        yOffset = int(np.abs(np.min(yOffset))+10)
        self.Offset = np.array([yOffset,xOffset,0])

    def shiftBestPoints(self):
        self.shiftedBestPoints = []
        for point in self.bestPoints:
            shiftedPoints = np.empty(1,Pixel)
            shiftedPoints  = self.pixels[point]
            self.shiftedBestPoints.append(shiftedPoints.pt)
        self.shiftedBestPoints = np.array(self.shiftedBestPoints)


    def shiftImage(self):
        imagesize = np.size(self.gray)
        yOffset = np.zeros((imagesize, 1))
        xOffset = np.zeros((imagesize, 1))
        count  = 0
        for i  in range(self.imageSize[0]):
            for j in range(self.imageSize[1]):
                pixelX = self.pixels[i,j].x + abs(self.Offset[0])
                pixelY = self.pixels[i,j].y + abs(self.Offset[1])
                yOffset[count] = pixelY
                xOffset[count] = pixelX
                self.pixels[i,j].applyShift(pixelX,pixelY)
                count += 1

    def moveImage(self, hMatrix):
        img = self.original
        imageSize = self.imageSize
        oldPixels = np.copy(self.pixels)
        newPixels = oldPixels
        for i in range(0, imageSize[0]):
            for j in range(0,imageSize[1]):
                newPixels[i, j].shift(hMatrix)
        self.pixels = np.copy(newPixels)
    def reconstructImage(self, offset = [0,0]):

        pixels = np.copy(self.pixels)
        imageSize = self.imageSize

        x = np.zeros(imageSize[:2])
        y = np.zeros(imageSize[:2])
        for i in range(0, imageSize[0]):
            for j in range(0,imageSize[1]):
                x[i,j] = int(pixels[i,j].x)
                y[i,j] = int(pixels[i,j].y)

        minX = np.min(np.min(x))
        maxX = np.max(np.max(x))
        minY = np.min(np.min(y))
        maxY = np.max(np.max(y))
        xOffset = abs(minX)
        yOffset = abs(minY)

        self.Offset = [xOffset,yOffset]

        maxX = int(xOffset + maxX+1)
        maxY = int(yOffset + maxY+1)
        image = np.zeros([maxX , maxY, 3])
        x = x + int(xOffset)
        y = y + int(yOffset)
        x = x.astype(int)
        y = y.astype(int)
        self.corners = np.array([[x[0,0],y[0,0]],
                                 [x[0,-1],y[0,-1]],
                                 [x[-1,0],y[-1,0]],
                                 [x[-1,-1],y[-1,-1]]], dtype=np.int32)
        for i in range(0, imageSize[0]):
            for j in range(0,imageSize[1]):
                image[x[i,j], y[i,j], 0] = pixels[i, j].blue
                image[x[i,j], y[i,j], 1] = pixels[i, j].green
                image[x[i,j], y[i,j], 2] = pixels[i, j].red
        screenSize = (1920,1080)
        if maxX > screenSize[0] or maxY > screenSize[1]:
            image = cv2.resize(image,screenSize)
            # cv2.imshow("resized image", image)
        # cv2.imshow("changed", image)
        cv2.waitKey(0)
        warpedImage = image
        return warpedImage