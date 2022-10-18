from skimage import color as skic
import numpy as np

def mean(img):
    """
    Returns the average of all values
    """
    return img.mean()

def std(img):
    """
    Returns standard deviation
    """
    return img.std()

def avgRed(img):
    """
    Gets average red value
    """
    return img[:,:,0].mean()

def avgGreen(img):
    """
    Gets average green value
    """
    return img[:,:,1].mean()

def avgBlue(img):
    """
    Gets average blue value
    """
    return img[:,:,2].mean()

def frequencyPeakRedRGB(img):
    """
    Get the most frequent value in the blue channel
    """
    unique, counts = np.unique(img[:,:,0].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def frequencyPeakGreenRGB(img):
    """
    Get the most frequent value in the blue channel
    """
    unique, counts = np.unique(img[:, :, 1].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def frequencyPeakBlueRGB(img):
    """
    Get the most frequent value in the blue channel
    """
    unique, counts = np.unique(img[:,:,2].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def maxPeakRed(img):
    """
    Gets max peak of red
    """
    y, x = np.histogram(img[:,:,0], bins=20)
    return x[np.where(y == y.max())][0]

def maxPeakGreen(img):
    """
    Gets max peak of green
    """
    y, x = np.histogram(img[:,:,1], bins=20)
    return x[np.where(y == y.max())][0]

def maxPeakBlue(img):
    """
    Gets max peak of blue
    """
    y, x = np.histogram(img[:,:,2], bins=20)
    return x[np.where(y == y.max())][0]

def meanYcbcr(img):
    """
    Returns the average of all values in the Ycbcr color space
    """

    return skic.rgb2ycbcr(img).mean()

def stdYcbcr(img):
    """
    Returns standard deviation in Ycbcr color space
    """
    return skic.rgb2ycbcr(img).std()

def avgY(img):
    """
    Gets average Y value in Ycbcr color space
    """
    return skic.rgb2ycbcr(img)[:,:,0].mean()
def avgcb(img):
    """
    Gets average cb value in Ycbcr color space
    """
    return skic.rgb2ycbcr(img)[:,:,1].mean()

def avgcr(img):
    """
    Gets average cr value in Ycbcr color space
    """
    return skic.rgb2ycbcr(img)[:,:,2].mean()
def frequencyPeakY(img):
    """
    Get the most frequent value in the Y channel in Ycbcr color space
    """
    img=skic.rgb2ycbcr(img)
    unique, counts = np.unique(img[:,:,0].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def frequencyPeakcb(img):
    """
    Get the most frequent value in the cb channel in Ycbcr color space
    """
    img=skic.rgb2ycbcr(img)
    unique, counts = np.unique(img[:,:,1].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]
def frequencyPeakcr(img):
    """
    Get the most frequent value in the cr channel in Ycbcr color space
    """
    img=skic.rgb2ycbcr(img)
    unique, counts = np.unique(img[:,:,2].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def upperLeftAvgRed(img):
    """
    Gets the avg of red in the left upper image
    """
    return img[:127, :127, 0].mean()

def upperLeftAvgGreen(img):
    """
    Gets the avg of green in the left upper image
    """
    return img[:127, :127, 1].mean()

def upperLeftAvgBlue(img):
    """
    Gets the avg of blue in the left upper image
    """
    return img[:127, :127, 2].mean()

def upperLeftHFRed(img):
    """
    Gets the Highest frequency of red in upper left
    """
    unique, counts = np.unique(img[:127, :127, 0].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def upperLeftHFGreen(img):
    """
    Gets the Highest frequency of green in the left upper image
    """
    unique, counts = np.unique(img[:127, :127, 1].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def upperLeftHFBlue(img):
    """
    Gets the Highest frequency of blue in the left upper image
    """
    unique, counts = np.unique(img[:127, :127, 2].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def upperLeftHistRed(img):
    """
    Gets max peak of red in upper left corner
    """
    y, x = np.histogram(img[:127,:127,0], bins=20)
    return x[np.where(y == y.max())][0]

def upperLeftHistGreen(img):
    """
    Gets max peak of green in upper left corner
    """
    y, x = np.histogram(img[:127,:127,1], bins=20)
    return x[np.where(y == y.max())][0]

def upperLeftHistBlue(img):
    """
    Gets max peak of blue in upper left corner
    """
    y, x = np.histogram(img[:127,:127,2], bins=20)
    return x[np.where(y == y.max())][0]

def upperRightAvgRed(img):
    """
    Gets the avg of red in the Right upper image
    """
    return img[128:, :127, 0].mean()

def upperRightAvgGreen(img):
    """
    Gets the avg of green in the Right upper image
    """
    return img[128:, :127, 1].mean()

def upperRightAvgBlue(img):
    """
    Gets the avg of blue in the Right upper image
    """
    return img[128:, :127, 2].mean()

def upperRightHFRed(img):
    """
    Gets the Highest frequency of red in upper Right
    """
    unique, counts = np.unique(img[128:, :127, 0].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def upperRightHFGreen(img):
    """
    Gets the Highest frequency of green in the Right upper image
    """
    unique, counts = np.unique(img[128:, :127, 1].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def upperRightHFBlue(img):
    """
    Gets the Highest frequency of blue in the Right upper image
    """
    unique, counts = np.unique(img[128:, :127, 2].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def upperRightHistRed(img):
    """
    Gets max peak of red in upper Right corner
    """
    y, x = np.histogram(img[128:,:127,0], bins=20)
    return x[np.where(y == y.max())][0]

def upperRightHistGreen(img):
    """
    Gets max peak of green in upper Right corner
def frequencyPeakcr(img):
    """

    y, x = np.histogram(img[128:,:127,1], bins=20)
    return x[np.where(y == y.max())][0]

def upperRightHistBlue(img):
    """
    img=skic.rgb2ycbcr(img)
    unique, counts = np.unique(img[:,:,2].flatten(), return_counts=True)
    Gets max peak of blue in upper Right corner
    """
    y, x = np.histogram(img[128:,:127,2], bins=20)
    return x[np.where(y == y.max())][0]

def lowerLeftAvgRed(img):
    """
    Gets the avg of red in the left lower image
    """
    return img[:127, 128:, 0].mean()

def lowerLeftAvgGreen(img):
    """
    Gets the avg of green in the left lower image
    """
    return img[:127, 128:, 1].mean()

def lowerLeftAvgBlue(img):
    """
    Gets the avg of blue in the left lower image
    """
    return img[:127, 128:, 2].mean()

def lowerLeftHFRed(img):
    """
    Gets the Highest frequency of red in lower left
    """
    unique, counts = np.unique(img[:127, 128:, 0].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def lowerLeftHFGreen(img):
    """
    Gets the Highest frequency of green in the left lower image
    """
    unique, counts = np.unique(img[:127, 128:, 1].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def lowerLeftHFBlue(img):
    """
    Gets the Highest frequency of blue in the left lower image
    """
    unique, counts = np.unique(img[:127, 128:, 2].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def lowerLeftHistRed(img):
    """
    Gets max peak of red in lower left corner
    """
    y, x = np.histogram(img[:127,128:,0], bins=20)
    return x[np.where(y == y.max())][0]

def lowerLeftHistGreen(img):
    """
    Gets max peak of green in lower left corner
    """
    y, x = np.histogram(img[:127,128:,1], bins=20)
    return x[np.where(y == y.max())][0]

def lowerLeftHistBlue(img):
    """
    Gets max peak of blue in lower left corner
    """
    y, x = np.histogram(img[:127,128:,2], bins=20)
    return x[np.where(y == y.max())][0]

def lowerRightAvgRed(img):
    """
    Gets the avg of red in the Right lower image
    """
    return img[128:, 128:, 0].mean()

def lowerRightAvgGreen(img):
    """
    Gets the avg of green in the Right lower image
    """
    return img[128:, 128:, 1].mean()

def lowerRightAvgBlue(img):
    """
    Gets the avg of blue in the Right lower image
    """
    return img[128:, 128:, 2].mean()

def lowerRightHFRed(img):
    """
    Gets the Highest frequency of red in lower Right
    """
    unique, counts = np.unique(img[128:, 128:, 0].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def lowerRightHFGreen(img):
    """
    Gets the Highest frequency of green in the Right lower image
    """
    unique, counts = np.unique(img[128:, 128:, 1].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def lowerRightHFBlue(img):
    """
    Gets the Highest frequency of blue in the Right lower image
    """
    unique, counts = np.unique(img[128:, 128:, 2].flatten(), return_counts=True)
    max_index = np.argmax(counts)
    return unique[max_index]

def lowerRightHistRed(img):
    """
    Gets max peak of red in lower Right corner
    """
    y, x = np.histogram(img[128:,128:,0], bins=20)
    return x[np.where(y == y.max())][0]

def lowerRightHistGreen(img):
    """
    Gets max peak of green in lower Right corner
    """
    y, x = np.histogram(img[128:,128:,1], bins=20)
    return x[np.where(y == y.max())][0]

def lowerRightHistBlue(img):
    """
    Gets max peak of blue in lower Right corner
    """
    y, x = np.histogram(img[128:,128:,2], bins=20)
    return x[np.where(y == y.max())][0]
