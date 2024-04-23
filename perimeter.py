from skimage.measure import label, regionprops
import skimage.io as io
# read in the image (enter the path where you downloaded it on your computer below
im = io.imread('Lungfilter after closing.png')
# To simplify things I am only using the first channel and thresholding
# to get a boolean image
bw = im[:,:,0] > 230
regions = regionprops(bw.astype(int))
print(regions[0].perimeter)
