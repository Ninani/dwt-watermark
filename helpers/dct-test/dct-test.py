import sys, os
from PIL import Image
import numpy
import scipy.fftpack

# sourceImage = sys.argv[1]
image_name = 'mis1.jpg'

image = Image.open('./pictures/' + image_name)
image = image.resize( (512, 512), 1 )
image = image.convert("L")

dctSize = image.size[0]

# get raw pixel values:
pixels = numpy.array(image.getdata(), dtype=numpy.float).reshape((dctSize, dctSize))
all_subdct = numpy.empty((dctSize, dctSize))
for i in range (0, pixels[0].__len__(), 8):
    for j in range (0, pixels[0].__len__(), 8):
        subpixels = pixels[i:i+8, j:j+8]
        subdct = scipy.fftpack.dct(scipy.fftpack.dct(subpixels.T, norm="ortho").T, norm="ortho")
        all_subdct[i:i+8, j:j+8] = subdct
        # subdct2 = subdct.clip(0, 255)
        # subdct2 = subdct2.astype("uint8")
        # subdct_img = Image.fromarray(subdct2)
        # subdct_img.save("./result/after_dct%d%d.png" % (i, j))

all_subdct2 = all_subdct.clip(0, 255)
all_subdct2 = all_subdct2.astype("uint8")
all_subdct_img = Image.fromarray(all_subdct2)
all_subdct_img.save("./result/after_dct_div8.png")

# perform 2-dimensional DCT (discrete cosine transform):
dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels.T, norm="ortho").T, norm="ortho")
dct2 = dct.clip(0, 255)
dct2 = dct2.astype("uint8")
dct_img = Image.fromarray(dct2)

dct_img.save("./result/after_dct.png") 



# image_subdct = Image.open('./result/after_dct_all_subdct.png')
# image_subdct = image_subdct.resize( (128, 128), 1 )
# image_subdct = image_subdct.convert("L")
# read_all_subdct = numpy.array(image_subdct.getdata(), dtype=numpy.float).reshape((dctSize, dctSize))

all_subidct = numpy.empty((dctSize, dctSize))
for i in range (0, pixels[0].__len__(), 8):
    for j in range (0, pixels[0].__len__(), 8):
        subidct = scipy.fftpack.idct(scipy.fftpack.idct(all_subdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
        # subidct = scipy.fftpack.dct(scipy.fftpack.dct(read_all_subdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
        all_subidct[i:i+8, j:j+8] = subidct

all_subidct2 = all_subidct.clip(0, 255)
all_subidct2 = all_subidct2.astype("uint8")
all_subidct_img = Image.fromarray(all_subidct2)
all_subidct_img.save("./result/after_idct_div8.png")




# create a series of images with increasingly larger parts of the DCT values being used:
# os.mkdir("frames/")
# for i in range(0, dctSize):
#     dct2 = dct.copy()

#     # zero out part of the higher frequencies of the DCT values:
#     dct2[i:,:] = 0
#     dct2[:,i:] = 0

#     # perform 2d inverse DCT to get pixels back:
#     idct = scipy.fftpack.idct(scipy.fftpack.idct(dct2.T, norm='ortho').T, norm='ortho')

#     # clip/convert pixel values obtained by IDCT, and create image:
#     idct = idct.clip(0, 255)
#     idct = idct.astype("uint8")
#     img = Image.fromarray(idct)

#     print img
#     img.save("frames/img_%04d.png" % i)


# os.system("convert -delay 30 -comment 'example of Discrete Cosine Transform (source image: %s)' frames/img_*.png dct.gif" % sourceImage)
