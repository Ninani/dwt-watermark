import numpy as np
import pywt
import cv2 
import os

current_path = str(os.path.dirname(__file__))  

image = 'mis1.jpg'   
watermark = 'qrcode.png' 

def convert_image(image, size):
    imArray = cv2.imread(current_path + '/pictures/' + image)

    imArray = cv2.resize(imArray, (size, size))
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    # print imArray[0][0]               #print color for pixel(0,0)
    cv2.imwrite(current_path + '/dataset/' + image, imArray)

    imArray =  np.float32(imArray) 
    imArray /= 255;
    # print imArray[0][0]               #qrcode white color = 1.0
    # print imArray[10][10]             #qrcode black color = 0.0           
    return imArray

def process_coefficients(imArray, model, level):
    coeffs=pywt.wavedec2(data = imArray, wavelet = model, level = level)
    print coeffs[0].__len__()
    coeffs_H=list(coeffs) 
   
    return coeffs_H


def embed_mod2(coeff_image, coeff_watermark):
    for i in xrange(coeff_watermark.__len__()):
        for j in xrange(coeff_watermark[i].__len__()):
            coeff_image[i*2][j*2] = coeff_watermark[i][j]

            
    
def embed_watermark(coeffs_image, watermark_image):

    levels = coeffs_image.__len__()-1

    # embed_mod2(coeffs_image[levels][0], watermark_image)
    #embed_mod2(coeffs_image[levels][1], watermark_image)
    embed_mod2(coeffs_image[levels][2], watermark_image)


def w2d(img):
    model = 'haar'
    image_array = convert_image(image, 256)
    watermark_array = convert_image(watermark, 64)


    coeffs_image = process_coefficients(image_array, model, level=2)
    coeffs_watermark = process_coefficients(watermark_array, model, level=1)

    embed_watermark(coeffs_image, watermark_array)


# reconstruction
    imArray_H=pywt.waverec2(coeffs_image, model)
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

#Save result
    cv2.imwrite(current_path + '/result/image_with_watermark.jpg', imArray_H)


w2d("test")