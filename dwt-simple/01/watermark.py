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


    return coeff_image
            
    
def embed_watermark(coeffs_image, coeffs_watermark):
    dst = [] 
    arr0 = embed_mod2(coeffs_image[0], coeffs_watermark[0])
    dst.append(arr0)

    for x in xrange(1, coeffs_image.__len__()):
        arr0 = embed_mod2(coeffs_image[x][0], coeffs_watermark[x][0])
        arr1 = embed_mod2(coeffs_image[x][1], coeffs_watermark[x][1])
        arr2 = embed_mod2(coeffs_image[x][2], coeffs_watermark[x][2])
        dst.append((arr0, arr1, arr2))

    return dst


def w2d(img):
    model = 'haar'
    imageArray = convert_image(image, 256)
    watermarkArray = convert_image(watermark, 128)

    coeffs_image = process_coefficients(imageArray, model, level=2)
    coeffs_watermark = process_coefficients(watermarkArray, model, level=2)

    dst = embed_watermark(coeffs_image, coeffs_watermark)


# reconstruction
    imArray_H=pywt.waverec2(coeffs_image, model)
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

#Save result
    cv2.imwrite(current_path + '/result/image_with_watermark.jpg', imArray_H)


w2d("test")