import numpy as np
import pywt
import os
from PIL import Image
from scipy.fftpack import dct
from scipy.fftpack import idct

current_path = str(os.path.dirname(__file__))  

image = 'mis1.jpg'   
watermark = 'qrcode.png' 

def convert_image(image_name, size):
    img = Image.open('./pictures/' + image_name).resize((size, size))
    img = img.convert('L')
    img.save('./dataset/' + image_name)

    image_array = np.array(img)
    image_array = np.float32(image_array) 
    image_array /= 255 
    print image_array[0][0]               #qrcode white color = 1.0
    print image_array[10][10]             #qrcode black color = 0.0  

    return image_array

def process_coefficients(imArray, model, level):
    coeffs=pywt.wavedec2(data = imArray, wavelet = model, level = level)
    # print coeffs[0].__len__()
    coeffs_H=list(coeffs) 
   
    return coeffs_H


def embed_mod2(coeff_image, coeff_watermark, offset=0):
    for i in xrange(coeff_watermark.__len__()):
        for j in xrange(coeff_watermark[i].__len__()):
            coeff_image[i*2+offset][j*2+offset] = coeff_watermark[i][j]

    return coeff_image

def embed_mod4(coeff_image, coeff_watermark):
    for i in xrange(coeff_watermark.__len__()):
        for j in xrange(coeff_watermark[i].__len__()):
            coeff_image[i*4][j*4] = coeff_watermark[i][j]

    return coeff_image



def put_watermark_pixels(subdct, subwat):
    wat = np.array(subwat).ravel() 
    wat_ind = 0
    for i in xrange(8):
        for j in xrange(7-i, 7-i+4):
            if j >= 0 and j < 8 and wat_ind < wat.__len__():
                subdct[i][j] = wat[wat_ind]
                wat_ind += 1

    return subdct


            
    
def embed_watermark(watermark_array, orig_image):
    watermark_array_size = watermark_array[0].__len__()
    wat_len = (watermark_array_size)/(orig_image[0].__len__()/8)
    wat_size = wat_len * wat_len

    subwatermarks = [] 
    for i in range (0, watermark_array_size, wat_len):
        for j in range (0, watermark_array_size, wat_len):
            subwatermarks.append(watermark_array[i:i+wat_len, j:j+wat_len])

    subwatermarks_ind = 0


    for x in range (0, orig_image.__len__(), 8):
        for y in range (0, orig_image.__len__(), 8):
            if subwatermarks_ind < subwatermarks.__len__():
                subdct = orig_image[x:x+8, y:y+8]
                subwat = subwatermarks[subwatermarks_ind]
                orig_image[x:x+8, y:y+8] = put_watermark_pixels(subdct, subwat)
                subwatermarks_ind += 1 


    return orig_image
      


def apply_dct(image_array):
    size = image_array[0].__len__()
    all_subdct = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            subpixels = image_array[i:i+8, j:j+8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_subdct[i:i+8, j:j+8] = subdct

    return all_subdct


def inverse_dct(all_subdct):
    size = all_subdct[0].__len__()
    all_subidct = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            subidct = idct(idct(all_subdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
            all_subidct[i:i+8, j:j+8] = subidct

    return all_subidct



def get_embeded(coeff_watermarked, mod=4, size=64):
    watermark = [[1 for x in range(size)] for y in range(size)]
    print coeff_watermarked.__len__()
    for i in xrange(size):
        for j in xrange(size):
            watermark[i][j] = coeff_watermarked[i*mod][j*mod]
    
    return watermark


def extract_subwatermark(subdct):
    subwat_ind = 0
    subwat1D = []
    # print subdct[0].__len__()

    for x in range(8):
        for y in range(7-x, 7-x+4):
            if y >= 0 and y < 8 and subwat_ind < 16:
                subwat1D.append(subdct[x][y])
                subwat_ind += 1

    return subwat1D


def get_watermark(dct_watermarked_coeff, watermark_size):
    watermark = [[0 for x in range(watermark_size)] for y in range(watermark_size)] 

    subwatermarks = []
    subwatermarks_ind = 0

    for x in range (0, dct_watermarked_coeff.__len__(), 8):
        for y in range (0, dct_watermarked_coeff.__len__(), 8):
            subwatermark = extract_subwatermark(dct_watermarked_coeff[x:x+8, y:y+8])
            subwatermarks.append(subwatermark)

    for i in range (0, watermark_size, 4):
        for j in range (0, watermark_size, 4):
            ind = 0
            for m in range(i, i+4):
                for n in range(j, j+4):
                    watermark[m][n] = subwatermarks[subwatermarks_ind][ind]

            subwatermarks_ind += 1


    return watermark


def recover_watermark(image_array, model='haar', level = 1):
    # img = Image.open('./result/image_with_watermark.jpg')
    # img = img.convert('L')


    # image_array = np.array(img)
    # image_array = np.float32(image_array) 
    # image_array /= 255 


    coeffs_watermarked_image = process_coefficients(image_array, model, level=level)
    dct_watermarked_coeff = apply_dct(coeffs_watermarked_image[0])
    

    watermark_array = get_watermark(dct_watermarked_coeff, 128)

    # watermark_array *= 255;
    watermark_array =  np.uint8(watermark_array)

#Save result
    img = Image.fromarray(watermark_array)
    img.save('./result/recovered_watermark.jpg')


def print_image_from_array(image_array, name):
    image_array *= 255;
    image_array =  np.uint8(image_array)
    img = Image.fromarray(image_array)
    img.save('./result/' + name)



def w2d(img):
    model = 'haar'
    level = 1
    image_array = convert_image(image, 512)
    watermark_array = convert_image(watermark, 128)

    coeffs_image = process_coefficients(image_array, model, level=level)

    dct_array = apply_dct(coeffs_image[0])

    dct_array = embed_watermark(watermark_array, dct_array)

    coeffs_image[0] = inverse_dct(dct_array)


# reconstruction
    image_array_H=pywt.waverec2(coeffs_image, model)

    print_image_from_array(image_array_H, 'image_with_watermark.jpg')



# recover images
    recover_watermark(image_array = image_array_H, model=model, level = level)


w2d("test")