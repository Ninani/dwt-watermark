import numpy as np
import pywt
from PIL import Image
import os

current_path = str(os.path.dirname(__file__))

image = 'cat.jpg'
watermark = 'pani.jpg'
model = 'haar'
level = 3
k = 0.85

def convert_image(image, size):
    img = Image.open('../pictures/' + image).resize((size, size))
    img = img.convert('L')
    img.save('dataset/' + image)

    image_array = np.array(img)
    image_array = np.float32(image_array)
    image_array /= 255
    return image_array


def process_coefficients(imArray, model, level):
    coeffs = pywt.wavedec2(data=imArray, wavelet=model, level=level)
    coeffs_H = list(coeffs)

    return coeffs_H


def w2d(img):
    imageArray = convert_image(image, 256)
    watermarkArray = convert_image(watermark, 256)

    coeffs_image = process_coefficients(imageArray, model, level=2)
    coeffs_watermark = process_coefficients(watermarkArray, model, level=1)

    wmi = embed_watermark(coeffs_image, coeffs_watermark)

    imArray_H = pywt.waverec2(wmi, model)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    img = Image.fromarray(imArray_H)
    img.save(current_path + '/result/image_with_watermark.jpg')

    return imArray_H

def embed_watermark(coeffs_image, coeffs_watermark):
    Ph1, Qh1, Rh1 = np.linalg.svd(coeffs_image[2][0])
    Pw, Qw, Rw = np.linalg.svd(coeffs_watermark[1][0])

    Q_new = []

    len = Qh1.__len__()
    for i in xrange(len):
            Q_new.append(Qh1[i] + k*Qw[i])

    reconstr = np.zeros((len, len))
    reconstr[:len, :len] = np.diag(Q_new)

    new_HL = np.dot(np.dot(Ph1, reconstr), Rh1)

    return coeffs_image[0], \
           (coeffs_image[1][0],coeffs_image[1][1],coeffs_image[1][2]), \
           (np.array(new_HL), coeffs_image[2][1], coeffs_image[2][2])


def extract_wm(coeffs_image, coeffs_img_wm, coeffs_watermark):
    Ph1, Qh1, Rh1 = np.linalg.svd(coeffs_image[2][0])

    Pwm, Qwm, Rwm = np.linalg.svd(coeffs_img_wm[2][0])

    Q_new = []

    len = Qh1.__len__()
    for i in xrange(len):
        Q_new.append((Qwm[i] - Qh1[i])/k)

    reconstr = np.zeros((len, len))
    reconstr[:len, :len] = np.diag(Q_new)

    Pw, Qw, Rw = np.linalg.svd(coeffs_watermark[1][0])

    WM_HL = np.dot(np.dot(Pw, reconstr), Rw)

    return coeffs_watermark[0], (WM_HL, coeffs_watermark[1][1], coeffs_watermark[1][2])


def extract(watermarked):
    img = Image.open('./result/image_with_watermark.jpg')
    img = img.convert('L')
    img = np.float32(img)
    img /= 255

    img_wm = Image.fromarray(watermarked)
    img_wm = img_wm.convert('L')
    img_wm = np.float32(img_wm)
    img_wm /= 255

    coeffs_img = process_coefficients(img, model, 2)
    coeffs_img_wm = process_coefficients(img_wm, model, 2)

    watermarkArray = convert_image(watermark, 256)
    coeffs_watermark = process_coefficients(watermarkArray, model, level=1)
    extracted = extract_wm(coeffs_img, coeffs_img_wm, coeffs_watermark)

    imArray_H = pywt.waverec2(extracted, model)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    img = Image.fromarray(imArray_H)
    img.save('./result/extracted_wm.jpg')


watermarked = w2d("test")
extract(watermarked)
