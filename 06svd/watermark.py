import numpy as np
import pywt
import cv2
import os

current_path = str(os.path.dirname(__file__))

image = 'cat.jpg'
watermark = 'pani.jpg'
model = 'haar'
level = 3
k = 0.9
q = 0.009


def convert_image(image, size):
    imArray = cv2.imread(current_path + '/../pictures/' + image)

    imArray = cv2.resize(imArray, (size, size))
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(current_path + '/dataset/' + image, imArray)

    imArray = np.float32(imArray)
    imArray /= 255
    return imArray


def process_coefficients(imArray, model, level):
    coeffs = pywt.wavedec2(data=imArray, wavelet=model, level=level)
    coeffs_H = list(coeffs)

    return coeffs_H


def embed_mod2(coeff_image, coeff_watermark):
    for i in xrange(coeff_watermark.__len__()):
        for j in xrange(coeff_watermark[i].__len__()):
            coeff_image[i][j] = k * coeff_image[i][j] + q * coeff_watermark[i][j]

    return coeff_image


def embed_watermark(coeffs_image, coeffs_watermark):
    Ph1, Qh1, Rh1 = np.linalg.svd(coeffs_image[2][0])

    Pw, Qw, Rw = np.linalg.svd(coeffs_watermark[1][0])

    Q_new = []

    len = Qh1.__len__()
    for i in xrange(len):
            Q_new.append(Qh1[i] + 10*Qw[i])

    reconstr = np.zeros((len, len))
    reconstr = np.diag(Q_new)

    new_HL = np.dot(np.dot(Ph1, reconstr), Rh1)

    return coeffs_image[0], (coeffs_image[1][0],coeffs_image[1][1],coeffs_image[1][2]), (np.array(new_HL), coeffs_image[2][1], coeffs_image[2][2])


def w2d(img):
    imageArray = convert_image(image, 256)
    watermarkArray = convert_image(watermark, 256)

    coeffs_image = process_coefficients(imageArray, model, level=2)
    coeffs_watermark = process_coefficients(watermarkArray, model, level=1)

    wmi = embed_watermark(coeffs_image, coeffs_watermark)

    imArray_H = pywt.waverec2(wmi, model)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    cv2.imwrite(current_path + '/result/image_with_watermark.jpg', imArray_H)

    return imArray_H


def extract_wm(coeffs_image, coeffs_img_wm):
    Ph1, Qh1, Rh1 = np.linalg.svd(coeffs_image[2][0])

    Pw, Qw, Rw = np.linalg.svd(coeffs_img_wm[2][0])

    Q_new = []

    len = Qh1.__len__()
    for i in xrange(len):
        Q_new.append((Qw[i] - Qh1[i])/10)

    reconstr = np.diag(Q_new)

    return np.dot(np.dot(Pw, reconstr), Rw)


def extract(watermarked):
    img = cv2.imread('dataset/' + image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.float32(img)
    img /= 255

    # img_wm = cv2.imread(current_path + '/result/image_with_watermark.jpg')
    # img_wm = cv2.cvtColor(img_wm, cv2.COLOR_RGB2GRAY)
    # img_wm = np.float32(img_wm)
    # img_wm /= 255
    img_wm = watermarked

    coeffs_img = process_coefficients(img, model, 2)
    coeffs_img_wm = process_coefficients(img_wm, model, 2)
    extracted = extract_wm(coeffs_img, coeffs_img_wm)

    imArray_H = extracted
    # imArray_H = pywt.waverec2(extracted, model)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    cv2.imwrite(current_path + '/result/extracted_wm.jpg', imArray_H)


watermarked = w2d("test")
extract(watermarked)
