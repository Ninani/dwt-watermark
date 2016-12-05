import numpy as np
import pywt
import cv2
import os

current_path = str(os.path.dirname(__file__))

image = 'cat.jpg'
watermark = 'rings.jpg'
model = 'haar'
level = 3
k = 0.9
q = 0.0009


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
    # dst = []
    # arr0 = embed_mod2(coeffs_image[0], coeffs_watermark[0])
    # dst.append(arr0)
    #
    # for x in xrange(1, coeffs_image.__len__()):
    #     arr0 = embed_mod2(coeffs_image[x][0], coeffs_watermark[x][0])
    #     arr1 = embed_mod2(coeffs_image[x][1], coeffs_watermark[x][1])
    #     arr2 = embed_mod2(coeffs_image[x][2], coeffs_watermark[x][2])
    #     dst.append((arr0, arr1, arr2))
    #
    # return dst

    LL3 = coeffs_image[0]
    WM3 = coeffs_watermark[0]
    len = coeffs_image[0].__len__()
    # WMI = np.zeros((len, len))

    for x in xrange(len):
        for y in xrange(len):
            coeffs_image[0][x][y] = k * LL3[x][y] + q * WM3[x][y]

    return coeffs_image


def w2d(img):
    imageArray = convert_image(image, 256)
    watermarkArray = convert_image(watermark, 256)

    coeffs_image = process_coefficients(imageArray, model, level=level)
    coeffs_watermark = process_coefficients(watermarkArray, model, level=level)

    wmi = embed_watermark(coeffs_image, coeffs_watermark)

    imArray_H = pywt.waverec2(wmi, model)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    cv2.imwrite(current_path + '/result/image_with_watermark.jpg', imArray_H)

    return imArray_H


def reverse_blend(coeff_image, coeff_watermark):
    len = coeff_watermark.__len__()
    for i in xrange(len):
        for j in xrange(coeff_watermark[i].__len__()):
            coeff_image[i][j] = coeff_watermark[i][j] - k * coeff_image[i][j]

    return coeff_image


def extract_wm(coeffs_img, coeffs_img_wm):
    # dst = []
    #
    # arr0 = reverse_blend(coeffs_img[0], coeffs_img_wm[0])
    # dst.append(arr0)
    #
    # for x in xrange(1, coeffs_img.__len__()):
    #     arr0 = reverse_blend(coeffs_img[x][0], coeffs_img_wm[x][0])
    #     arr1 = reverse_blend(coeffs_img[x][1], coeffs_img_wm[x][1])
    #     arr2 = reverse_blend(coeffs_img[x][2], coeffs_img_wm[x][2])
    #     dst.append((arr0, arr1, arr2))
    #
    # return dst

    LL3 = coeffs_img[0]
    WM3 = coeffs_img_wm[0]
    len = coeffs_img[0].__len__()
    watermarkArray = convert_image(watermark, 256)
    coeffs_watermark = process_coefficients(watermarkArray, model, level=level)
    # WMI = np.zeros((len, len))

    for x in xrange(len):
        for y in xrange(len):
            coeffs_watermark[0][x][y] = (WM3[x][y] - k * LL3[x][y])/q

    return coeffs_watermark


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

    coeffs_img = process_coefficients(img, model, level)
    coeffs_img_wm = process_coefficients(img_wm, model, level)
    extracted = extract_wm(coeffs_img, coeffs_img_wm)

    imArray_H = pywt.waverec2(extracted, model)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    cv2.imwrite(current_path + '/result/extracted_wm.jpg', imArray_H)


watermarked = w2d("test")
extract(watermarked)
