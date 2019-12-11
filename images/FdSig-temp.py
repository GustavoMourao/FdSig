# -*- coding: utf-8 -*-
"""
Created on Tue May  2 14:15:26 2017
Updated on Thu Dec  5 13:29:00 2019

@author: MidoriYakumo
@refactor1: GustavoMourao
"""

import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
from os import path
import matplotlib.pyplot as pyplot
import cv2
from scipy import ndimage
import scipy.io.wavfile
from pydub import AudioSegment
from pydub.playback import play


def normalized_rgb(img):
    """
    Normalize image.

    Args:
    ---------
        img: raw image

    Returns:
    ---------
        normalized image
    """
    if img.dtype == np.uint8:
        return img[:, :, :3] / 255.
    else:
        return img[:, :, :3].astype(np.float64)


def mix(a, b, u, keepImag=False):
    """
    Method that applies mix add mode mode on images.

    Args:
    ---------
        a:
        b:
        u:

    Returns:
    ---------
        mix coefs
    """
    if keepImag:
        return (a.real * (1 - u) + b.real * u) + a.imag * 1j
    else:
        return a * (1 - u) + b * u


def centralize(img, side=.06, clip=False):
    """
    """
    img = img.real.astype(np.float64)
    thres = img.size * side

    l = img.min()
    r = img.max()
    while l + 1 <= r:
        m = (l + r) / 2.
        s = np.sum(img < m)
        if s < thres:
            l = m
        else:
            r = m
    low = l

    l = img.min()
    r = img.max()
    while l + 1 <= r:
        m = (l + r) / 2.
        s = np.sum(img > m)
        if s < thres:
            r = m
        else:
            l = m

    high = max(low + 1, r)
    img = (img - low) / (high - low)

    if clip:
        img = np.clip(img, 0, 1)

    return img, low, high


def shuffle_gen(size, secret=None):
    """
    """
    r = np.arange(size)
    if secret:
        random.seed(secret)
        for i in range(size, 0, -1):
            j = random.randint(0, i)
            r[i-1], r[j] = r[j], r[i-1]
    return r


def xmap_gen(shape, secret=None):
    """
    """
    xh, xw = shuffle_gen(shape[0], secret), shuffle_gen(shape[1], secret)
    xh = xh.reshape((-1, 1))
    return xh, xw


def encode_image(oa, ob, xmap=None, margins=(1, 1), alpha=None, mix_mod='add'):
    """
    Encodes image ob into oa. First images are normalized.
    Then, original image (`oa`) will hide uncovered
    image (`ob`). To do that, `oa` is transform to frequency
    domain. After that is summed in each `oa` bin the respectivelly
    bins of `ob` image. Finally, the image is transform to spatial
    domain again.

    Args:
    ---------
        oa: original image
        ob: mask (hided) image/mark
        mix_mod: mode of add figures

    Returns:
    ---------
        encoded image (only visible into frequency domain)
    """
    na = normalized_rgb(oa)
    nb = normalized_rgb(ob)
    fa = np.fft.fft2(
        na,
        None,
        (0, 1)
    )
    # fa = np.fft.fftshift(
    #     fa,
    #     (0, 1)
    # )
    pb = np.zeros((na.shape[0]//2-margins[0]*2, na.shape[1]-margins[1]*2, 3))
    pb[:nb.shape[0], :nb.shape[1]] = nb

    # TODO: VERIFY ANULAR REGION TO MINIZE BORD EFFECT!
    low = 0
    if alpha is None:
        _, low, high = centralize(fa)
        alpha = (high - low)
        print("encode_image: alpha = {}".format(alpha))

    if xmap is None:
        xh, xw = xmap_gen(pb.shape)
    else:
        xh, xw = xmap[:2]

    if mix_mod == 'anular':
        # m, n = fa.shape[0], fa.shape[1]
        # ci = m//2 + 1
        # cj = (n//2) + 1
        # lr = m//4
        # hr = m//4 + 50
        # m = 1
        # for i in range(hr):
        #     for j in range(hr):
        #         r = np.sqrt(i*i + j*j)
        #         if ((r > lr) and (r < hr)):
        #             fa[i+ci, j+cj] += pb * alpha
        #             fa[i+ci, -j+cj] += pb * alpha
        #             fa[-i+ci, j+cj] += pb * alpha
        #             fa[-i+ci, -j+cj] += pb * alpha

        fa[+margins[0]+xh + 90, +margins[1]+xw + 90] += pb * alpha
        fa[-margins[0]-xh + 90, -margins[1]-xw + 90] += pb * alpha
        # fa[-margins[0]-xh + 1, -margins[1]-xw + 1] += pb * alpha
        # fa[+margins[0]+xh + 1, -margins[1]+xw + 1] += pb * alpha
        # fa[+margins[0]+xh + 1, -margins[1]+xw + 1] += pb * alpha

    if mix_mod == 'add':
        # Add mode.
        fa[+margins[0]+xh, +margins[1]+xw] += pb * alpha
        fa[-margins[0]-xh, -margins[1]-xw] += pb * alpha
    if mix_mod == 'mix':
        # mix mode
        fa[+margins[0]+xh, +margins[1]+xw] =\
            mix(fa[+margins[0]+xh, +margins[1]+xw], pb, alpha, True)
        fa[-margins[0]-xh, -margins[1]-xw] =\
            mix(fa[-margins[0]-xh, -margins[1]-xw], pb, alpha, True)
    if mix_mod == 'multiply':
        # multiply mode
        la = np.abs(fa[+margins[0]+xh, +margins[1]+xw])
        la[np.where(la < 1e-3)] = 1e-3
        fa[+margins[0]+xh, +margins[1]+xw] *= (la + pb * alpha) / la
        la = np.abs(fa[-margins[0]-xh, -margins[1]-xw])
        la[np.where(la < 1e-3)] = 1e-3
        fa[-margins[0]-xh, -margins[1]-xw] *= (la + pb * alpha) / la

    xa = np.fft.ifft2(fa, None, (0, 1))

    # Use real part.
    xa = xa.real
    xa = np.clip(xa, 0, 1)

    return xa, fa


def encode_text(oa, text, *args, **kwargs):
    """
    Encodes text into image.

    Args:
    ---------
        oa: original image
        text: text (string)

    Returns:
    ---------
        encoded image (only visible into frequency domain)
    """
    # font = ImageFont.truetype("arial.ttf", oa.shape[0] // 7)
    font = ImageFont.truetype("arial.ttf", oa.shape[0] // 5)
    renderSize = font.getsize(text)
    padding = min(renderSize) * 2 // 10
    renderSize = (renderSize[0] + padding * 2, renderSize[1] + padding * 2)
    textImg = Image.new('RGB', renderSize, (0, 0, 0))

    draw = ImageDraw.Draw(textImg)
    draw.text((padding, padding), text, (255, 255, 255), font=font)
    ob = np.asarray(textImg)

    # xa = pyplot.imread(decode)
    ob = adjust_image_sizes(oa, ob)

    # imshow_ex(
    #     ob,
    #     title="freq"
    # )
    # pyplot.show()

    return encode_image(oa, ob, *args, **kwargs)


def decode_image(xa, xmap=None, margins=(1, 1), oa=None, full=False):
    """
    Decodes image with hided mark (mask), only visible into frequency
    domain.

    Args:
    ---------
        xa: image with hided mark

    Returns:
    ---------
        dencoded image (frequency domain/spectra)
    """
    na = normalized_rgb(xa)
    fa = np.fft.fft2(na, None, (0, 1))
    imshow_ex(
        fa,
        title="freq"
    )
    pyplot.show()


    if xmap is None:
        xh = xmap_gen((xa.shape[0]//2-margins[0]*2, xa.shape[1]-margins[1]*2))
    else:
        xh, xw = xmap[:2]

    if oa is not None:
        noa = normalized_rgb(oa)
        foa = np.fft.fft2(noa, None, (0, 1))
        fa -= foa

    if full:
        nb, _, _ = centralize(fa, clip=True)
    else:
        nb, _, _ = centralize(fa[+margins[0]+xh, +margins[1]+xw], clip=True)
    return nb


def imshow_ex(img, *args, **kwargs):
    """
    Show image (space and frequency domain).

    Args:
    ---------
        img: image

    Returns:
    ---------
        show image
    """
    img, _, _ = centralize(img, clip=True)

    kwargs["interpolation"] = "nearest"
    if "title" in kwargs:
        pyplot.title(kwargs["title"])
        kwargs.pop("title")
    if len(img.shape) == 1:
        kwargs["cmap"] = "gray"
    pyplot.imshow(img, *args, **kwargs)


def imsaveEx(fn, img, *args, **kwargs):
    """
    Saves image.

    Args:
    ---------
        fn: image name
        img: image

    Returns:
    ---------
        saves image
    """
    kwargs["dpi"] = 1

    if img.dtype != np.uint8:
        print("Performing clamp and rounding")
        img, _, _ = centralize(img, clip=True)
        img = (img * 255).round().astype(np.uint8)

    pyplot.imsave(fn, img, *args, **kwargs)


def remove_channel(img):
    """
    In case of grayScale images the len(img.shape) == 2.
    Remove channel inconsistence (higher than 4).

    Args:
    ---------
        img: image

    Returns:
    ---------
        resized image0
    """
    if len(img.shape) > 2 and img.shape[2] == 4:
        # Convert the image from RGBA2RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def ajdust_decoded_image(oa, ob, flag_plot=False):
    """
    Adjust decoded mask image (that will be hided - `ob`) size
    based on principal image (`oa`). `ob` has to be smaller
    than `oa`.

    Args:
    ---------
        oa: original image
        ob: hided image

    Returns:
    ---------
        resized image
    """

    # Adjust number of channels.
    ob = remove_channel(ob)
    oa = remove_channel(oa)

    new_x = oa.shape[0]
    new_y = oa.shape[1]
    ob_res = cv2.resize(
        ob,
        (new_y // 1, new_x // 1),
        interpolation=cv2.INTER_AREA
    )

    if flag_plot:
        imshow_ex(ob_res)
        pyplot.show()

    return ob_res


def adjust_image_sizes(oa, ob, flag_plot=False):
    """
    Adjust mask image (that will be hided - `ob`) size
    based on principal image (`oa`). `ob` has to be smaller
    than `oa`.

    Args:
    ---------
        oa: original image
        ob: hided image

    Returns:
    ---------
        resized image
    """

    # Adjust number of channels.
    ob = remove_channel(ob)
    oa = remove_channel(oa)

    new_x = oa.shape[0]
    new_y = oa.shape[1]
    ob_res = cv2.resize(
        ob,
        # (new_y // 3, new_x // 3),
        (new_y // 16, new_x // 16),
        interpolation=cv2.INTER_AREA
    )

    if flag_plot:
        imshow_ex(ob_res)
        pyplot.show()

    return ob_res


def shows_processed_images(output, xmap, margins, oa):
    """
    Show sequence of processed images.

    Args:
    ---------
        oa: original image
        ob: hided image

    Returns:
    ---------
        resized image
    """
    xa = pyplot.imread(output)
    xb = decode_image(xa, xmap, margins, oa)

    pyplot.figure()
    pyplot.subplot(221)
    imshow_ex(
        ea,
        title="enco"
    )
    pyplot.subplot(222)
    imshow_ex(
        normalized_rgb(xa) - normalized_rgb(oa),
        title="delt"
    )
    pyplot.subplot(223)
    imshow_ex(
        fa,
        title="freq"
    )
    pyplot.subplot(224)
    imshow_ex(
        xb,
        title="deco"
    )
    pyplot.show()


if __name__ == "__main__" or True:
    """
    """
    # argparser = argparse.ArgumentParser(
    #     description="Frequency domain image steganography/waterprint/signature"
    # )
    # argparser.add_argument(
    #     "input_image",
    #     metavar="file",
    #     type=str,
    #     help="Original image filename."
    # )
    # argparser.add_argument(
    #     "-o",
    #     "--output",
    #     dest="output",
    #     type=str,
    #     help="Output filename."
    # )
    # argparser.add_argument(
    #     "-s",
    #     "--secret",
    #     dest="secret",
    #     type=str,
    #     help="Secret to generate index mapping.")
    # # encode
    # argparser.add_argument(
    #     "-i",
    #     "--image",
    #     dest="imagesign",
    #     type=str,
    #     help="Signature image filename."
    # )
    # argparser.add_argument(
    #     "-t",
    #     "--text",
    #     dest="textsign",
    #     type=str,
    #     help="Signature text."
    # )
    # argparser.add_argument(
    #     "-a",
    #     "--alpha",
    #     dest="alpha",
    #     type=float,
    #     help="Signature blending weight."
    # )
    # # decode
    # argparser.add_argument(
    #     "-d",
    #     "--decode",
    #     dest="decode",
    #     type=str,
    #     help="Image filename to be decoded."
    # )
    # # other
    # argparser.add_argument(
    #     "-v",
    #     dest="visual",
    #     action="store_true",
    #     default=False,
    #     help="Display image."
    # )
    # args = argparser.parse_args()

    # input_image = 'coca.jpg'
    input_image = 'gilete1.tif'
    secret = 'TEXTO INSERIDO'
    # secret = None
    # output = 'pepsi_coca_test_.jpg'
    output = 'gilete1-simplus-v1-test.jpg'
    # output = 'gilete1-simplus-v1-test-central-v2.jpg'
    # imagesign = 'pepsi.png'
    # imagesign = 'dot-matrix.png'
    imagesign = 'matrix-3.png'
    # imagesign = None
    # imagesign = 'simplus.png'
    # textsign = 'TEXTO INSERIDO'
    textsign = None
    # alpha = 50
    alpha = None
    decode = 'gilete1-simplus-v1-test.jpg'
    # decode = 'gilete1-simplus-v1-test.png'
    # decode = False
    visual = True
    oa = pyplot.imread(input_image)

    oa = pyplot.imread(input_image)
    # oa = ndimage.rotate(oa, 45)
    margins = (oa.shape[0] // 7, oa.shape[1] // 7)
    # margins = (1, 1)
    xmap = xmap_gen(
        (oa.shape[0]//2-margins[0]*2, oa.shape[1]-margins[1]*2),
        secret
    )

    # --------
    # Test: get image and see info
    oa_with_sound = pyplot.imread('gilete1-audio-crop-2.png')

    oa_with_sound_flatten = oa_with_sound.flatten()

    pyplot.plot(oa_with_sound_flatten)
    pyplot.show()
    # --------

    # --------
    # 1. Transform 2D data into one dimension
    # imshow_ex(oa, title="before")
    # pyplot.show()
    x_dim, y_dim, z_dim = oa.shape[0], oa.shape[1], oa.shape[2]
    print(type(oa))

    # Read sound to be added
    # sound = AudioSegment.from_wav(file='F051710.WAV')
    sound = scipy.io.wavfile.read('F051710.WAV')
    # play(sound)
    sound = sound[1]
    # pyplot.plot(sound)
    # pyplot.show()
    sound = np.abs(sound)

    oa_flatten = oa.flatten()

    sound = sound/np.max(oa_flatten)

    # Substituir os N últimos caracteres pelo áudio
    counter_sound = 0
    for i in range(len(oa_flatten)):
        if i < (len(oa_flatten) - len(sound)):
            oa_flatten[i] = oa_flatten[i]
        else:
            oa_flatten[i] = sound[counter_sound]
            counter_sound += 1

    # # Adicionar no meio da sequencia o audio
    # for i in range(len(oa_flatten)):
    #     if i < (len(oa_flatten)//2):

    # oa_resulted = np.concatenate((
    #     oa_flatten,
    #     sound
    # ), axis=None)
    # oa_resulted.flatten()

    oa_resulted = oa_flatten

    pyplot.plot(oa_resulted)
    pyplot.show()
    scipy.io.wavfile.write('image_sound.wav', 20000, oa_resulted)

    oa_back = oa_resulted.reshape(
        x_dim,
        y_dim,
        z_dim
    )
    imshow_ex(oa_back, title="after")
    pyplot.show()

    # SALVAR IMAGEM COM ÁUDIO!
    imsaveEx('gilete1-com-audio.jpg', oa_back)

    # 2. Get back to 2D

    # --------

    if decode:
        xa = pyplot.imread(decode)

        fa = np.fft.fft2(
            xa,
            None,
            (0, 1)
        )
        fa_imag = np.real(fa)
        pyplot.figure()
        imshow_ex(fa_imag, title="freq-decode")
        pyplot.show()

        xa = ajdust_decoded_image(oa, xa)
        xb = decode_image(xa, xmap, margins, oa)
        if output is None:
            base, ext = path.splitext(decode)
            output = base + "-" + path.basename(input_image) + ext
        imsaveEx(output, xb)
        print("Image saved.")
        if visual:
            pyplot.figure()
            imshow_ex(xb, title="deco")

            # 1. Padronize figure
            # xb = np.uint8(xb)
            xb = (xb*255).astype(np.uint8)

            # 2. Apply median filter
            xb = cv2.medianBlur(xb,5)
            xb = cv2.GaussianBlur(xb,(5,5),0)
            xb = cv2.GaussianBlur(xb,(7,7),0)
            xb = cv2.GaussianBlur(xb,(7,7),0)
            # xb = cv2.bilateralFilter(xb,9,75,75)

            # 3. Apply highpass filter
            # xb = cv2.Laplacian(xb,cv2.CV_64F)
            # xb = cv2.Sobel(xb,cv2.CV_64F,1,0,ksize=5)
            # xb = cv2.Sobel(xb,cv2.CV_64F,0,1,ksize=5)
            # xb = (xb*255).astype(np.uint8)

            # 3. Apply Canny at filter
            edges = cv2.Canny(xb,50,50)

            pyplot.subplot(121),pyplot.imshow(xb,cmap = 'gray')
            pyplot.title('Original Image'), pyplot.xticks([]), pyplot.yticks([])
            pyplot.subplot(122),pyplot.imshow(edges,cmap = 'gray')

            pyplot.show()
    else:
        if imagesign:
            ob = pyplot.imread(imagesign)
            ob_res = adjust_image_sizes(oa, ob, flag_plot=False)
            ea, fa = encode_image(oa, ob_res, xmap, margins, alpha)

            if output is None:
                base, ext = path.splitext(input_image)
                output = base + "+" + path.basename(imagesign) + ext
        elif textsign:
            ea, fa = encode_text(oa, textsign, xmap, margins, alpha)
            if output is None:
                base, ext = path.splitext(input_image)
                output = base + "_" + path.basename(textsign) + ext
        else:
            print("Neither image or text signature is not given.")
            exit(2)

        imsaveEx(output, ea)
        print("Image saved.")

        if visual:
            shows_processed_images(
                output,
                xmap,
                margins,
                oa
            )
