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
        None, (0, 1)
    )
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

    if mix_mod == 'add':
        # Add mode.
        # fa[+margins[0]+xh, +margins[1]+xw] += pb * alpha
        # fa[-margins[0]-xh, -margins[1]-xw] += pb * alpha

        fa[+margins[0]+xh, +margins[1]+xw] += pb * alpha
        fa[-margins[0]-xh, -margins[1]-xw] += pb * alpha
        fa[+margins[0]+xh, -margins[1]+xw] += pb * alpha
        fa[+margins[0]+xh, -margins[1]+xw] += pb * alpha
        fa[-margins[0]+xh, -margins[1]+xw] += pb * alpha
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
    font = ImageFont.truetype("consola.ttf", oa.shape[0] // 7)
    renderSize = font.getsize(text)
    padding = min(renderSize) * 2 // 10
    renderSize = (renderSize[0] + padding * 2, renderSize[1] + padding * 2)
    textImg = Image.new('RGB', renderSize, (0, 0, 0))
    draw = ImageDraw.Draw(textImg)
    draw.text((padding, padding), text, (255, 255, 255), font=font)
    ob = np.asarray(textImg)
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
        (new_y // 6, new_x // 6),
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
    argparser = argparse.ArgumentParser(
        description="Frequency domain image steganography/waterprint/signature"
    )
    argparser.add_argument(
        "input_image",
        metavar="file",
        type=str,
        help="Original image filename."
    )
    argparser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        help="Output filename."
    )
    argparser.add_argument(
        "-s",
        "--secret",
        dest="secret",
        type=str,
        help="Secret to generate index mapping.")
    # encode
    argparser.add_argument(
        "-i",
        "--image",
        dest="imagesign",
        type=str,
        help="Signature image filename."
    )
    argparser.add_argument(
        "-t",
        "--text",
        dest="textsign",
        type=str,
        help="Signature text."
    )
    argparser.add_argument(
        "-a",
        "--alpha",
        dest="alpha",
        type=float,
        help="Signature blending weight."
    )
    # decode
    argparser.add_argument(
        "-d",
        "--decode",
        dest="decode",
        type=str,
        help="Image filename to be decoded."
    )
    # other
    argparser.add_argument(
        "-v",
        dest="visual",
        action="store_true",
        default=False,
        help="Display image."
    )
    args = argparser.parse_args()

    oa = pyplot.imread(args.input_image)
    # oa = ndimage.rotate(oa, 45)
    margins = (oa.shape[0] // 7, oa.shape[1] // 7)
    margins = (1, 1)
    xmap = xmap_gen(
        (oa.shape[0]//2-margins[0]*2, oa.shape[1]-margins[1]*2),
        args.secret
    )

    if args.decode:
        xa = pyplot.imread(args.decode)
        xa = ajdust_decoded_image(oa, xa)
        xb = decode_image(xa, xmap, margins, oa)
        if args.output is None:
            base, ext = path.splitext(args.decode)
            args.output = base + "-" + path.basename(args.input_image) + ext
        imsaveEx(args.output, xb)
        print("Image saved.")
        if args.visual:
            pyplot.figure()
            imshow_ex(xb, title="deco")
            pyplot.show()
    else:
        if args.imagesign:
            ob = pyplot.imread(args.imagesign)
            ob_res = adjust_image_sizes(oa, ob, flag_plot=False)
            ea, fa = encode_image(oa, ob_res, xmap, margins, args.alpha)

            if args.output is None:
                base, ext = path.splitext(args.input_image)
                args.output = base + "+" + path.basename(args.imagesign) + ext
        elif args.textsign:
            ea, fa = encode_text(oa, args.textsign, xmap, margins, args.alpha)
            if args.output is None:
                base, ext = path.splitext(args.input_image)
                args.output = base + "_" + path.basename(args.textsign) + ext
        else:
            print("Neither image or text signature is not given.")
            exit(2)

        imsaveEx(args.output, ea)
        print("Image saved.")

        if args.visual:
            shows_processed_images(
                args.output,
                xmap,
                margins,
                oa
            )
