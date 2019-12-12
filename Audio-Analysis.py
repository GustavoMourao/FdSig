import argparse
import random
import numpy as np
import matplotlib.pyplot as pyplot
import scipy.io.wavfile


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


def read_signals_return_data(input_image):
    """
    Read signals and modulates covered pattern/signal.

    Args:
    ---------
        input_image: souce name of image

    Returns:
    ---------
        img: image (numpy.ndarray)
        sound: signal/sound to be covered
        imgx: x dim of image
        imgy: y dim of image
        imgz: z dim of image
    """
    img = pyplot.imread(input_image)
    imgx, imgy, imgz = img.shape[0], img.shape[1], img.shape[2]

    # TODO: CREATE SIN WAVE!
    signal = scipy.io.wavfile.read('images/F051710.WAV')
    signal = np.abs(signal[1])

    return img, signal, imgx, imgy, imgz


def image_colapses_normalize_signal(img, signal):
    """
    Colapses image into one dimension axis and normalizes.
    signal to be covered.

    Args:
    ---------
        img: image
        signal: signal

    Returns:
    ---------
        img_flatten: image flatten
        signal: normalized signal
    """
    img_flatten = img.flatten()
    signal = signal/np.max(img_flatten)

    return img_flatten, signal


def cover_information(img_flatten, signal):
    """
    Cover signal information into image at the end of signal.

    Args:
    ---------
        img_flatten: image as flatten mode
        signal: signal

    Returns:
    ---------
        img_flatten: image flatten with cover pattern/signal
    """
    counter_sound = 0
    for i in range(len(img_flatten)):
        if i < (len(img_flatten) - len(sound)):
            img_flatten[i] = img_flatten[i]
        else:
            img_flatten[i] = signal[counter_sound]
            counter_sound += 1

    return img_flatten


def encoded_image(img_flatten, imgx, imgy, imgz):
    """
    Converts encripted flatten image as an encripted image.

    Args:
    ---------
        img_flatten: image in flatten mode
        imgx: x dim of original image
        imgy: y dim of original image
        imgz: z dim of original image

    Returns:
    ---------
        encoded image
    """
    return img_flatten.reshape(
        imgx,
        imgy,
        imgz
    )


def decode_image(img_coded):
    """
    Get flatten image and shows the encoded signal.

    Args:
    ---------
        img_coded: source name of image with code to be verified

    Returns:
    ---------
        plots image with signal encripted
    """
    # TODO: CREATES A TEST BASED ON SIGNAL FREQUENCY
    oa_with_sound = pyplot.imread(img_coded)
    oa_with_sound_flatten = oa_with_sound.flatten()

    pyplot.plot(oa_with_sound_flatten)
    pyplot.title('Decoded image as signal')
    pyplot.show()


if __name__ == "__main__":

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

    if args.decode:
        decode_image(args.decode)
    elif args.output:
        oa, sound, x_dim, y_dim, z_dim = read_signals_return_data(
            args.input_image
        )

        oa_flatten, sound = image_colapses_normalize_signal(oa, sound)

        pyplot.subplot(121)
        pyplot.plot(oa_flatten)
        pyplot.title('Original Image')

        oa_resulted = cover_information(oa_flatten, sound)
        oa_back = encoded_image(oa_resulted, x_dim, y_dim, z_dim)

        imsaveEx(args.output, oa_back)

        pyplot.subplot(122)
        pyplot.plot(oa_resulted)
        pyplot.title('Original Image with Cover Info')
        pyplot.show()

    # # Test.
    # input_image = 'images/gilete1.tif'

    # oa, sound, x_dim, y_dim, z_dim = read_signals_return_data(input_image)

    # oa_flatten, sound = image_colapses_normalize_signal(oa, sound)

    # oa_flatten = cover_information(oa_flatten, sound)

    # oa_resulted = cover_information(oa_flatten, sound)

    # # pyplot.plot(oa_resulted)
    # # pyplot.show()
    # # scipy.io.wavfile.write('image_sound.wav', 20000, oa_resulted)

    # oa_back = encoded_image(oa_resulted, x_dim, y_dim, z_dim)

    # imshow_ex(oa_back, title="after")
    # pyplot.show()

    # # SALVAR IMAGEM COM ÁUDIO!
    # imsaveEx('gilete1-com-audio.jpg', oa_back)
