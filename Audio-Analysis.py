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


if __name__ == "__main__":

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

    input_image = 'gilete1.tif'
    oa = pyplot.imread(input_image)

    # # --------
    # # Test: get image and see info
    # oa_with_sound = pyplot.imread('gilete1-audio-crop-2.png')

    # oa_with_sound_flatten = oa_with_sound.flatten()

    # pyplot.plot(oa_with_sound_flatten)
    # pyplot.show()
    # # --------

    # --------
    # 1. Transform 2D data into one dimension
    # imshow_ex(oa, title="before")
    # pyplot.show()
    x_dim, y_dim, z_dim = oa.shape[0], oa.shape[1], oa.shape[2]

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
