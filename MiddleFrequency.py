import numpy as np
from matplotlib import pyplot


if __name__ == "__main__":
    """
    """
    input_image = 'coca.jpg'
    img = pyplot.imread(input_image)
    img_f = np.fft.fft2(img)
    img_f_shift = np.fft.fftshift(img_f)

    pyplot.imshow(img)
    pyplot.show()

    m, n = img_f.shape[0], img_f.shape[1]
    ci = m//2 + 1
    cj = (n//2) + 1

    # Dimensions of annular region for hiding.
    # In case of other image type
    lr = m//4
    hr = m//4 + 50
    message = 'Text for text evaluation'
    message_count = string.count(substring)
    mlen = len(str('Text for text evaluation'))
    m_ = 1

    for i in range(hr):
        for j in range(hr):
            r = np.sqrt(i*i + j*j)
            if (r > lr) and (r < hr):
                # if (m_ <= mlen):
                    # img_f_shift[i+ci, j+cj] =\
                    #     ('Text for text evaluation').split()[m_]
                    img_f_shift[i+ci, -j+cj] = img_f_shift[i+ci, j+cj]
                    img_f_shift[-i+ci, j+cj] = img_f_shift[i+ci, j+cj]
                    img_f_shift[-i+ci, -j+cj] = img_f_shift[i+ci, j+cj]
                    m_ = m_ + 1

    img_rec = np.fft.ifft2(img_f_shift, None, (0, 1))

    # Use real part.
    img_rec = img_rec.real
    img_rec = np.clip(img_rec, 0, 1)

    pyplot.imshow(np.abs(img_rec))
    pyplot.show()

    print(type(img_f_shift))
