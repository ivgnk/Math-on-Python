'''
Based on
https://docs.scipy.org/doc/scipy/reference/signal.html#filtering
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html#scipy.signal.wiener
'''
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def ref_2D_wienerfilt():
    '''
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html#scipy.signal.wiener
    '''
    from scipy.datasets import face
    from scipy.signal import wiener
    import matplotlib.pyplot as plt
    import numpy as np
    rng = np.random.default_rng()
    img = rng.random((40, 40))  # Create a random image
    filtered_img = wiener(img, (5, 5))  # Filter the image
    f, (plot1, plot2) = plt.subplots(1, 2)
    plot1.imshow(img)
    plot2.imshow(filtered_img)
    plt.show()

def my_1D_wienerfilt():
    # https://numpy.org/doc/stable/reference/random/index.html
    rng = np.random.default_rng() # Generator start
    x = rng.standard_normal(250)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.plot(x,label='ini')
    lst = [5, 15, 25, 35]  # range(5,30,8)
    for i in lst:
        print(i)
        y = signal.wiener(x, i)
        plt.plot(y,label='wiener '+str(i))
    # y = signal.medfilt(x, 15)
    # plt.plot(y)
    plt.legend()
    plt.grid()
    plt.show()

# ref_2D_wienerfilt()
my_1D_wienerfilt()
