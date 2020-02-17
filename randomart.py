import numpy as np
import click
import hashlib
import cv2
import sys

NONLINEAR_TABLE = {
    "log": lambda x: np.log(np.abs(x) + 0.01),
    "xlog": lambda x: x * np.log(np.abs(x) + 0.01),
    "tanh": lambda x: np.tanh(x),
    "log_p1": lambda x: np.log(np.abs(x) + 1),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "log_x2": lambda x: np.log(x**2 + 0.01),
    "log_x2_p1": lambda x: np.log(x**2 + 1),
    "sin": lambda x: np.sin(x),
    "abs_tanh": lambda x: np.abs(np.tanh(x)),
    "softplus": lambda x: np.log(1 + np.exp(x)),
    "softsign": lambda x: x / (1 + np.abs(x)),
    "identity": lambda x: x,
}


def bn(x, axis=1):
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    x = (x - mean) / std
    return x


def global_bn(x):
    mean = np.mean(x)
    std = np.std(x)
    x = (x - mean) / std
    return x


def fully_connected(x,
                    weight,
                    bias=None,
                    normalize=True,
                    global_normalize=False):
    x = np.matmul(x, weight)
    if bias is not None:
        x += bias
    if global_normalize:
        x = global_bn(x)
    elif normalize:
        x = bn(x)
    return x


def sha256(b):
    h = hashlib.sha256()
    h.update(b)
    s = h.digest()
    return list(map(int, s))


def mesh(w, h, t=1):
    ww = np.arange(start=1, stop=w + 1, step=1) / w
    hh = np.arange(start=1, stop=h + 1, step=1) / h
    x, y = np.meshgrid(ww, hh)
    coord = np.concatenate(
        (y.flatten()[:, np.newaxis], x.flatten()[:, np.newaxis]), axis=1)
    coord = np.concatenate((coord, np.ones((coord.shape[0], 1)) * t), axis=1)
    return coord


def type1(rng, width, height, t=1):
    LAYERS = [42, 42, 42, 42, 42, 3]
    NONLINEAR = ["sigmoid", "log", "sin", "xlog", "sin", "identity"]

    x = mesh(width, height, t=t).astype('float64')
    batch_size = x.shape[0]

    before = LAYERS[0]
    weight = rng.random(size=(3, before), dtype='d')
    x = fully_connected(x, weight, global_normalize=True)
    weight = rng.random(size=(before, before), dtype='d')
    x = fully_connected(x, weight) * 0.001

    for k in range(len(LAYERS)):
        after = LAYERS[k]
        weight = bn(rng.random(size=(before, after), dtype='d'))
        x = fully_connected(x, weight, normalize=False)
        x = NONLINEAR_TABLE[NONLINEAR[k]](x)
        before = after

    return x


def type2(rng, width, height, t=1):
    LAYERS = [30, 30, 30, 30, 3]
    NONLINEAR = ["sigmoid", "sin", "log", "identity", "identity"]

    x = mesh(width, height, t=t).astype('float64')
    batch_size = x.shape[0]
    before = 3

    for k in range(len(LAYERS)):
        after = LAYERS[k]
        weight = bn(rng.random(size=(before, after), dtype='d'))
        x = fully_connected(x, weight, normalize=False)
        x = NONLINEAR_TABLE[NONLINEAR[k]](x)
        before = after

    return x


def type3(rng, width, height, t=1):
    LAYERS = [30, 30, 30, 30, 3]
    NONLINEAR = ["sigmoid", "sin", "xlog", "sin", "abs_tanh"]

    x = mesh(width, height, t=t).astype('float64')
    batch_size = x.shape[0]
    before = 3

    for k in range(len(LAYERS)):
        after = LAYERS[k]
        weight = bn(rng.random(size=(before, after), dtype='d'))
        x = fully_connected(x, weight, normalize=False)
        x = NONLINEAR_TABLE[NONLINEAR[k]](x)
        before = after

    return x


def type3a(rng, width, height, t=1):
    LAYERS = [30, 30, 30, 30, 3]
    NONLINEAR = ["sigmoid", "sin", "xlog", "sin", "tanh"]

    x = mesh(width, height, t=t).astype('float64')
    batch_size = x.shape[0]
    before = 3

    for k in range(len(LAYERS)):
        after = LAYERS[k]
        weight = bn(rng.random(size=(before, after), dtype='d'))
        x = fully_connected(x, weight, normalize=False)
        x = NONLINEAR_TABLE[NONLINEAR[k]](x)
        before = after

    return x


def type4(rng, width, height, t=1):
    LAYERS = [30, 30, 30, 30, 30, 3]
    NONLINEAR = ["sigmoid", "sin", "log_x2_p1", "tanh", "softsign", "identity"]

    x = mesh(width, height, t=t).astype('float64')
    batch_size = x.shape[0]
    before = 3

    for k in range(len(LAYERS)):
        after = LAYERS[k]
        weight = bn(rng.random(size=(before, after), dtype='d'))
        x = fully_connected(x, weight, normalize=False)
        x = NONLINEAR_TABLE[NONLINEAR[k]](x)
        before = after

    return x


FUNCTIONS = {
    "1": type1,
    "2": type2,
    "3": type3,
    "3a": type3a,
    "4": type4,
}


@click.command()
@click.option("-i", "--input", help="Input string", type=str, default=None)
@click.option("-f", "--file", help="Input file", type=str, default=None)
@click.option("-w", "--width", help="Image width", type=int, default=256)
@click.option("-h", "--height", help="Image height", type=int, default=256)
@click.option("-t",
              "--randomtype",
              help="Randomart Type",
              type=str,
              default="1")
@click.option("-o",
              "--output",
              help="Output image path",
              type=str,
              default="output.png")
@click.option("-m",
              "--timedim",
              help="Value of time dimension",
              type=float,
              default=1.0)
def main(input, file, width, height, randomtype, output, timedim):
    if input is None and file is None:
        print("Input string or file must be indicated")
        sys.exit(2)

    b = b''
    if input is not None:
        b = input.encode('utf-8')
    else:
        with open(file, "rb") as f:
            b = f.read()

    rng = np.random.Generator(np.random.MT19937(sha256(b)))

    x = FUNCTIONS[randomtype](rng, width, height, t=timedim)

    pixels = ((x - np.min(x)) / (np.max(x) - np.min(x) + 1e-08) *
              255).astype('uint8')
    pixels = pixels.reshape((height, width, 3))
    cv2.imwrite(output, pixels)


if __name__ == "__main__":
    main()
