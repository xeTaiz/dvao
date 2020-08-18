from contextlib import contextmanager
from xml.dom import minidom
import sys, os
import numpy as np

@contextmanager
def hidden_prints():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

@contextmanager
def hidden_errors():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def normalize_minmax(x):
    return (x - x.min()) / (x.max() - x.min())

def read_inviwo_tf(fn):
    xmldoc = minidom.parse(str(fn))
    def parse_point(point):
        pos = float(point.getElementsByTagName('pos')[0].getAttribute('content'))
        opacity = float(point.getElementsByTagName('rgba')[0].getAttribute('w'))
        return pos, opacity
    points = sorted(map(parse_point, xmldoc.getElementsByTagName('Point')))
    l, r = points[0][1], points[-1][1]
    xp, yp = zip(*points)
    def apply_tf(x, normalize=False):
        if normalize: x = normalize_minmax(x)
        return np.interp(x, xp, yp, left=l, right=r)

    return apply_tf


__all__ = ['hidden_prints', 'hidden_errors', 'read_inviwo_tf', 'normalize_minmax']
