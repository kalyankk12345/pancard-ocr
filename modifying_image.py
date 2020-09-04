"""
This function is used to morph images
changing operation
changing structure element shape to ellipse(desired)
"""
from __future__ import print_function
import cv2 as cv


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def modify(img, modes, str_modes):
    sz = 10
    itn = 1
    opern = modes.split('/')
    sz = sz - 10
    op = opern[sz > 0]
    sz = sz * 2 + 1

    str_name = 'MORPH_' + str_modes.upper()
    operation_name = 'MORPH_' + op.upper()
    st = cv.getStructuringElement(getattr(cv, str_name), (sz, sz))
    res = cv.morphologyEx(img, getattr(cv, operation_name), st, iterations=itn)

    draw_str(res, (10, 20), 'mode: ' + modes)
    draw_str(res, (10, 40), 'operation: ' + operation_name)
    draw_str(res, (10, 60), 'structure: ' + str_name)
    draw_str(res, (10, 80), 'ksize: %d  iterations: %d' % (sz, itn))
    cv.imshow('morphology', res)
    cv.imwrite('result.jpg', img)


if __name__ == '__main__':
    # print(__doc__)  # re-uses documentation string and writes to terminal each time we run the script
    import sys

    try:
        fn = sys.argv[1]
    except:
        fn = '16.jpg'
    img_ = cv.imread(fn)
    if img_ is None:
        print('Failed to load image file:', fn)
        sys.exit(1)
    modes_ = 'erode/dilate'
    str_modes_ = 'ellipse'

    cv.namedWindow('morphology')
    cv.imshow('morphology',img_)
    cv.waitKey(0)
    modify(img_, modes_, str_modes_)
    cv.waitKey(0)
    cv.destroyAllWindows()
