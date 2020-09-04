# import the necessary packages
import numpy as np
import argparse
import cv2


def align_image(img):
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    cv2.imshow('grey',gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow('threshold', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    cord = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(cord)[-1]
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (height, width) = img.shape[:2]
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotate = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imshow("Input", img)
    cv2.imshow("Rotated", rotate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # drawing the correction angle on the image for validation:
    cv2.putText(rotate, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the output image
    print("angle: {:.3f}".format(angle))
    '''cv2.imshow("Input", img)
    cv2.imshow("Rotated", rotate)'''
    processed_(rotate, img)


# This function is used to show output
def processed_(x, y):
    cv2.imshow("Image", y)
    cv2.imshow("Rotated", x)
    cv2.imwrite("rotate.jpg", y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image file")
    args = vars(ap.parse_args())

    # load the image from disk
    image_ = cv2.imread(args["image"])
    align_image(image_)
