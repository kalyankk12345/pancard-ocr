import glob
import os
import random
import sys
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage.filters import rank_filter


def dilate(ary, N, iterations):
    """Dilate using an NxN '+' sign shape. ary is np.uint8."""

    kernel = np.zeros((N, N), dtype=np.uint8)
    kernel[(N - 1) // 2, :] = 1  # Bug solved with // (integer division)

    dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations)

    kernel = np.zeros((N, N), dtype=np.uint8)
    kernel[:, (N - 1) // 2] = 1  # Bug solved with // (integer division)
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
    return dilated_image


def props_for_contours(contours, ary):
    """Calculate bounding box & the number of set pixels for each contour."""
    c_info = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        c_im = np.zeros(ary.shape)
        cv2.drawContours(c_im, [c], 0, 255, -1)
        c_info.append({
            'x1': x,
            'y1': y,
            'x2': x + w - 1,
            'y2': y + h - 1,
            'sum': np.sum(ary * (c_im > 0)) / 255
        })
    return c_info


def union_crops(crop1, crop2):
    """Union two (x1, y1, x2, y2) rects."""
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)


def intersect_crops(crop1, crop2):
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)


def crop_area(crop):
    x1, y1, x2, y2 = crop
    return max(0, x2 - x1) * max(0, y2 - y1)


def find_border_components(contours, ary):
    # finding co-ordinates of bounding boxes
    borders = []
    area = ary.shape[0] * ary.shape[1]
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 0.5 * area:
            borders.append((i, x, y, x + w - 1, y + h - 1))
    return borders


# Rounding off to nearest right angle
def angle_from_right(deg):
    return min(deg % 90, 90 - (deg % 90))


def remove_b(contour, ary):
    # Remove everything outside a border contour
    c_im = np.zeros(ary.shape)
    r = cv2.minAreaRect(contour)
    degrees = r[2]
    if angle_from_right(degrees) <= 10.0:
        box = cv2.boxPoints(r)  # returns 4 points in the form of 2 tuples
        box = np.int0(box)
        cv2.drawContours(c_im, [box], 0, 255, -1)
        cv2.drawContours(c_im, [box], 0, 0, 4)
    else:
        x1, y1, x2, y2 = cv2.boundingRect(contour)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)

    return np.minimum(c_im, ary)  # returns minimum element array


def find_comp(edges, max_components=16):
    # Dilating the image till few components are left and returning contours
    # for these comppnents
    count = 21
    dilation = 5
    n = 12
    while count > 16:
        n += 1
        dilated_img = dilate(edges, N=3, iterations=n)
        dilated_img = np.uint8(dilated_img)
        contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
    return contours


def optimal_comp(contours, edges):
    # Returns an (x1, y1, x2, y2) tuple.

    c_info = props_for_contours(contours, edges)
    c_info.sort(key=lambda x: -x['sum'])
    total = np.sum(edges) / 255
    area = edges.shape[0] * edges.shape[1]

    c = c_info[0]
    del c_info[0]
    this_crop = c['x1'], c['y1'], c['x2'], c['y2']
    crop = this_crop
    covered_sum = c['sum']

    while covered_sum < total:
        changed = False
        recall = 1.0 * covered_sum / total
        prec = 1 - 1.0 * crop_area(crop) / area
        f1 = 2 * (prec * recall / (prec + recall))

        for i, c in enumerate(c_info):
            this_crop = c['x1'], c['y1'], c['x2'], c['y2']
            new_crop = union_crops(crop, this_crop)
            new_sum = covered_sum + c['sum']
            new_recall = 1.0 * new_sum / total
            new_prec = 1 - 1.0 * crop_area(new_crop) / area
            new_f1 = 2 * new_prec * new_recall / (new_prec + new_recall)
            remaining_frac = c['sum'] / (total - covered_sum)
            new_area_frac = 1.0 * crop_area(new_crop) / crop_area(crop) - 1
            if new_f1 > f1 or (
                    remaining_frac > 0.25 and new_area_frac < 0.15):
                print('%d %s -> %s / %s (%s), %s -> %s / %s (%s), %s -> %s' % (
                    i, covered_sum, new_sum, total, remaining_frac,
                    crop_area(crop), crop_area(new_crop), area, new_area_frac,
                    f1, new_f1))
                crop = new_crop
                covered_sum = new_sum
                del c_info[i]
                changed = True
                break

        if not changed:
            break

    return crop


def pad_crop(crop, contours, edges, border_contour, pad_px=15):
    # Expanding the crop to get full contours,including contours intersecting
    # and excluding those that expand past a border.

    bx1, by1, bx2, by2 = 0, 0, edges.shape[0], edges.shape[1]
    if border_contour is not None and len(border_contour) > 0:
        c = props_for_contours([border_contour], edges)[0]
        bx1, by1, bx2, by2 = c['x1'] + 5, c['y1'] + 5, c['x2'] - 5, c['y2'] - 5

    def crop_border(crop):
        x1, y1, x2, y2 = crop
        x1 = max(x1 - pad_px, bx1)
        y1 = max(y1 - pad_px, by1)
        x2 = min(x2 + pad_px, bx2)
        y2 = min(y2 + pad_px, by2)
        return crop

    crop = crop_border(crop)

    c_info = props_for_contours(contours, edges)
    changed = False
    for c in c_info:
        this_crop = c['x1'], c['y1'], c['x2'], c['y2']
        this_area = crop_area(this_crop)
        int_area = crop_area(intersect_crops(crop, this_crop))
        new_crop = crop_border(union_crops(crop, this_crop))
        if 0 < int_area < this_area and crop != new_crop:
            print('%s -> %s' % (str(crop), str(new_crop)))
            changed = True
            crop = new_crop

    if changed:
        return pad_crop(crop, contours, edges, border_contour, pad_px)
    else:
        return crop


def downscale_image(image, max_dimension=2048):
    # Resizing the dimensions to <= 2048
    # Returns scale(wrt 2048) and image
    width, height = image.size
    if max(width, height) <= max_dimension:
        return 1.0, image

    scale = 1.0 * max_dimension / max(width, height)
    new_im = image.resize((int(width * scale), int(height * scale)), Image.ANTIALIAS)
    return scale, new_im


def process_image(path, out_path):
    orig_im = Image.open(path)
    scale, im = downscale_image(orig_im)

    edges = cv2.Canny(np.asarray(im), 100, 200)

    # Dilating image before finding a border.
    major = cv2.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    borders = find_border_components(contours, edges)

    borders.sort(
        key=lambda i_x1_y1_x2_y2: (i_x1_y1_x2_y2[3] - i_x1_y1_x2_y2[1]) * (i_x1_y1_x2_y2[4] - i_x1_y1_x2_y2[2]))

    border_contour = None
    if len(borders):
        border_contour = contours[borders[0][0]]
        edges = remove_b(border_contour, edges)

    edges = 255 * (edges > 0).astype(np.uint8)

    # Remove  borders using a rank filter.

    max_rows = rank_filter(edges, -4, size=(1, 20))
    max_cols = rank_filter(edges, -4, size=(20, 1))
    deborder = np.minimum(np.minimum(edges, max_rows), max_cols)

    edges = deborder
    contours = find_comp(edges)

    if len(contours) == 0:
        print('%s -> (no text!)' % path)
        return

    crop = optimal_comp(contours, edges)
    crop = pad_crop(crop, contours, edges, border_contour)

    # upscale to the original image size.
    crop = [int(x / scale) for x in crop]

    text_im = orig_im.crop(crop)
    text_im.save(out_path)
    print('%s -> %s' % (path, out_path))


if __name__ == '__main__':

    if len(sys.argv) == 2:
        files = glob.glob(sys.argv[1])
        random.shuffle(files)
    else:
        files = sys.argv[1:]

    for path in files:
        out_path = path.replace('.jpg', '.crop.png')

        # out_path = path.replace('.png', '.crop.png')  # .png as input
        if os.path.exists(out_path):
            continue
        try:
            process_image(path, out_path)
        except Exception as e:
            print('%s %s' % (path, e))