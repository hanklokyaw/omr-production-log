# Credit to original omr code creator rbaron (https://github.com/rbaron),
# without his script I could not accomplish this pilot project.
# Thank you rbaron!

import argparse
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

# When the four corners are identified, we will do a four-point
# perspective transform such that the outmost points of each
# corner map to a TRANSF_SIZE x TRANSF_SIZE square image.


employee_id_names = {
    1: 'Richard',
    2: 'Omar',
    3: 'Victor',
    4: 'John',
    5: 'David',
    6: 'Frank',
    7: 'Julian',
    9: 'Michael',
    10: 'Tomas',
    33: 'Victor'
}

activity_id_names = {
    0: 'No Activity',
    1: 'Gold Polish',
    2: 'Non-Gold Polish',
    3: 'Gem Setting',
    4: 'Gold Welding',
    5: 'Non-Gold Polish',
    6: 'Ring',
    7: 'Tumbling',
    8: 'Drilling',
    9: 'Sanding',
    10: 'Cups',
    11: 'Steaming',
    12: 'Cast',
    13: 'Ti',
    14: 'Bronze',
    15: 'R-Par',
    16: 'R-Ti',
    17: 'R-Gold',
    18: 'Gold Hammer',
    19: 'Ball Department',
    20: 'Repair',
    21: 'Item Code',
    22: 'Gold Hammer'
}

TRANSF_SIZE = 980
PAGE_WIDTH = 700
PAGE_HEIGHT = 1049

THRESHOLD = 0.9

###### EMPLOYEE ID MARK PROPERTIES #####
EID_ROWS = 2
EID_NO_OF_CHOICES = 9
EID_PATCH_HEIGHT = 20
EID_PATCH_HEIGHT_WITH_MARGIN = 22
EID_PATCH_LEFT_MARGIN = 495
EID_PATCH_RIGHT_MARGIN = 20
EID_PATCH_TOP_Y = 33
EID_ALT_HEIGHT = 20
EID_ALT_WIDTH = 19
EID_ALT_WIDTH_WITH_MARGIN = 20
EID_ALT_TOP_Y = 696
EID_MARK_POINT_START_X = 40
EID_MARK_WIDTH_MULTIPLY = 1.1
EID_MARK_WIDTH_MULTIPLY_ADJ = 0.9

###### DATE PROPERTIES #####
DATE_ROWS = 2
DATE_NO_OF_CHOICES = 12
DATE_PATCH_HEIGHT = 20
DATE_PATCH_HEIGHT_WITH_MARGIN = 22
DATE_PATCH_LEFT_MARGIN = 435
DATE_PATCH_RIGHT_MARGIN = 22
DATE_PATCH_TOP_Y = 84
DATE_ALT_HEIGHT = 20
DATE_ALT_WIDTH = 19
DATE_ALT_WIDTH_WITH_MARGIN = 20
DATE_ALT_TOP_Y = 610
DATE_MARK_POINT_START_X = 40
DATE_MARK_WIDTH_MULTIPLY = 1.1
DATE_MARK_WIDTH_MULTIPLY_ADJ = 0.9

###### DATE10 PROPERTIES #####
DATE10_ROWS = 1
DATE10_NO_OF_CHOICES = 3
DATE10_PATCH_HEIGHT = 20
DATE10_PATCH_HEIGHT_WITH_MARGIN = 22
DATE10_PATCH_LEFT_MARGIN = 435
DATE10_PATCH_RIGHT_MARGIN = 205
DATE10_PATCH_TOP_Y = 126
DATE10_ALT_HEIGHT = 20
DATE10_ALT_WIDTH = 19
DATE10_ALT_WIDTH_WITH_MARGIN = 20
DATE10_ALT_TOP_Y = 610
DATE10_MARK_POINT_START_X = 40
DATE10_MARK_WIDTH_MULTIPLY = 1.1
DATE10_MARK_WIDTH_MULTIPLY_ADJ = 0.9

###### DATE1 PROPERTIES #####
DATE1_ROWS = 1
DATE1_NO_OF_CHOICES = 9
DATE1_PATCH_HEIGHT = 20
DATE1_PATCH_HEIGHT_WITH_MARGIN = 22
DATE1_PATCH_LEFT_MARGIN = 495
DATE1_PATCH_RIGHT_MARGIN = 24
DATE1_PATCH_TOP_Y = 126
DATE1_ALT_HEIGHT = 20
DATE1_ALT_WIDTH = 19
DATE1_ALT_WIDTH_WITH_MARGIN = 20
DATE1_ALT_TOP_Y = 693
DATE1_MARK_POINT_START_X = 40
DATE1_MARK_WIDTH_MULTIPLY = 1.05
DATE1_MARK_WIDTH_MULTIPLY_ADJ = 0.9

###### HOURS PROPERTIES #####
HOURS_ROWS = 1
HOURS_NO_OF_CHOICES = 12
HOURS_PATCH_HEIGHT = 20
HOURS_PATCH_HEIGHT_WITH_MARGIN = 22
HOURS_PATCH_LEFT_MARGIN = 115
HOURS_PATCH_RIGHT_MARGIN = 342
HOURS_PATCH_TOP_Y = 126
HOURS_ALT_HEIGHT = 20
HOURS_ALT_WIDTH = 20
HOURS_ALT_WIDTH_WITH_MARGIN = 20.5
HOURS_ALT_TOP_Y = 160
HOURS_MARK_POINT_START_X = 40
HOURS_MARK_WIDTH_MULTIPLY = 1.03
HOURS_MARK_WIDTH_MULTIPLY_ADJ = 0.9

###### DAILY_ACTIVITY_10 PROPERTIES #####
ACTIVITY10_ROWS = 10
ACTIVITY10_NO_OF_CHOICES = 2
ACTIVITY10_PATCH_HEIGHT = 19.5
ACTIVITY10_PATCH_HEIGHT_WITH_MARGIN = 20.5
ACTIVITY10_PATCH_LEFT_MARGIN = 19
ACTIVITY10_PATCH_RIGHT_MARGIN = 642
ACTIVITY10_PATCH_TOP_Y = 826
ACTIVITY10_ALT_HEIGHT = 19.5
ACTIVITY10_ALT_WIDTH = 20
ACTIVITY10_ALT_WIDTH_WITH_MARGIN = 20.5
ACTIVITY10_ALT_TOP_Y = 25
ACTIVITY10_MARK_POINT_START_X = 40
ACTIVITY10_MARK_WIDTH_MULTIPLY = 1.03
ACTIVITY10_MARK_WIDTH_MULTIPLY_ADJ = 0.9

###### DAILY_ACTIVITY_1 PROPERTIES #####
ACTIVITY1_ROWS = 10
ACTIVITY1_NO_OF_CHOICES = 9
ACTIVITY1_PATCH_HEIGHT = 19.5
ACTIVITY1_PATCH_HEIGHT_WITH_MARGIN = 20.5
ACTIVITY1_PATCH_LEFT_MARGIN = 59
ACTIVITY1_PATCH_RIGHT_MARGIN = 460
ACTIVITY1_PATCH_TOP_Y = 826
ACTIVITY1_ALT_HEIGHT = 19.5
ACTIVITY1_ALT_WIDTH = 20
ACTIVITY1_ALT_WIDTH_WITH_MARGIN = 20.5
ACTIVITY1_ALT_TOP_Y = 80
ACTIVITY1_MARK_POINT_START_X = 40
ACTIVITY1_MARK_WIDTH_MULTIPLY = 1.03
ACTIVITY1_MARK_WIDTH_MULTIPLY_ADJ = 0.9

###### ITEM PROPERTIES #####
ITEM_ROWS = 10
ITEM_NO_OF_CHOICES = 3
ITEM_PATCH_HEIGHT = 19.5
ITEM_PATCH_HEIGHT_WITH_MARGIN = 20.5
ITEM_PATCH_LEFT_MARGIN = 249
ITEM_PATCH_RIGHT_MARGIN = 392
ITEM_PATCH_TOP_Y = 826
ITEM_ALT_HEIGHT = 19.5
ITEM_ALT_WIDTH = 19
ITEM_ALT_WIDTH_WITH_MARGIN = 19.5
ITEM_ALT_TOP_Y = 350
ITEM_MARK_POINT_START_X = 40
ITEM_MARK_WIDTH_MULTIPLY = 1.03
ITEM_MARK_WIDTH_MULTIPLY_ADJ = 0.9

###### PCSTHOUSAND PROPERTIES #####
PCS_T_ROWS = 10
PCS_T_NO_OF_CHOICES = 2
PCS_T_PATCH_HEIGHT = 19.5
PCS_T_PATCH_HEIGHT_WITH_MARGIN = 20.5
PCS_T_PATCH_LEFT_MARGIN = 307
PCS_T_PATCH_RIGHT_MARGIN = 370
PCS_T_PATCH_TOP_Y = 826
PCS_T_ALT_HEIGHT = 19.5
PCS_T_ALT_WIDTH = 19
PCS_T_ALT_WIDTH_WITH_MARGIN = 19.5
PCS_T_ALT_TOP_Y = 350
PCS_T_MARK_POINT_START_X = 40
PCS_T_MARK_WIDTH_MULTIPLY = 1.03
PCS_T_MARK_WIDTH_MULTIPLY_ADJ = 0.9

###### PCSHUNDRED PROPERTIES #####
PCS_H_ROWS = 10
PCS_H_NO_OF_CHOICES = 9
PCS_H_PATCH_HEIGHT = 19.5
PCS_H_PATCH_HEIGHT_WITH_MARGIN = 20.5
PCS_H_PATCH_LEFT_MARGIN = 330
PCS_H_PATCH_RIGHT_MARGIN = 254
PCS_H_PATCH_TOP_Y = 826
PCS_H_ALT_HEIGHT = 19.5
PCS_H_ALT_WIDTH = 12.8
PCS_H_ALT_WIDTH_WITH_MARGIN = 12.8
PCS_H_ALT_TOP_Y = 462
PCS_H_MARK_POINT_START_X = 37
PCS_H_MARK_WIDTH_MULTIPLY = 0.8
PCS_H_MARK_WIDTH_MULTIPLY_ADJ = 0.9

###### PCS10 PROPERTIES #####
PCS_10_ROWS = 10
PCS_10_NO_OF_CHOICES = 9
PCS_10_PATCH_HEIGHT = 19.5
PCS_10_PATCH_HEIGHT_WITH_MARGIN = 20.5
PCS_10_PATCH_LEFT_MARGIN = 449
PCS_10_PATCH_RIGHT_MARGIN = 137
PCS_10_PATCH_TOP_Y = 826
PCS_10_ALT_HEIGHT = 19.5
PCS_10_ALT_WIDTH = 12.8
PCS_10_ALT_WIDTH_WITH_MARGIN = 12.8
PCS_10_ALT_TOP_Y = 627
PCS_10_MARK_POINT_START_X = 22
PCS_10_MARK_WIDTH_MULTIPLY = 1
PCS_10_MARK_WIDTH_MULTIPLY_ADJ = 0.6

###### PCS PROPERTIES #####
PCS_ROWS = 10
PCS_NO_OF_CHOICES = 9
PCS_PATCH_HEIGHT = 19.5
PCS_PATCH_HEIGHT_WITH_MARGIN = 20.5
PCS_PATCH_LEFT_MARGIN = 566
PCS_PATCH_RIGHT_MARGIN = 16
PCS_PATCH_TOP_Y = 826
PCS_ALT_HEIGHT = 19.5
PCS_ALT_WIDTH = 12.8
PCS_ALT_WIDTH_WITH_MARGIN = 12.8
PCS_ALT_TOP_Y = 794
PCS_MARK_POINT_START_X = 23
PCS_MARK_WIDTH_MULTIPLY = 1.03
PCS_MARK_WIDTH_MULTIPLY_ADJ = 0.9


def calculate_contour_features(contour):
    """Calculates interesting properties (features) of a contour.

    We use these features to match shapes (contours). In this script,
    we are interested in finding shapes in our input image that look like
    a corner. We do that by calculating the features for many contours
    in the input image and comparing these to the features of the corner
    contour. By design, we know exactly what the features of the real corner
    contour look like - check out the calculate_corner_features function.

    It is crucial for these features to be invariant both to scale and rotation.
    In other words, we know that a corner is a corner regardless of its size
    or rotation. In the past, this script implemented its own features, but
    OpenCV offers much more robust scale and rotational invariant features
    out of the box - the Hu moments.
    """
    moments = cv2.moments(contour)
    return cv2.HuMoments(moments)


def calculate_corner_features():
    """Calculates the array of features for the corner contour.
    In practice, this can be pre-calculated, as the corners are constant
    and independent from the inputs.

    We load the img/corner.png file, which contains a single corner, so we
    can reliably extract its features. We will use these features to look for
    contours in our input image that look like a corner.
    """
    corner_img = cv2.imread('img/corner.png')
    corner_img_gray = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(
        corner_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # We expect to see only two contours:
    # - The "outer" contour, which wraps the whole image, at hierarchy level 0
    # - The corner contour, which we are looking for, at hierarchy level 1
    # If in trouble, one way to check what's happening is to draw the found contours
    # with cv2.drawContours(corner_img, contours, -1, (255, 0, 0)) and try and find
    # the correct corner contour by drawing one contour at a time. Ideally, this
    # would not be done at runtime.
    if len(contours) != 2:
        raise RuntimeError(
            'Did not find the expected contours when looking for the corner')

    # Following our assumptions as stated above, we take the contour that has a parent
    # contour (that is, it is _not_ the outer contour) to be the corner contour.
    # If in trouble, verify that this contour is the corner contour with
    # cv2.drawContours(corner_img, [corner_contour], -1, (255, 0, 0))
    corner_contour = next(ct
                          for i, ct in enumerate(contours)
                          if hierarchy[0][i][3] != -1)

    return calculate_contour_features(corner_contour)


def normalize(im, block_size=77, c_value=10):
    """Converts `im` to black and white.

    Applying a threshold to a grayscale image will make every pixel either
    fully black or fully white. Before doing so, a common technique is to
    get rid of noise (or super high frequency color change) by blurring the
    grayscale image with a Gaussian filter."""
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Filter the grayscale image with a 3x3 kernel
    blurred = cv2.GaussianBlur(im_gray, (3, 3), 0)

    # Applies a Gaussian adaptive thresholding. In practice, adaptive thresholding
    # seems to work better than applying a single, global threshold to the image.
    # This is particularly important if there could be shadows or non-uniform
    # lighting on the answer sheet. In those scenarios, using a global thresholding
    # technique might yield particularly bad results.
    # The choice of the parameters blockSize and C is crucial and domain-dependent.
    # In practice, you might want to try different values for your specific answer
    # sheet.
    return cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value)


def get_approx_contour(contour, tol=.01):
    """Gets rid of 'useless' points in the contour."""
    epsilon = tol * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def get_contours(image_gray):
    contours, _ = cv2.findContours(
        image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return map(get_approx_contour, contours)


def get_corners(contours):
    """Returns the 4 contours that look like a corner the most.

    In the real world, we cannot assume that the corners will always be present,
    and we likely need to decide how good is good enough for contour to
    look like a corner.
    This is essentially a classification problem. A good approach would be
    to train a statistical classifier model and apply it here. In our little
    exercise, we assume the corners are necessarily there."""
    corner_features = calculate_corner_features()
    return sorted(
        contours,
        key=lambda c: features_distance(
            corner_features,
            calculate_contour_features(c)))[:4]


def get_bounding_rect(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)


def features_distance(f1, f2):
    return np.linalg.norm(np.array(f1) - np.array(f2))


def draw_point(point, img, radius=5, color=(0, 0, 255)):
    cv2.circle(img, tuple(point), radius, color, -1)


def get_centroid(contour):
    m = cv2.moments(contour)
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    # print(m)
    return (x, y)


def sort_points_counter_clockwise(points):
    origin = np.mean(points, axis=0)

    def positive_angle(p):
        x, y = p - origin
        ang = np.arctan2(y, x)
        return 2 * np.pi + ang if ang < 0 else ang

    return sorted(points, key=positive_angle)


def get_outmost_points(contours):
    all_points = np.concatenate(contours)
    return get_bounding_rect(all_points)


def perspective_transform(img, points):
    """Applies a 4-point perspective transformation in `img` so that `points`
    are the new corners."""
    source = np.array(
        points,
        dtype="float32")

    dest = np.array([
        [TRANSF_SIZE, TRANSF_SIZE],
        [0, TRANSF_SIZE],
        [0, 0],
        [TRANSF_SIZE, 0]],
        dtype="float32")

    transf = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(img, transf, (TRANSF_SIZE, TRANSF_SIZE))
    return warped


def sheet_coord_to_transf_coord(x, y):
    return list(map(lambda n: int(np.round(n)), (
        TRANSF_SIZE * x / PAGE_WIDTH,
        TRANSF_SIZE * y / PAGE_HEIGHT
    )))


def get_question_patch(transf, question_index, patch_left_margin, patch_right_margin, patch_top_y, patch_height,
                       patch_height_with_margin):
    """Exracts a region of interest (ROI) of a single question."""
    # Top left of question patch q_number
    tl = sheet_coord_to_transf_coord(
        patch_left_margin,
        patch_top_y + patch_height_with_margin * question_index
    )

    # Bottom right of question patch q_number
    br = sheet_coord_to_transf_coord(
        PAGE_WIDTH - patch_right_margin,
        patch_top_y +
        patch_height +
        patch_height_with_margin * question_index
    )
    # print(transf[tl[1]:br[1], tl[0]:br[0]])
    return transf[tl[1]:br[1], tl[0]:br[0]]


def get_question_patches(transf, number_of_rows, patch_left_margin, patch_right_margin, patch_top_y, patch_height,
                         patch_height_with_margin):
    for i in range(number_of_rows):
        yield get_question_patch(transf, i, patch_left_margin, patch_right_margin, patch_top_y, patch_height,
                                 patch_height_with_margin)


def get_alternative_patches(question_patch, options, alt_width, alt_width_with_margin):
    for i in range(options):
        x0, _ = sheet_coord_to_transf_coord(alt_width_with_margin * i, 0)
        x1, _ = sheet_coord_to_transf_coord(alt_width +
                                            alt_width_with_margin * i, 0)
        yield question_patch[:, x0:x1]


def draw_marked_alternative(question_patch, index, alt_height, alt_width, mark_start_point, mark_padding, mark_adj):
    cx, cy = sheet_coord_to_transf_coord(
        (alt_width * (mark_padding * index - mark_adj)),
        alt_height / 2)
    draw_point((mark_start_point + cx, cy), question_patch, radius=5, color=(255, 0, 0))


#### Original script
def get_marked_alternative(alternative_patches, threshold):
    """Decides which alternative is marked, if any.

    Given a list of alternative patches, we need to decide which one,
    if any, is marked. Here, we do a simple, hacky heuristic: the
    alternative patch with lowest brightness (darker), is marked if
    it is sufficiently darker than the _second_ darker alternative
    patch.

    In practice, a more robust, data-driven model is necessary."""
    means = list(map(np.mean, alternative_patches))
    sorted_means = sorted(means)

    # Simple heuristic
    if sorted_means[0] / sorted_means[1] > threshold:
        return None

    return np.argmin(means)


##### Modified
# def get_marked_alternatives(alternative_patches, threshold):
#     """Decides which alternatives are marked, if any.
#
#     Given a list of alternative patches, we need to decide which ones,
#     if any, are marked. Here, we use a simple heuristic: all alternatives
#     below a certain brightness threshold are considered marked.
#
#     In practice, a more robust, data-driven model is necessary."""
#     means = list(map(np.mean, alternative_patches))
#     marked_indices = [i for i, mean in enumerate(means) if mean < threshold * min(means)]
#
#     return marked_indices if marked_indices else None


def eid_get_letter(alt_index):
    options = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    return options[alt_index] if alt_index is not None else "0"


def date_get_letter(alt_index):
    options = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    return options[alt_index] if alt_index is not None else "0"


def date10_get_letter(alt_index):
    options = ["1", "2", "3"]
    return options[alt_index] if alt_index is not None else "0"


def date1_get_letter(alt_index):
    options = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    return options[alt_index] if alt_index is not None else "0"


def hours_get_letter(alt_index):
    options = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    return options[alt_index] if alt_index is not None else "0"


def activity10_get_letter(alt_index):
    options = ["1", "2"]
    return options[alt_index] if alt_index is not None else "0"


def activity1_get_letter(alt_index):
    options = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    return options[alt_index] if alt_index is not None else "0"


def item_get_letter(alt_index):
    options = ["1", "2", "3"]
    return options[alt_index] if alt_index is not None else "0"


def pcs_t_get_letter(alt_index):
    options = ["1", "0"]
    return options[alt_index] if alt_index is not None else "0"


def pcs_h_get_letter(alt_index):
    options = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    return options[alt_index] if alt_index is not None else "0"


def pcs_10_get_letter(alt_index):
    options = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    return options[alt_index] if alt_index is not None else "0"


def pcs_get_letter(alt_index):
    options = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    return options[alt_index] if alt_index is not None else "0"


def draw_patch_boundaries(image, corners, color=(255, 0, 0)):
    cv2.drawContours(image, [np.array(corners)], -1, color, 3)


def get_answers(source_file, threshold, patch_start_x, number_of_rows, number_of_choices, patch_left_margin,
                patch_right_margin,
                patch_top_y, patch_height, patch_height_with_margin, alt_width, alt_width_with_margin, alt_height,
                mark_point_start_x, mark_point_width_multiply, mark_point_width_adj, section_name):
    """Runs the full pipeline:

    - Loads input image
    - Normalizes image
    - Finds contours
    - Finds corners among all contours
    - Finds 'outmost' points of all corners
    - Applies perspective transform to get a bird's eye view
    - Draws red rectangles around each question and alternative patch
    - Scans each line for the marked alternative
    """
    im_orig = cv2.imread(source_file)

    im_normalized = normalize(im_orig, block_size=77, c_value=10)

    contours = get_contours(im_normalized)

    corners = get_corners(contours)

    cv2.drawContours(im_orig, corners, -1, (0, 255, 0), 3)

    outmost = sort_points_counter_clockwise(get_outmost_points(corners))

    color_transf = perspective_transform(im_orig, outmost)
    normalized_transf = perspective_transform(im_normalized, outmost)

    # Draw red rectangles around each question and alternative patch
    for i, q_patch in enumerate(
            get_question_patches(normalized_transf, number_of_rows, patch_left_margin, patch_right_margin, patch_top_y,
                                 patch_height, patch_height_with_margin)):
        tl = sheet_coord_to_transf_coord(
            patch_left_margin,
            patch_top_y + patch_height_with_margin * i
        )
        br = sheet_coord_to_transf_coord(
            PAGE_WIDTH - patch_right_margin,
            patch_top_y +
            patch_height +
            patch_height_with_margin * i
        )
        cv2.rectangle(color_transf, tuple(tl), tuple(br), (0, 0, 255),
                      2)  ### use this line to print the option collection box

        alt_patches = list(get_alternative_patches(q_patch, number_of_choices, alt_width, alt_width_with_margin))
        # print(q_patch)
        for j, alt_patch in enumerate(alt_patches):
            x0, _ = sheet_coord_to_transf_coord(alt_width_with_margin * j, 0)
            x1, _ = sheet_coord_to_transf_coord(alt_width +
                                                alt_width_with_margin * j, 0)
            cv2.rectangle(color_transf, (patch_start_x + x0, tl[1]), (patch_start_x + x1, br[1]), (255, 0, 0), 2)

    answers = []
    for i, q_patch in enumerate(
            get_question_patches(normalized_transf, number_of_rows, patch_left_margin, patch_right_margin, patch_top_y,
                                 patch_height, patch_height_with_margin)):
        # print(normalized_transf)
        alt_index = get_marked_alternative(
            get_alternative_patches(q_patch, number_of_choices, alt_width, alt_width_with_margin), threshold)
        # print(alt_index)
        if alt_index is not None:
            color_q_patch = get_question_patch(color_transf, i, patch_left_margin, patch_right_margin, patch_top_y,
                                               patch_height, patch_height_with_margin)
            draw_marked_alternative(color_q_patch, alt_index, alt_height, alt_width, mark_point_start_x,
                                    mark_point_width_multiply, mark_point_width_adj)

        if section_name == 'eid':
            answers.append(eid_get_letter(alt_index))
        elif section_name == 'date':
            answers.append(date_get_letter(alt_index))
        elif section_name == 'date10':
            answers.append(date10_get_letter(alt_index))
        elif section_name == 'date1':
            answers.append(date1_get_letter(alt_index))
        elif section_name == 'hours':
            answers.append(hours_get_letter(alt_index))
        elif section_name == 'activity10':
            answers.append(activity10_get_letter(alt_index))
        elif section_name == 'activity1':
            answers.append(activity1_get_letter(alt_index))
        elif section_name == 'item':
            answers.append(item_get_letter(alt_index))
        elif section_name == 'pcs_t':
            answers.append(pcs_t_get_letter(alt_index))
        elif section_name == 'pcs_h':
            answers.append(pcs_h_get_letter(alt_index))
        elif section_name == 'pcs_10':
            answers.append(pcs_10_get_letter(alt_index))
        elif section_name == 'pcs':
            answers.append(pcs_get_letter(alt_index))

    return answers, color_transf, outmost


def main():
    input_path = './input/'

    jpg_files = [f for f in os.listdir(input_path) if f.lower().endswith('.jpg')]

    for jpg_file in jpg_files:
        # Construct the full path to the JPG file
        jpg_path = os.path.join(input_path, jpg_file)

        ##### Process EID
        # eid_parser = argparse.ArgumentParser()
        # eid_parser.add_argument(
        #     "--show",
        #     action="store_true",
        #     help="Displays annotated image")
        # eid_args = eid_parser.parse_args()
        user_id_list, im, outmost = get_answers(jpg_path, THRESHOLD, EID_ALT_TOP_Y, EID_ROWS, EID_NO_OF_CHOICES,
                                                EID_PATCH_LEFT_MARGIN, EID_PATCH_RIGHT_MARGIN, EID_PATCH_TOP_Y,
                                                EID_PATCH_HEIGHT, EID_PATCH_HEIGHT_WITH_MARGIN, EID_ALT_WIDTH,
                                                EID_ALT_WIDTH_WITH_MARGIN, EID_ALT_HEIGHT, EID_MARK_POINT_START_X,
                                                EID_MARK_WIDTH_MULTIPLY, EID_MARK_WIDTH_MULTIPLY_ADJ,
                                                'eid')
        print(user_id_list)

        ### User ID calculation
        user_id = (int(user_id_list[0]) * 10) + int(user_id_list[1])
        print(f"Employee ID: {user_id}")

        # if eid_args.show:
        #     cv2.imshow('Annotated image', im)
        #     cv2.waitKey(0)

        ##### Process DATE
        # date_parser = argparse.ArgumentParser()
        # date_parser.add_argument(
        #     "--show",
        #     action="store_true",
        #     help="Displays annotated image")
        # date_args = date_parser.parse_args()
        date_mark, im, outmost = get_answers(jpg_path, THRESHOLD, DATE_ALT_TOP_Y, DATE_ROWS, DATE_NO_OF_CHOICES,
                                             DATE_PATCH_LEFT_MARGIN, DATE_PATCH_RIGHT_MARGIN, DATE_PATCH_TOP_Y,
                                             DATE_PATCH_HEIGHT, DATE_PATCH_HEIGHT_WITH_MARGIN, DATE_ALT_WIDTH,
                                             DATE_ALT_WIDTH_WITH_MARGIN, DATE_ALT_HEIGHT, DATE_MARK_POINT_START_X,
                                             DATE_MARK_WIDTH_MULTIPLY, DATE_MARK_WIDTH_MULTIPLY_ADJ,
                                             'date')  # Retrieve 'outmost'
        print(date_mark)

        ### Process dates
        year = ''
        month = ''
        ## parse year and month
        if date_mark[0] == '1':
            year = '2024'
        elif date_mark[0] == '2':
            year = '2025'
        elif date_mark[0] == '3':
            year = '2026'
        elif date_mark[0] == '4':
            year = '2027'
        elif date_mark[0] == '5':
            year = '2028'
        elif date_mark[0] == '6':
            year = '2029'
        elif date_mark[0] == '7':
            year = '2030'
        elif date_mark[0] == '8':
            year = '2031'
        elif date_mark[0] == '9':
            year = '2032'
        elif date_mark[0] == '10':
            year = '2033'
        elif date_mark[0] == '11':
            year = '2034'
        elif date_mark[0] == '12':
            year = '2035'
        else:
            year = '????'

        if date_mark[1] == '1':
            month = '01'
        elif date_mark[1] == '2':
            month = '02'
        elif date_mark[1] == '3':
            month = '03'
        elif date_mark[1] == '4':
            month = '04'
        elif date_mark[1] == '5':
            month = '05'
        elif date_mark[1] == '6':
            month = '06'
        elif date_mark[1] == '7':
            month = '07'
        elif date_mark[1] == '8':
            month = '08'
        elif date_mark[1] == '9':
            month = '09'
        elif date_mark[1] == '10':
            month = '10'
        elif date_mark[1] == '11':
            month = '11'
        elif date_mark[1] == '12':
            month = '12'
        else:
            month = '??'

        # if date_args.show:
        #     cv2.imshow('Annotated image', im)
        #     cv2.waitKey(0)

        ### Process date 10 digit
        # date10_parser = argparse.ArgumentParser()
        # date10_parser.add_argument(
        #     "--show",
        #     action="store_true",
        #     help="Displays annotated image")
        # date10_args = date10_parser.parse_args()
        date10_mark, im, outmost = get_answers(jpg_path, THRESHOLD, DATE10_ALT_TOP_Y, DATE10_ROWS,
                                               DATE10_NO_OF_CHOICES,
                                               DATE10_PATCH_LEFT_MARGIN, DATE10_PATCH_RIGHT_MARGIN, DATE10_PATCH_TOP_Y,
                                               DATE10_PATCH_HEIGHT, DATE10_PATCH_HEIGHT_WITH_MARGIN, DATE10_ALT_WIDTH,
                                               DATE10_ALT_WIDTH_WITH_MARGIN, DATE10_ALT_HEIGHT,
                                               DATE10_MARK_POINT_START_X,
                                               DATE10_MARK_WIDTH_MULTIPLY, DATE10_MARK_WIDTH_MULTIPLY_ADJ,
                                               'date10')  # Retrieve 'outmost'
        print(date10_mark)

        ### Process Date 10
        date10 = date10_mark[0]

        # if date10_args.show:
        #     cv2.imshow('Annotated image', im)
        #     cv2.waitKey(0)

        ### Process Days
        # date1_parser = argparse.ArgumentParser()
        # date1_parser.add_argument(
        #     "--show",
        #     action="store_true",
        #     help="Displays annotated image")
        # date1_args = date1_parser.parse_args()
        date1_mark, im, outmost = get_answers(jpg_path, THRESHOLD, DATE1_ALT_TOP_Y, DATE1_ROWS,
                                              DATE1_NO_OF_CHOICES,
                                              DATE1_PATCH_LEFT_MARGIN, DATE1_PATCH_RIGHT_MARGIN, DATE1_PATCH_TOP_Y,
                                              DATE1_PATCH_HEIGHT, DATE1_PATCH_HEIGHT_WITH_MARGIN, DATE1_ALT_WIDTH,
                                              DATE1_ALT_WIDTH_WITH_MARGIN, DATE1_ALT_HEIGHT, DATE1_MARK_POINT_START_X,
                                              DATE1_MARK_WIDTH_MULTIPLY, DATE1_MARK_WIDTH_MULTIPLY_ADJ,
                                              'date1')  # Retrieve 'outmost'
        print(date1_mark)

        ### Process days calculation
        date1 = date1_mark[0]
        day = (int(date10) * 10) + int(date1)
        if day == 0 or day > 31:
            day = '??'
        print(day)
        print(f"Year-Month: {year}-{month}-{day}")

        # if date1_args.show:
        #     cv2.imshow('Annotated image', im)
        #     cv2.waitKey(0)

        # if date10_args.show:
        #     cv2.imshow('Annotated image', im)
        #     cv2.waitKey(0)

        ##### Process Hours
        # hours_parser = argparse.ArgumentParser()
        # hours_parser.add_argument(
        #     "--show",
        #     action="store_true",
        #     help="Displays annotated image")
        # hours_args = hours_parser.parse_args()
        hours_mark, im, outmost = get_answers(jpg_path, THRESHOLD, HOURS_ALT_TOP_Y, HOURS_ROWS,
                                              HOURS_NO_OF_CHOICES,
                                              HOURS_PATCH_LEFT_MARGIN, HOURS_PATCH_RIGHT_MARGIN, HOURS_PATCH_TOP_Y,
                                              HOURS_PATCH_HEIGHT, HOURS_PATCH_HEIGHT_WITH_MARGIN, HOURS_ALT_WIDTH,
                                              HOURS_ALT_WIDTH_WITH_MARGIN, HOURS_ALT_HEIGHT, HOURS_MARK_POINT_START_X,
                                              HOURS_MARK_WIDTH_MULTIPLY, HOURS_MARK_WIDTH_MULTIPLY_ADJ,
                                              'hours')  # Retrieve 'outmost'
        print(hours_mark)
        hours = hours_mark[0]
        print(f"Hours: {hours}hrs")

        # if hours_args.show:
        #     cv2.imshow('Annotated image', im)
        #     cv2.waitKey(0)

        # if date10_args.show:
        #     cv2.imshow('Annotated image', im)
        #     cv2.waitKey(0)

        ##### Process Activity
        # activity10_parser = argparse.ArgumentParser()
        # activity10_parser.add_argument(
        #     "--show",
        #     action="store_true",
        #     help="Displays annotated image")
        # activity10_args = activity10_parser.parse_args()
        activity10_mark, im, outmost = get_answers(jpg_path, THRESHOLD, ACTIVITY10_ALT_TOP_Y,
                                                   ACTIVITY10_ROWS,
                                                   ACTIVITY10_NO_OF_CHOICES,
                                                   ACTIVITY10_PATCH_LEFT_MARGIN, ACTIVITY10_PATCH_RIGHT_MARGIN,
                                                   ACTIVITY10_PATCH_TOP_Y,
                                                   ACTIVITY10_PATCH_HEIGHT, ACTIVITY10_PATCH_HEIGHT_WITH_MARGIN,
                                                   ACTIVITY10_ALT_WIDTH,
                                                   ACTIVITY10_ALT_WIDTH_WITH_MARGIN, ACTIVITY10_ALT_HEIGHT,
                                                   ACTIVITY10_MARK_POINT_START_X,
                                                   ACTIVITY10_MARK_WIDTH_MULTIPLY, ACTIVITY10_MARK_WIDTH_MULTIPLY_ADJ,
                                                   'activity10')  # Retrieve 'outmost'
        print(activity10_mark)

        # if activity10_args.show:
        #     cv2.imshow('Annotated image', im)
        #     cv2.waitKey(0)

        # activity1_parser = argparse.ArgumentParser()
        # activity1_parser.add_argument(
        #     "--show",
        #     action="store_true",
        #     help="Displays annotated image")
        # activity1_args = activity1_parser.parse_args()
        activity1_mark, im, outmost = get_answers(jpg_path, THRESHOLD, ACTIVITY1_ALT_TOP_Y, ACTIVITY1_ROWS,
                                                  ACTIVITY1_NO_OF_CHOICES,
                                                  ACTIVITY1_PATCH_LEFT_MARGIN, ACTIVITY1_PATCH_RIGHT_MARGIN,
                                                  ACTIVITY1_PATCH_TOP_Y,
                                                  ACTIVITY1_PATCH_HEIGHT, ACTIVITY1_PATCH_HEIGHT_WITH_MARGIN,
                                                  ACTIVITY1_ALT_WIDTH,
                                                  ACTIVITY1_ALT_WIDTH_WITH_MARGIN, ACTIVITY1_ALT_HEIGHT,
                                                  ACTIVITY1_MARK_POINT_START_X,
                                                  ACTIVITY1_MARK_WIDTH_MULTIPLY, ACTIVITY1_MARK_WIDTH_MULTIPLY_ADJ,
                                                  'activity1')  # Retrieve 'outmost'
        print(activity1_mark)

        ### Process activity list
        activity_list = []
        for i in range(0, len(activity1_mark)):
            activity_list.append((int(activity10_mark[i]) * 10) + int(activity1_mark[i]))
        print(activity_list)

        ##### Process Items
        # item_parser = argparse.ArgumentParser()
        # item_parser.add_argument(
        #     "--show",
        #     action="store_true",
        #     help="Displays annotated image")
        # item_args = item_parser.parse_args()
        item_mark, im, outmost = get_answers(jpg_path, THRESHOLD, ITEM_ALT_TOP_Y, ITEM_ROWS, ITEM_NO_OF_CHOICES,
                                             ITEM_PATCH_LEFT_MARGIN, ITEM_PATCH_RIGHT_MARGIN, ITEM_PATCH_TOP_Y,
                                             ITEM_PATCH_HEIGHT, ITEM_PATCH_HEIGHT_WITH_MARGIN, ITEM_ALT_WIDTH,
                                             ITEM_ALT_WIDTH_WITH_MARGIN, ITEM_ALT_HEIGHT, ITEM_MARK_POINT_START_X,
                                             ITEM_MARK_WIDTH_MULTIPLY, ITEM_MARK_WIDTH_MULTIPLY_ADJ,
                                             'date')  # Retrieve 'outmost'
        print(item_mark)

        ### Process Items
        item_list = []
        for i in range(0, len(item_mark)):
            item_list.append(int(item_mark[i]))

        # if item_args.show:
        #     cv2.imshow('Annotated image', im)
        #     cv2.waitKey(0)

        ##### Process Pieces in Thousands
        # pcs_t_parser = argparse.ArgumentParser()
        # pcs_t_parser.add_argument(
        #     "--show",
        #     action="store_true",
        #     help="Displays annotated image")
        # pcs_t_args = pcs_t_parser.parse_args()
        pcs_t_mark, im, outmost = get_answers(jpg_path, THRESHOLD, PCS_T_ALT_TOP_Y, PCS_T_ROWS,
                                              PCS_T_NO_OF_CHOICES,
                                              PCS_T_PATCH_LEFT_MARGIN, PCS_T_PATCH_RIGHT_MARGIN, PCS_T_PATCH_TOP_Y,
                                              PCS_T_PATCH_HEIGHT, PCS_T_PATCH_HEIGHT_WITH_MARGIN, PCS_T_ALT_WIDTH,
                                              PCS_T_ALT_WIDTH_WITH_MARGIN, PCS_T_ALT_HEIGHT, PCS_T_MARK_POINT_START_X,
                                              PCS_T_MARK_WIDTH_MULTIPLY, PCS_T_MARK_WIDTH_MULTIPLY_ADJ,
                                              'pcs_t')  # Retrieve 'outmost'
        print(pcs_t_mark)

        # if pcs_t_args.show:
        #     cv2.imshow('Annotated image', im)
        #     cv2.waitKey(0)

        ### Process Pieces in Hundreds
        # pcs_h_parser = argparse.ArgumentParser()
        # pcs_h_parser.add_argument(
        #     "--show",
        #     action="store_true",
        #     help="Displays annotated image")
        # pcs_h_args = pcs_h_parser.parse_args()
        pcs_h_mark, im, outmost = get_answers(jpg_path, THRESHOLD, PCS_H_ALT_TOP_Y, PCS_H_ROWS,
                                              PCS_H_NO_OF_CHOICES,
                                              PCS_H_PATCH_LEFT_MARGIN, PCS_H_PATCH_RIGHT_MARGIN, PCS_H_PATCH_TOP_Y,
                                              PCS_H_PATCH_HEIGHT, PCS_H_PATCH_HEIGHT_WITH_MARGIN, PCS_H_ALT_WIDTH,
                                              PCS_H_ALT_WIDTH_WITH_MARGIN, PCS_H_ALT_HEIGHT, PCS_H_MARK_POINT_START_X,
                                              PCS_H_MARK_WIDTH_MULTIPLY, PCS_H_MARK_WIDTH_MULTIPLY_ADJ,
                                              'pcs_h')  # Retrieve 'outmost'
        print(pcs_h_mark)

        # if pcs_h_args.show:
        #     cv2.imshow('Annotated image', im)
        #     cv2.waitKey(0)

        ### Process pieces in 10 digits
        # pcs_10_parser = argparse.ArgumentParser()
        # pcs_10_parser.add_argument(
        #     "--show",
        #     action="store_true",
        #     help="Displays annotated image")
        # pcs_10_args = pcs_10_parser.parse_args()
        pcs_10_mark, im, outmost = get_answers(jpg_path, THRESHOLD, PCS_10_ALT_TOP_Y, PCS_10_ROWS,
                                               PCS_10_NO_OF_CHOICES,
                                               PCS_10_PATCH_LEFT_MARGIN, PCS_10_PATCH_RIGHT_MARGIN, PCS_10_PATCH_TOP_Y,
                                               PCS_10_PATCH_HEIGHT, PCS_10_PATCH_HEIGHT_WITH_MARGIN, PCS_10_ALT_WIDTH,
                                               PCS_10_ALT_WIDTH_WITH_MARGIN, PCS_10_ALT_HEIGHT,
                                               PCS_10_MARK_POINT_START_X,
                                               PCS_10_MARK_WIDTH_MULTIPLY, PCS_10_MARK_WIDTH_MULTIPLY_ADJ,
                                               'pcs_10')  # Retrieve 'outmost'
        print(pcs_10_mark)

        # if pcs_10_args.show:
        #     cv2.imshow('Annotated image', im)
        #     cv2.waitKey(0)

        ### Process Pieces
        # pcs_parser = argparse.ArgumentParser()
        # pcs_parser.add_argument(
        #     "--show",
        #     action="store_true",
        #     help="Displays annotated image")
        # pcs_args = pcs_parser.parse_args()
        pcs_mark, im, outmost = get_answers(jpg_path, THRESHOLD, PCS_ALT_TOP_Y, PCS_ROWS, PCS_NO_OF_CHOICES,
                                            PCS_PATCH_LEFT_MARGIN, PCS_PATCH_RIGHT_MARGIN, PCS_PATCH_TOP_Y,
                                            PCS_PATCH_HEIGHT, PCS_PATCH_HEIGHT_WITH_MARGIN, PCS_ALT_WIDTH,
                                            PCS_ALT_WIDTH_WITH_MARGIN, PCS_ALT_HEIGHT, PCS_MARK_POINT_START_X,
                                            PCS_MARK_WIDTH_MULTIPLY, PCS_MARK_WIDTH_MULTIPLY_ADJ,
                                            'pcs')  # Retrieve 'outmost'

        print(pcs_mark)

        date = f'{year}-{month}-{day}'
        # hours
        # user_id

        pcs_list = []
        for i in range(0, len(pcs_mark)):
            pcs_list.append((int(pcs_t_mark[i]) * 1000) + (int(pcs_h_mark[i]) * 100) + (int(pcs_10_mark[i]) * 10) + int(
                pcs_mark[i]))

        print("---------------------------")

        employee_name = ''
        activity_name_list = []

        for key, value in employee_id_names.items():
            print(f'key: {key}')
            print(f'user_id: {user_id}')
            if key == user_id:
                employee_name = value
                print(f'value: {value}')
                break
            else:
                employee_name = 'Employee name not registered'

        activity_name_list = [activity_id_names.get(activity, 'Activity name not registered') for activity in
                              activity_list]

        print(date)
        print(hours)
        print(user_id)
        print(employee_name)
        print(activity_list)
        print(activity_name_list)
        print(item_list)
        print(pcs_list)

        # print(f"Hours: {hours}hrs")

        # if activity1_args.show:
        #     cv2.imshow('Annotated image', im)
        #     cv2.waitKey(0)

        # if date_args.show:
        #     cv2.imshow('Annotated image', im)
        #     cv2.waitKey(0)

        # for i, answer in enumerate(answers):
        #     print("Item {}: {}".format(i + 1, answer))

        # Create a DataFrame with question numbers and answers
        df = pd.DataFrame({'EMPLOYEE_ID': [user_id] * len(pcs_list), 'EMPLOYEE_NAME': [employee_name] * len(pcs_list),
                           "DATE": [date] * len(pcs_list), "HOURS": [hours] * len(pcs_list),
                           "ACTIVITY_ID": activity_list, "ACTIVITY_NAME": activity_name_list, "ITEM": item_list,
                           "PIECES": pcs_list})

        df = df[(df['ACTIVITY_ID'] != 0) | (df['ITEM'] != 0) | (df['PIECES'] != 0)]

        # Reset index after dropping rows
        df = df.reset_index(drop=True)

        print(df)
        # entries = pd.DataFrame({"Item": range(1, EID_ROWS + 1), "Pieces": answers})
        #
        try:
            date_str = datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
            custom_file_name = f'{date_str}_{user_id}_{employee_name}.csv'

            output_file_path = './exports/' + custom_file_name

            # Save the DataFrame to the customized file
            df.to_csv(output_file_path, index=False)

            print('Record Item to {}.'.format(output_file_path))

        except:
            print("No output. Error!")

        # if pcs_args.show:
        #     cv2.imshow('Annotated image', im)
        #     cv2.waitKey(0)

        # if eid_args.show:
        #     cv2.imshow('Annotated image', im)
        #     cv2.waitKey(0)

        # Print the DataFrame
        # print(df)

        # Export the DataFrame to a CSV file if --output is provided
        # if args.output:
        #     df.to_csv(args.output, index=False)
        #     print('Record Item to {}.'.format(args.output))

        print(f"Processed {jpg_file}: {user_id_list}")


if __name__ == '__main__':
    main()
