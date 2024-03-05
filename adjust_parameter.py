import cv2
import numpy as np
import argparse

# Initialize parameters
N_QUESTIONS = 10

ANSWER_SHEET_WIDTH = 740 # GREEN
ANSWER_SHEET_HEIGHT = 1049 # GREEN

# ANSWER_PATCH_HEIGHT = 160  # 50 # RED
ANSWER_PATCH_HEIGHT_WITH_MARGIN = 218  # 80 # RED
ANSWER_PATCH_LEFT_MARGIN = 510  # 200 # RED
ANSWER_PATCH_RIGHT_MARGIN = -1450  # 90 # RED
FIRST_ANSWER_PATCH_TOP_Y = 560  # 200 # RED

ALTERNATIVE_HEIGHT = 218 #BLUE
# ALTERNATIVE_WIDTH = 40 #BLUE
ALTERNATIVE_WIDTH_WITH_MARGIN = 170 #BLUE
NUM_CHOICES = 10  # Update the number of choices


def draw_lines(image, lines, color):
    """Draw lines on the image with specified color."""
    for line in lines:
        cv2.line(image, line[0], line[1], color, 2)


def adjust_parameters(image):
    """Draw lines to adjust the parameters."""
    lines = []

    # Draw lines for ANSWER_SHEET_WIDTH (Green color)
    lines.append([(ANSWER_PATCH_LEFT_MARGIN, FIRST_ANSWER_PATCH_TOP_Y),
                  (ANSWER_SHEET_WIDTH - ANSWER_PATCH_RIGHT_MARGIN, FIRST_ANSWER_PATCH_TOP_Y)])

    # Draw lines for ANSWER_PATCH_HEIGHT_WITH_MARGIN (Red color)
    for i in range(N_QUESTIONS + 1):
        y = FIRST_ANSWER_PATCH_TOP_Y + i * ANSWER_PATCH_HEIGHT_WITH_MARGIN
        lines.append([(ANSWER_PATCH_LEFT_MARGIN, y), (ANSWER_SHEET_WIDTH - ANSWER_PATCH_RIGHT_MARGIN, y)])

    # Move ALTERNATIVE_WIDTH_WITH_MARGIN (Blue color) lines 100 pixels left and 150 pixels down
    for i in range(NUM_CHOICES + 1):
        x = i * ALTERNATIVE_WIDTH_WITH_MARGIN
        start_point = (x + 500, 560)
        end_point = (x + 500, ANSWER_SHEET_HEIGHT + 1690)
        lines.append([start_point, end_point])

    # Draw lines with different colors
    draw_lines(image, lines[:2], (0, 255, 0))  # Green color for ANSWER_SHEET_WIDTH
    draw_lines(image, lines[2:13], (0, 0, 255))  # Red color for ANSWER_PATCH_HEIGHT_WITH_MARGIN
    draw_lines(image, lines[13:], (255, 0, 0))  # Blue color for ALTERNATIVE_WIDTH_WITH_MARGIN

    return image



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--show",
        action="store_true",
        help="Displays annotated image")

    args = parser.parse_args()

    # Capture an image (replace this with your image capture logic)
    image = cv2.imread('img/rich2.jpg')

    # Adjust parameters interactively
    annotated_image = adjust_parameters(image)

    if args.show:
        # If --show is provided, create a resizable window
        cv2.namedWindow('Annotated image', cv2.WINDOW_NORMAL)
        # Resize the window to match the answer sheet dimensions
        cv2.resizeWindow('Annotated image', ANSWER_SHEET_WIDTH, ANSWER_SHEET_HEIGHT)
        # Display the annotated image
        cv2.imshow('Annotated image', annotated_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
