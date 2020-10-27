import cv2
import numpy as np

##########################
imgWidth = 640
imgHeight = 480
##########################

# Camera read
cap = cv2.VideoCapture(1)
cap.set(3, imgWidth)
cap.set(4, imgWidth)
cap.set(10, 150)  # Set Brightness


def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        hor_con = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                          scale)
            if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def get_contours(image):
    max_cnt_points = np.array([])
    max_area = 0
    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                max_cnt_points = approx
                max_area = area
    cv2.drawContours(imgContour, max_cnt_points, -1, (255, 0, 0), 20)
    return max_cnt_points


def pre_processing(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 200, 200)
    kernel = np.ones((5, 5))
    img_dial = cv2.dilate(img_canny, kernel, iterations=2)
    img_threshold = cv2.erode(img_dial, kernel, iterations=1)
    return img_threshold


def sort_second(val):
    return val[1]


def get_wrap(image, max_cnt_points):
    if len(max_cnt_points) == 4:
        max_cnt_points = sorted(max_cnt_points.reshape((4, 2)), key=sort_second)
        pts1 = np.float32(max_cnt_points)
        pts2 = np.float32([[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_output = cv2.warpPerspective(image, matrix, (imgWidth, imgHeight))

        img_cropped = img_output[10: img_output.shape[0] - 10, 10: img_output.shape[1] - 10]
        img_cropped = cv2.resize(img_cropped, (imgWidth, imgHeight))

        return img_cropped
    else:
        return np.zeros_like(img)


while True:
    success, img = cap.read()
    cv2.resize(img, (imgWidth, imgHeight))
    imgContour = img.copy()
    imgThreshold = pre_processing(img)
    max_cnt = get_contours(imgThreshold)

    # print(biggest)
    imgWrapped = get_wrap(img, max_cnt)

    # join stack images
    imgStack = stack_images(0.5,
                            ([img, imgThreshold],
                             [imgContour, imgWrapped]))

    cv2.imshow('Scanned Image', cv2.resize(imgWrapped, (480, 640)))
    cv2.imshow('Work Flow', imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
