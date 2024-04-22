from os import mkdir,listdir
from os.path import exists,isfile, join, isdir
from shutil import rmtree
import numpy as np
import cv2
import os
import sys
sys.path.append('')

sleeper_length = 50  # примерное расстояние между рельсами в пикселях (длина шпалы) в нижней части изображения
min_car_length = 8 # минимальная длина вагона с учетом погрешности расстояний

# путь к калибровочным данным
path_tmp = os.getcwd()
path_to_kzsg_prj = ""
path_to_calibration_data = ""
for i in path_tmp.split("/"):
    path_to_kzsg_prj += i + "/"
    if i == "kzsg":
        break
print("path to kzsg prj is {}".format(path_to_kzsg_prj))
path_to_calibration_data = path_to_kzsg_prj + "Calibration/"
# путь к линиям, ограничивающим пути
path_to_lines = path_to_calibration_data + "rail_lines/lines_"
# путь к матрицам преобразования
warping_points_path = path_to_calibration_data + "warping_coefs/warping_coefs_"

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def get_jpg_filenames(path, recursively: bool = False, find_png: bool = False):

    if find_png:
        extension = "png"
    else:
        extension = "jpg"

    if recursively:
        return [f for f in getListOfFiles(path) if isfile(f) and f.split(".")[-1] == extension]
    else:
        return [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f.split(".")[-1] == extension]


def get_folders_traindata(path):
    return [join(path, f) for f in listdir(path) if isdir(join(path, f)) and f.split("(")[0].split("_")[0] == "traindata"]


def check_sector_num(sector_num: int):
    hr, ud = False, False#horizontal rails, up_to_down
    if int(sector_num) not in (2, 4, 5, 6, 7, 8, 9, 10, 11, 12):#Если не вертикальный сектор
        hr = True
    if int(sector_num) in [3]:#Если рельсы направлены сверху вниз
        ud = True
    return hr, ud


def rotate_img(img, sector_num: int):
    hr,ud = check_sector_num(sector_num)
    if hr:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if ud:
        img = cv2.rotate(img, cv2.ROTATE_180)
    return img


def unrotate_img(img, sector_num: int):
    hr, ud = check_sector_num(sector_num)
    if hr:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if ud:
        img = cv2.rotate(img, cv2.ROTATE_180)
    return img


def euclid_dist(x1,y1,x2,y2):
    return np.sqrt(np.power((x1 - x2), 2) + np.power((y1 - y2), 2))

def rm_mk_dirs(dirs):
    max_trials = 5
    for path in dirs:
        for i in range(0,max_trials):
            try:
                if exists(path):
                    rmtree(path)
                mkdir(path)
                break
            except Exception as e:
                if i == max_trials-1:
                    raise e
                pass

def parce_name(name:str, num_sectors, default_sector_num = 11):
    sector_num = default_sector_num
    try:
        sector_num_tmp = int(name.split("_")[-1].split(".")[0].split("(")[0])
        if sector_num_tmp <= num_sectors:
            sector_num = sector_num_tmp
    except:
        pass

    return sector_num


def get_line(rho,theta,w,h):
    max_add=50 #чтобы отрезки приравнять к линиям
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + max_add * (-b))
    y1 = int(y0 + max_add * (a))
    x2 = int(x0 - max_add * (-b))
    y2 = int(y0 - max_add * (a))

#найдем пересечения с краем кадра (пересечения с двумя из четырех краев)
    x_up, y_up = get_intersection(x1,x2,0,w,y1,y2,0,0)
    x_right, y_right = get_intersection(x1, x2, w, w, y1, y2, 0,h)
    x_down, y_down = get_intersection(x1, x2, 0,w, y1, y2, h, h)
    x_left, y_left = get_intersection(x1, x2, 0, 0, y1, y2, 0,h)
    x_chosen = []
    y_chosen = []
    for x, y in [x_down,y_down],[x_right,y_right],[x_left,y_left],[x_up,y_up]:
        if 0 <= x <= w and 0 <= y <= h:
            x_chosen.append(x)
            y_chosen.append(y)
    try:
        return int(x_chosen[0]), int(y_chosen[0]), int(x_chosen[1]), int(y_chosen[1])
        # return 0,0,0,0
    except:
        return 0,0,0,0





#вычисление точки пересечения отрезков [x1 y1]:[x2 y2] и [x3 y3]:[x4 y4]
def get_intersection(x1 : int,x2 : int,x3 : int,x4 : int,y1 : int,y2 : int,y3 : int,y4 : int):
    x1 = int(x1)
    x2 = int(x2)
    x3 = int(x3)
    x4 = int(x4)
    y1 = int(y1)
    y2 = int(y2)
    y3 = int(y3)
    y4 = int(y4)
    try:
        u = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        x = x1 + u * (x2 - x1)
        y = y1 + u * (y2 - y1)
    except ZeroDivisionError:
        x = -1
        y = -1
    return x, y


def from_real_to_warped(x,y, M):
    pts = np.array([[x, y]], dtype="float32")
    pts = np.array([pts])
    x,y = cv2.perspectiveTransform(pts, M)[0][0]
    return int(x), int(y)


def from_warped_to_real(x,y,M):
    pts = np.array([[x, y]], dtype="float32")
    pts = np.array([pts])
    M_inv = np.linalg.pinv(M)
    x,y = cv2.perspectiveTransform(pts, M_inv )[0][0]
    return int(x), int(y)


def show_and_destroy_img(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 800, 800)
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyWindow(name)

def get_lines_coords(num_of_sector, is_main_detection=True):

    with open(path_to_lines+str(num_of_sector)) as file:
        numbers = []
        for line in file:
            nums = line.split(sep=',')
            nums = [int(float(num)) for num in nums]
            numbers.append(nums)
    return numbers

def draw_lines_on_mask(mask, sector_num):
    numbers = get_lines_coords(sector_num)
    xs = []
    for num in numbers:
        xs.append((num[1], num[2]))

    y = mask.shape[0]
    x2_old = 0
    print(xs)
    for x1, x2 in xs:
        print(x1, x2)
        mask = cv2.line(mask, (x1, 0), (x1, y), color=(0, 0, 0), thickness=30)
        mask = cv2.line(mask, (x2, 0), (x2, y), color=(0, 0, 0), thickness=30)
        for x in range(x2_old, x1, 10):
            mask = cv2.line(mask, (x, 0), (x, y), color=(0, 0, 0), thickness=30)
        # mask = cv2.line(mask, (round((x1-x2_old)/2), 0), (round((x1-x2_old)/2), y), color=(0, 0, 0), thickness=x1-x2_old)
        x2_old = x2
    for x in range(x2_old, mask.shape[1], 10):
        mask = cv2.line(mask, (x, 0), (x, y), color=(0, 0, 0), thickness=30)
    return mask

def get_contours_from_mask(mask, num_of_sector, show_contours=True):
    cnts = seg_utils.get_contours(mask, num_of_sector, show_contours)

    if show_contours:
        mask = cv2.drawContours(mask, cnts, -1, (255, 255, 255), 6)
        show_and_destroy_img('mask', mask)

def bounding_boxes(img):
    ret, threshed_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    xs = []
    ys = []
    ws = []
    hs = []
    for c in contours:
        # get the bounding rect
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        xs.append(x)
        ys.append(y)
        ws.append(w)
        hs.append(h)
        if rect[2] < 50 or rect[3] < 50: continue
        # draw a green rectangle to visualize the bounding rect
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # print("BOXES", box)b
        # draw a red 'nghien' rectangle
        cv2.drawContours(img, [box], 0, (0, 0, 255))

    # print(len(contours))
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

    print('COORDS', xs, ys, ws, hs)

    return img, rect

## создание папок и под папок
def creat_folder(num_of_sector):
    os.mkdir('traindata/Traindata({})'.format(num_of_sector))
    os.mkdir('traindata/Traindata({})/images'.format(num_of_sector))
    os.mkdir('traindata/Traindata({})/labels'.format(num_of_sector))

def get_num_of_sector(nameIm):
    a = nameIm.split('.')
    b = a[-2].split('_')
    try:
        c = int(b[-1])
        num = c
    except ValueError:
        c = b[-1].split('(')
        num = c[0]
    return num

def draw_rect_with_corners(x0,y0,x1,y1, img, color=(0,0,255)):
    alpha = 0.3
    overlay = img.copy()
    output = img.copy()

    cv2.rectangle(overlay, (x0, y0), (x1, y1),
                  color, -1)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                    0, output)

    draw_corners(x0, y0, x1, y1, output)

    return output

def draw_corners(x0,y0,x1,y1, img):
    line_length =int(min(abs(x1-x0),abs(y1-y0)) / 3)
    cv2.line(img, (x0, y0 + line_length), (x0, y0), (0, 255, 0), thickness=10)
    cv2.line(img, (x0 + line_length, y0), (x0, y0), (0, 255, 0), thickness=10)

    cv2.line(img, (x1, y0 + line_length), (x1, y0), (0, 255, 0), thickness=10)
    cv2.line(img, (x1 - line_length, y0), (x1, y0), (0, 255, 0), thickness=10)

    cv2.line(img, (x1, y1 - line_length), (x1, y1), (0, 255, 0), thickness=10)
    cv2.line(img, (x1 - line_length, y1), (x1, y1), (0, 255, 0), thickness=10)

    cv2.line(img, (x0, y1 - line_length), (x0, y1), (0, 255, 0), thickness=10)
    cv2.line(img, (x0 + line_length, y1), (x0, y1), (0, 255, 0), thickness=10)


# def on_railway(mask, num_of_sector):
#     numbers = get_lines_coords(num_of_sector)
#     extreme_pts, convex_hulls_img = seg_utils.get_contours(mask, num_of_sector, SHOW_IMG=True)

