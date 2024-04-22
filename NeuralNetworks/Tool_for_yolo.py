import sys, time

import os
import cv2 as cv
import nn_utils
import glob
global n
import tqdm
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
from yolov5 import detect



path_to_photo_warp = "yolov5/images/work_dir/"



class Rectangle():
    x1, y1, x2, y2 = 0, 0, 0, 0
    def __init__(self,x1,y1,x2,y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

class App():
    global n
    num_imag = -1
    path_dir = ''
    path_to_txt = path_dir + 'NeuralNetworks/yolov5/inference/output/' # путь к лейблам
    path_to_warpIm = path_dir + 'NeuralNetworks/yolov5/images/work_dir/' # путь к подготовленным для разметки изображениям
    path_to_save_image = path_dir + 'NeuralNetworks/traindata/images/' # путь для сохранения изоражений
    path_to_save_labels = path_dir + 'NeuralNetworks/traindata/labels/' # путь для сохранения лейблов к изображениям
    BLUE = [255, 0, 0]  # rectangle color

    # setting up flags
    rect = (0, 0, 1, 1)
    drawing = False  # flag for drawing curves
    rect_over = False  # flag to check if rect drawn
    rect_or_mask = 100  # flag for selecting rect or mask mode
    thickness = 3  # brush thickness
    x0 = 0
    y0 = 0
    classes = {'Kritiy_Vagon': 40, 'Poluvagon': 70, 'Transporter': 100, 'Tsisterna': 130, 'Lokomotiv': 160,
               'Hopper': 190, 'Else': 220}
    counter = 0
    x1, x2, y1, y2 = 0, 0, 0, 0
    rectangles = []
    rectangle = False
    rectangle_move = False
    rm = False
    createFold = True
    Action = True
    ctr = 0

    def check_image(self):
        # print(os.getcwd(), "getcwd")
        tree = os.listdir(self.path_to_warpIm)
        tree.sort()
        identy = 0
        length = len(tree)
        print(" Instructions:")
        print("   - Жми (n) и (p) для перемещения по списку фотографий. Жми (y) для выбора отображаемой фотки. ESC для выхода ")
        print("   - Жми (y) для выбора отображаемой фотки. ESC для выхода ")
        while True:
            path = os.path.join(self.path_to_warpIm, tree[identy])
            textpath = os.path.splitext(path)
            arrtextpath = textpath[0].split('/')
            nameim = arrtextpath[-1] + textpath[-1]
            cv.namedWindow('images')
            img = cv.imread(path)
            im2 = cv.resize(img, (1200, 700))
            cv.imshow('images', im2)
            k = cv.waitKey(1)
            if k == 27: # exit
                cv.destroyWindow('images')
                break
            elif k == ord('n'):  # go to next image
                if identy < length - 1:
                    identy += 1
                else:
                    print('взгляните вниз, вы обработали все фотки, ты добился своего ну что, и чего тебе это стоило?')
            elif k == ord('p') or k == ord('b'):  # go to prev image
                if identy > 0:
                    identy -= 1
                else:
                    print('вы находитесь у подножия горы фотографий, которые вам нужно разобрать')
            elif k == ord('y')  or k == ord('v'):  # выбор текущего изображения для разметки
                self.img = cv.resize(img, (1200, 700))
                self.orig_img = self.img
                self.img2 = self.img.copy()  # a copy of original image
                self.name = nn_utils.get_name(nameim)
                try:
                    ###  переместить часть кода в try
                    txtfile = self.path_to_txt + arrtextpath[-1] + '.txt'
                    points_list = nn_utils.get_points(txtfile, self.img)
                    print(img.shape[0], img.shape[1])
                    ## загрузка координат размеченных нейронкой
                    for points in points_list:
                        self.rectangles.append(Rectangle(points[0], points[1], points[2], points[3]))

                except Exception as e:
                    print('Не удалось открыть файл ')
                    print(e.__class__)
                finally:
                    ## исправление вызова контекстного меню через пкм
                    cv.imshow('input', self.img)
                    cv.destroyAllWindows()
                    self.imagesave = cv.imread(path)
                    self.run(self.img)
        print('Обработано', self.ctr, 'изображений(я/е)')

    def check_rectangle(self, x, y):
        flag = False
        for i in range(len(self.rectangles)):
            if x >= self.rectangles[i].x1 and x <= self.rectangles[i].x2 and y >= self.rectangles[i].y1 and y <= self.rectangles[i].y2:
                self.x1, self.y1 = self.rectangles[i].x1, self.rectangles[i].y1
                self.x2, self.y2 = self.rectangles[i].x2, self.rectangles[i].y2
                self.rectangles.pop(i)
                self.delete_rectangle = i
                flag = True
                break
        return flag

    def onmouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.Action = False
            if not self.rectangle_move:
                self.rectangle = True
                if not self.check_rectangle(x,y):
                    self.x2, self.y2 = x, y
                self.x1, self.y1 = x, y
            for rectangle in self.rectangles:
                cv.rectangle(self.img, (rectangle.x1, rectangle.y1), (rectangle.x2, rectangle.y2), (255, 0, 0), 4)

        elif event == cv.EVENT_MOUSEMOVE:
            if self.rectangle:
                self.x2, self.y2 = x, y
                self.img = self.img2.copy()
                cv.rectangle(self.img, (self.x1, self.y1), (x, y), self.BLUE, 2)

            elif self.rectangle_move:
                self.img = self.img2.copy()
                dif_x = x - self.x3
                dif_y = y - self.y3 
                self.x1 += dif_x
                self.x2 += dif_x
                self.y1 += dif_y
                self.y2 += dif_y

                cv.rectangle(self.img, (self.x1, self.y1), (self.x2, self.y2), self.BLUE, 2)

                self.x3, self.y3 = x, y
            for rectangle in self.rectangles:
                cv.rectangle(self.img, (rectangle.x1, rectangle.y1), (rectangle.x2, rectangle.y2), (255, 0, 0), 4)

        elif event == cv.EVENT_RBUTTONDOWN:
            self.Action = False

            if not self.rectangle:
                self.rectangle_move = True
                if not self.check_rectangle(x, y):
                    self.rectangle_move = False
                    self.rm = False
                else:
                    self.x3, self.y3 = x, y

        elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
            self.Action = True
            if self.rectangle or self.rectangle_move:
                self.rectangle = False
                self.rm = False
                self.rectangle_move = False
                x1, y1, x2, y2 = self.correctPoints(self.x1, self.y1, self.x2, self.y2)
                self.rectangles.append(Rectangle(x1, y1, x2, y2))
                self.dr = len(self.rectangles)-1

    def correctPoints(self, x1,y1,x2,y2):
        x1 = self.out_of_range(x1,1)
        x2 = self.out_of_range(x2,1)
        y1 = self.out_of_range(y1,0)
        y2 = self.out_of_range(y2,0)
        return x1, y1, x2, y2

    def out_of_range(self,p, i):
        if p > self.img.shape[i]:
            p = self.img.shape[i] - 1
        elif p < 0:
            p = 0
        return p

    def save_rectangles(self):
        os.makedirs(self.path_to_save_image, exist_ok=True)
        os.makedirs(self.path_to_save_labels, exist_ok=True)

        cv.imwrite(self.path_to_save_image + '/{}.jpg'.format(self.name), self.imagesave)
        ## подготовка данных для записи в txt файл
        with open(self.path_to_save_labels + '{}.txt'.format(self.name), "w") as file:
            print()
        for i in range(len(self.rectangles)):
            x_s = str((self.rectangles[i].x1 + self.rectangles[i].x2) / 2 / self.img.shape[1])
            y_s = str((self.rectangles[i].y1 + self.rectangles[i].y2) / 2 / self.img.shape[0])
            x1 = min(self.rectangles[i].x1, self.rectangles[i].x2)
            x0 = max(self.rectangles[i].x1, self.rectangles[i].x2)
            y1 = min(self.rectangles[i].y1, self.rectangles[i].y2)
            y0 = max(self.rectangles[i].y1, self.rectangles[i].y2)
            w = str(abs(x0 - x1) / self.img.shape[1])
            h = str(abs(y0 - y1) / self.img.shape[0])
            try:
                with open(self.path_to_save_labels + '{}.txt'.format(self.name), "a") as file:
                    klass = str(0)
                    file.write('{} {} {} {} {}'.format(klass, x_s, y_s, w, h) + "\n")
                print('координаты', i + 1, "прямоугольника сохранены в", self.name + '.txt')
            except Exception as e:
                print('Файл не сохранен ')
                print(e.__class__)
        self.createFold = False
        self.ctr += 1

    def run(self, img_warp):
        self.img = img_warp
        self.orig_img = self.img
        self.img2 = self.img.copy()  # a copy of original image
        cv.namedWindow('input')
        cv.setMouseCallback('input', self.onmouse)

        while (1):
            for rectangle in self.rectangles:
                cv.rectangle(self.img, (rectangle.x1, rectangle.y1), (rectangle.x2, rectangle.y2), (255, 0, 0), 4)
            cv.imshow('input', self.img)
            k = cv.waitKey(1)
            if k == 27:  # esc to exit
                self.flagSave = True
                self.rectangles.clear()
                cv.destroyAllWindows()
                self.num_imag+=1

                print(" Instructions:")
                print("   - Жми (n) и (p) для перемещения по списку фотографий. Жми (y) для выбора отображаемой фотки. ESC для выхода ")
                print("   - Жми (y) для выбора отображаемой фотки. ESC для выхода ")
                break
            elif k == ord('s'):  # сохранить координаты
                self.save_rectangles()

            elif k == ord('r'):  # reset everything
                print("resetting \n")
                self.rect = (0, 0, 1, 1)
                self.img = self.img2.copy()
                self.rectangles.clear()
            elif k == ord('d') and self.Action:  # delete item
                len1= len(self.rectangles)
                if not self.dr >= len1 and self.dr >= 0:
                    self.rectangles.pop(self.dr)
                    self.img = self.img2.copy()
                    for rectangle in self.rectangles:
                        cv.rectangle(self.img, (rectangle.x1, rectangle.y1), (rectangle.x2, rectangle.y2), (255, 0, 0), 4)
                    self.dr-=1
                else:
                    print('милорд, пожалуйста, хватит,вы уже все удалили')




if __name__ == '__main__':
    global n
    print('Подготовить новые фотки/предсказания press (e)')
    cv.namedWindow('small_black_window')
    while (1):
        cv.namedWindow('small_black_window')
        k = cv.waitKey(1)
        if k == 27:
            break
        elif k == ord('q'):
            try:
                App().check_image()
            except Exception as e:
                print('Папка ./yolov5/images пуста, подготовте изображения  press(y) а затем предсказания press(e)')
                print(e.__class__)


        elif k == ord('e'):
            #files = glob.glob(path_to_photo_warp)
            files = [file for file in glob.glob(path_to_photo_warp + "*")]
            if len(files) == 0:
                print('Отсутствуют фотки для предсказаний в папке images/')
            else:
                print('Работа нейронки(подготовка предсказаний)')
                #os.chdir('yolov5/')
                start = time.time()
                # box = nn_utils.yolo_work()
                for i in files:
                    name_frame = os.path.basename(i).split(".png")[0].split(".jpg")[0]

                    #print(name_frame, "name_frame")
                    image_frame = cv.imread(i)

                    box = detect.detection_function(image_frame,name_frame)


                print('Time work yolo:', time.time() - start)
                os.chdir('../')
                print('Выберите изображение')
                App().check_image()