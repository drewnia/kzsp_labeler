import cv2
import sys
sys.path.append("../")
from pylab import *
import kzsg_utils
#import tensorflow as tf
import subprocess
import os
from skimage import transform
import tqdm
from glob import glob

NEURAL_SIZE = 256
TEST_PATH = 'TEST/'
TEST_IMG_PATH = 'yolov5/test_warped/images/'
TXT_FILES_PATH = 'yolov5/inference/output/'
path_to_calibration_data = "Calibration/"


def frames_count(vidpath):
    """Функция возвращает количество кадров на видео"""
    video = cv2.VideoCapture(vidpath)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return total


class ExtremePoint(object):
    """Класс для хранения экстремальных точек, содержащих информацию о найденных областях
    на изображении, полученном от нейросети UNET"""
    def __init__(self, c: np.ndarray, t: np.ndarray, l: np.ndarray, r:np.ndarray, b: np.ndarray):
        self.center = c
        self.top = t
        self.left = l
        self.right = r
        self.bottom = b


def draw_lines_on_img(img, sector_num, is_main_detection=True):
    """Функция рисует линии на изображении согласно файлу lines_ (для UNET)"""
    add = ''
    if not is_main_detection:
        add = '../'
    img_copy = img.copy()
    path_to_line = add + '{}{}'.format(kzsg_utils.path_to_lines,sector_num)
    lines = np.array(np.loadtxt(path_to_line, delimiter=','), np.float32)
    for line in lines:
        cv2.line(img_copy, (line[1], 0), (line[1], img_copy.shape[0]), (0, 255, 0), thickness=10)
        cv2.line(img_copy, (line[2], 0), (line[2], img_copy.shape[0]), (0, 0, 255), thickness=10)
        img_copy = cv2.rotate(img_copy, cv2.ROTATE_90_CLOCKWISE)
        cv2.putText(img_copy, str(int(line[0])), (int(img_copy.shape[0]/2),int((line[1]+line[2])/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 7)
        img_copy = cv2.rotate(img_copy, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_copy


def get_way(cnt_or_points, sector_num, nn_type, is_main_detection):
    """Функция возвращает путь, к которому относится заданный контур (для UNET) или точки (для YOLO)"""
    path_to_line = '{}{}'.format(kzsg_utils.path_to_lines,sector_num)
    lines = np.array(np.loadtxt(path_to_line, delimiter=','), np.float32)
    min = 99999
    idx = -1
    if nn_type == 'unet':
        cx = cnt_or_points.center[0]
    elif nn_type == 'yolo':
        cx = cnt_or_points[0] + int((cnt_or_points[2] - cnt_or_points[0]) / 2)
    for line in lines:
        way_no = line[0]
        x_left = line[1]
        x_right = line[2]
        dist = abs(cx-(x_left+x_right)/2)
        if dist < min:
            idx = way_no
            min = dist
    return int(idx)


def create_diagram_cv(predicted_vals, img, start_point = 0, end_point = 500,way_nums = None, dist_detailing = 20,SHOW_IMG=False):
    """Функция создает диаграмму КЗСГ с помощью OpenCV"""
    w = img.shape[1]
    h = img.shape[0]
    if way_nums is None:
        way_nums = range(0,20)
    dnry, diagram = create_empty_diagram_cv(start_point,end_point,way_nums,dist_detailing,w,h)

    for val in predicted_vals:
        diagram = draw_line_on_diagram(diagram,dnry,val['start_point'],val['end_point'],val['way'])
    if SHOW_IMG:
        kzsg_utils.show_and_destroy_img("KZSG_diagram", diagram)
    return diagram


def create_diagram(num_of_sector, predicted_vals, SHOW_IMG = False):
    """Функция создает диаграмму КЗСГ с помощью Matplotlib"""
    way_nums = get_occupied_ways(num_of_sector)
    x = []
    y = []
    for val in predicted_vals:
        x.append(val['way'])
        y.append((val['start_point'], val['end_point']))
        print(val['way'], val['start_point'], val['end_point'])
    matplotlib.use('Agg')
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot((x, x), ([i for (i, j) in y], [j for (i, j) in y]), c='red', linewidth=10)

    try:
        for i in range(way_nums[0], way_nums[-1] + 1):
            ax.set_xticks(range(way_nums[0], way_nums[-1] + 1))
            ax.plot([i] * (max(max(y)) - min(min(y))), range(min(min(y)), max(max(y))), c='black', linewidth=2,
                    linestyle='dashed')
    except:
        pass

    plt.xlim(way_nums[0], way_nums[-1])
    plt.xlabel('Номер пути')
    plt.ylabel('Расстояние, м')
    plt.savefig('fig.png')
    f.canvas.draw()
    img = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(f.canvas.get_width_height()[::-1] + (3,))
    cv2.imwrite("111.png",img)
    # im = fig2img(f)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    #
    img = cv2.resize(img, (1920, 1080))
    img = cv2.resize(img, (1920, 1080))
    if SHOW_IMG:
        kzsg_utils.show_and_destroy_img("KZSG_diagram", img)
    return img


def get_occupied_ways(sector_num):
    """Функция возвращает номера путей в секторе"""
    way_nums = []
    path_to_line = '{}{}'.format(kzsg_utils.path_to_lines,sector_num)
    with open(path_to_line) as f:
        for line in f:
            ns = line.split(sep=',')
            if len(ns) == 3:
                way_nums.append(int(float(ns[0])))
    return way_nums


def get_contours(mask, num_of_sector, SHOW_IMG = False):
    """Функция возвращает информацию о контурах в маске (изображении), полученной от нейросети UNET"""
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(mask, (1, 201), 0)
    mask = kzsg_utils.draw_lines_on_mask(mask, num_of_sector)

    kzsg_utils.show_and_destroy_img('mask', mask)

    ret, thresh = cv2.threshold(mask, 220, 255, 0)

    kzsg_utils.show_and_destroy_img('thresh', thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 6)
    kzsg_utils.show_and_destroy_img('contours', img)

    convex_hulls = []
    for cnt in contours:
        convex_hulls.append(cv2.convexHull(cnt, False))

    convex_hulls_img = np.zeros(mask.shape, np.uint8)

    if SHOW_IMG:
        cv2.drawContours(img, convex_hulls, -1, (255, 255, 255), 6)
        kzsg_utils.show_and_destroy_img("convex_hulls1",img)

    ys = []
    extreme_pts = []
    try:
        for cnt in convex_hulls:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            c = np.asarray([cX, cY])
            left = cnt[cnt[:, :, 0].argmin()][0]
            r = cnt[cnt[:, :, 0].argmax()][0]
            t = cnt[cnt[:, :, 1].argmin()][0]
            b = cnt[cnt[:, :, 1].argmax()][0]
            extreme_pts.append(ExtremePoint(c, t, r, left, b))
            ys.append([min(b), max(t)])

    except:
        pass

    cv2.drawContours(convex_hulls_img, convex_hulls, -1, (255, 255, 255), 6)
    for pt in extreme_pts:
        cv2.circle(convex_hulls_img, (pt.center[0], pt.center[1]),2,(255,255,255),26)
        cv2.circle(convex_hulls_img, (pt.top[0], pt.top[1]), 2, (255, 255, 255), 26)
        cv2.circle(convex_hulls_img, (pt.right[0], pt.right[1]), 2, (255, 255, 255), 26)
        cv2.circle(convex_hulls_img, (pt.left[0], pt.left[1]), 2, (255, 255, 255), 26)
        cv2.circle(convex_hulls_img, (pt.bottom[0], pt.bottom[1]), 2, (255, 255, 255), 26)
    convex_hulls_img = np.expand_dims(convex_hulls_img, axis=-1)
    if SHOW_IMG:
        kzsg_utils.show_and_destroy_img("convex_hulls",convex_hulls_img)

    return extreme_pts, convex_hulls_img


def check_prediction(way, start_dist,end_dist):
    """Функция возвращает флаг, означающий, что область, ограниченная двумя точками, является подвижной единицей"""
    flag = True
    if way < 0 or (end_dist-start_dist) < kzsg_utils.min_car_length:
        flag = False
    return flag


def warp_img(img, m, w=None, h=None):
    """Функция возвращает изображение с преобразованной по матрице m перспективой"""
    if (w,h) == (None, None):
        h,w = img.shape[:2]
    img_warped = cv2.warpPerspective(img, m, (w, h))
    return img_warped


def get_predicted_vals_yolo(sectors_info, points_list, num_of_sector, M):
    """Функция возвращает предсказанные нейросетью YOLO значения в метрах"""
    predicted_vals = []
    for points in points_list:
        if points[1] > 0:
            way = get_way(points, num_of_sector, 'yolo', True)
            x_min = points[0]
            y_min = points[3]
            x_max = points[2]
            y_max = points[1]
            x_c = x_min + int((x_max-x_min)/2)
            start_dist = sectors_info[num_of_sector-1].get_D_via_point_warped((x_c, y_min), way, M)
            end_dist = sectors_info[num_of_sector-1].get_D_via_point_warped((x_c, y_max), way, M)
            predicted_vals.append({'way':way, 'start_point':start_dist, 'end_point':end_dist})
    return predicted_vals


def get_predicted_vals_unet(cv_imgp, num_of_sector, sectors_info, warping_matrix, SHOW_IMG: bool = False):
    """Функция возвращает предсказанные нейросетью UNET значения в метрах"""
    cnts,convex_hulls_img = get_contours(cv_imgp, num_of_sector, SHOW_IMG)
    predicted_vals = []

    for cnt in cnts:
        way = get_way(cnt, num_of_sector, 'unet')

        start_pix = kzsg_utils.from_warped_to_real(cnt.bottom[0], cnt.bottom[1], warping_matrix)
        start_dist = sectors_info[num_of_sector - 1].get_D_via_point((start_pix[0], start_pix[1]), way)
        left_pix = cnt.left[0]
        right_pix = cnt.right[0]
        end_pix = kzsg_utils.from_warped_to_real(cnt.top[0], cnt.top[1], warping_matrix)
        end_dist = sectors_info[num_of_sector - 1].get_D_via_point((end_pix[0], end_pix[1]), way)
        numbers = kzsg_utils.get_lines_coords(num_of_sector)
        condition = True
        for num in numbers:
            if num[0] == way:
                if left_pix < num[1]:
                    left = num[1]
                else:
                    left = num[1] - left_pix
                if right_pix < num[2]:
                    right = num[2] - right_pix
                else:
                    right = num[2]
                if right - left < round((num[2]-num[1])/2):
                    condition = False
        if check_prediction(way, start_dist, end_dist) and np.abs(left_pix - right_pix) > 50 and condition:
            predicted_vals.append({'way':way, 'start_point':start_dist, 'end_point':end_dist})
    return predicted_vals,convex_hulls_img


def get_distances(sectors_info, num_of_sector):
    """Функция возвращает значения стартовой и конечной точек для нормирования диаграммы КЗСГ"""
    sector_info = sectors_info[num_of_sector-1]
    start_point = sector_info.ADD
    max_val = 0
    for item in sector_info.L:
        if item > max_val:
            max_val = item
    end_point = max_val + start_point
    return start_point, end_point


def get_warping_matrix(num_of_sector, is_main_detection=False):
    """Функция возвращает матрицу преобразования перспективы по номеру сектора"""

    path_to_coefs = '{}{}'.format(kzsg_utils.warping_points_path, num_of_sector)
    if not os.path.exists(path_to_coefs):
        raise Exception('warp matrix for sector {} does not exists at {}'.format(num_of_sector,path_to_coefs))
    elif os.stat(path_to_coefs).st_size == 0:
        raise Exception('warp matrix for sector {} is empty'.format(num_of_sector))
    else:
        M = np.loadtxt(path_to_coefs, delimiter=',')
        shape = M.shape
        if shape != (3, 3):
            raise Exception('warp matrix has wrong shape')
        else:
            warping_matrix = np.array(M, np.float32)
    return warping_matrix


def prepare_img(frame_orig, num_of_sector, warping_matrix, pre_warp=False):
    """Функция возвращает преобразованное по матрице преобразования перспективы изображение,
    если необходимо, для входа нейросети UNET"""
    if pre_warp:
        frame = warp_img(frame_orig, warping_matrix, frame_orig.shape[1], frame_orig.shape[0])
    else:
        frame = frame_orig.copy()
    return frame


def predict_unet(frame, model, warping_matrix, pre_warp=False, use_gpu=False):
    """Функция возвращает предсказанную нейросетью UNET маску"""
    (H, W) = frame.shape[:2]
    img = transform.resize(frame, (NEURAL_SIZE, NEURAL_SIZE))

    TEST = np.zeros((1, NEURAL_SIZE, NEURAL_SIZE, 3), dtype='float32')  # float32
    TEST[0] = img
    if use_gpu:
        str_device = "/device:GPU:0"
    else:
        str_device = "/device:CPU:0"
    with tf.device(str_device):
        prediction = model.predict(TEST)

    imgp = transform.resize(prediction[0], (H, W, 3))
    cv_imgp = (imgp * 255).astype('uint8')
    if not pre_warp:
        cv_imgp = warp_img(cv_imgp, warping_matrix, W, H)
    return cv_imgp



def warp_images(files,num_sectors):
    """Функция принимат на вход список файлов, находящихся в директории TEST/, и номер сектора, и сохраняет фотографии
    с преобразованной перспективой в директорию test_warped/images/"""
    for file in files:
        img1 = cv2.imread(file)
        shape = img1.shape[:2][::-1]
        img1 = cv2.resize(img1, shape)
        warp_matrix = get_warping_matrix(num_sectors, True)
        warped_img = warp_img(img1, warp_matrix)
        filename = file.replace(TEST_PATH, TEST_IMG_PATH)
        cv2.imwrite(filename, warped_img)


def yolo_work():
    """Функция запускает работу нейронной сети YOLO для фотографий, находящихся в директории test_warped/images/"""
    bashCommand = "python3 detect.py --weights weights/yolov5s.pt --img 416  --save-txt --source test_warped/images/"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    outp, err = process.communicate()
    print(outp, err)


def bounding_boxes(img, filename):
    """Функция строит bounding boxes на черном изображении и создает маску"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0
    points = []
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        points.append((x, y, w, h))

    zeros = np.zeros(gray.shape)
    for point in points:
        x, y, w, h = point
        if w > 40 and w * h > 1600:
            zeros = cv2.rectangle(zeros, (x, y), (x + w, y + h), (200, 0, 0), 2)
    cv2.imwrite(filename, zeros)
    return points


def get_points(txtfile, img):
    """Функция принимает на вход txt-файлы с разметкой YOLO и возвращает массив массивов точкек [xmin, ymin, xmax, ymax]
     для всех прямоугольников"""
    points = list()
    if os.path.exists(txtfile):
        with open(txtfile, 'r') as f:
            points = []
            for line in f:
                nums = line.strip().split(sep=' ')
                nums = [float(num) for num in nums]
                xc = int(nums[1]*img.shape[1])
                yc = int((nums[2])*img.shape[0])
                w = int(nums[3]*img.shape[1])
                h = int(nums[4]*img.shape[0])
                xmin = xc - int(w/2)
                xmax = xc + int(w/2)
                ymin = yc - int(h/2)
                ymax = yc + int(h/2)
                if w * h > 10000:
                    points.append([xmin, ymin, xmax, ymax])
    else:
        print("there is no detection for {}".format(txtfile))
    return points


def draw_rect(txtfile, img):
    """Функция для рисования прямоугольника по txt-фалам разметки YOLO на изображении img"""
    points_list = get_points(txtfile, img)
    for points in points_list:
        img = cv2.rectangle(img, (points[0], points[1]), (points[2], points[3]), (255, 0, 0), 5)
    return img


def get_sectors_info(num_sectors):
    """Функция возвращает информацию о всех секторах (список объектов Sector)"""
    sectors_info = []
    for i in range(num_sectors):
        sectors_info.append(calibr.Sector(i + 1))
    return sectors_info


def check_matrix(warping_matrix):
    """Функция проверяет наличие матрицы преобразования перспективы"""
    flag = True
    if warping_matrix is None:
        flag = False
    return flag


def run_unet(num_sectors, PATH_TO_UNET, RM_PREDICTION_DIR, IS_VIDEO, ONLY_DAY, SHOW_IMG, USE_GPU, PRE_WARP, CONCAT, WRITE_VIDEO, SAVE_PREDICTION):
    """Функция запускает отработку проекта КЗСГ с помощью нейросети UNET"""
    cwd = os.getcwd()
    path = cwd + "/TEST/"
    predicted_path = cwd + "predicted/"
    model = tf.keras.models.load_model(PATH_TO_UNET)
    sectors_info = get_sectors_info(num_sectors=num_sectors)
    ## если надо пересоздать директорию с предсказаниями
    if RM_PREDICTION_DIR or not os.path.exists(predicted_path):
        kzsg_utils.rm_mk_dirs([predicted_path])
    ##

    ## вычитывание видео или папки с изображениями
    if IS_VIDEO:
        vs = cv2.VideoCapture("rec.mp4")
        frames_total = frames_count("rec.mp4")
    else:
        files = kzsg_utils.get_jpg_filenames(path)
        if not files:
            files = kzsg_utils.get_jpg_filenames(path, find_png=True)
        file_names = [f.split("/")[-1] for f in files]
        frames_total = len(files)
    print("Total number of frames: ", frames_total)
    ##

    writer = None
    ## Создание предиктора
    ##

    # pbar = tqdm(total=frames_total)
    for i in range(0, frames_total):
        # pbar.update(1)
        ## чтение кадра
        if IS_VIDEO:
            (grabbed, frame_orig) = vs.read()
            if not grabbed:
                break
            frame_title = str(i) + ".jpg"
        else:
            num_of_sector = kzsg_utils.parce_name(files[i], num_sectors, num_of_sector)
            frame_orig = cv2.imread(files[i])
            base = files[i].split("/")[-1].split(".")[0].split("_")[0]

            if ONLY_DAY:
                try:
                    date_time_obj = datetime.datetime.strptime(base, '%Y-%m-%d %H')
                    if not 4 < date_time_obj.hour < 14:
                        continue
                except Exception as e:
                    print(e)
                    pass
            frame_title = file_names[i]
        ##

        # ## проверка наличия матрицы для преобразования перспективы
        # if not check_matrix(num_of_sector):
        #     print("warp matrix is empty, please refer to rail_founder.py")
        #     continue
        ##
        ## подготовка изображения и получение изображения cv_imgp с предсказанными масками
        frame = prepare_img(frame_orig, num_of_sector, warping_matrix)

        if SHOW_IMG:
            kzsg_utils.show_and_destroy_img("frame " + frame_title, frame)
        frame = kzsg_utils.rotate_img(frame, num_of_sector)
        cv_imgp = predict_unet(frame, model, warping_matrix, use_gpu=USE_GPU)
        ##

        ## постпреобразование перспективы (если необходимо) и соединение двух изображений - оригинала и преобразованного
        if not PRE_WARP:

            frame_warped = warp_img(frame, get_warping_matrix(num_of_sector))
            if SHOW_IMG:
                frame_warped_with_lines = draw_lines_on_img(frame_warped, num_of_sector)
                kzsg_utils.show_and_destroy_img("warped_with_lines " + frame_title, frame_warped_with_lines)
            numbers = kzsg_utils.get_lines_coords(num_of_sector)
            x1 = numbers[0][1]
            x2 = numbers[-1][2]

            frame_warped = frame_warped[:, x1:x2]
            frame_warped = cv2.resize(frame_warped, (1920, 1080))
            left_frame = np.zeros((frame_warped.shape[0], 0, 3), np.uint8)
            right_frame = np.zeros((frame_warped.shape[0], 0, 3), np.uint8)

            frame_warped = cv2.resize(frame_warped, (
                frame_warped.shape[1] - left_frame.shape[1] - right_frame.shape[1], frame_warped.shape[0]))
            frame_warped = cv2.hconcat([left_frame, frame_warped, right_frame])

            frame = kzsg_utils.unrotate_img(frame, num_of_sector)
            frame = cv2.resize(frame, (frame_warped.shape[1], frame_warped.shape[0]))

            if CONCAT:
                frame = cv2.vconcat([frame, frame_warped])
        elif CONCAT:
            frame = cv2.vconcat([frame_orig, frame])

        if SHOW_IMG:
            kzsg_utils.show_and_destroy_img("warped " + frame_title, frame)
            kzsg_utils.show_and_destroy_img("mask " + frame_title, cv_imgp)
        ##

        ## получение предсказанных значений пути, начала и конца области для каждой маски
        predicted_vals, convex_hulls_img = get_predicted_vals_unet(cv_imgp, num_of_sector, sectors_info, warping_matrix, SHOW_IMG)

        blank_img = np.zeros((10, cv_imgp.shape[1], 1), dtype='uint8')
        blank_img.fill(255)
        convex_hulls_img = cv2.vconcat([cv_imgp[:, :, 0], blank_img, convex_hulls_img])
        ##
        ####################################################################################################################
        # convex_hulls_img = kzsg_utils.bounding_boxes(convex_hulls_img)

        ## получение диаграммы КЗСГ
        # start_point = 0
        start_point, end_point = get_distances(sectors_info, num_of_sector)
        way_nums = get_occupied_ways(num_of_sector)
        kzsg_img = create_diagram_cv(predicted_vals, frame_warped, start_point, end_point, way_nums, SHOW_IMG=SHOW_IMG)
        # kzsg_img = seg_utils.create_diagram(num_of_sector, predicted_vals)
        #### проверка наличия матрицы для преобразования перспективы

        frame = cv2.resize(frame, (1920, 2160))
        if CONCAT:
            cv_imgp = cv2.vconcat([frame, kzsg_img])

            if SHOW_IMG:
                kzsg_utils.show_and_destroy_img("processed frame " + frame_title, cv_imgp)
        if WRITE_VIDEO:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter('NIIAS.avi', fourcc, 30, (W, H), True)

            writer.write(cv_imgp)
        else:
            cv2.imwrite(predicted_path + frame_title, cv_imgp)
            if SAVE_PREDICTION:
                new_name = predicted_path + os.path.splitext(frame_title)[0] + "_mask" + os.path.splitext(frame_title)[
                    1]
                cv2.imwrite(new_name, np.expand_dims(convex_hulls_img, -1))
    if WRITE_VIDEO:
        writer.release()
        vs.release()
    close()


def run_yolo(num_sectors, way_nums, SHOW_IMG=False):
    """Функция запускает отработку проекта КЗСГ с помощью нейросети YOLO"""
    files = glob(TEST_IMG_PATH + '*.jpg')
    for file in files:
        os.remove(file)

    files = glob(TEST_PATH + '*.jpg')

    warp_images(files,num_sectors)
    os.chdir('yolov5/')
    yolo_work()
    os.chdir('../')
    sectors_info = get_sectors_info(num_sectors=num_sectors)
    for file in files:
        num_of_sector = kzsg_utils.parce_name(file,num_sectors)
        way_nums = calibr.Sector(num_of_sector).WAY
        warp_matrix = get_warping_matrix(num_of_sector, True)

        img1 = cv2.imread(file)
        img1 = kzsg_utils.rotate_img(img1, num_of_sector)

        warped_img = warp_img(img1, warp_matrix)

        id_ = file.replace('.jpg', '')
        id_ = id_.replace(TEST_IMG_PATH, '')
        id_ = id_.replace(TEST_PATH, '')

        img = cv2.imread(TEST_IMG_PATH + id_ + '.jpg')

        txtfile = TXT_FILES_PATH + id_ + '.txt'
        points_list = get_points(txtfile, img)
        out = img.copy()
        print(txtfile)
        for points in points_list:
            out = kzsg_utils.draw_rect_with_corners(points[0], points[1], points[2], points[3], out)
        points_list = get_points(txtfile, img)
        warp_matrix = get_warping_matrix(num_of_sector, True)


        predicted_vals = get_predicted_vals_yolo(sectors_info, points_list, num_of_sector, warp_matrix)

        start_point, end_point = get_distances(sectors_info, num_of_sector)
        diagram = create_diagram_cv(predicted_vals, img, start_point, end_point, way_nums, SHOW_IMG=False)
        print(predicted_vals)
        img1 = kzsg_utils.unrotate_img(img1, num_of_sector)
        pics = np.vstack((img1, out, diagram))
        if SHOW_IMG:
            kzsg_utils.show_and_destroy_img('output', pics)
        new_path = './predicted/'
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        new_path += file.replace('TEST/', '')
        cv2.imwrite(new_path, pics)


def creat_folder(num_of_sector):
    """Функция создает папки и подпапки Traindata"""
    os.mkdir('traindata/Traindata({})'.format(num_of_sector))
    os.mkdir('traindata/Traindata({})/images'.format(num_of_sector))
    os.mkdir('traindata/Traindata({})/labels'.format(num_of_sector))


def get_num_of_sector(nameIm):
    """Функция возвращает номер сектора по имени файла"""
    a = nameIm.split('.')
    b = a[-2].split('_')
    try:
        c = int(b[-1])
        num = c
    except ValueError:
        c = b[-1].split('(')
        num = c[0]
    return num


def get_name(nameIm):
    """Функция возвращает имя по имени файла (отрезает разширение)"""
    name = ''
    a = nameIm.split('.')
    for i in range(len(a) - 1):
        name += a[i]
    return  name
