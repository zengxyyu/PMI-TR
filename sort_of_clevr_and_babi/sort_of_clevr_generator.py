import cv2
import os
import numpy as np
import random
#import cPickle as pickle
import pickle
import warnings
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Sort-of-CLEVR dataset generator')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--t-subtype', type=int, default=-1,
                    help='Force ternary questions to be of a given type')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

train_size = 9800
test_size = 200
img_size = 75
size = 5
question_size = 18  ## 2 x (6 for one-hot vector of color), 3 for question type, 3 for question subtype
q_type_idx = 12
sub_q_type_idx = 15
"""Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""

nb_questions = 10
dirs = './data'

colors = [
    (0,0,255),##r
    (0,255,0),##g
    (255,0,0),##b
    (0,156,255),##o
    (128, 128, 128),##k
    (0,255,255)##y
]


try:
    os.makedirs(dirs)
except:
    print('directory {} already exists'.format(dirs))

def center_generate(objects):
    while True:
        pas = True
        # 成的两个随机整数分别表示物体中心点在图像中的横坐标和纵坐标
        center = np.random.randint(0+size, img_size - size, 2)        
        if len(objects) > 0:
            for name,c,shape in objects:
                # 使用了 NumPy 库来进行向量化计算，
                # 其中 ((center - c) ** 2).sum() 表示计算当前中心点与列表中每个对象的中心点之间的欧几里得距离，
                # 并将所有距离平方相加，这样可以避免使用循环来计算每个对象的距离，从而提高代码效率。
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                    pas = False
        if pas:
            return center



def build_dataset():
    # 将当前对象的颜色 ID、中心点 center 和形状（'r'表示矩形，'c'表示圆形）
    # 作为一个元组 (color_id, center, shape) 添加到对象列表 objects 中。
    objects = []
    # 创建大小为75*75的三通道图片，每个通道值为255，即75*75的白色图片
    img = np.ones((img_size,img_size,3),dtype=np.uint8) * 255
    # 利用for循环产生75*75的图片，背景为白色，包含6个物体（矩形或者圆），6种可选颜色
    for color_id,color in enumerate(colors):
        center = center_generate(objects)
        # 然后根据一个随机的概率（random.random() < 0.5）选择绘制一个矩形或圆形
        # 绘制矩形
        if random.random()<0.5:
            start = (center[0]-size, center[1]-size)
            end = (center[0]+size, center[1]+size)
            cv2.rectangle(img, start, end, color, -1)
            objects.append((color_id,center,'r'))
        # 绘制圆
        else:
            center_ = (center[0], center[1])
            cv2.circle(img, center_, size, color, -1)
            objects.append((color_id,center,'c'))

    # 输出产生的图片
    cv2.namedWindow('image_show', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image_show', 200, 200)
    cv2.imshow('image_show', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ternary_questions = []
    binary_questions = []
    # 非关系型问题
    norel_questions = []
    ternary_answers = []
    binary_answers = []
    norel_answers = []

    """Non-relational questions"""
    for _ in range(nb_questions):
        # question初始化为[0,0,0,...0] 18维
        question = np.zeros((question_size))
        # 随机选择一种颜色
        color = random.randint(0,5)
        question[color] = 1
        question[q_type_idx] = 1
        subtype = random.randint(0,2)
        question[subtype+sub_q_type_idx] = 1
        norel_questions.append(question)
        """Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""
        if subtype == 0:
            """query shape->rectangle/circle"""
            if objects[color][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 1:
            """query horizontal position->yes/no"""
            if objects[color][1][0] < img_size / 2:
                answer = 0
            else:
                answer = 1

        elif subtype == 2:
            """query vertical position->yes/no"""
            if objects[color][1][1] < img_size / 2:
                answer = 0
            else:
                answer = 1
        norel_answers.append(answer)
    
    """Binary Relational questions"""
    for _ in range(nb_questions):
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[q_type_idx+1] = 1
        subtype = random.randint(0,2)
        question[subtype+sub_q_type_idx] = 1
        binary_questions.append(question)

        if subtype == 0:
            """closest-to->rectangle/circle"""
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            dist_list[dist_list.index(0)] = 999
            closest = dist_list.index(min(dist_list))
            if objects[closest][2] == 'r':
                answer = 2
            else:
                answer = 3
                
        elif subtype == 1:
            """furthest-from->rectangle/circle"""
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            furthest = dist_list.index(max(dist_list))
            if objects[furthest][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 2:
            """count->1~6"""
            my_obj = objects[color][2]
            count = -1
            for obj in objects:
                if obj[2] == my_obj:
                    count +=1 
            answer = count+4

        binary_answers.append(answer)

    """Ternary Relational questions"""
    for _ in range(nb_questions):
        question = np.zeros((question_size))
        rnd_colors = np.random.permutation(np.arange(5))
        # 1st object
        color1 = rnd_colors[0]
        question[color1] = 1
        # 2nd object
        color2 = rnd_colors[1]
        question[6 + color2] = 1

        question[q_type_idx + 2] = 1
        
        if args.t_subtype >= 0 and args.t_subtype < 3:
            subtype = args.t_subtype
        else:
            subtype = random.randint(0, 2)

        question[subtype+sub_q_type_idx] = 1
        ternary_questions.append(question)

        # get coordiantes of object from question
        A = objects[color1][1]
        B = objects[color2][1]

        if subtype == 0:
            """between->1~4"""

            between_count = 0 
            # check is any objects lies inside the box
            for other_obj in objects:
                # skip object A and B
                if (other_obj[0] == color1) or (other_obj[0] == color2):
                    continue

                # Get x and y coordinate of third object
                other_objx = other_obj[1][0]
                other_objy = other_obj[1][1]

                if (A[0] <= other_objx <= B[0] and A[1] <= other_objy <= B[1]) or \
                   (A[0] <= other_objx <= B[0] and B[1] <= other_objy <= A[1]) or \
                   (B[0] <= other_objx <= A[0] and B[1] <= other_objy <= A[1]) or \
                   (B[0] <= other_objx <= A[0] and A[1] <= other_objy <= B[1]):
                    between_count += 1

            answer = between_count + 4
        elif subtype == 1:
            """is-on-band->yes/no"""
            
            grace_threshold = 12  # half of the size of objects
            epsilon = 1e-10  
            m = (B[1]-A[1])/((B[0]-A[0]) + epsilon ) # add epsilon to prevent dividing by zero
            c = A[1] - (m*A[0])

            answer = 1  # default answer is 'no'

            # check if any object lies on/close the line between object A and object B
            for other_obj in objects:
                # skip object A and B
                if (other_obj[0] == color1) or (other_obj[0] == color2):
                    continue

                other_obj_pos = other_obj[1]
                
                # y = mx + c
                y = (m*other_obj_pos[0]) + c
                if (y - grace_threshold)  <= other_obj_pos[1] <= (y + grace_threshold):
                    answer = 0
        elif subtype == 2:
            """count-obtuse-triangles->1~6"""

            obtuse_count = 0

            # disable warnings
            # the angle computation may fail if the points are on a line
            warnings.filterwarnings("ignore")
            for other_obj in objects:
                # skip object A and B
                if (other_obj[0] == color1) or (other_obj[0] == color2):
                    continue

                # get position of 3rd object
                C = other_obj[1]
                # edge length
                a = np.linalg.norm(B - C)
                b = np.linalg.norm(C - A)
                c = np.linalg.norm(A - B)
                # angles by law of cosine
                alpha = np.rad2deg(np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))
                beta = np.rad2deg(np.arccos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c)))
                gamma = np.rad2deg(np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))
                max_angle = max(alpha, beta, gamma)
                if max_angle >= 90 and max_angle < 180:
                    obtuse_count += 1

            warnings.filterwarnings("default")
            answer = obtuse_count + 4

        ternary_answers.append(answer)

    ternary_relations = (ternary_questions, ternary_answers)
    binary_relations = (binary_questions, binary_answers)
    norelations = (norel_questions, norel_answers)
    
    img = img/255.
    # 数据集格式
    dataset = (img, ternary_relations, binary_relations, norelations)
    return dataset


if __name__ == "__main__":
    print('building test datasets...')
    #使用 tqdm 包装 range(test_size)，从而创建一个迭代器，它可以在终端上显示一个进度条
    test_datasets = [build_dataset() for _ in tqdm(range(test_size))]
    print('building train datasets...')
    train_datasets = [build_dataset() for _ in tqdm(range(train_size))]


    #img_count = 0
    #cv2.imwrite(os.path.join(dirs,'{}.png'.format(img_count)), cv2.resize(train_datasets[0][0]*255, (512,512)))


    print('saving datasets...')
    filename = os.path.join(dirs,'sort-of-clevr_zxy.pickle')
    # with open(filename, 'wb') as f:
    #     pickle.dump((train_datasets, test_datasets), f)
    # print('datasets saved at {}'.format(filename))
