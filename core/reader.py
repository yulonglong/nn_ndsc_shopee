import logging

logger = logging.getLogger(__name__)

def read_csv_train(csv_path):
    itemid = []
    title = []
    image_path = []

    y_multi = []
    class_titles = []
    class_max_number = []
    num_classes = 0

    # Read file and write into corresponding arrays
    with open(csv_path, "r") as infile:
        first = True
        for line in infile:
            line = line.strip()
            content = line.split(',')

            if (first):
                first = False
                class_titles = content[3:]
                num_classes = len(class_titles)
                for i in range(num_classes):
                    class_max_number.append(-1)
                    y_multi.append([])
            else:
                itemid.append(content[0])
                title.append(content[1])
                image_path.append(content[2])
                for i in range(num_classes):
                    if (len(content[i+3]) > 0):
                        if class_max_number[i] < int(float(content[i+3])):
                            class_max_number[i] = int(float(content[i+3]))
                        y_multi[i].append(int(float(content[i+3])))
                    else:
                        y_multi[i].append(None)
    return itemid, title, image_path, y_multi, class_titles, class_max_number, num_classes


def read_csv_final_test(csv_path):
    itemid = []
    title = []
    image_path = []

    # Read file and write into corresponding arrays
    with open(csv_path, "r") as infile:
        first = True
        for line in infile:
            line = line.strip()
            content = line.split(',')
            if (first):
                first = False
            else:
                itemid.append(content[0])
                title.append(content[1])
                image_path.append(content[2])
    return itemid, title, image_path

def create_vocab(titles):
    """Create Vocab from title"""
    x_title = []
    vocab = {"<pad>": 0, "<unk>": 1}
    for title in titles:
        words = title.strip().lower().split()
        x_title_curr = []
        for word in words:
            idx = 1
            if word in vocab:
                idx = vocab[word]
            else:
                idx = len(vocab)
                vocab[word] = len(vocab)
            x_title_curr.append(idx)
        x_title.append(x_title_curr)
    return vocab, x_title

def convert_word_to_idx_using_vocab(vocab, titles):
    """Create sequence of indices from title"""
    x_title = []
    for title in titles:
        words = title.strip().lower().split()
        x_title_curr = []
        for word in words:
            idx = 1 # index 1 represents unknown word
            if word in vocab:
                idx = vocab[word]
            x_title_curr.append(idx)
        x_title.append(x_title_curr)
    return x_title

#####################################
## Create img representation 
#

import cv2
import multiprocessing

global_img_dim = 500
def process_image(full_path):
    """Convert jpg image to numpy representation"""
    if (full_path[-4:] != ".jpg"):
        full_path += ".jpg"
    
    curr_image = cv2.imread(full_path)
    curr_image = cv2.resize(curr_image, (global_img_dim, global_img_dim), interpolation = cv2.INTER_AREA)
    # curr_image = cv2.resize(curr_image, (args.img_dim, args.img_dim), interpolation = cv2.INTER_AREA)
    # Optional, can normalize here or normalize later
    curr_image = cv2.normalize(curr_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    if (curr_image.shape[0] != curr_image.shape[1]):
        logger.error(curr_image.shape)
    return curr_image

def process_image_multiprocess(img_dim, img_path_list):
    global_img_dim = img_dim
    # Multiprocess image processing
    # P = multiprocessing.Pool(processes=(4))
    # train_x_img = P.map(process_image, img_path_list)
    # P.close()
    # P.join()

    # Single core is apparently faster
    train_x_img = []
    for i in range(len(img_path_list)):
        train_x_img.append(process_image(img_path_list[i]))
    return train_x_img

#######################################