import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all paths in memory"""

    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images = data['x']
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = Image.fromarray(self.images[index])
        if isinstance(self.transform,list):
            sample1 = self.transform[0](x)
            sample2 = self.transform[1](x)
            y = self.labels[index]
            return [sample1, sample2], y
        else:
            x = self.transform(x)
            y = self.labels[index]
            return x, y

    # def __getitem__(self, index):
    #     """Generates one sample of data"""
    #     x = Image.fromarray(self.images[index])
    #     x = self.transform(x)
    #     y = self.labels[index]
    #     return x, y


def get_data(path, num_tasks, nc_first_task, validation, shuffle_classes, class_order=None, trn_lst=None, tst_lst=None):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    taskcla = []

    # read filenames and labels
    if trn_lst is None and tst_lst is None:
        trn_lines = np.loadtxt(os.path.join(path, 'train.txt'), dtype=str)
        tst_lines = np.loadtxt(os.path.join(path, 'test.txt'), dtype=str)
    else:
        trn_lines = np.loadtxt(os.path.join(path, trn_lst), dtype=str)
        tst_lines = np.loadtxt(os.path.join(path, tst_lst), dtype=str)

    if class_order is None:
        num_classes = len(np.unique(trn_lines[:, 1]))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()

    # if shuffle_classes:
    #     np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    if nc_first_task is None:
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < num_classes, "first task wants more classes than exist"
        remaining_classes = num_classes - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
        cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1

    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['val'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}

    # ALL OR TRAIN
    for this_image, this_label in trn_lines:
        if not os.path.isabs(this_image):
            this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x'].append(this_image)
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])

    # ALL OR TEST
    for this_image, this_label in tst_lines:
        if not os.path.isabs(this_image):
            this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x'].append(this_image)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    if validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0])
                rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['val']['x'].append(data[tt]['trn']['x'][rnd_img[ii]])
                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_img[ii]])
                    data[tt]['trn']['x'].pop(rnd_img[ii])
                    data[tt]['trn']['y'].pop(rnd_img[ii])

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order


from torchvision import datasets
class myImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        return path, target


def get_data_imagenet(path, num_tasks, nc_first_task, validation, shuffle_classes, class_order=None, trn_lst=None, tst_lst=None):
    """Prepare data: dataset splits, task partition, class order"""
    
    data = {}
    taskcla = []

    # read filenames and labels
    traindir = os.path.join(path, 'train')
    valdir = os.path.join(path, 'val')
    trainset = myImageFolder(traindir)
    testset =  myImageFolder(valdir)

    num_classes = len(class_order)
    class_order = class_order.copy()
    # if shuffle_classes:
    #     np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    if nc_first_task is None:
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < num_classes, "first task wants more classes than exist"
        remaining_classes = num_classes - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
        cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1

    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['val'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}

    # ALL OR TRAIN
    for this_image, this_label in trainset:
        # if not os.path.isabs(this_image):
        #     this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x'].append(this_image)
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])

    # ALL OR TEST
    for this_image, this_label in testset:
        # if not os.path.isabs(this_image):
        #     this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x'].append(this_image)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    if validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0])
                rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['val']['x'].append(data[tt]['trn']['x'][rnd_img[ii]])
                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_img[ii]])
                    data[tt]['trn']['x'].pop(rnd_img[ii])
                    data[tt]['trn']['y'].pop(rnd_img[ii])

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order



class myFullImageFolder(datasets.ImageFolder):
    def __init__(self, root, train_flag):
        self.train_flag = train_flag
        super().__init__(root, None)

    def __getitem__(self, index):
        path, target = self.samples[index]
        return path, target

    def find_classes(self, directory: str):
        # classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        # if not classes:
        #     raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        # class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        directory = './data/imagenet/train'
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

        data_path = './data/imagenet/meta/train.txt'
        data_lines = np.loadtxt(data_path, dtype=str)
        class_to_idx = {}
        for this_image, this_label in data_lines:
            class_to_idx.update({str(this_image.split('/')[0]): int(this_label)})
        return classes, class_to_idx

    def make_dataset(self, directory: str, class_to_idx = None,
        extensions = None, is_valid_file = None):
        instances = []
        if self.train_flag == 'train':
            data_path = './data/imagenet/meta/train.txt'
            data_lines = np.loadtxt(data_path, dtype=str)
        elif self.train_flag == 'val':
            data_path = './data/imagenet/meta/val.txt'
            data_lines = np.loadtxt(data_path, dtype=str)
        elif self.train_flag == 'test':
            data_path = './data/imagenet/meta/test.txt'
            data_lines = np.loadtxt(data_path, dtype=str)

        for this_image, this_label in data_lines:
            # print(this_image, this_label)
            if not os.path.isabs(this_image):
                this_image = os.path.join(directory, this_image)
                if this_image.split('.')[1] != 'JPEG':
                    this_image = this_image.split('.')[0] + '.JPEG'
                assert os.path.exists(this_image), print(this_image, this_label)
            this_label = int(this_label)
            item = this_image, this_label
            instances.append(item)
        return instances

def get_data_fullimagenet(path, num_tasks, nc_first_task, validation, shuffle_classes, class_order=None, trn_lst=None, tst_lst=None):
    """Prepare data: dataset splits, task partition, class order"""
    data = {}
    taskcla = []

    # read filenames and labels
    traindir = os.path.join(path, 'train')
    valdir = os.path.join(path, 'val')
    trainset = myFullImageFolder(traindir, 'train')
    testset =  myFullImageFolder(valdir, 'val')

    num_classes = len(class_order)
    class_order = class_order.copy()
    # if shuffle_classes:
    #     np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    if nc_first_task is None:
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < num_classes, "first task wants more classes than exist"
        remaining_classes = num_classes - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
        cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1

    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['val'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}

    # ALL OR TRAIN
    for this_image, this_label in trainset:
        # if not os.path.isabs(this_image):
        #     this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x'].append(this_image)
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])

    # ALL OR TEST
    for this_image, this_label in testset:
        # if not os.path.isabs(this_image):
        #     this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x'].append(this_image)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    if validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0])
                rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['val']['x'].append(data[tt]['trn']['x'][rnd_img[ii]])
                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_img[ii]])
                    data[tt]['trn']['x'].pop(rnd_img[ii])
                    data[tt]['trn']['y'].pop(rnd_img[ii])

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order
