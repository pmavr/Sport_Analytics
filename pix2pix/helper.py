
import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from numpy import savez_compressed
import os, glob, shutil
import tarfile

from utils import get_project_root


def read_homography_matrix(data):
    H = np.zeros((3, 3))
    for i in range(len(data)):
        H[i] = np.array([float(x) for x in data[i].strip().split()])
    return H


def show_image(img, msg=''):
    """
    Displays an image. Esc char to close window
    :param img: Image to be displayed
    :param msg: Optional message-title for the window
    :return:
    """
    cv2.imshow(msg, img)
    while 1:
        k = cv2.waitKey(0)
        if k == 27:
            break
    cv2.destroyWindow(msg)


def load_images(initial_path, relative_path):
    src_list, tar_list = list(), list()
    court_image_path = f'{initial_path}{relative_path}court_images/'
    homographies_path = f'{initial_path}{relative_path}homography_matrices/'
    bin_court = plt.imread(f'{initial_path}binary_court.jpg')

    dataset_length = os.listdir(court_image_path).__len__()

    for i in range(1, dataset_length):
        court_img = plt.imread('{}{}.jpg'.format(court_image_path, i))

        with open('{}{}.homographyMatrix'.format(homographies_path, i)) as homography_file:
            data = homography_file.readlines()
        matrix = read_homography_matrix(data)
        w = court_img.shape[0]
        h = court_img.shape[1]
        inv_matrix = np.linalg.inv(matrix)
        edge_map = cv2.warpPerspective(src=bin_court, M=inv_matrix, dsize=(h, w))

        src_list.append(court_img)
        tar_list.append(edge_map)
    return [np.asarray(src_list), np.asarray(tar_list)]


def load_real_samples(filename):
    data = np.load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


def generate_real_samples(dataset, n_samples, patch_shape):
    trainA, trainB = dataset
    ix = np.random.randint(0, trainA.shape[0], n_samples)
    X1, X2 = trainA[ix], trainB[ix]
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


def generate_fake_samples(g_model, samples, patch_shape):
    x = g_model.predict(samples)
    y = np.zeros((len(x), patch_shape, patch_shape, 1))
    return x, y


def summarize_performance(step, g_model, dataset, n_samples=3):
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_realA[i])

    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_fakeB[i])

    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples * 2 + i)
        plt.axis('off')
        plt.imshow(X_realB[i])

    filename1 = 'plot_%06d.png' % (step + 1)
    plt.savefig(filename1)
    plt.close()

    filename2 = 'model_%06d.h5' % (step + 1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    n_patch = d_model.output_shape[1]
    trainA, trainB = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs

    for i in range(n_steps):
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))

        if (i + 1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset)


def plot_images(src_img, gen_img, tar_img):
    images = np.vstack((src_img, gen_img, tar_img))
    images = (images + 1) / 2.0
    titles = ['Source', 'Generated', 'Expected']

    for i in range(len(images)):
        plt.subplot(1, 3, 1 + i)
        plt.axis('off')
        plt.imshow(images[i])
        plt.title(titles[i])
    plt.show()


def get_horizontal_flip_homography(court_img, homography_orig):
    # taken from Homayounfar et al. - 2017

    # court dimensions in yards
    field_width = 114.83
    field_height = 74.37
    img_width = court_img.shape[1]
    img_height = court_img.shape[0]

    homography_rev = np.array([[-1, 0, img_width], [0, 1, 0], [0, 0, 1]])
    homography_rev_inv = homography_rev
    homography_rev_model = np.array([[-1, 0, field_width], [0, 1, 0], [0, 0, 1]])

    homography = homography_orig.dot(homography_rev_inv)
    homography = homography_rev_model.dot(homography)
    return homography


def augment_soccer_dataset(initial_path, relative_path):
    court_image_path = f'{initial_path}{relative_path}court_images/'
    homographies_path = f'{initial_path}{relative_path}homography_matrices/'
    grass_mat_path = f'{initial_path}{relative_path}grass_mats/'

    dataset_length = os.listdir(court_image_path).__len__()

    for i in range(1, dataset_length + 1):
        court_img = cv2.imread('{}{}.jpg'.format(court_image_path, i))

        mat = scipy.io.loadmat('{}{}_grass_gt.mat'.format(grass_mat_path, i))
        grass_mask = mat.get('grass')

        with open('{}{}.homographyMatrix'.format(homographies_path, i)) as homography_file:
            data = homography_file.readlines()
        matrix = read_homography_matrix(data)

        flipped_court_img = cv2.flip(court_img, 1)
        flipped_grass_mask = cv2.flip(grass_mask, 1)
        flipped_mat = {'grass': flipped_grass_mask}
        flipped_court_homography = get_horizontal_flip_homography(court_img, matrix)

        flipped_image_code = i + dataset_length
        cv2.imwrite(f'{court_image_path}{flipped_image_code}.jpg', flipped_court_img)
        scipy.io.savemat(f'{grass_mat_path}{flipped_image_code}_grass_gt.mat', flipped_mat, do_compression=True)
        np.savetxt(f'{homographies_path}{flipped_image_code}.homographyMatrix',
                   flipped_court_homography, fmt='%.7e', delimiter='\t')


def export_dataset_to_npz(initial_path, relative_path, filename):
    [src_images, tar_images] = load_images(initial_path, relative_path)

    src_images = [cv2.resize(img, (256, 256)) for img in src_images]
    tar_images = [cv2.resize(img, (256, 256)) for img in tar_images]

    src_images = np.asarray(src_images)
    tar_images = np.asarray(tar_images)

    savez_compressed(filename, src_images, tar_images)
    print('Saved dataset: ', filename)


def divide_dataset_to_folders(initial_path, relative_path):
    path = f'{initial_path}{relative_path}'
    os.mkdir(f'{path}court_images')
    os.mkdir(f'{path}homography_matrices')
    os.mkdir(f'{path}grass_mats')

    court_image_files = glob.glob(f'{path}*.jpg')
    homography_files = glob.glob(f'{path}*.homographyMatrix')
    mat_files = glob.glob(f'{path}*.mat')

    for file in court_image_files:
        shutil.move(file, f'{path}court_images')

    for file in homography_files:
        shutil.move(file, f'{path}homography_matrices')

    for file in mat_files:
        shutil.move(file, f'{path}grass_mats')


if __name__ == '__main__':

    initial_path = f'{get_project_root()}/datasets/world_cup_2014/'
    dataset_tar_file = 'soccer_data.tar.gz'

    tar_file = tarfile.open(f'{initial_path}{dataset_tar_file}')
    tar_file.extractall(path=f'{initial_path}')
    tar_file.close()

    relative_train_path = 'raw/train_val/'
    relative_test_path = 'raw/test/'

    divide_dataset_to_folders(initial_path, relative_train_path)
    divide_dataset_to_folders(initial_path, relative_test_path)

    augment_soccer_dataset(initial_path, relative_train_path)
    augment_soccer_dataset(initial_path, relative_test_path)

    export_dataset_to_npz(initial_path, relative_train_path, filename=f'{initial_path}train_maps_256.npz')
    export_dataset_to_npz(initial_path, relative_test_path, filename=f'{initial_path}test_maps_256.npz')
