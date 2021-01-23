import cv2
import numpy as np

import torch
from torchvision.transforms import ToTensor, Normalize, Compose
import torch.backends.cudnn as cudnn

from models.pix2pix.Pix2Pix import Pix2Pix
import utils


class GrassMaskEstimator:

    def __init__(self, model_filename):

        self.transform = Compose([
            ToTensor(),
            Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))])

        pix2pix = Pix2Pix(is_train=False)
        model, _, _, _, _, _ = Pix2Pix.load_model(model_filename, pix2pix)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = model.to(self.device)
            cudnn.benchmark = True

        self.model.eval()

    def __call__(self, court_img):

        img = self.transform(court_img)
        img = torch.stack([img])
        with torch.no_grad():
            img = img.to(self.device)
            pred_grass_mask = self.model.infer(img)
        pred_grass_mask = tensor2im(pred_grass_mask)

        # standardize mask
        pred_grass_mask = pred_grass_mask/255

        #apply gradient mask
        masked_court_image = np.uint8(court_img*pred_grass_mask)

        return masked_court_image



def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_grass_mask_estimator_output_images(court_images_, masked_court_images_):
    for i in range(len(court_images_)):
        img_delimiter = np.zeros((256, 2, 3), dtype=np.uint8)
        img_to_be_saved = np.concatenate(
            (court_images_[i], img_delimiter, masked_court_images_[i]), axis=1)
        cv2.imwrite(f'{utils.get_project_root()}datasets/test/court_images/grass_mask_estimator/court_masked_{i}.jpg', img_to_be_saved)


if __name__ == '__main__':
    import sys
    import glob

    image_filelist = glob.glob(f"{utils.get_project_root()}datasets/test/court_images/*.jpg")
    court_images = []
    masked_court_images = []
    estimator = GrassMaskEstimator(
        model_filename=f'{utils.get_generated_models_path()}grass_mask_estimator_200.pth')

    for image_filename in image_filelist:
        court_image = cv2.imread(image_filename)
        court_image = cv2.resize(court_image, (256, 256))

        masked_court_image = estimator(court_image)

        court_images.append(court_image)
        masked_court_images.append(masked_court_image)

    save_grass_mask_estimator_output_images(court_images, masked_court_images)

    sys.exit()
