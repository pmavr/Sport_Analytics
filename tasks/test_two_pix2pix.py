
from models.two_pix2pix.test_options import opt
from models.two_pix2pix.data.data_loader import CreateDataLoader
from models.two_pix2pix.models.pix2pix_model import Pix2PixModel

opt = opt()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.continue_train = False

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = Pix2PixModel()
model.initialize(opt)
print("model [%s] was created" % (model.name()))

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()

    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    visuals = model.get_current_visuals()


    # visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

# webpage.save()
