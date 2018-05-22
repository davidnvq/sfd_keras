import numpy as np
import cv2
from skimage import io

count = 0
Path = '/home/ubuntu/Datasets/AFW/testimages/'

f = open('sfd_afw_dets.txt', 'wt')
for Name in open('afw_img_list.txt'):
    Image_Path = Path + Name[:-1] + '.jpg'
    image = io.imread(Image_Path)
    heigh = image.shape[0]
    width = image.shape[1]
    print(heigh, width)
    print(image.shape)
    im_shrink = 640.0 / max(image.shape[0], image.shape[1])
    image = cv2.resize(image, None, None, fx=im_shrink, fy=im_shrink, interpolation=cv2.INTER_LINEAR)
    print("After", image.shape)

    count += 1
    if count > 0:
        break

    net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    detections = net.forward()['detection_out']
    det_conf = detections[0, 0, :, 2]
    det_xmin = detections[0, 0, :, 3]
    det_ymin = detections[0, 0, :, 4]
    det_xmax = detections[0, 0, :, 5]
    det_ymax = detections[0, 0, :, 6]

    for i in range(det_conf.shape[0]):
        xmin = int(round(det_xmin[i] * width))
        ymin = int(round(det_ymin[i] * heigh))
        xmax = int(round(det_xmax[i] * width))
        ymax = int(round(det_ymax[i] * heigh))

        # simple fitting to AFW, because the gt box of training data (i.e., WIDER FACE) is longer than the gt box of AFW
        ymin += 0.2 * (ymax - ymin + 1)
        score = det_conf[i]
        if score < 0:
            continue

        # Example: "113234 0.93 0 0 10 10"
        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                format(Name[:-1], score, xmin, ymin, xmax, ymax))
    count += 1
    print('%d/205' % count)