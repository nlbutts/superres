import cv2
import argparse


def dosuper(imgfile, do_edsr, do_espcn, do_fsrcnn, do_lapsrn):
    if do_edsr:
        print('Creating EDSR image')
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = "EDSR_x4.pb"
        sr.readModel(path)
        sr.setModel("edsr", 4) # set the model by passing the value and the upsampling ratio
        img = cv2.imread(imgfile)
        result = sr.upsample(img) # upscale the input image
        cv2.imwrite('edsr.bmp', result)
    if do_espcn:
        print('Creating ESPCN image')
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = "ESPCN_x4.pb"
        sr.readModel(path)
        sr.setModel("espcn", 4) # set the model by passing the value and the upsampling ratio
        img = cv2.imread(imgfile)
        result = sr.upsample(img) # upscale the input image
        cv2.imwrite('espcn.bmp', result)
    if do_fsrcnn:
        print('Creating FSRCNN image')
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = "FSRCNN_x4.pb"
        sr.readModel(path)
        sr.setModel("fsrcnn", 4) # set the model by passing the value and the upsampling ratio
        img = cv2.imread(imgfile)
        result = sr.upsample(img) # upscale the input image
        cv2.imwrite('fsrcnn.bmp', result)
    if do_lapsrn:
        print('Creating LapSRN image')
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = "LapSRN_x4.pb"
        sr.readModel(path)
        sr.setModel("lapsrn", 4) # set the model by passing the value and the upsampling ratio
        img = cv2.imread(imgfile)
        result = sr.upsample(img) # upscale the input image
        cv2.imwrite('lapsrn.bmp', result)


if __name__ == "__main__":
    infoStr = """This program analyzes the output of the loadbox data
    """

    parser = argparse.ArgumentParser(description=infoStr)
    parser.add_argument('-f', '--file',
                        type=str,
                        required=True,
                        help='The log file')
    parser.add_argument('--edsr',
                        action='store_true',
                        default=False,
                        help='Use EDSR')
    parser.add_argument('--espcn',
                        action='store_true',
                        default=False,
                        help='use ESPCN')
    parser.add_argument('--fsrcnn',
                        action='store_true',
                        default=False,
                        help='use FSRCNN')
    parser.add_argument('--lapsrn',
                        action='store_true',
                        default=False,
                        help='use LapSRN')


    args = parser.parse_args()

    dosuper(args.file, args.edsr, args.espcn, args.fsrcnn, args.lapsrn)