
import util, numpy as np

# (0, 255) ; (0, 255)

def prep_image(folder):
    numpyImage, numpyOrigin, numpySpacing = util.load_dicom_image(folder)
    image = util.resample(numpyImage, numpySpacing, (1,1,1))

    # truncated_image = util.truncate_HU_uint8(resampled_image)
    image = util.normalize(image, (0,255))

    return image


if __name__ == '__main__':
    test_dcm_dir = "/cvibraid/data32/image/qiws32/10070_SCSMODEF0328/2018-04-16/2152100017434"
    image = prep_image(test_dcm_dir)
    np.save("/radraid/apps/personal/wasil/trash/lcs_example.npy", image)
    import matplotlib.pyplot as plt 
    plt.imsave("/radraid/apps/personal/wasil/trash/lcs_example.png", image[int(image.shape[0]/2),  :, :])