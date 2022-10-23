from PIL import Image, ImageChops, ImageEnhance
# import matplotlib.pyplot as plt
import numpy as np
import cv2

# img np.array
def my_brightness(img, factor):
    degenerate_img = np.zeros(img.shape, img.dtype)
    return np.clip((factor*img + (1-factor) * degenerate_img), 0, 255).astype(np.uint8)


def my_contrast(img, factor):
    degenerate_img = np.ones(img.shape, img.dtype)
    img_mean = int(np.mean(np.array(Image.fromarray(img).convert('RGB').convert("L")), axis=(0,1)) + 0.5)
    degenerate_img = degenerate_img * img_mean
    return np.clip((factor*img + (1-factor) * degenerate_img), 0, 255).astype(np.uint8)


def my_color(img, factor):
    gray_img = np.array(Image.fromarray(img).convert('RGB').convert("L"))
    degenerate_img = a1 = np.stack([gray_img, gray_img, gray_img], axis=2)
    return np.clip((factor*img + (1-factor) * degenerate_img), 0, 255).astype(np.uint8)


def my_sharpness(img, factor):
    smooth_filter = 1 / 13 * np.asarray([[1, 1, 1], [1, 5, 1], [1, 1, 1]])
   
    img_image = Image.fromarray(img)
    from PIL import  ImageFilter # Pillow 的模糊操作
    smoF = img_image.filter(ImageFilter.SMOOTH)
    smoF_np = np.array(smoF)

    
    degenerate_img = cv2.filter2D(img, -1, smooth_filter)
    # img1 = factor*img
    # img2 = (1-factor) * degenerate_img
    # img3 = factor*img + (1-factor) * degenerate_img
    # # print(((degenerate_img - smoF_np).mean()))
    return np.clip((factor*img + (1-factor) * degenerate_img), 0, 255).astype(np.uint8)

#  two images of different local brightness to blend the images
def center_seam(img1, img2):
    img1_hsv_v = cv2.cvtColor(img_blend1_rgb,cv2.COLOR_RGB2HSV)[:,:,2]
    img2_hsv_v = cv2.cvtColor(img_blend2_rgb,cv2.COLOR_RGB2HSV)[:,:,2]
    imgDist1=cv2.distanceTransform(img1_hsv_v, distanceType=cv2.DIST_L2, maskSize=5)
    imgDist2=cv2.distanceTransform(img2_hsv_v, distanceType=cv2.DIST_L2, maskSize=5)
    imgAlpha = imgDist1/(imgDist1+imgDist2+1e-5)
    imgBlend12 = img1.copy()
    for c in range(3):
        imgBlend12[:,:,c]=img1[:,:,c] * imgAlpha + img2[:,:,c]*(1-imgAlpha)
    return np.clip(imgBlend12, 0, 255).astype(np.uint8)



if __name__ == "__main__":
    img = Image.open(r"./imgs/dark.png")
    
    parms = [-0.5, 0, 0.5, 1, 1.5]
    # brightness
    img_brightness_ress = []
    for idx in range(len(parms)):
        img_name = "img_brightness_" + str(parms[idx]) + ".png"
        img_res = ImageEnhance.Brightness(img).enhance(parms[idx])
        img_res_np = np.array(img_res)
        img_my_res = my_brightness(np.array(img), parms[idx])
        print("Bright params: {}  result equal: {}".format(str(parms[idx]),str(np.allclose(np.array(img_res), img_my_res))))
        img_brightness_ress.append(img_res)
        img_res.save(img_name)

    # color
    img_color_ress = []
    for idx in range(len(parms)):
        img_name = "img_color_" + str(parms[idx]) + ".png"
        img_res = ImageEnhance.Color(img).enhance(parms[idx])
        img_my_res = my_color(np.asarray(img), parms[idx])
        print("Color params: {}  result equal: {}".format(str(parms[idx]),str(np.allclose(np.array(img_res), img_my_res))))
        img_color_ress.append(img_res)
        img_res.save(img_name)

    # contrast
    img_contrast_ress = []
    for idx in range(len(parms)):
        img_name = "img_contrast_" + str(parms[idx]) + ".png"
        img_res = ImageEnhance.Contrast(img).enhance(parms[idx])
        mg_res_np = np.array(img_res)
        img_my_res = my_contrast(np.asarray(img), parms[idx])
        print("Contrast params: {}  result equal: {}".format(str(parms[idx]), np.allclose(np.asarray(img_res), img_my_res)))
        img_contrast_ress.append(img_res)
        img_res.save(img_name)

    # sharpness
    img_sharpness_ress = []
    for idx in range(len(parms)):
        img_name = "img_sharpness_" + str(parms[idx]) + ".png"
        img_res = ImageEnhance.Sharpness(img).enhance(parms[idx])
        img_res_np = np.array(img_res)
        img_np = np.array(img)
        img_my_res = my_sharpness(np.asarray(img), parms[idx])
        print(np.mean(np.array(img)-img_res_np))
        print("Sharpness params: {}  result equal: {}".format(str(parms[idx]), np.allclose(np.asarray(img_res), img_my_res)))
        print("res - my_res:{}".format(str((img_my_res - img_res).mean())))
        img_sharpness_ress.append(img_res)
        img_res.save(img_name)
    

    # test center seam
    img_blend1_rgb = cv2.imread('./imgs/img_blend1.png')[:,:,::-1]
    img_blend2_rgb = cv2.imread('./imgs/img_blend2.png')[:,:,::-1]
    img_blend12_rgb= center_seam(img_blend1_rgb,img_blend2_rgb)
    cv2.imwrite("img_blend12.png",img_blend12_rgb[:,:,::-1])