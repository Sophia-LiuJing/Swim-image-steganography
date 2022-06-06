# logistic encrypted for image
import sys

from PIL import Image
import numpy as np
# logic
# def logic_encrypt(im, x0=0.51, mu=3.7):
#     xsize, ysize = im.shape[0], im.shape[1]
#     im = np.array(im).flatten()
#     num = len(im)
#     for i in range(100):
#         x0 = mu * x0 * (1-x0)
#     E = np.zeros(num)
#     E[0] = x0
#     for i in range(0,num-1):
#         E[i+1] = mu * E[i]* (1-E[i])
#     E = np.round(E*255).astype(np.uint8)
#     im = np.bitwise_xor(E,im)
#     im = im.reshape(xsize,ysize,-1)
#     im = np.squeeze(im)
#     return im
def logistic(x0=0.51,mu=3.7,num=0):
    for i in range(100):
        x0 = mu * x0 * (1-x0)
    E = np.zeros(num)
    E[0] = x0
    for i in range(0,num-1):
        E[i+1] = mu * E[i]* (1-E[i])
    return E
def logic_scramble_encrypt(im, x0=0.51, mu=3.7):
    xsize, ysize = im.shape[0], im.shape[1]
    im = np.array(im).flatten()
    # num=len(im)
    # num = xsize*ysize
    im=im.reshape(xsize*ysize,3)
    num=im.shape[0]
    for i in range(100):
        x0 = mu * x0 * (1-x0)
    E = np.zeros(num)
    E[0] = x0
    k = np.zeros(num)
    k[0] = 1
    for i in range(0,num-1):
        E[i+1] = mu * E[i]* (1-E[i])
        k[i + 1] = i + 2
    E = np.round(E*num).astype(np.uint64)
    k = k.astype(np.uint64)
    for i in range(0, num):
        t = E[i]
        temp = im[t].copy()
        im[t] = im[i].copy()
        im[i] = temp
    # for i in range(0, num):
    #     t = E[num - 1 - i]
    #     temp = im[t]
    #     im[t] = im[num - 1 - i]
    #     im[num - 1 - i] = temp
    # print(k)
    # im = np.array(im).flatten()
    im = im.reshape(xsize,ysize,-1)
    # im = np.squeeze(im)
    return im


def logic_scramble_decrypt(im, x0=0.51, mu=3.7):
    xsize, ysize = im.shape[0], im.shape[1]
    im = np.array(im).flatten()
    num = len(im)
    # im = im.reshape(xsize * ysize, 3)
    # num = im.shape[0]
    num = xsize*ysize
    im = im.reshape(-1, 3)
    for i in range(100):
        x0 = mu * x0 * (1-x0)
    E = np.zeros(num)
    E[0] = x0
    k = np.zeros(num)
    k[0] = 1
    for i in range(0,num-1):
        E[i+1] = mu * E[i]* (1-E[i])
        k[i + 1] = i + 2
    E = np.round(E*num).astype(np.uint64)
    k = k.astype(np.uint64)
    # for i in range(0, num):
    #     t = E[i]
    #     temp = im[t]
    #     im[t] = im[i]
    #     im[i] = temp
    for i in range(0, num):
        t = E[num - 1 - i]
        temp = im[t].copy()
        im[t] = im[num - 1 - i].copy()
        im[num - 1 - i] = temp
    # print(k)
    # im = np.array(im).flatten()
    im = im.reshape(xsize,ysize,-1)
    # im = np.squeeze(im)
    return im


def generatex0(x0,mu):
    xtotal=[]
    xtotal.append(x0)
    for i in range(201):
        xtotal.append(mu*xtotal[-1]*(1-xtotal[-1]))
    return [xtotal[10],xtotal[25],xtotal[50],xtotal[100],xtotal[200]]

def recursion_daluan(img):
    # 返回矩阵的行和列
    s_1 = img.shape[0]
    r = np.random.choice(s_1, size=s_1, replace=False, p=None)
    RGBS = img[r, :, :]
    s_2 = img.shape[1]
    c = np.random.choice(s_2, size=s_2, replace=False, p=None)
    RGBSS = RGBS[:, c, :]

    return RGBSS

def unrecursion_daluan(img):
    c=img.shape[1]
    r=img.shape[0]
    i = 0
    f = np.arange(0, len(c))
    while i <= len(c):
        # find（r==j）是返回r中等于j的值的索引，可以是多个，赋值给f的第j个位置
        for (k, val) in enumerate(c):
            if val == i:
                f[i] = k
        i = i + 1

    j = 0
    g = np.arange(0, len(r))
    RGBE = img[:, f, :]
    while j <= len(r):
        # find（r==j）是返回r中等于j的值的索引，可以是多个，赋值给f的第j个位置
        for (k, val) in enumerate(r):
            if val == j:
                g[j] = k
        j = j + 1

    RGBEE = RGBE[g, :, :]
    return RGBEE



def arnold_encode(image, shuffle_times=3, a=1, b=7):
    """ Arnold shuffle for rgb image
    Args:
        image: input original rgb image
        shuffle_times: how many times to shuffle
    Returns:
        Arnold encode image
    """
    # 1:创建新图像
    arnold_image = np.zeros(shape=image.shape,dtype=np.int)

    # 2：计算N
    h, w = image.shape[0], image.shape[1]
    N = h  # 或N=w

    # 3：遍历像素坐标变换
    for time in range(shuffle_times):
        for ori_x in range(h):
            for ori_y in range(w):
                # 按照公式坐标变换
                new_x = (1 * ori_x + b * ori_y) % N
                new_y = (a * ori_x + (a * b + 1) * ori_y) % N

                arnold_image[new_x, new_y, :] = image[ori_x, ori_y, :]
        image=arnold_image.copy()

    return arnold_image


def arnold_decode(image, shuffle_times=3, a=1, b=7):
    """ decode for rgb image that encoded by Arnold
    Args:
        image: rgb image encoded by Arnold
        shuffle_times: how many times to shuffle
    Returns:
        decode image
    """
    # 1:创建新图像
    decode_image = np.zeros(shape=image.shape,dtype=np.int)

    # 2：计算N
    h, w = image.shape[0], image.shape[1]
    N = h  # 或N=w

    # 3：遍历像素坐标变换
    for time in range(shuffle_times):
        for ori_x in range(h):
            for ori_y in range(w):
                # 按照公式坐标变换
                new_x = ((a * b + 1) * ori_x + (-b) * ori_y) % N
                new_y = ((-a) * ori_x + ori_y) % N
                decode_image[new_x, new_y, :] = image[ori_x, ori_y, :]
        image = decode_image.copy()
    return decode_image

def recursion_encrypt(img,xsize,ysize):
    x = int(xsize / 2)
    y = int(ysize / 2)
    if x<=1 or y <= 1:
        return



    else:

        recursion_encrypt(img[0: x, 0: y], x, y)
        # img[0: x, 0: y]=arnold_encode(img[0: x, 0: y])

        img[0: x, 0: y] =  logic_scramble_encrypt(img[0: x, 0: y])

        recursion_encrypt(img[0: x, y : ysize], x, y)
        # img[0: x, y : ysize]=arnold_encode(img[0: x, y: ysize])
        img[0: x, y: ysize] = logic_scramble_encrypt(img[0: x, y: ysize])


        recursion_encrypt(img[x:xsize, 0: y], x, y)
        # img[x : xsize, 0: y]=arnold_encode(img[x:xsize, 0: y])
        img[x: xsize, 0: y] = logic_scramble_encrypt(img[x:xsize, 0: y])

        recursion_encrypt(img[x : xsize, y : ysize], x, y)
        # img[x : xsize, y : ysize]=arnold_encode(img[x : xsize, y : ysize])
        img[x: xsize, y: ysize] = logic_scramble_encrypt(img[x: xsize, y: ysize])


        # img[0: xsize, 0: ysize]=arnold_encode(img[0: xsize, 0: ysize])
        img[0: xsize, 0: ysize] = logic_scramble_encrypt(img[0: xsize, 0: ysize])


def recursion_unencrypt(img,xsize,ysize):
    x = int(xsize / 2)
    y = int(ysize / 2)
    if x <= 1 or ysize <= 1:
        return



    else:


        # img[0: xsize, 0: ysize] = arnold_decode(img[0: xsize, 0: ysize])
        # img[0: x, 0: y] = arnold_decode(img[0: x, 0: y])
        # img[0: x, y: ysize] = arnold_decode(img[0: x, y: ysize])
        # img[x: xsize, 0: y] = arnold_decode(img[x:xsize, 0: y])
        # img[x: xsize, y: ysize] = arnold_decode(img[x: xsize, y: ysize])

        img[0: xsize, 0: ysize] = logic_scramble_decrypt(img[0: xsize, 0: ysize])
        img[0: x, 0: y] = logic_scramble_decrypt(img[0: x, 0: y])
        img[0: x, y: ysize] = logic_scramble_decrypt(img[0: x, y: ysize])
        img[x: xsize, 0: y] = logic_scramble_decrypt(img[x:xsize, 0: y])
        img[x: xsize, y: ysize] = logic_scramble_decrypt(img[x: xsize, y: ysize])

        recursion_unencrypt(img[0: x, 0: y], x, y)
        recursion_unencrypt(img[0: x, y: ysize], x, y)
        recursion_unencrypt(img[x:xsize, 0: y], x, y)
        recursion_unencrypt(img[x: xsize, y: ysize], x, y)



def recursion_gray_encrypt(img,xsize,ysize):


    if xsize<=36 or ysize <= 36:
        return



    else:
        x = int(xsize / 2)
        y = int(ysize / 2)
        recursion_encrypt(img[0: x, 0: y], x, y)
        # img[0: x, 0: y]=arnold_encode(img[0: x, 0: y])
        img[0: x, 0: y] =  logic_scramble_encrypt(img[0: x, 0: y])

        recursion_encrypt(img[0: x, y : ysize], x, y)
        # img[0: x, y : ysize]=arnold_encode(img[0: x, y: ysize])
        img[0: x, y: ysize] = logic_scramble_encrypt(img[0: x, y: ysize])


        recursion_encrypt(img[x:xsize, 0: y], x, y)
        # img[x : xsize, 0: y]=arnold_encode(img[x:xsize, 0: y])
        img[x: xsize, 0: y] = logic_scramble_encrypt(img[x:xsize, 0: y])

        recursion_encrypt(img[x : xsize, y : ysize], x, y)
        # img[x : xsize, y : ysize]=arnold_encode(img[x : xsize, y : ysize])
        img[x: xsize, y: ysize] = logic_scramble_encrypt(img[x: xsize, y: ysize])


        # img[0: xsize, 0: ysize]=arnold_encode(img[0: xsize, 0: ysize])
        img[0: xsize, 0: ysize] = logic_scramble_encrypt(img[0: xsize, 0: ysize])


def recursion_gray_unencrypt(img,xsize,ysize):


    if xsize <= 36 or ysize <= 36:
        return



    else:
        x = int(xsize / 2)
        y = int(ysize / 2)

        # img[0: xsize, 0: ysize] = arnold_decode(img[0: xsize, 0: ysize])
        # img[0: x, 0: y] = arnold_decode(img[0: x, 0: y])
        # img[0: x, y: ysize] = arnold_decode(img[0: x, y: ysize])
        # img[x: xsize, 0: y] = arnold_decode(img[x:xsize, 0: y])
        # img[x: xsize, y: ysize] = arnold_decode(img[x: xsize, y: ysize])

        img[0: xsize, 0: ysize] = logic_scramble_decrypt(img[0: xsize, 0: ysize])
        img[0: x, 0: y] = logic_scramble_decrypt(img[0: x, 0: y])
        img[0: x, y: ysize] = logic_scramble_decrypt(img[0: x, y: ysize])
        img[x: xsize, 0: y] = logic_scramble_decrypt(img[x:xsize, 0: y])
        img[x: xsize, y: ysize] = logic_scramble_decrypt(img[x: xsize, y: ysize])

        recursion_unencrypt(img[0: x, 0: y], x, y)
        recursion_unencrypt(img[0: x, y: ysize], x, y)
        recursion_unencrypt(img[x:xsize, 0: y], x, y)
        recursion_unencrypt(img[x: xsize, y: ysize], x, y)




# import matplotlib.pyplot as plt
#
# def img_hist(im):
#     im = np.array(im)
#     plt.hist(im.flatten(), bins = 256)
#     plt.show()
#
# import torchvision.utils as vutils
# im = Image.open("/share/home/wangzy/PycharmProjects/PyTorch-Deep-Image-Steganography/PyTorch-Deep-Image-Steganography/orisecret.png")
# im_en=Image.open("/share/home/wangzy/PycharmProjects/PyTorch-Deep-Image-Steganography/PyTorch-Deep-Image-Steganography/orisecret.png")
# # im = im.convert("L")
# x0 = 0.51
# mu = 3.7
# x=generatex0(x0,mu)
# im_en=np.array(im_en)
# recursion_encrypt(im_en,im_en.shape[0],im_en.shape[1],x)
# im_de=im_en.copy()
#
# recursion_encrypt(im_de,im_de.shape[0],im_de.shape[1],x)
#
# # im_en = logic_encrypt(im, x0, mu)
# # im_de = logic_encrypt(im_en, x0, mu)
# im_en = Image.fromarray(im_en)
# im_de = Image.fromarray(im_de)
#
# # img_hist(im_en)
# im.show()
# im_en.show()
# im_en.save("./PyTorch-Deep-Image-Steganography/encryptionData/3.JPEG")
# im_de.show()

# def encrypt(pt_secret_img):

# import torch
# x0 = 0.51
# mu = 3.7
# x=generatex0(x0,mu)
# input = torch.randn((3, 3, 144, 144)).cuda()
# min_max=(0, 1)
# for idx in range(0,input.shape[0]):
#     secret=input[idx,:,:,:]
#     # secret=secret.permute(1,2,0)
#     # secret=secret.numpy()
#     tensor=secret.squeeze().float().cpu().clamp_(*min_max)
#     tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
#     img_np = tensor.numpy()
#     img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
#     img_np = (img_np * 255.0).round()
#     img_np=img_np.astype(np.uint8)
#     im_en=img_np.copy()
#
#
#     recursion_encrypt(im_en, im_en.shape[0], im_en.shape[1], x)
#     # img_dp=img_np.copy()
#     #
#     # recursion_encrypt(img_dp, img_dp.shape[0], img_dp.shape[1], x)
#     #
#     # im_ori=Image.fromarray(ori_img)
#     # im_en = Image.fromarray(img_np)
#     # im_dn=Image.fromarray(img_dp)
#     #
#     # im_ori.show()
#     # im_en.show()
#     # im_dn.show()
#     # from IPython import embed
#     # embed()

import torch
def encryption(tensors,min_max=(0, 1)):
    import utils.transformed as transforms
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
    transform2=transforms.Compose([
        transforms.ToPILImage()
    ])
    with torch.no_grad():

        tensor = tensors[0, :, :, :]
        tensor=transform2(tensor.cpu())
        img_np = tensor
        img_np = np.array(img_np)

        recursion_encrypt(img_np, img_np.shape[0], img_np.shape[1])


    # im_en_result= Image.fromarray(im_en_result)
        im_en_result = transform(img_np)
        im_en_result = torch.unsqueeze(im_en_result, dim=0)




        for i in range(1,tensors.shape[0]):
            tensor=tensors[i,:,:,:]
            tensor = transform2(tensor.cpu())

            img_np = tensor
            img_np = np.array(img_np)
            recursion_encrypt(img_np, img_np.shape[0], img_np.shape[1])


            # im_en = Image.fromarray(im_en)
            im_en=transform(img_np)

            im_en = torch.unsqueeze(im_en, dim=0)
            im_en_result=torch.cat((im_en_result, im_en), 0)

    return im_en_result


import torch


def gray_encryption(tensors, min_max=(0, 1)):
    import utils.transformed as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform2 = transforms.Compose([
        transforms.ToPILImage()
    ])
    with torch.no_grad():
        tensor = tensors[0, :, :, :]
        tensor = transform2(tensor.cpu())
        img_np = tensor
        img_np = np.array(img_np)

        img_np  = img_np [:,:,np.newaxis]
        recursion_encrypt(img_np, img_np.shape[0], img_np.shape[1])

        # im_en_result= Image.fromarray(im_en_result)
        im_en_result = transform(img_np)
        im_en_result = torch.unsqueeze(im_en_result, dim=0)

        for i in range(1, tensors.shape[0]):
            tensor = tensors[i, :, :, :]
            tensor = transform2(tensor.cpu())

            img_np = tensor
            img_np = np.array(img_np)
            img_np = img_np[:, :, np.newaxis]
            recursion_encrypt(img_np, img_np.shape[0], img_np.shape[1])

            # im_en = Image.fromarray(im_en)
            im_en = transform(img_np)

            im_en = torch.unsqueeze(im_en, dim=0)
            im_en_result = torch.cat((im_en_result, im_en), 0)

    return im_en_result


def unencryption(tensors,min_max=(0, 1)):
    import utils.transformed as transforms
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
    transform2=transforms.Compose([
        transforms.ToPILImage()
    ])
    with torch.no_grad():

        tensor = tensors[0, :, :, :]
        tensor=transform2(tensor.cpu())
        img_np = tensor
        img_np = np.array(img_np)

        recursion_unencrypt(img_np, img_np.shape[0], img_np.shape[1])
    # im_en_result= Image.fromarray(im_en_result)
        im_en_result = transform(img_np)
        im_en_result = torch.unsqueeze(im_en_result, dim=0)




        for i in range(1,tensors.shape[0]):
            tensor=tensors[i,:,:,:]
            tensor = transform2(tensor.cpu())

            img_np = tensor
            img_np = np.array(img_np)
            recursion_unencrypt(img_np, img_np.shape[0], img_np.shape[1])


            # im_en = Image.fromarray(im_en)
            im_en=transform(img_np)

            im_en = torch.unsqueeze(im_en, dim=0)
            im_en_result=torch.cat((im_en_result, im_en), 0)

    return im_en_result

