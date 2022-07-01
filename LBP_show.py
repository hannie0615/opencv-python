
# 이미지를 인풋으로 받아서(비디오 아님) cv2.show 하고 결과값을 출력하는 코드

import math
import cv2
import numpy as np
import matplotlib.pylab as plt

# LBP
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def s(x):
    temp = x>0
    return temp.astype(float)

def lbpCode(im_gray, threshold):
    width, height = im_gray.shape
    interpOff = math.sqrt(2)/2 # 약 0.7071
    I = im2double(im_gray)
    pt = cv2.copyMakeBorder(I,1,1,1,1,cv2.BORDER_REPLICATE) # 이미지에 1만큼 가장자리를 만들어 줌
    right = pt[1:-1, 2:]
    left = pt[1:-1, :-2]
    above = pt[:-2, 1:-1]
    below = pt[2:, 1:-1];
    aboveRight = pt[:-2, 2:]
    aboveLeft = pt[:-2, :-2]
    belowRight = pt[2:, 2:]
    belowLeft = pt[2:, :-2]
    interp0 = right
    interp1 = (1-interpOff)*((1-interpOff) * I + interpOff * right) + interpOff *((1-interpOff) * above + interpOff * aboveRight)

    interp2 = above;
    interp3 = (1-interpOff)*((1-interpOff) * I + interpOff * left ) + interpOff *((1-interpOff) * above + interpOff * aboveLeft)

    interp4 = left;
    interp5 = (1-interpOff)*((1-interpOff) * I + interpOff * left ) + interpOff *((1-interpOff) * below + interpOff * belowLeft)

    interp6 = below;
    interp7 = (1-interpOff)*((1-interpOff) * I + interpOff * right ) + interpOff *((1-interpOff) * below + interpOff * belowRight)

    s0 = s(interp0 - I-threshold)
    s1 = s(interp1 - I-threshold)
    s2 = s(interp2 - I-threshold)
    s3 = s(interp3 - I-threshold)
    s4 = s(interp4 - I-threshold)
    s5 = s(interp5 - I-threshold)
    s6 = s(interp6 - I-threshold)
    s7 = s(interp7 - I-threshold)
    LBP81 = s0 * 1 + s1 * 2+s2 * 4   + s3 * 8+ s4 * 16  + s5 * 32  + s6 * 64  + s7 * 128
    LBP81.astype(int)

    U = np.abs(s0 - s7) + np.abs(s1 - s0) + np.abs(s2 - s1) + np.abs(s3 - s2) + np.abs(s4 - s3) + np.abs(s5 - s4) + np.abs(s6 - s5) + np.abs(s7 - s6)
    LBP81riu2 = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
    LBP81riu2[U > 2] = 9;

    return LBP81riu2

def lbpSharpness(im_gray, s, threshold):
    lbpmap  = lbpCode(im_gray, threshold)
    window_r = (s-1)//2;
    h, w = im_gray.shape[:2]
    map =  np.zeros((h, w), dtype=float)
    lbpmap_pad = cv2.copyMakeBorder(lbpmap, window_r, window_r, window_r, window_r, cv2.BORDER_REPLICATE)

    lbpmap_sum = (lbpmap_pad==6).astype(float) + (lbpmap_pad==7).astype(float) + (lbpmap_pad==8).astype(float) + (lbpmap_pad==9).astype(float)
    integral = cv2.integral(lbpmap_sum);
    integral = integral.astype(float)

    map = (integral[s-1:-1, s-1:-1]-integral[0:h, s-1:-1]-integral[s-1:-1, 0:w]+integral[0:h, 0:w])/math.pow(s,2);

    return map



# img 구하기
def ImgSum(image, k):
    num = 0
    num += np.sum(image)

    if num == 25 :  # (k*k*2/3)
        # print(num)
        num = 1
    else:
        num = 0

    return num

def ImgSum2(image, k):
    count = 0

    # 이미지 사이즈 가져오기 (400, 600)
    (ih, iw) = image.shape[:2]

    # 컨볼루션 결과
    output = 0

    # 컨볼루션 진행
    for x in np.arange(0, iw, k):  # 커널 k씩 이동 (0, iw, k)
        for y in np.arange(0, ih, k):
            # roi 생성
            roi = image[y:y + k, x:x + k] # 순서는 y->x 순

            output = ImgSum(roi, k)

            if output == 0:
                continue
            else:
                count = count + 1

    return count

def img_lbp(img_file):

    # - Color
    img = cv2.imread(img_file)

    # - Gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img is None:
        print('No image file')
    else:
        img_gray = cv2.resize(img_gray, dsize=(480,320))
        sharpness_map = lbpSharpness(img_gray, 21, 0.016)
        sharpness_map = (sharpness_map - np.min(sharpness_map)) / (np.max(sharpness_map - np.min(sharpness_map)))

        sharpness_map = (sharpness_map * 255).astype("uint8")
        lbp = np.stack((sharpness_map,), -1)
        # concat = np.concatenate((img_gray, lbp), axis=1)

        # method1
        blur = cv2.GaussianBlur(lbp, (5, 5), 0) # img_gray / lbp
        ret1, thresh_blur = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)  # 오츠
        b = np.sum(thresh_blur)/255
        print(b)

        # method7
        ret, lbp_blur = cv2.threshold(lbp, 60, 255,  cv2.THRESH_BINARY)
        lbp_blur = lbp_blur/255
        c = ImgSum2(lbp_blur, 5)    # img_gray / lbp
        print(c)

        return lbp_blur




if __name__ =="__main__":

    img_git1 = 'C:/A-LBP-focus/IMG/unfo6.jpg'

    git1 = cv2.imread(img_git1)
    git1_gray = cv2.cvtColor(git1, cv2.COLOR_BGR2GRAY)

    git1_gray = cv2.resize(git1_gray, dsize=(480, 320))
    git1 = cv2.resize(git1, dsize=(480, 320))
    cv2.imshow('Unfocused Image', git1)
    cv2.imshow('LBP', img_lbp(img_git1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()





















