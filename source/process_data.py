import os
import numpy as np
import cv2
from os import listdir, mkdir
from os.path import isfile, join, isdir
import dlib
from PIL import Image
import argparse

def get_lndm(path_img, path_out, start_id = 0, dlib_path=""):
    dir_proc = {'msk':'msk', 'org':'orig', 'clr':'clr', 'lnd':'lndm'}

    for dir_it in dir_proc:
        if os.path.isdir(path_out + dir_proc[dir_it]) == False:
            os.mkdir(path_out + dir_proc[dir_it])

    folder_list = [f for f in listdir(path_img)]
    folder_list.sort()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_path+"shape_predictor_68_face_landmarks.dat")

    line_px = 1
    res_w = 178
    res_h = 218

    for fld in folder_list[:]:
        imglist_all = [f[:-4] for f in listdir(join(path_img, fld)) if isfile(join(path_img, fld, f)) and f[-4:] == ".jpg"]
        imglist_all.sort(key=int)
        imglist_all = imglist_all[start_id:]

        for dir_it in dir_proc:
            if os.path.isdir(join(path_out, dir_proc[dir_it], fld)) == False:
                os.mkdir(join(path_out, dir_proc[dir_it], fld))

        land_mask = True
        crop_coord = []
        for it in range(len(imglist_all)):
            clr = cv2.imread(join(path_img, fld, imglist_all[it]+".jpg"), cv2.IMREAD_ANYCOLOR)
            img = clr.copy()
            img_dlib = np.array(clr[:, :, :], dtype=np.uint8)
            dets = detector(img_dlib, 1)

            for k_it, d in enumerate(dets):
                if k_it != 0:
                    continue
                landmarks = predictor(img_dlib, d)

                # centering
                c_x = int((landmarks.part(42).x + landmarks.part(39).x) / 2)
                c_y = int((landmarks.part(42).y + landmarks.part(39).y) / 2)
                w_r = int((landmarks.part(42).x - landmarks.part(39).x)*4)
                h_r = int((landmarks.part(42).x - landmarks.part(39).x)*5)
                w_r = int(h_r/res_h*res_w)

                w, h = int(w_r * 2), int(h_r * 2)
                pd = int(w) # padding size
                
                img_p = np.zeros((img.shape[0]+pd*2, img.shape[1]+pd*2, 3), np.uint8) * 255
                img_p[:, :, 0] = np.pad(img[:, :, 0], pd, 'edge')
                img_p[:, :, 1] = np.pad(img[:, :, 1], pd, 'edge')
                img_p[:, :, 2] = np.pad(img[:, :, 2], pd, 'edge')
                
                visual = img_p[c_y - h_r+pd:c_y + h_r+pd, c_x - w_r+pd:c_x + w_r+pd]

                crop_coord.append([c_y - h_r, c_y + h_r, c_x - w_r, c_x + w_r, pd, imglist_all[it]+".jpg"])
                t_x, t_y = int(c_x - w_r), int(c_y - h_r)

                ratio_w, ratio_h = res_w/w, res_h/h

                visual = cv2.resize(visual, dsize=(res_w, res_h), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(join(path_out, dir_proc['clr'], fld, imglist_all[it]+".jpg"), visual) #saving crop
                cv2.imwrite(join(path_out, dir_proc['org'], fld, imglist_all[it]+".jpg"), clr) # saving original

                if land_mask:
                    img_lndm = np.ones((res_h, res_w, 3), np.uint8) * 255

                    def draw_line(offset, pt_st, pt_end):
                        cv2.line(img_lndm, (int((landmarks.part(offset + pt_st).x - t_x) * ratio_w), int((landmarks.part(offset + pt_st).y - t_y) * ratio_h)), (int((landmarks.part(offset + pt_end).x - t_x) * ratio_w), int((landmarks.part(offset + pt_end).y - t_y) * ratio_h)), (0, 0, 255), line_px)

                    for i in range(16):
                        draw_line(0, i, i+1)

                    for i in range(3):
                        draw_line(27, i, i+1)

                    for i in range(4):
                        draw_line(60, i, i+1)

                    for i in range(3):
                        draw_line(64, i, i+1)

                    draw_line(0, 67, 60)

                    result = Image.fromarray((img_lndm).astype(np.uint8))
                    result.save(join(path_out, dir_proc['lnd'], fld, imglist_all[it]+".jpg"))

                    img_msk = np.ones((res_h, res_w, 3), np.uint8) * 255

                    contours = np.zeros((0, 2))
                    contours = np.concatenate((contours, np.array([[(landmarks.part(0).x - t_x) * ratio_w, (landmarks.part(19).y - t_y) * ratio_h]])), axis=0)
                    for p in range(17):
                        contours = np.concatenate((contours, np.array([[(landmarks.part(p).x - t_x) * ratio_w, (landmarks.part(p).y - t_y) * ratio_h]])),axis=0)
                    contours = np.concatenate((contours, np.array([[(landmarks.part(16).x - t_x) * ratio_w, (landmarks.part(24).y - t_y) * ratio_h]])),axis=0)
                    contours = contours.astype(int)
                    cv2.fillPoly(img_msk, pts=[contours], color=(0, 0, 0))
                    result = Image.fromarray((img_msk).astype(np.uint8))
                    result.save(join(path_out, dir_proc['msk'], fld, imglist_all[it]+".jpg"))

        #np.save(join(path_out, dir_proc['org'], fld, 'crop_coord.npy'), crop_coord) #crop coordinates
        print("folder done",fld)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, help='directory with input data', default='../dataset/celeba/clr/')
    parser.add_argument('--output', type=str, help='directory for output', default='../output/')
    parser.add_argument('--dlib', type=str, help='directory with dlib predictor', default='')
    args = parser.parse_args()
    
    get_lndm(args.input, args.output, dlib_path=args.dlib)
    