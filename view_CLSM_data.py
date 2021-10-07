
import os
import sys
import pickle
from glob import glob

import numpy as np
import skimage.io
from skimage.util import img_as_uint
from tifffile import imsave, imread
import cv2

from my_utils import humanbytes, print_metadata, get_metadata
import matplotlib.pyplot as plt

import javabridge
import bioformats

def start_java_vm():
    javabridge.start_vm(class_path=bioformats.JARS)
    myloglevel="ERROR"  # user string argument for logLevel.
    rootLoggerName = javabridge.get_static_field("org/slf4j/Logger","ROOT_LOGGER_NAME", "Ljava/lang/String;")
    rootLogger = javabridge.static_call("org/slf4j/LoggerFactory","getLogger", "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
    logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level",myloglevel, "Lch/qos/logback/classic/Level;")
    javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)

def oir_to_tiffstack(oir_file, channel, sizey, sizex):
    with bioformats.ImageReader(oir_file) as reader:
        img = img_as_uint((reader.read(c=channel)))
        if scale_12bit_to16bit:
            img *= 2**4
        if rotate:
            img = np.swapaxes(img, 0, 1)
        return img

def process_dir(all_files, inp_dir, outp_dir, notes):
    for oir_file in all_files:
        area_index = oir_file.find('_G0')+1
        area = oir_file[area_index:area_index+4]
        print('\n\n', area)

        # metadata handling
        sizey, sizex, sizet, channel_dict, summary, meta_dict = get_metadata(oir_file, notes, 0)
        with open(f"{outp_dir}/{area}_meatadata.pkl", "wb") as pkl_file:
            pickle.dump(meta_dict, pkl_file)
            print(f'{area}_meatadata.pkl saved.\n')

        all_channels = []
        chnl_colors = []
        for channel_color, channel in channel_dict.items():
            chnl_colors.append(channel_color)
            fname = f'{area}_{channel_color}_compr.tif'
            print(f'\tReading in {channel_color} channel:\n{channel["summary"]}')
            img = oir_to_tiffstack(oir_file, channel['id'], sizey, sizex)

            # save single channels
            if save_tifs:
                print(f'\tSaving {humanbytes(sys.getsizeof(img))} image (compression=deflate).')
                imsave(f'{outp_dir}/{fname}', img, photometric='minisblack', 
                        compression='deflate', bigtiff=True)
            
            if save_merged_annotated:
                all_channels.append(img)
            
        if save_merged_annotated:
            bg = cv2.UMat(cv2.cvtColor(all_channels[0], cv2.COLOR_BGR2RGB))
            red_green = cv2.UMat(cv2.merge([*all_channels[1:], np.zeros_like(all_channels[1])] ))
            img_stacked = cv2.addWeighted(bg,0.3, red_green,0.7, 0)

            text = f'{summary}\n\n{channel["summary"]}'
            for i, line in enumerate(text.split('\n')):
                cv2.putText(img_stacked, line, 
                            (20,(i+1)*(int(70*(textsize/2)))), cv2.FONT_HERSHEY_DUPLEX, 
                            1*textsize, (2**16,2**16,2**16), 3)

            img_stacked = cv2.UMat.get(img_stacked).astype(float) /2**16
            imsave(f'{outp_dir}/{area}_annotated.tif', img_stacked, compression='deflate')

inp_dir = '/run/media/loaloa/lbb_ssd/timelapse13_onemonthlater/'
outp_dir = '/run/media/loaloa/lbb_ssd/timelapse13_onemonthlater_processed/'
notes = 'Retrograde Ruby virus after 2d, GFP-actin from a month ago (timelapse)'
all_files = sorted(glob(inp_dir+'Stitch*_G0*.oir'))
print_files = '\n\t'.join([f for f in all_files])
print(f'Found {len(all_files)} areas: \n\t{print_files}')

save_tifs = False
rotate = True
save_merged_annotated = True
scale_12bit_to16bit = True
textsize = 3

def main():
    start_java_vm()
    os.makedirs(outp_dir, exist_ok=True)
    process_dir(all_files, inp_dir, outp_dir, notes)
    javabridge.kill_vm()
if __name__ == '__main__':
    main()