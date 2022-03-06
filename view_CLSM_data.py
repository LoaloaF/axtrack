
"""
to run this you need the bioformats package 

try 
pip install bioformats
then run the script.

or try 
pip install javabridge
pip install bioformats
then run the script.
"""

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
            if show:
                plt.imshow(img_stacked)
                plt.show()
            skimage.io.imsave(f'{outp_dir}/{area}_annotated.png', img_stacked)




# ============= change below to your pathhhhhh / files =================
# name1 = 'Exp14_DIV14_prim01_Cycle'
# name2 = 'Exp14_DIV14_prim02_Cycle'
# name3 = 'Exp14_DIV14_prim02_real_Cycle'
# name4 = 'Exp14_DIV14_prim03_Cycle'
# name5 = 'Exp14_DIV14_prim04_Cycle_01'
# name6 = 'Exp14_DIV14_rest_Cycle'
# name7 = 'Exp14overview_Cycle'
name8 = 'main_Cycle_01'
path = '/run/media/loaloa/lbb_ssd/primitives_13.10/exp12_DIV14/'
inp_dir = f'{path}/{name8}/'
outp_dir = f'{path}/exp12_DIV14_processed_nostitch/'
notes = 'DIV14, Exp14, GFP RGC, RFP RGC, faint green=thalamus+GCamp'
# ============= change above to your pathhhhhh / files =================

# inp_dir = '/home/loaloa/ETZ_drive/biohybrid-signal-p/20.10_primitives/Exp14_DIV14_prim02_Cycle/'
# outp_dir = '/home/loaloa/ETZ_drive/biohybrid-signal-p/20.10_primitives/exp14_DIV14_processed_2/'

# inp_dir = '/home/loaloa/ETZ_drive/biohybrid-signal-p/20.10_primitives/Exp14_DIV14_prim02_real_Cycle/'
# outp_dir = '/home/loaloa/ETZ_drive/biohybrid-signal-p/20.10_primitives/exp14_DIV14_processed_3/'

# inp_dir = '/home/loaloa/ETZ_drive/biohybrid-signal-p/20.10_primitives/Exp14_DIV14_prim03_Cycle/'
# outp_dir = '/home/loaloa/ETZ_drive/biohybrid-signal-p/20.10_primitives/exp14_DIV14_processed_4/'

# inp_dir = '/home/loaloa/ETZ_drive/biohybrid-signal-p/20.10_primitives/Exp14_DIV14_prim04_Cycle_01/'
# outp_dir = '/home/loaloa/ETZ_drive/biohybrid-signal-p/20.10_primitives/exp14_DIV14_processed_5/'

# inp_dir = '/home/loaloa/ETZ_drive/biohybrid-signal-p/20.10_primitives/Exp14_DIV14_rest_Cycle/'
# outp_dir = '/home/loaloa/ETZ_drive/biohybrid-signal-p/20.10_primitives/exp14_DIV14_processed_6/'

# inp_dir = '/home/loaloa/ETZ_drive/biohybrid-signal-p/20.10_primitives/Exp14overview_Cycle_01/'
# outp_dir = '/home/loaloa/ETZ_drive/biohybrid-signal-p/20.10_primitives/exp14_DIV14_processed_7/'



# ============ change regular expression below. * indicate whatever is in between the filename =================
all_files = sorted(glob(inp_dir+'/main*_G02*.oir'))
# ============ change regular expression above. * indicate whatever is in between the filename =================

print_files = '\n\t'.join([f for f in all_files])
print(f'Found {len(all_files)} areas: \n\t{print_files}')

save_tifs = True
rotate = False
save_merged_annotated = True
scale_12bit_to16bit = True
show = False
textsize = 1

def main():
    start_java_vm()
    os.makedirs(outp_dir, exist_ok=True)
    process_dir(all_files, inp_dir, outp_dir, notes)
    javabridge.kill_vm()
if __name__ == '__main__':
    main()