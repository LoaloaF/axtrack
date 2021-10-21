"""
Processes .oir files from the CLSM microscope, using python-bioformats port.
First find all the .oir files in a given directory, this will match the number
of regions in timelapse recording (G001, G002, ...). Save them as .tiff sequences
using tifffile package. Optionally, save print metadata, flip the recording 
horizontally, compress the .tiff files using "deflate" and finally output an 
annotated video of the main channel merged with the transmission channel. The 
annotation includes a note about the experimental details. Output ist saved in 
`outp_dir`.
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

import javabridge
import bioformats

def start_java_vm():
    javabridge.start_vm(class_path=bioformats.JARS)
    myloglevel="ERROR"  # user string argument for logLevel.
    rootLoggerName = javabridge.get_static_field("org/slf4j/Logger","ROOT_LOGGER_NAME", "Ljava/lang/String;")
    rootLogger = javabridge.static_call("org/slf4j/LoggerFactory","getLogger", "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
    logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level",myloglevel, "Lch/qos/logback/classic/Level;")
    javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)

def oir_to_tiffstack(oir_file, channel, sizey, sizex, sizet, do_flip):
    with bioformats.ImageReader(oir_file) as reader:
        # ffmpeg can't handle odd pixel numbers...
        if sizex %2:
            sizex -= 1
        if sizey %2:
            sizey -= 1
        stacked_img = np.zeros((sizet, sizey, sizex), dtype=np.uint16)
        for t in range(sizet):
            print(f'\tt {t+1}/{sizet}', end='..')
            img = img_as_uint(reader.read(t=t, c=channel))
            if img.shape[0] != sizey:
                img = img[:-1,:]
            if img.shape[1] != sizex:
                img = img[:,:-1]
            stacked_img[t, :, :] = img
        print('Done.')
        # make all recordings have the tissue wells at the bottom
        if sizey > sizex:
            sizex, sizey = sizey, sizex
            stacked_img = np.swapaxes(stacked_img, 1, 2)
        
        if do_flip:
            stacked_img = np.flip(stacked_img, axis=1).copy()
        return stacked_img

def process_dir(all_files, inp_dir, outp_dir, notes, designs, flips):
    designgroup_order = [1, 3, 2, 4, 5]
    dt_idx = inp_dir.rfind('dt')
    dt = float(inp_dir[dt_idx+2:inp_dir.find('_', dt_idx)])
    print(f'Found {len(all_files)} areas - dt={dt}')
    for i, oir_file in enumerate(all_files):
        if i >=25:
            continue
        area_index = oir_file.find('_G0')+1
        area = oir_file[area_index:area_index+4]
        
        # structure_idx = int(area[1:])
        # designgroup_idx = (structure_idx-1) // 8
        # which_designgroup = designgroup_order[designgroup_idx] -1
        # design_idx = (structure_idx-1) % 4
        # design = f'D{which_designgroup*4 + design_idx +1:0>2}'
        design = f'D{designs[i]:0>2}'

        print(f'Processing {area} now - {design}...')
        # metadata handling
        sizey, sizex, sizet, channel_dict, summary, meta_dict = get_metadata(oir_file, notes, dt)
        summary = f'Area {area}, {design} ' + summary
        with open(f"{outp_dir}/{which_tl}_{design}_{area}_meatadata.pkl", "wb") as pkl_file:
            pickle.dump(meta_dict, pkl_file)
            print(f'{area}_meatadata.pkl saved.\n')

        for channel_color, channel in channel_dict.items():
            fname = f'{which_tl}_{design}_{area}_{channel_color}_compr.{compress}.tif'
            print(f'\tReading in {channel_color} channel...\n{channel["summary"]}')
            stacked_img = oir_to_tiffstack(oir_file, channel['id'], sizey, sizex, 
                                        sizet if channel_color != 'Transmission' else 1, flips[i])

            # save single channels
            if save_tifs:
                print(f'\tSaving {humanbytes(sys.getsizeof(stacked_img))} image'
                      f' sequence (compression={compress})...', end='')
                imsave(f'{outp_dir}/{fname}', stacked_img, 
                            photometric='minisblack', compression=compress, bigtiff=True)
                print('Done.')

            # produces one video for each channel+transmission channel
            if make_video:
                print('\tConverting to rgb video...', end='')
                if channel_color == 'Transmission':
                    background = np.stack([stacked_img[0]*3]*3, -1)
                    background = np.stack([background]*sizet, 0)

                else:
                    chnl_id_rgb = 0 if channel_color == 'RFP' else 1
                    rgb_vid = background.copy()
                    
                    rgb_vid[:,:,:,chnl_id_rgb] += stacked_img*13
                    del stacked_img

                    fname = fname[:fname.rfind('_')] + '.mp4'
                    # frames are saved tmp bc ffmpeg gave me loooots of trouble
                    [os.remove(f) for f in glob(f'{outp_dir}../tmp/*')]
                    time = 0.
                    for t in range(sizet):
                        frame = rgb_vid[t]
                        if annotate_video:
                            text = f'{summary}\n\n{channel["summary"]}'
                            for i, line in enumerate(text.split('\n')):
                                cv2.putText(frame, line, 
                                            (20,(i+1)*70), cv2.FONT_HERSHEY_DUPLEX, 
                                            1*textsize, (2**15,2**15,2**15), 3)
                            
                            frame_time = f'{int(time//60)}h{int(time%60)}min'
                            cv2.putText(frame, frame_time, 
                                        (sizex-250,70), cv2.FONT_HERSHEY_DUPLEX, 
                                        1*textsize, (2**15,2**15,2**15), 3)
                            time += dt

                        if sizex*sizey > (15*10**6):
                            x, y = [int(frame.shape[1]*.7), int(frame.shape[0]*.7)]
                            x = x-1 if x%2 else x
                            y = y-1 if y%2 else y
                            frame = cv2.resize(frame, (x,y))
                        skimage.io.imsave(f'{outp_dir}../tmp/frame_{t:0>3}.tif', frame, check_contrast=False)
                        if t == sizet-1:
                            skimage.io.imsave(f'{outp_dir}../tmp/frame_{t+1:0>3}.tif', frame, check_contrast=False)
                            skimage.io.imsave(f'{outp_dir}../tmp/frame_{t+2:0>3}.tif', frame, check_contrast=False)
                            skimage.io.imsave(f'{outp_dir}../tmp/frame_{t+3:0>3}.tif', frame, check_contrast=False)
                    del rgb_vid

                    makevideo = f'ffmpeg -y -framerate 4 -i {outp_dir}../tmp/'+'frame_%03d.tif -hide_banner -loglevel error -c:v libx264 -crf 20 -pix_fmt yuv420p '+f'{outp_dir}/{fname}'
                    os.system(makevideo)
                    [os.remove(f) for f in glob(f'{outp_dir}../tmp/*')]
                print('Done.\n')

# INPUT TO DATA PROCESSING PIPELINE
notes = []
inp_dir = []
outp_dir = []
all_files = []
all_designs = []
all_flips = []

designs = [1, 2,    4, 5, 6, 7, 8,
           5, 6, 7, 8, 1, 2, 3, 4,
           9, 10, 11, 12, 13, 14, 15, 16,
           13, 14, 15, 16, 9, 10, 11, 12,
           17, 18, 19, 20, 
           17, 18, 19, 20,
           21, 21
]
flips = [True,True,      True,True,True,True,True,
        False,False,False,False,False,False,False,False,
        True,True,True,True,True,True,True,True,
        False,False,False,False,False,False,False,False,
        True,True,True,True,
        False,False,False,False,
        True,True
]



notes.append("idk dt=31:00min")
inp_dir.append('/run/media/loaloa/lbb_ssd/timelapse14_Exp15/Exp15_infect_dt32_Cycle/')
outp_dir.append('/run/media/loaloa/lbb_ssd/timelapse14_processed/')
all_files.append(sorted(glob(inp_dir[-1]+'/Stitch*_G0*.oir')))
print(sorted(glob(inp_dir[-1]+'/Stitch*_G0*.oir')))
all_designs.append(designs)
all_flips.append(flips)

# notes.append("tl12, 1.5h incubation-2hinc, Exp7,\ndt=42:42min")
# inp_dir.append('/run/media/loaloa/lbb_ssd/timelapse12_Exp7/Exp7_dt42.42_Cycle/')
# outp_dir.append('/run/media/loaloa/lbb_ssd/timelapse12_processed/')
# all_files.append(sorted(glob(inp_dir[-1]+'Stitch*_G0*.oir')))

# notes.append("tl10.1, oldCtx_20hinc_5hinc, Exp5.1,")
# inp_dir.append('/run/media/loaloa/lbb_ssd/timelapse10.1_Exp5.1/Exp5.1_dt12_1to10GFOvir_20+5hinc_Cycle/')
# outp_dir.append('/run/media/loaloa/lbb_ssd/timelapse10.1_processed/')
# all_files.append(sorted(glob(inp_dir[-1]+'Stitch*_G0*.oir')))

# notes.append("tl10.2, oldCtx_20hinc_5hinc, Exp5.2,")
# inp_dir.append('/run/media/loaloa/lbb_ssd/timelapse10.2_Exp5.1/Exp5.1_tl10.2_longinc_dt6.10_Cycle/')
# outp_dir.append('/run/media/loaloa/lbb_ssd/timelapse10.2_processed/')
# all_files.append(sorted(glob(inp_dir[-1]+'Stitch*_G0*.oir')))

# notes.append("tl11.1, spheroids_zproblems_1.5dinc_20hinc, Exp5.2,\ndt=31:00min, 19h total")
# inp_dir.append('/run/media/loaloa/lbb_ssd/tl11.1/Exp6.2_spheroids_zproblems_1.5dinc_20hinc_dt31_Cycle/')
# outp_dir.append('/run/media/loaloa/lbb_ssd/timelapse11.1_processed/')
# all_files.append(sorted(glob(inp_dir[-1]+'Stitch*_G0*.oir')))

save_tifs = True
compress = 'deflate'
make_video = False
annotate_video = True
textsize = 1.3
which_tl = 'tl14'


def main():
    start_java_vm()
    for data_idx in range(len(notes)):
        print('\n\n\n\n\n\n')
        os.makedirs(outp_dir[data_idx], exist_ok=True)
        process_dir(all_files[data_idx], inp_dir[data_idx], outp_dir[data_idx], 
                    notes[data_idx], all_designs[data_idx], all_flips[data_idx])
    javabridge.kill_vm()

if __name__  ==  '__main__':
    main()