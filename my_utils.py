from collections import OrderedDict
import bioformats
import xmltodict
import datetime

def humanbytes(B):
   'Return the given bytes as a human friendly KB, MB, GB, or TB string'
   B = float(B)
   KB = float(1024)
   MB = float(KB ** 2) # 1,048,576
   GB = float(KB ** 3) # 1,073,741,824
   TB = float(KB ** 4) # 1,099,511,627,776

   if B < KB:
      return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
   elif KB <= B < MB:
      return '{0:.2f} KB'.format(B/KB)
   elif MB <= B < GB:
      return '{0:.2f} MB'.format(B/MB)
   elif GB <= B < TB:
      return '{0:.2f} GB'.format(B/GB)
   elif TB <= B:
      return '{0:.2f} TB'.format(B/TB)

def print_metadata(meta_dict):
    def iterate_dict(meta_dict, indent):
        for key in meta_dict.keys():
            print(indent+key, end='') 
            if type(meta_dict[key]) == OrderedDict:
                print()
                indent = '\t'+indent
                iterate_dict(meta_dict[key], indent)
            elif type(meta_dict[key]) is str:
                print(f' = \"{meta_dict[key]}\";')
            elif type(meta_dict[key]) is list:
                print(f' = list with {len(meta_dict[key])} dicts')
                for i, dict_in_list in enumerate(meta_dict[key]):
                    print(f'\t{indent} index:{i}:')
                    iterate_dict(dict_in_list, '\t'+indent)
                    print()
                print('\n\n')
    indent = ''
    iterate_dict(meta_dict, indent)

def get_metadata(oir_file, notes, dt):
    metadata = bioformats.get_omexml_metadata(oir_file)
    meta_dict = xmltodict.parse(metadata)
    all_ann = meta_dict["OME"]['StructuredAnnotations']['XMLAnnotation']

    sizey = int(meta_dict['OME']["Image"]["Pixels"]['@SizeY'])
    sizex = int(meta_dict['OME']["Image"]["Pixels"]['@SizeX'])
    sizet = int(meta_dict['OME']["Image"]["Pixels"]['@SizeT'])

    acqui_date = meta_dict['OME']["Image"]["AcquisitionDate"]
    acqui_date = [symbol if symbol.isnumeric() else '.' for symbol in acqui_date]
    acqui_date = [int(item) for item in ''.join(acqui_date).split('.')]
    acqui_date[-1] *= 1000 # to nanoseconds
    acqui_date[3] += 2 # to GMT+1
    acqui_date = str(datetime.datetime(*acqui_date))[:-10]
    duration = f'{int(dt*sizet//60)}h:{int(dt*sizet%60)}min'

    objective = meta_dict['OME']["Instrument"]["Objective"]["@Model"]
    objective = objective[objective.rfind(' ')+1:]
    pixel_size = meta_dict['OME']["Image"]["Pixels"]['@PhysicalSizeX'][:4]+' um'
    for entry in all_ann:
        entry = entry['Value']['OriginalMetadata']
        if 'opticalResolution x' in entry['Key']:
            optical_resolution = entry['Value'][:4] + ' um'
            break

    cmap = {'670.0':['RFP', 'LD561'], '540.0': ["GFP", 'LD488'], '600.0': ["GFP", 'LD488'], '672.0':['RFP', 'LD561']}
    channel_dict = OrderedDict()
    channels = meta_dict['OME']["Image"]["Pixels"]['Channel']
    for i, chnl in enumerate(channels):

        if '@EmissionWavelength' in chnl:
            color = cmap[chnl['@EmissionWavelength']][0]
            laser = cmap[chnl['@EmissionWavelength']][1]
            pinhole = f'{chnl["@PinholeSize"]} um'
            smplrate = chnl['@SamplesPerPixel']
            detector = chnl['DetectorSettings']['@ID']
            volt_gain = [f'{dets["@Voltage"]} V' for dets in meta_dict["OME"]["Instrument"]["Detector"] if dets['@ID']==detector][0]

            for entry in all_ann:
                entry = entry['Value']['OriginalMetadata']
                if entry['Key'] == f'- Laser {laser} transmissivity':
                    laser_power = entry['Value']
                    break

            summary = (f"Channel={color}, Pinhole={pinhole}, Samplerate={smplrate}\n"
                       f"Gain={volt_gain}, Laser={laser_power}%\n")

            channel_dict[color] = {'laser': laser,
                                   'id': i,
                                   'pinhole': pinhole,
                                   'smplrate': smplrate,
                                   'volt_gain': volt_gain,
                                   'laser_power': laser_power,
                                   'summary': summary}

        else:
            volt_gain = -1
            channel_dict['Transmission'] = {'id': i,
                                            'volt_gain': volt_gain,
                                            'summary': ''}
    
    summary = (f"Recording start: {acqui_date} - duration: {duration}\n"
               f"Notes: {notes}\n"
               f"Obj={objective}, Optic.Res.={optical_resolution}, PixelSize={pixel_size}\n"
               f"Y-X-T={sizey}-{sizex}-{sizet}, Area={sizey*float(pixel_size[:4])}um x {sizex*float(pixel_size[:4])}um")
    print(summary)


    channel_dict = OrderedDict(reversed(list(channel_dict.items())))
    return sizey, sizex, sizet, channel_dict, summary, meta_dict