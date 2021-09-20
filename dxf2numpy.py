"""
Credit to Hamza442004: https://github.com/Hamza442004/DXF2img
"""

import matplotlib.pyplot as plt
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import numpy as np


default_img_res = 2000
def convert_dxf2img(dxf_fname, out_fname, img_res=default_img_res):
    doc = ezdxf.readfile(dxf_fname)
    msp = doc.modelspace()
    # Recommended: audit & repair DXF document before rendering
    auditor = doc.audit()
    # The auditor.errors attribute stores severe errors,
    # which *may* raise exceptions when rendering.
    if len(auditor.errors) != 0:
        raise Exception("The DXF document is damaged and can't be converted!")
    else:
        fig = plt.figure(dpi=img_res)
        ax = fig.add_axes([0, 0, 1, 1])
        
        ctx = RenderContext(doc)
        ctx.set_current_layout(msp)
        ctx.current_layout.set_colors(bg='#FFFFFF')
        out = MatplotlibBackend(ax)
        Frontend(ctx, out).draw_layout(msp, finalize=True)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.ubyte)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))[:,:,0]

        print(data.shape)

        plt.savefig(out_fname)
        return data

if __name__ == '__main__':
    filename = '/home/loaloa/gdrive/projects/biohybrid MEA/PDMS frame files/wafer_version2/wafer_v5_cutlinesonlyplusring.dxf'
    convert_dxf2img(filename, filename.replace('.dxf', '.svg'))
    
#    import glob
 #   inp_files = glob.glob('../PDMS frame files/wafer_version2/design_masks/timelapse_designs_mask_D*.dxf')
  #  for i in range(len(inp_files)):
   #     print(inp_files[i])
    #    convert_dxf2img(inp_files[i], inp_files[i].replace('.dxf', '.svg'))
