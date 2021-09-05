"""
Credit to Hamza442004: https://github.com/Hamza442004/DXF2img
"""

import matplotlib.pyplot as plt
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import numpy as np

default_img_format = '.png'
default_img_res = 500
def convert_dxf2img(dxf_fname, img_format=default_img_format, img_res=default_img_res):
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
        # ctx.current_layout.set_colors(bg='#FFFFFF')
        out = MatplotlibBackend(ax)
        Frontend(ctx, out).draw_layout(msp, finalize=True)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.ubyte)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))[:,:,0]

        print(data.shape)

        plt.show()
        return data

if __name__ == '__main__':
    convert_dxf2img('./../PDMS frame files/dxf exports/design17_Dorg.dxf')