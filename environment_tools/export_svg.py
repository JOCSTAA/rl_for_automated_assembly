""" Export FreeCAD objects to SVG."""
import sys
sys.path.append("C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/FreeCAD_0.19.22284-Win-Conda_vc14.x-x86_64/bin")
import FreeCAD

from get_svg import get_svg_of_dimension


def get_svg_representation_of_obj(obj,
                                  direction=FreeCAD.Vector(1, -1, 1),  # Isometric
                                  scale=1,
                                  rotation=60,
                                  line_width=1,
                                  hidden_line_width=.5,
                                  show_hidden_line=False):
    """ Get SVG representation of a given object."""
    if hasattr(obj, 'Label'):
        view = FreeCAD.ActiveDocument.addObject('Drawing::FeatureViewPart', 'View_' + obj.Label)
    else:
        view = FreeCAD.ActiveDocument.addObject('Drawing::FeatureViewPart', 'View')
    view.Source = obj
    view.Direction = direction
    view.Rotation = rotation
    view.Scale = scale
    view.LineWidth = line_width
    view.HiddenWidth = hidden_line_width
    view.ShowHiddenLines = show_hidden_line
    # FreeCAD.ActiveDocument.recompute()
    view.recompute()
    svg = view.ViewResult
    # FreeCAD.ActiveDocument.removeObject(view.Name)
    return svg

def export_svg(objs,
               direction=FreeCAD.Vector(1, -1, 1),  # Isometric
               scale = None,
               rotation=60,
               line_width=1,
               hidden_line_width=.5,
               show_hidden_line=True,
               dims=None,
               file_path=None,
                view_box = 1):
    hidden_line_width = line_width/10
    show_hidden_line = True
    """ Exports FreeCAD objects to SVG.
    If file_path is None then it will return a SVG string
    otherwise it will save content to a given file_path."""
    svg = '''<?xml version="1.0"?>
             <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
             "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">'''
    # TODO: Calculate viewbox according to object size.
    # for table
    if view_box == 1:
        svg += '''<svg viewBox="-350 -650 1000 1000" xmlns="http://www.w3.org/2000/svg" xmlns:freecad="FreeCAD" version="1.1">'''

    else:
        svg += '''<svg viewBox="-550 -550 1000 1000" xmlns="http://www.w3.org/2000/svg" xmlns:freecad="FreeCAD" version="1.1">'''

    for obj in objs:
        svg += get_svg_representation_of_obj(obj,
                                             direction,
                                             scale,
                                             rotation,
                                             line_width,
                                             hidden_line_width,
                                             show_hidden_line)

    if dims:
        svg += get_svg_path_dimensions(dims, direction, scale, rotation, line_width, 75)

    svg += '''</svg>'''
    if file_path:
        with open(file_path, 'w') as f:
            f.write(svg)
            f.close()
    else:
        return svg

def get_svg_path_dimensions(dims,
                            direction = FreeCAD.Vector(1, -1, 1),
                            scale=1,
                            rotation=60,
                            line_width=5,
                            fontsize=10):
    """ Get SVG path of given dimensions."""
    svg_path = '''<g transform = "rotate({}, 0, 0) translate(0, 0) scale({},{})">'''.format(rotation, scale, scale)
    for dim in dims:
        svg_path += get_svg_of_dimension(obj=dim,
                                         direction=FreeCAD.Vector(1, -1, 1),
                                         rotation=60,
                                         linewidth=line_width,
                                         fontsize=fontsize,
                                         techdraw=True)
    svg_path += '''</g>'''
    return svg_path