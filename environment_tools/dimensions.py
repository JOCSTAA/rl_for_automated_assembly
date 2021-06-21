import sys
sys.path.append("C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/FreeCAD_0.19.22284-Win-Conda_vc14.x-x86_64/bin")
import FreeCAD
import Draft


def create_boundary_dimensions(obj,
                               dis_bw_obj_and_dim=100):
    """ Create dimensions of length, width and height of given object."""
    ldimline = FreeCAD.Vector(0,
                              obj.Shape.BoundBox.YMin - dis_bw_obj_and_dim,
                              obj.Shape.BoundBox.ZMin)
    ldim = Draft.makeDimension(obj.Shape.BoundBox.getPoint(4),
                               obj.Shape.BoundBox.getPoint(5),
                               ldimline)
    wdimline = FreeCAD.Vector(obj.Shape.BoundBox.XMax + dis_bw_obj_and_dim,
                              0,
                              obj.Shape.BoundBox.ZMin)
    wdim = Draft.makeDimension(obj.Shape.BoundBox.getPoint(5),
                               obj.Shape.BoundBox.getPoint(6),
                               wdimline)
    hdimline = FreeCAD.Vector(obj.Shape.BoundBox.XMin - dis_bw_obj_and_dim,
                              obj.Shape.BoundBox.YMin,
                              0)
    hdim = Draft.makeDimension(obj.Shape.BoundBox.getPoint(0),
                               obj.Shape.BoundBox.getPoint(4),
                               hdimline)
    FreeCAD.ActiveDocument.recompute()
    return ldim, wdim, hdim