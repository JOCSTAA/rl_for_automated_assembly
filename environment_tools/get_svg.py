""" Export dimensions into SVG."""
import sys
sys.path.append("C:/Users/josho/Desktop/school_work/PROJECT/free_cad_stuffs/FreeCAD_0.19.22284-Win-Conda_vc14.x-x86_64/bin")
import math

import FreeCAD
import DraftVecUtils
import WorkingPlane
import Part
import DraftGeomUtils

from Draft import getType, getrgb


def get_svg_of_dimension(obj,
                         scale=1,
                         linewidth=0.35,
                         fontsize=12,
                         fillstyle='shape color',
                         direction=None,
                         linestyle=None,
                         color='#0000FF',
                         linespacing=None,
                         techdraw=False,
                         rotation=0):
    """ Returns a string containing a SVG representation of the given object,
    with the given linewidth and fontsize (used if the given object contains
    any text). You can also supply an arbitrary projection vector. the
    scale parameter allows to scale linewidths down, so they are resolution-independant."""

    # if this is a group, gather all the svg views of its children
    if hasattr(obj, 'isDerivedFrom'):
        if obj.isDerivedFrom('App::DocumentObjectGroup'):
            svg = ""
            for child in obj.Group:
                svg += get_svg_of_dimension(child,
                                            scale,
                                            linewidth,
                                            fontsize,
                                            fillstyle,
                                            direction,
                                            linestyle,
                                            color,
                                            linespacing,
                                            techdraw)
            return svg

    pathdata = []
    svg = ""
    linewidth = float(linewidth) / scale
    fontsize = (float(fontsize) / scale) / 2
    if linespacing:
        linespacing = float(linespacing) / scale
    else:
        linespacing = 0.5
    pointratio = 0.75  # the number of times the dots are smaller than the arrow size
    plane = None
    if direction:
        if isinstance(direction, FreeCAD.Vector):
            if direction != FreeCAD.Vector(0, 0, 0):
                plane = WorkingPlane.plane()
                plane.alignToPointAndAxis_SVG(FreeCAD.Vector(0, 0, 0), direction.negative().negative(), 0)
        elif isinstance(direction, WorkingPlane.plane):
            plane = direction
    stroke = '#000000'
    if color:
        if '#' in color:
            stroke = color
        else:
            stroke = getrgb(color)

    def get_line_style():
        """Returns a linestyle."""
        p = FreeCAD.ParamGet('User parameter:BaseApp/Preferences/Mod/Draft')
        l = None
        if linestyle == 'Dashed':
            l = p.GetString('svgDashedLine', '0.09,0.05')
        elif linestyle == 'Dashdot':
            l = p.GetString('svgDashdotLine', '0.09,0.05,0.02,0.05')
        elif linestyle == 'Dotted':
            l = p.GetString('svgDottedLine', '0.02,0.02')
        elif linestyle:
            if ',' in linestyle:
                l = linestyle
        if l:
            l = l.split(',')
            try:
                # scale dashes
                l = ','.join([str(float(d) / scale) for d in l])
            except:
                return 'none'
            else:
                return l
        return 'none'

    def get_proj(vec):
        if not plane:
            return vec
        nx = DraftVecUtils.project(vec, plane.u)
        lx = nx.Length
        if abs(nx.getAngle(plane.u)) > 0.1:
            lx = -lx
        ny = DraftVecUtils.project(vec, plane.v)
        ly = ny.Length
        if abs(ny.getAngle(plane.v)) > 0.1:
            ly = -ly
        #if techdraw: buggy - we now simply do it at the end
        #    ly = -ly
        return FreeCAD.Vector(lx, ly, 0)

    def get_discretized(edge):
        ml = FreeCAD.ParamGet('User parameter:BaseApp/Preferences/Mod/Draft').GetFloat('svgDiscretization', 10.0)
        if ml == 0:
            ml = 10
        d = int(edge.Length / ml)
        if d == 0:
            d = 1
        edata = ""
        for i in range(d + 1):
            v = get_proj(edge.valueAt(edge.FirstParameter + ((float(i) / d) * (edge.LastParameter - edge.FirstParameter))))
            if not edata:
                edata += 'M ' + str(v.x) +' '+ str(v.y) + ' '
            else:
                edata += 'L ' + str(v.x) +' '+ str(v.y) + ' '
        return edata

    def get_pattern(pat):
        if pat in svgpatterns():
            return svgpatterns()[pat][0]
        return ''

    def get_path(edges=[], wires=[], pathname=None):
        svg = "<path "
        if pathname is None:
            svg += 'id="%s" ' % obj.Name
        elif pathname != "":
            svg += 'id="%s" ' % pathname
        svg += ' d="'
        if not wires:
            egroups = Part.sortEdges(edges)
        else:
            egroups = []
            for w in wires:
                w1 = w.copy()
                w1.fixWire()
                egroups.append(Part.__sortEdges__(w1.Edges))
        for egroupindex, edges in enumerate(egroups):
            edata = ""
            vs = ()  # skipped for the first edge
            for edgeindex, e in enumerate(edges):
                previousvs = vs
                # vertexes of an edge (reversed if needed)
                vs = e.Vertexes
                if previousvs:
                    if (vs[0].Point - previousvs[-1].Point).Length > 1e-6:
                        vs.reverse()
                if edgeindex == 0:
                    v = get_proj(vs[0].Point)
                    edata += 'M '+ str(v.x) +' '+ str(v.y) + ' '
                else:
                    if (vs[0].Point - previousvs[-1].Point).Length > 1e-6:
                        raise ValueError('edges not ordered')
                iscircle = DraftGeomUtils.geomType(e) == 'Circle'
                isellipse = DraftGeomUtils.geomType(e) == 'Ellipse'
                if iscircle or isellipse:
                    if hasattr(FreeCAD, 'DraftWorkingPlane'):
                        drawing_plane_normal = FreeCAD.DraftWorkingPlane.axis
                    else:
                        drawing_plane_normal = FreeCAD.Vector(0, 0, 1)
                    if plane:
                        drawing_plane_normal = plane.axis
                    c = e.Curve
                    if round(c.Axis.getAngle(drawing_plane_normal), 2) in [0, 3.14]:
                        occversion = Part.OCC_VERSION.split('.')
                        done = False
                        if (occversion[0] >= 7) and (occversion[1] >= 1):
                            # if using occ >= 7.1, use HLR algorithm
                            import Drawing
                            snip = Drawing.projectToSVG(e,drawing_plane_normal)
                            if snip:
                                try:
                                    a = "A " + snip.split("path d=\"")[1].split("\"")[0].split("A")[1]
                                except:
                                    pass
                                else:
                                    edata += a
                                    done = True
                        if not done:
                            if len(e.Vertexes) == 1 and iscircle:  # complete curve
                                svg = get_circle(e)
                                return svg
                            elif len(e.Vertexes) == 1 and isellipse:
                                endpoints = (get_proj(c.value((c.LastParameter - \
                                        c.FirstParameter) / 2.0)), \
                                        get_proj(vs[-1].Point))
                            else:
                                endpoints = (get_proj(vs[-1].Point),)
                            # arc
                            if iscircle:
                                rx = ry = c.Radius
                                rot = 0
                            else:  # ellipse
                                rx = c.MajorRadius
                                ry = c.MinorRadius
                                rot = math.degrees(c.AngleXU * (c.Axis * \
                                    FreeCAD.Vector(0, 0, 1)))
                                if rot > 90:
                                    rot -=180
                                if rot < -90:
                                    rot += 180
                                # be careful with the sweep flag
                            flag_large_arc = (((e.ParameterRange[1] - \
                                    e.ParameterRange[0]) / math.pi) % 2) > 1
                            # flag_sweep = (c.Axis * drawing_plane_normal >= 0) \
                            #         == (e.LastParameter > e.FirstParameter)
                            #        == (e.Orientation == "Forward")
                            # other method: check the direction of the angle between tangents
                            t1 = e.tangentAt(e.FirstParameter)
                            t2 = e.tangentAt(e.FirstParameter + (e.LastParameter - e.FirstParameter) / 10)
                            flag_sweep = (DraftVecUtils.angle(t1, t2, drawing_plane_normal) < 0)
                            for v in endpoints:
                                edata += 'A %s %s %s %s %s %s %s ' % \
                                        (str(rx), str(ry), str(rot),\
                                        str(int(flag_large_arc)),\
                                        str(int(flag_sweep)), str(v.x), str(v.y))
                    else:
                        edata += get_discretized(e)
                elif DraftGeomUtils.geomType(e) == 'Line':
                    v = get_proj(vs[-1].Point)
                    edata += 'L ' + str(v.x) + ' ' + str(v.y) + ' '
                else:
                    bspline = e.Curve.toBSpline(e.FirstParameter, e.LastParameter)
                    if bspline.Degree > 3 or bspline.isRational():
                        try:
                            bspline = bspline.approximateBSpline(0.05, 50, 3, 'C0')
                        except RuntimeError:
                            print('Debug: unable to approximate bspline')
                    if bspline.Degree <= 3 and not bspline.isRational():
                        for bezierseg in bspline.toBezier():
                            if bezierseg.Degree>3:  # should not happen
                                raise AssertionError
                            elif bezierseg.Degree == 1:
                                edata += 'L '
                            elif bezierseg.Degree == 2:
                                edata += 'Q '
                            elif bezierseg.Degree == 3:
                                edata += 'C '
                            for pole in bezierseg.getPoles()[1:]:
                                v = get_proj(pole)
                                edata += str(v.x) + ' ' + str(v.y) + ' '
                    else:
                        print('Debug: one edge (hash ', e.hashCode(), \
                                ') has been discretized with parameter 0.1')
                        for linepoint in bspline.discretize(0.1)[1:]:
                            v = get_proj(linepoint)
                            edata += 'L ' + str(v.x) + ' ' + str(v.y) + ' '
            if fill != 'none':
                edata += 'Z '
            if edata in pathdata:
                # do not draw a path on another identical path
                return ''
            else:
                svg += edata
                pathdata.append(edata)
        svg += '" '
        svg += 'stroke="' + stroke + '" '
        svg += 'stroke-width="' + str(linewidth) + ' px" '
        svg += 'style="stroke-width:'+ str(linewidth)
        svg += ';stroke-miterlimit:4'
        svg += ';stroke-dasharray:' + lstyle
        svg += ';fill:' + fill
        try:
            svg += ';fill-opacity:' + str(fill_opacity)
        except NameError:
            pass
        svg += ';fill-rule: evenodd "'
        svg += '/>\n'
        return svg

    def get_circle(edge):
        cen = get_proj(edge.Curve.Center)
        rad = edge.Curve.Radius
        if hasattr(FreeCAD, 'DraftWorkingPlane'):
            drawing_plane_normal = FreeCAD.DraftWorkingPlane.axis
        else:
            drawing_plane_normal = FreeCAD.Vector(0, 0, 1)
        if plane:
            drawing_plane_normal = plane.axis
        if round(edge.Curve.Axis.getAngle(drawing_plane_normal), 2) == 0:
            # perpendicular projection: circle
            svg = '<circle cx="' + str(cen.x)
            svg += '" cy="' + str(cen.y)
            svg += '" r="' + str(rad) + '" '
        else:
            # any other projection: ellipse
            svg = '<path d="'
            svg += get_discretized(edge)
            svg += '" '
        svg += 'stroke="' + stroke + '" '
        svg += 'stroke-width="' + str(linewidth) + ' px" '
        svg += 'style="stroke-width:'+ str(linewidth)
        svg += ';stroke-miterlimit:4'
        svg += ';stroke-dasharray:' + lstyle
        svg += ';fill:' + fill + '"'
        svg += '/>\n'
        return svg

    def get_ellipse(edge):
        cen = get_proj(edge.Curve.Center)
        mir = edge.Curve.MinorRadius
        mar = edge.Curve.MajorRadius
        svg = '<ellipse cx="' + str(cen.x)
        svg += '" cy="' + str(cen.y)
        svg += '" rx="' + str(mar)
        svg += '" ry="' + str(mir) + '" '
        svg += 'stroke="' + stroke + '" '
        svg += 'stroke-width="' + str(linewidth) + ' px" '
        svg += 'style="stroke-width:' + str(linewidth)
        svg += ';stroke-miterlimit:4'
        svg += ';stroke-dasharray:' + lstyle
        svg += ';fill:' + fill + '"'
        svg += '/>\n'
        return svg

    def get_arrow(arrowtype, point, arrowsize, color, linewidth, angle=0):
        svg = ""
        if arrowtype == 'Circle':
            svg += '<circle cx="' + str(point.x) + '" cy="' + str(point.y)
            svg += '" r="' + str(arrowsize) + '" '
            svg += 'fill="none" stroke="' + color + '" '
            svg += 'style="stroke-width:'+ str(linewidth) + ';stroke-miterlimit:4;stroke-dasharray:none" '
            svg += 'freecad:skip="1"'
            svg += '/>\n'
        elif arrowtype == 'Dot':
            svg += '<circle cx="' + str(point.x) + '" cy="' + str(point.y)
            svg += '" r="' + str(arrowsize) + '" '
            svg += 'fill="' + color +'" stroke="none" '
            svg += 'style="stroke-miterlimit:4;stroke-dasharray:none" '
            svg += 'freecad:skip="1"'
            svg += '/>\n'
        elif arrowtype == 'Arrow':
            svg += '<path transform="rotate(' + str(math.degrees(angle))
            svg += ',' + str(point.x) + ',' + str(point.y) + ') '
            svg += 'translate(' + str(point.x) + ',' + str(point.y) + ') '
            svg += 'scale(' + str(arrowsize) + ',' + str(arrowsize) + ')" freecad:skip="1" '
            svg += 'fill="' + color + '" stroke="none" '
            svg += 'style="stroke-miterlimit:4;stroke-dasharray:none" '
            svg += 'd="M 0 0 L 4 1 L 4 -1 Z"/>\n'
        elif arrowtype == 'Tick':
            svg += '<path transform="rotate(' + str(math.degrees(angle))
            svg += ',' + str(point.x) + ',' + str(point.y) + ') '
            svg += 'translate(' + str(point.x) + ',' + str(point.y) + ') '
            svg += 'scale(' + str(arrowsize) + ',' + str(arrowsize) + ')" freecad:skip="1" '
            svg += 'fill="' + color + '" stroke="none" '
            svg += 'style="stroke-miterlimit:4;stroke-dasharray:none" '
            svg += 'd="M -1 -2 L 0 2 L 1 2 L 0 -2 Z"/>\n'
        elif arrowtype == 'Tick-2':
            svg += '<line transform="rotate(' + str(math.degrees(angle) + 45)
            svg += ',' + str(point.x) + ',' + str(point.y) + ') '
            svg += 'translate(' + str(point.x) + ',' + str(point.y) + ') '
            svg += '" freecad:skip="1" '
            svg += 'fill="none" stroke="' + color + '" '
            svg += 'style="stroke-dasharray:none;stroke-linecap:square;'
            svg += 'stroke-width:' + str(linewidth) + '" '
            svg += 'x1="-' + str(arrowsize*2) + '" y1="0" '
            svg += 'x2="' + str(arrowsize*2) + '" y2="0" />\n'
        else:
                print("getSVG: arrow type not implemented")
        return svg

    def get_overshoot(point, shootsize, color, linewidth, angle=0):
        svg = '<line transform="rotate(' + str(math.degrees(angle))
        svg += ',' + str(point.x) + ',' + str(point.y) + ') '
        svg += 'translate(' + str(point.x) + ',' + str(point.y) + ') '
        svg += '" freecad:skip="1" '
        svg += 'fill="none" stroke="' + color + '" '
        svg += 'style="stroke-dasharray:none;stroke-linecap:square;'
        svg += 'stroke-width:' + str(linewidth) + '" '
        svg += 'x1="0" y1="0" '
        svg += 'x2="' + str(shootsize*-1) + '" y2="0" />\n'
        return svg

    def get_text(color, fontsize, fontname, angle, base, text, linespacing=0.5, align="center", flip=True):
        if isinstance(angle, FreeCAD.Rotation):
            if not plane:
                angle = angle.Angle
            else:
                if plane.axis.getAngle(angle.Axis) < 0.001:
                    angle = angle.Angle
                elif abs(plane.axis.getAngle(angle.Axis) - math.pi) < 0.001:
                    return ''  # text is perpendicular to view, so it shouldn't appear
                else:
                    angle = 0  # TODO maybe there is something better to do here?
        if not isinstance(text, list):
            text = text.split('\n')
        if align.lower() == 'center':
            anchor = 'middle'
        elif align.lower() == 'left':
            anchor = 'start'
        else:
            anchor = 'end'
        if techdraw:
            svg = ""
            for i in range(len(text)):
                t = text[i]
                #if not isinstance(t,unicode):
                #    t = t.decode("utf8")
                # possible workaround if UTF8 is unsupported
                #    import unicodedata
                #    t = u"".join([c for c in unicodedata.normalize("NFKD",t) if not unicodedata.combining(c)]).encode("utf8")
                svg += '<text fill="' + color +'" font-size="' + str(fontsize) + '" '
                svg += 'style="text-anchor:' + anchor + ';text-align:' + align.lower() + ';'
                svg += 'font-family:' + fontname + '" '
                svg += 'transform="rotate(' + str(math.degrees(angle))
                svg += ',' + str(base.x) + ',' + str(base.y - linespacing * i) + ') '
                svg += 'translate(' + str(base.x) + ',' + str(base.y - linespacing * i) + ') '
                svg += 'scale(1,-1)" '
                #svg += '" freecad:skip="1"'
                svg += '>\n' + t + '</text>\n'
        else:
            svg = '<text fill="'
            svg += color + '" font-size="'
            svg += str(fontsize) + '" '
            svg += 'style="text-anchor:' + anchor + ';text-align:' + align.lower() + ';'
            svg += 'font-family:' + fontname + '" '
            svg += 'transform="rotate(' + str(math.degrees(angle))
            svg += ','+ str(base.x) + ',' + str(base.y) + ') '
            if flip:
                svg += 'translate(' + str(base.x) + ',' + str(base.y) + ')'
            else:
                svg += 'translate(' + str(base.x) + ',' + str(-base.y) + ')'
            # svg += 'scale('+str(tmod/2000)+',-'+str(tmod/2000)+') '
            if flip:
                svg += ' scale(1,-1) '
            else:
                svg += ' scale(1,1) '
            svg += '" freecad:skip="1"'
            svg += '>\n'
            if len(text) == 1:
                try:
                    svg += text[0]
                except:
                    svg += text[0].decode('utf8')
            else:
                for i in range(len(text)):
                    if i == 0:
                        svg += '<tspan>'
                    else:
                        svg += '<tspan x="0" dy="' + str(linespacing) + '">'
                    try:
                        svg += text[i]
                    except:
                        svg += text[i].decode('utf8')
                    svg += '</tspan>\n'
            svg += '</text>\n'
        return svg


    # calculate the 4 points
    p1 = obj.Start
    p4 = obj.End
    base = None
    if hasattr(obj, 'Direction'):
        if not DraftVecUtils.isNull(obj.Direction):
            v2 = p1.sub(obj.Dimline)
            v3 = p4.sub(obj.Dimline)
            v2 = DraftVecUtils.project(v2, obj.Direction)
            v3 = DraftVecUtils.project(v3, obj.Direction)
            p2 = obj.Dimline.add(v2)
            p3 = obj.Dimline.add(v3)
            if DraftVecUtils.equals(p2, p3):
                base = None
                proj = None
            else:
                base = Part.LineSegment(p2, p3).toShape()
                proj = DraftGeomUtils.findDistance(p1, base)
                if proj:
                    proj = proj.negative()
    if not base:
        if DraftVecUtils.equals(p1, p4):
            base = None
            proj = None
        else:
            base = Part.LineSegment(p1, p4).toShape()
            proj = DraftGeomUtils.findDistance(obj.Dimline, base)
        if proj:
            p2 = p1.add(proj.negative())
            p3 = p4.add(proj.negative())
        else:
            p2 = p1
            p3 = p4
    if proj:
        # if hasattr(obj.ViewObject,"ExtLines"):
        extlines = 0
        DisplayMode = '2D'
        dmax = extlines
        if dmax and (proj.Length > dmax):
                if (dmax > 0):
                    p1 = p2.add(DraftVecUtils.scaleTo(proj, dmax))
                    p4 = p3.add(DraftVecUtils.scaleTo(proj, dmax))
                else:
                    rest = proj.Length + dmax
                    p1 = p2.add(DraftVecUtils.scaleTo(proj, rest))
                    p4 = p3.add(DraftVecUtils.scaleTo(proj, rest))
    else:
        proj = p3.sub(p2).cross(FreeCAD.Vector(0, 0, 1))
    if getType(obj) == 'Dimension':
        fontsize = 60
        ts = (len(str(obj.Distance)) * fontsize) / 4.0
        rm = ((p3.sub(p2)).Length / 2.0) - ts
        p2a = get_proj(p2.add(DraftVecUtils.scaleTo(p3.sub(p2), rm)))
        p2b = get_proj(p3.add(DraftVecUtils.scaleTo(p2.sub(p3), rm)))
        p1 = get_proj(p1)
        p2 = get_proj(p2)
        p3 = get_proj(p3)
        p4 = get_proj(p4)

        # calculate the text position and orientation
        if hasattr(obj, 'Normal'):
            if DraftVecUtils.isNull(obj.Normal):
                if proj:
                    norm = (p3.sub(p2).cross(proj)).negative()
                else:
                    norm = Vector(0, 0, 1)
            else:
                norm = FreeCAD.Vector(obj.Normal)
        else:
            if proj:
                norm = (p3.sub(p2).cross(proj)).negative()
            else:
                norm = Vector(0, 0, 1)
        if not DraftVecUtils.isNull(norm):
            norm.normalize()
        u = p3.sub(p2)
        u.normalize()
        v1 = norm.cross(u)
        rot1 = FreeCAD.Placement(DraftVecUtils.getPlaneRotation(u, v1, norm)).Rotation.Q
        FLIPARROWS = True
        if FLIPARROWS:
            u = u.negative()
        v2 = norm.cross(u)
        rot2 = FreeCAD.Placement(DraftVecUtils.getPlaneRotation(u, v2, norm)).Rotation.Q
        if p1 != p2:
            u3 = p1.sub(p2)
            u3.normalize()
            v3 = norm.cross(u3)
            rot3 = FreeCAD.Placement(DraftVecUtils.getPlaneRotation(u3, v3, norm)).Rotation.Q
        TEXTSPACING=60
        if TEXTSPACING:
            offset = DraftVecUtils.scaleTo(v1, TEXTSPACING)
        else:
            offset = DraftVecUtils.scaleTo(v1, 0.05)
        rott = rot1
        FLIPTEXT = True
        if FLIPTEXT:
            rott = FreeCAD.Rotation(*rott).multiply(FreeCAD.Rotation(norm, 180)).Q
            offset = offset.negative()

        tbase = (p2.add((p3.sub(p2).multiply(0.5)))).add(offset)
        angle = - DraftVecUtils.angle(p3.sub(p2))

        # drawing lines
        svg = '<path '
        if DisplayMode == '2D':
            tangle = angle
            if tangle > math.pi / 2:
                tangle = tangle - math.pi
            if rotation != 0:
                if abs(tangle + math.radians(rotation)) < 0.0001:
                    tangle += math.pi
                    tbase = tbase.add(DraftVecUtils.rotate(FreeCAD.Vector(0, 2 / scale, 0), tangle))
            svg += 'd="M ' + str(p1.x) + ' ' + str(p1.y) + ' '
            svg += 'L ' + str(p2.x) + ' ' + str(p2.y) + ' '
            svg += 'L '+str(p3.x) + ' ' + str(p3.y) + ' '
            svg += 'L ' + str(p4.x) + ' '+str(p4.y) + '" '
        else:
            tangle = 0
            if rotation != 0:
                tangle = - math.radians(rotation)
            tbase = tbase.add(FreeCAD.Vector(0, -2.0 / scale, 0))
            svg += 'd="M ' + str(p1.x) + ' ' + str(p1.y) + ' '
            svg += 'L ' + str(p2.x) + ' ' + str(p2.y) + ' '
            svg += 'L ' + str(p2a.x) + ' ' + str(p2a.y) + ' '
            svg += 'M ' + str(p2b.x) + ' ' + str(p2b.y) + ' '
            svg += 'L ' + str(p3.x) + ' ' + str(p3.y) + ' '
            svg += 'L ' + str(p4.x) + ' ' + str(p4.y) + '" '

        svg += 'fill="none" stroke="'
        svg += stroke + '" '
        svg += 'stroke-width="' + str(linewidth) + ' px" '
        svg += 'style="stroke-width:' + str(linewidth)
        svg += ';stroke-miterlimit:4;stroke-dasharray:none" '
        svg += 'freecad:basepoint1="' + str(p1.x) + ' ' + str(p1.y) + '" '
        svg += 'freecad:basepoint2="' + str(p4.x) + ' ' + str(p4.y) + '" '
        svg += 'freecad:dimpoint="' + str(p2.x) + ' ' + str(p2.y) + '"'
        svg += '/>\n'

#        # drawing dimension and extension lines overshoots
#        if hasattr(obj.ViewObject,"DimOvershoot") and obj.ViewObject.DimOvershoot.Value:
#            shootsize = obj.ViewObject.DimOvershoot.Value/pointratio
#            svg += get_overshoot(p2,shootsize,stroke,linewidth,angle)
#            svg += get_overshoot(p3,shootsize,stroke,linewidth,angle+math.pi)
#        if hasattr(obj.ViewObject,"ExtOvershoot") and obj.ViewObject.ExtOvershoot.Value:
#            shootsize = obj.ViewObject.ExtOvershoot.Value/pointratio
#            shootangle = -DraftVecUtils.angle(p1.sub(p2))
#            svg += get_overshoot(p2,shootsize,stroke,linewidth,shootangle)
#            svg += get_overshoot(p3,shootsize,stroke,linewidth,shootangle)
#
#        # drawing arrows
        #if hasattr(obj.ViewObject,"ArrowType"):

        FlipArrows = False
        ArrowSize = 10
        ArrowType = 'Arrow'
        arrowsize = ArrowSize / pointratio
        if FlipArrows:
            angle = angle + math.pi
        svg += get_arrow(ArrowType, p2, arrowsize, stroke, linewidth, angle)
        svg += get_arrow(ArrowType, p3, arrowsize, stroke, linewidth, angle + math.pi)

        # drawing text
        # mid_point = FreeCAD.Vector((p2.x + p3.x) / 2, (p2.y + p3.y) / 2,(p2.z + p3.z) / 2)
        # mid_point = mid_point.add(offset)

        FontName = ''
        total_length = (obj.Start.sub(obj.End)).Length
        obj.Distance = total_length
        svg += get_text(stroke, fontsize, FontName, tangle, tbase, str(obj.Distance))

    # techdraw expects bottom-to-top coordinates
    if techdraw:
        svg = '<g transform ="scale(1,-1)">' + svg + '</g>'
    return svg