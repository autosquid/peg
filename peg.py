"""3D geometry in Python"""
import numpy as np
import cv2
from sklearn import linear_model

def normFrom3Points(C, L0, L1):
    return np.cross(L0-C, L1-C)


def planeFrom3Points(C, L0, L1):
    """Returns: Plane Object"""
    n = normFrom3Points(C, L0, L1)
    return Plane(C, n)


class Plane:
    """given a point P on plane, and the normal direction of the plane"""
    def __init__(self, P, N):
        self.P = P
        self.N = N

    def intersection_line(self, S):
        """Returns intersection point"""
        P0 = S[0]
        P1 = S[1]

        s = np.dot(self.N , self.P - P0)/ np.dot(self.N, P1 - P0)
        return P0 + s * (P1-P0)

    def intersection_plane(self, three_points_on_other_Plane):
        """Return: a line of a tuple of two points"""
        V0, V1, V2 = three_points_on_other_Plane
        P0 = self.intersection_line([V0, V1])
        P1 = self.intersection_line([V0, V2])
        return (P0, P1)

def planeIntersection(P1, P2):
    return P1.intersection_plane(P2)

def planeLineIntersection(P, L):
    return P.intersection_line(L)

def lineIsInPoly(rb, L):
    """L is in polygon rb if its two ends are in rb"""
    l = np.array(L).reshape(4)
    return cv2.pointPolygonTest(rb, tuple(l[0:2]), False) >= 0 and cv2.pointPolygonTest(rb, tuple(l[2:4]), False) >= 0

def segmentPolygonDistance(s, poly, samples=10, signed=False):
    """segment-polygon distance by averaging distance between polygon and points on segment"""
    s = np.array(s).reshape((-1,2))
    ps =  [s[0] * (1-t) + s[1] * t for t in np.linspace(0, 1., num=samples)]
    if signed:
        vs = [cv2.pointPolygonTest(poly, tuple(p), True) for p in ps]
    else:
        vs = np.fabs([cv2.pointPolygonTest(poly, tuple(p), True) for p in ps])
    return np.average(vs)


def distanceBetweenParallelLines(l, L):
    e = L[1]-L[0]
    d = e/np.linalg.norm(e)
    m = (l[0]+l[1])/2.0
    e2 = m - L[0]
    return np.linalg.norm(np.dot(e2, d)*d - e2)


def sortLinesAccordingToX(Ls, y=0.0):
    """Assume all lines are relatively vertical, sort them in order of left-to-right"""
    def get_x(a, b, y=0.0):
        t = (b[1]-y)/(b[1]-a[1])
        x = b[0] - t * (b[0]-a[0])
        return x

    xs = [(L, get_x(L[0], L[1], y)) for L in Ls]

    return [x[0] for x in sorted(xs, cmp=lambda x,y: np.sign(x[1]-y[1]).astype(np.int))]


def distPoint2Line(P,  L):
    """ get the distance of a point to a line

        Input:  a Point P and a Line L (in any dimension)
        Return: the shortest distance from P to L
    """
    if len(L) != 2:
        print "Error: Line has 2 end-points, plz check shape"
        return None

    v = L[1] - L[0]
    w = P - L[0]

    c1 = np.dot(w,v)
    c2 = np.dot(v,v)
    b = c1 / c2

    Pb = L[0] +  v * b
    return np.linalg.norm(P- Pb)


def perp2D(u, v):
    return u[0]*v[1] - v[0]*u[1]


class Primitive2D():
    def __init__(self, P0, P1):
        self.P0 = P0
        self.P1 = P1



def intersect2D_2Lines(S1, S2, SMALL_NUM=1e-6):
    """ intersect2D_2Segments(): find the 2D intersection of 2 finite segments

    Input:  two finite segments S1 and S2
    Output: *I0 = intersect point (when it exists)
            *I1 =  endpoint of intersect segment [I0,I1] (when it exists)
    Return: 0=disjoint (no intersect)
            1=intersect  in unique point I0
            2=overlap  in segment from I0 to I1
    """

    S1, S2 = Primitive2D(S1[0], S1[1]), Primitive2D(S2[0], S2[1])

    u = S1.P1 - S1.P0
    v = S2.P1 - S2.P0
    w = S1.P0 - S2.P0
    D = perp2D(u,v)

    if (np.fabs(D) < SMALL_NUM) :           # S1 and S2 are parallel
        return None

    # the segments are skew and may intersect in a point
    # get the intersect parameter for S1
    sI = perp2D(v,w) / D

    I0 = S1.P0 + sI * u                # compute S1 intersect point
    return I0



def intersect2D_2Segments(S1, S2):
    """ intersect2D_2Segments(): find the 2D intersection of 2 finite segments

    Input:  two finite segments S1 and S2
    Output: *I0 = intersect point (when it exists)
            *I1 =  endpoint of intersect segment [I0,I1] (when it exists)
    Return: 0=disjoint (no intersect)
            1=intersect  in unique point I0
            2=overlap  in segment from I0 to I1
    """
    u = S1.P1 - S1.P0
    v = S2.P1 - S2.P0
    w = S1.P0 - S2.P0
    D = perp2D(u,v)

    if (np.fabs(D) < SMALL_NUM):           # S1 and S2 are parallel
        if (perp2D(u,w) != 0 or perp2D(v,w) != 0)  :
            return []                    # they are NOT collinear

        # they are collinear or degenerate
        # check if they are degenerate  points
        du =np.dot(u,u)
        dv =np.dot(v,v)
        if du==0 and dv==0 :            # both segments are points
            if S1.P0 !=  S2.P0:         # they are distinct  points
                 return []
            I0 = S1.P0                 # they are the same point
            return [I0,]

        if du==0 :                     # S1 is a single point
            if  inSegment(S1.P0, S2) == 0:  # but is not in S2
                 return []
            I0 = S1.P0
            return [I0,]

        if dv==0 :                     # S2 a single point
            if  inSegment(S2.P0, S1) == 0:  # but is not in S1
                 return []
            I0 = S2.P0
            return [I0,]

        # they are collinear segments - get  overlap (or not)
        w2 = S1.P1 - S2.P0
        if v.x != 0 :
            t0 = w.x / v.x
            t1 = w2.x / v.x
        else :
            t0 = w.y / v.y
            t1 = w2.y / v.y
        if t0 > t1 :                   # must have t0 smaller than t1
            t0, t1 = t1, t0

        if t0 > 1 or t1 < 0 :
            return []

        t0 = 0 if t0<0 else t0               # clip to min 0
        t1 = 1 if t1>1 else t1               # clip to max 1
        if t0 == t1 :                  # intersect is a point
            I0 = S2.P0 +  t0 * v
            return [I0,]

        # they overlap in a valid subsegment
        I0 = S2.P0 + t0 * v
        I1 = S2.P0 + t1 * v
        return [I0, I1]

    # the segments are skew and may intersect in a point
    # get the intersect parameter for S1
    sI = perp2D(v,w) / D
    if sI < 0 or sI > 1:                # no intersect with S1
        return []

    # get the intersect parameter for S2
    tI = perp2D(u,w) / D
    if tI < 0 or tI > 1:                # no intersect with S2
        return []

    I0 = S1.P0 + sI * u                # compute S1 intersect point
    return [I0]


def inSegment(P, S):
    """inSegment(): determine if a point is inside a segment

        Input:  a point P, and a collinear segment S
        Return: 1 = P is inside S
                0 = P is  not inside S
    """
    if S.P0.x != S.P1.x:    # S is not  vertical
        if S.P0.x <= P.x and P.x <= S.P1.x:
            return 1
        if S.P0.x >= P.x and P.x >= S.P1.x:
            return 1
    else :    # S is vertical, so test y  coordinate
        if S.P0.y <= P.y and P.y <= S.P1.y:
            return 1
        if S.P0.y >= P.y and P.y >= S.P1.y:
            return 1
    return 0


def sampledDistBetweenSegments(L1, L2, n=10):
    d = [distPoint2Line(L1[0] * (1.0-r) + L1[0] * r, L2) for r in np.linspace(0.0, 1.0, n, True)] + [distPoint2Line(L2[0] * (1.0-r) + L2[0] * r, L1) for r in np.linspace(0.0, 1.0, n, True)]
    return np.average(d)


def lineAngleDifference(line, ruler):
    return abs(np.dot(line,ruler)/np.linalg.norm(line)/np.linalg.norm(ruler))


def lineDirection(line):
    L = np.array(line).reshape((2,2))
    return (L[1] - L[0])/np.linalg.norm(L[1]-L[0])
