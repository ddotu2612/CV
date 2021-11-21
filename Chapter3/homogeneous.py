from numpy import *
from numpy import vstack, ones, diag, dot, zeros, linalg, concatenate, sqrt

# Chuẩn hoá và chuyển tới homogeneous coordinates
def normalize(points):
    """ Chuẩn hoá tất cả các điểm để hàng cuối cùng bằng 1 """
    for row in points:
        row /= points[-1]
    return points

def make_homog(points):
    """ Chuyển tập n điểm 2 chiều theo chiều dọc 
    và thêm hàng toàn 1 vào cuối thành ma trận 3 * n"""
    # Nối vào đuôi của ma trận points 1 hàng toàn 1
    return vstack((points, ones((1, points.shape[1]))))

def H_from_points(fp, tp):
    """ Find homography H, such that fp is mapped to tp
    using the linear DLT method. Points are conditioned
    automatically. """

    if fp.shape != tp.shape:
        raise RuntimeError('Number of points do not match')
    
    # condition points (important for numerical reasons)
    # --from points--
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = dot(C1,fp)

    # --to points--
    m = mean(tp[:2], axis=1)
    maxstd = max(std(tp[:2], axis=1)) + 1e-9
    C2 = diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp = dot(C2,tp)

    # create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = fp.shape[1]
    A = zeros((2*nbr_correspondences,9))
    for i in range(nbr_correspondences):
        A[2*i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0,
        tp[0][i]*fp[0][i], tp[0][i]*fp[1][i], tp[0][i]]
        A[2*i+1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1,
        tp[1][i]*fp[0][i], tp[1][i]*fp[1][i], tp[1][i]]
    
    U, S, V = linalg.svd(A)
    H = V[8].reshape((3, 3))

    # decondition
    H = dot(linalg.inv(C2), dot(H, C1))

    # normalize and return
    return H / H[2, 2]

# Affine Transformations
def Haffine_from_points(fp, tp):
    """ Find H, affine transformation, such that
    tp is affine transf of fp. """

    if fp.shape != tp.shape:
        raise RuntimeError('Number of points do not match')
    
    # condition points
    # --from points--
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp_cond = dot(C1, fp)

    # --to points--
    m = mean(tp[:2], axis=1)
    C2 = C1.copy() #must use same scaling for both point sets
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp_cond = dot(C2, tp)

    # conditioned points have mean zero, so translation is zero
    A = concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
    U, S, V = linalg.svd(A.T)

    # create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = concatenate((dot(C,linalg.pinv(B)),zeros((2,1))), axis=1)
    H = vstack((tmp2,[0,0,1]))

    # decondition
    H = dot(linalg.inv(C2),dot(H,C1))

    return H / H[2,2]

class RansacModel(object):
    """ Class for testing homography fit with ransac.py from
    http://www.scipy.org/Cookbook/RANSAC"""

    def __init__(self, debug=False):
        self.debug = debug
    
    def fit(self, data):
        """ Fit homography to four selected correspondences. """
        """ Fit đồng nhất cho 4 điểm tương ứng được chọn"""
        # Chuyển vị để H_from_points()
        data = data.T

        # from points
        fp = data[:3,:4]
        # target points
        tp = data[3:, :4]

        # fit homography and return
        return H_from_points(fp, tp)

    def get_error(self, data, H):
        """ Apply homography to all correspondences,
        return error for each transformed point. """
        """ Áp dụng phép đồng nhất tới tất cả các điểm tương ứng và 
        trả ra lỗi tới mỗi điểm dịch chuyển """

        data = data.T

        # from points
        fp = data[:3]
        # target points
        tp = data[3:]

        # transform fp
        fp_transformed = dot(H, fp)

        # normalize hom. coordinates
        for i in range(3):
            fp_transformed[i] /= fp_transformed[2]
        
        # Lỗi trên mỗi điểm
        return sqrt( sum((tp-fp_transformed)**2,axis=0) )

def H_from_ransac(fp, tp, model, maxiter=1000, match_theshold=10):
    """ Robust estimation of homography H from point
    correspondences using RANSAC
    Ước lượng phép đồng nhất H từ các điểm correspondences sử dụng Ransac
    input: fp,tp (3*n arrays) points in hom. coordinates. """
    import ransac

    # Nhóm các điểm tương ứng
    data = vstack((fp,tp))

    # compute H and return
    H, ransac_data = ransac.ransac(data.T, model, 4, maxiter, match_theshold, 10, return_all=True)

    return H, ransac_data['inliers']

# Áp dụng RANSAC tới các điểm





    



    