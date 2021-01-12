import numpy as np
import cv2


class Camera:
    def __init__(self, camera_params):
        """
        :param fl:
        :param u:
        :param v:
        :param cc:
        :param rod_rot:
        """
        u, v, fl = camera_params[0:3]
        rod_rot = camera_params[3:6]
        cc = camera_params[6:9]

        self.calibration_matrix_K = self._set_calibration(fl, u, v)
        self.camera_center = self._set_camera_center(cc)
        self.rotation = self._set_rotation(rod_rot)
        self.projection_matrix_P = self._compute_matrix()

    @staticmethod
    def _set_calibration(fl, u, v):
        return np.asarray([[fl, 0, u],
                            [0, fl, v],
                            [0, 0, 1]])

    @staticmethod
    def _set_camera_center(cc):
        assert cc.shape[0] == 3
        camera_center = np.zeros(3)
        camera_center[0] = cc[0]
        camera_center[1] = cc[1]
        camera_center[2] = cc[2]
        return camera_center

    @staticmethod
    def _set_rotation(rod_rot):
        """
        :param rod_rot: Rodrigues vector
        :return:
        """
        assert rod_rot.shape[0] == 3
        rotation = np.zeros(3)
        rotation[0] = rod_rot[0]
        rotation[1] = rod_rot[1]
        rotation[2] = rod_rot[2]
        return rotation

    def get_homography(self):
        """
        homography matrix from the projection matrix
        :return:
        """
        h = self.projection_matrix_P[:, [0, 1, 3]]
        return h

    def _compute_matrix(self):
        P = np.zeros((3, 4))
        for i in range(3):
            P[i][i] = 1.0

        for i in range(3):
            P[i][3] = -self.camera_center[i]

        r, _ = cv2.Rodrigues(self.rotation)
        return self.calibration_matrix_K @ r @ P

    def project_point_on_frame(self, x, y, w=1.0):
        p = np.zeros(3)
        p[0], p[1], p[2] = x, y, w
        homography_matrix = self.get_homography()
        q = homography_matrix @ p

        assert q[2] != 0.0
        projected_x = np.rint(q[0] / q[2]).astype(np.int)
        projected_y = np.rint(q[1] / q[2]).astype(np.int)
        return projected_x, projected_y

    def project_point_on_topview(self, x, y, w=1.0):
        p = np.zeros(3)
        p[0], p[1], p[2] = x, y, w
        H = self.get_homography()

        # flip template in y direction
        m1 = np.asarray([[1, 0, 0],
                         [0, -1, 68],
                         [0, 0, 1]])
        homography_matrix = H @ m1
        inverted_homography_matrix = np.linalg.inv(homography_matrix)
        q = inverted_homography_matrix @ p

        assert q[2] != 0.0
        projected_x = np.rint(q[0]/q[2]).astype(np.int)
        projected_y = np.rint(q[1]/q[2]).astype(np.int)
        return projected_x, projected_y