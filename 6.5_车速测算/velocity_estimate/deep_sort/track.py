# vim: expandtab:ts=4:sw=4
import numpy as np
from collections import deque

def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age, 
                 pre_class=None, feature=None):
        self.mean = mean
        self.last_mean = mean
        self.covariance = covariance
        self.direction = np.array([0,0,0,0])
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.speed = 0.
        self.freeze_speed = False
        self.speed_deque = deque(maxlen=4)
        '''
        self.cur_center = mean[:2].copy()
        self.cur_vel = [0,0]
        self.cur_accl = [0,0]
        self.accl_deque = deque(maxlen=5)
        self.accl_deque.append(self.cur_accl)
        '''
        self.state = TrackState.Tentative
        self.pre_class = pre_class
        self.class_vote = {}
        self.class_vote[pre_class] = 1
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret
    def center(self):
        point = self.mean[:2].copy()
        return tuple([int(x) for x in point])

    def last_center(self):
        point = self.last_mean[:2].copy()
        return tuple([int(x) for x in point])

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        #avg_accl = np.mean(self.accl_deque,axis=0)
        self.last_mean = self.mean
        self.mean, self.covariance = kf.predict(self.mean, self.covariance, self.direction)#, avg_accl)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection, line=[(0,0),(0,0)]):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance, self.direction = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        if detection.confidence >= 0.9:
            if self.class_vote.get(detection.pre_class):
                self.class_vote[detection.pre_class] +=1
            else:
                self.class_vote[detection.pre_class] = 1
        self.pre_class = max(self.class_vote,key=self.class_vote.get)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        P0 = self.last_mean[:2].copy()
        P1 = self.mean[:2].copy()
        if not self.freeze_speed:
            estm_speed = kf.estm_speed(P0,P1)
            self.speed_deque.append(estm_speed)
            self.speed = np.mean(self.speed_deque)
        '''new_center = detection.to_xyah()[:2].copy()
        cur_vel = new_center - self.cur_center
        self.cur_accl = cur_vel - self.cur_vel
        self.cur_vel = cur_vel
        self.cur_center = new_center
        self.accl_deque.append(self.cur_accl)
        '''
        
        if intersect(P0,P1,line[0],line[1]):
            self.freeze_speed = True
            if P1[1] > P0[1]:
                return 1
            else:
                return -1
        else:
            return 0

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
