import numpy as np

from libc.math cimport pow, log


ctypedef Py_ssize_t intp_t
ctypedef float float32_t
ctypedef double float64_t


def eval_mle_cython_w(float ldelta, float[:] S, double[:] UTw, double[:] UTy):
    cdef int n_samples = S.shape[0]

    cdef float delta = pow(10.0, ldelta)

    cdef int i
    cdef double[:] Sd = np.zeros(n_samples)
    cdef double yPy = 0.0
    cdef double yPw = 0.0
    cdef double wPw = 0.0

    cdef double ldet_H = 0.0
    cdef float mle

    for i in range(n_samples):
        Sd[i] = delta * S[i] + 1

        yPy += UTy[i] * UTy[i] / Sd[i]
        yPw += UTy[i] * UTw[i] / Sd[i]
        wPw += UTw[i] * UTw[i] / Sd[i]
    
    # y_P1_y
    yPy = yPy - yPw * yPw / wPw
    
    # ldet_H
    for i in range(n_samples):
        ldet_H += log(Sd[i])

    # mle
    mle = (n_samples/2) * log(n_samples/(2*np.pi)) - (n_samples/2) - (ldet_H/2) - (n_samples/2) * log(yPy)

    return mle


def eval_mle_cython_x(float ldelta, float[:] S, double[:] UTw, double[:] UTx, double[:] UTy):
    cdef int n_samples = S.shape[0]

    cdef float delta = pow(10.0, ldelta)

    cdef int i
    cdef double[:] Sd = np.zeros(n_samples)
    cdef double yPy = 0.0
    cdef double yPw = 0.0
    cdef double wPw = 0.0
    cdef double yPx = 0.0
    cdef double xPw = 0.0
    cdef double xPx = 0.0

    cdef double ldet_H = 0.0
    cdef float mle

    for i in range(n_samples):
        Sd[i] = delta * S[i] + 1

        yPy += UTy[i] * UTy[i] / Sd[i]
        yPw += UTy[i] * UTw[i] / Sd[i]
        wPw += UTw[i] * UTw[i] / Sd[i]

        yPx += UTy[i] * UTx[i] / Sd[i]
        xPw += UTx[i] * UTw[i] / Sd[i]

        xPx += UTx[i] * UTx[i] / Sd[i]

    # y_P1_y
    yPy = yPy - yPw * yPw / wPw

    # y_P2_y
    yPx = yPx - yPw * xPw / wPw
    xPx = xPx - xPw * xPw / wPw
    yPy = yPy - yPx * yPx / xPx

    # ldet_H
    for i in range(n_samples):
        ldet_H += log(Sd[i])

    # mle
    mle = (n_samples/2) * log(n_samples/(2*np.pi)) - (n_samples/2) - (ldet_H/2) - (n_samples/2) * log(yPy)

    return mle


def eval_mle_1st_cython_w(float ldelta, float[:] S, double[:] UTw, double[:] UTy):
    cdef int n_samples = S.shape[0]

    cdef float delta = pow(10.0, ldelta)

    cdef int i
    cdef double[:] Sd = np.zeros(n_samples)
    cdef double yPy = 0.0
    cdef double yPw = 0.0
    cdef double wPw = 0.0
    cdef double yPPy = 0.0
    cdef double yPPw = 0.0
    cdef double wPPw = 0.0

    cdef double trace_Hi = 0.0
    cdef double trace_HiG

    cdef double yPGPy
    cdef double mle_1st

    for i in range(n_samples):
        Sd[i] = delta * S[i] + 1

        yPy += UTy[i] * UTy[i] / Sd[i]
        yPw += UTy[i] * UTw[i] / Sd[i]
        wPw += UTw[i] * UTw[i] / Sd[i]

        yPPy += UTy[i] * UTy[i] / (Sd[i] * Sd[i])
        yPPw += UTy[i] * UTw[i] / (Sd[i] * Sd[i])
        wPPw += UTw[i] * UTw[i] / (Sd[i] * Sd[i])

        trace_Hi += 1 / Sd[i]

    # y_P1_y
    yPy = yPy - yPw * yPw / wPw

    # y_P1P1_y
    yPPy = yPPy + yPw * yPw * wPPw / (wPw * wPw) - 2 * yPw * yPPw / wPw

    # trace_HiG
    trace_HiG = (n_samples - trace_Hi) / delta

    # y_P1GP1_y
    yPGPy = (yPy - yPPy) / delta

    # mle 1st
    mle_1st = -trace_HiG/2 + (n_samples/2) * yPGPy/yPy

    return mle_1st


def eval_mle_1st_cython_x(float ldelta, float[:] S, double[:] UTw, double[:] UTx, double[:] UTy):
    cdef int n_samples = S.shape[0]

    cdef float delta = pow(10.0, ldelta)

    cdef int i
    cdef double[:] Sd = np.zeros(n_samples)
    cdef double yPy = 0.0
    cdef double yPw = 0.0
    cdef double wPw = 0.0
    cdef double yPx = 0.0
    cdef double xPw = 0.0
    cdef double xPx = 0.0

    cdef double yPPy = 0.0
    cdef double yPPw = 0.0
    cdef double wPPw = 0.0
    cdef double yPPx = 0.0
    cdef double xPPw = 0.0
    cdef double xPPx = 0.0

    cdef double trace_Hi = 0.0
    cdef double trace_HiG

    cdef double yPGPy
    cdef double mle_1st

    for i in range(n_samples):
        Sd[i] = delta * S[i] + 1

        yPy += UTy[i] * UTy[i] / Sd[i]
        yPw += UTy[i] * UTw[i] / Sd[i]
        wPw += UTw[i] * UTw[i] / Sd[i]

        yPx += UTy[i] * UTx[i] / Sd[i]
        xPw += UTx[i] * UTw[i] / Sd[i]

        xPx += UTx[i] * UTx[i] / Sd[i]

        yPPy += UTy[i] * UTy[i] / (Sd[i] * Sd[i])
        yPPw += UTy[i] * UTw[i] / (Sd[i] * Sd[i])
        wPPw += UTw[i] * UTw[i] / (Sd[i] * Sd[i])

        yPPx += UTy[i] * UTx[i] / (Sd[i] * Sd[i])
        xPPw += UTx[i] * UTw[i] / (Sd[i] * Sd[i])

        xPPx += UTx[i] * UTx[i] / (Sd[i] * Sd[i])

        trace_Hi += 1 / Sd[i]
    
    # y_P1_y
    yPy = yPy - yPw * yPw / wPw

    # y_P2_y
    yPx = yPx - yPw * xPw / wPw
    xPx = xPx - xPw * xPw / wPw
    yPy = yPy - yPx * yPx / xPx

    # y_P1P1_y
    yPPy = yPPy + yPw * yPw * wPPw / (wPw * wPw) - 2 * yPw * yPPw / wPw

    # y_P2P2_y
    yPPx = yPPx + yPw * xPw * wPPw / (wPw * wPw) - yPw * xPPw / wPw - xPw * yPPw / wPw
    xPPx = xPPx + xPw * xPw * wPPw / (wPw * wPw) - 2 * xPw * xPPw / wPw

    yPPy = yPPy + yPx * yPx * xPPx / (xPx * xPx) - 2 * yPx * yPPx / xPx

    # trace_HiG
    trace_HiG = (n_samples - trace_Hi) / delta

    # y_P1GP1_y
    yPGPy = (yPy - yPPy) / delta

    # mle 1st
    mle_1st = -trace_HiG/2 + (n_samples/2) * yPGPy/yPy

    return mle_1st


def eval_mle_2nd_cython_w(float ldelta, float[:] S, double[:] UTw, double[:] UTy):
    cdef int n_samples = S.shape[0]

    cdef float delta = pow(10.0, ldelta)

    cdef int i
    cdef double[:] Sd = np.zeros(n_samples)
    cdef double yPy = 0.0
    cdef double yPw = 0.0
    cdef double wPw = 0.0
    cdef double yPPy = 0.0
    cdef double yPPw = 0.0
    cdef double wPPw = 0.0
    cdef double yPPPy = 0.0
    cdef double yPPPw = 0.0
    cdef double wPPPw = 0.0

    cdef double trace_Hi = 0.0
    cdef double trace_HiHi = 0.0
    # cdef double trace_HiG
    cdef double trace_HiGHiG

    cdef double yPGPy
    cdef double yPGPGPy
    cdef double mle_2nd

    for i in range(n_samples):
        Sd[i] = delta * S[i] + 1

        yPy += UTy[i] * UTy[i] / Sd[i]
        yPw += UTy[i] * UTw[i] / Sd[i]
        wPw += UTw[i] * UTw[i] / Sd[i]

        yPPy += UTy[i] * UTy[i] / (Sd[i] * Sd[i])
        yPPw += UTy[i] * UTw[i] / (Sd[i] * Sd[i])
        wPPw += UTw[i] * UTw[i] / (Sd[i] * Sd[i])

        yPPPy += UTy[i] * UTy[i] / (Sd[i] * Sd[i] * Sd[i])
        yPPPw += UTy[i] * UTw[i] / (Sd[i] * Sd[i] * Sd[i])
        wPPPw += UTw[i] * UTw[i] / (Sd[i] * Sd[i] * Sd[i])

        trace_Hi += 1 / Sd[i]
        trace_HiHi += 1 / (Sd[i] * Sd[i])

    # y_P1_y
    yPy = yPy - yPw * yPw / wPw

    # y_P1P1_y
    yPPy = yPPy + yPw * yPw * wPPw / (wPw * wPw) - 2 * yPw * yPPw / wPw

    # y_P1P1P1_y
    yPPPy = (yPPPy \
             - yPw * yPw * wPPw / (wPw * wPw * wPw) \
             - 2 * yPw * yPPPw / wPw \
             - yPPw * yPPw / wPw \
             + 2 * yPw * yPPw * wPPw / (wPw * wPw) \
             + yPw * yPw * wPPPw / (wPw * wPw))



    # trace_HiG
    # trace_HiG = (n_samples - trace_Hi) / delta

    # trace_HiGHiG
    trace_HiGHiG = (n_samples + trace_HiHi - 2 * trace_Hi) / (delta * delta)

    # y_P1GP1_y
    yPGPy = (yPy - yPPy) / delta

    # y_P1GP1GP1_y
    yPGPGPy = (yPy + yPPPy - 2 * yPPy) / (delta * delta)

    # mle 1st
    mle_2nd = trace_HiGHiG/2 - (n_samples/2) * (yPGPGPy * yPy - yPGPy * yPGPy) / (yPy * yPy)

    return mle_2nd


def eval_mle_2nd_cython_x(float ldelta, float[:] S, double[:] UTw, double[:] UTx, double[:] UTy):
    cdef int n_samples = S.shape[0]

    cdef float delta = pow(10.0, ldelta)

    cdef int i
    cdef double[:] Sd = np.zeros(n_samples)
    cdef double yPy = 0.0
    cdef double yPw = 0.0
    cdef double wPw = 0.0
    cdef double yPx = 0.0
    cdef double xPw = 0.0
    cdef double xPx = 0.0

    cdef double yPPy = 0.0
    cdef double yPPw = 0.0
    cdef double wPPw = 0.0
    cdef double yPPx = 0.0
    cdef double xPPw = 0.0
    cdef double xPPx = 0.0

    cdef double yPPPy = 0.0
    cdef double yPPPw = 0.0
    cdef double wPPPw = 0.0
    cdef double yPPPx = 0.0
    cdef double xPPPw = 0.0
    cdef double xPPPx = 0.0

    cdef double trace_Hi = 0.0
    cdef double trace_HiHi = 0.0
    cdef double trace_HiGHiG

    cdef double yPGPy
    cdef double yPGPGPy
    cdef double mle_2nd

    for i in range(n_samples):
        Sd[i] = delta * S[i] + 1

        yPy += UTy[i] * UTy[i] / Sd[i]
        yPw += UTy[i] * UTw[i] / Sd[i]
        wPw += UTw[i] * UTw[i] / Sd[i]

        yPx += UTy[i] * UTx[i] / Sd[i]
        xPw += UTx[i] * UTw[i] / Sd[i]

        xPx += UTx[i] * UTx[i] / Sd[i]

        yPPy += UTy[i] * UTy[i] / (Sd[i] * Sd[i])
        yPPw += UTy[i] * UTw[i] / (Sd[i] * Sd[i])
        wPPw += UTw[i] * UTw[i] / (Sd[i] * Sd[i])

        yPPx += UTy[i] * UTx[i] / (Sd[i] * Sd[i])
        xPPw += UTx[i] * UTw[i] / (Sd[i] * Sd[i])

        xPPx += UTx[i] * UTx[i] / (Sd[i] * Sd[i])

        yPPPy += UTy[i] * UTy[i] / (Sd[i] * Sd[i] * Sd[i])
        yPPPw += UTy[i] * UTw[i] / (Sd[i] * Sd[i] * Sd[i])
        wPPPw += UTw[i] * UTw[i] / (Sd[i] * Sd[i] * Sd[i])

        yPPPx += UTy[i] * UTx[i] / (Sd[i] * Sd[i] * Sd[i])
        xPPPw += UTx[i] * UTw[i] / (Sd[i] * Sd[i] * Sd[i])

        xPPPx += UTx[i] * UTx[i] / (Sd[i] * Sd[i] * Sd[i])


        trace_Hi += 1 / Sd[i]
        trace_HiHi += 1 / (Sd[i] * Sd[i])
    
    # y_P1_y
    yPy = yPy - yPw * yPw / wPw

    # y_P2_y
    yPx = yPx - yPw * xPw / wPw
    xPx = xPx - xPw * xPw / wPw
    yPy = yPy - yPx * yPx / xPx

    # y_P1P1_y
    yPPy = yPPy + yPw * yPw * wPPw / (wPw * wPw) - 2 * yPw * yPPw / wPw

    # y_P2P2_y
    yPPx = yPPx + yPw * xPw * wPPw / (wPw * wPw) - yPw * xPPw / wPw - xPw * yPPw / wPw
    xPPx = xPPx + xPw * xPw * wPPw / (wPw * wPw) - 2 * xPw * xPPw / wPw

    yPPy = yPPy + yPx * yPx * xPPx / (xPx * xPx) - 2 * yPx * yPPx / xPx

    # y_P1P1P1_y
    yPPPy = (yPPPy \
             - yPw * yPw * wPPw / (wPw * wPw * wPw) \
             - 2 * yPw * yPPPw / wPw \
             - yPPw * yPPw / wPw \
             + 2 * yPw * yPPw * wPPw / (wPw * wPw) \
             + yPw * yPw * wPPPw / (wPw * wPw))
    
    # y_P2P2P2_y
    yPPPx = (yPPPx \
             - yPw * xPw * wPPw / (wPw * wPw * wPw) \
             - yPw * xPPPw / wPw \
             - xPw * yPPPw / wPw \
             - yPPw * xPPw / wPw \
             + yPw * xPPw * wPPw / (wPw * wPw) \
             + xPw * yPPw * wPPw / (wPw * wPw) \
             + yPw * xPw * wPPPw / (wPw * wPw))
    
    xPPPx = (xPPPx \
             - xPw * xPw * wPPw / (wPw * wPw * wPw) \
             - 2 * xPw * xPPPw / wPw \
             - xPPw * xPPw / wPw \
             + 2 * xPw * xPPw * wPPw / (wPw * wPw) \
             + xPw * xPw * wPPPw / (wPw * wPw))
    
    yPPPy = (yPPPy \
             - yPx * yPx * xPPx / (xPx * xPx * xPx) \
             - 2 * yPx * yPPPx / xPx \
             - yPPx * yPPx / xPx \
             + 2 * yPx * yPPx * xPPx / (xPx * xPx) \
             + yPx * yPx * xPPPx / (xPx * xPx))
    

    # trace_HiGHiG
    trace_HiGHiG = (n_samples + trace_HiHi - 2 * trace_Hi) / (delta * delta)

    # y_P2GP2_y
    yPGPy = (yPy - yPPy) / delta

    # y_P2GP2GP2_y
    yPGPGPy = (yPy + yPPPy - 2 * yPPy) / (delta * delta)

    # mle 2nd
    mle_2nd = trace_HiGHiG/2 - (n_samples/2) * (yPGPGPy * yPy - yPGPy * yPGPy) / (yPy * yPy)

    return mle_2nd
