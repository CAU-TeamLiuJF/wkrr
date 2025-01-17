import numpy as np

from libc.math cimport exp, log


ctypedef Py_ssize_t intp_t
ctypedef float float32_t
ctypedef double float64_t


def eval_mle_cython_w(float32_t ldelta, float32_t[:] S, float64_t[:] UTw, float64_t[:] UTy):
    cdef intp_t n_samples = S.shape[0]

    cdef float32_t delta = exp(ldelta)

    cdef intp_t i
    cdef float64_t[:] Sd = np.zeros(n_samples)
    cdef float64_t yPy = 0.0
    cdef float64_t yPw = 0.0
    cdef float64_t wPw = 0.0

    cdef float64_t ldet_H = 0.0
    cdef float32_t mle

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


def eval_mle_cython_x(float32_t ldelta, float32_t[:] S, float64_t[:] UTw, float64_t[:] UTx, float64_t[:] UTy):
    cdef intp_t n_samples = S.shape[0]

    cdef float32_t delta = exp(ldelta)

    cdef intp_t i
    cdef float64_t[:] Sd = np.zeros(n_samples)
    cdef float64_t yPy = 0.0
    cdef float64_t yPw = 0.0
    cdef float64_t wPw = 0.0
    cdef float64_t yPx = 0.0
    cdef float64_t xPw = 0.0
    cdef float64_t xPx = 0.0

    cdef float64_t ldet_H = 0.0
    cdef float32_t mle

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


def eval_mle_1st_cython_w(float32_t ldelta, float32_t[:] S, float64_t[:] UTw, float64_t[:] UTy):
    cdef intp_t n_samples = S.shape[0]

    cdef float32_t delta = exp(ldelta)

    cdef intp_t i
    cdef float64_t[:] Sd = np.zeros(n_samples)
    cdef float64_t yPy = 0.0
    cdef float64_t yPw = 0.0
    cdef float64_t wPw = 0.0
    cdef float64_t yPPy = 0.0
    cdef float64_t yPPw = 0.0
    cdef float64_t wPPw = 0.0

    cdef float64_t trace_Hi = 0.0
    cdef float64_t trace_HiG

    cdef float64_t yPGPy
    cdef float64_t mle_1st

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


def eval_mle_1st_cython_x(float32_t ldelta, float32_t[:] S, float64_t[:] UTw, float64_t[:] UTx, float64_t[:] UTy):
    cdef intp_t n_samples = S.shape[0]

    cdef float32_t delta = exp(ldelta)

    cdef intp_t i
    cdef float64_t[:] Sd = np.zeros(n_samples)
    cdef float64_t yPy = 0.0
    cdef float64_t yPw = 0.0
    cdef float64_t wPw = 0.0
    cdef float64_t yPx = 0.0
    cdef float64_t xPw = 0.0
    cdef float64_t xPx = 0.0

    cdef float64_t yPPy = 0.0
    cdef float64_t yPPw = 0.0
    cdef float64_t wPPw = 0.0
    cdef float64_t yPPx = 0.0
    cdef float64_t xPPw = 0.0
    cdef float64_t xPPx = 0.0

    cdef float64_t trace_Hi = 0.0
    cdef float64_t trace_HiG

    cdef float64_t yPGPy
    cdef float64_t mle_1st

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

