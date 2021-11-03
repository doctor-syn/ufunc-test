#include <Python.h>
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"


// https://numpy.org/doc/stable/user/c-info.ufunc-tutorial.html

static PyMethodDef LogitMethods[] = {
    {NULL, NULL, 0, NULL}
};

static void double_logit(
    char **args,
    npy_intp const *dimensions,
    npy_intp const *strides,
    void *innerloopdata)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = strides[0], out_step = strides[1];

    // printf("in_step=%d\n", (int)in_step);
    // printf("out_step=%d\n", (int)out_step);

    if (in_step == sizeof(double) && out_step == sizeof(double))
    {
        double *src = (double *)in;
        double *dest = (double *)out;
        for (i = 0; i < n; i++)
        {
            double tmp = *src++;
            // *dest++ = log(tmp) - log(1.0-tmp);
            *dest++ = log(tmp / (1.0-tmp));
        }
    }
    else
    {
        for (i = 0; i < n; i++)
        {
            /*BEGIN main ufunc computation*/
            double tmp = *(double *)in;
            tmp /= 1 - tmp;
            *((double *)out) = log(tmp);
            /*END main ufunc computation*/

            in += in_step;
            out += out_step;
        }
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {double_logit};

/* These are the input and return dtypes of logit.*/
static char types[2] = {NPY_DOUBLE, NPY_DOUBLE};

static void *data[1] = {NULL};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npufunc",
    NULL,
    -1,
    LogitMethods,
    NULL,
    NULL,
    NULL,
    NULL};

PyMODINIT_FUNC PyInit_npufunc(void)
{
    PyObject *m, *logit, *d;
    m = PyModule_Create(&moduledef);
    if (!m)
    {
        return NULL;
    }

    import_array();
    import_umath();

    logit = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1,
                                    PyUFunc_None, "logit",
                                    "logit_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "logit", logit);
    Py_DECREF(logit);

    return m;
}
