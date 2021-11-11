#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"




#include<math.h>

typedef double f64;
typedef long long i64;
typedef unsigned long long u64;
typedef long long bool;

#define REP(X) {X, X, X, X, X, X, X, X}
#define REINTERP(from, F, T) union { F f; T t; } u; u.f = from; return u.t;

inline f64 f64_mul_add(f64 a, f64 b, f64 c) {
    return a * b + c;
}

inline f64 f64_select(bool a, f64 b, f64 c) {
    return a ? b : c;
}

inline f64 f64_round(f64 a) {
    return round(a);
}

inline f64 f64_f(double f) {
    return (f64)f;
}

inline u64 f64_mkuty(long long v) {
    return (u64)v;
}

inline f64 f64_mkfty(long long v) {
    REINTERP(v, long long, double)
}

inline u64 f64_reinterpret_fty_uty(f64 f) {
    REINTERP(f, f64, u64)
}

inline f64 f64_reinterpret_uty_fty(u64 f) {
    REINTERP(f, u64, f64)
}

const f64 PI = M_PI;
const f64 LOG2_E = M_LOG2E;
const f64 LOG2_10 = M_LN10 / M_LN2;
const f64 MIN_POSITIVE = 2.2250738585072014E-308;
f64 f64_log2(f64 arg) {
  u64 arg_bits = f64_reinterpret_fty_uty(arg);
  u64 exponent = (arg_bits >> f64_mkuty(52ull)) - f64_mkuty(1023ull);
  f64 x = f64_reinterpret_uty_fty((arg_bits & f64_mkuty(4503599627370496ull - 1ull)) | f64_mkuty(4607182418800017408ull)) - f64_f(1.5);
  f64 y = f64_mul_add(f64_mul_add(f64_mul_add(f64_mul_add(f64_mul_add(f64_mul_add(f64_mul_add(f64_mul_add(f64_mul_add(f64_mul_add(f64_mul_add(f64_mul_add(f64_mul_add(f64_mul_add(f64_mul_add(f64_mul_add(f64_mul_add(f64_mul_add((-f64_mkfty(4546374278107678680ull)), x, f64_mkfty(4549472900654298694ull)), x, -f64_mkfty(4548163352978971471ull)), x, f64_mkfty(4550979242292870686ull)), x, -f64_mkfty(4555268298898397793ull)), x, f64_mkfty(4558493969767053974ull)), x, -f64_mkfty(4561674950935626047ull)), x, f64_mkfty(4564626928674686496ull)), x, -f64_mkfty(4567915592780460350ull)), x, f64_mkfty(4571457650865613616ull)), x, -f64_mkfty(4574762503947417879ull)), x, f64_mkfty(4578107178233634012ull)), x, -f64_mkfty(4581741589614488110ull)), x, f64_mkfty(4585636752412570626ull)), x, -f64_mkfty(4589798106271600139ull)), x, f64_mkfty(4594301705899034598ull)), x, -f64_mkfty(4599447016227734533ull)), x, f64_mkfty(4606838314010019088ull)), x, f64_mkfty(4603444093345823441ull));
  return y + ((f64)exponent);
}

f64 f64_ln(f64 arg) {
  return f64_log2(arg) * f64_f(1.0 / LOG2_E);
}


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

    // for (int i = 20; i != 100; ++i ) {
    //     f64 x = (double)i / 10.0;
    //     f64 y1 = f64_ln(x);
    //     f64 y2 = log(x);
    //     printf("x=%26.22f ln(x)=%26.22f lib ln(x)=%26.22f %26.22f\n", x, y1, y2, log(fabs(y1 - y2))/M_LN2);
    // }

    // printf("in_step=%d\n", (int)in_step);
    // printf("out_step=%d\n", (int)out_step);

    if (in_step == sizeof(double) && out_step == sizeof(double))
    {
        double *src = (double *)in;
        double *dest = (double *)out;
        for (i = 0; i < n; i++)
        {
            double tmp = *src++;
            *dest++ = f64_ln(tmp / (1.0-tmp));
        }
    }
    else
    {
        // for (i = 0; i < n; i++)
        // {
        //     /*BEGIN main ufunc computation*/
        //     double tmp = *(double *)in;
        //     tmp /= 1 - tmp;
        //     *((double *)out) = log(tmp);
        //     /*END main ufunc computation*/

        //     in += in_step;
        //     out += out_step;
        // }
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
