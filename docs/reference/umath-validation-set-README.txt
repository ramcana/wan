---
category: reference
last_updated: '2025-09-13T18:40:09.024933'
original_path: fresh_env\Lib\site-packages\numpy\_core\tests\data\umath-validation-set-README.txt
tags:
- troubleshooting
title: Umath Validation Set Readme
---

Steps to validate transcendental functions:
1) Add a file 'umath-validation-set-<ufuncname>.txt', where ufuncname is name of
   the function in NumPy you want to validate
2) The file should contain 4 columns: dtype,input,expected output,ulperror
    a. dtype: one of np.float16, np.float32, np.float64
    b. input: floating point input to ufunc in hex. Example: 0x414570a4
       represents 12.340000152587890625
    c. expected output: floating point output for the corresponding input in hex.
       This should be computed using a high(er) precision library and then rounded to
       same format as the input.
    d. ulperror: expected maximum ulp error of the function. This
       should be same across all rows of the same dtype. Otherwise, the function is
       tested for the maximum ulp error among all entries of that dtype.
3) Add file umath-validation-set-<ufuncname>.txt to the test file test_umath_accuracy.py
   which will then validate your ufunc.
