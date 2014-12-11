SciPy's wav file reader does not support 24 bit sample widths. This has come up
many times on stackoverflow and on the scipy mailing list. In this gist, the
function `readwav` supports reading 24 bit files. It uses the `wave` module in
Python's standard library. It does not support compressed wav files.

Note: I've checked that it works on an assortment of wav files, but I won't say
that it has been *thoroughly* tested.  The module also defines the function
`writewav24` for writing 24 bit wav files from a numpy array.
