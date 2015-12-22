from tempfile import NamedTemporaryFile
from svmlight_loader import load_svmlight_file as _load_svmlight_file
from joblib import Parallel, delayed

import numpy as np
import multiprocessing as mp
import os
import scipy.sparse as sp


_ONE_MB = (1 << 20)


def _find_offsets(haystack, needle, idx, acc):
    offs = -1
    while 1:
        offs = haystack.find(needle, offs+1)
        if offs == -1:
            break
        else:
            acc.append(idx + offs)


def _positions(fp, ch, bufsize):
    """Enumerate all the positions in the file where the character is ch

    """
    inds = []
    idx = 0
    while 1:
	buf = fp.read(bufsize)
	if not buf:
	    break
	_find_offsets(buf, ch, idx, inds)
	idx += len(buf)
    return inds


def _do_partition_file(fobj, partitions, bufsize):
    tempfiles = []
    for partition_start, partition_end in partitions:
        tempfile = NamedTemporaryFile()
        fobj.seek(partition_start)
        pos = partition_start
        while pos < partition_end:
            buf = fobj.read(min(bufsize, partition_end - pos))
            if not buf:
                raise IOError("file was misspecified")
            tempfile.write(buf)
            pos += len(buf)
        tempfiles.append(tempfile)
        tempfile.flush()
    return tempfiles


def _partition_file(fobj, partitions, bufsize=_ONE_MB):
    if partitions <= 0:
        raise ValueError("need >=1 partition")

    # This is always so janky
    fobj.seek(0, os.SEEK_END)
    filelen = fobj.tell()
    fobj.seek(0)

    newlines = _positions(fobj, "\n", bufsize)

    # reset the fobj
    fobj.seek(0)

    if not newlines:
        partitions = 1
    if partitions > len(newlines):
        partitions = len(newlines)
    if partitions == 1:
        return [fobj]

    # TODO: no attempt is made to load balance partitions
    # based on length. it is assumed that each line is relatively
    # uniform in length

    partition_size = len(newlines) / partitions
    assert partition_size > 0

    coalesced_newlines = []
    for idx in xrange(partitions):
        if idx + 1 == partitions:
            coalesced_newlines.append(newlines[-1])
        else:
            end = partition_size * (idx + 1)
            coalesced_newlines.append(newlines[end - 1])

    #print "newlines:", newlines
    #print "coalesced_newlines:", coalesced_newlines

    startends = []
    for idx, position in enumerate(coalesced_newlines):
        #print "idx={}, position={}".format(idx, position)
        if idx == 0:
            startends.append((0, position + 1))
        else:
            startends.append((startends[-1][1], position + 1))
    if startends[-1][1] < filelen:
        startends.append((startends[-1][1], filelen))

    #print "startends:", startends

    return _do_partition_file(fobj, startends, bufsize)


def _pad_out_csr_matrix(X, n_cols):
    assert X.shape[1] <= n_cols
    return sp.csr_matrix(
            (X.data, X.indices, X.indptr),
            shape=(X.shape[0], n_cols))


def load_svmlight_file(file_path, n_features=None, dtype=None,
                       buffer_mb=40, zero_based="auto", n_jobs=None):
    """


    """

    if n_jobs is None or n_jobs <= 0:
        # TODO(stephentu): check for Hyperthreading?
        n_jobs = mp.cpu_count()

    with open(file_path, 'r') as fp:
        tempfiles = _partition_file(fp, n_jobs, 40 * _ONE_MB)
        assert len(tempfiles) == n_jobs

    fn = delayed(_load_svmlight_file)
    pairs = Parallel(n_jobs=n_jobs, verbose=0)(
        fn(tempfile.name, n_features, dtype, buffer_mb, zero_based)
        for tempfile in tempfiles)
    n_cols = max(X.shape[1] for X, _ in pairs)
    Xlist = [_pad_out_csr_matrix(X, n_cols) for X, _ in pairs]
    Ylist = [Y for _, Y in pairs]
    X, Y = sp.vstack(Xlist), np.vstack(Ylist)
    return X, Y
