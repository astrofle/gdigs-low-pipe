"""
"""

from gdigs_low_pipe.utils import get_vegas_sdfits_files


def test_get_vegas_sdfits_files():
    """Test """

    path = "testdata"
    files = get_vegas_sdfits_files(path)
    flist = list(files)
    assert len(flist) == 2
    assert flist[0].name == "test.raw.vegas.A.fits"
    assert flist[1].name == "test.raw.vegas.B.fits"
