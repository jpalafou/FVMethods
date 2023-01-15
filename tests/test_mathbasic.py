from util.mathbasic import fact, gcf, lcm


def test_fact():
    """
    test the fact function
    """
    assert fact(0) == 1
    assert fact(1) == 1
    assert fact(10) == 3628800


def test_gcf():
    """
    test greatest the gcf function
    """
    assert gcf([15, 30, 60]) == 15
    assert gcf([2, -4, 8, -16, 32, -64, 128]) == 2
    assert gcf([1, 1]) == 1


def test_lcm():
    """
    test the lcm function
    """
    assert lcm(3, 2) == 6
    assert lcm(100, -10) == -100
    assert lcm(1, 1) == 1
