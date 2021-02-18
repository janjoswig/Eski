import pytest

from eski import atoms


class TestAtom:

    @pytest.mark.parametrize(
        "args,kwargs,expected",
        [
            (("C"), {}, ("C", "C", "C", "UNK", 0., 0.)),
            (("Ar", "AX"), {"mass": 40}, ("Ar", "AX", "Ar", "UNK", 40., 0.))
            ]
        )
    def test_create(self, args, kwargs, expected):
        a = atoms.Atom(*args, **kwargs)
        check = dict(
            zip(
                ["aname", "atype", "element", "residue", "mass", "charge"],
                expected
               )
            )
        for attr, value in check.items():
            assert getattr(a, attr) == value

        attr_strs = [f"{k}={v}" for k, v in check.items()]
        assert repr(a) == f"{a.__class__.__name__}({', '.join(attr_strs)})"
