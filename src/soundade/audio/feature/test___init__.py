from unittest import TestCase

from soundade.audio.feature import Feature


class TestFeature(TestCase):

    def test_Feature_init_basic(self):
        f = Feature('test', lambda a: a)

        assert 'test' not in f.kwargs
        assert 'one' not in f.kwargs

    def test_Feature_init_kwargs(self):
        f = Feature('test', lambda a: a, test=0, one=1)

        assert 'test' in f.kwargs
        assert 'one' in f.kwargs

    def test_Feature_init_star(self):
        f = Feature('test', lambda a: a, **{'test': 0, 'one': 1})

        assert 'test' in f.kwargs
        assert 'one' in f.kwargs

    def test_compute_star(self):
        f = Feature('test', lambda y, **kwargs: kwargs)
        kwa = f.compute(1, **{'test': 0, 'one': 1})

        assert 'test' in kwa
        assert 'one' in kwa


    def test_compute_kwargs(self):
        f = Feature('test', lambda y, **kwargs: kwargs)
        kwa = f.compute(1, test=0, one=1)

        assert 'test' in kwa
        assert 'one' in kwa

    def test_compute_star_override(self):
        f = Feature('test', lambda y, **kwargs: kwargs, **{'test': 0, 'one': 1})
        kwa = f.compute(1, **{'test': 10, 'one': 1, 'two': 2})

        assert 'test' in kwa
        assert 'one' in kwa
        assert 'two' in kwa

        assert kwa['test'] == 10
        assert kwa['one'] == 1
        assert kwa['two'] == 2

    def test_compute_kwargs_override(self):
        f = Feature('test', lambda y, **kwargs: kwargs, test=0, one=1)
        kwa = f.compute(1, test=10, one=1, two=2)

        assert 'test' in kwa
        assert 'one' in kwa
        assert 'two' in kwa

        assert kwa['test'] == 10
        assert kwa['one'] == 1
        assert kwa['two'] == 2
