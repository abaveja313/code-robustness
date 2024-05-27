from utils import verify_visitor

from mutations.booleans.boolean_demorgans import BooleanDemorgansVisitor
from mutations.booleans.boolean_inversion import FirstInversionVisitor, SecondInversionVisitor
from mutations.booleans.expand_booleans import ExpandBooleansVisitor


class TestBooleanDemorgans:
    def test_boolean_demorgans_all_cases(self):
        code = """
        a = not True or not False
        if not 3 < 2 and not 4 > 5:
            pass
        c = not (a is b or a is not b)
        return not (a+c % 6 == 5 and a is not True)
        """
        expected = [
            """
            a = not (True and False)
            if not 3 < 2 and (not 4 > 5):
                pass
            c = not (a is b or a is not b)
            return not (a + c % 6 == 5 and a is not True)
            """,
            """
            a = not True or not False
            if not (3 < 2 or 4 > 5):
                pass
            c = not (a is b or a is not b)
            return not (a + c % 6 == 5 and a is not True)
            """,
            """
            a = not True or not False
            if not 3 < 2 and (not 4 > 5):
                pass
            c = not a is b and (not a is not b)
            return not (a + c % 6 == 5 and a is not True)
            """,
            """
            a = not True or not False
            if not 3 < 2 and (not 4 > 5):
                pass
            c = not (a is b or a is not b)
            return not a + c % 6 == 5 or not a is not True
            """,
        ]
        verify_visitor(BooleanDemorgansVisitor, code, expected)

    def test_boolean_demorgans_compound(self):
        # we don't handle nested
        code = """
        while not (a and b) or not (not c or not d):
            pass
        """
        expected = [
            """
            while not ((a and b) and (not (c and d))):
                pass
            """
        ]
        verify_visitor(BooleanDemorgansVisitor, code, expected)

    def test_boolean_demorgans_positive_compound(self):
        code = """
        return a and b
        """
        expected = []
        verify_visitor(BooleanDemorgansVisitor, code, expected)


class TestFirstBooleanInversion:
    def test_boolean_inversion_simple_1(self):
        code = """
        if foo:
            pass
        """
        expected = """
        if not foo:
            pass
        """
        verify_visitor(FirstInversionVisitor, code, expected)

    def test_boolean_inversion_simple_2(self):
        code = """
        if not foo is not 5:
            pass
        """
        expected = """
        if foo is not 5:
            pass
        """
        verify_visitor(FirstInversionVisitor, code, expected)

    def test_boolean_inversion_simple_3(self):
        code = """
        while c - 6 == 5:
            pass
        """
        expected = """
        while c - 6 != 5:
            pass
        """
        verify_visitor(FirstInversionVisitor, code, expected)

    def test_boolean_inversion_complex_1(self):
        code = """
        if not (a < 5 and b > 6) or not not c == 5:
            pass
        """
        expected = """
        if (a < 5 and b > 6) and c != 5:
            pass
        """
        verify_visitor(FirstInversionVisitor, code, expected)

    def test_boolean_inversion_complex_2(self):
        code = """
        if (not (a < 5 and not b > 6)) == False:
            pass
        """
        expected = """
        if (a >= 5 or b > 6) != False:
            pass
        """
        verify_visitor(FirstInversionVisitor, code, expected)

    def test_boolean_inversion_complex_3(self):
        code = """
        if  (a < 5 and (b > 6 or c == 5)):
            pass
        """
        expected = """
        if a >= 5 or (b <= 6 and c != 5):
            pass
        """
        verify_visitor(FirstInversionVisitor, code, expected)

    def test_boolean_inversion_demorgans(self):
        code = """
        def foo():
            if a and b:
                pass
        """
        expected = """
        def foo():
            if not a or not b:
                pass
        """
        verify_visitor(FirstInversionVisitor, code, expected)


class TestSecondBooleanInversion:
    def test_second_boolean_inversion(self):
        code = """
        def po():
            if not a and not b:
                pass
        """
        expected = """
        def po():
            if not (a or b):
                pass
        """
        verify_visitor(SecondInversionVisitor, code, expected)


class TestExpandBooleans:
    def test_expand_booleans_simple(self):
        code = """
        result = not m
        """

        expected = """
        result = not m and True
        """
        verify_visitor(ExpandBooleansVisitor, code, expected)

    def test_expand_booleans_complex(self):
        code = """
        result = not m and not (a or b)
        """

        expected = [
            """
            result = (not m and True) and (not (a or b))
            """,
            """
            result = not m and (not (a or b) and True)
            """
        ]
        verify_visitor(ExpandBooleansVisitor, code, expected)
