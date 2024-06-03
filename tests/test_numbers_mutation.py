from mutations.numbers.integer_replacement import IntegerReplacementVisitor
from mutations.numbers.numeric_literals import IntegerBinTransformer, IntegerHexTransformer, IntegerOctTransformer
from utils import verify_visitor, verify_transformation


class TestIntegerReplacement:
    def test_integer_replacement_simple(self):
        code = """
        a = 30
        """
        expected = """
        a = 35 + -5
        """
        verify_visitor(IntegerReplacementVisitor, code, expected)

    def test_integer_replacement_nested(self):
        code = """
        def f():
            b = (32 * 43) - 2
        """
        expected = [
            """
            def f():
                b = 32 * 43 - (7 + -5)
            """,
            """
            def f():
                b = (37 + -5) * 43 - 2
            """,
            """
            def f():
                b = 32 * (48 + -5) - 2
            """,
        ]
        verify_visitor(IntegerReplacementVisitor, code, expected)


class TestNumericLiterals:
    def test_base_conversion_docstring(self):
        code = '''
        def extract_even(test_tuple):
            """
            Write a function to remove uneven elements in the nested mixed tuple.
            assert extract_even((4, 5, (7, 6, (0b10, 4)), 6, 8)) == (4, (6, (2, 4)), 6, 8),
            """
            pass
        '''
        verify_transformation(IntegerBinTransformer, code, [])

    def test_base_2_conversion(self):
        code = """
        print(4)
        def foo(a, b):
            if a + b < -100:
                return 2 * a
        """
        expected = [
            """
            print(0b100)
            def foo(a, b):
                if a + b < -100:
                    return 2 * a
            """,
            """
            print(4)
            def foo(a, b):
                if a + b < -0b1100100:
                    return 2 * a
            """,
            """
            print(4)
            def foo(a, b):
                if a + b < -100:
                    return 0b10 * a
            """,
        ]
        verify_transformation(IntegerBinTransformer, code, expected)

    def test_base_8_conversion(self):
        code = """
        def doWork(a):
            a = a + 100 - 2
            return -a
        """
        expected = [
            """
            def doWork(a):
                a = a + 0o144 - 2
                return -a
            """,
            """
            def doWork(a):
                a = a + 100 - 0o2
                return -a
            """,
        ]
        verify_transformation(IntegerOctTransformer, code, expected)

    def test_base_16_conversion(self):
        code = """
        a = 3 * 2
        b = -1000
        """
        expected = [
            """
            a = 0x3 * 2
            b = -1000
            """,
            """
            a = 3 * 0x2
            b = -1000
            """,
            """
            a = 3 * 2
            b = -0x3e8
            """
        ]
        verify_transformation(IntegerHexTransformer, code, expected)