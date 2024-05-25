import textwrap

from utils import verify_visitor

from mutations.arrays.length_to_generator import LenToGeneratorVisitor
from mutations.arrays.list_initializer_unpack import ListInitializerUnpackVisitor
from mutations.arrays.nested_array_initializer import NestedArrayInitializerVisitor
from mutations.arrays.reverse_range import ReverseRangeVisitor
from mutations.arrays.single_element_initializer import SingleElementInitializerVisitor
from mutations.arrays.string_to_char_array import StringToCharArrayVisitor


class TestLenToGeneratorVisitor:
    def test_len_to_generator(self):
        code = """
        def myFunc(arr):
            return len(arr)
        """
        expected = ["""
        def myFunc(arr):
            return sum([1 for _ in arr])
        """]
        verify_visitor(LenToGeneratorVisitor, code, expected)

    def test_len_to_generator_nested(self):
        code = """
        def myFunc(mylist):
            return len(mylist[len(mylist) - 1])
        """
        expected = [
            """
            def myFunc(mylist):
                return sum([1 for _ in mylist[len(mylist) - 1]])""",
            """
            def myFunc(mylist):
                return len(mylist[sum([1 for _ in mylist]) - 1])
            """
        ]

        verify_visitor(LenToGeneratorVisitor, code, expected)

    def test_len_to_generator_range(self):
        code = """
        l = range(len(foo))
        """
        expected = """
        l = range(sum([1 for _ in foo]))
        """
        verify_visitor(LenToGeneratorVisitor, code, expected)


class TestListInitializerUnpackVisitor:
    def test_list_initializer_unpack(self):
        code = """
        def doSomething2():
            a = [1, 2, 3]
            return a
        """
        expected = """
        def doSomething2():
            a = list(*[[1, 2, 3]])
            return a
        """
        verify_visitor(ListInitializerUnpackVisitor, code, expected)

    def test_list_initializer_unpack_nonassign(self):
        code = """[]"""
        expected = []
        verify_visitor(ListInitializerUnpackVisitor, code, expected)

    def test_list_initializer_unpack_complex(self):
        code = """
        def myTest():
            myComplex = [1, True, {"a": 1, "b": 2}]
            myComplex = myComplex[:1]
        """
        expected = """
        def myTest():
            myComplex = list(*[[1, True, {'a': 1, 'b': 2}]])
            myComplex = myComplex[:1]
        """
        verify_visitor(ListInitializerUnpackVisitor, code, expected)


class TestNestedArrayInitializerVisitor:

    def test_nested_array_initializer(self):
        code = """
        def testNested(arr, foo):
            a = [1, 2, 3]
            if len(a) > 0:
                a = [None]
                return a
        """
        expected = [
            """
            def testNested(arr, foo):
                a = [[1, 2, 3]]
                if len(a) > 0:
                    a = [None]
                    return a
            """,
            """
            def testNested(arr, foo):
                a = [1, 2, 3]
                if len(a) > 0:
                    a = [[None]]
                    return a
            """
        ]
        verify_visitor(NestedArrayInitializerVisitor, code, expected)

    def test_nested_array_initializer_unpacking(self):
        code = """a, b = [1, 2]"""
        expected = []
        verify_visitor(NestedArrayInitializerVisitor, code, expected)

    def test_nested_array_initializer_compound(self):
        code = """a, b = [1, 2], [3, 4]"""
        expected = []
        verify_visitor(NestedArrayInitializerVisitor, code, expected)


class TestReverseRangeVisitor:
    def test_reverse_range_constant_vals_positive_step(self):
        code = """
        def loop():
            for i, j in enumerate(range(10)):
                print(i, j)
                for k in range(-1, 10, 2):
                    print(range(-30, -20))
        """
        expected = [
            """
            def loop():
                for (i, j) in enumerate(range(9, -1, -1)):
                    print(i, j)
                    for k in range(-1, 10, 2):
                        print(range(-30, -20))
            """,
            """
            def loop():
                for (i, j) in enumerate(range(10)):
                    print(i, j)
                    for k in range(9, -2, -2):
                        print(range(-30, -20))
            """,
            """
            def loop():
                for (i, j) in enumerate(range(10)):
                    print(i, j)
                    for k in range(-1, 10, 2):
                        print(range(-21, -31, -1))
             """
        ]
        verify_visitor(ReverseRangeVisitor, code, expected)

    def test_reverse_range_constant_vals_negative_step(self):
        code = """
        a = range(-5, -4)
        b = range(10, 0, -1)
        """
        expected = [
            """
            a = range(-5, -6, -1)
            b = range(10, 0, -1)
            """,
            """
            a = range(-5, -4)
            b = range(1, 11, 1)
            """
        ]
        verify_visitor(ReverseRangeVisitor, code, expected)

    def test_reverse_range_dynamic_vals(self):
        code = """
        def loop(n):
            for i in range(n):
                print(i)
        """
        expected = """
        def loop(n):
            for i in range(n - 1, -1, -1):
                print(i)
        """
        verify_visitor(ReverseRangeVisitor, code, expected)


class TestSingleElementInitializer:
    def test_single_element_initializer(self):
        code = """
        a = []
        b = [None]
        [None]
        """
        expected = """
        a = [None]
        b = [None]
        [None]
        """
        verify_visitor(SingleElementInitializerVisitor, code, expected)


class TestStringToCharArray:
    def test_string_to_char_array(self):
        code = """
        def doWork():
            '''oh no'''
            my_name = "Amrit\\nis cool"
            return my_name
        """
        expected = """
        def doWork():
            \"\"\"oh no\"\"\"
            my_name = ['A', 'm', 'r', 'i', 't', '\\n', 'i', 's', ' ', 'c', 'o', 'o', 'l']
            return my_name
        """
        verify_visitor(StringToCharArrayVisitor, code, expected)

    def test_string_to_char_array_empty(self):
        code = "foo = ''"
        expected = "foo = []"
        return verify_visitor(StringToCharArrayVisitor, code, expected)
