from mutations.dicts.dict_to_array import DictToArrayVisitor
from mutations.dicts.array_to_dict import ArrayToDictVisitor
from mutations.dicts.dict_initializer_unpack import DictInitializerUnpackVisitor

from utils import verify_visitor


class TestArrayToDict:
    def test_array_to_dict_simple(self):
        code = """
        a = []
        def foo():
            b = []
        """
        expected = [
            """
            a = {}
            
            def foo():
                b = []
            """,
            """
            a = []
            
            def foo():
                b = {}
            """
        ]
        verify_visitor(ArrayToDictVisitor, code, expected)

    def test_array_to_dict_full(self):
        code = """
        p = [1, 2]
        """
        expected = []
        verify_visitor(ArrayToDictVisitor, code, expected)


class TestDictInitializerUnpack:
    def test_dict_initializer_unpack_simple(self):
        code = """
        a = {1: 2, 2: 3, 3: 4}
        """
        expected = [
            """
            a = {**{1: 2, 2: 3, 3: 4}, **{}}
            """
        ]
        verify_visitor(DictInitializerUnpackVisitor, code, expected)

    def test_dict_initializer_unpack_arg(self):
        code = """
        code = {}
        print({'foo': 1, 'bar': 2})
        """
        expected = [
            """
            code = {**{}, **{}}
            print({'foo': 1, 'bar': 2})
            """
        ]
        verify_visitor(DictInitializerUnpackVisitor, code, expected)


class TestDictToArray:
    def test_dict_to_array_simple(self):
        code = """
        a = {}
        def foo():
            b = {}
        """
        expected = [
            """
            a = []

            def foo():
                b = {}
            """,
            """
            a = {}

            def foo():
                b = []
            """
        ]
        verify_visitor(DictToArrayVisitor, code, expected)

    def test_array_to_dict_full(self):
        code = """
        p = {1: 2}
        """
        expected = []
        verify_visitor(DictToArrayVisitor, code, expected)
