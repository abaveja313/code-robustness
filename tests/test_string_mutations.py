from mutations.strings.array_to_string import EmptyArrayToStringVisitor
from mutations.strings.constant_splitting import ConstantSplittingVisitor
from mutations.strings.string_concat_to_formatted import StringConcatToFStringVisitor
from mutations.strings.string_concat_to_join import StringConcatToJoinVisitor
from mutations.strings.string_to_bytestring import StringToBytestringVisitor
from utils import verify_visitor, verify_transformation


class TestArrayToString:
    def test_array_to_string(self):
        code = """
        a = []
        if i != 0:
            b = []
            a.append([1, 2, 3])
        """
        expected = [
            """
            a = ''
            if i != 0:
                b = []
                a.append([1, 2, 3])
            """,
            """
            a = []
            if i != 0:
                b = ''
                a.append([1, 2, 3])
            """
        ]
        verify_visitor(EmptyArrayToStringVisitor, code, expected)

    def test_array_to_string_ooc(self):
        code = """
        def foo():
            print([1, 2, 3])
            return [2, 4, 6]
        """
        expected = []
        verify_visitor(EmptyArrayToStringVisitor, code, expected)


class TestConstantSplitting:
    def test_constant_splitting(self):
        code = """
        a = '123456'
        b = "abc"
        c = 'smilyface'
        d = ''
        """
        expected = [
            """
            a = '123' + '456'
            b = 'abc'
            c = 'smilyface'
            d = ''
            """,
            """
            a = '123456'
            b = 'abc'
            c = 'smi' + 'lyface'
            d = ''            
            """
        ]
        verify_visitor(ConstantSplittingVisitor, code, expected)

    def test_constant_splitting_test_already_concat(self):
        code = """
        def doSomeWork():
            a = "123" + "456"
            c = "smily" + "face"
            d = ["abc"]
            return "AHHHH"
        """
        expected = [
            """
            def doSomeWork():
                a = '123' + '456'
                c = 'smi' + 'ly' + 'face'
                d = ['abc']
                return 'AHHHH'
            """,
            """
            def doSomeWork():
                a = '123' + '456'
                c = 'smily' + ('fac' + 'e')
                d = ['abc']
                return 'AHHHH'
            """,
            """
            def doSomeWork():
                a = '123' + '456'
                c = 'smily' + 'face'
                d = ['abc']
                return 'AHH' + 'HH'
            """,
        ]
        verify_visitor(ConstantSplittingVisitor, code, expected)


class TestStringConcatToFormatted:
    def test_string_concat_simple(self):
        code = """a = 'hello' + 'world'"""
        expected = """a = f'helloworld'"""
        verify_visitor(StringConcatToFStringVisitor, code, expected)

    def test_string_concat_to_formatted(self):
        code = """
        def poly(x):
            return "x^2 + "  + str(x) + " + 1"
        """
        expected = [
            """
            def poly(x):
                return f'x^2 + {str(x)}' + ' + 1'
            """,
            """
           def poly(x):
               return f'x^2 + {str(x)} + 1'
            """,
        ]
        verify_visitor(StringConcatToFStringVisitor, code, expected)

    def test_string_concat_to_formatted_complex(self):
        code = """
        def greet(first_name, last_name):
            name = 'Mr. ' + last_name + ', ' + first_name
            exclaimed = f'Hello, {name}!'
            if name.isupper():
                double_exclaimed = exclaimed + '!!'      
        """
        expected = [
            """
            def greet(first_name, last_name):
                name = f'Mr. {last_name}' + ', ' + first_name
                exclaimed = f'Hello, {name}!'
                if name.isupper():
                    double_exclaimed = exclaimed + '!!'             
            """,
            """
            def greet(first_name, last_name):
                name = f'Mr. {last_name}, ' + first_name
                exclaimed = f'Hello, {name}!'
                if name.isupper():
                    double_exclaimed = exclaimed + '!!'             
            """,
            """
            def greet(first_name, last_name):
                name = 'Mr. ' + last_name + ', ' + first_name
                exclaimed = f'Hello, {name}!'
                if name.isupper():
                    double_exclaimed = f'{exclaimed}!!'             
            """,
        ]
        verify_visitor(StringConcatToFStringVisitor, code, expected)


class TestStringConcatToJoin:
    def test_string_concat_simple(self):
        code = """a = 'hello' + 'world'"""
        expected = """a = ''.join(['hello', 'world'])"""
        verify_visitor(StringConcatToJoinVisitor, code, expected)

    def test_string_concat_to_formatted(self):
        code = """
        def poly(x):
            return "x^2 + "  + str(x) + " + 1"
        """
        expected = [
            """
            def poly(x):
                return ''.join(['x^2 + ', str(x)]) + ' + 1'
            """,
            """
            def poly(x):
                return ''.join(['x^2 + ', str(x), ' + 1'])
            """,
        ]
        verify_visitor(StringConcatToJoinVisitor, code, expected)


class TestStringToBytestring:
    def test_string_to_bytestring(self):
        code = """
        a = 'hello'
        b = "world"
        """
        expected = [
            """
            a = b'hello'
            b = 'world'
            """,
            """
            a = 'hello'
            b = b'world'
            """
        ]
        verify_visitor(StringToBytestringVisitor, code, expected)

    def test_string_to_bytestring_complex_assign(self):
        code = """
        def foo():
            a = 'hello'
            b = "world"
            c = 'smilyface'
            d = ''
            return a + b + c + d
        """
        expected = [
            """
            def foo():
                a = b'hello'
                b = 'world'
                c = 'smilyface'
                d = ''
                return a + b + c + d
            """,
            """
            def foo():
                a = 'hello'
                b = b'world'
                c = 'smilyface'
                d = ''
                return a + b + c + d
            """,
            """
            def foo():
                a = 'hello'
                b = 'world'
                c = b'smilyface'
                d = ''
                return a + b + c + d
            """
        ]