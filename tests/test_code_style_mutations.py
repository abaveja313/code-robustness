from utils import verify_transformation, verify_visitor

from mutations.code_style.add_parens import AddParensTransformer
from mutations.code_style.comments_block import BlockCommentsTransformer
from mutations.code_style.comments_inline import InlineCommentsTransformer
from mutations.code_style.expand_augmented_assign import ExpandAugmentedAssignVisitor
from mutations.code_style.identifiers_rename import IdentifierRenameVisitor
from mutations.code_style.identity_assignment import IdentityAssignmentVisitor
from mutations.code_style.merge_statements import MergeStatementsTransformer
from mutations.code_style.print_injector import PrintInjectionVisitor
from mutations.code_style.string_quote import StringQuoteDoubleTransformer, StringQuoteSingleTransformer
from mutations.code_style.unused_variables import UnusedVariableVisitor


class TestAddParens:
    def test_simple_add_parens(self):
        code = """
        def foo():
            r = 1 + 2 + 3
            return r * r
        """
        expected = [
            """ 
            def foo():
                r = (1 + 2) + 3
                return r * r
            """,
            """ 
            def foo():
                r = (1 + 2 + 3)
                return r * r
            """,
            """ 
            def foo():
                r = 1 + 2 + 3
                return (r * r)
            """""
        ]
        verify_transformation(AddParensTransformer, code, expected)

    def test_add_parens_docstring(self):
        code = """
        def foo():
            '''
            I am a docstring
            
            assert 1 + 2 + 3 == 6
            '''
            pass
        """
        verify_transformation(AddParensTransformer, code, [])

    def test_add_parens_docstring_nested(self):
        code = """
        def foo():
            '''
            I am a docstring

            assert 1 + 2 + 3 == 6
            '''
            def bar():
                \"\"\"
                a = True + False
                \"\"\"
                return True
            pass
        """
        verify_transformation(AddParensTransformer, code, [])

    def test_add_parens_with_bools(self):
        code = """
        a = not (a or b)
        if a and b or a < 2:
            pass
        """
        expected = [
            """
            a = (not (a or b))
            if a and b or a < 2:
                pass
            """,
            """
            a = not ((a or b))
            if a and b or a < 2:
                pass
            """,
            """
            a = not (a or b)
            if (a and b) or a < 2:
                pass
            """,
            """
           a = not (a or b)
           if (a and b or a < 2):
               pass
           """,
            """
               a = not (a or b)
               if a and b or (a < 2):
                   pass
               """,
        ]
        verify_transformation(AddParensTransformer, code, expected)


class TestCommentsBlock:
    def test_block_comments(self):
        code = """
        def factorial(n):
            if n == 0:
                return 1
            return n * factorial(n - 1)
        """
        expected = [
            """
            def factorial(n):
                # I am a block comment
                # I am a block comment
                if n == 0:
                    return 1
                return n * factorial(n - 1)
            """,
            """
            def factorial(n):
                if n == 0:            
                    # I am a block comment
                    # I am a block comment
                    return 1
                return n * factorial(n - 1)
            """,
            """
            def factorial(n):
                if n == 0:            
                    return 1
                # I am a block comment
                # I am a block comment
                return n * factorial(n - 1)
            """
        ]
        verify_transformation(BlockCommentsTransformer, code, expected)

    def test_block_comments_docstring(self):
        code = '''
        def factorial(n):
            """
            assert factorial(0) == 1 + 0
            """
            def inner(n):
                """
                assert inner(0) == 1 + 0
                """
        '''
        expected = [
            """
            def factorial(n):
                \"\"\"
                assert factorial(0) == 1 + 0
                \"\"\"
                # I am a block comment
                # I am a block comment
                def inner(n):
                    \"\"\"
                    assert inner(0) == 1 + 0
                    \"\"\"
            """
        ]
        verify_transformation(BlockCommentsTransformer, code, expected)

    def test_block_comments_empty_line(self):
        code = """
        """
        expected = []
        verify_transformation(BlockCommentsTransformer, code, expected)


class TestCommentsInline:
    def test_inline_comments_multi_func(self):
        code = """
        def funcA(a, b):
            return funcB(a)
        
        def funcB(a):
            pass
        """
        expected = [
            """
            def funcA(a, b):
                return funcB(a)  # I am a comment
            
            def funcB(a):
                pass
            """,
            """
            def funcA(a, b):
                return funcB(a) 
    
            def funcB(a):
                pass  # I am a comment
            """
        ]
        verify_transformation(InlineCommentsTransformer, code, expected)

    def test_inline_comments(self):
        code = """
        def doSomething(a, b):
            if name == 'foo':
                doSomething(1, 2)
            return a + b
       """
        expected = [
            """
            def doSomething(a, b):
                if name == 'foo':  # I am a comment
                    doSomething(1, 2)
                return a + b
            """,
            """
            def doSomething(a, b):
                if name == 'foo':
                    doSomething(1, 2)  # I am a comment
                return a + b
            """,
            """
            def doSomething(a, b):
                if name == 'foo':
                    doSomething(1, 2)
                return a + b  # I am a comment
            """
        ]
        verify_transformation(InlineCommentsTransformer, code, expected)

    def test_inline_comments_docstring(self):
        code = '''
        def doSomething(a, b):
            """
            assert doSomething(1, 2) == 3
            """
        '''
        verify_transformation(InlineCommentsTransformer, code, [])


class TestExpandAugmentedAssign:
    def test_expand_augmented_assign_simple(self):
        code = """
        a = 1
        a += 2
        """
        expected = [
            """
            a = 1
            a = a + 2
            """
        ]
        verify_visitor(ExpandAugmentedAssignVisitor, code, expected)

    def test_expand_augmented_assign_weird(self):
        code = """
        c = 94
        c %= 2
        c *= 3
        """
        expected = [
            """
            c = 94
            c = c % 2
            c *= 3
            """,
            """
            c = 94
            c %= 2
            c = c * 3
            """,
        ]
        verify_visitor(ExpandAugmentedAssignVisitor, code, expected)


class TestIdentifierRename:
    def test_identifier_rename_simple(self):
        code = """
        def foo(c, d):
            a = 1
            b = 2
            return a + b
       """
        expected = [
            """
            def foo(e, f):
                a = 1
                b = 2
                return a + b
            """,
            """
            def foo(c, d):
                e = 1
                b = 2
                return a + b
            """,
            """
            def foo(c, d):
                a = 1
                e = 2
                return a + b
            """
        ]
        verify_visitor(IdentifierRenameVisitor, code, expected)

    def test_identifier_rename_complex(self):
        code = """
         def foo(a, b):
            d,c = 6,4
            for e in range(a + d):
                g = 3
                c += g
             return c
        """
        expected = [
            """
            def foo(f, h):
                (d, c) = (6, 4)
                for e in range(a + d):
                    g = 3
                    c += g
                return c
            """,
            """
            def foo(a, b):
                (d, c) = (6, 4)
                for f in range(a + d):
                    g = 3
                    c += g
                return c
            """,
            """
            def foo(a, b):
                (d, c) = (6, 4)
                for e in range(a + d):
                    f = 3
                    c += g
                return c
            """
        ]
        verify_visitor(IdentifierRenameVisitor, code, expected)


class TestIdentityAssignment:
    def test_identity_assignment_simple(self):
        code = """
        def doSomething():
            a = 4
            print(a)
            (c, d) = (4, 5)
        """
        expected = [
            """
            def doSomething():
                a = 4
                a = a
                print(a)
                (c, d) = (4, 5)
            """
        ]
        verify_visitor(IdentityAssignmentVisitor, code, expected)

    def test_identity_assignment_tuple(self):
        code = """
        def doSomething():
            a = 4
            print(a)
        """
        expected = [
            """
            def doSomething():
                a = 4
                a = a
                print(a)
            """
        ]
        verify_visitor(IdentityAssignmentVisitor, code, expected)

    def test_identity_assignment_nested(self):
        code = """
        if __name == "__main__":
            f = main
            f()
        """
        expected = """
        if __name == '__main__':
            f = main
            f = f
            f()
        """
        verify_visitor(IdentityAssignmentVisitor, code, expected)


class TestMergeStatements:
    def test_merge_statements_simple(self):
        code = """
        a = 1
        b = 2
        c = 3
        """
        expected = [
            """
            a = 1; b = 2
            c = 3
            """,
            """
            a = 1
            b = 2; c = 3
            """
        ]
        verify_transformation(MergeStatementsTransformer, code, expected)

    def test_merge_statements_blocks(self):
        code = """
        def foo(a):
            a = 3
            if a == 3:
                b = 6
                b -= 4
                print(b)
                return b
            else:
                c = 4
                c += 3
                return c
        """
        expected = [
            """
            def foo(a):
                a = 3
                if a == 3:
                    b = 6; b -= 4
                    print(b)
                    return b
                else:
                    c = 4
                    c += 3
                    return c
            """,
            """
            def foo(a):
                a = 3
                if a == 3:
                    b = 6
                    b -= 4; print(b)
                    return b
                else:
                    c = 4
                    c += 3
                    return c
            """,
            """
            def foo(a):
                a = 3
                if a == 3:
                    b = 6
                    b -= 4
                    print(b)
                    return b
                else:
                    c = 4; c += 3
                    return c
            """
        ]
        verify_transformation(MergeStatementsTransformer, code, expected)

    def test_merge_statements_nested(self):
        code = """
        def bar():
            if a == 3:
                a = 4
                if b == 4:
                    b = 5
        """
        expected = []
        verify_transformation(MergeStatementsTransformer, code, expected)

    def test_merge_statements_docstring(self):
        code = """
        def foo():
            '''
            a = 1
            b = 2
            print(a + b)
            '''
            pass
            """
        verify_transformation(MergeStatementsTransformer, code, [])


class TestPrintInjector:
    def test_print_injector_simple(self):
        code = """
        def foo(a, b):
            a = 6
            return a + b
        """
        expected = [
            """
            def foo(a, b):
                a = 6
                print('This line was reached for debugging!')
                return a + b
            """
        ]
        verify_visitor(PrintInjectionVisitor, code, expected)

    def test_print_injector_blocks(self):
        code = """
        if foo:
            a = True
            doSomething(a)
        else:
            return c
        """
        expected = [
            """
            if foo:
                a = True
                print('This line was reached for debugging!')
                doSomething(a)
            else:
                return c
            """,
            """
            if foo:
                a = True
                doSomething(a)
                print('This line was reached for debugging!')
            else:
                return c
            """
        ]
        verify_visitor(PrintInjectionVisitor, code, expected)


class TestStringQuote:
    def test_string_quote_single(self):
        code = """
        a = "foo"
        print('bar' + "hello" + a)
        """
        expected = [
            """
            a = 'foo'
            print('bar' + "hello" + a)
            """,
            """
            a = "foo"
            print('bar' + 'hello' + a)
            """
        ]
        verify_transformation(StringQuoteSingleTransformer, code, expected)

    def test_string_quote_docstring(self):
        code = """
        def foo():
            '''
            a = ""
            '''
            pass
        """
        verify_transformation(StringQuoteSingleTransformer, code, [])

    def test_string_quote_double(self):
        code = """
        def doSomething():
            l = ['foo', "bar", 'fodo']
            return map(lambda x: x + 'hello', l)
        """
        expected = [
            """
            def doSomething():
                l = ["foo", "bar", 'fodo']
                return map(lambda x: x + 'hello', l)
            """,
            """
            def doSomething():
                l = ['foo', "bar", 'fodo']
                return map(lambda x: x + "hello", l)
            """,
            """
            def doSomething():
                l = ['foo', "bar", "fodo"]
                return map(lambda x: x + 'hello', l)
            """,
        ]
        verify_transformation(StringQuoteDoubleTransformer, code, expected)


class TestUnusedVariables:
    def test_unused_variable_simple(self):
        code = """
        def foo(a, b):
            a = 6
            if True:
                print(a)
            return a
        """
        expected = [
            """
            def foo(a, b):
                a = 6
                foo = 3
                if True:
                    print(a)
                return a
            """,
            """
            def foo(a, b):
                a = 6
                if True:
                    print(a)
                    foo = 3
                return a
            """
        ]
        verify_visitor(UnusedVariableVisitor, code, expected)
