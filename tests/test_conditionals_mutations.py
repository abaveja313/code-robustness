from mutations.conditionals.if_to_conditional import IfToConditionalVisitor
from mutations.conditionals.if_to_while_loop import IfToWhileLoopVisitor
from utils import verify_visitor


class TestIfToConditional:
    def test_if_to_conditional(self):
        code = """
        if name == "Alice":
            a = 3
        else:
            a = 2
        """
        expected = """
        a = 3 if name == 'Alice' else 2
        """
        verify_visitor(IfToConditionalVisitor, code, expected)

    def test_if_to_conditional_with_elif(self):
        code = """
        def doWork(name):
            if name == "Alice":
                a = 3
            elif name == "Bob":
                a = 2
            else:
                a = 1
        """
        expected = """
        def doWork(name):
            if name == 'Alice':
                a = 3
            else:
                a = 2 if name == 'Bob' else 1
        """
        verify_visitor(IfToConditionalVisitor, code, expected)

    def test_if_to_conditional_nested(self):
        code = """
        if name == "Alice":
            if age == 20:
                a = 3
            else:
                a = 2
        else:
            a = 1
        """
        expected = """
        if name == 'Alice':
            a = 3 if age == 20 else 2
        else:
            a = 1
        """
        verify_visitor(IfToConditionalVisitor, code, expected)

    def test_if_to_conditional_diff_variables(self):
        code = """
        if True:
            a = False
        else:
            b = False
        """
        expected = []
        verify_visitor(IfToConditionalVisitor, code, expected)


class TestIfToWhileLoop:
    def test_if_to_while_elif(self):
        code = """
        if a == 3:
            a = 2
        elif a == 2:
            b = 2
        else:
            c = 3
        """
        expected = [
            """
            while a == 3:
                pass
            """,
        ]
        verify_visitor(IfToWhileLoopVisitor, code, expected)

    def test_if_to_while_loop(self):
        code = """
        if a == 3:
            a = 2
        else:
            a = 1
        """
        expected = """
        while a == 3:
            pass
        """
        verify_visitor(IfToWhileLoopVisitor, code, expected)

    def test_if_to_while_loop_with_elif(self):
        code = """
        if a == 3:
            a = 2
        elif a == 2:
            a = 1
        else:
            a = 0
        """
        expected = [
            """
            while a == 3:
                pass
            """
        ]
        verify_visitor(IfToWhileLoopVisitor, code, expected)

    def test_if_to_while_loop_nested(self):
        code = """
        if a == 3:
            if b == 4:
                a = 2
            else:
                a = 1
        else:
            a = 0
        """
        expected = [
            """
            while a == 3:
                pass  
            """,
            """
            if a == 3:
                while b == 4:
                    pass
            else:
                a = 0
            """
        ]
        verify_visitor(IfToWhileLoopVisitor, code, expected)
