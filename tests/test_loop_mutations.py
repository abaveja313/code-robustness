from utils import verify_visitor
from mutations.loops.enumerate_for import EnumerateForVisitor
from mutations.loops.for_to_while import ForToWhileVisitor
from mutations.loops.while_to_if import WhileToIfVisitor


class TestEnumerateFor:
    def test_enumerate_for_simple(self):
        code = """
        def doWork():
            for i in range(10):
                print(i)
                
                for j in [1,2,3]:
                    print(i + j)
        """

        expected = [
            """
            def doWork():
                for (cidx, i) in enumerate(range(10)):
                    print(i)
                    for j in [1, 2, 3]:
                        print(i + j)
            """,
            """
            def doWork():
                for i in range(10):
                    print(i)
                    for (cidx, j) in enumerate([1, 2, 3]):
                        print(i + j)
            """,
        ]
        verify_visitor(EnumerateForVisitor, code, expected)

    def test_enumerate_for_already_enumerated(self):
        code = """
        for (idx, i) in enumerate(range(10)):
            print(i)
        """
        expected = []
        verify_visitor(EnumerateForVisitor, code, expected)

    def test_enumerate_for_tuple(self):
        code = """
        for (i, j) in zip(a, b):
            print(i, j)
        """
        expected = """
        for (cidx, i, j) in enumerate(zip(a, b)):
            print(i, j)
        """
        verify_visitor(EnumerateForVisitor, code, expected)


class TestForToWhile:
    def test_for_to_while_basic(self):
        code = """
        for i in range(10):
            print(i)
            
            for j in range(2, 7):
                print(j)
        """
        expected = [
            """
            i = 0
            while i < 10:
                print(i)
                for j in range(2, 7):
                    print(j)
                i += 1
            """,
            """
            for i in range(10):
                print(i)
                j = 2
                while j < 7:
                    print(j)
                    j += 1
            """
        ]
        verify_visitor(ForToWhileVisitor, code, expected)

    def test_while_loop_no_action(self):
        code = """
        i = 0
        while i < 10:
            pass
        """
        expected = []
        verify_visitor(ForToWhileVisitor, code, expected)

    def test_complex_loop(self):
        code = """
        for i, j in range(4):
            print(i)
        """
        expected = []
        verify_visitor(ForToWhileVisitor, code, expected)

    def test_for_to_while_non_range(self):
        code = """
        for i in a:
            pass
        """
        expected = []
        verify_visitor(ForToWhileVisitor, code, expected)

    def test_for_to_while_negative_range(self):
        code = """
        def doIt():
            for po in range(10, -10, -2):
                print(i)
        """
        expected = """
        def doIt():
            po = 10
            while po > -10:
                print(i)
                po -= 2
        """
        verify_visitor(ForToWhileVisitor, code, expected)


class TestWhileToIf:
    def test_while_to_if(self):
        code = """
        while i < 10:
            print(i)
        """
        expected = """
        if i < 10:
            pass
        """
        verify_visitor(WhileToIfVisitor, code, expected)
