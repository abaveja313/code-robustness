from mutations.math.arithmetic_to_bitshift import MultiplyBy2ToBitshiftVisitor, DivideBy2ToBitshiftVisitor, \
    NegationToComplementVisitor
from mutations.math.math_inverter import ModuloInversionVisitor, AdditionInversionVisitor, SubtractionInversionVisitor, \
    MultiplicationInversionVisitor, DivisionInversionVisitor
from utils import verify_visitor


class TestArithmeticToBitshift:
    def test_mul_by_two(self):
        code = """
        a = 2
        while a * 2 < 10:
            print(a)
        """
        expected = """
        a = 2
        while a << 1 < 10:
            print(a)
        """
        verify_visitor(MultiplyBy2ToBitshiftVisitor, code, expected)

    def test_mul_by_two_with_assignment(self):
        code = """
        a = 2
        b = a * 2
        """
        expected = """
        a = 2
        b = a << 1
        """
        verify_visitor(MultiplyBy2ToBitshiftVisitor, code, expected)

    def test_div_by_two(self):
        code = """
        if a / 2 % 2 == 0:
            p = 6 / 2
        """
        expected = [
            """
            if (a >> 1) % 2 == 0:
                p = 6 / 2
            """,
            """
            if a / 2 % 2 == 0:
                p = 6 >> 1
            """
        ]
        verify_visitor(DivideBy2ToBitshiftVisitor, code, expected)

    def test_complement(self):
        code = """
        a = -b
        b = -a
        """
        expected = [
            """
            a = ~b + 1
            b = -a
            """,
            """
            a = -b
            b = ~a + 1
            """
        ]
        verify_visitor(NegationToComplementVisitor, code, expected)

    def test_multi_complement(self):
        code = """
        if -(-a) == a:
            print('yes')
        """
        expected = [
            """
            if -(~a + 1) == a:
                print('yes')
            """,
            """
            if ~-a + 1 == a:
                print('yes')
            """
        ]
        verify_visitor(NegationToComplementVisitor, code, expected)


class TestMathInverter:
    def test_modulus(self):
        code = """
        if a % 2 == 0:
            print((c + d) % (d + e))
        """
        expected = [
            """
            if a - 2 * (a // 2) == 0:
                print((c + d) % (d + e))
            """,
            """
            if a % 2 == 0:
                print(c + d - (d + e) * ((c + d) // (d + e)))
            """
        ]
        verify_visitor(ModuloInversionVisitor, code, expected)

    def test_addition(self):
        code = """
        if a + b == c:
            print(d - a)
        """
        expected = [
            """
            if a - -b == c:
                print(d - a)
            """
        ]
        verify_visitor(AdditionInversionVisitor, code, expected)

    def test_subtraction(self):
        code = """
        print(a - (b - c))
        """
        expected = [
            """
            print(a + -(b - c))
            """,
            """
            print(a - (b + -c))
            """
        ]
        verify_visitor(SubtractionInversionVisitor, code, expected)

    def test_multiplication(self):
        code = """
        a = 0
        while a * 8 < 10:
            print(a)
        """
        expected = [
            """
            a = 0
            while a / (1.0 / 8) < 10:
                print(a)
            """
        ]
        verify_visitor(MultiplicationInversionVisitor, code, expected)

    def test_division(self):
        code = """
        a = 0
        if a / 8 < 10:
            print(a)
        """
        expected = [
            """
            a = 0
            if a * 8 ** -1 < 10:
                print(a)
            """
        ]
        verify_visitor(DivisionInversionVisitor, code, expected)
