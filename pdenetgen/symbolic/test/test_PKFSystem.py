import unittest
from sympy import Derivative, Integer
from pydap.symbolic import SymbolicPKF, Expectation, Eq
from pydap.symbolic.dynamics import burgers


class TestSymbolicPKF_Burgers(unittest.TestCase):

    def test_all_system(self):
        # 1) Extract information from Burgers dynamics
        u, = burgers.prognostic_functions
        t, x = burgers.coordinates
        kappa, = burgers.constants

        # 2) Apply PKF
        pkf = SymbolicPKF(burgers)

        # 3) Prepare theoretical value
        mfield = pkf.fields[u]

        Eu = Expectation(mfield.random)
        eu = mfield.error
        epsilon = mfield.epsilon
        Vu = mfield.variance
        gu = mfield.metric[0]
        nuu = mfield.diffusion[0]
        uc_term = Expectation(epsilon * Derivative(epsilon, x, 4))

        # 4) Set the theoretical system or equation to compare with the computed ones
        error_system = [Eq(
            Derivative(eu, t),
            kappa * Derivative(eu, (x, 2)) - eu * Derivative(Eu, x) - Eu * Derivative(eu, x)
        )]

        expectation_system = [Eq(
            Derivative(Eu, t),
            kappa * Derivative(Eu, x, 2) - Eu * Derivative(Eu, x) - Derivative(Vu, x) / Integer(2)
        )]

        variance_system = [Eq(
            Derivative(Vu, t),
            -2 * kappa * Vu * gu + kappa * Derivative(Vu, (x, 2)) - kappa * Derivative(Vu, x) ** 2 / (
                    2 * Vu) - 2 * Vu * Derivative(Eu, x) - \
            Eu * Derivative(Vu, x)
        )]

        epsilon_system = [
            Eq(
                Derivative(epsilon, t),
                kappa * epsilon * gu + kappa * Derivative(epsilon, (x, 2)) + \
                kappa * Derivative(Vu, x) * Derivative(epsilon, x) / Vu - Eu * Derivative(epsilon, x))
        ]

        metric_system = [Eq(
            Derivative(gu, t),
            2 * kappa * gu ** 2 - 2 * kappa * Expectation(epsilon * Derivative(epsilon, (x, 4))) - \
            3 * kappa * Derivative(gu, (x, 2)) + \
            2 * kappa * gu * Derivative(Vu, (x, 2)) / Vu + kappa * Derivative(Vu, x) * Derivative(gu, x) / Vu - \
            2 * kappa * gu * Derivative(Vu, x) ** 2 / Vu ** 2 - 2 * gu * Derivative(Eu, x) - Eu * Derivative(gu, x)
        )]

        kurtosis_closure = {uc_term: 3 * gu ** 2 - 2 * Derivative(gu, x, 2)}

        pkf.set_closure(kurtosis_closure)

        closed_diffusion_eq = Eq(
            Derivative(nuu, t),
            kappa * Derivative(nuu, (x, 2)) + 2 * kappa - 2 * kappa * Derivative(nuu, x) ** 2 / nuu - \
            2 * kappa * nuu * Derivative(Vu, (x, 2)) / Vu + kappa * Derivative(Vu, x) * Derivative(nuu, x) / Vu + \
            2 * kappa * nuu * Derivative(Vu, x) ** 2 / Vu ** 2 - u * Derivative(nuu, x) + 2 * nuu * Derivative(u, x)
        )

        # 5) Do tests
        self.assertEqual(pkf.error_system, error_system)
        self.assertEqual(pkf.expectation_system, expectation_system)
        self.assertEqual(pkf.variance_system, variance_system)
        self.assertEqual(pkf.epsilon_system, epsilon_system)
        self.assertEqual(pkf.metric_system, metric_system)
        self.assertEqual(pkf.in_diffusion[2], closed_diffusion_eq)

if __name__ == '__main__':
    unittest.main()
