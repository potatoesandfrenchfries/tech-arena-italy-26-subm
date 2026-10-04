import pytest

from compat_galerkin import checks, fem

TOL = 1e-10  # README Gate 0 threshold

CAVITIES = {
    "square_uniform": lambda: fem.build_cavity(fem.unit_square(3)),
    "square_inclusion": lambda: fem.build_cavity(
        m := fem.unit_square(3), fem.inclusion_eps(m)
    ),
    "lshape_inclusion": lambda: fem.build_cavity(
        m := fem.l_shape(3), fem.inclusion_eps(m, center=(0.4, 0.4))
    ),
}


@pytest.fixture(params=CAVITIES, scope="module")
def report(request):
    return checks.gate0(CAVITIES[request.param]())


@pytest.mark.parametrize(
    "key", ["dof_convention", "curl_grad", "lift_identity", "pec_trace", "gauss_reduced", "power_balance"]
)
def test_identities_hold(report, key):
    assert report[key] < TOL, report


def test_gauss_control_has_teeth(report):
    # the same test on functions outside the reduced space must fail by a wide margin
    assert report["gauss_control"] > 1e-4, report
