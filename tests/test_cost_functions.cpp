#include <utils2/test.hpp>
#include <utils2/cost_functions.hpp>
#include <cmath>
#include <vector>
#include <array>

TEST_CASE("Dual number arithmetic") {
    using D = utils2::Dual<double, 2>;

    D a(3.0, 0); // x = 3, dx/dx = 1, dx/dy = 0
    D b(4.0, 1); // y = 4, dy/dx = 0, dy/dy = 1

    auto sum = a + b;
    REQUIRE_NEAR(sum.value, 7.0, 1e-12);
    REQUIRE_NEAR(sum.derivatives[0], 1.0, 1e-12);
    REQUIRE_NEAR(sum.derivatives[1], 1.0, 1e-12);

    auto diff = a - b;
    REQUIRE_NEAR(diff.value, -1.0, 1e-12);

    auto prod = a * b;
    REQUIRE_NEAR(prod.value, 12.0, 1e-12);
    // d(x*y)/dx = y = 4, d(x*y)/dy = x = 3
    REQUIRE_NEAR(prod.derivatives[0], 4.0, 1e-12);
    REQUIRE_NEAR(prod.derivatives[1], 3.0, 1e-12);

    auto quot = a / b;
    REQUIRE_NEAR(quot.value, 0.75, 1e-12);
    // d(x/y)/dx = 1/y = 0.25
    REQUIRE_NEAR(quot.derivatives[0], 0.25, 1e-12);
    // d(x/y)/dy = -x/y^2 = -3/16
    REQUIRE_NEAR(quot.derivatives[1], -3.0 / 16.0, 1e-12);

    auto neg = -a;
    REQUIRE_NEAR(neg.value, -3.0, 1e-12);
    REQUIRE_NEAR(neg.derivatives[0], -1.0, 1e-12);
}

TEST_CASE("Dual constant promotion") {
    using D = utils2::Dual<double, 1>;

    D x(2.0, 0);
    D c(5.0); // constant, no derivatives

    auto r = x * c;
    REQUIRE_NEAR(r.value, 10.0, 1e-12);
    REQUIRE_NEAR(r.derivatives[0], 5.0, 1e-12); // d(5x)/dx = 5
}

TEST_CASE("Dual math functions sin/cos") {
    using D = utils2::Dual<double, 1>;

    D x(std::numbers::pi / 6.0, 0); // 30 degrees

    auto s = utils2::sin(x);
    REQUIRE_NEAR(s.value, 0.5, 1e-12);
    REQUIRE_NEAR(s.derivatives[0], std::cos(std::numbers::pi / 6.0), 1e-12);

    auto c = utils2::cos(x);
    REQUIRE_NEAR(c.value, std::cos(std::numbers::pi / 6.0), 1e-12);
    REQUIRE_NEAR(c.derivatives[0], -std::sin(std::numbers::pi / 6.0), 1e-12);
}

TEST_CASE("Dual math functions exp") {
    using D = utils2::Dual<double, 1>;

    D x(1.0, 0);
    auto r = utils2::exp(x);
    REQUIRE_NEAR(r.value, std::exp(1.0), 1e-12);
    REQUIRE_NEAR(r.derivatives[0], std::exp(1.0), 1e-12); // d(exp(x))/dx = exp(x)
}

TEST_CASE("Dual math functions sqrt") {
    using D = utils2::Dual<double, 1>;

    D x(4.0, 0);
    auto r = utils2::sqrt(x);
    REQUIRE_NEAR(r.value, 2.0, 1e-12);
    REQUIRE_NEAR(r.derivatives[0], 0.25, 1e-12); // d(sqrt(x))/dx = 1/(2*sqrt(x)) = 0.25
}

TEST_CASE("Dual math functions log") {
    using D = utils2::Dual<double, 1>;

    D x(2.0, 0);
    auto r = utils2::log(x);
    REQUIRE_NEAR(r.value, std::log(2.0), 1e-12);
    REQUIRE_NEAR(r.derivatives[0], 0.5, 1e-12); // d(ln(x))/dx = 1/x
}

TEST_CASE("autodiff gradient computation") {
    // f(x, y) = x^2 + 3*x*y + y^2
    // df/dx = 2x + 3y, df/dy = 3x + 2y
    auto f = [](const auto& p) {
        return p[0] * p[0] + p[0] * p[1] * decltype(p[0])(3.0) + p[1] * p[1];
    };

    std::array<double, 2> params = {2.0, 1.0};
    auto [val, grad] = utils2::autodiff<2>(f, params);

    REQUIRE_NEAR(val, 4.0 + 6.0 + 1.0, 1e-12); // 11.0
    REQUIRE_NEAR(grad[0], 2.0 * 2.0 + 3.0 * 1.0, 1e-12); // 7.0
    REQUIRE_NEAR(grad[1], 3.0 * 2.0 + 2.0 * 1.0, 1e-12); // 8.0
}

TEST_CASE("Jacobian computation") {
    // f(x, y) = [x + y, x * y]
    // J = [[1, 1], [y, x]]
    auto f = [](const auto& p) {
        using T = std::remove_cvref_t<decltype(p[0])>;
        return std::array<T, 2>{p[0] + p[1], p[0] * p[1]};
    };

    std::array<double, 2> params = {3.0, 5.0};
    auto J = utils2::jacobian<2, 2>(f, params);

    REQUIRE_NEAR(J[0][0], 1.0, 1e-12); // d(x+y)/dx
    REQUIRE_NEAR(J[0][1], 1.0, 1e-12); // d(x+y)/dy
    REQUIRE_NEAR(J[1][0], 5.0, 1e-12); // d(x*y)/dx = y
    REQUIRE_NEAR(J[1][1], 3.0, 1e-12); // d(x*y)/dy = x
}

TEST_CASE("Levenberg-Marquardt fit a line") {
    // Fit y = m*x + b to data points: (0,1), (1,3), (2,5), (3,7)
    // True solution: m=2, b=1.
    struct Data {
        std::vector<double> x{0.0, 1.0, 2.0, 3.0};
        std::vector<double> y{1.0, 3.0, 5.0, 7.0};
    };
    Data data;

    auto residual_func = [&data](std::span<const double> params) -> std::vector<double> {
        double m = params[0];
        double b = params[1];
        std::vector<double> r(data.x.size());
        for (std::size_t i = 0; i < data.x.size(); ++i)
            r[i] = m * data.x[i] + b - data.y[i];
        return r;
    };

    std::vector<double> initial = {0.0, 0.0};
    utils2::LMConfig<double> cfg;
    cfg.max_iterations = 100;
    cfg.tolerance = 1e-12;

    auto result = utils2::levenberg_marquardt(residual_func, initial, 4, cfg);

    REQUIRE(result.converged);
    REQUIRE_NEAR(result.params[0], 2.0, 1e-4); // m
    REQUIRE_NEAR(result.params[1], 1.0, 1e-4); // b
    REQUIRE_NEAR(result.final_cost, 0.0, 1e-6);
}

TEST_CASE("Levenberg-Marquardt fit quadratic") {
    // Fit y = a*x^2 + b*x + c to (0,1), (1,2), (2,5), (3,10)
    // True: a=1, b=0, c=1.
    std::vector<double> xs{0.0, 1.0, 2.0, 3.0};
    std::vector<double> ys{1.0, 2.0, 5.0, 10.0};

    auto residual_func = [&](std::span<const double> p) -> std::vector<double> {
        std::vector<double> r(xs.size());
        for (std::size_t i = 0; i < xs.size(); ++i)
            r[i] = p[0] * xs[i] * xs[i] + p[1] * xs[i] + p[2] - ys[i];
        return r;
    };

    auto result = utils2::levenberg_marquardt(
        residual_func, std::vector<double>{0.0, 0.0, 0.0}, 4);

    REQUIRE(result.converged);
    REQUIRE_NEAR(result.params[0], 1.0, 1e-3);
    REQUIRE_NEAR(result.params[1], 0.0, 1e-3);
    REQUIRE_NEAR(result.params[2], 1.0, 1e-3);
}

TEST_CASE("gradient_descent minimize quadratic") {
    // Minimize f(x, y) = (x - 3)^2 + (y + 1)^2
    auto f = [](const auto& p) {
        auto dx = p[0] - decltype(p[0])(3.0);
        auto dy = p[1] + decltype(p[1])(1.0);
        return dx * dx + dy * dy;
    };

    std::array<double, 2> initial = {0.0, 0.0};
    auto result = utils2::gradient_descent<2>(f, initial, 0.1, 1000, 1e-10);

    REQUIRE_NEAR(result[0], 3.0, 1e-4);
    REQUIRE_NEAR(result[1], -1.0, 1e-4);
}

TEST_CASE("DistanceCost") {
    utils2::DistanceCost<double, 3> cost{.target = {1.0, 2.0, 3.0}};

    double point[] = {1.0, 2.0, 3.0};
    double val = cost(point);
    REQUIRE_NEAR(val, 0.0, 1e-12);

    double point2[] = {4.0, 2.0, 3.0};
    double val2 = cost(point2);
    REQUIRE_NEAR(val2, 9.0, 1e-12);
}

TEST_CASE("Dual comparison") {
    using D = utils2::Dual<double, 1>;
    D a(3.0);
    D b(5.0);
    REQUIRE(a < b);
    REQUIRE(a == D(3.0));
    REQUIRE(a != b);
}

TEST_CASE("Dual compound assignment") {
    using D = utils2::Dual<double, 1>;
    D x(2.0, 0);
    D c(3.0);

    x += c;
    REQUIRE_NEAR(x.value, 5.0, 1e-12);
    REQUIRE_NEAR(x.derivatives[0], 1.0, 1e-12);

    x -= c;
    REQUIRE_NEAR(x.value, 2.0, 1e-12);

    x *= c;
    REQUIRE_NEAR(x.value, 6.0, 1e-12);
    REQUIRE_NEAR(x.derivatives[0], 3.0, 1e-12);

    x /= c;
    REQUIRE_NEAR(x.value, 2.0, 1e-12);
    REQUIRE_NEAR(x.derivatives[0], 1.0, 1e-12);
}

TEST_CASE("Dual abs function") {
    using D = utils2::Dual<double, 1>;

    D pos(3.0, 0);
    auto r_pos = utils2::abs(pos);
    REQUIRE_NEAR(r_pos.value, 3.0, 1e-12);
    REQUIRE_NEAR(r_pos.derivatives[0], 1.0, 1e-12);

    D neg(-3.0, 0);
    auto r_neg = utils2::abs(neg);
    REQUIRE_NEAR(r_neg.value, 3.0, 1e-12);
    REQUIRE_NEAR(r_neg.derivatives[0], -1.0, 1e-12);
}

TEST_CASE("Dual pow function") {
    using D = utils2::Dual<double, 1>;

    D base(2.0, 0);
    D exponent(3.0);  // constant

    auto r = utils2::pow(base, exponent);
    REQUIRE_NEAR(r.value, 8.0, 1e-10);
    // d/dx (x^3) = 3*x^2 = 12.0
    REQUIRE_NEAR(r.derivatives[0], 12.0, 1e-10);
}

TEST_CASE("Dual atan2 function") {
    using D = utils2::Dual<double, 1>;

    D y(1.0, 0);
    D x(1.0);  // constant

    auto r = utils2::atan2(y, x);
    REQUIRE_NEAR(r.value, std::atan2(1.0, 1.0), 1e-12);
    // d/dy atan2(y, x) = x / (x^2 + y^2) = 1/2
    REQUIRE_NEAR(r.derivatives[0], 0.5, 1e-12);
}

TEST_CASE("Dual multivariable chain rule") {
    using D = utils2::Dual<double, 2>;

    // f(x,y) = sin(x) * exp(y)
    // df/dx = cos(x) * exp(y)
    // df/dy = sin(x) * exp(y)
    D x(1.0, 0);
    D y(0.5, 1);

    auto result = utils2::sin(x) * utils2::exp(y);
    REQUIRE_NEAR(result.value, std::sin(1.0) * std::exp(0.5), 1e-12);
    REQUIRE_NEAR(result.derivatives[0], std::cos(1.0) * std::exp(0.5), 1e-12);
    REQUIRE_NEAR(result.derivatives[1], std::sin(1.0) * std::exp(0.5), 1e-12);
}

TEST_CASE("RegularizationCost") {
    utils2::RegularizationCost<double> reg{.weight = 0.5};

    double params[] = {3.0, 4.0};
    double val = reg(params, 2);
    // 0.5 * (9 + 16) = 12.5
    REQUIRE_NEAR(val, 12.5, 1e-12);
}

TEST_CASE("DistanceCost with nonzero distance") {
    utils2::DistanceCost<double, 2> cost{.target = {1.0, 2.0}};

    double point[] = {4.0, 6.0};
    double val = cost(point);
    // (4-1)^2 + (6-2)^2 = 9 + 16 = 25
    REQUIRE_NEAR(val, 25.0, 1e-12);
}

TEST_CASE("Levenberg-Marquardt circle fit") {
    // Fit a circle: points on circle with center (2, 3) and radius 5
    // Parameters: cx, cy, r
    std::vector<double> xs, ys;
    constexpr double pi = std::numbers::pi;
    for (int i = 0; i < 8; ++i) {
        double angle = 2.0 * pi * i / 8.0;
        xs.push_back(2.0 + 5.0 * std::cos(angle));
        ys.push_back(3.0 + 5.0 * std::sin(angle));
    }

    auto residual_func = [&](std::span<const double> p) -> std::vector<double> {
        double cx = p[0], cy = p[1], r = p[2];
        std::vector<double> res(xs.size());
        for (std::size_t i = 0; i < xs.size(); ++i) {
            double dx = xs[i] - cx, dy = ys[i] - cy;
            res[i] = std::sqrt(dx * dx + dy * dy) - r;
        }
        return res;
    };

    auto result = utils2::levenberg_marquardt(
        residual_func, std::vector<double>{0.0, 0.0, 1.0}, 8);

    REQUIRE(result.converged);
    REQUIRE_NEAR(result.params[0], 2.0, 1e-3);
    REQUIRE_NEAR(result.params[1], 3.0, 1e-3);
    REQUIRE_NEAR(result.params[2], 5.0, 1e-3);
}

TEST_CASE("autodiff higher degree polynomial") {
    // f(x) = x^4 - 2x^2 + 1, df/dx = 4x^3 - 4x
    auto f = [](const auto& p) {
        auto x = p[0];
        return x * x * x * x - decltype(x)(2.0) * x * x + decltype(x)(1.0);
    };

    std::array<double, 1> params = {2.0};
    auto [val, grad] = utils2::autodiff<1>(f, params);

    REQUIRE_NEAR(val, 16.0 - 8.0 + 1.0, 1e-12);     // 9.0
    REQUIRE_NEAR(grad[0], 32.0 - 8.0, 1e-12);         // 24.0
}

TEST_CASE("Dual default construction") {
    using D = utils2::Dual<double, 3>;
    D d;
    REQUIRE_NEAR(d.value, 0.0, 1e-12);
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE_NEAR(d.derivatives[i], 0.0, 1e-12);
    }
}

UTILS2_TEST_MAIN()
