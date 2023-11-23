#include <iostream>
#include <functional>
#include <cassert>
#include <cmath>
#include <omp.h>

#define MAX(a, b) (a >= b ? a : b)


template<size_t M, size_t N>
struct net
{
    double a1;
    double b1;
    double a2;
    double b2;
    double h1;
    double h2;

    net(const double a1, const double b1, const double a2, const double b2)
            : a1(a1),
              b1(b1),
              a2(a2),
              b2(b2),
              h1((b1 - a1) / M),
              h2((b2 - a2) / N)
    {}

    double
    x(const double i) const
    { return a1 + h1 * i; }

    double
    y(const double j) const
    { return a2 + h2 * j; }
};

template<typename T, size_t M, size_t N>
class matrix;

template<typename T, size_t M>
class vector;

template<typename T, size_t M, size_t N>
class matrix
{
    T base[M][N];

public:
    matrix() = default;

    static matrix
    zero()
    {
        matrix res{};
        for (size_t i = 0; i < M; ++i) { for (size_t j = 0; j < N; ++j) { res.base[i][j] = static_cast<T>(0); }}
        return res;
    }

    T
    norm() const
    {
        T res = static_cast<T>(0);
#pragma omp parallel for default(none) shared(res)
        for (size_t i = 0; i < M; ++i) { for (size_t j = 0; j < N; ++j) { res += base[i][j] * base[i][j]; }}
        return std::sqrt(res);
    }

    T
    squaredNorm() const
    {
        T res = static_cast<T>(0);
#pragma omp parallel for default(none) shared(res)
        for (size_t i = 0; i < M; ++i) { for (size_t j = 0; j < N; ++j) { res += base[i][j] * base[i][j]; }}
        return res;
    }

    vector<T, M * N>
    reshaped() const
    {
        vector<T, M * N> res;
#pragma omp parallel for default(none) shared(res)
        for (size_t i = 0; i < M; ++i) { for (size_t j = 0; j < N; ++j) { res(i * M + j) = (*this)(i, j); }}
        return res;
    }

    T &
    operator()(size_t i, size_t j)
    { return base[i][j]; }

    const T &
    operator()(size_t i, size_t j) const
    { return base[i][j]; }

    friend matrix
    operator+(const matrix &left, const matrix &right)
    {
        matrix res;
#pragma omp parallel for default(none) shared(res, left, right)
        for (size_t i = 0; i < M; ++i) { for (size_t j = 0; j < N; ++j) { res(i, j) = left(i, j) + right(i, j); }}
        return res;
    }


    friend matrix
    operator-(const matrix &left, const matrix &right)
    {
        matrix res{};
#pragma omp parallel for default(none) shared(res, left, right)
        for (size_t i = 0; i < M; ++i) { for (size_t j = 0; j < N; ++j) { res(i, j) = left(i, j) - right(i, j); }}
        return res;
    }

    friend matrix
    operator*(const matrix &left, T k)
    {
        matrix res;
#pragma omp parallel for default(none) shared(res, k, left)
        for (size_t i = 0; i < M; ++i) { for (size_t j = 0; j < N; ++j) { res(i, j) = left(i, j) * k; }}
        return res;
    }

    friend matrix
    operator*(T k, const matrix &right)
    {
        matrix res{};
#pragma omp parallel for default(none) shared(res, k, right)
        for (size_t i = 0; i < M; ++i) { for (size_t j = 0; j < N; ++j) { res(i, j) = right(i, j) * k; }}
        return res;
    }

    matrix
    operator-() const
    {
        matrix res{};
#pragma omp parallel for default(none) shared(res)
        for (size_t i = 0; i < M; ++i) { for (size_t j = 0; j < N; ++j) { res(i, j) = -base[i][j]; }}
        return res;
    }

    template<size_t K>
    friend matrix
    operator*(const matrix<T, M, K> &left, const matrix<T, K, N> &right)
    {
        matrix res = matrix::zero();
#pragma omp parallel for default(none) shared(res, left, right)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                for (size_t k = 0; k < K; ++k) {
                    res(i, j) += left(i, k) * right(k, j);
                }
            }
        }
        return res;
    }

    matrix &
    operator+=(const matrix &other)
    { return *this = *this + other; }

    matrix &
    operator-=(const matrix &other)
    { return *this = *this - other; }

    matrix &
    operator*=(T k)
    { return *this = *this * k; }

    template<size_t K>
    matrix<T, M, K> &
    operator*=(const matrix<T, N, K> &other)
    { return *this = *this * other; }

    friend std::ostream &
    operator<<(std::ostream &out, const matrix matrix)
    {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) { out << matrix(i, j) << ' '; }
            out << std::endl;
        }
        return out;
    }
};

template<typename T, size_t M>
class vector : public matrix<T, M, 1>
{
public:
    using matrix<T, M, 1>::norm;
    using matrix<T, M, 1>::squaredNorm;

    T &
    operator()(size_t i)
    { return matrix<T, M, 1>::operator()(i, 1); }

    const T &
    operator()(size_t i) const
    { return matrix<T, M, 1>::operator()(i, 1); }

    template<size_t N>
    friend vector
    operator*(const matrix<T, M, N> &left, const vector<T, N> &right)
    { return left * static_cast<matrix<T, N, 1>>(right); }

    T
    dot(const vector &other)
    {
        T res = static_cast<T>(0);
#pragma omp parallel for default(none) shared(res, other)
        for (size_t i = 0; i < M; ++i) { res += (*this)(i) * other(i); }
        return res;
    }
};

template<size_t M, size_t N>
matrix<double, M, N>
least_residuals(
        std::function<matrix<double, M, N>(matrix<double, M, N>)> A,
        const matrix<double, M, N> &b,
        double delta = 0.00001)
{
    matrix<double, M, N> w = matrix<double, M, N>::zero();
    matrix<double, M, N> wPrev{};
    do {
        wPrev = w;
        matrix<double, M, N> r = A(w) - b;
        if (r.norm() <= delta) { break; }
        auto tmp = A(r);
        double t = tmp.reshaped().dot(r.reshaped()) / tmp.squaredNorm();
        w -= t * r;
    } while ((w - wPrev).norm() >= delta);
    return w;
}

template<size_t M, size_t N>
matrix<double, M, N>
right_x_derivative(const matrix<double, M, N> &w, double h)
{
    matrix<double, M, N> res = matrix<double, M, N>::zero();
#pragma omp parallel for default(none) shared(res, w, h)
    for (size_t i = 1; i < M - 1; ++i) {
        for (size_t j = 1; j < N - 1; ++j) {
            res(i, j) = (w(i + 1, j) - w(i, j)) / h;
        }
    }
    return res;
}

template<size_t M, size_t N>
matrix<double, M, N>
left_x_derivative(const matrix<double, M, N> &w, double h)
{
    matrix<double, M, N> res = matrix<double, M, N>::zero();
#pragma omp parallel for default(none) shared(res, w, h)
    for (size_t i = 1; i < M - 1; ++i) {
        for (size_t j = 1; j < N - 1; ++j) {
            res(i, j) = (w(i, j) - w(i - 1, j)) / h;
        }
    }
    return res;
}

template<size_t M, size_t N>
matrix<double, M, N>
right_y_derivative(const matrix<double, M, N> &w, double h)
{
    matrix<double, M, N> res = matrix<double, M, N>::zero();
#pragma omp parallel for default(none) shared(res, w, h)
    for (size_t i = 1; i < M - 1; ++i) {
        for (size_t j = 1; j < N - 1; ++j) {
            res(i, j) = (w(i, j + 1) - w(i, j)) / h;
        }
    }
    return res;
}

template<size_t M, size_t N>
matrix<double, M, N>
left_y_derivative(const matrix<double, M, N> &w, double h)
{
    matrix<double, M, N> res = matrix<double, M, N>::zero();
#pragma omp parallel for default(none) shared(res, w, h)
    for (size_t i = 1; i < M - 1; ++i) {
        for (size_t j = 1; j < N - 1; ++j) {
            res(i, j) = (w(i, j) - w(i, j - 1)) / h;
        }
    }
    return res;
}

double
integrate(
        const std::function<double(double)> &function,
        const std::pair<double, double> &xlim,
        const double dx = 0.001)
{
    double res = 0;
    for (int i = 0; i < (xlim.second - xlim.first) / dx; ++i) {
        const double x = xlim.first + i * dx;
        res += function(x) * dx;
    }
    return res;
}

double
integrate2D(
        const std::function<double(double, double)> &function,
        const std::pair<double, double> &xlim,
        const std::pair<double, double> &ylim,
        const double dx = 0.001,
        const double dy = 0.001)
{
    double res = 0.0;
    for (int i = 0; i < (xlim.second - xlim.first) / dx; ++i) {
        double const x = xlim.first + i * dx;
        for (int j = 0; j < (ylim.second - ylim.first) / dy; ++j) {
            const double y = ylim.first + j * dy;
            res += function(x, y) * dx * dy;
        }
    }
    return res;
}

template<size_t M, size_t N>
std::pair<net<M, N>, matrix<double, M, N>>
solve(const std::function<bool(double, double)> &D, double A1, double B1, double A2, double B2)
{
    assert(B1 > A1 && B2 > A2);
    net<M, N> net(A1, B1, A2, B2);
    const double h = MAX(net.h1, net.h2);
    double eps = h * h;
    auto k = [eps, &D](const double x, const double y) { return D(x, y) ? 1 : eps; };
    matrix<double, M, N> w{};
    matrix<double, M, N> a = matrix<double, M, N>::zero();
    matrix<double, M, N> b = matrix<double, M, N>::zero();
#pragma omp parallel for default(none) shared(net, k, a, b)
    for (int i = 1; i < static_cast<int>(M) - 1; ++i) {
        for (int j = 1; j < static_cast<int>(N) - 1; ++j) {
            a(i, j) = integrate([&net, i, &k](const double t) { return k(net.x(i - 0.5), t); },
                                { net.y(j - 0.5), net.y(j + 0.5) }) / net.h2;
            b(i, j) = integrate([&net, j, &k](const double t) { return k(t, net.y(j - 0.5)); },
                                { net.x(i - 0.5), net.x(i + 0.5) }) / net.h1;
        }
    }
    auto A = [&a, &b, &net](matrix<double, M, N> x) {
        return -right_x_derivative<M, N>(a * left_x_derivative<M, N>(x, net.h1), net.h1) -
               right_y_derivative<M, N>(b * left_y_derivative<M, N>(x, net.h2), net.h2);
    };
    matrix<double, M, N> F = matrix<double, M, N>::zero();
#pragma omp parallel for default(none) shared(F, D, net)
    for (int i = 1; i < static_cast<int>(M) - 1; ++i) {
        for (int j = 1; j < static_cast<int>(N) - 1; ++j) {
            F(i, j) =
                    integrate2D([&D](const double x, const double y) { return D(x, y) ? 1 : 0; },
                                { net.x(i - 0.5), net.x(i + 0.5) },
                                { net.y(j - 0.5), net.y(j + 0.5) }) / net.h1 / net.h2;
        }
    }
    return { net, least_residuals<M, N>(A, F) };
}


int
main(int argc, char **argv)
{
    constexpr size_t m = 160, n = 160;
    constexpr size_t numThreads = 8;
    int size, rank = 0;
    omp_set_dynamic(false);
    omp_set_num_threads(numThreads);
    const auto solution = solve<m, n>([](const double x, const double y) { return x * x + 4 * y * y < 1; }, -1, 1, -0.5,
                                      0.5);
    if (rank == 0) {
        const auto net = solution.first;
        auto dots = solution.second;
        for (int i = 0; i < static_cast<int>(m); ++i) {
            for (int j = 0; j < static_cast<int>(n); ++j) { printf("(%lf,%lf,%lf) ", net.x(i), net.y(j), dots(i, j)); }
            printf("\n\n");
        }
    }
}
