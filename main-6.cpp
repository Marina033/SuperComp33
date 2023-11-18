#include <iostream>
#include <functional>
#include <cassert>
#include <cmath>
#include <omp.h>

#define MAX(a, b) (a >= b ? a : b)
#define NUM_THREADS 8


template<size_t M, size_t N>
struct Net {
    double A1;
    double B1;
    double A2;
    double B2;
    double h1;
    double h2;

    Net(double A1, double B1, double A2, double B2) : A1(A1), B1(B1), A2(A2), B2(B2), h1((B1 - A1) / M),
                                                      h2((B2 - A2) / N) {}

    [[nodiscard]] double x(double i) const {
        return A1 + h1 * i;
    }

    [[nodiscard]] double y(double j) const {
        return A2 + h2 * j;
    }
};

template<typename T, size_t M, size_t N>
class Matrix;

template<typename T, size_t M>
class Vector;

template<typename T, size_t M, size_t N>
class Matrix {
    T base[M][N];

public:
    Matrix() = default;

    static Matrix<T, M, N> Zero() {
        Matrix<T, M, N> res{};
        for (auto &row: res.base) {
            for (auto &elem: row) {
                elem = static_cast<T>(0);
            }
        }
        return res;
    }

    [[nodiscard]] T norm() const {
        T res = static_cast<T>(0);
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                res += base[i][j] * base[i][j];
            }
        }
        return std::sqrt(res);
    }

    [[nodiscard]] T squaredNorm() const {

        T res = static_cast<T>(0);
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                res += base[i][j] * base[i][j];
            }
        }
        return res;
    }

    [[nodiscard]] Vector<T, M * N> reshaped() const {
        Vector<T, M * N> res;
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                res(i * M + j) = (*this)(i, j);
            }
        }
        return res;
    }

    T &operator()(size_t i, size_t j) {
        return base[i][j];
    }

    const T &operator()(size_t i, size_t j) const {
        return base[i][j];
    }

    friend Matrix<T, M, N> operator+(const Matrix<T, M, N> &left, const Matrix<T, M, N> &right) {
        Matrix<T, M, N> res;
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                res(i, j) = left(i, j) + right(i, j);
            }
        }
        return res;
    }


    friend Matrix<T, M, N> operator-(const Matrix<T, M, N> &left, const Matrix<T, M, N> &right) {
        Matrix<T, M, N> res{};
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                res(i, j) = left(i, j) - right(i, j);
            }
        }
        return res;
    }

    friend Matrix<T, M, N> operator*(const Matrix<T, M, N> &left, T k) {
        Matrix<T, M, N> res;
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                res(i, j) = left(i, j) * k;
            }
        }
        return res;
    }

    friend Matrix<T, M, N> operator*(T k, const Matrix<T, M, N> &right) {
        Matrix<T, M, N> res{};
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                res(i, j) = right(i, j) * k;
            }
        }
        return res;
    }

    Matrix<T, M, N> operator-() const {
        Matrix<T, M, N> res{};
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                res(i, j) = -base[i][j];
            }
        }
        return res;
    }

    template<size_t K>
    friend Matrix<T, M, N> operator*(const Matrix<T, M, K> &left, const Matrix<T, K, N> &right) {
        Matrix<T, M, N> res = Matrix<T, M, N>::Zero();
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                for (size_t k = 0; k < K; ++k) {
                    res(i, j) += left(i, k) * right(k, j);
                }
            }
        }
        return res;
    }

    Matrix<T, M, N> &operator+=(const Matrix<T, M, N> &other) {
        return *this = *this + other;
    }

    Matrix<T, M, N> &operator-=(const Matrix<T, M, N> &other) {
        return *this = *this - other;
    }

    Matrix<T, M, N> &operator*=(T k) {
        return *this = *this * k;
    }

    template<size_t K>
    Matrix<T, M, K> &operator*=(const Matrix<T, N, K> &other) {
        return *this = *this * other;
    }

    friend std::ostream &operator<<(std::ostream &out, const Matrix<T, M, N> &matrix) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                out << matrix(i, j) << ' ';
            }
            out << std::endl;
        }
        return out;
    }
};

template<typename T, size_t M>
class Vector : public Matrix<T, M, 1> {
public:
    using Matrix<T, M, 1>::norm, Matrix<T, M, 1>::squaredNorm;

    T &operator()(size_t i) {
        return Matrix<T, M, 1>::operator()(i, 1);
    }

    const T &operator()(size_t i) const {
        return Matrix<T, M, 1>::operator()(i, 1);
    }

    template<size_t N>
    friend Vector<T, M> operator*(const Matrix<T, M, N> &left, const Vector<T, N> &right) {
        return left * static_cast<Matrix<T, N, 1>>(right);
    }

    T dot(const Vector<T, M> &other) {
        T res = static_cast<T>(0);
        for (size_t i = 0; i < M; ++i) {
            res += (*this)(i) * other(i);
        }
        return res;
    }
};

template<size_t M, size_t N>
Matrix<double, M, N>
least_residuals(std::function<Matrix<double, M, N>(Matrix<double, M, N>)> A,
                const Matrix<double, M, N> &B,
                double delta = 0.00001) {
    Matrix<double, M, N> w = Matrix<double, M, N>::Zero();
    Matrix<double, M, N> w_prev{};
    do {
        w_prev = w;
        Matrix<double, M, N> r = A(w) - B;
        if (r.norm() <= delta) {
            break;
        }
        auto tmp = A(r);
        double t = tmp.reshaped().dot(r.reshaped()) / tmp.squaredNorm();
        w -= t * r;
    } while ((w - w_prev).norm() >= delta);
    return w;
}

template<size_t M, size_t N>
Matrix<double, M, N> right_x_derivative(const Matrix<double, M, N> &w, double h) {
    Matrix<double, M, N> res = Matrix<double, M, N>::Zero();
    for (size_t i = 1; i < M - 1; ++i) {
        for (size_t j = 1; j < N - 1; ++j) {
            res(i, j) = (w(i + 1, j) - w(i, j)) / h;
        }
    }
    return res;
}

template<size_t M, size_t N>
Matrix<double, M, N> left_x_derivative(const Matrix<double, M, N> &w, double h) {
    Matrix<double, M, N> res = Matrix<double, M, N>::Zero();
    for (size_t i = 1; i < M - 1; ++i) {
        for (size_t j = 1; j < N - 1; ++j) {
            res(i, j) = (w(i, j) - w(i - 1, j)) / h;
        }
    }
    return res;
}

template<size_t M, size_t N>
Matrix<double, M, N> right_y_derivative(const Matrix<double, M, N> &w, double h) {
    Matrix<double, M, N> res = Matrix<double, M, N>::Zero();
    for (size_t i = 1; i < M - 1; ++i) {
        for (size_t j = 1; j < N - 1; ++j) {
            res(i, j) = (w(i, j + 1) - w(i, j)) / h;
        }
    }
    return res;
}

template<size_t M, size_t N>
Matrix<double, M, N> left_y_derivative(const Matrix<double, M, N> &w, double h) {
    Matrix<double, M, N> res = Matrix<double, M, N>::Zero();
    for (size_t i = 1; i < M - 1; ++i) {
        for (size_t j = 1; j < N - 1; ++j) {
            res(i, j) = (w(i, j) - w(i, j - 1)) / h;
        }
    }
    return res;
}

double integrate(const std::function<double(double)> &function, std::pair<double, double> xlim, double dx = 0.001) {
    double res = 0;
    for (int i = 0; i < (xlim.first - xlim.second) / dx; ++i) {
        double x = xlim.first + i * dx;
        res += function(x) * dx;
    }
    return res;
}

double integrate2D(const std::function<double(double, double)> &function,
                   std::pair<double, double> xlim,
                   std::pair<double, double> ylim,
                   double dx = 0.001,
                   double dy = 0.001) {
    double res = 0.0;
    for (int i = 0; i < (xlim.second - xlim.first) / dx; ++i) {
        double x = xlim.first + i * dx;
        for (int j = 0; j < (ylim.second - ylim.first) / dy; ++j) {
            double y = ylim.first + j * dy;
            res += function(x, y) * dx * dy;
        }
    }
    return res;
}

template<size_t M, size_t N>
Matrix<double, M, N> solve(const std::function<bool(double, double)> &D, double A1, double B1, double A2, double B2) {
    assert(B1 > A1 && B2 > A2);
    struct Net<M, N> net(A1, B1, A2, B2);
    double h = MAX(net.h1, net.h2);
    double eps = h * h;
    auto k = [eps, &D](double x, double y) { return D(x, y) ? 1 : eps; };
    Matrix<double, M, N> w = Matrix<double, M, N>::Zero();
    Matrix<double, M, N> a = Matrix<double, M, N>::Zero();
    Matrix<double, M, N> b = Matrix<double, M, N>::Zero();
#pragma opm parallel default(none) for
    for (int i = 1; i < M - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            a(i, j) = integrate([&net, i, &k](double t) { return k(net.x(i - 0.5), t); },
                                {net.y(j - 0.5), net.y(j + 0.5)}) / net.h2;
            b(i, j) = integrate([&net, j, &k](double t) { return k(t, net.y(j - 0.5)); },
                                {net.x(i - 0.5), net.x(i + 0.5)}) / net.h1;
        }
    }

    auto A = [&a, &b, &net](Matrix<double, M, N> w) {
        return -right_x_derivative<M, N>(a * left_x_derivative<M, N>(w, net.h1), net.h1) -
               right_y_derivative<M, N>(b * left_y_derivative<M, N>(w, net.h2), net.h2);
    };
    Matrix<double, M, N> F = Matrix<double, M, N>::Zero();
#pragma omp parallel default(none) shared(F, D, net)
    {
#pragma omp for
        for (int i = 1; i < M - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                F(i, j) =
                        integrate2D([&D](double x, double y) { return D(x, y) ? 1 : 0; },
                                    {net.x(i - 0.5), net.x(i + 0.5)},
                                    {net.y(j - 0.5), net.y(j + 0.5)}) / net.h1 / net.h2;
            }
        }
    }
    return least_residuals<M, N>(A, F);
}


int main() {
    /*
     *
     * Для запуска использовать функцию solve.
     * Аргументиы функции:
     * D - область. Возвращает true если точка x, y принадлежит боласти D.
     * A1, B1, A2, B2 - указанные в условии границы прямоугольника, находятся аналитически.
     * Возвращаемое значение - сетка M на N, являющаяся решением исходной задачи.
     *
     * */
    omp_set_dynamic(false);
    omp_set_num_threads(NUM_THREADS);
    const size_t M = 40, N = 40;
    std::cout << solve<M, N>([](double x, double y) { return x * x + 4 * y * y < 1; }, -1, 1, -0.5, 0.5); 
}
