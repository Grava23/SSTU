import numpy as np
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed


def generate_linear_vector(N, min_val, max_val):
    return np.linspace(min_val, max_val, N)

def generate_reverse_linear_vector(N, min_val, max_val):
    return np.linspace(max_val, min_val, N)

def generate_sinusoidal_vector(N, min_val, max_val):
    delta_alpha = 2 * np.pi / N
    return ((max_val - min_val) / 2) * np.sin(np.arange(0, 2 * np.pi, delta_alpha)) + ((max_val + min_val) / 2)

def generate_matrix_A(N, min_val, max_val, method):
    if method == 'linear':
        return np.linspace(min_val, max_val, N**2).reshape(N, N)
    elif method == 'reverse_linear':
        return np.linspace(max_val, min_val, N**2).reshape(N, N)
    elif method == 'sinusoidal':
        delta_alpha = 2 * np.pi / N**2
        return ((max_val - min_val) / 2) * np.sin(np.arange(0, 2 * np.pi, delta_alpha)).reshape(N, N) + ((max_val + min_val) / 2)
    elif method == 'random':
        return np.random.uniform(min_val, max_val, (N, N))
    else:
        raise ValueError("Некорректный метод генерации")

def linear_transformation(A, B):
    return np.dot(A, B)

def calculate_vector_element(A, B, i):
    return np.dot(A[i], B)

def parallel_linear_transformation(A, B, M):
    N = len(B)
    results = [None] * N
    with ThreadPoolExecutor(max_workers=M) as executor:
        futures = [executor.submit(calculate_vector_element, A, B, i) for i in range(N)]
        for future in as_completed(futures):
            results[futures.index(future)] = future.result()
    return results


def main():
    N = int(input("Введите размерность N: "))
    M = int(input("Введите количество потоков M: "))
    A_min, A_max = -10, 0
    B_min, B_max = -10, 5

    # Выберите методы для генерации матрицы A и вектора B
    A_method = 'random'  # Например, 'linear', 'reverse_linear', 'sinusoidal', 'random'
    B_method = 'random'  # Например, 'linear', 'reverse_linear', 'sinusoidal', 'random'

    A = generate_matrix_A(N, A_min, A_max, A_method)
    if B_method == 'linear':
        B = generate_linear_vector(N, B_min, B_max)
    elif B_method == 'reverse_linear':
        B = generate_reverse_linear_vector(N, B_min, B_max)
    elif B_method == 'sinusoidal':
        B = generate_sinusoidal_vector(N, B_min, B_max)
    elif B_method == 'random':
        B = np.random.uniform(B_min, B_max, N)


    start_time = time.time()
    C = parallel_linear_transformation(A, B, M)
    end_time = time.time()

    print("Матрица A:")
    print(A)
    print("\nВектор B:")
    print(B)
    print("\nРаспределение элементов вектора C по потокам:")
    for i in range(N):
        print(f"Поток {i % M + 1}: C[{i}] = {C[i]}")
    print("\nРезультат (Вектор C):")
    print(C)
    print("\nВремя выполнения программы: {:.6f} секунд".format(end_time - start_time))
    print("\nГправитис Н.А.")

if __name__ == "__main__":
    main()
