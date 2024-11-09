# Перемножение матриц

Программа состоит из двух основных частей: умножение матриц на CPU и умножение матриц на GPU с использованием CUDA.

## Умножение матриц на CPU
Функция multiplyCPU реализует умножение матриц на центральном процессоре (CPU).

Создается результирующая матрица C размером n x n.
В трех вложенных циклах происходит перемножение элементов матриц A и B, и результат записывается в матрицу C.

## Умножение матриц на GPU
Функция multiplyMatrixOnGPU реализует умножение матриц на графическом процессоре (GPU) с использованием CUDA.
1. Выделяется память на GPU для матриц A, B и результирующей матрицы C.
2. Данные матриц A и B копируются из CPU в GPU.
3. Запускается CUDA Kernel multiplyGPU для выполнения умножения матриц.
4. Результат копируется обратно из GPU в CPU.
5. Освобождается память на GPU.

## Выводы

| Размер матриц | Время на CPU | Время на GPU |
| ----------- | ----------- |  ----------- |
| 100x100    | 0.0095655 | 1.34866 |
| 600x600    | 2.80474 | 0.0188052 |
| 1100x1100  | 14.6097 | 0.0931462 |
| 1600x1600  | 41.7976 | 0.416504 |

При небольших размерах матриц видно, что использование GPU не дает приемущества, скорее наоборот затраты слишком большие по сравнению с CPU.
Однако можно заметить, что с увеличением размера матриц выигрыш GPU растёт.


# Сумма элементов вектора

Программа вычисляет сумму элементов массива, используя CUDA для распараллеливания вычислений на GPU, и сравнивает результаты с последовательным вычислением на CPU.

## Распараллеливание: 
Вычисление суммы элементов массива распараллеливается на GPU с использованием CUDA.
Каждый блок содержит несколько потоков (у меня 1024 потока на блок).
Каждый поток обрабатывает один элемент массива.

## Пример работы программы
![image](https://github.com/user-attachments/assets/0dc7908c-939d-44b5-b84c-c1dc25f6f926)


## Выводы

| Количество элементов | Скорость на GPU (секунды) | Скорость на CPU (секунды) |
|----------------------|---------------------------|---------------------------|
| 1000                | 0.229115                   | 2.9e-06                    |
| 10000               | 0.0010346                  | 2.4e-05                    |
| 100000              | 0.0010451                  | 0.0002614                  |
| 1000000            | 0.0051775                  | 0.0027249                  |
| 10000000           | 0.0231228                  | 0.0241502                  |
| 100000000          | 0.226132                   | 0.286698                   |

Видно, что при небольшом количестве элементов вектора использование GPU не эффективно, при больших объемах данных результат станогвится заметнее.
Также хотелось бы отметить, что использовался не самый мощный графический процессор и более продвинутый CPU, поэтому большого выигрыша в скорости GPU не дал.

# Соль и перец
Программа реализовывает CUDA-версию 9-точечного медианного фильтра. Результат записывается в output.bmp.
Для реализации использовалась библиотека с открытым исходным кодом EasyBMP.
## Пример работы программы
Изображение до:
![image](https://github.com/user-attachments/assets/c8695fa6-0383-4235-82c1-1025eb25362d)
Изображение после:
![image](https://github.com/user-attachments/assets/ce9e7356-383d-4de3-a51e-f9b710f516fb)

# Алгоритм Харриса

