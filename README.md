# Перемножение матриц

Программа состоит из двух основных частей: умножение матриц на CPU и умножение матриц на GPU с использованием CUDA.

## Умножение матриц на CPU
Функция multiplyCPU реализует умножение матриц на центральном процессоре (CPU).

Создается результирующая матрица C размером n x n.
В трех вложенных циклах происходит перемножение элементов матриц A и B, и результат записывается в матрицу C.

## Умножение матриц на GPU
Функция multiplyMatrixOnGPU реализует умножение матриц на графическом процессоре (GPU) с использованием CUDA.
-Выделяется память на GPU для матриц A, B и результирующей матрицы C.
-Данные матриц A и B копируются из CPU в GPU.
-Запускается CUDA Kernel multiplyGPU для выполнения умножения матриц.
-Результат копируется обратно из GPU в CPU.
-Освобождается память на GPU.

## Пример работы программы
![image](https://github.com/user-attachments/assets/4f160c72-681d-45c3-ba85-a81b7186b99b)

## Выводы

| Размер матриц | Время на CPU | Время на GPU |
| ----------- | ----------- |  ----------- |
| 100x100    | 0.0095655 | 1.34866 |
| 600x600    | 2.80474 | 0.0188052 |
| 1100x1100  | 14.6097 | 0.0931462 |
| 1600x1600  | 41.7976 | 0.416504 |

При небольших размерах матриц видно, что использование GPU не дает приемущества, скорее наоборот затраты слишком большие по сравнению с CPU.
Однако можно заметить, что с увеличением размера матриц выигрыш GPU растёт.
