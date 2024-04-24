import tensorflow as tf
from tensorflow.keras.initializers import Orthogonal
import numpy as np
import time
from constants import DATA_CUR
from parse import find_rows_by_first_cell_value

def test_model(values: list, number: int or str):

    for folder, array in DATA_CUR.items():
        if str(number) in array:
            cur_folder = folder

    start_time = time.time()
    model_path = f'models/{cur_folder}/indicator{number}_model.h5'
    model = tf.keras.models.load_model(model_path, custom_objects={'Orthogonal': Orthogonal})

    data = np.array(values).reshape((-1, 1))

    response = model.predict(data, batch_size=64)
    print('-' * 10)
    print(f'Ответ модели №{number} (ЦУР: {cur_folder}): {response[-1][0]:.2f}')
    print(f'Было затрачено времени: {time.time() - start_time} секунд')
    print('-' * 10)

def main():
    file_name = 'data/data.xlsx'
    value = 'Ростовская область'
    data_cur = find_rows_by_first_cell_value(file_name, value)
    for item in data_cur:
        test_model(item['Values'], item['Indicator'])

if __name__ == "__main__":
    main()