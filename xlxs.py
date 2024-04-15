import pandas as pd

def read_xlsx_to_list():
    file_path = "diaodu.xlsx"
    data = pd.read_excel(file_path)
    return data.to_dict('records')

