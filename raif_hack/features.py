import pandas as pd
import re
import numpy as np
from raif_hack.utils import UNKNOWN_VALUE

def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропущенные категориальные переменные
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()
    fillna_cols = ['region','city','street','realty_type', 'floor']
    df_new[fillna_cols] = df_new[fillna_cols].fillna(UNKNOWN_VALUE)
    return df_new


def prepare_floor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обработка
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()

    def parse_floor(floor):
        try:
            return str(int(float(floor)))
        except Exception as e:
            if isinstance(floor, str):
                under = 1
                if 'подвал' in floor or 'цоколь' in floor:
                    under = -1
                if re.findall('\d+', floor):
                    return int(re.findall('\d+', floor)[0]) * under
            elif np.isnan(floor):
                return "nan"

    def parse_floor_cat(floor):
        MANY_FLOORS = 'many'
        UNDER = 'underfloor'

        if isinstance(floor, str):
            splits = [x.strip() for x in floor.split(',') if x.strip() != '']
            if len(splits) == 1:
                if re.match("[0-9]\s*-\s*[0-9]", floor):
                    return MANY_FLOORS
                if re.match("[0-9]\s*-\s*\D", floor):
                    return floor.split('-')[0].strip()
                if "+" in floor:
                    return MANY_FLOORS
                if 'подвал' in floor.lower() or 'цоколь' in floor.lower():
                    return UNDER
                if re.findall('\d+', floor):
                    digit = int(re.findall('\d+', floor)[0])
                    if digit > 10:
                        return 'very high'
                    elif digit > 2:
                        return 'high'
                    elif digit < 0:
                        return UNDER
                    else:
                        return 'ground'
                return 'another'
            else:
                return MANY_FLOORS
        if np.isnan(floor):
            return "nan"

    df_new['floor_cat'] = df_new['floor'].apply(parse_floor_cat)
    df_new['floor'] = df_new['floor'].apply(parse_floor)

    return df_new