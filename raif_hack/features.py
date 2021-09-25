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
        OUTSIDE = 'ouside'

        if isinstance(floor, str):
            splits = [x.strip() for x in floor.split(',') if x.strip() != '']
            if len(splits) == 1:
                if re.match("[0-9]\s*-\s*[0-9]", floor):
                    return MANY_FLOORS
                if re.match("[0-9]\s*-\s*\D", floor):
                    return floor.split('-')[0].strip()
                if "+" in floor:
                    return MANY_FLOORS
                if 'подва' in floor.lower() or 'цоколь' in floor.lower():
                    return UNDER
                if 'антресоль' in floor.lower() or 'чердак' in floor.lower() or 'мансарда' in floor.lower():
                    return OUTSIDE
                if 'тех' in floor.lower():
                    return 'tech'
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


def add_economic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет экономические данные
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()
    df_eco_stats = pd.read_csv('../data/economic_stats_regions.csv', index_col=0)

    def merge_columns(df_source, df_dist, column):
        return df_dist.merge(df_source, on=column)

    df_new = merge_columns(df_eco_stats, df_new, 'region')

    return df_new.sort_values('id')


def prepare_square(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обработка square
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()

    def cat_square(total_square):
        if total_square < 20 and total_square > 0:
            return 'very small'
        elif total_square < 100 and total_square >= 20:
            return 'small'
        elif total_square < 300 and total_square >= 100:
            return 'medium'
        elif total_square < 1000 and total_square >= 300:
            return 'medium big'
        elif total_square < 5000 and total_square >= 1000:
            return 'big'
        elif total_square > 5000:
            return 'very big'
        else:
            return UNKNOWN_VALUE

    df_new['square_cat'] = df_new['total_square'].apply(cat_square)

    return df_new


def prepare_building(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обработка osm_building_points
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()

    df_new['building_cat'] = df_new.apply(lambda x: 1 if ((x['osm_building_points_in_0.001'] > 0)
                                                          & (x['osm_building_points_in_0.0075'] > 0)
                                                          & (x['osm_building_points_in_0.01'] > 1)) else 0, axis=1)

    return df_new


def prepare_amenity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обработка osm_amenity_points
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()

    df_new['amenity_cat'] = df_new.apply(lambda x: 1 if ((x['osm_amenity_points_in_0.001'] > 2)
                                                          & (x['osm_amenity_points_in_0.0075'] > 10)
                                                          & (x['osm_amenity_points_in_0.01'] > 20)) else 0, axis=1)

    return df_new

def prepare_historic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обработка osm_historic_points
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()

    df_new['historic_cat'] = df_new.apply(lambda x: 1 if ((x['osm_historic_points_in_0.005'] > 2)
                                                          & (x['osm_historic_points_in_0.0075'] > 3)
                                                          & (x['osm_historic_points_in_0.01'] > 10)) else 0, axis=1)

    return df_new