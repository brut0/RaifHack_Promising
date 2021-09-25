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


def add_metro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание признака наличия метро
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    new_df = df.copy()
    metro_cities = ['Москва', 'Санкт-Петербург', 'Казань', 'Екатеринбург', 'Нижний Новгород', 'Новосибирск', 'Самара']
    # train
    is_metro = []
    for city in new_df['city']:
        if city in metro_cities:
            is_metro.append(1)
        else:
            is_metro.append(0)
    new_df['is_metro'] = is_metro

    new_osm_subway_closest_dist = []
    subway_500 = []
    subway_1000 = []
    subway_2000 = []
    for dist, metro_bool in zip(new_df['osm_subway_closest_dist'], new_df['is_metro']):
        if is_metro == 0:
            new_osm_subway_closest_dist.append(0)
        elif dist > 3:
            new_osm_subway_closest_dist.append(0.33)
        elif dist < 0.1:
            new_osm_subway_closest_dist.append(10)
        else:
            new_osm_subway_closest_dist.append(1 / dist)

        if dist < 0.5:
            subway_500.append(1)
            subway_1000.append(1)
            subway_2000.append(1)
        elif dist < 1:
            subway_500.append(0)
            subway_1000.append(1)
            subway_2000.append(1)
        elif dist < 2:
            subway_500.append(0)
            subway_1000.append(0)
            subway_2000.append(1)
        else:
            subway_500.append(0)
            subway_1000.append(0)
            subway_2000.append(0)

    new_df['osm_subway_closest_dist'] = new_osm_subway_closest_dist
    new_df['subway_500'] = subway_500
    new_df['subway_1000'] = subway_1000
    new_df['subway_2000'] = subway_2000

    return new_df

def change_region(df):
    new_df = df.copy()
    change_region_dict = {
        'Адыгея':'Республика Адыгея',
        'Татарстан':'Республика Татарстан',
        'Мордовия': 'Республика Мордовия',
        'Коми': 'Республика Коми',
        'Карелия': 'Республика Карелия',
        'Башкортостан': 'Республика Башкортостан',
        'Ханты-Мансийский АО':'Ханты-Мансийский автономный округ - Югра',
        'Удмуртия':'Удмуртская республика'
    }

    change_city_dict = {
        'Иркутский район, Маркова рп, Зеленый Берег мкр':'Маркова',
        'Иркутский район, Маркова рп, Стрижи кв-л':'Маркова',
        'город Светлый':'Светлый',
        'Орел':'Орёл'
    }


    def custom_func(region,city):
        if (region == 'Ленинградская область') and (city=='Санкт-Петербург'):
            return 'Санкт-Петербург'
        if (region == 'Тюменская область') and (city=='Нижневартовск'):
            return 'Ханты-Мансийский автономный округ - Югра'
        if (region == 'Тюменская область') and (city=='Сургут'):
            return 'Ханты-Мансийский автономный округ - Югра'
        return region

    for err, new in zip(list(change_city_dict.keys()),list(change_city_dict.values())):
        new_df.replace(err, new, inplace=True)

    for err, new in zip(list(change_region_dict.keys()),list(change_region_dict.values())):
        new_df.replace(err, new, inplace=True)
        new_df['region'] = new_df.apply(lambda x: custom_func(x['region'],x['city']),axis=1)

    return new_df
