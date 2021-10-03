import pandas as pd
import feature_preprocessor

def input_data_organizer(raw_data):

    xg_data = pd.DataFrame()
    xg_data = raw_data[['date','air_temp','wind_speed','wind_direction','rain','humidity','casi','rail_direction','rail_temp']]
    xg_data.date = pd.to_datetime(xg_data.date)

    # Greenwich time offset -9 / 그리니치 시간에 맞춰야 하기 때문에, 9시간 전으로 test셋을 변환한 후 TSI, azimuth, altitude 값을 구함
    # 천문우주지식정보 사이트에서 구한 값과 차이를 구할 수도 있음.
    gr_off = pd.DataFrame()
    gr_off['date'] = xg_data.date - pd.offsets.Hour(9)

    # Total Solar Incidence
    xg_data['TSI'] = gr_off['date'].map(lambda x: feature_preprocessor.TSI(x))
    # Azimuth, Altitude
    xg_data['azimuth'] = gr_off['date'].map(lambda x: feature_preprocessor.el_az_changer(x)[0])
    xg_data['altitude'] = gr_off['date'].map(lambda x: feature_preprocessor.el_az_changer(x)[1])

    print(xg_data.describe())

    return xg_data
