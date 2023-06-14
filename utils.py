import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
import numpy as np


def dateVisualizer(df, dateType):
    details = ['_year', '_month', '_week', '_day', '_dayofweek']
    fig, axs = plt.subplots(len(details), 1, dpi=95, figsize=(15, 10))
    i = 0
    for col in details:
        col = dateType + col
        try:
            axs[i].boxplot(df[col], vert=False)
            axs[i].set_ylabel(col)
        except:
            df[col] = df[col].apply(mpl_dates.date2num)
            axs[i].boxplot(df[col], vert=False)
            axs[i].set_ylabel(col)
        i += 1
    plt.show()


def dataVisualizer(df):
    df_ignore = ['booking_no', 'booking_date',
                 'arrival_date', 'departure_date']
    fig, axs = plt.subplots(len(df.columns.values) -
                            len(df_ignore), 1, dpi=95, figsize=(15, 10))
    i = 0
    for col in df.columns:
        if col in df_ignore:
            continue
        try:
            axs[i].boxplot(df[col], vert=False)
            axs[i].set_ylabel(col)
        except:
            df[col] = df[col].apply(mpl_dates.date2num)
            axs[i].boxplot(df[col], vert=False)
            axs[i].set_ylabel(col)
        i += 1
    plt.show()


def dateParser(df, df_ignore):
    # new column for converting '2022-06-06' to '2022-06-06 01:00:00' in following for loop.
    # format has to be converted to access week of the year and day of week variables from datetime
    df['hour'] = 1

    for key in df_ignore:
        df[key] += pd.to_timedelta(df.hour, unit='h')
        # TODO: think of year parameter to be used.
        df[key + '_year'] = df[key].dt.year
        # df[key+'_month'] = df[key].dt.month
        df[key+'_mnth_sin'] = np.sin((df[key].dt.month-1)*(2.*np.pi/12))
        df[key+'_mnth_cos'] = np.cos((df[key].dt.month-1)*(2.*np.pi/12))
        df[key+'_week'] = df[key].dt.isocalendar().week
        # df[key+'_day'] = df[key].dt.day
        df[key+'_day_sin'] = np.sin((df[key].dt.day-1)*(2.*np.pi/31))
        df[key+'_day_cos'] = np.cos((df[key].dt.day-1)*(2.*np.pi/31))
        # TODO: think of using one hot encoding, sunday denoted as 6 monday-0.
        # this may trick model to assign higher priority ex: to sunday considered with saturday.  features = pd.get_dummies(features)
        # df[key+'_dayofweek'] = df[key].dt.dayofweek
        df[key +
            '_dayofweek_sin'] = np.sin((df[key].dt.dayofweek-1)*(2.*np.pi/7))
        df[key +
            '_dayofweek_cos'] = np.cos((df[key].dt.dayofweek-1)*(2.*np.pi/7))
    return df.drop(['hour'], axis=1)


def processData(projectDir):
    '''returns hotel data encoded'''

    df = pd.read_excel(os.path.join(projectDir, 'hotel.xlsx'))
    # drop empty or unused columns
    df = df.drop(['hotel_id', 'name', 'required_prepayment', 'repaid_amount', 'accommodation_property', 'payment_method',
                 'cancellation_date', 'status', 'extra_services', 'customer', 'gender', 'special_offer', 'agent_rate_plan', 'discounts'], axis=1)

    # encode string data to integer classes
    categories = ['point_of_sale', 'reference_source', 'room_type', 'country']
    for cat in categories:
        # factorize returns data and labels
        df[cat], _ = pd.factorize(df[cat])

    return df


def dataPreprocess(df):
    # check whether a null data or not
    is_null_or_not = df.isnull().sum()
    # statistical analysis: if returns -1 in any columns that means that cell is 0.
    df.describe()

    # Box plots for desired columns in one plot
    # ignores unreliable date data. if date data must be included, except block will be used
    # dataVisualizer(df)

    # adds year, month, week, day, week of the year, day of the week columns for each one of
    # ['booking_date', 'arrival_date', 'departure_date']
    df = dateParser(df, ['booking_date', 'arrival_date', 'departure_date'])

    # visualize booking year, month, week, day, week of the year, day of the week for
    # ['booking_date', 'arrival_date', 'departure_date']
    # dateVisualizer(df, 'booking_date')

    # TODO: This gives infinity error check, until then use total amount as average amount: calculating daily average as desired output
    # count nights spend as 1, if visit has same arrival and departure date
    for index, row in df.iterrows():
        if row['arrival_date'] == row['departure_date']:
            df.loc[index, 'night'] = 1

    df['average_amount'] = df['total_amount'] / df['night']

    df = df.drop(['booking_date', 'arrival_date',
                 'departure_date', 'booking_no'], axis=1)
    return df
    # return X_train, Y_train
