import pandas as pd 
import numpy as np 

class Cleaner: 

    def renameColumns(self, df):
        df.columns = df.columns.str.replace(r'\d+.', '')
        return df

    #Set Data Types function accordingly
    def set_dtypes(self, dataf):
        """Sets Data Type to specific columns"""
        dataf['Enroll Date'] = pd.to_datetime(dataf['Enroll Date'],infer_datetime_format=True)
        dataf['Exit Date'] = pd.to_datetime(dataf['Exit Date'],infer_datetime_format=True)
        dataf['CurrentDate'] = pd.to_datetime(dataf['CurrentDate'],infer_datetime_format=True)
        dataf['Date of First Contact (Beta)'] = pd.to_datetime(dataf['Date of First Contact (Beta)'],infer_datetime_format=True)
        dataf['Date of First ES Stay (Beta)'] = pd.to_datetime(dataf['Date of First ES Stay (Beta)'],infer_datetime_format=True)
        dataf['Date of Last Contact (Beta)'] = pd.to_datetime(dataf['Date of Last Contact (Beta)'],infer_datetime_format=True)
        dataf['Date of Last ES Stay (Beta)'] = pd.to_datetime(dataf['Date of Last ES Stay (Beta)'],infer_datetime_format=True)
        dataf['Engagement Date'] = pd.to_datetime(dataf['Engagement Date'],infer_datetime_format=True)
        dataf['Homeless Start Date'] = pd.to_datetime(dataf['Homeless Start Date'],infer_datetime_format=True)
        return dataf


    # Use apply to assign values in dataframe to categories

    def recategorization(self, dataf):
        values_dict = {
            # Permanent Exits
            'Staying or living with family, permanent tenure' : 'Permanent Exit',
            'Staying or living with friends, permanent tenure' : 'Permanent Exit',
            'Permanent housing (other than RRH) for formerly homeless persons' : 'Permanent Exit',
            'Rental by client with RRH or equivalent subsidy' : 'Permanent Exit',
            'Rental by client, no ongoing housing subsidy' : 'Permanent Exit',
            'Rental by client, other ongoing housing subsidy' : 'Permanent Exit',
            'Owned by client, no ongoing housing subsidy' : 'Permanent Exit',
            # Temporary Exits
            'Place not meant for habitation (e.g., a vehicle, an abandoned building, bus/train/subway station/airport or anywhere outside)' : 'Unknown/Other',
            'Staying or living with family, temporary tenure (e.g., room, apartment or house)' : 'Temporary Exit',
            'Staying or living with friends, temporary tenure (e.g., room, apartment or house)' : 'Temporary Exit',
            'Hotel or Motel paid for without Emergency Shelter Voucher' : 'Temporary Exit',
            # Emergency Shelter
            'Emergency shelter, including hotel or motel paid for with emergency shelter voucher, or RHY-funded Host Home shelter' : 'Emergency Shelter',
            # Transitional Housing
            'Transitional Housing for homeless persons (including homeless youth)' : 'Transitional Housing',
            'Safe Haven' : 'Transitional Housing',
            'Substance Abuse Treatment or Detox Center' : 'Transitional Housing',
            'Foster Care Home or Foster Care Group Home' : 'Transitional Housing',
            'Psychiatric Hospital or Other Psychiatric Facility' : 'Transitional Housing',
            # Unknown/Other
            'No exit interview completed' : 'Unknown/Other',
            'Client refused' : 'Unknown/Other',
            'Other' : 'Unknown/Other',
            'Client doesn\'t know' : 'Unknown/Other',
            np.NaN : 'Unknown/Other'
            }
        dataf['Recategorized'] = dataf['Exit Destination'].map(values_dict)
        return dataf


       