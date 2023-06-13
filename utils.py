import pandas as pd
import os 

def processData(projectDir):
    '''returns hotel data encoded'''
    
    df = pd.read_excel(os.path.join(projectDir, 'hotel.xlsx'))
    #drop empty or unused columns
    df = df.drop(['hotel_id','name', 'required_prepayment', 'repaid_amount', 'accommodation_property', 'payment_method', 'cancellation_date', 'status', 'extra_services', 'customer', 'gender', 'special_offer', 'agent_rate_plan', 'discounts'], axis=1)
    
    #encode string data to integer classes
    categories = ['point_of_sale', 'reference_source', 'room_type', 'country']
    for cat in categories:
        #factorize returns data and labels
        df[cat], _ = pd.factorize(df[cat])

    return df


