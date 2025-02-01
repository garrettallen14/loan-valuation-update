from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# Function to get the exact number of days between two dates
def get_days_between(later_date, earlier_date):
    date_difference = later_date - earlier_date
    days_difference = date_difference.days

    return days_difference

# Function to get a list of months, going by payment date, not by the first of the month
def get_monthly_dates(calcuation_start_date, next_payment_date, amortization_period):
    
    # Ensure the dates are in the correct format
    if isinstance(calcuation_start_date, str):
        calcuation_start_date = datetime.strptime(calcuation_start_date, "%m/%d/%Y")
    if isinstance(next_payment_date, str):
        next_payment_date = datetime.strptime(next_payment_date, "%m/%d/%Y")
    
    # Calculate month prior to the next payment date
    dates = [calcuation_start_date, next_payment_date]
    for _ in range(2, amortization_period):
        next_payment_date += relativedelta(months=+1)
        dates.append(next_payment_date)


    return dates