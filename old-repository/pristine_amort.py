import numpy_financial as npf
import numpy as np
from dates import get_monthly_dates, get_days_between
import pandas as pd

# Runs a Pristine Amortization, then returns the corresponding Amortization Factor array
# Error with unpaid_balance being the UPB, it should always just be opening_balance because its pristine and we start at the jawn
def get_amortization_factor(total_monthly_payment, opening_balance, interest_rate, original_term):
    
    # Create arrays of zeros to store calculations
    upb = np.zeros(original_term)
    interest = np.zeros(original_term)
    principal = np.zeros(original_term)
    amortization_factor = np.zeros(original_term)


    # Initialize the first month
    upb[0] = opening_balance
    amortization_factor[0] = 1
    # Calculate the interest and principal for each month
    for i in range(1, original_term):

        # Last month's unpaid balance / 365 * number of days between payments * ANNUAL interest rate
        interest[i] = upb[i-1]* interest_rate/12

        # The minimum of last month's unpaid balance versus the total monthly payment - current interest payment
        principal[i] = max(0, min(upb[i-1], total_monthly_payment - interest[i]))

        # Last month's unpaid balance - principal payment
        upb[i] = upb[i-1] - principal[i]

        # Current unpaid balance / our given starting unpaid balance
        amortization_factor[i] = upb[i] / opening_balance

        # If the unpaid balance is 0, we break the loop
        if upb[i-1] <= 0:
            break

    return amortization_factor