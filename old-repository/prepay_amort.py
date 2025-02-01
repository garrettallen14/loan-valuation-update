import numpy_financial as npf
from pyxirr import xirr
import numpy as np
import pandas as pd
from dates import get_monthly_dates, get_days_between
from pristine_amort import get_amortization_factor
from pandas import Timestamp

class PrepaymentVectorHandler:
    def __init__(self):
        pass
        
    def process_vector(self, vector_str, loan_age=0, original_term=None, rempmts=None):
        """
        Process a prepayment vector with complete context of the loan
        
        Args:
            vector_str: The vector as a string or scalar
            loan_age: Current age of loan in months (AOTRM - REMPMTS)
            original_term: Original term of the loan
            rempmts: Remaining payments
        """
        # First convert to array
        if isinstance(vector_str, np.ndarray):
            vector = vector_str.astype(float)
        elif isinstance(vector_str, str) and ' ' in vector_str:
            vector = np.array(vector_str.split(), dtype=float)
        else:
            vector = np.array([float(vector_str)])
            
        # Convert percentages if needed - only for CPR, CDR which are typically expressed as percentages
        # We can identify these by checking if any value is > 1 and < 100
        if np.any((vector > 1) & (vector < 100)):
            vector = vector / 100
            
        # Calculate the correct starting index based on loan age
        # loan_age represents how many months into the vector we should start
        start_index = max(0, original_term - rempmts if original_term and rempmts else loan_age)
        
        # Trim vector to start at the correct point
        if len(vector) > start_index:
            vector = vector[start_index:]
        else:
            # If we're beyond the vector length, use the last value
            vector = np.array([vector[-1]])
                
        # Extend if needed to match remaining term
        if rempmts:
            target_length = rempmts
            if len(vector) < target_length:
                vector = np.append(vector, [vector[-1]] * (target_length - len(vector)))
            else:
                vector = vector[:target_length]
                
        # Ensure at least two elements for calculations
        if len(vector) < 2:
            vector = np.full(max(2, rempmts or 2), vector[0])
            
        return vector
        
    def process_loan_vectors(self, cpr, cdr, severity, coupon_mult, loan_age, original_term, rempmts):
        """Process all vectors for a single loan"""
        cpr_vector = self.process_vector(cpr, loan_age, original_term, rempmts)
        cdr_vector = self.process_vector(cdr, loan_age, original_term, rempmts)
        severity_vector = self.process_vector(severity, loan_age, original_term, rempmts)
        coupon_mult_vector = self.process_vector(coupon_mult, loan_age, original_term, rempmts)
        return cpr_vector, cdr_vector, severity_vector, coupon_mult_vector

def xnpv(rate, values, dates):
    if rate <= -1.0:
        return float('inf')
    d0 = dates[0]    # or min(dates)
    return sum([ vi / (1.0 + rate)**((di - d0).days / 365.0) for vi, di in zip(values, dates)])

# Calculate and return the cash flows from a given loan
def get_cash_flows(apmt1, rempmts, original_balance, unpaid_balance, interest_rate, original_term, 
                  calculation_start_date, next_payment_date, cpr, cdr, severity, price, 
                  coupon_mult, original_yield=None, output=False, xirr_calc=False, index=None, 
                  arpay=-1, maturity_date=None, contract_date=None, reference_date=None, debug=False):
    try:
        # Store original term before modification for payment calculation
        orig_term_for_pmt = int(original_term)
        original_term = int(original_term) + 1
        rempmts = int(rempmts) + 1
        
        # Calculate loan age - simplified and more explicit calculation
        loan_age = 0
        if contract_date and reference_date:
            days_difference = (reference_date - contract_date).days
            loan_age = max(0, int(days_difference / 30.436875))
            
        # Process vectors
        vector_handler = PrepaymentVectorHandler()
        cpr, cdr, severity, coupon_mult = vector_handler.process_loan_vectors(
            cpr, cdr, severity, coupon_mult,
            loan_age=loan_age,
            original_term=original_term,
            rempmts=rempmts
        )
        
        # Calculate SMM and MDR
        smm = 1 - (1 - cpr) ** (1/12)
        mdr = 1 - (1 - cdr) ** (1/12)

        total_monthly_payment = apmt1

        # Calculate payment using original term (before +1 adjustment)
        if arpay == 0 and original_balance == unpaid_balance:
            total_monthly_payment = npf.pmt(interest_rate/12, orig_term_for_pmt, -original_balance)

        amortization_factor = get_amortization_factor(total_monthly_payment, original_balance, interest_rate, original_term)

        # Create arrays of zeros to store calculations
        upb = np.zeros(original_term)
        performing_upb = np.zeros(original_term)
        gross_charge_offs = np.zeros(original_term)
        net_loss = np.zeros(original_term)
        recoveries = np.zeros(original_term)
        scheduled_principal = np.zeros(original_term)
        voluntary_prepayments = np.zeros(original_term)
        principal_cash_flow = np.zeros(original_term)
        interest_cash_flow = np.zeros(original_term)
        lost_interest = np.zeros(original_term)

        # Create dates array
        dates = get_monthly_dates(calculation_start_date, next_payment_date, original_term)

        # Initializing month zero
        upb[0] = unpaid_balance
        gross_charge_offs[1] = max(0, upb[0] * mdr[1])
        performing_upb[0] = upb[0] - gross_charge_offs[1]

        # Each iteration will calculate the row values for the current month [i]
        for i in range(1, original_term):
            try:
                # This month's GCOs * this month's severity
                net_loss[i] = max(0, gross_charge_offs[i] * severity[i])

                # Safe division for voluntary prepayments with numerical stability check
                EPSILON = 1e-10  # Small threshold to prevent division by very small numbers
                if amortization_factor[i-1] > EPSILON:
                    voluntary_prepayments[i] = upb[i-1] * smm[i] * amortization_factor[i] / amortization_factor[i-1]
                else:
                    voluntary_prepayments[i] = 0

                # Safe division for scheduled principal with numerical stability check
                if amortization_factor[i-1] > EPSILON:
                    scheduled_principal[i] = (upb[i-1] - net_loss[i]) * (1 - (amortization_factor[i] / amortization_factor[i-1]))
                else:
                    scheduled_principal[i] = upb[i-1] - net_loss[i]  # If amortization factor is too small, assume full payment

                upb[i] = max(0, upb[i-1] - gross_charge_offs[i] - scheduled_principal[i] - voluntary_prepayments[i])

                if i+1 >= original_term:
                    gcos = 0
                else:
                    gross_charge_offs[i+1] = max(0, upb[i] * mdr[i+1])
                    gcos = gross_charge_offs[i+1]

                performing_upb[i] = max(0, performing_upb[i-1] - scheduled_principal[i] - voluntary_prepayments[i] - gcos)
                recoveries[i] = max(0, gross_charge_offs[i] - net_loss[i])
                principal_cash_flow[i] = scheduled_principal[i] + voluntary_prepayments[i] + recoveries[i]
                interest_cash_flow[i] = (performing_upb[i-1] * interest_rate * 
                    get_days_between(dates[i], dates[i-1]) / 365 * 
                    coupon_mult[i])
                lost_interest[i] = (upb[i-1] - performing_upb[i-1]) * interest_rate * get_days_between(dates[i], dates[i-1]) / 365

                if upb[i] <= 0 or amortization_factor[i] == 0:
                    break
            except Exception as e:
                if debug:
                    print(f"Error in iteration {i}: {str(e)}")
                break

        # Handle balloon payment case
        if arpay == 0 and maturity_date:
            # Calculate months until maturity using positive day count
            reference_date = reference_date or calculation_start_date
            days_to_maturity = (maturity_date - reference_date).days
            months_til_mat = min(len(dates), max(1, int(days_to_maturity / 30.436875)))

            # Initialize arrays for balloon payment scenario
            upb = [upb[0]] * months_til_mat
            performing_upb = [upb[0]] * months_til_mat
            gross_charge_offs = [0] * months_til_mat
            net_loss = [0] * months_til_mat
            recoveries = [0] * months_til_mat
            scheduled_principal = [0] * months_til_mat
            voluntary_prepayments = [0] * months_til_mat
            principal_cash_flow = np.array([0] * months_til_mat)
            interest_cash_flow = np.array([0] * months_til_mat)
            lost_interest = [0] * months_til_mat

            # Set balloon payment at maturity
            principal_cash_flow[months_til_mat-1] = upb[0]

            if len(dates[:months_til_mat]) != len(upb):
                error_msg = f"Array length mismatch: dates={len(dates[:months_til_mat])}, upb={len(upb)}"
                if debug:
                    print(error_msg)
                    return None, None, None
                else:
                    raise ValueError(error_msg)

            x = dates[:months_til_mat].copy()

            if output:
                df = pd.DataFrame({
                    'Date': x,
                    'UPB': upb,
                    'Performing UPB': performing_upb,
                    'Gross Charge Offs': gross_charge_offs,
                    'Net Loss': net_loss,
                    'Recoveries': recoveries,
                    'Scheduled Principal': scheduled_principal,
                    'Voluntary Prepayments': voluntary_prepayments,
                    'Principal Cash Flow': principal_cash_flow,
                    'Interest Cash Flow': interest_cash_flow,
                    # 'Lost Interest': lost_interest,
                    'Total Cash Flow': principal_cash_flow + interest_cash_flow
                })
                # print(df)
                return df, principal_cash_flow, interest_cash_flow


        # If the user wants to see the output, print a DataFrame of the entire thing
        if output:
            df = pd.DataFrame({
                'Date': dates,
                'UPB': upb,
                'Performing UPB': performing_upb,
                'Gross Charge Offs': gross_charge_offs,
                'Net Loss': net_loss,
                'Recoveries': recoveries,
                'Scheduled Principal': scheduled_principal,
                'Voluntary Prepayments': voluntary_prepayments,
                'Principal Cash Flow': principal_cash_flow,
                'Interest Cash Flow': interest_cash_flow,
                # 'Lost Interest': lost_interest,
                'Total Cash Flow': principal_cash_flow + interest_cash_flow
            })
            # print(df)
            return df, principal_cash_flow, interest_cash_flow
        
        elif xirr_calc:
            cash_flows = principal_cash_flow + interest_cash_flow
            cash_flows[0] = -original_balance * price/100
            yield_value = xirr(zip(dates, cash_flows))
            if debug:
                print(f"\nOriginal Balance: ${original_balance:,.2f}")
                print(f"Price: {price:.2f}")
                print(f"Purchase Yield: {yield_value*100:.2f}%")
                print(f"First 6 months cash flows:")
                for i in range(min(6, len(cash_flows))):
                    print(f"Month {i}: ${cash_flows[i]:,.2f}")
            return yield_value
        
        cash_flows = principal_cash_flow + interest_cash_flow
        npv = xnpv(original_yield, cash_flows, dates)
        
        if debug:
            print(f"\nNPV Calculation:")
            print(f"Original Yield Used: {original_yield*100:.2f}%")
            print(f"NPV: ${npv:,.2f}")
            print(f"New Price: {npv*100/unpaid_balance:.2f}")
            print(f"First 6 months cash flows:")
            for i in range(min(6, len(cash_flows))):
                print(f"Month {i}: ${cash_flows[i]:,.2f}")

        return npv, cpr, cdr, severity, coupon_mult

    except Exception as e:
        print(f"ERROR in get_cash_flows: {e}")
        print(f"Input parameters:")
        print(f"original_term: {original_term}, rempmts: {rempmts}")
        print(f"unpaid_balance: {unpaid_balance}")
        print(f"cpr: {cpr}, cdr: {cdr}, severity: {severity}, coupon_mult: {coupon_mult}")
        raise  # Re-raise the exception after printing details