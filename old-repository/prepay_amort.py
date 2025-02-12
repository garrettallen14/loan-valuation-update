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
        
    def process_vector(self, vector_str, loan_age=0, original_term=None, rempmts=None, is_amort_factor=False):
        """
        Process a prepayment vector with complete context of the loan
        
        Args:
            vector_str: The vector as a string or scalar
            loan_age: Current age of loan in months (AOTRM - REMPMTS)
            original_term: Original term of the loan
            rempmts: Remaining payments
            is_amort_factor: Whether this vector is an amortization factor
        """
        # print("\n=== Processing Vector ===")
        # print("Input Parameters:")
        # print(f"- Loan age: {loan_age}")
        # print(f"- Original term: {original_term}")
        # print(f"- Remaining payments: {rempmts}")
        # print(f"- Is amortization factor: {is_amort_factor}")
        
        # First convert to array
        # print("\nStep 1: Converting to array")
        # print(f"Input type: {type(vector_str)}")
        if isinstance(vector_str, np.ndarray):
            # print("Input is numpy array:")
            # print(f"Full vector: {vector_str}")
            vector = vector_str.astype(float)
        elif isinstance(vector_str, str) and ' ' in vector_str:
            # print("Input is space-separated string:")
            # print(f"Raw string: {vector_str}")
            vector = np.array(vector_str.split(), dtype=float)
        else:
            # print("Input is scalar:")
            # print(f"Value: {vector_str}")
            vector = np.array([float(vector_str)])
            
        # print(f"After conversion - length: {len(vector)}")
        # print(f"Full vector after conversion: {vector}")
            
        # Convert percentages if needed
        # print("\nStep 2: Percentage conversion check")
        # print(f"Vector range: min={np.min(vector):.4f}, max={np.max(vector):.4f}")
        if np.any((vector > 1) & (vector < 100)):
            # print("Converting from percentage")
            # print(f"After conversion: {vector}")
            vector = vector / 100
            
        # Calculate starting index
        # print("\nStep 3: Calculate starting index")
        if is_amort_factor:
            # For amortization factor with aged loan, start one period later
            start_index = max(0, (original_term - rempmts + 2) if original_term and rempmts else (loan_age + 1))
            # print(f"Amortization factor with loan_age > 0, adding 1 to start_index")
        else:
            start_index = max(0, original_term - rempmts if original_term and rempmts else loan_age)
            
        # print(f"Method 1: original_term - rempmts = {original_term - rempmts if original_term and rempmts else 'N/A'}")
        # print(f"Method 2: loan_age = {loan_age}")
        # print(f"Selected start_index = {start_index}")

        # Trim vector
        # print("\nStep 4: Vector trimming")
        # print(f"Current vector: {vector}")
        if len(vector) > start_index:
            # print(f"Trimming {start_index} elements from start")
            # print(f"After trimming: {vector}")
            vector = vector[start_index:]
        else:
            # print(f"Vector too short ({len(vector)} < {start_index}), using last value")
            # print(f"New vector: {vector}")
            vector = np.array([vector[-1]])
                
        # Extend if needed
        # print("\nStep 5: Length adjustment")
        if rempmts:
            target_length = rempmts
            # print(f"Target length: {target_length}")
            # print(f"Current length: {len(vector)}")
            if len(vector) < target_length:
                # print(f"Extending with {target_length - len(vector)} copies of last value: {vector[-1]}")
                vector = np.append(vector, [vector[-1]] * (target_length - len(vector)))
            else:
                # print(f"Trimming to {target_length} elements")
                vector = vector[:target_length]
            # print(f"After length adjustment: {vector}")
                
        # Ensure minimum length
        # print("\nStep 6: Minimum length check")
        if len(vector) < 2:
            min_length = max(2, rempmts or 2)
            # print(f"Padding to minimum length {min_length}")
            # print(f"After padding: {vector}")
            vector = np.full(min_length, vector[0])
            
        # print("\nFinal Vector:")
        # print(f"Length: {len(vector)}")
        # print(f"Values: {vector}")
        # print("=== End Processing ===\n")
        return vector
        
    def process_loan_vectors(self, cpr, cdr, severity, coupon_mult, loan_age, original_term, rempmts):
        """Process all vectors for a single loan"""
        # print("\n=== Processing All Vectors ===")
        # print(f"Original term: {original_term}, Remaining payments: {rempmts}")
        is_aged_loan = original_term != rempmts
        
        # print(f"Processing CPR vector...")
        cpr_vector = self.process_vector(
            cpr, 
            loan_age=loan_age, 
            original_term=original_term, 
            rempmts=rempmts,
            is_amort_factor=is_aged_loan
        )
        
        # print(f"\nProcessing CDR vector...")
        cdr_vector = self.process_vector(
            cdr, 
            loan_age=loan_age, 
            original_term=original_term, 
            rempmts=rempmts,
            is_amort_factor=is_aged_loan
        )
        
        # print(f"\nProcessing Severity vector...")
        severity_vector = self.process_vector(
            severity, 
            loan_age=loan_age, 
            original_term=original_term, 
            rempmts=rempmts,
            is_amort_factor=is_aged_loan
        )
        
        # print(f"\nProcessing Coupon Multiplier vector...")
        coupon_mult_vector = self.process_vector(
            coupon_mult, 
            loan_age=loan_age, 
            original_term=original_term, 
            rempmts=rempmts,
            is_amort_factor=is_aged_loan
        )
        
        # print("\n=== Vector Alignment Check ===")
        # print(f"All vectors should have length {rempmts}")
        # print(f"CPR vector length: {len(cpr_vector)}")
        # print(f"CDR vector length: {len(cdr_vector)}")
        # print(f"Severity vector length: {len(severity_vector)}")
        # print(f"Coupon mult vector length: {len(coupon_mult_vector)}")
        
        # print("\nFirst 5 values of each vector:")
        # print(f"CPR:     {cpr_vector[:5]}")
        # print(f"CDR:     {cdr_vector[:5]}")
        # print(f"Severity:{severity_vector[:5]}")
        # print(f"Coupon:  {coupon_mult_vector[:5]}")
        # print("=== End Vector Processing ===\n")
        
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

        # Initialize total monthly payment
        total_monthly_payment = apmt1
            
        # Calculate payment using original term (before +1 adjustment)
        if arpay == 0 and original_balance == unpaid_balance:
            total_monthly_payment = npf.pmt(interest_rate/12, orig_term_for_pmt, -original_balance)

        # Get full amortization factor vector
        full_amort_factor = get_amortization_factor(total_monthly_payment, original_balance, interest_rate, original_term)
        # print("\n=== Initial Setup ===")
        # print(f"Input parameters:")
        # print(f"- Total monthly payment: {total_monthly_payment:.2f}")
        # print(f"- Original balance: {original_balance:.2f}")
        # print(f"- Interest rate: {interest_rate:.4f}")
        # print(f"- Original term: {original_term}")
        # print(f"- Loan age: {loan_age}")
        # print(f"- Remaining payments: {rempmts}")
        # print("\nFull amortization factor vector:")
        # print(f"Length: {len(full_amort_factor)}")
        # print(f"All values: {full_amort_factor}")
        # print("=== End Initial Setup ===\n")
        
        # Process vectors and trim them based on loan age
        vector_handler = PrepaymentVectorHandler()
        
        # print("\n=== Starting Vector Processing ===")
        # print("Initial vector states:")
        # print(f"CPR input: {cpr}")
        # print(f"CDR input: {cdr}")
        # print(f"Severity input: {severity}")
        # print(f"Coupon mult input: {coupon_mult}")
        # print(f"Original term: {original_term}")
        # print(f"Loan age: {loan_age}")
        # print(f"Remaining payments: {rempmts}")
        
        cpr, cdr, severity, coupon_mult = vector_handler.process_loan_vectors(
            cpr, cdr, severity, coupon_mult,
            loan_age=loan_age,
            original_term=original_term,
            rempmts=rempmts
        )

        # Trim amortization factor to match other vectors
        # print("\n=== Amortization Factor Processing ===")
        # print("Before trimming:")
        # print(f"Full amort factor length: {len(full_amort_factor)}")
        # print(f"Full amort factor values: {full_amort_factor}")
        
        # When loan is aged (original_term != rempmts), shift all vectors including amortization factor
        is_aged_loan = original_term != rempmts
        # print(f"Is aged loan: {is_aged_loan} (original_term: {original_term}, rempmts: {rempmts})")
        amortization_factor = vector_handler.process_vector(
            full_amort_factor,
            loan_age=loan_age,
            original_term=original_term,
            rempmts=rempmts,
            is_amort_factor=is_aged_loan
        )
        
        # print("\n=== Final Vector Length Check ===")
        # print(f"Expected length (rempmts): {rempmts}")
        # print(f"CPR length: {len(cpr)}")
        # print(f"CDR length: {len(cdr)}")
        # print(f"Severity length: {len(severity)}")
        # print(f"Coupon mult length: {len(coupon_mult)}")
        # print(f"Amortization factor length: {len(amortization_factor)}")
        
        # print("\nFirst 5 values of all vectors:")
        # print(f"CPR:      {cpr[:5]}")
        # print(f"CDR:      {cdr[:5]}")
        # print(f"Severity: {severity[:5]}")
        # print(f"Coupon:   {coupon_mult[:5]}")
        # print(f"Amort:    {amortization_factor[:5]}")
        # print("=== End Vector Setup ===\n")
        
        # Calculate SMM and MDR
        smm = 1 - (1 - cpr) ** (1/12)
        mdr = 1 - (1 - cdr) ** (1/12)

        # print("\n=== SMM and MDR Processing ===")
        # print("Original arrays:")
        # print(f"SMM: {smm}")
        # print(f"MDR: {mdr}")
        
        # Shift arrays one position forward, wrapping last value to first position
        smm = np.roll(smm, 1)
        mdr = np.roll(mdr, 1)
        
        # print("\nAfter rolling arrays:")
        # print(f"SMM: {smm}")
        # print(f"MDR: {mdr}")
        
        # Set first value to 0
        smm[0] = 0
        mdr[0] = 0
        
        # print("\nAfter setting first values to 0:")
        # print(f"SMM: {smm}")
        # print(f"MDR: {mdr}")
        # print("=== End SMM and MDR Processing ===\n")

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
        # print("\n=== Initial Month (0) Setup ===")
        # print(f"Initial UPB[0]: {upb[0]:.2f}")
        # print(f"Initial MDR[1]: {mdr[1]:.4f}")
        gross_charge_offs[1] = max(0, upb[0] * mdr[1])
        # print(f"Initial gross_charge_offs[1]: {gross_charge_offs[1]:.2f}")
        performing_upb[0] = upb[0] - gross_charge_offs[1]
        # print(f"Initial performing_upb[0]: {performing_upb[0]:.2f}")
        # print("=== End Initial Month Setup ===\n")

        # Each iteration will calculate the row values for the current month [i]
        for i in range(1, original_term):
            try:
                # print(f"\n=== Month {i} Calculations ===")
                # print(f"Previous month UPB[{i-1}]: {upb[i-1]:.2f}")
                # print(f"Previous month performing_upb[{i-1}]: {performing_upb[i-1]:.2f}")
                # print(f"Current amort_factor[{i}]: {amortization_factor[i]:.4f}")
                # print(f"Previous amort_factor[{i-1}]: {amortization_factor[i-1]:.4f}")
                # print(f"Current SMM[{i}]: {smm[i]:.4f}")
                # print(f"Current MDR[{i}]: {mdr[i]:.4f}")
                # print(f"Current severity[{i}]: {severity[i]:.4f}")

                # This month's GCOs * this month's severity
                net_loss[i] = max(0, gross_charge_offs[i] * severity[i])
                # print(f"\n1. Net Loss Calculation:")
                # print(f"   gross_charge_offs[{i}]: {gross_charge_offs[i]:.2f}")
                # print(f"   severity[{i}]: {severity[i]:.4f}")
                # print(f"   net_loss[{i}] = {net_loss[i]:.2f}")

                # Safe division for voluntary prepayments with numerical stability check
                EPSILON = 1e-10  # Small threshold to prevent division by very small numbers
                # print(f"\n2. Voluntary Prepayments Calculation:")
                # print(f"   upb[{i-1}]: {upb[i-1]:.2f}")
                # print(f"   smm[{i}]: {smm[i]:.4f}")
                # print(f"   amort_factor[{i}]: {amortization_factor[i]:.4f}")
                # print(f"   amort_factor[{i-1}]: {amortization_factor[i-1]:.4f}")
                
                if amortization_factor[i-1] > EPSILON:
                    voluntary_prepayments[i] = upb[i-1] * smm[i] * amortization_factor[i] / amortization_factor[i-1]
                    # print(f"   voluntary_prepayments[{i}] = {voluntary_prepayments[i]:.2f}")
                else:
                    voluntary_prepayments[i] = 0
                    # print(f"   amort_factor too small, voluntary_prepayments[{i}] = 0")

                # Safe division for scheduled principal with numerical stability check
                # print(f"\n3. Scheduled Principal Calculation:")
                # print(f"   upb[{i-1}]: {upb[i-1]:.2f}")
                # print(f"   net_loss[{i}]: {net_loss[i]:.2f}")
                # print(f"   amort_factor ratio: {amortization_factor[i]/amortization_factor[i-1]:.4f}")
                
                if amortization_factor[i-1] > EPSILON:
                    scheduled_principal[i] = (upb[i-1] - net_loss[i]) * (1 - (amortization_factor[i] / amortization_factor[i-1]))
                    # print(f"   scheduled_principal[{i}] = {scheduled_principal[i]:.2f}")
                else:
                    scheduled_principal[i] = upb[i-1] - net_loss[i]
                    # print(f"   amort_factor too small, full payment: {scheduled_principal[i]:.2f}")

                # Calculate new UPB
                # print(f"\n4. New UPB Calculation:")
                # print(f"   Previous UPB[{i-1}]: {upb[i-1]:.2f}")
                # print(f"   Previous performing_upb[{i-1}]: {performing_upb[i-1]:.2f}")
                # print(f"   Subtracting:")
                # print(f"   - gross_charge_offs[{i}]: {gross_charge_offs[i]:.2f} (from current month)")
                # print(f"   - scheduled_principal[{i}]: {scheduled_principal[i]:.2f}")
                # print(f"   - voluntary_prepayments[{i}]: {voluntary_prepayments[i]:.2f}")
                
                upb[i] = max(0, upb[i-1] - gross_charge_offs[i] - scheduled_principal[i] - voluntary_prepayments[i])
                # print(f"   New UPB[{i}] = {upb[i]:.2f}")

                # Calculate next month's gross charge offs
                # print(f"\n5. Next Month's Gross Charge Offs:")
                if i+1 >= original_term:
                    gcos = 0
                    # print(f"   At term end, gcos = 0")
                else:
                    # print(f"   Current UPB[{i}]: {upb[i]:.2f}")
                    # print(f"   Next month's MDR[{i+1}]: {mdr[i+1]:.4f}")
                    gross_charge_offs[i+1] = max(0, upb[i] * mdr[i+1])
                    gcos = gross_charge_offs[i+1]
                    # print(f"   Next month's gross_charge_offs[{i+1}] = {gross_charge_offs[i+1]:.2f}")
                    # print(f"   This means {gross_charge_offs[i+1]:.2f} will be charged off at the start of next month")

                # Calculate performing UPB and cash flows
                # print(f"\n6. Performing UPB and Cash Flows:")
                # print(f"   Previous performing_upb[{i-1}]: {performing_upb[i-1]:.2f}")
                # print(f"   Subtracting:")
                # print(f"   - scheduled_principal[{i}]: {scheduled_principal[i]:.2f}")
                # print(f"   - voluntary_prepayments[{i}]: {voluntary_prepayments[i]:.2f}")
                # print(f"   - next month's gcos: {gcos:.2f}")
                performing_upb[i] = max(0, performing_upb[i-1] - scheduled_principal[i] - voluntary_prepayments[i] - gcos)
                # print(f"   New performing_upb[{i}] = {performing_upb[i]:.2f}")
                recoveries[i] = max(0, gross_charge_offs[i] - net_loss[i])
                principal_cash_flow[i] = scheduled_principal[i] + voluntary_prepayments[i] + recoveries[i]
                
                # print("\n7. Interest Cash Flow Calculation:")
                days_between = get_days_between(dates[i], dates[i-1])
                # print(f"   Date range: {dates[i-1]} to {dates[i]}")
                # print(f"   Days between: {days_between}")
                # print(f"   Previous performing_upb[{i-1}]: {performing_upb[i-1]:.2f}")
                # print(f"   Annual interest rate: {interest_rate:.4f}")
                # print(f"   Coupon multiplier[{i}]: {coupon_mult[i]:.4f}")
                # print(f"   Calculation: {performing_upb[i-1]:.2f} * {interest_rate:.4f} * {days_between}/365 * {coupon_mult[i]:.4f}")
                interest_cash_flow[i] = (performing_upb[i-1] * interest_rate * days_between / 365 * coupon_mult[i])
                # print(f"   Interest cash flow[{i}] = {interest_cash_flow[i]:.2f}")
                
                # print(f"   recoveries[{i}] = {recoveries[i]:.2f}")
                # print(f"   principal_cash_flow[{i}] = {principal_cash_flow[i]:.2f}")
                # print(f"   interest_cash_flow[{i}] = {interest_cash_flow[i]:.2f}")
                # print(f"   total_cash_flow[{i}] = {principal_cash_flow[i] + interest_cash_flow[i]:.2f}")
                # print("=== End Month Calculations ===\n")

                if upb[i] <= 0 or amortization_factor[i] == 0:
                    # print(f"Breaking loop: UPB <= 0 or amort_factor = 0")
                    break
            except Exception as e:
                if debug:
                    # print(f"Error in iteration {i}: {str(e)}")
                    pass
                break

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
                # print(f"\nOriginal Balance: ${original_balance:,.2f}")
                # print(f"Price: {price:.2f}")
                # print(f"Purchase Yield: {yield_value*100:.2f}%")
                # print(f"First 6 months cash flows:")
                # for i in range(min(6, len(cash_flows))):
                #     print(f"Month {i}: ${cash_flows[i]:,.2f}")
                pass
            return yield_value
        
        cash_flows = principal_cash_flow + interest_cash_flow
        npv = xnpv(original_yield, cash_flows, dates)
        
        if debug:
            # print(f"\nNPV Calculation:")
            # print(f"Original Yield Used: {original_yield*100:.2f}%")
            # print(f"NPV: ${npv:,.2f}")
            # print(f"New Price: {npv*100/unpaid_balance:.2f}")
            # print(f"First 6 months cash flows:")
            # for i in range(min(6, len(cash_flows))):
            #     print(f"Month {i}: ${cash_flows[i]:,.2f}")
            pass

        return npv, cpr, cdr, severity, coupon_mult

    except Exception as e:
        # print(f"ERROR in get_cash_flows: {e}")
        # print(f"Input parameters:")
        # print(f"original_term: {original_term}, rempmts: {rempmts}")
        # print(f"unpaid_balance: {unpaid_balance}")
        # print(f"cpr: {cpr}, cdr: {cdr}, severity: {severity}, coupon_mult: {coupon_mult}")
        raise  # Re-raise the exception after printing details