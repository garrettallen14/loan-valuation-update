"""Core calculation engine for loan valuation and cash flow projections.

This module contains the primary logic for calculating loan yields, prices, and cash flows.
It handles various loan types (amortizing and balloon) and incorporates prepayment vectors
(CPR, CDR, severity, coupon multipliers) into the calculations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, Union
import numpy as np
import numpy_financial as npf
import pandas as pd
from pyxirr import xirr

@dataclass
class PrepaymentVectors:
    """Container for prepayment and default vectors that affect cash flows.
    
    Attributes:
        cpr: Conditional Prepayment Rate - rate at which loans are expected to prepay
        cdr: Conditional Default Rate - rate at which loans are expected to default
        severity: Loss severity - percentage of defaulted balance that is lost
        coupon_mult: Coupon multiplier - adjusts interest payments over time
    """
    cpr: np.ndarray  # As decimal (e.g., 0.05 for 5%)
    cdr: np.ndarray  # As decimal
    severity: np.ndarray  # As decimal
    coupon_mult: np.ndarray  # Multiplier (e.g., 1.0 for no adjustment)

    @classmethod
    def from_strings(cls, cpr: str, cdr: str, severity: str, coupon_mult: str):
        """Create vectors from space-separated string inputs (e.g., "5.0 6.0 7.0")"""
        return cls(
            cpr=np.array(cpr.split(), dtype=float),
            cdr=np.array(cdr.split(), dtype=float),
            severity=np.array(severity.split(), dtype=float),
            coupon_mult=np.array(coupon_mult.split(), dtype=float)
        )

@dataclass
class LoanTerms:
    """Container for loan payment terms and balances.
    
    Attributes:
        monthly_payment: Fixed monthly payment amount
        remaining_payments: Number of payments left
        original_balance: Original loan amount
        current_balance: Current unpaid principal
        interest_rate: Annual interest rate as decimal
        original_term: Original number of payments
    """
    monthly_payment: float
    remaining_payments: int
    original_balance: float
    current_balance: float
    interest_rate: float  # As decimal (e.g., 0.05 for 5%)
    original_term: int

@dataclass
class LoanDates:
    """Container for important loan dates.
    
    Attributes:
        calculation_date: Date to start calculations from
        next_payment_date: Next scheduled payment date
        contract_date: Original loan start date
        maturity_date: Final payment due date
        reference_date: Date for age calculations (usually ABKDTDT)
    """
    calculation_date: datetime
    next_payment_date: datetime
    contract_date: Optional[datetime] = None
    maturity_date: Optional[datetime] = None
    reference_date: Optional[datetime] = None

class VectorProcessor:
    """Handles preparation and adjustment of prepayment vectors based on loan characteristics."""
    
    @staticmethod
    def process_vector(vector: np.ndarray, loan_age: int, term: int, remaining_term: int) -> np.ndarray:
        """Adjusts a vector based on loan age and remaining term.
        
        For current cash flows:
            1. Trims vector based on loan age (time since contract date)
            2. Extends vector if needed to cover remaining term
            3. Ensures at least 2 elements for interpolation
            
        For original cash flows (term == remaining_term):
            1. Uses full vector from start (loan_age = 0)
            2. Extends vector if needed to cover original term
            3. Ensures at least 2 elements for interpolation
        """

        # Convert from percentage to decimal if needed
        if np.any((vector > 1) & (vector < 100)):
            vector = vector / 100
            
        # Trim vector based on loan age - only for current cash flows
        if len(vector) > loan_age:
            vector = vector[loan_age:]
        else:
            # If we've gone past the end of the vector, use the last value
            vector = np.array([vector[-1]])
            
        # Extend vector if needed to cover the remaining term
        if len(vector) < remaining_term:
            vector = np.append(vector, [vector[-1]] * (remaining_term - len(vector)))
        else:
            vector = vector[:remaining_term]
                
        return vector if len(vector) >= 2 else np.full(max(2, remaining_term or 2), vector[0])

    def process_vectors(self, vectors: PrepaymentVectors, loan_age: int, term: int, remaining_term: int) -> PrepaymentVectors:
        """Process all prepayment vectors for a loan.
        
        For current cash flows:
            - Trim vectors based on loan age (time since contract date)
            - Extend to cover remaining term if needed
            
        For original cash flows (when calculating from contract date):
            - Use full vectors from the start
            - Extend to cover original term if needed
        """
        # For original cash flows (when calculating from contract date), use vectors from start
        # For current cash flows, use loan age to trim vectors
        is_original_cashflow = loan_age == 0 and term == remaining_term
        start_index = 0 if is_original_cashflow else loan_age
        
        return PrepaymentVectors(
            cpr=self.process_vector(vectors.cpr, start_index, term, remaining_term),
            cdr=self.process_vector(vectors.cdr, start_index, term, remaining_term),
            severity=self.process_vector(vectors.severity, start_index, term, remaining_term),
            coupon_mult=self.process_vector(vectors.coupon_mult, start_index, term, remaining_term)
        )

class CashFlowCalculator:
    """Main calculator for loan cash flows, yields, and prices."""
    
    def __init__(self):
        self.vector_processor = VectorProcessor()

    # change to first payment date
    def _calculate_loan_age(self, dates: LoanDates) -> int:
        """Calculate loan age based on reference date, always using contract date."""
        if dates.contract_date and dates.reference_date:
            days_difference = (dates.reference_date - dates.contract_date).days
            return max(0, int(days_difference / 30.436875))
        return 0

    def _get_monthly_dates(self, start_date: datetime, next_payment: datetime, term: int) -> List[datetime]:
        """Generate payment dates for the loan term, handling month-end dates correctly."""
        dates = []
        current_date = start_date
        
        # Generate dates for the full term (including start date)
        for _ in range(term):
            dates.append(current_date)
            
            # For the first iteration, jump to next_payment
            if current_date == start_date:
                current_date = next_payment
                continue
                
            # Calculate next month while handling year transitions
            year = current_date.year + ((current_date.month + 1) - 1) // 12
            month = ((current_date.month + 1) - 1) % 12 + 1
            
            # Handle month-end dates correctly
            if month == 12:
                next_month = datetime(year + 1, 1, 1)
            else:
                next_month = datetime(year, month + 1, 1)
            last_day = (next_month - pd.Timedelta(days=1)).day
            
            target_day = min(current_date.day, last_day)
            current_date = datetime(year, month, target_day)
            
        return dates

    def _get_days_between(self, date1: datetime, date2: datetime) -> int:
        """Calculate actual days between two dates for interest calculations."""
        return abs((date2 - date1).days)

    def _calculate_amortization_factor(self, payment: float, balance: float, rate: float, term: int) -> np.ndarray:
        """Calculate amortization factors by running a pristine amortization schedule.
        
        The amortization factor at each point represents the present value
        of remaining payments divided by the payment amount. We calculate this
        by running a clean amortization schedule without any prepayment vectors.
        """
        factors = np.zeros(term)
        monthly_rate = rate / 12
        
        # Handle edge case where payment is 0 or very small
        if payment < 1e-6:
            return factors
            
        # Initialize the pristine schedule
        pristine_upb = np.zeros(term)
        pristine_upb[0] = balance
        
        # Calculate pristine scheduled principal and interest
        for i in range(1, term):
            # Calculate interest portion
            interest = pristine_upb[i-1] * monthly_rate
            
            # Principal is payment minus interest
            principal = min(payment, pristine_upb[i-1] + interest)
            
            # Update UPB
            pristine_upb[i] = max(0, pristine_upb[i-1] + interest - principal)
            
            # Calculate factor as remaining balance / payment
            factors[i] = pristine_upb[i] / payment
            
        # First month factor is initial balance / payment
        factors[0] = balance / payment
            
        return factors

    def _calculate_monthly_payment(self, terms: LoanTerms) -> float:
        """Calculate the monthly payment for an amortizing loan."""
        # If payment is already provided and non-zero, use it
        if terms.monthly_payment > 0:
            return terms.monthly_payment
            
        # Otherwise calculate the amortizing payment
        return npf.pmt(terms.interest_rate/12, terms.original_term, -terms.original_balance)

    def calculate_yield(self, terms: LoanTerms, dates: LoanDates, vectors: PrepaymentVectors, price: float) -> float:
        """Calculate yield to maturity using internal rate of return.
        
        1. Projects cash flows
        2. Sets initial cash flow as negative purchase price
        3. Uses XIRR to calculate yield considering actual payment dates
        """
        df, principal_cf, interest_cf, _ = self.project_cashflows(terms, dates, vectors)
        cash_flows = principal_cf + interest_cf
        cash_flows[0] = -terms.original_balance * price/100  # Initial investment
        return xirr(zip(df['Date'], cash_flows))

    def calculate_price(self, terms: LoanTerms, dates: LoanDates, vectors: PrepaymentVectors, required_yield: float) -> Tuple[float, PrepaymentVectors]:
        """Calculate price given a required yield.
        
        1. Projects cash flows
        2. Discounts cash flows using required yield
        3. Returns NPV and processed vectors
        """
        df, principal_cf, interest_cf, processed_vectors = self.project_cashflows(terms, dates, vectors)
        cash_flows = principal_cf + interest_cf
        
        if required_yield <= -1.0:
            return float('inf'), vectors
            
        # Calculate NPV using actual days between payments
        d0 = df['Date'].iloc[0]
        npv = sum([cf / (1.0 + required_yield)**((d - d0).days / 365.0) 
                  for cf, d in zip(cash_flows, df['Date'])])
                  
        return npv, processed_vectors

    def project_cashflows(self, terms: LoanTerms, dates: LoanDates, vectors: PrepaymentVectors) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, PrepaymentVectors]:
        """
        Project detailed monthly cash flows.

        This is the core calculation engine that:
        1. Processes prepayment vectors
        2. Calculates monthly default and prepayment rates
        3. Projects principal and interest payments
        4. Handles defaults, recoveries, and losses
        5. Returns detailed cash flow projections
        """
        if terms.interest_rate == 0:
            terms.interest_rate = 0.2

        # Setup initial calculations
        term = terms.remaining_payments + 5  # Add extra months
        loan_age = self._calculate_loan_age(dates)

        # Process vectors to match loan characteristics
        processed_vectors = self.vector_processor.process_vectors(
            vectors, loan_age, term, terms.remaining_payments + 2
        )
        
        # Convert annual to monthly rates
        smm = 1 - (1 - processed_vectors.cpr) ** (1/12)  # Single Monthly Mortality (prepayment)
        mdr = 1 - (1 - processed_vectors.cdr) ** (1/12)  # Monthly Default Rate
        
        # Calculate monthly payment
        monthly_payment = self._calculate_monthly_payment(terms)
            
        # Get amortization schedule factors
        factors = self._calculate_amortization_factor(
            monthly_payment, 
            terms.original_balance, 
            terms.interest_rate, 
            term
        )
        
        # Initialize tracking arrays
        upb = np.zeros(term)
        performing_upb = np.zeros(term)
        gross_charge_offs = np.zeros(term)
        net_loss = np.zeros(term)
        recoveries = np.zeros(term)
        scheduled_principal = np.zeros(term)
        voluntary_prepayments = np.zeros(term)
        principal_cash_flow = np.zeros(term)
        interest_cash_flow = np.zeros(term)
        
        # Generate payment dates
        payment_dates = self._get_monthly_dates(
            dates.calculation_date, 
            dates.next_payment_date, 
            term
        )

        # Initialize first month
        upb[0] = terms.current_balance
        performing_upb[0] = terms.current_balance * (1-mdr[0])

        # Calculate monthly cash flows
        for i in range(1, term):
            try:
                # 1. Calculate interest (based on prior month's performing UPB)
                days_in_period = self._get_days_between(payment_dates[i-1], payment_dates[i])
                interest_cash_flow[i] = max(
                    0, 
                    performing_upb[i-1] 
                    * terms.interest_rate 
                    * days_in_period / 365 
                    * processed_vectors.coupon_mult[i]
                )

                # 2. Calculate defaults and recoveries
                gross_charge_offs[i] = max(0, upb[i-1] * mdr[i])
                net_loss[i] = max(0, gross_charge_offs[i] * processed_vectors.severity[i])
                recoveries[i] = max(0, gross_charge_offs[i] - net_loss[i])

                # 3. Calculate principal payments
                # Handle edge case where factors[i-1] is 0
                factor_ratio = factors[i]/factors[i-1] if factors[i-1] > 0 else 0
                voluntary_prepayments[i] = min(
                    max(0, upb[i-1]-gross_charge_offs[i]),
                    smm[i] * factor_ratio * upb[i-1]
                )
                
                # Calculate scheduled principal using factor ratio
                remaining_after_prepay = max(0, upb[i-1] - gross_charge_offs[i])
                scheduled_principal[i] = min(
                    remaining_after_prepay - voluntary_prepayments[i],
                    remaining_after_prepay * (1 - factor_ratio)
                )

                # 4. Update balances for next month
                upb[i] = max(
                    0, 
                    upb[i-1] 
                    - gross_charge_offs[i] 
                    - scheduled_principal[i] 
                    - voluntary_prepayments[i]
                )
                
                # Calculate performing UPB using next period's defaults
                next_mdr = mdr[i+1] if i+1 < len(mdr) else mdr[-1]
                next_gco = max(0, upb[i] * next_mdr) if upb[i] > 0 else 0
                performing_upb[i] = max(
                    0,
                    performing_upb[i-1]
                    - next_gco
                    - scheduled_principal[i]
                    - voluntary_prepayments[i]
                )
                
                # Principal cash flow includes recoveries
                principal_cash_flow[i] = scheduled_principal[i] + voluntary_prepayments[i] + recoveries[i]
                
                # Clean up any values less than $0.01
                if upb[i] < 0.01:
                    upb[i] = 0
                if performing_upb[i] < 0.01:
                    performing_upb[i] = 0
                if gross_charge_offs[i] < 0.01:
                    gross_charge_offs[i] = 0
                if net_loss[i] < 0.01:
                    net_loss[i] = 0
                if recoveries[i] < 0.01:
                    recoveries[i] = 0
                if scheduled_principal[i] < 0.01:
                    scheduled_principal[i] = 0
                if voluntary_prepayments[i] < 0.01:
                    voluntary_prepayments[i] = 0
                if principal_cash_flow[i] < 0.01:
                    principal_cash_flow[i] = 0
                if interest_cash_flow[i] < 0.01:
                    interest_cash_flow[i] = 0

            except Exception as e:
                if i < len(processed_vectors.severity):
                    print(f"Error in month {i}: {str(e)}")
                break
        
        # Create detailed cash flow DataFrame
        df = pd.DataFrame({
            'Date': payment_dates,
            'UPB': upb,
            'Performing UPB': performing_upb,
            'Gross Charge Offs': gross_charge_offs,
            'Net Loss': net_loss,
            'Recoveries': recoveries,
            'Scheduled Principal': scheduled_principal,
            'Voluntary Prepayments': voluntary_prepayments,
            'Principal Cash Flow': principal_cash_flow,
            'Interest Cash Flow': interest_cash_flow,
            'Total Cash Flow': principal_cash_flow + interest_cash_flow
        })
        
        return df, principal_cash_flow, interest_cash_flow, processed_vectors
