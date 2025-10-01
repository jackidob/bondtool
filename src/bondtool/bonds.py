from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Bond:
    face: float              # Face value (par)
    coupon_rate: float       # Annual coupon rate as decimal (e.g., 0.05)
    price: float             # Clean price (assume no accrued for simplicity)
    ytm: Optional[float]     # Yield to maturity as decimal if known; otherwise None
    maturity_years: float    # Years to maturity
    freq: int = 2            # Coupon payments per year (2=semiannual, 1=annual, 4=quarterly)

    def cash_flows(self) -> List[float]:
        c = self.coupon_rate * self.face / self.freq
        n = int(round(self.maturity_years * self.freq))
        flows = [c] * n
        flows[-1] += self.face
        return flows

    def price_from_ytm(self, ytm: float) -> float:
        r = ytm / self.freq
        flows = self.cash_flows()
        return sum(cf / ((1 + r) ** (i + 1)) for i, cf in enumerate(flows))

    def ytm_from_price(self, guess: float = 0.05, tol: float = 1e-10, max_iter: int = 100) -> float:
        # Newton-Raphson on f(ytm) = price_from_ytm(ytm) - price
        y = guess
        for _ in range(max_iter):
            f = self.price_from_ytm(y) - self.price
            if abs(f) < tol:
                return y
            # derivative df/dy using analytical derivative of PV wrt y
            r = y / self.freq
            flows = self.cash_flows()
            df = sum(-(i+1) * cf / (self.freq) * (1 + r) ** (-(i+2)) for i, cf in enumerate(flows))
            if df == 0:
                break
            y_new = y - f / df
            # keep y in sensible bounds
            if y_new < -0.99:
                y_new = -0.99
            y = y_new
        return y  # return last iterate even if not converged

    def macaulay_duration(self, ytm: Optional[float] = None) -> float:
        y = self.ytm if ytm is None else ytm
        if y is None:
            y = self.ytm_from_price()
        r = y / self.freq
        flows = self.cash_flows()
        pv = 0.0
        wsum = 0.0
        for i, cf in enumerate(flows, start=1):
            disc = (1 + r) ** i
            pv_cf = cf / disc
            t_years = i / self.freq
            pv += pv_cf
            wsum += t_years * pv_cf
        return wsum / pv

    def modified_duration(self, ytm: Optional[float] = None) -> float:
        y = self.ytm if ytm is None else ytm
        if y is None:
            y = self.ytm_from_price()
        mac_dur = self.macaulay_duration(y)
        return mac_dur / (1 + y / self.freq)

    def convexity(self, ytm: Optional[float] = None) -> float:
        y = self.ytm if ytm is None else ytm
        if y is None:
            y = self.ytm_from_price()
        r = y / self.freq
        flows = self.cash_flows()
        convex = 0.0
        price = 0.0
        for i, cf in enumerate(flows, start=1):
            disc = (1 + r) ** i
            pv_cf = cf / disc
            price += pv_cf
            t = i / self.freq
            convex += pv_cf * t * (t + 1/self.freq)
        # Return annualized convexity (per 1% change, assuming freq compounding)
        return convex / (price * (1 + r)**2)
