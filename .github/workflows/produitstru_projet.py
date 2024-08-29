#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:20:25 2024

@author: macinfo
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st



######### Obligations à taux fixe 

class Bond: 
    def __init__(self,nominal,coupon_rate,maturity, risk_free_rate,freq): 
        self.nominal = nominal
        self.coupon_rate=coupon_rate
        self.maturity = maturity
        self.risk_free_rate = risk_free_rate
        self.freq = freq
        
    def rate_eq(self): 
        return (1 + self.risk_free_rate)**(1 / self.freq) - 1
        
    def bond_price(self):
        flux = [self.nominal*self.coupon_rate/self.freq/ (1 + self.rate_eq())**i for i in range(1, self.maturity * self.freq + 1)]
        flux[-1] += self.nominal  
        return sum(flux)
    
bond = Bond(nominal =100, coupon_rate = 0.05, maturity =7, risk_free_rate =0.03,freq=2)
price= bond.bond_price()
print('Le prix de lobligation est : ', price)

######## Options vanilles

class VanillaOption:
    def __init__(self, spot, strike, maturity, risk_free_rate, volatility, option_type='call',div=0):
        self.spot = spot
        self.strike = strike
        self.maturity = maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.option_type = option_type  # 'call' ou 'put'
        self.div = div
    
    def option_price(self):
        d1 = (np.log(self.spot / self.strike) + (self.risk_free_rate-self.div + 0.5 * self.volatility**2) * self.maturity) / (self.volatility * np.sqrt(self.maturity))
        d2 = d1 - self.volatility * np.sqrt(self.maturity)
        s = self.spot * np.exp(-self.div* self.maturity) 
        if self.option_type == 'call':
            price = (s * st.norm.cdf(d1) - self.strike * np.exp(-self.risk_free_rate * self.maturity) * st.norm.cdf(d2))
        else:  # 'put'
            price = (self.strike * np.exp(-self.risk_free_rate * self.maturity) * st.norm.cdf(-d2) - s * st.norm.cdf(-d1))
        return price
    
    def straddle_price(self):
        self.option_type = 'call'  # Assurer que le type d'option est 'call'
        call_price = self.option_price()  # Calculer le prix du call

        self.option_type = 'put'  # Changer le type d'option pour 'put'
        put_price = self.option_price()  # Calculer le prix du put
        return call_price + put_price 
    
    def strangle_price(self, strike_put, strike_call):
        # Pour un strangle, les strikes pour le call et le put sont différents
        self.strike = strike_put
        self.option_type = 'put'
        put_price = self.option_price()
        
        self.strike = strike_call
        self.option_type = 'call'
        call_price = self.option_price()
        
        return call_price + put_price
    
    def butterfly_price(self, strike_lower, strike_middle, strike_upper):
        # Pour un butterfly spread, on utilise trois strikes différents
        self.strike = strike_lower
        self.option_type = 'call'
        call_lower_price = self.option_price()
        
        self.strike = strike_middle
        call_middle_price = self.option_price()
        
        self.strike = strike_upper
        call_upper_price = self.option_price()
        
        return call_lower_price - 2 * call_middle_price + call_upper_price
    
option = VanillaOption(spot=100, strike=80, maturity=7, risk_free_rate=0.03, volatility=0.2, option_type='put', div=0.03)
price = option.option_price()

print('Le prix de loption est : ', price)

########Produits à stratégie optionnelle

# Straddle
straddle_price = option.straddle_price()
print('Le prix du Straddle est :', straddle_price)

# Strangle
strangle_price = option.strangle_price(strike_put=95, strike_call=105)
print('Le prix du Strangle est :', strangle_price)

# Butterfly Spread
butterfly_price = option.butterfly_price(strike_lower=95, strike_middle=100, strike_upper=105)
print('Le prix du Butterfly Spread est :', butterfly_price)


####### Options à barrière (KI, KO)

class BarrierOption(VanillaOption):
    def __init__(self, spot, strike, barrier, maturity, risk_free_rate, volatility, option_type='call', barrier_type='KO'):
        super().__init__(spot, strike, maturity, risk_free_rate, volatility, option_type)
        self.barrier = barrier
        self.barrier_type = barrier_type  # KO ou KI
        
    def simulate_paths(self, num_paths, num_steps):
        dt = self.maturity / num_steps
        s= self.spot * np.exp(-self.div* self.maturity)
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = s
        for t in range(1, num_steps + 1):
            z = np.random.standard_normal(num_paths)
            paths[:, t] = paths[:, t - 1] * np.exp((self.risk_free_rate - 0.5 * self.volatility**2) * dt + self.volatility * np.sqrt(dt) * z)
        return paths
    
    def option_price_monte_carlo(self, num_paths=10000, num_steps=252):
        paths = self.simulate_paths(num_paths, num_steps)
        payoffs = np.zeros(num_paths)

        for i in range(num_paths):
            path = paths[i]
            if self.barrier_type == 'KO' and np.any(path >= self.barrier):
                payoffs[i] = 0  # KO arrivé
            elif self.barrier_type == 'KI' and not np.any(path >= self.barrier):
                payoffs[i] = 0  # KI pas arrivé
            else:
                if self.option_type == 'call':
                    payoffs[i] = max(path[-1] - self.strike, 0)
                else:
                    payoffs[i] = max(self.strike - path[-1], 0)

        price = np.exp(-self.risk_free_rate * self.maturity) * np.mean(payoffs)
        return price

barrier_option = BarrierOption(spot=100, strike=100, barrier=110, maturity=1, risk_free_rate=0.05, volatility=0.2, option_type='call', barrier_type='KI')
barrier_option_price = barrier_option.option_price_monte_carlo()

print('Le prix de loption barrière est : ', barrier_option_price)


##### Options binaires

class BinaryOption:
    def __init__(self, spot, strike, maturity, risk_free_rate, volatility, payoff=1, option_type='call',div=0):
        self.spot = spot  
        self.strike = strike  
        self.maturity = maturity  
        self.risk_free_rate = risk_free_rate 
        self.volatility = volatility 
        self.payoff = payoff  # Le montant fixe à payer si l'option est dans la monnaie
        self.option_type = option_type  # 'call' ou 'put'
        self.div=div
    
    def d1(self):
        return (np.log(self.spot / self.strike) + (self.risk_free_rate + 0.5 * self.volatility**2) * self.maturity) / (self.volatility * np.sqrt(self.maturity))

    def d2(self):
        return self.d1() - self.volatility * np.sqrt(self.maturity)
    
    def option_price(self):
        d2 = self.d2()
        s=self.spot * np.exp(-self.div * self.maturity) 
        if self.option_type == 'call':
            price = np.exp(-self.risk_free_rate * self.maturity) * st.norm.cdf(d2) * self.payoff * s / self.spot
        else:  
            price = np.exp(-self.risk_free_rate * self.maturity) * st.norm.cdf(-d2) * self.payoff * s / self.spot
        return price

binary_call = BinaryOption(spot=100, strike=100, maturity=1, risk_free_rate=0.05, volatility=0.2, payoff=1, option_type='call',div=0.03)
binary_put = BinaryOption(spot=100, strike=100, maturity=1, risk_free_rate=0.05, volatility=0.2, payoff=1, option_type='put',div=0.03)

call_price = binary_call.option_price()
put_price = binary_put.option_price()

print(f"Prix de l'option binaire call : {call_price:.2f}")
print(f"Prix de l'option binaire put : {put_price:.2f}")
