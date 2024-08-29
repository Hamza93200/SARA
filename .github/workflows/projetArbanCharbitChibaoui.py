#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:37:11 2024

@author: karenarban
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


######### Obligations à taux fixe 

class Bond:
    """
    Classe pour représenter une obligation avec des méthodes pour calculer son prix,
    sa durée, sa convexité, et la sensibilité de son prix aux changements de taux d'intérêt.
    """
    def __init__(self, nominal, coupon_rate, maturity, risk_free_rate, freq):
        """
        Initialise une nouvelle instance de l'obligation.
        
        :param nominal: Le montant nominal de l'obligation.
        :param coupon_rate: Le taux du coupon annuel de l'obligation.
        :param maturity: La maturité de l'obligation, en années.
        :param risk_free_rate: Le taux sans risque annuel.
        :param freq: La fréquence de paiement des coupons par an.
        """
        self.nominal = nominal
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self.risk_free_rate = risk_free_rate
        self.freq = freq

    def rate_eq(self):
        """
        Calcule le taux d'intérêt équivalent pour la périodicité des paiements.
        
        :return: Le taux d'intérêt équivalent.
        """
        return (1 + self.risk_free_rate)**(1 / self.freq) - 1
        
    def bond_price(self):
        """
        Calcule le prix de l'obligation.
        
        :return: Le prix de l'obligation.
        """
        flux = [self.nominal * self.coupon_rate / self.freq / (1 + self.rate_eq())**i for i in range(1, int(self.maturity * self.freq) + 1)]
        flux[-1] += self.nominal  
        return sum(flux)
    
    def duration_and_convexity(self):
        """
        Calcule la durée et la convexité de l'obligation.
        
        :return: Un tuple contenant la durée et la convexité de l'obligation.
        """
        price = self.bond_price()
        duration = sum([(i / self.freq) * (self.nominal * self.coupon_rate / self.freq) / (1 + self.rate_eq())**i for i in range(1, int(self.maturity * self.freq) + 1)]) / price
        duration += (self.maturity * self.nominal) / ((1 + self.rate_eq())**(self.maturity * self.freq) * price)
        convexity = sum([(i / self.freq) * (i / self.freq + 1) * (self.nominal * self.coupon_rate / self.freq) / (1 + self.rate_eq())**(i + 2) for i in range(1, int(self.maturity * self.freq) + 1)]) / price
        convexity += (self.maturity * (self.maturity + 1) * self.nominal) / ((1 + self.rate_eq())**(self.maturity * self.freq + 2) * price)
        return duration, convexity
    
    def modified_duration(self):
        """
        Calcule la durée modifiée de l'obligation.
        
        :return: La durée modifiée de l'obligation.
        """
        macaulay_duration, _ = self.duration_and_convexity()
        return macaulay_duration / (1 + self.rate_eq())

    def price_sensitivity(self):
        """
        Calcule la sensibilité du prix de l'obligation à un changement de 1% du taux d'intérêt.
        
        :return: La variation estimée du prix de l'obligation.
        """
        mod_duration = self.modified_duration()
        return -mod_duration * self.bond_price() * 0.01
    
    def display_bond_details(self):
        """
        Affiche les détails et les calculs relatifs à l'obligation.
        """
        price = self.bond_price()
        duration, convexity = self.duration_and_convexity()
        mod_duration = self.modified_duration()
        sensitivity = self.price_sensitivity()
    
        print(f"Le prix de l'obligation est : {price:.2f} €")
        print(f"La duration de Macaulay est : {duration:.2f} années")
        print(f"La convexité est : {convexity:.2f}")
        print(f"La duration modifiée est : {mod_duration:.2f}")
        print(f"La sensibilité du prix pour un changement de 1% du taux d'intérêt est : {sensitivity:.2f} €")

print("\n### Obligations à taux fixe ###\n")
# Création d'une instance de Bond
bond = Bond(nominal=100, coupon_rate=0.05, maturity=7, risk_free_rate=0.03, freq=2)

# Affichage des résultats en utilisant la méthode de la classe
bond.display_bond_details()



######## Options vanilles

class VanillaOption:
    """
    Classe pour représenter une option vanille et calculer ses propriétés.
    """
    def __init__(self, spot, strike, maturity, risk_free_rate, volatility, option_type='call', div=0):
        """
        Initialise une nouvelle option vanille.
        """
        self.spot = spot
        self.strike = strike
        self.maturity = maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.option_type = option_type
        self.div = div

    def d1(self):
        """
        Calcule et retourne d1 utilisé dans les formules de Black-Scholes.
        """
        return (np.log(self.spot / self.strike) + (self.risk_free_rate - self.div + 0.5 * self.volatility**2) * self.maturity) / (self.volatility * np.sqrt(self.maturity))

    def d2(self):
        """
        Calcule et retourne d2 utilisé dans les formules de Black-Scholes.
        """
        return self.d1() - self.volatility * np.sqrt(self.maturity)

    def option_price(self):
        """
        Calcule et retourne le prix de l'option.
        """
        d1, d2 = self.d1(), self.d2()
        s = self.spot * np.exp(-self.div * self.maturity)
        if self.option_type == 'call':
            return (s * st.norm.cdf(d1) - self.strike * np.exp(-self.risk_free_rate * self.maturity) * st.norm.cdf(d2))
        else:
            return (self.strike * np.exp(-self.risk_free_rate * self.maturity) * st.norm.cdf(-d2) - s * st.norm.cdf(-d1))

    def delta(self):
        """
        Calcule et retourne le delta de l'option.
        """
        d1 = self.d1()
        if self.option_type == 'call':
            return np.exp(-self.div * self.maturity) * st.norm.cdf(d1)
        else:
            return np.exp(-self.div * self.maturity) * (st.norm.cdf(d1) - 1)

    def prob_of_exercise(self):
        """
        Calcule et retourne la probabilité que l'option soit exercée.
        """
        d2 = self.d2()
        if self.option_type == 'call':
            return st.norm.cdf(d2)
        else:
            return st.norm.cdf(-d2)

    def plot_payoff_and_profit(self):
        """
        Trace le graphique de payoff et de profit de l'option à l'échéance.
        """
        option_price = self.option_price()
        S = np.linspace(0, 2 * self.strike, 100)
        payoff = np.maximum(S - self.strike, 0) if self.option_type == 'call' else np.maximum(self.strike - S, 0)
        profit = payoff - option_price

        plt.figure(figsize=(10, 6))
        plt.plot(S, payoff, label=f'{self.option_type.capitalize()} Option Payoff')
        plt.plot(S, profit, label=f'{self.option_type.capitalize()} Option Profit', linestyle='--')
        plt.xlabel('Prix de l\'actif sous-jacent')
        plt.ylabel('Payoff / Profit')
        plt.title(f'Option {self.option_type.capitalize()} Payoff and Profit at Maturity')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def straddle_price(self):
        """
        Calcule et retourne le prix total d'un straddle, qui est la somme des prix d'une option call
        et d'une option put avec les mêmes paramètres d'entrée.

        :return: Le prix total du straddle.
        """
        original_option_type = self.option_type  # Sauvegarder l'option_type original

        self.option_type = 'call'  # Calculer le prix du call
        call_price = self.option_price()

        self.option_type = 'put'  # Calculer le prix du put
        put_price = self.option_price()

        self.option_type = original_option_type  # Restaurer l'option_type original
        return call_price + put_price 

    def straddle_delta(self):
        """
        Calcule et retourne le delta total d'un straddle, qui est la somme des deltas d'une option call
        et d'une option put avec les mêmes paramètres d'entrée.

        :return: Le delta total du straddle.
        """
        original_option_type = self.option_type  # Sauvegarder l'option_type original

        self.option_type = 'call'  # Calculer le delta du call
        call_delta = self.delta()

        self.option_type = 'put'  # Calculer le delta du put
        put_delta = self.delta()

        self.option_type = original_option_type  # Restaurer l'option_type original
        return call_delta + put_delta
    
    def plot_straddle_payoff_and_profit(self):
        """
        Trace le graphique du payoff et du profit d'un straddle à l'échéance en fonction
        du prix de l'actif sous-jacent. Le straddle est composé d'une option call et d'une option put
        avec les mêmes paramètres d'entrée.
        """
        straddle_price = self.straddle_price()
        S = np.linspace(0.5 * self.strike, 1.5 * self.strike, 100)
        
        call_payoff = np.maximum(S - self.strike, 0)
        put_payoff = np.maximum(self.strike - S, 0)
        straddle_payoff = call_payoff + put_payoff
        straddle_profit = straddle_payoff - straddle_price
        
        plt.figure(figsize=(10, 6))
        plt.plot(S, straddle_payoff, label='Straddle Payoff', color='blue')
        plt.plot(S, straddle_profit, label='Straddle Profit', linestyle='--', color='red')
        plt.xlabel('Prix de l\'actif sous-jacent')
        plt.ylabel('Payoff / Profit')
        plt.title('Straddle Payoff and Profit at Maturity')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def strangle_price(self, strike_put, strike_call):
        """
        Calcule et retourne le prix total d'un strangle, qui est la somme des prix d'une option put et d'une
        option call avec différents prix d'exercice.

        :param strike_put: Le prix d'exercice de l'option put.
        :param strike_call: Le prix d'exercice de l'option call.
        :return: Le prix total du strangle.
        """
        original_strike = self.strike  # Sauvegarder le strike original
        
        self.strike = strike_put
        self.option_type = 'put'
        put_price = self.option_price()
        
        self.strike = strike_call
        self.option_type = 'call'
        call_price = self.option_price()
        
        self.strike = original_strike  # Restaurer le strike original
        return call_price + put_price
    
    def strangle_delta(self, strike_put, strike_call):
        """
        Calcule et retourne le delta total d'un strangle, qui est la somme des deltas d'une option put et d'une
        option call avec différents prix d'exercice.

        :param strike_put: Le prix d'exercice de l'option put.
        :param strike_call: Le prix d'exercice de l'option call.
        :return: Le delta total du strangle.
        """
        original_strike = self.strike  # Sauvegarder le strike original
        
        self.strike = strike_put
        self.option_type = 'put'
        put_delta = self.delta()

        self.strike = strike_call
        self.option_type = 'call'
        call_delta = self.delta()

        self.strike = original_strike  # Restaurer le strike original
        return call_delta + put_delta

    def plot_strangle_payoff_and_profit(self, strike_put, strike_call):
        """
        Trace le graphique du payoff et du profit d'un strangle à l'échéance en fonction
        du prix de l'actif sous-jacent. Le strangle est composé d'une option put et d'une option call
        avec différents prix d'exercice.

        :param strike_put: Le prix d'exercice de l'option put.
        :param strike_call: Le prix d'exercice de l'option call.
        """
        strangle_price = self.strangle_price(strike_put, strike_call)
        S = np.linspace(0.5 * strike_put, 1.5 * strike_call, 100)
        
        call_payoff = np.maximum(S - strike_call, 0)
        put_payoff = np.maximum(strike_put - S, 0)
        strangle_payoff = call_payoff + put_payoff
        strangle_profit = strangle_payoff - strangle_price
        
        plt.figure(figsize=(10, 6))
        plt.plot(S, strangle_payoff, label='Strangle Payoff', color='blue')
        plt.plot(S, strangle_profit, label='Strangle Profit', linestyle='--', color='red')
        plt.xlabel('Prix de l\'actif sous-jacent')
        plt.ylabel('Payoff / Profit')
        plt.title('Strangle Payoff and Profit at Maturity')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def butterfly_price(self, strike_lower, strike_middle, strike_upper):
        """
        Calcule et retourne le coût net d'une stratégie Butterfly Spread, 
        qui est construite en achetant un call avec un prix d'exercice inférieur, 
        en vendant deux calls avec un prix d'exercice moyen, et 
        en achetant un call avec un prix d'exercice supérieur.

        :param strike_lower: Le prix d'exercice de l'option call achetée avec le prix le plus bas.
        :param strike_middle: Le prix d'exercice des options call vendues.
        :param strike_upper: Le prix d'exercice de l'option call achetée avec le prix le plus élevé.
        :return: Le coût net de la stratégie Butterfly Spread.
        """
        original_strike = self.strike  # Sauvegarder le strike original

        self.strike = strike_lower
        call_lower_price = self.option_price()
        
        self.strike = strike_middle
        call_middle_price = self.option_price()
        
        self.strike = strike_upper
        call_upper_price = self.option_price()
        
        self.strike = original_strike  # Restaurer le strike original
        return call_lower_price - 2 * call_middle_price + call_upper_price
    
    def butterfly_delta(self, strike_lower, strike_middle, strike_upper):
        """
        Calcule et retourne le delta total d'une stratégie Butterfly Spread.

        :param strike_lower: Le prix d'exercice de l'option call achetée avec le prix le plus bas.
        :param strike_middle: Le prix d'exercice des options call vendues.
        :param strike_upper: Le prix d'exercice de l'option call achetée avec le prix le plus élevé.
        :return: Le delta total de la stratégie Butterfly Spread.
        """
        original_strike = self.strike  # Sauvegarder le strike original

        self.strike = strike_lower
        delta_lower = self.delta()
        
        self.strike = strike_middle
        delta_middle = self.delta() * 2
        
        self.strike = strike_upper
        delta_upper = self.delta()
        
        self.strike = original_strike  # Restaurer le strike original
        return delta_lower - delta_middle + delta_upper
    
    def plot_butterfly_payoff_and_profit(self, strike_lower, strike_middle, strike_upper):
        """
        Trace le graphique du payoff et du profit d'une stratégie Butterfly Spread à l'échéance,
        en fonction du prix de l'actif sous-jacent.

        :param strike_lower: Le prix d'exercice de l'option call achetée avec le prix le plus bas.
        :param strike_middle: Le prix d'exercice des options call vendues.
        :param strike_upper: Le prix d'exercice de l'option call achetée avec le prix le plus élevé.
        """
        butterfly_price = self.butterfly_price(strike_lower, strike_middle, strike_upper)
        S = np.linspace(0.5 * strike_lower, 1.5 * strike_upper, 100)
        
        call_lower_payoff = np.maximum(S - strike_lower, 0)
        call_middle_payoff = np.maximum(S - strike_middle, 0) * 2
        call_upper_payoff = np.maximum(S - strike_upper, 0)
        
        butterfly_payoff = call_lower_payoff - call_middle_payoff + call_upper_payoff
        butterfly_profit = butterfly_payoff - butterfly_price
        
        plt.figure(figsize=(10, 6))
        plt.plot(S, butterfly_payoff, label='Butterfly Payoff', color='blue')
        plt.plot(S, butterfly_profit, label='Butterfly Profit', linestyle='--', color='red')
        plt.xlabel('Prix de l\'actif sous-jacent')
        plt.ylabel('Payoff / Profit')
        plt.title('Butterfly Spread Payoff and Profit at Maturity')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def call_spread_price(self, strike_buy, strike_sell):
        """
        Calcule et retourne le coût net d'un Bull Call Spread.

        :param strike_buy: Le prix d'exercice de l'option call achetée.
        :param strike_sell: Le prix d'exercice de l'option call vendue.
        :return: Le coût net du Bull Call Spread.
        """
        original_strike = self.strike  # Sauvegarder le strike original

        self.strike = strike_buy
        self.option_type = 'call'
        call_buy_price = self.option_price()
        
        self.strike = strike_sell
        call_sell_price = self.option_price()
        
        self.strike = original_strike  # Restaurer le strike original
        return call_buy_price - call_sell_price
    
    def call_spread_delta(self, strike_buy, strike_sell):
        """
        Calcule le delta net d'un Bull Call Spread.

        :param strike_buy: Le prix d'exercice de l'option call achetée.
        :param strike_sell: Le prix d'exercice de l'option call vendue.
        :return: Le delta net du Bull Call Spread.
        """
        original_strike = self.strike  # Sauvegarder le strike original

        self.strike = strike_buy
        delta_buy = self.delta()
        
        self.strike = strike_sell
        delta_sell = self.delta()
        
        self.strike = original_strike  # Restaurer le strike original
        return delta_buy - delta_sell
    
    def plot_call_spread_payoff_and_profit(self, strike_buy, strike_sell):
        """
        Trace le graphique du payoff et du profit d'un Bull Call Spread à l'échéance,
        en fonction du prix de l'actif sous-jacent.

        :param strike_buy: Le prix d'exercice de l'option call achetée.
        :param strike_sell: Le prix d'exercice de l'option call vendue.
        """
        call_spread_price = self.call_spread_price(strike_buy, strike_sell)
        S = np.linspace(0.5 * strike_buy, 1.5 * strike_sell, 100)
        
        call_buy_payoff = np.maximum(S - strike_buy, 0)
        call_sell_payoff = -np.maximum(S - strike_sell, 0)  # Payoff négatif car l'option est vendue
        
        # Le payoff total du call spread est la somme des payoffs de l'achat et de la vente
        call_spread_payoff = call_buy_payoff + call_sell_payoff
        
        # Le profit est le payoff du spread
        call_spread_profit = call_spread_payoff - call_spread_price
        
        plt.figure(figsize=(10, 6))
        plt.plot(S, call_spread_payoff, label='Call Spread Payoff', color='blue')
        plt.plot(S, call_spread_profit, label='Call Spread Profit', linestyle='--', color='red')
        
        plt.xlabel('Prix de l\'actif sous-jacent')
        plt.ylabel('Payoff / Profit')
        plt.title('Call Spread Payoff and Profit at Maturity')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def put_spread_price(self, strike_sell, strike_buy):
        """
        Calcule et retourne le coût net d'un Bear Put Spread.

        :param strike_sell: Le prix d'exercice de l'option put vendue.
        :param strike_buy: Le prix d'exercice de l'option put achetée.
        :return: Le coût net du Bear Put Spread.
        """
        original_strike = self.strike  # Sauvegarder le strike original
        
        # Vendre un put à un prix d'exercice inférieur
        self.strike = strike_sell
        self.option_type = 'put'
        put_sell_price = self.option_price()
        
        # Acheter un put à un prix d'exercice supérieur
        self.strike = strike_buy
        put_buy_price = self.option_price()
        
        self.strike = original_strike  # Restaurer le strike original
        return put_buy_price - put_sell_price
    
    def put_spread_delta(self, strike_sell, strike_buy):
        """
        Calcule le delta net d'un Bear Put Spread.

        :param strike_sell: Le prix d'exercice de l'option put vendue.
        :param strike_buy: Le prix d'exercice de l'option put achetée.
        :return: Le delta net du Bear Put Spread.
        """
        original_strike = self.strike  # Sauvegarder le strike original
        
        # Delta pour l'option put achetée à un prix d'exercice supérieur
        self.strike = strike_buy
        delta_buy = self.delta()
        
        # Delta pour l'option put vendue à un prix d'exercice inférieur
        self.strike = strike_sell
        delta_sell = self.delta()
        
        self.strike = original_strike  # Restaurer le strike original
        return delta_buy - delta_sell

    def plot_put_spread_payoff_and_profit(self, strike_sell, strike_buy):
        """
        Trace le graphique du payoff et du profit d'un Bear Put Spread à l'échéance,
        en fonction du prix de l'actif sous-jacent.

        :param strike_sell: Le prix d'exercice de l'option put vendue.
        :param strike_buy: Le prix d'exercice de l'option put achetée.
        """
        put_spread_price = self.put_spread_price(strike_sell, strike_buy)
        S = np.linspace(0.5 * strike_sell, 1.5 * strike_buy, 100)
        
        put_buy_payoff = np.maximum(strike_buy - S, 0) - put_spread_price
        put_sell_payoff = -np.maximum(strike_sell - S, 0)  # Payoff négatif car l'option est vendue
        
        # Le payoff total du put spread est la somme des payoffs des options achetée et vendue
        put_spread_payoff = put_buy_payoff + put_sell_payoff
        
        # Le profit est le payoff du spread moins le coût initial pour établir le spread
        put_spread_profit = put_spread_payoff
        
        plt.figure(figsize=(10, 6))
        plt.plot(S, put_spread_payoff, label='Put Spread Payoff', color='blue')
        plt.plot(S, put_spread_profit, label='Put Spread Profit', linestyle='--', color='red')
        
        plt.xlabel('Prix de l\'actif sous-jacent')
        plt.ylabel('Payoff / Profit')
        plt.title('Put Spread Payoff and Profit at Maturity')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def strip_price(self):
        """
        Calcule et retourne le coût net d'un Strip, qui consiste en l'achat d'un call
        et de deux puts sur le même actif sous-jacent, avec les mêmes prix d'exercice
        et dates d'expiration.

        :return: Le coût net du Strip.
        """
        self.option_type = 'call'
        call_price = self.option_price()
        self.option_type = 'put'
        put_price = self.option_price()
        # Un Strip est composé de 1 call et 2 puts.
        return call_price + 2 * put_price
    
    def strip_delta(self):
        """
        Calcule le delta net d'un Strip, qui consiste en l'achat d'un call et de deux puts.

        :return: Le delta net du Strip, basé sur la combinaison du delta du call et de deux fois
        le delta du put.
        """
        self.option_type = 'call'
        delta_call = self.delta()
        self.option_type = 'put'
        delta_put = self.delta()
        # Le delta total du Strip est la somme du delta du call et de deux fois le delta du put.
        return delta_call + 2 * delta_put

    def plot_strip_payoff_and_profit(self):
        """
        Trace le graphique du payoff et du profit d'un Strip à l'échéance, en fonction
        du prix de l'actif sous-jacent.
        """
        strip_price = self.strip_price()
        S = np.linspace(0.5 * self.strike, 1.5 * self.strike, 100)
        
        call_payoff = np.maximum(S - self.strike, 0)
        put_payoff = np.maximum(self.strike - S, 0)
        
        # Le payoff total du Strip est le payoff du call plus deux fois le payoff du put.
        strip_payoff = call_payoff + 2 * put_payoff
        
        # Le profit est le payoff total moins le coût initial du Strip.
        strip_profit = strip_payoff - strip_price
        
        plt.figure(figsize=(10, 6))
        plt.plot(S, strip_payoff, label='Strip Payoff', color='blue')
        plt.plot(S, strip_profit, label='Strip Profit', linestyle='--', color='red')
        
        plt.xlabel('Prix de l\'actif sous-jacent')
        plt.ylabel('Payoff / Profit')
        plt.title('Strip Payoff and Profit at Maturity')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def strap_price(self):
        """
        Calcule et retourne le coût net d'un Strap, qui consiste en l'achat de deux calls
        et d'un put sur le même actif sous-jacent, avec les mêmes prix d'exercice
        et dates d'expiration.

        :return: Le coût net du Strap.
        """
        self.option_type = 'call'
        call_price = self.option_price()
        self.option_type = 'put'
        put_price = self.option_price()
        # Un Strap est composé de 2 calls et 1 put.
        return 2 * call_price + put_price
    
    def strap_delta(self):
        """
        Calcule le delta net d'un Strap, qui consiste en l'achat de deux calls et d'un put.

        :return: Le delta net du Strap, basé sur la combinaison de deux fois le delta du call et du delta du put.
        """
        self.option_type = 'call'
        delta_call = self.delta()
        self.option_type = 'put'
        delta_put = self.delta()
        # Le delta total du Strap est la somme de deux fois le delta du call et du delta du put.
        return 2 * delta_call + delta_put

    def plot_strap_payoff_and_profit(self):
        """
        Trace le graphique du payoff et du profit d'un Strap à l'échéance, en fonction
        du prix de l'actif sous-jacent.
        """
        strap_price = self.strap_price()
        S = np.linspace(0.5 * self.strike, 1.5 * self.strike, 100)
        
        call_payoff = np.maximum(S - self.strike, 0)
        put_payoff = np.maximum(self.strike - S, 0)
        
        # Le payoff total du Strap est deux fois le payoff des calls plus le payoff du put.
        strap_payoff = 2 * call_payoff + put_payoff
        
        # Le profit est le payoff total moins le coût initial du Strap.
        strap_profit = strap_payoff - strap_price
        
        plt.figure(figsize=(10, 6))
        plt.plot(S, strap_payoff, label='Strap Payoff', color='blue')
        plt.plot(S, strap_profit, label='Strap Profit', linestyle='--', color='red')
        
        plt.xlabel('Prix de l\'actif sous-jacent')
        plt.ylabel('Payoff / Profit')
        plt.title('Strap Payoff and Profit at Maturity')
        plt.legend()
        plt.grid(True)
        plt.show()

    
def display_option_details(description, price, delta, probability=None):
    print(f"{description} - Prix: {price:.2f}, Delta: {delta:.2f}", end="")
    if probability is not None:
        print(f", Probabilité d'exercice: {probability:.2f}")
    else:
        print()  # Nouvelle ligne pour les cas sans probabilité d'exercice

# Création d'une instance d'option
option = VanillaOption(spot=100, strike=80, maturity=7, risk_free_rate=0.03, volatility=0.2, option_type='put', div=0.03)

# Affichage des détails de l'option vanille
display_option_details("Option vanille", option.option_price(), option.delta(), option.prob_of_exercise())
option.plot_payoff_and_profit()

# Stratégies optionnelles
print("\n### Stratégies Optionnelles ###\n")

# Straddle
display_option_details("Straddle", option.straddle_price(), option.straddle_delta())
option.plot_straddle_payoff_and_profit()

# Strangle
display_option_details("Strangle", option.strangle_price(75, 85), option.strangle_delta(75, 85))
option.plot_strangle_payoff_and_profit(75, 85)

# Butterfly
display_option_details("Butterfly", option.butterfly_price(75, 80, 85), option.butterfly_delta(75, 80, 85))
option.plot_butterfly_payoff_and_profit(75, 80, 85)

# Bull Call Spread
display_option_details("Call Spread", option.call_spread_price(80, 85), option.call_spread_delta(80, 85))
option.plot_call_spread_payoff_and_profit(80, 85)

# Bear Put Spread
display_option_details("Put Spread", option.put_spread_price(75, 80), option.put_spread_delta(75, 80))
option.plot_put_spread_payoff_and_profit(75, 80)

# Strip
display_option_details("Strip", option.strip_price(), option.strip_delta())
option.plot_strip_payoff_and_profit()

# Strap
display_option_details("Strap", option.strap_price(), option.strap_delta())
option.plot_strap_payoff_and_profit()

####### Options à barrière (KI, KO)


class BarrierOption(VanillaOption):
    """
    Représente une option à barrière, étendant la fonctionnalité d'une option vanille
    pour inclure les comportements des options à barrière Knock-In (KI) et Knock-Out (KO).
    """
    def __init__(self, spot, strike, barrier, maturity, risk_free_rate, volatility, option_type='call', barrier_type='KO'):
        """
        Initialise une nouvelle option à barrière.

        :param barrier: Le prix de la barrière qui active ou désactive l'option.
        :param barrier_type: Le type de barrière ('KO' pour Knock-Out, 'KI' pour Knock-In).
        """
        super().__init__(spot, strike, maturity, risk_free_rate, volatility, option_type)
        self.barrier = barrier
        self.barrier_type = barrier_type

    def simulate_paths(self, num_paths, num_steps):
        """
        Simule des trajectoires de prix pour l'actif sous-jacent en utilisant le modèle
        géométrique de Brownian Motion.

        :param num_paths: Nombre de trajectoires à simuler.
        :param num_steps: Nombre de pas de temps pour chaque trajectoire.
        :return: Un array NumPy contenant les trajectoires simulées.
        """
        dt = self.maturity / num_steps
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.spot
        for t in range(1, num_steps + 1):
            z = np.random.standard_normal(num_paths)
            paths[:, t] = paths[:, t - 1] * np.exp((self.risk_free_rate - 0.5 * self.volatility**2) * dt + self.volatility * np.sqrt(dt) * z)
        return paths

    def option_price_monte_carlo(self, num_paths=10000, num_steps=252):
        """
        Calcule le prix de l'option à barrière en utilisant la méthode de Monte Carlo.

        :param num_paths: Nombre de trajectoires à simuler pour le calcul.
        :param num_steps: Nombre de pas de temps pour chaque trajectoire.
        :return: Le prix estimé de l'option à barrière.
        """
        paths = self.simulate_paths(num_paths, num_steps)
        payoffs = np.zeros(num_paths)

        for i in range(num_paths):
            path = paths[i]
            if self.barrier_type == 'KO' and np.any(path >= self.barrier):
                payoffs[i] = 0
            elif self.barrier_type == 'KI' and not np.any(path >= self.barrier):
                payoffs[i] = 0
            else:
                if self.option_type == 'call':
                    payoffs[i] = max(path[-1] - self.strike, 0)
                else:  # 'put'
                    payoffs[i] = max(self.strike - path[-1], 0)

        return np.exp(-self.risk_free_rate * self.maturity) * np.mean(payoffs)
    
    def option_price_delta_monte_carlo(self, num_paths=10000, num_steps=252, spot_delta=1.0):
        """
        Calcule le delta de l'option à barrière en utilisant la méthode de Monte Carlo.

        :param spot_delta: Le petit changement appliqué au prix spot pour calculer le delta.
        :return: Le delta estimé de l'option à barrière.
        """
        original_spot = self.spot
        original_price = self.option_price_monte_carlo(num_paths, num_steps)

        # Augmenter légèrement le spot et recalculer le prix
        self.spot += spot_delta
        new_price = self.option_price_monte_carlo(num_paths, num_steps)

        # Restaurer le spot original
        self.spot = original_spot

        # Calculer et retourner le delta
        delta = (new_price - original_price) / spot_delta
        return delta
    
    def probability_of_exercise(self, num_paths=10000, num_steps=252):
        """
        Estime la probabilité que l'option soit exercée, en utilisant la simulation de Monte Carlo.

        :return: La probabilité estimée que l'option soit exercée.
        """
        
        paths = self.simulate_paths(num_paths, num_steps)
        in_the_money_count = 0
    
        for i in range(num_paths):
            path = paths[i]
            final_price = path[-1]
            if self.option_type == 'call':
                condition = final_price > self.strike
            else:  # 'put'
                condition = final_price < self.strike
    
            if self.barrier_type == 'KO':
                barrier_breached = np.any(path >= self.barrier) if self.option_type == 'call' else np.any(path <= self.barrier)
                if condition and not barrier_breached:
                    in_the_money_count += 1
            elif self.barrier_type == 'KI':
                barrier_breached = np.any(path >= self.barrier) if self.option_type == 'call' else np.any(path <= self.barrier)
                if condition and barrier_breached:
                    in_the_money_count += 1
    
        probability = in_the_money_count / num_paths
        return probability

    def plot_payoff(self):
        """
        Trace le graphique de payoff de l'option à barrière à l'échéance, en fonction du prix final de l'actif sous-jacent.
        """
        
        # Gamme de prix de l'actif sous-jacent à la maturité
        S = np.linspace(0.5 * self.barrier, 1.5 * self.barrier, 100)
        payoffs = np.zeros_like(S)

        # Calculer le payoff pour chaque prix à la maturité
        for i, s in enumerate(S):
            if self.option_type == 'call':
                payoff = max(s - self.strike, 0)
            else:  # 'put'
                payoff = max(self.strike - s, 0)
            
            # Ajustements pour les options barrières
            if self.barrier_type == 'KO':
                if (self.option_type == 'call' and s >= self.barrier) or (self.option_type == 'put' and s <= self.barrier):
                    payoff = 0  # Option désactivée si le prix traverse la barrière
            elif self.barrier_type == 'KI':
                if (self.option_type == 'call' and s < self.barrier) or (self.option_type == 'put' and s > self.barrier):
                    payoff = 0  # Option non activée si le prix ne touche pas la barrière
            
            payoffs[i] = payoff

        # Tracer le graphique de payoff
        plt.figure(figsize=(10, 6))
        plt.plot(S, payoffs, label='Payoff at Maturity')
        plt.axhline(0, color='black', lw=1)
        plt.axvline(self.strike, color='grey', linestyle='--', label='Strike Price')
        if self.barrier_type == 'KO':
            plt.axvline(self.barrier, color='red', linestyle='--', label='Barrier')
        elif self.barrier_type == 'KI':
            plt.axvline(self.barrier, color='green', linestyle='--', label='Barrier')
        plt.xlabel('Price of Underlying Asset at Maturity')
        plt.ylabel('Payoff')
        plt.title('Barrier Option Payoff at Maturity')
        plt.legend()
        plt.grid(True)
        plt.show()

print("\n### Option Barriere ###\n")
barrier_option = BarrierOption(spot=100, strike=100, barrier=110, maturity=1, risk_free_rate=0.05, volatility=0.2, option_type='call', barrier_type='KI')
barrier_option_price = barrier_option.option_price_monte_carlo()
barrier_option_delta = barrier_option.option_price_delta_monte_carlo()
print('Le prix de l option barrière est : ', barrier_option_price)
print('Le delta de l option barrière est : ', barrier_option_delta)
print('La probabilite d exercice est ', barrier_option.probability_of_exercise())
barrier_option.plot_payoff()


##### Options binaires

class BinaryOption:
    """
    Représente une option binaire, qui paie un montant fixe si certaines conditions sont remplies
    à l'échéance.
    """

    def __init__(self, spot, strike, maturity, risk_free_rate, volatility, payoff=1, option_type='call', div=0):
        """
        Initialise une nouvelle option binaire.

        :param spot: Prix actuel de l'actif sous-jacent.
        :param strike: Prix d'exercice de l'option.
        :param maturity: Temps jusqu'à l'échéance (en années).
        :param risk_free_rate: Taux d'intérêt sans risque annuel.
        :param volatility: Volatilité annuelle de l'actif sous-jacent.
        :param payoff: Montant du paiement si l'option est dans la monnaie.
        :param option_type: Type de l'option ('call' ou 'put').
        :param div: Taux de dividende annuel de l'actif sous-jacent.
        """
        self.spot = spot
        self.strike = strike
        self.maturity = maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.payoff = payoff
        self.option_type = option_type
        self.div = div
        
    def d1(self):
        return (np.log(self.spot / self.strike) + (self.risk_free_rate + 0.5 * self.volatility**2) * self.maturity) / (self.volatility * np.sqrt(self.maturity))

    def d2(self):
        return self.d1() - self.volatility * np.sqrt(self.maturity)

    def option_price(self):
        """
        Calcule et retourne le prix de l'option binaire en utilisant une formule fermée.

        :return: Le prix de l'option binaire.
        """
        d2 = self.d2()
        s = self.spot * np.exp(-self.div * self.maturity)
        if self.option_type == 'call':
            return np.exp(-self.risk_free_rate * self.maturity) * st.norm.cdf(d2) * self.payoff
        else:
            return np.exp(-self.risk_free_rate * self.maturity) * st.norm.cdf(-d2) * self.payoff

    def delta(self):
        """
        Calcule et retourne le delta de l'option binaire.

        :return: Le delta de l'option binaire.
        """
        d2 = self.d2()
        pdf_d2 = st.norm.pdf(d2)
        common_factor = np.exp(-self.risk_free_rate * self.maturity) * pdf_d2 / (self.spot * self.volatility * np.sqrt(self.maturity))
        if self.option_type == 'call':
            delta = common_factor
        else:  # 'put'
            delta = -common_factor
        return delta
    
    def probability_of_exercise(self):
        """
        Calcule et retourne la probabilité que l'option soit exercée.

        :return: La probabilité d'exercice de l'option binaire.
        """
        d2 = self.d2()
        if self.option_type == 'call':
            return st.norm.cdf(d2)
        else:
            return st.norm.cdf(-d2)
    
    def plot_payoff(self):
       """
       Trace le graphique de payoff de l'option binaire à l'échéance.
       """
       S = np.linspace(0.5 * self.strike, 1.5 * self.strike, 100)
       payoffs = np.zeros_like(S)
       
       if self.option_type == 'call':
           payoffs[S >= self.strike] = self.payoff
       else:  # 'put'
           payoffs[S <= self.strike] = self.payoff
       plt.figure(figsize=(10, 6))
       plt.plot(S, payoffs, label='Binary Option Payoff')
       plt.axhline(0, color='black', lw=1)
       plt.axvline(self.strike, color='grey', linestyle='--', label='Strike Price')
       plt.xlabel('Price of Underlying Asset')
       plt.ylabel('Payoff')
       plt.title(f'Binary {self.option_type.capitalize()} Option Payoff at Maturity')
       plt.legend()
       plt.grid(True)
       plt.show()


        
print("\n### Option Binaire ###\n")
binary_call = BinaryOption(spot=100, strike=100, maturity=1, risk_free_rate=0.05, volatility=0.2, payoff=1, option_type='call', div=0.03)
print('Binary Call Option Price:', binary_call.option_price())
print('Binary Call Option Delta:', binary_call.delta())
print('La probabilite d exercice est ', binary_call.probability_of_exercise())
binary_call.plot_payoff()

class StructuredProducts:
    def __init__(self, spot, strike, maturity, risk_free_rate, volatility, div=0, payoff=1, participation_rate=1):
        self.spot = spot
        self.strike = strike
        self.maturity = maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.div = div
        self.payoff = payoff  # Montant fixe pour le Reverse Convertible
        self.participation_rate = participation_rate  # Pour Certificat Outperformance
        self.option_type = 'call'  # Initialiser avec 'call', peut être changé dynamiquement

    def d1(self):
        return (np.log(self.spot / self.strike) + (self.risk_free_rate - self.div + 0.5 * self.volatility**2) * self.maturity) / (self.volatility * np.sqrt(self.maturity))

    def d2(self):
        return self.d1() - self.volatility * np.sqrt(self.maturity)
    
    def reverse_convertible_price(self, bond_redemption_value):
        self.payoff = bond_redemption_value  # Mettre à jour le payoff avec la valeur de rédemption
        put_price = self.put_option_price()
        bond_price = np.exp(-self.risk_free_rate * self.maturity) * bond_redemption_value
        return bond_price - put_price

    def put_option_price(self):
        self.option_type = 'put'
        d1 = self.d1()
        d2 = self.d2()
        put_price = (st.norm.cdf(-d2) * self.strike * np.exp(-self.risk_free_rate * self.maturity) - st.norm.cdf(-d1) * self.spot * np.exp(-self.div * self.maturity))
        return put_price

    def call_option_price(self):
        self.option_type = 'call'
        d1 = self.d1()
        d2 = self.d2()
        call_price = (st.norm.cdf(d1) * self.spot * np.exp(-self.div * self.maturity) - st.norm.cdf(d2) * self.strike * np.exp(-self.risk_free_rate * self.maturity))
        return call_price

    def reverse_convertible_payoff(self, final_price):
        if final_price < self.strike:
            return final_price / self.strike * self.payoff
        else:
            return self.payoff

    def outperformance_certificate_price(self):
        call_price = self.call_option_price()
        return call_price * self.participation_rate

    def outperformance_certificate_payoff(self, final_price):
        if final_price > self.strike:
            return (final_price - self.strike) * self.participation_rate
        else:
            return 0
    def reverse_convertible_delta(self):
        # Le delta du Reverse Convertible est essentiellement le delta de l'option put vendue
        self.option_type = 'put'  # Temporairement définir comme put pour calculer le delta
        delta_put = -self.delta()  # Négatif car l'option est vendue
        self.option_type = 'call'  # Restaurer le type d'option initial si nécessaire
        return delta_put

    def outperformance_certificate_delta(self):
        # Le delta du Certificat Outperformance peut être approximé par le delta d'une option call
        self.option_type = 'call'  # S'assurer que le type est défini sur call
        delta_call = self.delta() * self.participation_rate  # Ajuster par le taux de participation
        return delta_call

    def delta(self):
        # Calcul de base du delta pour une option call ou put
        d1= self.d1()
        if self.option_type == 'call':
            return np.exp(-self.div * self.maturity) * st.norm.cdf(d1)
        else:
            return np.exp(-self.div * self.maturity) * (st.norm.cdf(d1) - 1)
        
    def plot_reverse_convertible_payoff(self):
        final_prices = np.linspace(0.5 * self.strike, 1.5 * self.strike, 100)
        payoffs = [self.reverse_convertible_payoff(price) for price in final_prices]
        plt.figure(figsize=(10, 6))
        plt.plot(final_prices, payoffs, label='Reverse Convertible Payoff')
        plt.xlabel('Final Price of Underlying Asset')
        plt.ylabel('Payoff')
        plt.title('Reverse Convertible Payoff')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_outperformance_certificate_payoff(self):
        final_prices = np.linspace(0.5 * self.strike, 1.5 * self.strike, 100)
        payoffs = [self.outperformance_certificate_payoff(price) for price in final_prices]
        plt.figure(figsize=(10, 6))
        plt.plot(final_prices, payoffs, label='Outperformance Certificate Payoff')
        plt.xlabel('Final Price of Underlying Asset')
        plt.ylabel('Payoff')
        plt.title('Outperformance Certificate Payoff')
        plt.legend()
        plt.grid(True)
        plt.show()
        
# Paramètres d'exemple pour un Reverse Convertible
spot_rc = 100
strike_rc = 100
maturity_rc = 1
risk_free_rate_rc = 0.05
volatility_rc = 0.2
div_rc = 0.03
bond_redemption_value_rc = 105  # La valeur à laquelle l'obligation sera remboursée

# Création de l'instance pour un Reverse Convertible
reverse_convertible = StructuredProducts(spot=100, strike=100, maturity=1, 
                                         risk_free_rate=0.05, volatility=0.2, div=0.03)

print("\n### Produits Structures ###\n")
# Calcul et affichage du prix et du delta pour le Reverse Convertible
rc_price = reverse_convertible.reverse_convertible_price(bond_redemption_value=105)
rc_delta = reverse_convertible.reverse_convertible_delta()
print(f"Reverse Convertible Price: {rc_price}")
print(f"Reverse Convertible Delta: {rc_delta}")

# Tracé du payoff pour le Reverse Convertible
reverse_convertible.plot_reverse_convertible_payoff()

# Paramètres d'exemple pour un Certificat Outperformance
spot_op = 100
strike_op = 100
maturity_op = 1
risk_free_rate_op = 0.05
volatility_op = 0.2
div_op = 0.03
participation_rate_op = 1.5  # Taux de participation à la performance de l'actif

# Création de l'instance pour un Certificat Outperformance
outperformance_certificate = StructuredProducts(spot=100, strike=100, maturity=1, 
                                                risk_free_rate=0.05, volatility=0.2, div=0.03, 
                                                participation_rate=1.5)

# Calcul et affichage du prix et du delta pour le Certificat Outperformance
op_price = outperformance_certificate.outperformance_certificate_price()
op_delta = outperformance_certificate.outperformance_certificate_delta()
print(f"Outperformance Certificate Price: {op_price}")
print(f"Outperformance Certificate Delta: {op_delta}")

# Tracé du payoff pour le Certificat Outperformance
outperformance_certificate.plot_outperformance_certificate_payoff()

