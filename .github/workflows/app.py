import streamlit as st
import numpy as np
import scipy.stats as stt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from io import BytesIO
from projetArbanCharbitChibaoui import Bond, VanillaOption, BinaryOption, BarrierOption, StructuredProducts
st.set_option('deprecation.showPyplotGlobalUse', False)


st.title('Calculateur de prix de produits financiers')

# Choix du produit financier
product_type = st.selectbox(
    'Choisir le type de produit financier',
    ['Obligation', 'Option Vanille', 'Option Binaire', 'Option Barrière', 'Produit Structuré']
)

# Paramètres communs
risk_free_rate = st.number_input('Taux sans risque', min_value=0.0, value=0.03)
maturity = st.number_input('Maturité', min_value=0.0, value=1.0)

# Paramètres spécifiques à chaque produit
if product_type == 'Obligation':
    nominal = st.number_input('Valeur nominale', min_value=0.0, value=100.0, key='nominal_obligation')
    st.header('Paramètres de l\'obligation')
    nominal = st.number_input('Valeur nominale', min_value=0.0, value=100.0)
    coupon_rate = st.number_input('Taux de coupon', min_value=0.0, value=0.05)
    freq = st.number_input('Fréquence de paiement', min_value=1, value=2)

    # Calculer le prix et les détails de l'obligation
    if st.button('Calculer les détails de l\'obligation'):
        bond = Bond(nominal, coupon_rate, maturity, risk_free_rate, freq)
        bond_price = bond.bond_price()
        duration, convexity = bond.duration_and_convexity()
        mod_duration = bond.modified_duration()
        price_sensitivity = bond.price_sensitivity()

        st.write(f"Le prix de l'obligation est : {bond_price:.2f} €")
        st.write(f"La duration de Macaulay est : {duration:.2f} années")
        st.write(f"La convexité est : {convexity:.2f}")
        st.write(f"La duration modifiée est : {mod_duration:.2f}")
        st.write(f"La sensibilité du prix pour un changement de 1% du taux d'intérêt est : {price_sensitivity:.2f} €")

    

elif product_type == 'Option Vanille':
    st.header('Paramètres de l\'option vanille')
    spot = st.number_input('Prix spot', min_value=0.01, max_value=10000.0, value=100.0, step=0.01)
    strike = st.number_input('Prix d\'exercice', min_value=0.01, max_value=10000.0, value=100.0, step=0.01)
    volatility = st.number_input('Volatilité', min_value=0.01, max_value=2.0, value=0.20, step=0.01)
    div_yield = st.number_input('Rendement du dividende', min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    option_type = st.selectbox('Type d\'option', ['call', 'put'])
    strategy_type = st.selectbox(
    'Stratégie',
    ['Simple', 'Straddle', 'Strangle', 'Butterfly', 'Bull Call Spread', 'Bear Put Spread', 'Strip', 'Strap']
)

    # Stratégie simple
    if strategy_type == 'Simple':
        if st.button('Calculer le prix de l\'option vanille'):
            option = VanillaOption(spot, strike, maturity, risk_free_rate, volatility, option_type, div_yield)
            option_price = option.option_price()
            option_delta = option.delta()
            st.write(f"Le prix de l'option vanille est: {option_price:.2f} €")
            st.write(f"Le delta de l'option est: {option_delta:.2f}")

    # Stratégie Straddle
    elif strategy_type == 'Straddle':
        if st.button('Calculer le prix et le delta du straddle'):
            option = VanillaOption(spot, strike, maturity, risk_free_rate, volatility, 'call', div_yield)
            straddle_price = option.straddle_price()
            straddle_delta = option.straddle_delta()
            st.write(f"Le prix du straddle est: {straddle_price:.2f} €")
            st.write(f"Le delta du straddle est: {straddle_delta:.2f}")
            option.plot_straddle_payoff_and_profit()  # Uncomment this line if you implement plotting
            st.pyplot()
    # Stratégie Strangle
    elif strategy_type == 'Strangle':
        strike_put = st.number_input('Prix d\'exercice Put', min_value=0.01, max_value=10000.0, value=strike * 0.95, step=0.01)
        strike_call = st.number_input('Prix d\'exercice Call', min_value=0.01, max_value=10000.0, value=strike * 1.05, step=0.01)
        if st.button('Calculer le prix et le delta du strangle'):
            option = VanillaOption(spot, strike, maturity, risk_free_rate, volatility, 'call', div_yield)
            strangle_price = option.strangle_price(strike_put, strike_call)
            strangle_delta = option.strangle_delta(strike_put, strike_call)
            st.write(f"Le prix du strangle est: {strangle_price:.2f} €")
            st.write(f"Le delta du strangle est: {strangle_delta:.2f}")
            option.plot_strangle_payoff_and_profit(strike_put, strike_call)  # Uncomment this line if you implement plottin
            st.pyplot()

    elif strategy_type == 'Butterfly':
        strike_lower = st.number_input('Prix d\'exercice inférieur', min_value=0.01, max_value=10000.0, value=strike * 0.95, step=0.01)
        strike_middle = st.number_input('Prix d\'exercice moyen', min_value=0.01, max_value=10000.0, value=strike, step=0.01)
        strike_upper = st.number_input('Prix d\'exercice supérieur', min_value=0.01, max_value=10000.0, value=strike * 1.05, step=0.01)
        if st.button('Calculer le prix et les détails du Butterfly'):
            option = VanillaOption(spot, strike_middle, maturity, risk_free_rate, volatility, 'call', div_yield)
            butterfly_price = option.butterfly_price(strike_lower, strike_middle, strike_upper)
            butterfly_delta = option.butterfly_delta(strike_lower, strike_middle, strike_upper)
        
            st.write(f"Le prix du Butterfly est: {butterfly_price:.2f} €")
            st.write(f"Le delta du Butterfly est: {butterfly_delta:.2f}")

        # Tracé du graphique de payoff pour le Butterfly
            option.plot_butterfly_payoff_and_profit(strike_lower, strike_middle, strike_upper)
            st.pyplot()


    elif strategy_type == 'Bull Call Spread':
        strike_buy = st.number_input('Prix d\'exercice de l\'achat Call', min_value=0.01, max_value=10000.0, value=strike * 0.9, step=0.01)
        strike_sell = st.number_input('Prix d\'exercice de la vente Call', min_value=0.01, max_value=10000.0, value=strike * 1.1, step=0.01)
        if st.button('Calculer le prix et les détails du Bull Call Spread'):
            option = VanillaOption(spot, strike_buy, maturity, risk_free_rate, volatility, 'call', div_yield)
            call_spread_price = option.call_spread_price(strike_buy, strike_sell)
            call_spread_delta = option.call_spread_delta(strike_buy, strike_sell)
            
            st.write(f"Le coût net du Bull Call Spread est: {call_spread_price:.2f} €")
            st.write(f"Le delta net du Bull Call Spread est: {call_spread_delta:.2f}")
            
            option.plot_call_spread_payoff_and_profit(strike_buy, strike_sell)
            st.pyplot()

    # Stratégie Bear Put Spread
    elif strategy_type == 'Bear Put Spread':
        strike_sell = st.number_input('Prix d\'exercice de la vente Put', min_value=0.01, max_value=10000.0, value=strike * 0.9, step=0.01)
        strike_buy = st.number_input('Prix d\'exercice de l\'achat Put', min_value=0.01, max_value=10000.0, value=strike * 1.1, step=0.01)
        if st.button('Calculer le prix et les détails du Bear Put Spread'):
            option = VanillaOption(spot, strike_sell, maturity, risk_free_rate, volatility, 'put', div_yield)
            put_spread_price = option.put_spread_price(strike_sell, strike_buy)
            put_spread_delta = option.put_spread_delta(strike_sell, strike_buy)
            
            st.write(f"Le coût net du Bear Put Spread est: {put_spread_price:.2f} €")
            st.write(f"Le delta net du Bear Put Spread est: {put_spread_delta:.2f}")
            
            option.plot_put_spread_payoff_and_profit(strike_sell, strike_buy)
            st.pyplot()
    # Stratégie Strip
    elif strategy_type == 'Strip':
        if st.button('Calculer le prix et le delta du Strip'):
            option = VanillaOption(spot, strike, maturity, risk_free_rate, volatility, 'call', div_yield)
            strip_price = option.strip_price()
            strip_delta = option.strip_delta()
            st.write(f"Le prix du Strip est: {strip_price:.2f} €")
            st.write(f"Le delta du Strip est: {strip_delta:.2f}")
            option.plot_strip_payoff_and_profit()
            st.pyplot()

    # Stratégie Strap
    elif strategy_type == 'Strap':
        if st.button('Calculer le prix et le delta du Strap'):
            option = VanillaOption(spot, strike, maturity, risk_free_rate, volatility, 'call', div_yield)
            strap_price = option.strap_price()
            strap_delta = option.strap_delta()
            st.write(f"Le prix du Strap est: {strap_price:.2f} €")
            st.write(f"Le delta du Strap est: {strap_delta:.2f}")
            option.plot_strap_payoff_and_profit()
            st.pyplot()


elif product_type == 'Option Barrière':
    st.header('Paramètres de l\'option barrière')
    spot = st.number_input('Prix spot pour l\'option barrière', min_value=0.0, value=100.0)
    strike = st.number_input('Prix d\'exercice pour l\'option barrière', min_value=0.0, value=100.0)
    barrier = st.number_input('Barrière', min_value=0.0, value=110.0)
    volatility = st.number_input('Volatilité pour l\'option barrière', min_value=0.0, value=0.2)
    option_type = st.selectbox('Type d\'option barrière', ['call', 'put'])
    barrier_type = st.selectbox('Type de barrière', ['KO', 'KI'])
    div_yield = st.number_input('Dividende pour l\'option barrière', min_value=0.0, value=0.03)

    if st.button('Calculer le prix, le delta et la probabilité d\'exercice de l\'option barrière'):
        barrier_option = BarrierOption(spot, strike, barrier, maturity, risk_free_rate, volatility, option_type, barrier_type)
        barrier_option_price = barrier_option.option_price_monte_carlo()
        barrier_option_delta = barrier_option.option_price_delta_monte_carlo()
        barrier_option_proba = barrier_option.probability_of_exercise()

        st.write(f'Le prix de l\'option barrière est: {barrier_option_price:.2f}')
        st.write(f'Le delta de l\'option barrière est: {barrier_option_delta:.2f}')
        st.write(f'La probabilité d\'exercice est: {barrier_option_proba:.2f}')
        
        barrier_option.plot_payoff()
        st.pyplot()

elif product_type == 'Option Binaire':
    st.header('Paramètres de l\'option binaire')
    spot = st.number_input('Prix spot pour l\'option binaire', min_value=0.0, value=100.0)
    strike = st.number_input('Prix d\'exercice pour l\'option binaire', min_value=0.0, value=100.0)
    volatility = st.number_input('Volatilité pour l\'option binaire', min_value=0.0, value=0.2)
    payoff = st.number_input('Payoff', min_value=0.0, value=1.0)
    option_type = st.selectbox('Type d\'option binaire', ['call', 'put'])
    div = st.number_input('Dividende pour l\'option binaire', min_value=0.0, value=0.03)

    if st.button('Calculer le prix et les détails de l\'option binaire'):
        binary_option = BinaryOption(spot, strike, maturity, risk_free_rate, volatility, payoff, option_type, div)
        binary_option_price = binary_option.option_price()
        binary_option_delta = binary_option.delta()
        binary_option_proba = binary_option.probability_of_exercise()

        st.write(f"Prix de l'option binaire {option_type}: {binary_option_price:.2f}")
        st.write(f"Delta de l'option binaire: {binary_option_delta:.2f}")
        st.write(f"Probabilité d'exercice de l'option binaire: {binary_option_proba:.2f}")

        # Tracer et afficher le graphique de payoff
        binary_option.plot_payoff()
        st.pyplot()

elif product_type == 'Produit Structuré':
    st.header('Paramètres du Produit Structuré')
    structured_product_type = st.selectbox('Choisir le type de produit structuré', ['Reverse Convertible', 'Outperformance Certificate'])

    spot = st.number_input('Prix spot', min_value=0.01, max_value=10000.0, value=100.0, step=0.01)
    strike = st.number_input('Prix d\'exercice', min_value=0.01, max_value=10000.0, value=100.0, step=0.01)
    maturity = st.number_input('Maturité', min_value=0.01, max_value=30.0, value=1.0, step=0.01)
    risk_free_rate = st.number_input('Taux sans risque', min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    volatility = st.number_input('Volatilité', min_value=0.0, max_value=2.0, value=0.20, step=0.01)
    div = st.number_input('Dividende', min_value=0.0, max_value=1.0, value=0.03, step=0.01)
    payoff = st.number_input('Payoff', min_value=0.0, value=100.0)
    participation_rate = st.number_input('Taux de participation', min_value=0.0, value=1.5)

    structured_product = StructuredProducts(spot, strike, maturity, risk_free_rate, volatility, div, payoff, participation_rate)

    if structured_product_type == 'Reverse Convertible':
        bond_redemption_value = st.number_input('Valeur de rédemption de l\'obligation', min_value=0.0, value=100.0)
        if st.button('Calculer Reverse Convertible'):
            rc_price = structured_product.reverse_convertible_price(bond_redemption_value)
            rc_delta = structured_product.reverse_convertible_delta()
            st.write(f"Prix du Reverse Convertible: {rc_price:.2f}")
            st.write(f"Delta du Reverse Convertible: {rc_delta:.2f}")
            structured_product.plot_reverse_convertible_payoff()
            st.pyplot()

    elif structured_product_type == 'Outperformance Certificate':
        if st.button('Calculer Outperformance Certificate'):
            op_price = structured_product.outperformance_certificate_price()
            op_delta = structured_product.outperformance_certificate_delta()
            st.write(f"Prix du Certificat d\'Outperformance: {op_price:.2f}")
            st.write(f"Delta du Certificat d\'Outperformance: {op_delta:.2f}")
            structured_product.plot_outperformance_certificate_payoff()
            st.pyplot()
   