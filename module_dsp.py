# -*- coding: utf-8 -*- 
"""
Versie 29 juni 2023

Module voor het vak Digitale signaalbewerking.
Verwijzingen naar formules uit het boek van Weji Wang, 'Introduction to
Digital Signal and System Analysis'.
Daarnaast ook verwijzingen naar de DSP Guide ('The Scientist and Engineer's 
Guide to Digital Signal Processing', by Steven W. Smith, Ph.D.)

@author: wen
"""

#%% ==========================================================================#
# --- Importeer modules
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import special
from numpy.polynomial import Polynomial
import numpy.testing as tst

#%% ==========================================================================#
def unit_impulse(N,n0=0):
    ''' Functie die een eenheidspuls (unit pulse) maakt
            
Invoer
-----
   N : aantal elementen in de uitvoer array
   
   n0: positie waar signaal gelijk is aan 1. Op andere posities is het
   signaal gelijk aan 0

Uitvoer
-------
   d: numpy array d = array(d[0], d[1], d[2], ... , d[N-1]), met 
   d[n0] = 1 en alle andere elementen gelijk aan 0.
        
        '''
    # Creeer eenheidspuls
    d     = np.zeros(N)
    d[n0] = 1.0
    
    # Return output
    return d

#%% ==========================================================================#
def unit_step(N,n0=0):
    ''' Functie die een eenheidsstap (unit step) maakt
            
Invoer
-----
   N : aantal elementen in de array
   
   n0: positie waar de overgang van signaal = 0 naar signaal = 1 
   plaatsvindt

Uitvoer
------
   u: numpy array u = array(u[0], u[1], u[2], ... , u[N-1]), met 
   u[j] = 0 voor j < n0 en u[j] = 1 voor j >= n0
        
        '''
    # Creeer unit step
    u       = np.zeros(N)
    u[n0:N] = 1.0
    
    # Return output
    return u

#%% ==========================================================================#
def unit_ramp(N,n0=0):
    ''' Functie die een lineaire functie (unit ramp) maakt
            
Invoer
------
   N : aantal elementen in de array
   
   n0: positie waar de ramp begint

Uitvoer
------
   r: numpy array r = array(r[0], r[1], r[2], ... , r[N-1]), met 
   r[j] = 0 voor j < n0, r[n0] = 0, r[n0+1] = 1, r[n0+2] = 2, etc.
        '''
    # Creeer unit ramp
    r       = np.zeros(N)
    r[n0:N] = np.array( range(N-n0) )
    
    # Return output
    return r

#%% ==========================================================================#
def lti_tijdrespons(x_inp,a_val,b_val,mode='scipy'):
    ''' Functie die de tijdrespons van een LTI systeem bepaalt adhv input x_inp 
en coefficienten in LTI coefficienten in arrays a_val en b_val. 
        
Formule (3.9) uit Wang:
   a[0]*y[n] + a[1]*y[n-1] + a[2]*y[n-2] + ... + a[N]*y[n-N] =
   b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] + ... + b[M]*x[n-M]
    
Bij recursief rekenen wordt deze formule als volgt gebruikt:

y[n] = ( -a[1]*y[n-1] - a[2]*y[n-2] - ... - a[N]*y[n-N] +
b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] + ... + b[M]*x[n-M] ) / a[0]
           
Voorbeeld
---------
y[n] = -0.7*y[n-1] + 0.2*y[n-2] + 0.5*x[n-1] - 0.8*x[n-2]
leidt tot: a_val = [1.0 , 0.7, -0.2] en b_val = [ 0.0, 0.5, -0.8]
   
Invoer
------
   x_inp: array met N_x elementen die de input x van het LTI systeem 
   beschrijft. Correspondeert met x in de formule hierboven.
   
   a_val: array met N+1 variabelen die correspondeert met a in de  
   formule hierboven
   
   b_val: array met M+1 variabelen die correspondeert met b in de 
   formule hierboven
   
   mode: indicates the way in which the time-response is computed.
   All modes yield the same results. 
       
       mode = 'scipy': makes use of efficient scipy functions (default)

       mode = 'explicit': is based on a code in which the response is computed 
               using visible loops. Aim is to make commputation of correlation 
               visible

Uitvoer
------
   y_out: array met N_x elementen die de output y van het LTI systeem 
   bevat. Correspondeert met y in de formule hierboven       
        '''
    # Bepaal enkele constante waarden
    N_x = len(x_inp)
    N   = len(a_val) - 1
    M   = len(b_val) - 1
    
    if ( mode == 'explicit' ):
        # Initialiseer uitvoer
        y_out = np.empty(N_x)
        
        # Bereken de output waarden middels een loop
        for n in range(0,N_x):
            # Bereken de term met a*y coefficienten. Houd rekening met feit dat
            # y[-1], y[-2], ... gelijk aan nul (niet gedefinieerd) zijn
            sum_a_y = 0.0
            for k in range( 1,min(N,n)+1 ):
                sum_a_y += a_val[k]*y_out[n-k]
                
            # Bereken de term met b*x coefficienten. Houd rekening met feit dat
            # x[-1], x[-2], ... nul (niet gedefinieerd) zijn
            sum_b_x = 0.0
            for k in range( 0,min(M,n)+1 ):
                sum_b_x += b_val[k]*x_inp[n-k]
                
            # Bepaal y[n]
            y_out[n] = (-sum_a_y + sum_b_x) / a_val[0]
            
    elif ( 'scipy' ):
        # Gebruik scipy functie
        y_out = signal.lfilter( b_val, a_val, x_inp )

#    elif ( mode == 'fft' ):
#        print('Deze manier werkt nog niet goed, want geeft geen identieke ')
#        print('resultaten als de expliciete manier, omdat spectrum periodiciteit')
#        print('verondersteld')
#        return
#        # Bereken spectrum van invoer signaal
#        dt = 1.0      # Waarde gelijk gekozen aan 1 om tijdas te kunnen maken
#        t  = dt*np.arange(N_x)
#        f,X_inp_f = dsp.tijdsignaal_naar_spectrum(t,x_inp)
#        
#        # Bepaal relatieve hoekfrequentie
#        fs = 1/dt
#        Omega = 2*np.pi*f/fs
#        
#        # Compute impulse response in freq domain
#        H = dsp.lti_freqrespons(Omega,a_val,b_val)
#       
#        # Compute spectrum of output signal
#        Y_out_f = H * X_inp_f
#        
#        # Compute inverse fourier of output signal
#        t_dummy,y_out = dsp.complex_spectrum_naar_tijdsignaal1(f,Y_out_f)
  
        
    # Return output
    return y_out

#%% ==========================================================================#
def lti_freqrespons(Omega,a_val,b_val,mode='scipy'):
    ''' Functie die de frequentie overdracht van een LTI systeem bepaalt adhv 
input relatieve hoekfrequentie Omega en coefficienten in LTI-arrays 
a_val en b_val. 
        
Formule (3.9) uit Wang:
   a[0]*y[n] + a[1]*y[n-1] + a[2]*y[n-2] + ... + a[N]*y[n-N] =
   b[0]*x[n] + b[1]*x[n-1] + b[2]*y[n-2] + ... + b[M]*y[n-M]
            
Bij berekenen frequentie-respons wordt deze formule als volgt genoteerd
(zie vgl (4.8) uit Wang):
        
   H(Omega) = Y(Omega) / X(Omega)] = 
   (b[0] + b[1]*z^(-1) + ... + b[M]*z^(-M)) / 
   (a[0] + a[1]*z^(-1) + ... + a[N]*z^(-N))
  
met z = exp(j*Omega).
          
Invoer
------
   Omega: array met relatieve hoekfrequentie.
   
   a_val: array met N+1 variabelen die corresponderen met a in de  
   formule hierboven
   
   b_val: array met M+1 variabelen die corresponderen met b in de 
   formule hierboven
    
   mode: indicates the way in which the frequency-response is computed.
   All modes yield the same results. 
       
       mode = 'scipy': makes use of efficient scipy functions (default)

       mode = 'explicit': is based on a code in which the response is computed 
               using visible loops. Aim is to make commputation of correlation 
               visible
   
Uitvoer
------
   H_omega: array met (complexe) overdrachtsfunctie voor alle 
   hoekfrequenties in Omega. Correspondeert met H(Omega) in de formule 
   hierboven       
        '''
        
    # Bepaal response
    if ( mode == 'explicit' ):
        # Bepaal enkele constante waarden
        Nf     = len(Omega)
        N      = len(a_val) - 1
        M      = len(b_val) - 1
        z_min1 = np.exp(-1j*Omega)
        z_minN = np.ones(Nf,'complex')
        
        # Initialiseer teller en noemer
        teller = np.zeros(Nf,'complex')
        noemer = np.zeros(Nf,'complex')
        
        # Bereken de teller en noemer middels een loop
        for nm in range(0,max(N,M)+1):
            # Bereken noemer
            if ( nm <= N ):
                noemer += a_val[nm]*z_minN
            
            # Berkenen teller
            if ( nm <= M ):
                teller += b_val[nm]*z_minN
                
            # Update z_minN
            z_minN *= z_min1
            
        # Bereken overdracht
        H_omega = teller / noemer
        
    elif ( mode == 'scipy' ):
        # Aanroep van scipy functie
        Omega_dum,H_omega = signal.freqz(b_val,a_val,Omega,False,None)
        
        # Test of omega_dum gelijk is aan omega
        tst.assert_allclose(Omega_dum,Omega, atol=1e-7 )
  
    # Return output
    return H_omega

#%% ==========================================================================#
def lti_freqrespons_zeros_poles(Omega,K,zeros_val,poles_val,mode='scipy'):
    ''' Functie die de frequentie overdracht van een LTI systeem bepaalt adhv  
input relatieve hoekfrequentie Omega, de versterkingsfactor K en de 
polen en nulpunten in arrays poles_val en zeros_val.
        
Formule (5.4) uit Wang:
    H(Omega) = Y(Omega) / X(Omega)] = 
    K * (z-zeros_val[0]) * (z-zeros_val[1]) * ... * (z-zeros_val[Nz-1])  
    /((z-poles_val[0]) * (z-poles_val[1]) * ... * (z-poles_val[Np-1]))
  
met z = exp(j*Omega)
          
Invoer
------
   Omega: array met relatieve hoekfrequentie.
   
   K: versterkingsfactor
   
   zeros_val: (meestal complexe) array met Nz waarden van de nulpunten
   
   poles_val: (meestal complexe) array met Np waarden van de polen 

   mode: indicates the way in which the frequency-response is computed.
   All modes yield the same results. 
       
       mode = 'scipy': makes use of efficient scipy functions (default)

       mode = 'explicit': is based on a code in which the response is computed 
               using visible loops. Aim is to make commputation of correlation 
               visible
 
Uitvoer
-------
   H_freq: array met (complexe) overdrachtsfunctie voor alle 
   relatieve hoekfrequenties in Omega. Correspondeert met H_freq in de formule 
   hierboven       
        '''
        
    if ( mode == 'explicit' ):
        # Bepaal enkele constante waarden
        Nf = len(Omega)
        Np = len(poles_val)
        Nz = len(zeros_val)
        z  = np.exp(1j*Omega)
        
        # Initialiseer teller en noemer
        teller = np.ones(Nf,'complex')
        noemer = np.ones(Nf,'complex')
        
        # Bereken de noemer middels een loop
        for m in range(Np):
            noemer *= (z - poles_val[m])
        
        # Bereken de teller middels een loop
        for n in range(Nz):
            teller *= (z - zeros_val[n])
    
        # Bereken overdracht
        H_freq = K * z**(Np-Nz) * teller / noemer
        
    elif ( mode == 'scipy' ):
        # Aanroep van scipy functie
        Omega_dum,H_freq = signal.freqz_zpk(zeros_val,poles_val,K,Omega,False)
        
        # Test of omega_dum gelijk is aan omega
        tst.assert_allclose(Omega_dum,Omega, atol=1e-7 )

  
    # Return output
    return H_freq


#%% ==========================================================================#
def lti_val_to_zeros_poles( a_val, b_val ):
    ''' Functie die obv de LTI coefficienten de polen en nulpunten bepaalt.
      
De LTI coefficienten a_val en b_val staan in formule (3.9) uit Wang:
   a[0]*y[n] + a[1]*y[n-1] + a[2]*y[n-2] + ... + a[N]*y[n-N] =
   b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] + ... + b[M]*x[n-M]
            
Bij berekenen frequentie-respons wordt deze formule als volgt genoteerd
(zie vgl (4.8) uit Wang):
        
   H(Omega) = Y(Omega) / X(Omega)] = 
   (b[0] + b[1]*z^(-1) + ... + b[M]*z^(-M)) / 
   (a[0] + a[1]*z^(-1) + ... + a[N]*z^(-N))
  
met z = exp(j*Omega).

Neem Q = max( M,N ).

De zeros zijn de nulpunten van de teller, nadat deze met z^Q zijn 
vermenigvuldigd:
    b[0]*z^Q + b[1]*z^(Q-1) + ... + b[M]*z^(Q-M) = 0

De poles zijn de nulpunten van de noemer nadat deze met z^Q zijn 
vermenigvuldigd:
    a[0]*z^Q + a[1]*z^(Q-1) + ... + a[N]*z^(Q-N) = 0
          
Invoer
------
   a_val: array met N+1 variabelen die corresponderen met a in de  
   formule hierboven
   
   b_val: array met M+1 variabelen die corresponderen met b in de 
   formule hierboven
   
Uitvoer
-------
   K: versterkingsfactor
   
   zeros_val: (meestal complexe) array met M waarden van de nulpunten
   
   poles_val: (meestal complexe) array met N waarden van de polen 
'''
    # Op dit moment alleen scipy implementatie werkend
    mode = 'scipy'
    if ( mode == 'scipy' ):
        # Bepaal enkele parameters
        N = len(a_val) - 1
        M = len(b_val) - 1
        Q = max( M,N )
        
        # Maak arrays met lengte Q+1
        a_val_ext = np.zeros( Q+1 )
        b_val_ext = np.zeros( Q+1 )
        a_val_ext[0:N+1] = a_val
        b_val_ext[0:M+1] = b_val
        
        # Bepaal de polen, nulpunten en K
        zeros_val, poles_val, K = signal.tf2zpk( b_val_ext, a_val_ext )
        
    else:
        # Bepaal order van teller en noemer
        N = len( a_val ) - 1
        M = len( b_val ) - 1
        
        # Teller om zeros te bepalen
        t = Polynomial( b_val[::-1] )
        zeros_val = t.roots()
       
        # Noemer om polen te bepalen
        n = Polynomial( a_val[::-1] )
        poles_val = n.roots()
        
        K = np.nan   # Nog niet geimplementeerd
   
    # Return uitvoer
    return K, zeros_val, poles_val

#%% ==========================================================================#
def lti_coeff_discr2cont( a_val, b_val, dt, method=1 ):
    ''' Functie die obv de discrete LTI coefficienten de waarden van de 
    continue overdrachtscoefficienten bepaalt.
      
De LTI coefficienten a_val en b_val staan in formule (3.9) uit Wang:
   a[0]*y[n] + a[1]*y[n-1] + a[2]*y[n-2] + ... + a[N]*y[n-N] =
   b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] + ... + b[M]*x[n-M]
            
Bij berekenen frequentie-respons wordt deze formule als volgt genoteerd
(zie vgl (4.8) uit Wang), ook wel de discrete overdrachtsfunctie:
        
   H(Omega) = Y(Omega) / X(Omega)] = 
   (b[0] + b[1]*z^(-1) + ... + b[M]*z^(-M)) / 
   (a[0] + a[1]*z^(-1) + ... + a[N]*z^(-N))
  
met z = exp(j*Omega) = ex[(j*omega*dt) = exp(s*dt) en s = j*omega, met omega
de dimensievolle frequentie. Neem Q = max( M,N ).

Omschrijven leidt tot de volgende continue overdrachtsfunctie

    H(s) = (b_c[0] + b_c[1]*s^(1) + ... + b_c[Q]*s^(Q)) / 
    (a_c[0] + a_c[1]*s^(1) + ... + a_c[Q]*s^(Q))

met uitvoer-coefficienten a_val_cont = a_c en b_val_cont = b_c

Er zijn 2 manieren voor het omschrijven geimplementeerd:
    
Method = 1:
    
    Toepassing van Taylor-reeks:
    z^p = exp(p*dt*s) = sum_{k=0}^Q (p*dt*s)^k / k!
   
Method = 2:   
    
    Toepassing van de benadering (eerste orde Taylor-reeks)
    z^p = (1 + s*dt)^p = sum_{k=0}^p binom(p,k) * (p*dt)^k
          
Invoer
------
   a_val: array met N+1 variabelen die corresponderen met a in de  
   formule hierboven voor de discrete overdrachtsfunctie
   
   b_val: array met M+1 variabelen die corresponderen met b in de 
   formule hierboven voor de discrete overdrachtsfunctie
   
   dt: discrete tijdstap
   
   method: gelijk aan 1 (default) of 2
   
Uitvoer
-------
   a_val_cont: array met N+1 variabelen die corresponderen met a in de  
   formule hierboven voor de continue overdrachtsfunctie
   
   b_val_cont: array met M+1 variabelen die corresponderen met b in de 
   formule hierboven voor de continue overdrachtsfunctie
'''
    # Bepaal enkele parameters
    N = len(a_val) - 1
    M = len(b_val) - 1
    Q = max( M,N )
    
    # Maak arrays met lengte Q+1
    a_val_ext = np.zeros( Q+1 )
    b_val_ext = np.zeros( Q+1 )
    a_val_ext[0:N+1] = a_val
    b_val_ext[0:M+1] = b_val
    
    # Initialiseer uitvoer-arrays met lengte Q+1
    a_val_cont = np.zeros( Q+1 )
    b_val_cont = np.zeros( Q+1 )
    
    if ( method == 1 ):
        # Taylor-reeks. Loop om alle coefficienten te bepalen
        k_factorial = 1    # k-factorial k!
        for k in range( 0, Q+1 ):
            # Bepaal de coefficienten
            for i_a in range( 0,N+1 ):
                a_val_cont[k] += a_val_ext[i_a] * (Q-i_a)**k
            for i_b in range( 0,M+1 ):
                b_val_cont[k] += b_val_ext[i_b] * (Q-i_b)**k
            #
            k_factorial   *= max(1,k)    # max is nodig ivm de situatie voor k=0        
            a_val_cont[k] *= dt**k / k_factorial
            b_val_cont[k] *= dt**k / k_factorial
        
    elif ( method == 2 ):
        # Eerste orde Taylor reeks: loop om alle coefficienten te bepalen
        for k in range( 0, Q+1 ):
            for i in range( k, Q+1 ):
                bn             = special.binom( i,k )
                a_val_cont[k] += a_val_ext[Q-i] * bn
                b_val_cont[k] += b_val_ext[Q-i] * bn
            #
            a_val_cont[k] *= (dt)**k
            b_val_cont[k] *= (dt)**k
    
    # Return uitvoer
    return a_val_cont, b_val_cont
   

#%% ==========================================================================#
#%% ==========================================================================#
def zeros_poles_to_lti_val( K, zeros_val, poles_val ):
    ''' Functie die obv de polen en nulpunten de LTI coefficienten bepaalt.
        
Formule (5.4) uit Wang:
    H(Omega) = Y(Omega) / X(Omega)] = 
    K * (z-zeros_val[0]) * (z-zeros_val[1]) * ... * (z-zeros_val[Q-R-1])  
    /((z-poles_val[0]) * (z-poles_val[1]) * ... * (z-poles_val[Q-1]))   
   
met z = exp(j*Omega), en zeros_val en poles_val de nulpunten van de teller 
en noemer.

Uitvermenigvuldigen geeft (zie vgl (4.8) uit Wang):
        
   H(Omega) = Y(Omega) / X(Omega)] = 
   (b[0] + b[1]*z^(-1) + ... + b[M]*z^(-M)) / 
   (a[0] + a[1]*z^(-1) + ... + a[N]*z^(-N))
  
De LTI coefficienten a_val en b_val staan in formule (3.9) uit Wang:
   a[0]*y[n] + a[1]*y[n-1] + a[2]*y[n-2] + ... + a[N]*y[n-N] =
   b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] + ... + b[M]*x[n-M]
   
Neem Q = max( M,N ). 

De waarde van R wordt bepaald door het aantal b-coefficienten aan begin van
array b_val dat gelijk is aan 0. Als b_val = [0,0,0,x,...], met x ongelijk aan
nul, dan geldt: R = 3. Bij een minimum-fase systeem geldt: R = 0, en dan geldt:
aantal polen = aantal nulpunten = Q

Invoer
-------
   K: versterkingsfactor
   
   zeros_val: (meestal complexe) array met Q-R waarden van de nulpunten
   
   poles_val: (meestal complexe) array met Q waarden van de polen 
   
Uitvoer
------
   a_val: array met N+1 variabelen die corresponderen met a in de  
   formule hierboven
   
   b_val: array met M+1 variabelen die corresponderen met b in de 
   formule hierboven
   
'''
    # Op dit moment alleen scipy implementatie werkend
    mode = 'scipy'
    if ( mode == 'scipy' ):
        # Het aantal elementen aan begin van b dat gelijk is aan 0
        R = len(poles_val) - len(zeros_val)
        
        # Bepaal de LTI coefficienten
        [b_val,a_val] = signal.zpk2tf( zeros_val, poles_val, K )
        
        # Voeg nul-elementen toe aan het begin bij b
        for i in range(R):
            b_val = np.append( 0,b_val )
        
    else:
        # Onderstaande is nog niet afgerond
        
        # Bepaal de gevraagde coefficienten
        a_val_tmp = Polynomial.fromroots( poles_val )
        b_val_tmp = Polynomial.fromroots( zeros_val )
        
        a_val = a_val_tmp.coef[::-1]
        b_val = b_val_tmp.coef[::-1]
        
        K = np.nan   # Berekening van K moet nog geimplementeerd worden
   
    # Return uitvoer
    return a_val, b_val  

#%% ==========================================================================#
def convolution(x_inp,y_inp,mode='scipy'):
    ''' Functie die de lineaire convolutie van twee signalen x_inp en y_inp 
    bepaalt
    NB Voor cyclische convolutie, zie dsp.convolution_cyclic
        
Formule:
   z[n] = x[n]*y[n] = sum(k=-\inf to inf) x[k]*y[n-k] =
   sum(k=0 to N-1) x[k]*y[n-k]
   voor alle n = 0,1,2,...,N-1.
           
Hierbij nemen we x en y gelijk aan nul voor n<0 en n>N-1. Dat betekent dus
dat het signaal NIET cyclisch verondersteld wordt. Het is dus een lineaire 
convolutie. Hierin is N = max(Nx,Ny), dus het maximum van het aantal elementen 
in arrays x_inp en y_inp. 
            
Invoer
-----
   x_inp: array met Nx elementen 
   
   y_inp: array met Ny elementen 
   
   mode: geeft aan hoe de lineaire convolutie berekend wordt. Alle methoden 
   geven hetzelfde resultaat, maar verschillen in rekentijd
   
        mode = 'scipy': gebruikt module scipy, en is (voor grote arrays)
        het snelst  
        
        mode = 'numpy': gebruikt module numpy
              
        mode = 'explicit': dit is een code die expliciet laat zien hoe
        de lineaire convolutie berekend wordt. Dit is de traagste van de drie

Uitvoer
------
   z_out: array met N elementen met lineaire convolutie van x_inp en y_inp
        '''
    # Bepaal enkele constante waarden
    Nx = len(x_inp)
    Ny = len(y_inp)
    N  = max(Nx,Ny) 
    # Pad shorter array with zeros
    x_inpc = np.zeros(N)
    y_inpc = np.zeros(N)
    x_inpc[0:Nx] = x_inp
    y_inpc[0:Ny] = y_inp
    
    # Hieronder staan drie stukken code die hetzelfde resultaat geven
    if ( mode == 'explicit' ):
        # Code 1. Expliciete berekening
        # Initialiseer uitvoer
        z_out = np.zeros(N)
        
        # Bereken de convolutie middels een loop. Bedenkt hierbij dat x en y 
        # gelijk zijn aan nul voor n<0 en n>N-1 (niet cyclisch dus)
        for n in range(0,N):
            for k in range( 0,min(N,n)+1 ):
                z_out[n] += x_inpc[k]*y_inpc[n-k]
            
    elif ( mode == 'numpy' ):
        # Code 2. Met Numpy
        z_out_full = np.convolve(x_inpc,y_inpc,mode='full')
        z_out = z_out_full[0:N]
        
    elif ( mode == 'scipy' ):
        # Code 3. Met scipy.
        # NB signal.fftconvolve geeft hetzelfde resultaat, maar signal.convolve
        # bepaalt zelf welke berekeningswijze (hetzij directe berekening, 
        # hetzij via FFT) het snelst is
        z_out_full = signal.convolve(x_inpc,y_inpc,mode='full')
        z_out = z_out_full[0:N]
  
    # Return output
    return z_out


#%% ==========================================================================#
def convolution_cyclic(x_inp,y_inp,mode='numpy'):
    ''' Functie die de cyclische convolutie van twee signalen x_inp en y_inp 
    bepaalt
    NB Voor lineaire convolutie, zie dsp.convolution
        
Formule:
   z[n] = x[n]*y[n] = sum(k=-\inf to inf) x[k]*y[n-k] =
   sum(k=0 to n) x[k]*y[n-k] + sum(k=n+1 to N-1) x[k]*y[N+n-k]
   voor alle n = 0,1,2,...,N-1.
           
Hierbij nemen we x en y cyclisch, met periode N. Hierin is N = max(Nx,Ny), 
dus het maximum van het aantal elementen in arrays x_inp en y_inp. Cyclisch
betekent: x[n] = x[n+N] en y[n] = y[n+N]
            
Invoer
-----
   x_inp: array met Nx elementen 
   
   y_inp: array met Ny elementen 
   
   mode: geeft aan hoe de cyclische convolutie berekend wordt. Alle methoden 
   geven hetzelfde resultaat, maar verschillen in rekentijd
           
        mode = 'numpy': gebruikt module numpy, en de eigenschap dat cyclische 
        convolutie x*y identiek is aan inverse FFT van product van Xf en Yf, 
        met Xf en Yf de FFT van x en y. Voor grote arrays is deze het snelst.
              
        mode = 'explicit': dit is een code die expliciet laat zien hoe
        de cyclische convolutie berekend wordt. Dit is de traagste van de twee


Uitvoer
------
   z_out: array met N elementen met cyclische convolutie van x_inp en y_inp
        '''
    # Bepaal enkele constante waarden
    Nx = len(x_inp)
    Ny = len(y_inp)
    N  = max(Nx,Ny) 
    # Pad shorter array with zeros
    x_inpc = np.zeros(N)
    y_inpc = np.zeros(N)
    x_inpc[0:Nx] = x_inp
    y_inpc[0:Ny] = y_inp
    
    # Hieronder staan twee stukken code die hetzelfde resultaat geven
    if ( mode == 'explicit' ):
        # Code 1. Expliciete berekening
        # Initialiseer uitvoer
        z_out = np.zeros(N)
        
        # Bereken de cyclische convolutie middels een loop. Bedenkt hierbij dat 
        # x en y cyclisch zijn, dus x[n] = x[n+N] en y[n] = y[n+N]
        for n in range(0,N):
            for k in range( 0,n+1 ):
                z_out[n] += x_inpc[k]*y_inpc[n-k]
            for k in range( n+1,N ):
                z_out[n] += x_inpc[k]*y_inpc[N+n-k]
            
    elif ( mode == 'numpy' ):
        # Code 2. Numpy
        x_fft   = np.fft.fft(x_inpc)
        y_fft   = np.fft.fft(y_inpc)
        xy_fft  = x_fft * y_fft
        z_out   = np.real( np.fft.ifft( xy_fft ) )
          
    # Return output
    return z_out

#%% ==========================================================================#
def crosscorr_circ( x1, x2, mode='scipy' ):
    '''
    Circular cross correlation.
    
    Input:
        x1: 1D array with N1 elements
        
        x2: 1D array with N2 elements
        
        mode: indicates the way in which the cross-correlation is computed.
        All modes should yield the same results
            mode = 'scipy': uses scipy functions, and is fastest
            
            mode = 'numpy': uses numpy functions
            
            mode = 'explicit': is based on a code in which the cross
            correlation is computed using visible loops. Aim is to
            make commputation of correlation visible
                     
    Output
        x12: 1D array with N=max(N1,N2) elements containing circular 
        cross-correlation.
        
    Input are two 1D arrays x1 and x2, with N1 and N2 elements. Let 
    N = max(N1,N2). The shorter array is padded with zeros, so that both arrays
    have length N. This leads to (extended) arrays x1c and x2c. 
    The circular cross correlation is obtained by computing the correlation 
    between x1c and x2c over all elements N, where x1c and x2c are circularly 
    extended (i.e., are periodic with period N).
    Output is correlation x12, which has N elements, and is also periodic with
    period N. 
    
    Circular cross-correlation is defined as followed:
        x12[p] = ( sum_{n} x1c[ np.mod(n+p,N) ] * x2c[n] ) 
    where p has N elements running from 0 to N (p = 0,1,...,(N-1)), and the 
    summation runs over N elements n = 0,1,...,(N-1). 
    
    Example 1. Suppose N=50 and the maximum of x12 lies at element 10. Then
    signal x2 runs 10 elements ahead of signal x1. Equivalently, x2 runs 40
    elements behind x1.
    
    Example 2. Suppose N=50 and the maximum of x12 lies at element 45. Then
    signal x2 runs 5 elements behind signal x1. Equivalently, x2 runs 45 
    elements ahead of x1.    
    '''
    # Number of elements
    N1 = len(x1)
    N2 = len(x2)
    N  = max(N1,N2)
    # Pad shorter array with zeros
    x1c = np.zeros(N)
    x2c = np.zeros(N)
    x1c[0:N1] = x1
    x2c[0:N2] = x2
    
    # Below there are three pieces of code that aim at doing the same. 
    if ( mode == 'explicit' ):
        # Code 1. Explicit computation. 
        # THe code below shows explicitly what is happening. But the 
        # disadvantage is that it is slow for large data sets
        
        # Initialize
        x12 = np.zeros(N)
        # Loop to compute cross correlation
        for p in range(N):
            for n in range(N):
                x12[p] += x1c[ np.mod(n+p,N) ]*x2c[n]
                
    elif ( mode == 'numpy' ):
        # Code 2. Using numpy functions
        # This code is faster
        
        # Extend array x2c in order to ensure circular computation
        x2c_ext = np.concatenate((x2c,x2c))  
        x12 = np.correlate(x1c,x2c_ext,mode='valid')
        # The last element of x12 must be omitted
        x12 = x12[0:-1]
        
    elif ( mode == 'scipy' ):
        # Code 3. Using scipy functions
        # This code is faster for large arrays
        
        # Extend array x2c in order to ensure circular computation
        x2c_ext = np.concatenate((x2c,x2c))  
        x12 = signal.correlate(x1c,x2c_ext,mode='valid')
        # The last element of x12 must be omitted
        x12 = x12[0:-1]
        
    return x12

#%% ==========================================================================#
def z_transform(x_inp,z_val):
    ''' Functie die de z-transformatie van array x bepaalt voor gegeven 
(complexe) waarden van z
        
Formule:
   X_out(z) = sum(n=0 to and including N-1) x_inp[n]*z_val^(-n)

Hierbij nemen we x gelijk aan nul voor n>N-1. Hierin is
N het aantal elementen in arrays x_inp en y_inp
    
Invoer
-----
   x_inp: array met N (reele) elementen 
   
   z_val: array met (complexe) getalswaarden z

Uitvoer
------
   X_out: array met (complexe) getalswaarden van de z-transformatie
   van x_inp. Het aantal elementen is gelijk aan het aantal elementen 
   in z_val
        '''
    # Bepaal enkele constante waarden
    N_x    = len(x_inp)
    N_z    = len(z_val)
    z_min1 = 1/z_val                  # Array met z^(-1) waarden
    z_minN = np.ones(N_z,'complex')   # Array [z^(0),z^(-1),...,z^(-N_x+1)]
    
    # Initialiseer uitvoer
    X_out = np.zeros(N_z)
    
    # Bereken de convolutie middels een loop
    for n in range(0,N_x):
        X_out  = X_out + x_inp[n]*z_minN
        z_minN = z_minN*z_min1     # Array met waarden [z^(-n),z^(-n),...]
  
    # Return output
    return X_out

#%% ==========================================================================#
def dft_idft_core(x,N,scaling,sign_power):
    ''' Functie die het rekenhart van de DFT (discrete fourier transformatie) 
        en de IDFT (inverse discrete fourier transformatie) bepaalt
    '''

    # Bepaal de term exp( sign_power * 2*pi*k*n/N ) voor k=n=1. 
    # Hierbij wordt gebruik gemaakt van: exp(j*fase) = cos(fase) + j*sin(fase),
    # met j de imaginaire eenheid
    fase         = sign_power * (2.0*np.pi/N)
    exp_term_1_1 = np.complex( np.cos(fase), np.sin(fase) )
    
    # Initialiseer uitvoer
    x_out = np.zeros(N,'complex')

    # Bereken de (I)DFT middels een loop
    for n in range(N):
        for k in range(N):
            exp_term_n_k = exp_term_1_1**(k*n)
            x_out[k]     = x_out[k] + x[n]*exp_term_n_k
            
    # Pas schaling toe
    x_out = scaling*x_out
    
    # Return output
    return x_out

#%% ==========================================================================#
def dft(x_time):
    ''' Functie die de DFT (discrete fourier transformatie) van tijdsignaal
x_time bepaalt. Deze functie dsp.dft doet hetzelfde als numpy.fft.fft
        
Formule:
   x_freq(k) = sum(n=0 to N-1) x_time[n]*exp(-j*2*pi*k*n/N)

Het aantal elementen in x_time is gelijk aan N. Het aantal elementen in
x_freq is ook gelijk aan N
    
Invoer
-----
   x_time: input tijd-array met N elementen 

Uitvoer
-------
   X_freq: array met N elementen, bevattende de DFT van x_time
        '''
    # Bepaal enkele constante waarden
    Nt         = len(x_time)
    scaling    = 1.0      # Schaalfactor waarmee de som vermenigvuldigd wordt
    sign_power = -1.0     # Teken in de complexe e-macht

    # Bereken DFT in het rekenhart
    x_freq = dft_idft_core(x_time,Nt,scaling,sign_power)
    
    # Return output
    return x_freq

#%% ==========================================================================#
def idft(x_freq):
    ''' Functie die de IDFT (inverse discrete fourier transformatie) van 
frequentiesignaal x_freq bepaalt. Deze functie doet hetzelfde als 
numpy.fft.ifft
        
Formule:
   x_time(k) = (1/N) sum(n=0 to N-1) x_freq[n]*exp(+j*2*pi*k*n/N)

Het aantal elementen in x_freq is gelijk aan N. Het aantal elementen in
x_time is ook gelijk aan N
    
Invoer
-----
   x_freq: input frequentie-array met N elementen 

Uitvoer
------
   x_time: array met N elementen, bevattende de IDFT van x_freq
        '''
    # Bepaal enkele constante waarden
    Nf         = len(x_freq)
    scaling    = 1.0/Nf   # Schaalfactor waarmee de som vermenigvuldigd wordt
    sign_power = 1.0      # Teken in de complexe e-macht

    # Bereken DFT in het rekenhart
    x_time = dft_idft_core(x_freq,Nf,scaling,sign_power)
    
    # Return output
    return x_time

#%% ==========================================================================#
def tijdsignaal_naar_spectrum(t,x_time):
    ''' Functie die een tijdsignaal omzet naar een enkelzijdig complex 
spectrum.        
        
Invoer
------
   t: tijd-as met Nt equidistante elementen
   
   x_time: tijd-array met Nt elementen die het tijdsignaal beschrijven
   
Uitvoer
------
   f: frequentie-as met Nf equidistante elementen
   
   x_freq: array met Nf elementen, bevattende de DFT van x_time
   
In de berekening van de FFT, wordt het laatste element van x_time overgeslagen
in het geval dat het aantal elementen in x_time oneven is. Met dit overslaan
van het laatste element moet ook rekening gehouden worden bij het berekenen
van de frequentie as   
        '''
        
    # Bepaal enkele constanten
    Nt = len(t)
    dt = t[1] - t[0]
    if ( Nt % 2 == 0 ):
        # Nt is even
        Nt_e = Nt
    else:
        # Nt is oneven: sla het laatste element van de tijdserie over
        Nt_e = Nt - 1
    
    # Bereken RFFT met numpy module
    x_freq  = np.fft.rfft(x_time[0:Nt_e])
    
    # Creeer frequentie as (N.B. het weggecommentarieerde deel geeft exact 
    # hetzelfde als freq_rfft)
        # df = 1/(Nt_e*dt)
        # Nf = Nt_e/2  + 1  # aantal frequenties
        # f  = np.arange(Nf)*df   # frequentie as
    f = np.fft.rfftfreq( Nt_e, dt )
    
    # Return output
    return f, x_freq

#%% ==========================================================================#
def maak_complex_spectrum(f,fun_abs_S, **kwargs):
    ''' Functie die een complex spectrum maakt op basis van een gegeven 
frequentie as, een functienaam en bijbehorende argumenten. Het complexe 
spectrum krijgt random fasen voor alle frequenties
    
Invoer
------
    f : frequentie-as met Nf equidistante elementen
    
    fun_abs_S : een functie die door de gebruiker gemaakt moet zijn dat
    de modulus van het spectrum als functie van f en eventueel
    andere argumenten berekent
           
    **kwargs : de eventueel andere argumenten die horen bij fun_abs_S

Uitvoer
-------
    S_complex : een complex spectrum voor alle frequenties in f
           
Voorbeeld
---------
>>> def fun_abs_spectrum(f, **kwargs ):   
>>> # Functie voorschrift voor modulus van het spectrum 
>>> a     = kwargs.get('a', None)
>>> b     = kwargs.get('b', None)
>>> S_abs = a*f + b
>>> return S_abs
>>>   
>>> f = np.linspace(0,2.0,20)
>>> S_complex = dsp.maak_complex_spectrum(f,fun_abs_spectrum, a=5.0, 
    b=4.0, seed_val = 2 )            
        
        '''
        
    # Bepaal aantal frequenties
    Nf = len(f)
    
    # Kies een seed voor random getallen. Default is 0
    seed_val = kwargs.get('seed_val', 0)
    np.random.seed(seed=seed_val)
    
    # Creeer Nf random fasen met waarden tussen 0 en 2*pi
    random_phase = np.random.uniform(0.0,2.0*np.pi,Nf)
    
    # Bereken absolute waarde (grootte = modulus) van spectrum als functie van 
    # frequentie f en eventueel andere argumenten
    S_abs = fun_abs_S(f, **kwargs)
    
    # Bereken complexe spectrum door de random fasen mee te nemen
    S_complex = S_abs * np.exp( 1j * random_phase )
    
    # Return output    
    return S_complex

#%% ==========================================================================#
def complex_spectrum_naar_tijdsignaal1(f,x_freq):
    ''' Functie die een enkelzijdig complex spectrum inclusief gegeven 
frequentie-as omzet naar een tijdsignaal met bijpassende tijd-as   
        
Invoer
-----
   f: frequentie-as met Nf equidistante elementen
   
   x_freq: array met Nf elementen, bevattende de DFT van x_time

Uitvoer
------
   t: tijd-as met Nt equidistante elementen
   
   x_time: tijd-array met Nt elementen 
        '''
        
    # Bepaal enkele constanten
    Nf = len(f)
    df = f[1] - f[0]
    
    # Bereken RFFT met numpy module
    x_time  = np.fft.irfft(x_freq)
    
    # Creeer tijd as 
    Nt   = 2*Nf - 2
    Tend = 1/df
    dt   = Tend/Nt
    t    = np.arange(Nt)*dt

    # Return output
    return t, x_time

#%% ==========================================================================#
def complex_spectrum_naar_tijdsignaal2(t,f,x_freq):
    ''' Functie die een enkelzijdig complex spectrum inclusief gegeven 
frequentie-as omzet naar een tijdsignaal op een gegeven tijdas.        
        
Invoer
------
   t: tijd-as met Nt equidistante elementen
       
   f: frequentie-as met Nf equidistante elementen
   
   x_freq: array met Nf elementen, bevattende de DFT van x_time.

Uitvoer
-------
   x_time: tijdsignaal in array met Nt elementen op tijd-as t
        '''
        
    # Bepaal enkele constanten
    Nf         = len(f)
    x_time     = np.zeros( len(t) )
    fase1      = 2.0*np.pi*f[1]*t
    exp_fase_1 = np.exp( 1j*fase1 )
    # Aantal tijdstappen horende bij Nf. Na Nt_rfft tijdstappen zal het 
    # gecreerde tijdsignaal periodiek zijn. Merk op dat Nt_rfft in het 
    # algemeen ongelijk is aan len(t)
    Nt_rfft    = 2*Nf - 2   
    
    # Bepaal RFFT mbv sommatie over alle frequentie componenten
    for ifreq in range(Nf):
        # Component f[ifreq] = ifreq*f[1]
        exp_fase_n = exp_fase_1**ifreq
        # Add this frequentie component
        x_time = x_time + x_freq[ifreq]*exp_fase_n
        
    # Pas schaling toe en neem daarvan het reele deel. De factor 2 is 
    # afkomstig van het feit dat we als invoer het enkelzijdige spectrum hebben
    # De factor 2 neemt dus de invloed van het (even grote) deel van het 
    # spectrum boven de Nyquist frequentie mee
    x_time = np.real(2.0*x_time) / Nt_rfft

    # Return output
    return x_time

#%% ==========================================================================#
def fun_real_spectrum_naar_tijdsignaal(t,fun_abs_S, **kwargs):
    ''' Functie die een tijdserie maakt obv een gewenste tijdas en een door 
de gebruiker gegeven functie van de modulus van een spectrum. De 
gekozen fases zijn random, maak de seed kan mbv parameter seed_val 
worden geinitialiseerd

Invoer
------
   t: tijd-as met Nt equidistante elementen
   
   fun_abs_S: een functie die door de gebruiker gemaakt moet zijn dat
   de modulus van het spectrum als functie van f en eventueel
   andere argumenten berekent
   
   **kwargs: de eventueel andere argumenten die horen bij fun_abs_S en
   voor seed_val. Deze argumenten zijn nodig voor de aanroep van
   dsp.maak_complex_spectrum en dsp.complex_spectrum_naar_tijdsignaal1

Uitvoer
-------
   x_time: tijd-as met Nt equidistante elementen die het tijdsignaal 
   beschrijven
   
Voorbeeld van een aanroep
-------------------------
>>> # Door de gebruiker gemaakte functie van spectrum
>>> def fun_abs_spectrum2(f, **kwargs ):
>>> f_s = kwargs.get('f_signal', None)
>>> S_abs = np.zeros( len(f))
>>>  for i_f_val in range(len(f)):
>>>     if ( f[i_f_val] == f_s ):
>>>         S_abs[i_f_val] = 2.0
>>> return S_abs
>>> 
>>> t = np.linspace(0,50.0,201)        
>>> z = dsp.fun_real_spectrum_naar_tijdsignaal(t,fun_abs_spectrum2, 
        f_signal = 0.5, seed_val = 1 )
        '''
    # Bepaal enkele constanten
    dt      = t[1] - t[0]            # Tijdstap
    Nt      = len(t)                 # Aantal tijd elementen
    Nt_even = 2*np.int(Nt/2)         # Gelijk aan Nt als Nt even, 
                                     #   gelijk aan Nt-1 als Nt oneven
    df      = 1.0/(Nt_even*dt)       # Frequentie stap
    Nf      = 1 + np.int(Nt_even/2)  # Aantal frequenties
    f       = np.arange(Nf)*df       # Frequentie as
    x_time  = np.zeros(Nt)           # Initialiseer uitvoer met gewenste aantal
                                     #   elementen
    
    # Bereken complexe spectrum
    S_cmplx = maak_complex_spectrum(f,fun_abs_S, **kwargs )
    
    # Bereken de eerste Nt_even elementen van x_time
    t_even,x_time[0:Nt_even] = complex_spectrum_naar_tijdsignaal1(f,S_cmplx)
    
    # Indien Nt oneven is, maak x_time periodiek door het laatste element 
    # gelijk aan het eerste elementen te kiezen
    if ( Nt - Nt_even == 1 ):
        x_time[Nt_even] = x_time[0]
        
    # Controleer of tijdas correct is
    if ( max( abs(t[0:Nt_even] - t_even) ) > 0.1*dt ):
        print("\n=== FOUTMELDING BIJ AANROEP VAN " + 
              "DSP.SPECTRUM_NAAR_TIJDSIGNAAL2 ===")
        print("Foutmelding: Fout in tijd-arrays")
        return
        
    # Return output
    return x_time

#%% ==========================================================================#
def real_spectrum_naar_tijdsignaal(t,f,S_abs,seed_val=0):
    ''' Functie die een tijdserie maakt obv een gewenste tijdas en een door 
de gebruiker gegeven (reel) spectrum S op frequentie-as f. De 
gekozen fases zijn random, maak de seed kan mbv parameter seed_val 
worden geinitialiseerd

Invoer
------
   t: tijd-as met Nt equidistante elementen
   
   f: frequentie-as met Nf equidistante elementen
   
   S_abs: spectrum (absolute waarde) met Nf elementen
   
   seed_val: optioneel argument dat de waarde van de seed geeft. Default is 0
   
Uitvoer
-------
   x_time: tijd-as met Nt equidistante elementen die het tijdsignaal 
   beschrijven
        '''
        
    # Bepaal aantal frequenties
    Nf = len(f)
    
    # Kies een seed voor random getallen. Default is 0
    np.random.seed(seed=seed_val)
    
    # Creeer Nf random fasen met waarden tussen 0 en 2*pi
    random_phase = np.random.uniform(0.0,2.0*np.pi,Nf)
        
    # Bereken complexe spectrum door de random fasen mee te nemen
    S_complex = S_abs * np.exp( 1j * random_phase )

    # Zet complexe spectrum om naar tijdserie
    x_time = complex_spectrum_naar_tijdsignaal2(t,f,S_complex)
    
    # Return output
    return x_time

#%% ==========================================================================#
def real_spectrum_constante_blokken(f, f_ranges, S_vals):
    ''' Functie die een stuksgewijze constant reeel spectrum (modulus) maakt 
op een gegeven frequentie-as. De stuksgewijs constante waarden worden
door middel van de tuples f_range en S_val bepaald.

Invoer
-----
   f: frequentie-as met Nf equidistante elementen
   
   f_ranges: lijst bestaande uit Ns lijsten met ieder 2 elementen. Het
   eerste element is een ondergrens en het tweede element is een 
   bovengrens van de frequentie
   
   S_vals: lijst met Ns elementen. Ieder element hoort bij de 
   corresponderende lijst uit f_range

Uitvoer
-------
   S_abs: stuksgewijs constante spectrum (absolute waarde) met Nf 
   elementen
   
Voorbeeld1 van een aanroep
--------------------------
>>> f = np.linspace(0,5.0,100)       
>>> f_ranges = [[1.2,3.0],[3.3,3.6],[4.2,6.0]]     
>>> S_vals   = [12.1, 33.1, 43.8]
   
   Dit leidt tot spectrum dat gelijk is aan 0.0 voor frequenties tussen
   0.0 en 1.2, 3.0 en 3.3, en 3.6 en 4.2. Het spectrum is gelijk aan 
   12.1 voor frequenties tussen 1.2 en 3.0 (inclusief deze waarden). 
   Het spectrum is gelijk aan 33.1 voor frequenties tussen 3.3 en 3.6 
   (inclusief deze waarden). Het spectrum is gelijk aan 
   43.8 voor frequenties tussen 4.2 en 5.0 (inclusief deze waarden).
   
Voorbeeld 2 van een aanroep (laagdoorlaatfilter)
-------------------------------------------------
>>> f = np.linspace(0,5.0,100)       
>>> f_ranges = [[0.0,2.2]]        
>>> S_vals   = [10.0] 
   
   Cut-off frequentie is 2.2.
 
Voorbeeld 3 van een aanroep (hoogdoorlaatfilter)
------------------------------------------------
>>> f = np.linspace(0,5.0,100)       
>>> f_ranges = [[3.2,6.0]]    
>>> S_vals   = [10.0]
   
   Cut-off frequentie is 3.2.

        '''
    # Bepaal enkele constanten
    Nf    = len( f )              # Aantal frequenties
    Ns    = len( f_ranges )       # Aantal constante blokken
    S_abs = np.zeros( Nf )        # Initialiseer uitvoer spectrum

    # Controleer invoer
    if ( Ns != len(S_vals) ):
        print("\n=== FOUTMELDING BIJ AANROEP VAN " + 
              "DSP.REAL_SPECTRUM_CONSTANTE_BLOKKEN ===")
        print("Foutmelding: Aantal lijsten in f_ranges is niet gelijk aan " +
              "aantal elementen in S_vals.")
        return

    # Loop over alle frequenties om spectrum te vullen
    for ifreq in range(Nf):
        f_val = f[ifreq]       # Frequentie waarde
        # Loop over alle constante blokken
        for iss in range(Ns):
            f_range = f_ranges[iss]
            if ( f_val >= f_range[0] and f_val <= f_range[1] ):
                S_abs[ifreq] += S_vals[iss] 
         
    # Return output
    return S_abs

#%% ==========================================================================#
def real_spectrum_pieken(f, f_peaks, S_peak_vals):
    ''' Functie die een reel spectrum maakt op een gegeven frequentie-as, 
waarbij het spectrum bestaat uit discrete (Dirac) pieken op frequenties 
gegeven in f_peaks en de waarde van de spectrumpieken is gegeven in
S_peaks_vals.

Invoer
------
   f: frequentie-as met Nf equidistante elementen
   
   f_peaks: lijst bestaande uit Np frequenties waar de pieken liggen.
   
   S_peak_vals: lijst met Np waarden die de spectrale waarde op de 
   frequenties in f_peaks bevat

Uitvoer
-------
   S_abs: gewenste spectrum (absolute waarde) met Nf elementen
   
   f_peaks_in_S_abs: lijst bestaande uit Np frequenties waar de pieken in
   spectrum S_abs liggen
   
Omdat frequentie-as f discreet is, zullen in het algemeen de waarden in
f_peaks niet precies op de waarden in f liggen. Daarom worden de pieken 
geplaatst op de frequenties in f die het dichtsbij de gegeven waarden 
in f_peaks liggen. De precieze frequentie-waarden van de pieken in 
S_abs worden in f_peaks_in_S_abs weggeschreven
        '''
    # Bepaal enkele constanten
    Nf               = len( f )              # Aantal frequenties
    Np               = len( f_peaks )        # Aantal pieken
    S_abs            = np.zeros( Nf )        # Initialiseer uitvoer spectrum
    f_peaks_in_S_abs = np.zeros( Np )        # Uitvoer frequentiewaarden

    # Controleer invoer
    if ( Np != len(S_peak_vals) ):
        print("\n=== FOUTMELDING BIJ AANROEP VAN " + 
              "DSB.REAL_SPECTRUM_PIEKEN ===")
        print("Foutmelding: Aantal elementen in f_peaks is niet gelijk aan " +
              "aantal elementen in S_peak_vals.")
        return

    # Loop over alle frequentie pieken
    for ip in range(Np):
        f_peak = f_peaks[ip]        # Frequentie waarde van piek
        
        dist_f = abs( f - f_peak )  # Bepaal afstand van alle frequenties tot 
                                    #   f_peak
        ip_min = np.argmin(dist_f)  # Argum van f dichtsbij f_peak_val
        
        # Vul uitvoer waarden in
        S_abs[ip_min]        = S_peak_vals[ip]
        f_peaks_in_S_abs[ip] = f[ip_min]
                 
    # Return output
    return S_abs, f_peaks_in_S_abs

#%% ==========================================================================#
def window(N,win_type):
    ''' Functie die een window creeert
            
Invoer
------
   N : aantal elementen in de array
   
   win_type: string die het type window aangeeft. 
   Toegestaan zijn: 
       - 'rectangular': geen window
       - 'triangular': driehoekswindow
       - 'hamming': Hamming window
       - 'blackmann': Blackmann window

Uitvoer
-------
   w: numpy array met N elementen dat het window bevat
        
        '''
    # Creeer window afhankelijk van win_type
    if ( win_type == 'rectangular' ):
        w = np.ones(N)
        
    elif ( win_type == 'triangular' ):
        n = np.arange(N)
        w = 1 - np.abs(2*n - N + 1) / N
        
    elif ( win_type == 'hamming' ):
        n = np.arange(N)
        # w = 0.54 + 0.46*np.cos( (2*n - N + 1)*np.pi / N )    Wang formula
        w = 0.54 - 0.46*np.cos( 2*n*np.pi / (N-1) )   # DSP guide and wiki

    elif ( win_type == 'blackmann' ):
        n = np.arange(N)
        w = 0.42-0.5*np.cos( 2*n*np.pi/(N-1) )+0.08*np.cos( 4*n*np.pi/(N-1) )

    else:
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.WINDOW ===")
        print("Foutmelding: parameter win_type is niet correct.")
    
    # Return output
    return w

#%% ==========================================================================#
def studentnummer_all(studentnummer):
    ''' Functie die een studentnummer (bestaande uit 8 cijfers) omzet naar
de acht individuele opeenvolgende getallen (deze getallen zijn natuurlijk 
allemaal tussen 0 en 9).
    
Invoer
------
    studentnummer : studentnummer bestaande uit 8 cijfers
    
Uitvoer
-------
    stud_nr_1 : eerste cijfer uit studentnummer
    
    stud_nr_2 : tweedee cijfer uit studentnummer
    
    stud_nr_3 : derde cijfer uit studentnummer
    
    stud_nr_4 : vierde cijfer uit studentnummer
    
    stud_nr_5 : vijfde cijfer uit studentnummer
    
    stud_nr_6 : zesde cijfer uit studentnummer
    
    stud_nr_7 : zevende cijfer uit studentnummer
    
    stud_nr_8 : achtste cijfer uit studentnummer
    
Voorbeeld
---------
    >>> studentnummer = 19239845
    >>> s1,s2,s3,s4,s5,s6,s7,s8 = dsp.studentnummer_all(studentnummer)
    
    Dan is: s1 = 1, s2 = 9, s3 = 2, s4 = 3, s5 = 9, s6 = 8, s7 = 4 en s8 = 5
    
    '''

    # Check of studentnummer een getal is dat uit 8 cijfers bestaat
    if ( studentnummer < 10000000 or studentnummer > 99999999 ):
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.STUDENTNUMMER_ALL ===")
        print("Foutmelding: Het opgegeven studentnummer voldoet niet.")
 
    # Zet studentnummer om naar string en haal hier vervolgens het getal uit
    stud_nr_str = str(studentnummer)
    stud_nr_1   = int( stud_nr_str[0] )
    stud_nr_2   = int( stud_nr_str[1] )
    stud_nr_3   = int( stud_nr_str[2] )
    stud_nr_4   = int( stud_nr_str[3] )
    stud_nr_5   = int( stud_nr_str[4] )
    stud_nr_6   = int( stud_nr_str[5] )
    stud_nr_7   = int( stud_nr_str[6] )
    stud_nr_8   = int( stud_nr_str[7] )

    # Return output
    return stud_nr_1, stud_nr_2, stud_nr_3, stud_nr_4, stud_nr_5, stud_nr_6, \
    stud_nr_7, stud_nr_8 
    
#%% ==========================================================================#
def studentnummer_one(studentnummer, nummer ):
    ''' Functie die een cijfer uit een studentnummer (bestaande uit 8 cijfers) 
geeft. 
    
Invoer
------
    studentnummer : studentnummer bestaande uit 8 cijfers
    
    nummer        : getal tussen 1 en 8 dat het hoeveelste cijfer aangeeft
    uit het studentnummer dat uitgevoerd moet worden
    
    
Uitvoer
-------
    stud_nr : cijfer uit studentnummerdat uitgevoerd wordt    
    
Voorbeeld
---------
    >>> studentnummer = 19239845
    >>> stud_nr = dsp.studentnummer_one(studentnummer,4)
    
    Dan is: stud_nr = 3, want dit is het vierde cijfer in het studentnummer
    
    '''

    # Check of studentnummer een getal is dat uit 8 cijfers bestaat
    if ( studentnummer < 10000000 or studentnummer > 99999999 ):
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.STUDENTNUMMER_ONE ===")
        print("Foutmelding: Het opgegeven studentnummer voldoet niet.")

    # Check of nummer een getal is tussen 1 en 8
    if ( nummer < 1 or nummer > 8 ):
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.STUDENTNUMMER_ONE ===")
        print("Foutmelding: Het opgegeven nummer voldoet niet.")

    # Zet studentnummer om naar string en haal hier vervolgens het getal uit.
    # Bedenk dat het eerste cijfer gelijk is aan element [0], het tweede cijfer
    # gelijk aan element [1], etc.
    stud_nr_str = str(studentnummer)
    stud_nr     = int( stud_nr_str[nummer-1] )

    # Return output
    return stud_nr

#%% ==========================================================================#
def fig_plot(fig_info,fig_style):
    ''' Functie om een plot te maken van N verschillende datasets.
            
Invoer
------
fig_info is een dictionary met de volgende keys:
    'xlabel': default value 'x' als fig_style='lineplot' of 'stemplot'
    default value '$\Omega$' als fig_style='freqplot'

    'ylabel': default value 'y' als fig_style='lineplot' of 'stemplot'
    default value '|G($\Omega$)|' als fig_style='freqplot'
    
    'title' : default value: leeg
    
    'xdata' : default value: integer arrays beginnend bij 0 en passend
    bij de lengte van de array(s) in 'ydata'. M.a.w. gelijk 
    aan [0,1,2,3,...,(lengte van y-data) - 1]
    user value: lijst bestaande uit N arrays van de 
    onafhankelijke variabele horende bij 'ydata'
              
    'ydata' : default value: is er niet. Gebruiker MOET 'ydata' geven
    user value: lijst bestaande uit N data arrays
              
    'markerstyle': default value: python markerstyle
    user value: lijst bestaande uit N markerstyles
              
    'label' : default value: 'data0', 'data1', data2'
    user value: lijst bestaande uit N labels
    
    'filename': default value: leeg.
    Als er geen filename is, dan wordt het figuur niet weggeschreven
    naar file. Als er wel een filename is, dan wordt het figuur
    weggeschreven naar file.

fig_style is een string met 'lineplot', 'stemplot' of 'freq_plot'
    .
    
Voorbeeld 1 van functie-aanroep voor 'lineplot' of 'stemplot'
-------------------------------------------------------------
>>> x1 = np.linspace(0,10,50)
>>> y1 = np.sin(x1)   
>>> x2 = np.linspace(0,9,40)   
>>> y2 = np.cos(x2)     
>>> fig_info = {'xlabel':'x', 'ylabel':'y', 'title':'sinus en cosinus',
    'xdata':[x1,x2], 'ydata':[y1,y2], 
    'markerstyle':['ro','k*'], 'label':['sin(x)','cos(x)'],
    'filename':'figure01.png'}
>>> fig_style = 'lineplot'
>>> dsp.fig_plot(fig_info,fig_style)

Voorbeeld 2 van functie-aanroep voor 'freqplot'
----------------------------------------------
>>> omega = np.linspace(0,2*np.pi,50)
>>> H = 0.5/(1 - 0.5*np.exp(-1j*omega))
>>> fig_info = {'xlabel':'$\Omega$', 'ylabel':'H($\Omega$)',
    'title':'abs and argument freqplot',
    'xdata':[omega], 'ydata':[H], 'markerstyle':['r']},
    'label':['H'],'filename':'figure05.pdf'}
>>> dsp.fig_plot(fig_info,'freqplot')
        '''
        
    # Check correctheid van de data-invoer
    Ny = len(fig_info['ydata'])
    if ( 'xdata' in fig_info ):
        Nx = len(fig_info['xdata'])
        if ( Nx != Ny ):
            print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.FIG_PLOT ===")
            print("Foutmelding: Aantal arrays in xdata en ydata is ongelijk.")
            print("Pas invoer aan")
            return
    else:
        # Creeer integer-arrays voor horizontale data as
        xdata_array = []
        for ic in range(Ny):
            Nx = len( fig_info['ydata'][ic] )
            xdata_array.append( np.array(range(Nx)) )
        fig_info['xdata'] = xdata_array
        
    # Check compleetheid van invoer, en vul zonodig aan: xlabel
    if ( 'xlabel' in fig_info ):
        pass
    else:
        if ( fig_style == 'lineplot' or fig_style == 'stemplot' ): 
            fig_info['xlabel'] = 'x'
        elif ( fig_style == 'freqplot' ):
            fig_info['xlabel'] = '$\Omega$'
        
    # Check compleetheid van invoer, en vul zonodig aan: ylabel
    if ( 'ylabel' in fig_info ):
        pass
    else:
        if ( fig_style == 'lineplot' or fig_style == 'stemplot' ): 
            fig_info['ylabel'] = 'y'
        elif ( fig_style == 'freqplot' ):
            fig_info['ylabel'] = 'G($\Omega$)'


    # Check compleetheid van invoer, en vul zonodig aan: title
    if ( 'title' in fig_info ):
        pass
    else:
        fig_info['title'] = ' '
    
    # Check compleetheid en correctheid: markerstyle
    if ( 'markerstyle' in fig_info ):
        if ( Ny != len(fig_info['markerstyle']) ):
            print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.FIG_PLOT ===")
            print("Foutmelding: Aantal elementen in key 'markerstyle' komt niet ")
            print("   overeen met aantal data-arrays")
            print("Pas invoer aan")
            return
    else:
        markerstyle_list = []
        for ic in range(Ny):
            markerstyle_list.append('')
        fig_info['markerstyle'] = markerstyle_list

    # Check compleetheid en correctheid: label
    if ( 'label' in fig_info ):
        if ( Ny != len(fig_info['label']) ):
            print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.FIG_PLOT ===")
            print("Foutmelding: Aantal elementen in key 'label' komt niet ")
            print("   overeen met aantal data-arrays")
            print("Pas invoer aan")
            return
    else:
        label_list = []
        for ic in range(Ny):
            label_list.append('data' + str(ic))
        fig_info['label'] = label_list

    # Creeer de figuur
    if ( fig_style == 'lineplot' or fig_style == 'stemplot' ):
        plt.figure()
        for idata in range(Ny):
            if ( fig_style == 'lineplot' ):
                plt.plot(fig_info['xdata'][idata],fig_info['ydata'][idata],
                         fig_info['markerstyle'][idata],linestyle='-',
                         linewidth=1,label=fig_info['label'][idata])
            elif ( fig_style == 'stemplot' ):
                plt.stem(fig_info['xdata'][idata],fig_info['ydata'][idata],
                         linefmt='k--',markerfmt=fig_info['markerstyle'][idata],
                         use_line_collection=True,label=fig_info['label'][idata])
    
          
        plt.xlabel(fig_info['xlabel'], fontsize=12)
        plt.ylabel(fig_info['ylabel'], fontsize=12)
        plt.legend(loc=0, fontsize=15)
        #plt.xlim(0.0,1.0)
        plt.title(fig_info['title'], fontsize=12)
        plt.grid()
        plt.tight_layout()
        
    elif ( fig_style == 'freqplot' ):
        plt.figure()
        plt.subplot(2,1,1)
        for idata in range(Ny):
            plt.plot(fig_info['xdata'][idata],
                     np.abs( fig_info['ydata'][idata] ),
                     fig_info['markerstyle'][idata],linestyle='-',
                     linewidth=3,label=fig_info['label'][idata])
        plt.xlabel(fig_info['xlabel'], fontsize=12)
        plt.ylabel('|' + fig_info['ylabel'] + '|', fontsize=12)
        plt.legend(loc=0, fontsize=12)
        #plt.xlim(0.0,1.0)
        plt.title(fig_info['title'], fontsize=12)
        plt.grid()
        plt.tight_layout()
        
        plt.subplot(2,1,2)
        for idata in range(Ny):
            # Get angle and give it a value between -pi and +pi
            angle_val = np.angle( fig_info['ydata'][idata] )
            angle_val = np.remainder(angle_val+np.pi,2.0*np.pi) - np.pi
            
            plt.plot(fig_info['xdata'][idata],
                     angle_val,
                     fig_info['markerstyle'][idata],linestyle='-',
                     linewidth=3,label=fig_info['label'][idata])
        plt.xlabel(fig_info['xlabel'], fontsize=12)
        plt.ylabel('arg(' + fig_info['ylabel'] + ')', fontsize=12)
        plt.legend(loc=0, fontsize=12)
        plt.grid()
        plt.tight_layout()  

    # Eventueel opslaan van figuren en vervolgens het figuur ook tonen
    if ( 'filename' in fig_info ):
        plt.savefig(fig_info['filename'], dpi=300)
    plt.show()

#%% ==========================================================================#
def zeros_poles_plot(zeros_val,poles_val, filename = False):
    ''' Functie om een plot te maken van polen en nulpunten in het complexe 
vlak.

Invoer
----        
   zeros_val: array met Nz waarden van de nulpunten
   
   poles_val: array met Np waarden van de polen 
   
   filename: optioneel argument. Bevat (indien het bestaat) de naam
   van de file waarin de figuur wordt weggeschreven

Uitvoer
------
   (geen uitvoerargument) Figuur met poles en nulpunten      
        '''
    # Creeer eenheidscirkel
    phi    = np.linspace(0,2.0*np.pi,100)
    circle = np.exp(1j*phi)
    
    # Maak figuur
    plt.figure()
    plt.plot(np.real(circle),np.imag(circle),'k-')
    for izero in range(len(zeros_val)):
        plt.plot( np.real(zeros_val[izero]),np.imag(zeros_val[izero]), 'ko',
                 markersize = 15)
    for ipole in range(len(poles_val)):
        plt.plot( np.real(poles_val[ipole]),np.imag(poles_val[ipole]), 'kx', 
                 markersize = 15)
    plt.xlabel('reele as', fontsize=12)
    plt.ylabel('imaginaire as', fontsize=12)
    plt.grid()
    plt.title('Polen en nulpunten', fontsize=12)
    plt.axis('equal')
    plt.tight_layout()
    if ( filename != False ):
        plt.savefig(filename, dpi=300)
    plt.show()

#%% ==========================================================================#
def fig_4plot(fig_info,fig_style):
    ''' Functie om een plot te maken bestaande uit vier subfiguren.
In iedere subfiguur wordt een dataset getoond. Iedere subfiguur is een
'normale' x-y plot. Deze functie is geschikt voor het plotten van bijvoorbeeld
tijseries of van (absolute) waarde van spectrum als functie van frequentie.
            
Invoer
------
fig_info is een dictionary met de volgende keys:
    'xlabel': een lijst bestaande uit uit 4 strings. Iedere string hoort
    bij de x-as van een van de subfiguren

    'ylabel': een lijst bestaande uit uit 4 strings. Iedere string hoort
    bij de y-as van een van de subfiguren
    
    'title' : een lijst bestaande uit uit 4 strings. Iedere string geeft
    de titel van een van de subfiguren
    
    'xdata' : een lijst bestaande uit 4 arrays horende bij de x-data van ieder
    van de vier subfiguren'
              
    'xdata' : een lijst bestaande uit 4 arrays horende bij de y-data van ieder
    van de vier subfiguren'

    'markerstyle' : een lijst bestaande uit uit 4 strings. Iedere string geeft
    de marker van de data in van een van de subfiguren
    
    'filename': default value: leeg.
    Als er geen filename is, dan wordt het figuur niet weggeschreven
    naar file. Als er wel een filename is, dan wordt het figuur
    weggeschreven naar file.

fig_style is lijst bestaande uit 4 strings met 'lineplot', 'stemplot' of
    'poleszeros_plot'
    
NB. Wanneer fig_style = 'poleszeros_plot', dan worden de waarden voor de keys
'xlabel', 'ylabel' en 'markerstyle' niet gebruikt. Ze moeten wel opgegeven
worden, maar fungeren dus als dummy waarden. 
Verder moeten de (complexe) nulputen opgegeven worden bij de 'xdata', en de
(complexe) polen bij de 'ydata'
    .
    
    
Voorbeeld 1 van functie-aanroep voor 'lineplot' en 'stemplot'
-------------------------------------------------------------
>>> T    = 64.0
>>> dt   = 2.0
>>> A1   = 1.0
>>> T1   = 16.0
>>> phi1 = 0.0
>>> A2   = 2.0
>>> T2   = 18.0
>>> phi2 = 1.1
>>> t    = np.arange(0.0,T,dt)
>>> x1   = A1*np.sin(2.0*np.pi*t/T1 + phi1)
>>> x2   = A2*np.sin(2.0*np.pi*t/T2 + phi2)
>>> 
>>> # Berekenen spectra
>>> f1,z1 = dsp.tijdsignaal_naar_spectrum(t,x1)
>>> f2,z2 = dsp.tijdsignaal_naar_spectrum(t,x2)
>>> 
>>> fig_info = {'xlabel':['t [s]','t [s]','f [Hz]','f [Hz]'],
>>>            'ylabel':['x [m]','x [m]','|X| [m/Hz]','|X| [m/Hz]'], 
>>>            'title':['A','B','C','D'],
>>>            'xdata':[t,t,f1,f2], 
>>>            'ydata':[x1,x2,np.abs(z1),np.abs(z2)], 
>>>            'markerstyle':['k-..','r.','ko','r-o'],
>>>            'filename':'plot4_A.png'}
>>> fig_style = ['lineplot','lineplot','stemplot','stemplot']
>>> dsp.fig_4plot(fig_info,fig_style)

Voorbeeld 2 van functie-aanroep voor o.a. 'poleszeros_plot'
-----------------------------------------------------------
>>> zeros_val = [0.0]                     # Nulpunten
>>> poles_val = [-0.8]                     # Polen
>>> Omega     = np.linspace(0,np.pi,10)   # Dimensieloze hoekfrequentie
>>> K         = 1.0                       # Versterkingsfactor
>>>            
>>> # Bereken complexe frequentie-respons op eenheidsimpuls
>>> G = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val,poles_val)
>>> f = Omega/(2.0*np.pi)
>>> n,h = dsp.complex_spectrum_naar_tijdsignaal1(f,G)
>>> 
>>> # Creeer vier-plot
>>> fig_info = {'xlabel':['','$\Omega$ [-]','$\Omega$ [-]','n'],
>>>             'ylabel':['','!G!','arg|G|','h[n]'], 
>>>             'title':['A','B','C','D'],
>>>             'xdata':[zeros_val,Omega,Omega,n], 
>>>             'ydata':[poles_val,np.abs(G),np.angle(G),h], 
>>>             'markerstyle':['.','r.','ko','ro'],
>>>             'filestyle':'plot4_C.png'}
>>> fig_style = ['poleszeros_plot','lineplot','lineplot','stemplot']
>>> dsp.fig_4plot(fig_info,fig_style)
        '''
    
    # Check correctheid van de data-invoer
    if ( len(fig_info['xlabel']) != 4 ):
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.FIG_4PLOT ===")
        print("Foutmelding: Aantal elementen in key 'xlabel' is ")
        print("   ongelijk aan 4")
        print("Pas invoer aan")
        return
    if ( len(fig_info['ylabel']) != 4 ):
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.FIG_4PLOT ===")
        print("Foutmelding: Aantal elementen in key 'ylabel' is ")
        print("   ongelijk aan 4")
        print("Pas invoer aan")
        return
    if ( len(fig_info['title']) != 4 ):
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.FIG_4PLOT ===")
        print("Foutmelding: Aantal elementen in key 'title' is ")
        print("   ongelijk aan 4")
        print("Pas invoer aan")
        return
    if ( len(fig_info['xdata']) != 4 ):
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.FIG_4PLOT ===")
        print("Foutmelding: Aantal elementen in key 'xdata' is ")
        print("   ongelijk aan 4")
        print("Pas invoer aan")
        return
    if ( len(fig_info['ydata']) != 4 ):
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.FIG_4PLOT ===")
        print("Foutmelding: Aantal elementen in key 'ydata' is ")
        print("   ongelijk aan 4")
        print("Pas invoer aan")
        return
    if ( len(fig_info['markerstyle']) != 4 ):
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.FIG_4PLOT ===")
        print("Foutmelding: Aantal elementen in key 'markerstyle' is ")
        print("   ongelijk aan 4")
        print("Pas invoer aan")
        return
    if ( len(fig_style) != 4 ):
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.FIG_4PLOT ===")
        print("Foutmelding: Aantal elementen in fig_style is ")
        print("   ongelijk aan 4")
        print("Pas invoer aan")
        return

    # Creeer de figuur
    plt.figure()
    for ifig in range(4):
        plt.subplot(2,2,ifig+1)
        plt.xlabel(fig_info['xlabel'][ifig],fontsize=10)
        plt.ylabel(fig_info['ylabel'][ifig],fontsize=10)
        plt.title(fig_info['title'][ifig])
        # Stop data in de figuren, afhankelijk van het figuur-type
        if ( fig_style[ifig] == 'lineplot' ):
            # Lineplot
            plt.plot(fig_info['xdata'][ifig],fig_info['ydata'][ifig],
                    fig_info['markerstyle'][ifig], linestyle='-',linewidth=1)
        elif ( fig_style[ifig] == 'stemplot' ):
            # Stemplot
            plt.stem(fig_info['xdata'][ifig],fig_info['ydata'][ifig],
                     markerfmt=fig_info['markerstyle'][ifig],linefmt='k--',
                     use_line_collection=True)
        elif ( fig_style[ifig] == 'poleszeros_plot' ):
            # Plot met polen en nulpunten
            #    Creeer eenheidscirkel
            phi    = np.linspace(0,2.0*np.pi,100)
            circle = np.exp(1j*phi)
            #
            zeros_val = fig_info['xdata'][ifig]
            poles_val = fig_info['ydata'][ifig]
            #
            plt.plot(np.real(circle),np.imag(circle),'k-')
            for izero in range(len(zeros_val)):
                plt.plot( np.real(zeros_val[izero]),np.imag(zeros_val[izero]), 
                         'ko', markersize = 12)
            for ipole in range(len(poles_val)):
                plt.plot( np.real(poles_val[ipole]),np.imag(poles_val[ipole]),  
                         'kx',markersize = 12)
            plt.axis('equal')
            plt.xlabel('reele as',fontsize=10)
            plt.ylabel('imag as',fontsize=10)

        # Voor iedere figuur een grid en tight layout
        plt.grid()
        plt.tight_layout()   
    
    # Eventueel opslaan van figuren en vervolgens het figuur ook tonen
    if ( 'filename' in fig_info ):
        plt.savefig(fig_info['filename'], dpi=300)
    plt.show()



#%% ==========================================================================#
#%% ========   Hieronder staan de functies alleen voor docenten   ============#
#%% ==========================================================================#
def moving_average_signal(x,M,avg_type='one-sided'):
    ''' Functie die de moving average met M punten voor een signal x bepaalt.
    
Er zijn twee types voor de moving average geimplementeerd: 
    
    avg_type = 'one-sided':
        y[i] = (x[i-M+1] + ... x[i-1] + x[i]) / M
    avg_type = 'symmetric':
        y[i] = (x[i-p] + x[i-p+1] + ... x[i+p-1] + x[i+p]) / M, met p = (M-1)/2
        
We gaan er van uit dat de waarden x[-1], x[-2], ... en x[N], x[N+1], ... allen
gelijk aan nul zijn.

Zie hoofdstuk 15 van de DSP guide.

Invoer
-----
   x: invoersignaal bestaande uit N elementen
   
   M: het aantal punten in het bepalen van het gemiddelde. Indien 
   avg_type = 'symmetric', dan moet M een oneven getal zijn
   
   avg_type: de manier waarop de moving average bepaald wordt:
       
       avg_type = 'one-sided': Dit is een causale bepaling van de moving 
       average, waarbij alleen signalen uit het verleden mee genomen worden
       
       avg_type = 'symmetric': een symmetrische bepaling van het moving average 
   
Uitvoer
-------
   y: het moving average signaal, bestaande uit N elementen
   
        '''    
    
    # Initialiseer
    N = len(x)         # Aantal elementen in array x
    y = np.zeros(N)    # Initialiseer uitvoer array
    
    # Bepaal de moving average
    if ( avg_type == 'one-sided' ):
        s = 0.0
        for i in range(M):
            s += x[i]
            y[i] = s
        for i in range(M,N):
            s += x[i] - x[i-M]
            y[i] = s
        y = y / M  # Pas schaling toe
        
    elif ( avg_type == 'symmetric' ):
        # Verifieer dat M een oneven getal is
        if ( (avg_type == 'symmetric') and (np.mod(M,2) == 0) ):
            print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.MOVING_AVERAGE_SIGNAL ===")
            print("Foutmelding: Invoerargument M is een even getal, en dat is ")
            print("in combinatie met avg_type = symmetric is niet toegestaan. ")
            print("Pas invoer aan")
            return
        p = int((M-1)/2)
        s = 0.0
        for i in range(p):
            s += x[i]
        for i in range(p+1):
            s += x[i+p]
            y[i] = s
        for i in range(p+1,N-p):
            s += x[i+p] - x[i-p-1]
            y[i] = s
        for i in range(N-p,N):
            s -= x[i-p-1]
            y[i] = s
        y = y / M # Pas schaling toe
        
    else:
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.MOVING_AVERAGE_SIGNAL ===")
        print("Foutmelding: Invoerargument avg_type is niet correct ")
        print("Pas invoer aan")
        return
    
    # --- Return uitvoer
    return y
    

#%% ==========================================================================#
def moving_average_lti_val(M,avg_type='one-sided'):
    ''' Functie die waarden voor LTI coefficienten a_val en b_val bepaalt voor 
de moving average filter met M punten.      

Er zijn twee types voor de moving average geimplementeerd:
    
    avg_type = 'one-sided':
        y[i] = (x[i-M+1] + ... x[i-1] + x[i]) / M
    avg_type = 'symmetric':
        y[i] = (x[i-p] + x[i-p+1] + ... x[i+p-1] + x[i+p]) / M, met p = (M-1)/2
        
We gaan er van uit dat de waarden x[-1], x[-2], ... en x[N], x[N+1], ... allen
gelijk aan nul zijn.       
Uitvoer zijn de waarden van a_val en b_val (formule (3.9) uit Wang):
   a[0]*y[n] + a[1]*y[n-1] + a[2]*y[n-2] + ... + a[N]*y[n-N] =
   b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] + ... + b[M-1]*x[n-M+1]
    
Invoer
-----
   M: het aantal punten in het bepalen van het gemiddelde. Indien 
   avg_type = 'symmetric', dan moet M een oneven getal zijn
   
   avg_type: de manier waarop de moving average bepaald wordt:
       
       avg_type = 'one-sided': Dit is een causale bepaling van de moving 
       average, waarbij alleen signalen uit het verleden mee genomen worden
       
       avg_type = 'symmetric': een symmetrische bepaling van het moving average 
   
Uitvoer
-------
   a_val: array met N+1 variabelen die correspondeert met a in de  
   formule hierboven
   
   b_val: array met M variabelen die correspondeert met b in de 
   formule hierboven
   
        '''    

    # Array b_val
    b_val = np.ones(M) / M
    
    # Bepaal array a_val
    if ( avg_type == 'one-sided' ):
        a_val = np.array( [1.0] )
        
    elif ( avg_type == 'symmetric' ):
        # Verifieer dat M een oneven getal is
        if ( (avg_type == 'symmetric') and (np.mod(M,2) == 0) ):
            print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.MOVING_AVERAGE_SIGNAL ===")
            print("Foutmelding: Invoerargument M is een even getal, en dat is ")
            print("in combinatie met avg_type = symmetric is niet toegestaan. ")
            print("Pas invoer aan")
            return
        p = int((M-1)/2)
        a_val = np.zeros(p+1)
        a_val[p] = 1.0
        
    else:
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.MOVING_AVERAGE_SIGNAL ===")
        print("Foutmelding: Invoerargument avg_type is niet correct ")
        print("Pas invoer aan")
        return
    
    # --- Return uitvoer
    return a_val, b_val

#%% ==========================================================================#
def windowed_sinc_lti_val(M,fc,fs,K=1.0,win_type='blackmann'):
    ''' Functie die waarden voor LTI coefficienten a_val en b_val bepaalt voor
de windowed sinc filters

Zie hiervoor bijvoorbeeld H16 van de DSP guide
    
Invoer
-----
   M: even getal dat gerelateerd is aan het totaal aantal punten in de filter
   (wat gelijk is aan (M+1)) 
   
   fc: cut-off frequentie, in Hertz
   
   fs: sample frequentie, in Hertz
   
   K : versterkingsfactor (default = 1.0)
   
   win_type: string die het type window aangeeft. 
   Toegestaan zijn: 
       - 'blackmann': Blackmann window (default waarde)
       - 'rectangular': geen window
       - 'triangular': driehoekswindow
       - 'hamming': Hamming window
       
Uitvoer
-------
   a_val: array met LTI coefficienten a voor de windowed sinc filters. Hier:
   a_val [1]
   
   b_val: array met LTI coefficienten b voor de windowed sinc filters. Hier:
   b_val bestaat uit in totaal M+1 coefficienten
   
        '''    
    # Verifieer dat M een even getal is
    if ( np.mod(M,2) == 1 ):
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.WINDOWED_SINC_LTI_VAL ===")
        print("Foutmelding: Invoerargument M is een oneven getal ")
        print("Verander dit getal in een even getal")
        return

    # Relatieve (hoek)frequentie
    fc_rel  = fc / fs
    Omega_c = 2.0 * np.pi * fc_rel
    
    # Hulpvariabele dat de lengte van b_val aangeeft
    N = M + 1
    
    # Bepaal window
    w = window(N,win_type)
    
    # Bepaal sinc filter, en daarmee b_val. NB np.sinc(x) = sin(pi*x) / (pi*x)
    n     = np.arange(N)
    x     = 2 * fc_rel * ( n - M/2 )    # Argument of sinc-function
    b_val = Omega_c * np.sinc(x) * w    # Yet without normalisation
    C     = np.sum( b_val )             # Normalisation factor
    b_val = K * b_val / C               # Normalise b_val, and apply K
    
    # Bepaal LTI coefficient a
    a_val = np.ones(1)
    
    # --- Return uitvoer
    return a_val, b_val

#%% ==========================================================================#
def spectral_inversion(a_val_in,b_val_in):
    ''' Functie die, obv gegeven LTI coefficienten, de bijberhorende LTI 
coefficienten voor de spectrale inversie bepaalt. 

Gegeven de LTI coefficienten a_val_in en b_val_in van een filter (bv low-pass 
filter). Bij een willekeurige invoer x geeft dit filter een uitvoer x_in (dus
bijvoorbeeld een signaal met alleen lage frequenties). Het spectraal inverse 
filter is een filter (met coefficienten a_val_out en b_val_out) dat als uitvoer 
x_out geeft, zodanig dat x = x_in + x_out.
Dus wanneer a_val_in en b_val_in bij een low-pass filter horen, dan horen
a_val_out en b_val_out bij een high-pass filter.
    
Invoer
-------
   a_val_in: array met N+1 variabelen die corresponderen met de LTI 
   coefficienten voor a van het invoerfilter
   
   b_val_in: array met M+1 variabelen die corresponderen met de LTI 
   coefficienten voor b van het invoerfilter
   
Uitvoer
-------
   a_val_uit: array met N+1 variabelen die corresponderen met de LTI 
   coefficienten voor a van de spectraal inverse van het invoerfilter
   
   b_val_uit: array met max(N,M)M+1 variabelen die corresponderen met de LTI 
   coefficienten voor b van de spectraal inverse van het invoerfilter
        '''    
    # Bepaal N, M en P = max(M,N)
    N = len( a_val_in ) - 1
    M = len( b_val_in ) - 1
    P = max(M,N)
    
    # Initialiseer hulp arrays met nullen
    a_val_tmp = np.zeros( P+1 )
    b_val_tmp = np.zeros( P+1 )
    
    # Vul hulp arrays
    a_val_tmp[0:N+1] = a_val_in
    b_val_tmp[0:M+1] = b_val_in
    
    # Bepaal de LTI coefficienten voor de spectrale inverse
    a_val_out = a_val_in
    b_val_out = a_val_tmp - b_val_tmp
    
    # --- Return uitvoer
    return a_val_out, b_val_out


#%% ==========================================================================#
def lti_val_dsp_to_default(a_val_dsp,b_val_dsp):
    ''' Functie die LTI coefficient waarden a_val en b_val, zoals gedefinieerd
in de DSP Guide, omzet naar LTI coefficient waarden zoals gebruikt in deze
module.

De DSP guide (zie met name Hoofdstuk 19) gebruikt een andere definitie voor de
filtercoefficienten a_val en b_val dan zoals gebruikt in de rest van deze 
module_dsp.py. De door ons gebruikte definitie is gebaseerd op (onder andere) 
het boek van Wang en de wikipedia pagina's over digitale filters (zie bijv 
https://en.wikipedia.org/wiki/Infinite_impulse_response). Om de aansluiting
met de DSP guide zo inzichtelijk mogelijk te maken (en om eventuele fouten te
vermijden), wordt in de op de DSP guide gebaseerde routines de notatie van de 
DSP guide gebruikt. Na aanroeping van de voorliggende routine verkrijgt de 
gebruiker de filter coefficienten zoals te gebruiken in de rest van 
module_dsp.py

Notuatie DSP guide (vergelijking 19-1)
    b_val_dsp[0]*y[n] - b_val_dsp[1]*y[n-1] - ... - b_val_dsp[N]*y[n-N] = 
    a_val_dsp[0]*x[n] + a_val_dsp[1]*x[n-1] + ... + a_val_dsp[M]*y[n-M]
Merk op dat coefficient b_val_dsp[0] in de DSP guide niet voorkomt. Deze
coefficient moet gelijk aan 1 zijn. Deze term is opgenomen in array b_val_dsp
omwille van de consistentie. Er geldt dus:
    a_val_dsp = [a0,a1,a2,...], met a0,a1,a2,... de a-coefficienten in (19-1)
    
    b_val_dsp = [ 1,b1,b2,...], met b1,b2,... de b-coefficienten in (19-1)
    
Notatie in deze module: 
    a_val_mod[0]*y[n] + a_val_mod[1]*y[n-1] + ... + a_val_mod[N]*y[n-N] = 
    b_val_mod[0]*x[n] + b_val_mod[1]*x[n-1] + ... + b_val_mod[M]*y[n-M]

Invoer
-------
   a_val_dsp: array met waarden die corresponderen met de LTI coefficienten 
   voor a zoals gedefinieerd in de DSP guide
   
   b_val_dsp: array met waarden die corresponderen met de LTI coefficienten 
   voor b zoals gedefinieerd in de DSP guide
       
Uitvoer
-------
   a_val_mod: array met waarden die corresponderen met de LTI coefficienten 
   voor a zoals gedefinieerd in de voorliggende module (module_dsp)
   
   b_val_mod: array met waarden die corresponderen met de LTI coefficienten 
   voor b zoals gedefinieerd in de voorliggende module (module_dsp)
   
        '''    
    # Verifieer dat b_val_dsp[0] gelijk aan 1 is
    if ( np.abs( b_val_dsp[0] - 1.0 ) > 1e-8 ):
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.LTI_VAL_DSP_TO_DEFAULT ===")
        print("Foutmelding: Invoerargument b_val_dsp[0] moet gelijk zijn aan 1 ")
        print("Pas dit aan")
        return

    # Bepaal de filter coefficienten
    a_val_mod    = -np.copy( b_val_dsp )
    a_val_mod[0] = b_val_dsp[0]
    b_val_mod    = np.copy( a_val_dsp )
    
    # --- Return uitvoer
    return a_val_mod, b_val_mod

#%% ==========================================================================#
def single_pole_lti_val(fc,fs,filter_type):
    ''' Functie die waarden voor LTI coefficienten a_val en b_val bepaalt voor
enkele single-pole filters

Zie hiervoor bijvoorbeeld H19 van de DSP guide. De conversie van de filter 
coefficienten naar de binnen module_dsp gebruikte notatie (a voor feedback, en
b voor feedfordward) wordt gedaan binnen dsp.single_pole_lti_val
    
Invoer
-----
   fc: cut-off frequentie, in Hertz
   
   fs: sample frequentie, in Hertz
   
   filter_type: string die het type filter aangeeft. 
   Toegestaan zijn: 
       - 'low-pass': single-pole low-pass filter (single stage) - DSP guide 
          eq. (19-2)
       - 'high-pass-dsp': single-pole high-pass filter (single stage) - DSP guide
          eq. (19-3)
       - 'high-pass': single-pole high-pass filter (single stage) - de spectrale
          inverse van 'low-pass'.
          NB 'high-pass' en 'high-pass-dsp' zijn niet identiek!
       - 'low-pass-4': single-pole low-pass filter (four stages) - DSP guide
          eq. (19-6)
       - 'high-pass-4': single-pole high-pass filter (four stages) - de 
          spectrale inverse van 'low-pass-4'
       
Uitvoer
-------
   a_val_mod: array met LTI coefficienten a voor de single-pole filter
   
   b_val_mod: array met LTI coefficienten b voor de single-pole filter
        '''    
    # Relatieve frequentie
    fc_rel = fc / fs
    
    # Parameter x (DSP guide, eq (19-5) en (19-6))
    if ( filter_type == 'low-pass-4' or filter_type == 'high-pass-4' ):
        x = np.exp( -14.445*fc_rel )
    else:
        x = np.exp( -2.0*np.pi*fc_rel )
    
    # Bepaal inhoud van de LTI filter arrays
    if ( filter_type == 'low-pass' ):
        # Low-pass - DSP guide, eq. (19-2)
        a0 = 1.0 - x
        b1 = x
        
        # Fill DSP coefficient arrays
        a_val_dsp = np.array([a0])
        b_val_dsp = np.array([1.0,b1])
        
        # Omzetting naar de binnen deze module gebruikelijke coefficienten
        a_val_mod, b_val_mod = lti_val_dsp_to_default(a_val_dsp,b_val_dsp)
        
    elif ( filter_type == 'high-pass-dsp' ):
        # High-pass - DSP guide, eq. (19-3)
        a0 =  (1.0 + x) / 2.0
        a1 = -(1.0 + x) / 2.0
        b1 = x
        
        # Fill DSP coefficient arrays
        a_val_dsp = np.array([a0,a1])
        b_val_dsp = np.array([1.0,b1])
        
        # Omzetting naar de binnen deze module gebruikelijke coefficienten
        a_val_mod, b_val_mod = lti_val_dsp_to_default(a_val_dsp,b_val_dsp)
        
    elif ( filter_type == 'high-pass' ):
        # High-pass - de spectrale inversie van het low-pass filter volgens DSP 
        # guide, eq. (19-2)
        a0 = 1.0 - x
        b1 = x
        
        # Fill DSP coefficient arrays
        a_val_dsp = np.array([a0])
        b_val_dsp = np.array([1.0,b1])
        
        # Omzetting naar de binnen deze module gebruikelijke coefficienten
        a_val_tmp, b_val_tmp = lti_val_dsp_to_default(a_val_dsp,b_val_dsp)
        
        # Spectrale inversie
        a_val_mod, b_val_mod = spectral_inversion(a_val_tmp, b_val_tmp)
        
    elif ( filter_type == 'low-pass-4' ):
        # Low-pass-4 - low-pass filter with 4 stages, DSP guide, eq. (19-6)
        a0 = (1 - x)**4
        b1 = 4*x
        b2 = -6*x**2
        b3 = 4*x**3
        b4 = -x**4
        
        # Fill DSP coefficient arrays
        a_val_dsp = np.array([a0])
        b_val_dsp = np.array([1.0,b1,b2,b3,b4])
        
        # Omzetting naar de binnen deze module gebruikelijke coefficienten
        a_val_mod, b_val_mod = lti_val_dsp_to_default(a_val_dsp,b_val_dsp)
      
    elif ( filter_type == 'high-pass-4' ):
        # High-pass-4 - de spectrale inversie van het low-pass filter met vier
        # stages, volgens DSP eq. (19-6)
        a0 = (1 - x)**4
        b1 = 4*x
        b2 = -6*x**2
        b3 = 4*x**3
        b4 = -x**4
        
        # Fill DSP coefficient arrays
        a_val_dsp = np.array([a0])
        b_val_dsp = np.array([1.0,b1,b2,b3,b4])
        
        # Omzetting naar de binnen deze module gebruikelijke coefficienten
        a_val_tmp, b_val_tmp = lti_val_dsp_to_default(a_val_dsp,b_val_dsp)
        
        # Spectrale inversie
        a_val_mod, b_val_mod = spectral_inversion(a_val_tmp, b_val_tmp)

    else:
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.SINGLE_POLE_LTI_VAL ===")
        print("Foutmelding: Invoerargument filter_type is niet gedefinieerd ")
        print("Pas dit invoerargument aan")
        return
    
    # --- Return uitvoer
    return a_val_mod, b_val_mod


#%% ==========================================================================#
def narrow_band_lti_val(fc,bw,fs,filter_type):
    ''' Functie die waarden voor LTI coefficienten a_val en b_val bepaalt voor
enkele narrow-band filters

Zie hiervoor bijvoorbeeld H19 van de DSP guide. De conversie van de filter 
coefficienten naar de binnen module_dsp gebruikte notatie (a voor feedback, en
b voor feedfordward) wordt gedaan binnen dsp.narrow_band_lti_val
    
Invoer
-----
   fc: cut-off frequentie, in Hertz
   
   bw: band-width frequentie, in hertz
   
   fs: sample frequentie, in Hertz
      
   filter_type: string die het type filter aangeeft. 
   Toegestaan zijn: 
       - 'band-pass': band-pass filter - DSP guide eq. (19-7)
       - 'band_reject': band-reject filter, also called a notch filter - DSP 
       guide eq. (19-8)
       
Uitvoer
-------
   a_val_mod: array met LTI coefficienten a voor de narrow-band filter
   
   b_val_mod: array met LTI coefficienten b voor de narrow-band filter
        '''    
    # Relatieve frequentie en relatieve bandbreedte
    fc_rel = fc / fs
    bw_rel = bw / fs
    
    # Parameters R en K
    R = 1 - 3*bw_rel
    c = np.cos( 2*np.pi*fc_rel )   # Vaak voorkomende cosinus term
    K = (1 - 2*R*c + R**2) / (2 - 2*c)
    
    # Bepaal inhoud van de LTI filter arrays
    if ( filter_type == 'band-pass' ):
        # Band-pass - DSP guide, eq. (19-7)
        a0 = 1 - K
        a1 = 2 * (K - R) * c
        a2 = R**2 - K
        b1 = 2 * R * c
        b2 = -R**2
        
        # Fill DSP coefficient arrays
        a_val_dsp = np.array([a0,a1,a2])
        b_val_dsp = np.array([1.0,b1,b2])
        
        # Omzetting naar de binnen deze module gebruikelijke coefficienten
        a_val_mod, b_val_mod = lti_val_dsp_to_default(a_val_dsp,b_val_dsp)
        
    elif ( filter_type == 'band-reject' ):
        # Band-reject, also called notch filter - DSP guide, eq. (19-8)
        a0 = K
        a1 = -2 * K * c
        a2 = K
        b1 = 2 * R * c
        b2 = -R**2
        
        # Fill DSP coefficient arrays
        a_val_dsp = np.array([a0,a1,a2])
        b_val_dsp = np.array([1.0,b1,b2])
        
        # Omzetting naar de binnen deze module gebruikelijke coefficienten
        a_val_mod, b_val_mod = lti_val_dsp_to_default(a_val_dsp,b_val_dsp)
        
    else:
        print("\n=== FOUTMELDING BIJ AANROEP VAN DSP.NARROW_BAND_LTI_VAL ===")
        print("Foutmelding: Invoerargument filter_type is niet gedefinieerd ")
        print("Pas dit invoerargument aan")
        return
    
    # --- Return uitvoer
    return a_val_mod, b_val_mod
#%% ==========================================================================#
