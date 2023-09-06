# -*- coding: utf-8 -*-
"""
Module met opdrachten

@author: wen
"""

import numpy as np
import module_dsp as dsp
import matplotlib.pyplot as plt

# =============================================================================
def dsp_notch_filter(f_0,f_c,dt,icoef_val=1):
    ''' Functie die enkele aspecten van een notch filter bepaald.
    
Voor een continu notch fitler geldt:     
    
H(s) = (s^2 + omega_0^2) / (s^2 + omega_c*s + omega_0^2)    

Daarin staat omega_0 voor de centrale hoekfrequentie van frequentie rejectie,
omega_c voor de breedte van de frequentie-rejectie, en s = j*omega.

Voor een digitaal filter geldt (Formule (3.9) uit Wang):
   a[0]*y[n] + a[1]*y[n-1] + a[2]*y[n-2] = b[0]*x[n] + b[1]*x[n-1] + b[2]*y[n-2]
            
Bij berekenen frequentie-respons wordt deze formule als volgt genoteerd
(zie vgl (4.8) uit Wang):
        
   H(Omega) = Y(Omega) / X(Omega)] = 
   (b[0]*z^2 + b[1]*z + b[2]) / (a[0]*z^2 + a[1]*z + a[2]) =
   (b[0] + b[1]*z^(-1) + b[2]*z^(-2)) / (a[0] + a[1]*z^(-1) + a[2]*z^(-2))
  
met z = exp(j*Omega).

Deze vergelijking is ook te schrijven als (vgl. (5.4) uit Wang):
    H(Omega) = Y(Omega) / X(Omega)] = 
    K*(z-zeros_val[0]) * (z-zeros_val[1])) /((z-poles_val[0]) * (z-poles_val[1]) 
  
met z = exp(j*Omega)
            
Invoer
-----
   f_0 : centrale frequentie (in Hertz) van frequentie rejectie, dwz een
   frequentie respons gelijk aan nul.
   
   f_c: breedte (in Hertz) van gesperde frequentieband
   
   dt: tijdstap van filter (in seconde)
   
   icoef_val: integer die aangeeft op welke manier de coefficienten a_val en 
   b_val berekend worden:
       icoef_val = 1: coefficienten berekend obv bilineaire transformatie
       icoef_val = 2: coefficienten berekend obv differenties

Uitvoer
-------
   a_val: array met 3 variabelen die corresponderen met a in de  
   formule hierboven
   
   b_val: array met 3 variabelen die corresponderen met b in de 
   formule hierboven
        
   zeros_val: (meestal complexe) array met 2 waarden van de nulpunten
   
   poles_val: (meestal complexe) array met 2 waarden van de polen 
   
   K: versterkingsfactor. Zodanig gekozen dat vgl. (5.4) getalsmatig dezelfde
   frequentieresponsie geeft als vgl.(4.8). (Zie hierboven voor de 
   vergelijkingen)
   
        '''    
        
    # --- Bereken enkele parameters
    w_0     = 2.0 * np.pi * f_0
    w_c     = 2.0 * np.pi * f_c
    Omega_0 = w_0 * dt
    Omega_c = w_c * dt
    
    # --- Initialiseer arrays met coefficienten
    a_val = np.zeros(3)
    b_val = np.zeros(3)
    
    # --- Bereken de coefficienten a_val en b_val
    if ( icoef_val == 1 ):   # Bilineaire transformatie
        # --- Enkele hulpvariabelen
        Omega_0_fac = 0.25 * Omega_0 * Omega_0
        Omega_c_fac = 0.50 * Omega_c
    
        # --- Vul arrays met coefficienten
        a_val[0] = 1.0 + Omega_c_fac + Omega_0_fac
        a_val[1] = -2.0 + 2.0*Omega_0_fac
        a_val[2] = 1.0 - Omega_c_fac + Omega_0_fac
        b_val[0] = 1.0 + Omega_0_fac
        b_val[1] = a_val[1]
        b_val[2] = b_val[0]
    elif ( icoef_val == 2 ):  # Differenties
        # --- Enkele hulpvariabelen
        Omega_0_fac = Omega_0 * Omega_0
        Omega_c_fac = Omega_c
    
        # --- Vul arrays met coefficienten
        a_val[0] = 1.0 + Omega_c_fac + Omega_0_fac
        a_val[1] = -2.0 - 2.0*Omega_0_fac
        a_val[2] = 1.0
        b_val[0] = 1.0 + Omega_0_fac
        b_val[1] = -2.0
        b_val[2] = 1.0
        #
    
    # --- Bereken polen en nulpunten
    zeros_val = np.roots( b_val )
    poles_val = np.roots( a_val )
    
    # --- Bepaal versterkingsfactor
    nom   = (1.0 - poles_val[0]) * (1.0 - poles_val[1])
    denom = (1.0 - zeros_val[0]) * (1.0 - zeros_val[1])
    K     = nom/denom
    
    # --- Schrijf parameter naar scherm
    #print('Omega_0 = ' + str(Omega_0) + '. Dit moet (veel) kleiner zijn dan 1')
    
    # --- Return uitvoer
    return a_val,b_val,zeros_val,poles_val,K
    

# =============================================================================


# =========================================================================== #
def dsp_opdracht001(studentnummer):

    # Creeer tijdas
    Nt = 60
    t = np.linspace(0,2.0,Nt,endpoint=False)
    
    # Verschillende componenten
    f1 = 2.0      # Frequentie van het tijdsignaal
    A1 = 1.5      # Amplitude
    phi1 = 0.6    # Fase
    
    f2 = 5.0      # Frequentie van het tijdsignaal
    A2 = 0.9      # Amplitude
    phi2 = 1.9    # Fase
        
    # Het tijdsignaal zelf
    x_tijd = A1*np.cos(2*np.pi*f1*t+phi1) +  A2*np.cos(2*np.pi*f2*t+phi2)
    
    # Bereken het complexe spectrum
    f,X_freq = dsp.tijdsignaal_naar_spectrum(t,x_tijd)
    
    # Return data
    return t,x_tijd,f,X_freq

# =========================================================================== #
def dsp_opdracht010(studentnummer):

    # --- Studentnummer
    s1,s2,s3,s4,s5,s6,s7,s8=dsp.studentnummer_all(studentnummer)
    
    # --- Maak een piek-spectrum ----
    f           = np.linspace(0,20.0,201)
    f_peaks     = [s1+2.0,s2+2.0,s3+2.0]    
    S_peak_vals = [3.0, 2.5, 2.0]
    S,f_peak_vals_in_S = dsp.real_spectrum_pieken(f, f_peaks, S_peak_vals)
    
    # --- Maak tijdserie
    t = np.linspace(0,6.0,201)
    x = dsp.real_spectrum_naar_tijdsignaal(t,f,S)
        
    # Return data
    return t,x

# =========================================================================== #
def dsp_opdracht020(studentnummer):

    # --- Maak een piek-spectrum ----
    f         = np.linspace(0,20.0,2001)   
    ff        = [3.0,13.0,16.1]
    f_ranges  = [[ff[0],ff[0]+2.0],[ff[1]-3.0,ff[1]],[ff[2]-0.1,ff[2]]]       
    S_vals    = [10,20,30] 
    S         = dsp.real_spectrum_constante_blokken(f, f_ranges, S_vals)

    # --- Maak tijdserie
    t = np.arange(0,40.0,0.02)
    x = dsp.real_spectrum_naar_tijdsignaal(t,f,S,studentnummer)
     
    # Return data
    return t,x

# =========================================================================== #
def dsp_opdracht020a(studentnummer):

    # --- Maak een piek-spectrum ----
    f         = np.linspace(0,20.0,2001)    
    f_ranges  = [[3.0,5.0],[10.0,13.0],[16.0,16.1]]       
    S_vals    = [10,20,30] 
    S         = dsp.real_spectrum_constante_blokken(f, f_ranges, S_vals)

    # --- Maak tijdserie
    t = np.arange(0,40.0,0.02)
    x = dsp.real_spectrum_naar_tijdsignaal(t,f,S,studentnummer)
    
    # Return data
    return t,x,f,S


## =========================================================================== #
def dsp_opdracht030(studentnummer):
    
    # --- Zet de seed
    seed_val = studentnummer
    
    # --- Maak een aantal band-spectra ----
    f = np.linspace(0,5.0,100)
    
    f_ranges1 = [[0,2.5]]       
    S_vals1   = [10] 
    S_abs1    = dsp.real_spectrum_constante_blokken(f, f_ranges1, S_vals1)
    
    f_ranges2 = [[2.5,4.7]]       
    S_vals2   = [10] 
    S_abs2    = dsp.real_spectrum_constante_blokken(f, f_ranges2, S_vals2)
    
    f_ranges3 = [[2.0,2.2]]       
    S_vals3   = [10] 
    S_abs3    = dsp.real_spectrum_constante_blokken(f, f_ranges3, S_vals3)
    
    f_ranges4 = [[4.5,4.7]]       
    S_vals4   = [10] 
    S_abs4    = dsp.real_spectrum_constante_blokken(f, f_ranges4, S_vals4)
    
    # --- Bepaal bijbehorende tijdseries
    t = np.linspace(0,10.0,2000)
    x1 = dsp.real_spectrum_naar_tijdsignaal(t,f,S_abs1,seed_val=0)
    x2 = dsp.real_spectrum_naar_tijdsignaal(t,f,S_abs2,seed_val=0)
    x3 = dsp.real_spectrum_naar_tijdsignaal(t,f,S_abs3,seed_val=0)
    x4 = dsp.real_spectrum_naar_tijdsignaal(t,f,S_abs4,seed_val=0)
    
    # f1,X_freq1 = dsp.tijdsignaal_naar_spectrum(t,x1)
    
    # --- Store enkele zaken in lijsten
    t_totaal     = [t,t,t,t]
    x_totaal     = [x1,x2,x3,x4]
    f_totaal     = [f,f,f,f]
    S_abs_totaal = [S_abs1,S_abs2,S_abs3,S_abs4]
    t_fig_label  = ['Fig 1','Fig 2','Fig 3','Fig 4']
    S_fig_label  = ['Fig A','Fig B','Fig C','Fig D']
    
    # --- Permuteer frequentie assen
    np.random.seed(seed=seed_val)
    f_perm_totaal = np.random.permutation( f_totaal )
    
    # --- Permuteer spectra
    np.random.seed(seed=seed_val)
    S_abs_perm_totaal = np.random.permutation( S_abs_totaal )
    
    # dsp_opdracht_perm01
    sol = dsp_opdracht_perm01(studentnummer,t_fig_label,S_fig_label)
    
    # Return data
    return t_fig_label,t_totaal,x_totaal,S_fig_label,f_perm_totaal, \
    S_abs_perm_totaal
    
## =========================================================================== #
def dsp_opdracht040(studentnummer):
    
    # Golfcomponent
    T   = 2.0      # Periode
    A   = 1.5      # Amplitude
    phi = 0.6      # Fase

    # Creeer tijdas
    dt = 0.2
    t1 = np.arange(0,5.0*T,dt)
    t2 = np.arange(0,5.5*T,dt)
            
    # Het tijdsignaal zelf
    x1 = A*np.cos(2*np.pi*t1/T+phi)
    x2 = A*np.cos(2*np.pi*t2/T+phi)
    
    # Bereken het complexe spectrum
    f1,X_freq1 = dsp.tijdsignaal_naar_spectrum(t1,x1)
    
    # Bereken het complexe spectrum
    f2,X_freq2 = dsp.tijdsignaal_naar_spectrum(t2,x2)
    
    # Return data
    return t1,t2,x1,x2,f1,f2,X_freq1,X_freq2
     
## =========================================================================== #
def dsp_opdracht041(studentnummer):
    
    # Golfcomponent
    T   = 0.8      # Periode
    A   = 1.5      # Amplitude
    phi = 0.6      # Fase

    # Creeer tijdas
    dt = 0.1
    t  = np.arange(0,7.5*T,dt)
            
    # Het tijdsignaal zelf
    x_orig = A*np.cos(2*np.pi*t/T+phi)
    
    # Bereken windows
    Nt = len(t)
    w_rect = dsp.window(Nt,'rectangular')
    w_tria = dsp.window(Nt,'triangular')
    w_hamm = dsp.window(Nt,'hamming')
    
    # Bereken tijdseries vermenigvuldigd met windows
    x_rect = x_orig * w_rect
    x_tria = x_orig * w_tria
    x_hamm = x_orig * w_hamm

    # Bereken het complexe spectrum
    f_orig,X_freq_orig = dsp.tijdsignaal_naar_spectrum(t,x_orig)
    f_rect,X_freq_rect = dsp.tijdsignaal_naar_spectrum(t,x_rect)
    f_tria,X_freq_tria = dsp.tijdsignaal_naar_spectrum(t,x_tria)
    f_hamm,X_freq_hamm = dsp.tijdsignaal_naar_spectrum(t,x_hamm)
    
    # Return data
    return t,x_orig,x_rect,x_tria,x_hamm,f_orig,f_rect,f_tria,f_hamm, \
    X_freq_orig,X_freq_rect,X_freq_tria,X_freq_hamm
   
## =========================================================================== #
def dsp_opdracht050(studentnummer):
    # --- Zet de seed
    seed_val = studentnummer
    
    # Omega en K
    Omega = np.linspace(0,np.pi,100)   # Dimensieloze hoekfrequentie
    K     = 1.0                        # Versterkingsfactor
    
    # Invoergegevens
    zeros_val1 = [0.0,0.0]
    poles_val1 = [0.9,-0.9]
    zeros_val2 = [0.9,-0.9]
    poles_val2 = [0.0,0.0]
    zeros_val3 = [0.5+0.5*1j,0.5-0.5*1j]
    poles_val3 = [0.0,0.0]
    zeros_val4 = [0.0,0.0]
    poles_val4 = [0.5+0.5*1j,0.5-0.5*1j]
    
               
    # Bereken complexe frequentie-respons op eenheidsimpuls
    G1 = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val1,poles_val1)
    G2 = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val2,poles_val2)
    G3 = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val3,poles_val3)
    G4 = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val4,poles_val4)
    
    # --- Store enkele zaken in lijsten
    zeros_totaal = [zeros_val1,zeros_val2,zeros_val3,zeros_val4]
    poles_totaal = [poles_val1,poles_val2,poles_val3,poles_val4]
    Omega_totaal = [Omega,Omega,Omega,Omega]
    G_abs_totaal = [np.abs(G1),np.abs(G2),np.abs(G3),np.abs(G4)]
    pz_fig_label = ['Fig 1','Fig 2','Fig 3','Fig 4']
    G_fig_label  = ['Fig A','Fig B','Fig C','Fig D']
    
    # --- Permuteer frequentie assen
    np.random.seed(seed=seed_val)
    Omega_perm_totaal = np.random.permutation( Omega_totaal )
    
    # --- Permuteer spectra
    np.random.seed(seed=seed_val)
    G_abs_perm_totaal = np.random.permutation( G_abs_totaal )
    
    # dsp_opdracht_perm01
    sol = dsp_opdracht_perm01(studentnummer,pz_fig_label,G_fig_label)
    
    return pz_fig_label,zeros_totaal,poles_totaal,G_fig_label, \
    Omega_perm_totaal, G_abs_perm_totaal

## =========================================================================== #
def dsp_opdracht051(studentnummer):
    # --- Zet de seed
    seed_val = studentnummer
    
    # Omega en K
    Omega = np.linspace(0,np.pi,100)   # Dimensieloze hoekfrequentie
    K     = 1.0                        # Versterkingsfactor
    pi    = np.pi
    
    # Invoergegevens
    zeros_val1 = [0.9*np.exp(1j*1*pi/6),0.9*np.exp(-1j*1*pi/6),
                  0.8*np.exp(1j*5*pi/6),0.8*np.exp(-1j*5*pi/6)]
    poles_val1 = [0.0,0.0,0.0,0.0]
    zeros_val2 = [0.0,0.0,0.0,0.0]
    poles_val2 = [0.9*np.exp(1j*1*pi/6),0.9*np.exp(-1j*1*pi/6),
                  0.8*np.exp(1j*5*pi/6),0.8*np.exp(-1j*5*pi/6)]
    zeros_val3 = [1.0*np.exp(1j*2*pi/6),1.0*np.exp(-1j*2*pi/6)]
    poles_val3 = [0.8*np.exp(1j*5*pi/6),0.8*np.exp(-1j*5*pi/6)]
    zeros_val4 = [0.8*np.exp(1j*5*pi/6),0.8*np.exp(-1j*5*pi/6)]
    poles_val4 = [0.0,0.0]
    
               
    # Bereken complexe frequentie-respons op eenheidsimpuls
    G1 = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val1,poles_val1)
    G2 = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val2,poles_val2)
    G3 = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val3,poles_val3)
    G4 = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val4,poles_val4)
    
    # --- Store enkele zaken in lijsten
    zeros_totaal = [zeros_val1,zeros_val2,zeros_val3,zeros_val4]
    poles_totaal = [poles_val1,poles_val2,poles_val3,poles_val4]
    Omega_totaal = [Omega,Omega,Omega,Omega]
    G_abs_totaal = [np.abs(G1),np.abs(G2),np.abs(G3),np.abs(G4)]
    pz_fig_label = ['Fig 1','Fig 2','Fig 3','Fig 4']
    G_fig_label  = ['Fig A','Fig B','Fig C','Fig D']
    
    # --- Permuteer frequentie assen
    np.random.seed(seed=seed_val)
    Omega_perm_totaal = np.random.permutation( Omega_totaal )
    
    # --- Permuteer spectra
    np.random.seed(seed=seed_val)
    G_abs_perm_totaal = np.random.permutation( G_abs_totaal )
    
    # dsp_opdracht_perm01
    sol = dsp_opdracht_perm01(studentnummer,pz_fig_label,G_fig_label)
    
    # Return output
    return pz_fig_label,zeros_totaal,poles_totaal,G_fig_label, \
    Omega_perm_totaal, G_abs_perm_totaal

## =========================================================================== #
def dsp_opdracht052(studentnummer):
    # --- Zet de seed
    seed_val = studentnummer
    
    # Omega en K
    Omega = np.linspace(0,np.pi,100)   # Dimensieloze hoekfrequentie
    K     = 1                          # Versterkingsfactor
    pi    = np.pi
    
    # Invoergegevens
    zeros_val1 = [0.3+0.4*1j,0.3-0.4*1j]
    poles_val1 = [0,0]
    zeros_val2 = [-0.3+0.4*1j,-0.3-0.4*1j]
    poles_val2 = [0,0]
    zeros_val3 = [0,0,0,0]
    poles_val3 = [0.3+0.4*1j,0.3-0.4*1j,-0.3+0.4*1j,-0.3-0.4*1j]
    zeros_val4 = [0.3+0.4*1j,0.3-0.4*1j,-0.3+0.4*1j,-0.3-0.4*1j]
    poles_val4 = [0.0,0.0]
     
    # Bereken complexe frequentie-respons op eenheidsimpuls
    G1 = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val1,poles_val1)
    G2 = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val2,poles_val2)
    G3 = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val3,poles_val3)
    G4 = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val4,poles_val4)
    
    # Schaal frequentie-respons zodanig dat G=1 voor frequentie = 0
    G1 = G1/np.abs(G1[0])
    G2 = G2/np.abs(G2[0])
    G3 = G3/np.abs(G3[0])
    G4 = G4/np.abs(G4[0])
    
    # Pas Omega as aan
    f = 20*Omega / pi
    
    # --- Store enkele zaken in lijsten
    zeros_totaal = [zeros_val1,zeros_val2,zeros_val3,zeros_val4]
    poles_totaal = [poles_val1,poles_val2,poles_val3,poles_val4]
    f_totaal     = [f,f,f,f]
    G_abs_totaal = [np.abs(G1),np.abs(G2),np.abs(G3),np.abs(G4)]
    pz_fig_label = ['Fig 1','Fig 2','Fig 3','Fig 4']
    G_fig_label  = ['Fig A','Fig B','Fig C','Fig D']
        
    # --- Permuteer frequentie assen
    np.random.seed(seed=seed_val)
    f_perm_totaal = np.random.permutation( f_totaal )
    
    # --- Permuteer spectra
    np.random.seed(seed=seed_val)
    G_abs_perm_totaal = np.random.permutation( G_abs_totaal )
    
    # dsp_opdracht_perm01
    sol = dsp_opdracht_perm01(studentnummer,pz_fig_label,G_fig_label)
    
    # Return output
    return pz_fig_label,zeros_totaal,poles_totaal,G_fig_label, \
    f_perm_totaal, G_abs_perm_totaal
    
## =========================================================================== #
def dsp_opdracht053(studentnummer):
    # --- Zet de seed
    seed_val = studentnummer
    
    # Omega en K
    Omega = np.linspace(0,np.pi,100)   # Dimensieloze hoekfrequentie
    K     = 1.0                        # Versterkingsfactor
    pi    = np.pi
    
    # Invoergegevens
    exp_phi_0_p = np.exp(+1j*pi/5)
    exp_phi_0_m = np.exp(-1j*pi/5)
    r_zeros     = [0.99,0.99,0.60,0.99]
    r_poles     = [0.60,0.80,0.99,0.99]
    zeros_val1  = [r_zeros[0]*exp_phi_0_p,r_zeros[0]*exp_phi_0_m]
    poles_val1  = [r_poles[0]*exp_phi_0_p,r_poles[0]*exp_phi_0_m]
    zeros_val2  = [r_zeros[1]*exp_phi_0_p,r_zeros[1]*exp_phi_0_m]
    poles_val2  = [r_poles[1]*exp_phi_0_p,r_poles[1]*exp_phi_0_m]
    zeros_val3  = [r_zeros[2]*exp_phi_0_p,r_zeros[2]*exp_phi_0_m]
    poles_val3  = [r_poles[2]*exp_phi_0_p,r_poles[2]*exp_phi_0_m]
    zeros_val4  = [r_zeros[3]*exp_phi_0_p,r_zeros[3]*exp_phi_0_m]
    poles_val4  = [r_poles[3]*exp_phi_0_p,r_poles[3]*exp_phi_0_m]
      
    # Bereken complexe frequentie-respons op eenheidsimpuls
    G1 = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val1,poles_val1)
    G2 = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val2,poles_val2)
    G3 = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val3,poles_val3)
    G4 = dsp.lti_freqrespons_zeros_poles(Omega,K,zeros_val4,poles_val4)
    
    # Schaal spectrum zodanig dat G[0] = 1
    G1 = G1/G1[0]
    G2 = G2/G2[0]
    G3 = G3/G3[0]
    G4 = G4/G4[0]
    
    # --- Store enkele zaken in lijsten
    zeros_totaal = [zeros_val1,zeros_val2,zeros_val3,zeros_val4]
    poles_totaal = [poles_val1,poles_val2,poles_val3,poles_val4]
    Omega_totaal = [Omega,Omega,Omega,Omega]
    G_abs_totaal = [np.abs(G1),np.abs(G2),np.abs(G3),np.abs(G4)]
    pz_fig_label = ['Fig 1','Fig 2','Fig 3','Fig 4']
    G_fig_label  = ['Fig A','Fig B','Fig C','Fig D']
    
    # --- Permuteer frequentie assen
    np.random.seed(seed=seed_val)
    Omega_perm_totaal = np.random.permutation( Omega_totaal )
    
    # --- Permuteer spectra
    np.random.seed(seed=seed_val)
    G_abs_perm_totaal = np.random.permutation( G_abs_totaal )
    
    # dsp_opdracht_perm01
    sol = dsp_opdracht_perm01(studentnummer,pz_fig_label,G_fig_label)
    
    # Return output
    return pz_fig_label,zeros_totaal,poles_totaal,G_fig_label, \
    Omega_perm_totaal, G_abs_perm_totaal
    
## =========================================================================== #
def dsp_opdracht060(studentnummer):
    # --- Zet de seed
    seed_val = studentnummer
    
    # --- Low-pass filter
    dt  = 0.005
    eps = 0.1
    Nt  = 3000
    
    ## --- Creeer een tijdsignaal
    f_inputsig   = np.arange(0,51.0,0.025)    
    f_range_laag = [[0.46,0.50]]    
    f_range_hoog = [[45.0,50.0]]     
    S_vals_laag  = [100] 
    S_vals_hoog  = [5] 
    S_laag       = dsp.real_spectrum_constante_blokken(f_inputsig, f_range_laag,
                                                       S_vals_laag)
    S_hoog       = dsp.real_spectrum_constante_blokken(f_inputsig, f_range_hoog, 
                                                       S_vals_hoog)
    # --- Maak tijdserie
    n_val  = np.arange(Nt)
    t      = n_val*dt
    fs     = 1/dt
    x_laag = dsp.real_spectrum_naar_tijdsignaal(t,f_inputsig,S_laag,seed_val)
    x_hoog = dsp.real_spectrum_naar_tijdsignaal(t,f_inputsig,S_hoog,seed_val)
    x_time = x_laag + x_hoog
    
    # Return output data
    return t,x_time,eps

## =========================================================================== #
def dsp_opdracht061(studentnummer,dt,Nt):
    # --- Zet de seed
    seed_val = studentnummer
    
    # --- Bereken enkele parameters
    fs   = 1/dt
    fnyq = 0.5*fs
    
    # ---- Creeer spectra
    f       = np.linspace(0,fnyq,300)
    f_vals1 = [[0.0,fnyq]]
    S_vals1 = [10.0]
    S1      = dsp.real_spectrum_constante_blokken(f,f_vals1,S_vals1)
    f_vals2 = [[0.0,0.4*fnyq],[0.4*fnyq,0.6*fnyq],[0.6*fnyq,0.8*fnyq],
               [0.8*fnyq,fnyq]]
    S_vals2 = [10.0,6.0,4.0,2.5]
    S2      = dsp.real_spectrum_constante_blokken(f,f_vals2,S_vals2)
    
    # --- Maak tijdseries
    n_val  = np.arange(Nt)
    t      = n_val*dt
    x1     = dsp.real_spectrum_naar_tijdsignaal(t,f,S1,seed_val)
    y2     = dsp.real_spectrum_naar_tijdsignaal(t,f,S2,seed_val)
    
    # Return output data
    return t,x1,y2

## =========================================================================== #
def dsp_opdracht062(studentnummer):
    
    # --- Zet de seed en enkele andere parameters
    seed_val = studentnummer
    f_0      = 3.0 + dsp.studentnummer_one(studentnummer, 3 )
    f_c      = 2.0
    dt       = 2.5e-3
    Tend     = 6.9 / f_0
    Nt       = np.int(Tend/dt)
    
    # --- Bereken enkele parameters
    fs   = 1/dt
    fnyq = 0.5*fs
    
    # ---- Creeer spectra
    f       = np.linspace(0,fnyq,Nt)
    f_vals  = [[f_0+5*f_c,0.5*fnyq],[0.5*fnyq,0.8*fnyq]]
    S_vals  = [100.0,30.0]
    S       = dsp.real_spectrum_constante_blokken(f,f_vals,S_vals)
    
    # --- Maak tijdserie obv spectrum
    n_val     = np.arange(Nt)
    t         = n_val*dt
    x_gewenst = dsp.real_spectrum_naar_tijdsignaal(t,f,S,seed_val)
    
    # --- Maak stoorsignaal
    x_stoor = 4.0 * np.sin(2*np.pi*f_0*t) 
    
    # Return output data
    return t,x_gewenst,x_stoor,f_0,f_c

## =========================================================================== #
def dsp_opdracht063(studentnummer):
    
    # --- Zet de seed en enkele andere parameters
    seed_val = studentnummer
    f_0      = 4.0 
    f_c      = 2.0
    dt       = 2.5e-3
    Tend     = 6.9 / f_0
    Nt       = np.int(Tend/dt)
    
    # --- Bereken enkele parameters
    fs   = 1/dt
    fnyq = 0.5*fs
    
    # ---- Creeer spectra
    f       = np.linspace(0,fnyq,Nt)
    f_vals  = [[f_0+5*f_c,0.5*fnyq],[0.5*fnyq,0.8*fnyq]]
    S_vals  = [50.0,10.0]
    S       = dsp_real_spectrum_constante_blokken(f,f_vals,S_vals)
    
    # --- Maak tijdserie obv spectrum
    n_val     = np.arange(Nt)
    t         = n_val*dt
    x_gewenst = dsp_real_spectrum_naar_tijdsignaal(t,f,S,seed_val)
    
    # --- Maak stoorsignaal
    x_stoor = 1.3 * np.sin(2*np.pi*f_0*t + 0.9) 
    
    # --- Bepaal totale signaal
    x_totaal = x_gewenst + x_stoor
    
    # Return output data
    return t,x_totaal,f_0,f_c

## =========================================================================== #
def dsp_opdracht_perm01(studentnummer,t_fig_label,S_fig_label):
    
    # --- Zet de seed
    seed_val = studentnummer
    
    # --- Permuteer tijdlabels (voor bepaal resultaat)
    np.random.seed(seed=seed_val)
    t_fig_label_perm = np.random.permutation( t_fig_label )
    
    # --- Zet resultaat in strings
    sol_string0 = S_fig_label[0] + ' hoort bij ' + t_fig_label_perm[0]
    sol_string1 = S_fig_label[1] + ' hoort bij ' + t_fig_label_perm[1]
    sol_string2 = S_fig_label[2] + ' hoort bij ' + t_fig_label_perm[2]
    sol_string3 = S_fig_label[3] + ' hoort bij ' + t_fig_label_perm[3]
    
    sol = [sol_string0,sol_string1,sol_string2,sol_string3]
    
    # Return data
    return sol

    
    