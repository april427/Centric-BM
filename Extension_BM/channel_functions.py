# %%
import numpy as np
import math


# %%
def dft_codebook(n, m):
    """
    Creates a DFT codebook for a ULA BS.

    Args:
        n (int): The number of antenna elements in the ULA.
        m (int): The number of beams in the codebook.
        angles_deg (list): A list of angles (in degrees) to form the beams.
        d (float): The antenna spacing (in wavelengths).

    Returns:
        codebook (numpy.ndarray): A numpy array of shape (n, m) representing the codebook.
    """
    dft_matrix = np.ones((m, m), dtype=np.complex64) * complex(1, 0) 
    for i in range(1, m+1):
        dft_matrix[i-1,:] = 1/np.sqrt(n) * np.array([np.exp(-1j * np.pi * j *(2 * i - 1 - m)/m) for j in range(n)])
    return dft_matrix

def compute_AoD_at_BS(BS_pos, UE_pos):
    """
        Computes AoD at ULA BS given positions.

        Args:
            BS_pos (numpy.ndarray): Float array with the coordinates of the BS.
            UE_pos [numpy.ndarray]: Float array with the coordinates of the UE.
        Returns:
            AoD (numpy.ndarray): Angle of departure in radians.
        """
    # Compute angle of departure at BS
    phi_radi = np.arctan2( UE_pos[1] - BS_pos[1], - BS_pos[0] + UE_pos[0]) 
    AoD = phi_radi#np.array([phi_radi])
    if AoD < 0:
        AoD = np.add(AoD,np.pi)
    return AoD

def compute_ULA_response(M, AoD):
    # Compute array response
    a_BS = 1/np.sqrt(M) * np.array([np.exp(-1j * np.pi * i * np.cos(AoD)) for i in range(M)])
    return a_BS

def compute_mmWave_LoS_NLoS_channel(M, a_BS, d, wavelength, blockage_coefficient=1):
    """
        Compute mmWave LoS channel. Dimension is (1,M)
        Consider Rayleigh block fading. * 1/np.sqrt(2)* complex(np.random.normal(0, 1), np.random.normal(0, 1))

        Args:  
            M (int): The number of antenna elements in the ULA.
            a_BS (numpy.ndarray): The array response at the BS.
            d (float): The distance between the BS and the UE.
            blockage_coefficient (float): The blockage coefficient.  
             
    """
    mmWave_LOS_channel = np.sqrt(M) * wavelength / (4 *np.pi * d) * blockage_coefficient  * np.conj(a_BS).T

    non_Los_channel = np.sqrt(1/2) * wavelength / (4 *np.pi * d) * blockage_coefficient * (np.random.randn(1, M) + 1j * np.sqrt(1/2) * np.random.randn(1, M))

    K_db = 10
    K = 10**(K_db/10)

    mmWave_LOS_channel = np.sqrt(K/(1+K))*mmWave_LOS_channel + np.sqrt(1/(1+K))*non_Los_channel

    return mmWave_LOS_channel

def freqency_communication_channel(Nsym, Nc, N, Nt, sigma_2_alpha, tau=None, fai1=None):
        # Nsym -- Number of OFDM symbols
        # Nc -- Number of subcarriers
        # N -- Number of paths 
        # sigma_2_alph -- Variance of the amplitude of each path 
        # Nt -- Number of transmit antennas
        # tau -- Delay of each path
        # fai1 -- Angle of departure of each path

    Tau = Nc  # Maximum delay spread
    if tau is None:
        tau = np.random.rand(1,N)*Tau 

    #n_r = np.mat(range(Nr)).reshape(Nr,1)
    n_t = np.array(range(Nt))
    if fai1 is None:
        fai1 = np.random.rand(N)*2*np.pi # Random AoD
    else:
        fai1 = np.array([fai1])
    A_t = np.zeros([Nt,N],dtype = complex)
    alpha = np.zeros([N],dtype = complex)
    H_f = np.array(np.zeros([Nt, Nsym,Nc],dtype = complex))
    for i in range(N):
        A_t[:,i] = 1/np.sqrt(Nt) * np.exp(-1j*np.pi*n_t*np.sin(fai1[i])) # Normalized steering vector, it is the same as the one in frequency_sensing_channel 
        aa = np.sqrt(sigma_2_alpha/N)
        alpha[i] = aa 
            
    for k in range(Nc):
        P = alpha*np.exp(-1j*2*np.pi*tau*k/Nc)
        if np.shape(P) != ():
            P = np.diag(P)
        H_f[:,:,k] = np.tile(np.dot(np.conj(A_t), P), [1,Nsym]) # Conjugate of A_t so H can be directly used ApplyOFDMChannel
        
    return H_f,A_t,alpha

def freqency_sensing_channel(batch_size, Nsym, Nc, Nt, N_UTs, subcarrier_spacing, ofdm_symbol_duration, reflection_coefficient, ut_velocities, wavelength, tau, fai1=None):
        # Nsym -- Number of OFDM symbols
        # Nc -- Number of subcarriers
        # N = 1 -- 1 path is considered for the sensing channel 
        # reflection_coefficient -- path coefficient, NO need to take squre root, [1, N_UE] 
        # Nt -- Number of transmit antennas
        # tau -- Two-way delay of each path [1, N_UE]
        # fai1 -- Angle of departure of each path [1, N_UE]

    # Tau = Nc  # Maximum delay spread

    n_t = np.array(range(Nt)).reshape(Nt,1)
    if fai1 is None:
        fai1 = np.random.rand(1)*2*np.pi # Random angles of departure
    else:
        fai1 = np.array([fai1])
    A_t = np.array(np.zeros([Nt, 1, N_UTs],dtype = complex))
    H_f = np.array(np.zeros([batch_size, 1, Nt, 1, Nt, Nsym, Nc],dtype = complex))

    omega_A = - np.pi * np.sin(fai1) # batch_size x N_UTs
    omega_R = - 2*np.pi * subcarrier_spacing* tau # batch_size x N_UTs
    omega_v = 4*np.pi * ofdm_symbol_duration * ut_velocities[:,:,0] / wavelength # batch_size x N_UTs   

    A_t = (1/np.sqrt(Nt)) * np.exp(1j*np.kron(omega_A, n_t)).reshape((Nt, 1, N_UTs), order = 'F')

    # Vectorized computation
    reflection_coefficient = reflection_coefficient[:, :, np.newaxis, np.newaxis, np.newaxis]
    omega_R = omega_R[:, :, np.newaxis, np.newaxis, np.newaxis]
    omega_v = omega_v[:, :, np.newaxis, np.newaxis, np.newaxis]

    exp_omega_R = np.exp(1j * np.arange(Nc) * omega_R)
    exp_omega_v = np.exp(1j * np.arange(Nsym)[:, np.newaxis] * omega_v)

    for u in range(N_UTs):
        A_t_conj = np.conj(A_t[:, 0, u])
        A_t_outer = np.outer(A_t[:, 0, u], A_t_conj)
        H_f[0, 0, :, 0, :, :, :] += reflection_coefficient[0, u] * A_t_outer[:, :, np.newaxis, np.newaxis] * exp_omega_R[0, u] * exp_omega_v[0, u]
    for b in range(1, batch_size):
        H_f[b, :] = H_f[0, :]
    
    return H_f, A_t
    
def compute_mono_static_channel(M, AoD, reflection_coefficient):
    """
        Compute monostatic echo channel from one UE. LoS. (M, M)
    """
    a_BS = compute_ULA_response(M, AoD)
    mono_LoS_channel = M * reflection_coefficient * np.outer(a_BS, np.conj(a_BS))
    return mono_LoS_channel


def compute_communication_rate(noiseless_output, signal, noise_power, N_UEs, N):
    """
        Compute the communication rate, consider inter user interference.
        Args:
            noiseless_output (numpy.ndarray): The noiseless output of the channel.
            noise_power (float): The noise power.
            N_UEs (int): The number of UEs.
            N (int): The number of samples within one TTI.
    """
    # Generate complex Gaussian noise
    noise = (np.random.normal(0, np.sqrt(noise_power/2), size = (N_UEs, N)) + 1j* np.random.normal(0, np.sqrt(noise_power/2), size = (N_UEs, N)))
    
    interference = noiseless_output - signal
    # Compute the communication rate
    rate = np.log2(1 + np.abs(signal)**2 / ( np.abs(interference)**2 + np.abs(noise) ** 2 ))

    # Compute SINR in dB
    sinr = 10*np.log10(np.abs(signal)**2 / ( np.abs(interference)**2 + np.abs(noise) ** 2 ))
    aa = np.mean(sinr,axis=1) 
    aa[np.isinf(aa)] = -20

    interference_db = np.mean(10*np.log10(np.abs(interference)**2),axis = 1) if np.abs(np.abs(interference)**2).any() else np.zeros((N_UEs,))
    interference_db[np.isnan(interference_db)] = 50
    interference_db[np.isinf(interference_db)] = 50
    rate = np.mean(rate, axis=1)
    return rate, aa, interference_db

def estimate_AoD_from_echo(y_BS, waveform, N_UEs, N, N_fft, number_of_antennas):
    """
        Estimate AoDs from the echo.
        Args:
            y_BS (numpy.ndarray): The received signal at the BS.
            waveform (numpy.ndarray): The waveform.
            N_UEs (int): The number of UEs.
            N (int): The number of samples within one TTI.
            N_fft (int): The number of samples in the FFT.
            env (Environment): The environment.
    """
    AoD_est = np.zeros(N_UEs)
    for u in range(N_UEs):
        nitta = np.matmul(y_BS, waveform[u, :])
        # estimate AoD
        w_padded = np.concatenate((nitta, np.zeros(N_fft - number_of_antennas)))
        w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
        # Map the FFT bins to angles in radians
        theta_bins = np.arccos(np.linspace(-1, 1, N_fft)) # in radians
        AoD_est[u] = np.pi - theta_bins[np.argmax(w_fft_dB)]
    return AoD_est

def dynamic_power_allocation(d, xi, Pt, Pt_mW, ue_type, threshold, N_UEs, datapool):
    """
        Compute dynamic power allocation.
        Args:
            d (float): The distance between the BS and the UE.
            xi (float): The blockage coefficient.
            Pt_mW (float): The total transmit power in mW.
        
    """

    counter = -1
    if d.ndim != 1:
        d = np.squeeze(d, 0)
    d = d * (datapool != 0)
    d[xi==0] = 0
    if np.all(d == 0):
        return Pt
    sorted_user = np.argsort(d)
    weak_ue_ind = sorted_user[counter]
    if ue_type[weak_ue_ind] == 1: # CU
        Pt[weak_ue_ind] = threshold[1] / xi[weak_ue_ind]
    else: # SU
        Pt[weak_ue_ind] = threshold[0] / xi[weak_ue_ind]
    for u in range(N_UEs):
        Pt[u] = Pt[weak_ue_ind] * (xi[u]/xi[weak_ue_ind]) * (threshold[ue_type[weak_ue_ind]]/threshold[ue_type[u]])
    Pt[np.isinf(Pt)] = Pt_mW
    Pt = Pt * (datapool != 0)

    if np.sum(Pt) - Pt_mW < 0: # more power is available
        Pt[datapool!=0] = Pt[datapool!=0] + (Pt_mW - np.sum(Pt)) / (np.count_nonzero(d)) # equal allocation of the remaining power
    else:
        while np.sum(Pt) - Pt_mW > 0 :
            Pt[sorted_user[counter]] -= (np.sum(Pt) - Pt_mW)
            if Pt[sorted_user[counter]] < 0:
                Pt[sorted_user[counter]] = 0 
            if counter == -N_UEs:
                Pt[sorted_user[counter]] = Pt_mW
                return Pt
            counter -= 1
    
    if np.any(np.isnan(Pt)):
        Pt[np.isnan(Pt)] = 0
        Pt[~np.isnan(Pt)] = Pt_mW / np.count_nonzero(Pt[~np.isnan(Pt)])

    return Pt

# ausiliary functions: convert decimal to base b
def base(b,n, N_UEs):
    size = math.ceil(math.log(max(1,n),b))
    convert_result = [place
        for i in range(size,-1,-1)
        if (place := n//b**i%b) or i<size] or [0]
    if len(convert_result) < N_UEs:
        convert_result = [0]*(N_UEs-len(convert_result)) + convert_result
    return convert_result

# %%
def hyperbolic_func(x, A1, A2, c):
    '''
    hyperbolic reward function
    Args:
        x (float): input value
        A (float): scaling parameter how much it closes to c
        c (float): location c 
    '''
    x = np.atleast_1d(x)
    y=np.zeros(np.size(x))
    y[x>=c] = 1.5*(1 - np.exp(- A1 * (x[x>=c] - c)))
    y[x<c] = - 0.5
    #y[x<c] = np.exp(A1 * (x[x<c] - c)) -1
    b = 12
    y[x>=c+b] = np.exp(-A2*(x[x>=c+b] - c - b )) 
    
    # y[x>=c] = (2 - np.exp(A1 * (x[x>=c] - c)))
    # y[x<c] = 0 
    # y[y<0] = 0
    return y

#%%

# %%
# import numpy as np
# import matplotlib.pyplot as plt
# x = np.linspace(0, 40, 1000)
# A1 = 0.8
# A2 = 0
# c= 10
# b = 12
# y = hyperbolic_func(x, A1, A2, c)
# textstr = '\n'.join(( r'$y = exp(- %.1f( x - %.1f)), x >= %.1f$' % (A2, c+b, c+b),
#     r'$y = (1 - exp(- %.1f( x - %.1f)), x >= %.1f$' % (A1, c, c),
#     r'$y = -1, x < %.1f$' % (c)))
# fig, ax = plt.subplots()
# ax.plot(x, y)
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax.text(16, -0.4, textstr)
# ax.grid(True, which='both')
# ax.axhline(y=0, color='k')
# ax.axvline(x=0, color='k')
#%%


