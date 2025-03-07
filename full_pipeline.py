import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
from scipy import signal
from scipy.signal import welch
from scipy.stats import wasserstein_distance, entropy
import gym
from gym import spaces

# Create a timestamp for the results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# 1. Define the RL Environment for Signal Filtering
class SignalFilterEnv(gym.Env):
    def __init__(self, raw_signal, sample_rate):
        super(SignalFilterEnv, self).__init__()
        self.raw_signal = raw_signal
        self.fs = sample_rate
        self.signal_length = len(raw_signal)
        
        # Define action and observation space
        # Action: [low_cut_freq, high_cut_freq, gain]
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.2, 0.1]),
            high=np.array([30.0, 100.0, 10.0]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.signal_length,),
            dtype=np.float32
        )
        
        # Reference EEG signal properties
        self.reference_psd = self.compute_psd(raw_signal)
    
    def compute_psd(self, signal):
        """Compute power spectral density"""
        freqs, psd = welch(signal, self.fs, nperseg=min(1024, len(signal)))
        return psd
    
    def apply_filters(self, action):
        """Apply bandpass filter with given parameters"""
        lowcut, highcut, gain = action[0], action[1], action[2]
        # Ensure highcut > lowcut
        if lowcut >= highcut:
            lowcut, highcut = min(lowcut, highcut), max(lowcut, highcut) + 1.0
            
        nyq = 0.5 * self.fs
        low = max(0.1, lowcut) / nyq
        high = min(0.99, highcut / nyq)
        
        # Design and apply bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, self.raw_signal)
        
        # Apply gain
        max_amp = np.max(np.abs(self.raw_signal))
        amplified_signal = np.clip(gain * filtered_signal, -max_amp, max_amp)
        
        return amplified_signal
    
    def calculate_reward(self, filtered_signal):
        """Calculate reward based on signal quality"""
        # Compute PSD of filtered signal
        psd_filtered = self.compute_psd(filtered_signal)
        
        # Ensure PSDs are of the same length
        min_length = min(len(self.reference_psd), len(psd_filtered))
        ref_psd = self.reference_psd[:min_length]
        filt_psd = psd_filtered[:min_length]
        
        # Calculate spectral distance
        spectral_distance = wasserstein_distance(ref_psd, filt_psd)
        
        # Calculate noise reduction
        noise = self.raw_signal - filtered_signal
        noise_power = np.mean(noise**2)
        signal_power = np.mean(filtered_signal**2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Compare entropy
        entropy_diff = abs(entropy(ref_psd) - entropy(filt_psd))
        
        # Reward is higher for better SNR and less distortion
        reward = snr - spectral_distance - entropy_diff
        return reward, snr
    
    def step(self, action):
        """Apply action and return new state, reward, done flag, and info"""
        filtered_signal = self.apply_filters(action)
        reward, snr = self.calculate_reward(filtered_signal)
        done = False  # In continuous filtering, we don't have episode termination
        info = {'SNR': snr}
        return filtered_signal, reward, done, info
    
    def reset(self):
        """Reset the environment"""
        return self.raw_signal

# 2. Feature Extraction from EEG
def extract_features(eeg_signal, fs):
    """Extract features from EEG signal for seizure detection"""
    # Time domain features
    mean = np.mean(eeg_signal)
    std = np.std(eeg_signal)
    min_val = np.min(eeg_signal)
    max_val = np.max(eeg_signal)
    range_val = max_val - min_val
    energy = np.sum(eeg_signal**2)
    line_length = np.sum(np.abs(np.diff(eeg_signal)))
    
    # Frequency domain features
    freqs, psd = welch(eeg_signal, fs, nperseg=min(1024, len(eeg_signal)))
    
    # Extract power in different frequency bands
    delta_idx = np.logical_and(freqs >= 0.5, freqs < 4)
    theta_idx = np.logical_and(freqs >= 4, freqs < 8)
    alpha_idx = np.logical_and(freqs >= 8, freqs < 13)
    beta_idx = np.logical_and(freqs >= 13, freqs < 30)
    gamma_idx = np.logical_and(freqs >= 30, freqs < 100)
    
    delta_power = np.mean(psd[delta_idx]) if np.any(delta_idx) else 0
    theta_power = np.mean(psd[theta_idx]) if np.any(theta_idx) else 0
    alpha_power = np.mean(psd[alpha_idx]) if np.any(alpha_idx) else 0
    beta_power = np.mean(psd[beta_idx]) if np.any(beta_idx) else 0
    gamma_power = np.mean(psd[gamma_idx]) if np.any(gamma_idx) else 0
    
    # Combine all features into a dictionary
    features = {
        'mean_amplitude': mean,
        'std_amplitude': std,
        'min_amplitude': min_val,
        'max_amplitude': max_val,
        'range_amplitude': range_val,
        'energy': energy,
        'line_length': line_length,
        'delta_power': delta_power,
        'theta_power': theta_power,
        'alpha_power': alpha_power,
        'beta_power': beta_power,
        'gamma_power': gamma_power,
        'delta_theta_ratio': delta_power / (theta_power + 1e-10),
        'theta_alpha_ratio': theta_power / (alpha_power + 1e-10),
        'alpha_beta_ratio': alpha_power / (beta_power + 1e-10),
        'mean_correlation': 0.5,  # Placeholder for more advanced features
        'correlation_std': 0.1,    # Placeholder for more advanced features
        'amplitude_ratio': max_val / (mean + 1e-10),
        'frequency_change': beta_power / (alpha_power + 1e-10),
        'entropy': entropy(psd) if len(psd) > 0 else 0
    }
    
    return features

# 3. Generate synthetic EEG data for training
def generate_synthetic_eeg(duration=60, fs=625.34, has_seizure=False):
    """Generate synthetic EEG data with optional seizure-like activity"""
    t = np.linspace(0, duration, int(duration * fs))
    
    # Background activity
    background = (
        10 * np.sin(2 * np.pi * 10 * t) +  # Alpha rhythm (~10 Hz)
        5 * np.sin(2 * np.pi * 5 * t) +    # Theta rhythm (~5 Hz)
        3 * np.sin(2 * np.pi * 20 * t)     # Beta rhythm (~20 Hz)
    )
    
    # Add noise
    noise = np.random.normal(0, 2, len(t))
    signal = background + noise
    
    if has_seizure:
        # Add seizure-like activity (high-frequency oscillations)
        seizure_start = int(len(t) * 0.4)  # Start at 40% of the signal
        seizure_duration = int(len(t) * 0.2)  # Last for 20% of the signal
        seizure = 20 * np.sin(2 * np.pi * 20 * t[seizure_start:seizure_start + seizure_duration])
        signal[seizure_start:seizure_start + seizure_duration] += seizure
    
    return signal

# 4. Function to simulate live EEG data streaming
def receive_new_eeg_segment(file_path=None, position=None, segment_size=1000):
    """Simulate receiving a new segment of EEG data"""
    # Initialize static variables on first call
    if not hasattr(receive_new_eeg_segment, "data"):
        try:
            if file_path and os.path.exists(file_path):
                data = pd.read_csv(file_path, comment='#')
                receive_new_eeg_segment.data = data['Channel 1 (V)'].values
            else:
                # Generate synthetic data if file is not available
                print("No data file provided or found. Using synthetic data.")
                receive_new_eeg_segment.data = generate_synthetic_eeg()
        except Exception as e:
            print(f"Error reading EEG data: {e}. Using synthetic data.")
            receive_new_eeg_segment.data = generate_synthetic_eeg()
        
        receive_new_eeg_segment.position = 0
        receive_new_eeg_segment.fs = 625.34  # Sampling rate
    
    # Use provided position if specified
    if position is not None:
        receive_new_eeg_segment.position = position
    
    # Extract segment
    data_length = len(receive_new_eeg_segment.data)
    end_position = min(receive_new_eeg_segment.position + segment_size, data_length)
    segment = receive_new_eeg_segment.data[receive_new_eeg_segment.position:end_position]
    
    # Update position for next call
    receive_new_eeg_segment.position = end_position
    
    # Loop back to beginning if we've reached the end
    if receive_new_eeg_segment.position >= data_length:
        receive_new_eeg_segment.position = 0
    
    return segment, receive_new_eeg_segment.fs

# 5. Train seizure detection model
def train_seizure_classifier(num_samples=1000):
    """Train a seizure detection classifier using synthetic data"""
    # Generate training data
    X = []
    y = []
    
    for i in range(num_samples):
        has_seizure = np.random.random() > 0.7  # 30% of samples have seizures
        signal = generate_synthetic_eeg(duration=10, has_seizure=has_seizure)
        features = extract_features(signal, fs=625.34)
        X.append(list(features.values()))
        y.append(has_seizure)
    
    X = np.array(X)
    y = np.array(y)
    
    # Create and train classifier
    classifier = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    classifier.fit(X, y)
    return classifier

# 6. The Complete Pipeline
def run_complete_pipeline(data_file=None, duration=60, segment_size=1000, overlap=500):
    """Run the complete EEG processing pipeline"""
    print("Initializing the comprehensive EEG pipeline...")
    
    # Train seizure classifier
    print("Training seizure classifier...")
    classifier = train_seizure_classifier()
    
    # Initialize RL environment with first segment
    print("Initializing RL-based EEG filtering environment...")
    initial_segment, fs = receive_new_eeg_segment(data_file, segment_size=segment_size)
    env = SignalFilterEnv(initial_segment, fs)
    
    # Find optimal filter parameters using RL
    print("Training RL agent to find optimal filter parameters...")
    num_steps = 10
    best_action = None
    best_reward = -float('inf')
    actions_log = []
    
    for step in range(num_steps):
        # Progress from low to high frequencies
        action = np.array([1.0 + step * 2, 40.0 + step * 2, 1.5])
        filtered_signal, reward, _, info = env.step(action)
        
        actions_log.append({
            'action': action.tolist(),
            'reward': reward,
            'snr': info['SNR']
        })
        
        if reward > best_reward:
            best_reward = reward
            best_action = action
        
        print(f"Step {step + 1}, Action: {action}, SNR: {info['SNR']:.2f} dB")
    
    print(f"\nBest filter parameters found:")
    print(f"Low cut: {best_action[0]:.2f} Hz")
    print(f"High cut: {best_action[1]:.2f} Hz")
    print(f"Gain: {best_action[2]:.2f}")
    
    # Setup real-time plotting
    plt.ion()
    fig = plt.figure(figsize=(12, 8))
    
    # Main processing loop
    print("\nStarting real-time EEG processing...")
    start_time = time.time()
    processing_log = []
    seizure_count = 0
    
    while time.time() - start_time < duration:
        # Get new EEG segment
        raw_segment, fs = receive_new_eeg_segment(data_file, segment_size=segment_size)
        
        # Apply RL-optimized filter
        env.raw_signal = raw_segment
        filtered_signal, reward, _, info = env.step(best_action)
        
        # Extract features and detect seizures
        features = extract_features(filtered_signal, fs)
        feature_vector = np.array([list(features.values())])
        
        # Make prediction
        seizure_prob = classifier.predict_proba(feature_vector)[0, 1]
        seizure_detected = classifier.predict(feature_vector)[0]
        
        if seizure_detected:
            seizure_count += 1
        
        # Log results
        current_time = time.time() - start_time
        processing_log.append({
            'time': current_time,
            'snr': info['SNR'],
            'seizure_probability': float(seizure_prob),
            'seizure_detected': bool(seizure_detected)
        })
        
        # Update plots
        plt.clf()
        t = np.arange(len(raw_segment)) / fs
        
        # Raw signal
        plt.subplot(3, 1, 1)
        plt.plot(t, raw_segment, 'b-', label='Raw EEG')
        plt.title('Real-time EEG Processing')
        plt.ylabel('Amplitude (µV)')
        plt.legend()
        
        # Filtered signal
        plt.subplot(3, 1, 2)
        plt.plot(t, filtered_signal, 'g-', label='Filtered EEG')
        plt.ylabel('Amplitude (µV)')
        plt.legend()
        
        # Frequency band powers
        plt.subplot(3, 1, 3)
        bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        powers = [features['delta_power'], features['theta_power'],
                 features['alpha_power'], features['beta_power'],
                 features['gamma_power']]
        plt.bar(bands, powers)
        plt.ylabel('Power')
        
        if seizure_detected:
            plt.figtext(0.5, 0.01, 'SEIZURE DETECTED!',
                       ha='center', color='red', fontsize=12,
                       bbox={'facecolor': 'yellow', 'alpha': 0.5})
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
        # Print status
        print(f"Time: {current_time:.1f}s, SNR: {info['SNR']:.2f} dB, "
              f"Seizure probability: {seizure_prob:.3f}")
        
        # Small delay to simulate real-time processing
        time.sleep(0.1)
    
    plt.ioff()
    
    # Save results
    print(f"\nProcessing complete!")
    print(f"Detected {seizure_count} potential seizure events.")
    
    # Save processing log
    with open(os.path.join(results_dir, 'processing_log.json'), 'w') as f:
        json.dump(processing_log, f, indent=2)
    
    # Save classifier and filter parameters
    joblib.dump(classifier, os.path.join(results_dir, 'seizure_classifier.joblib'))
    with open(os.path.join(results_dir, 'filter_params.json'), 'w') as f:
        json.dump({
            'best_action': best_action.tolist(),
            'best_reward': float(best_reward),
            'actions_log': actions_log
        }, f, indent=2)
    
    # Final summary plot
    plt.figure(figsize=(12, 6))
    
    times = [entry['time'] for entry in processing_log]
    snrs = [entry['snr'] for entry in processing_log]
    probs = [entry['seizure_probability'] for entry in processing_log]
    detections = [entry['seizure_detected'] for entry in processing_log]
    
    plt.subplot(2, 1, 1)
    plt.plot(times, snrs, 'b-')
    plt.title('Signal Quality over Time')
    plt.ylabel('SNR (dB)')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(times, probs, 'g-')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Detection threshold')
    for t, d in zip(times, detections):
        if d:
            plt.axvline(x=t, color='r', alpha=0.3)
    plt.title('Seizure Detection Results')
    plt.xlabel('Time (s)')
    plt.ylabel('Seizure Probability')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'summary.png'))
    plt.show()
    
    return {
        'classifier': classifier,
        'best_filter_params': best_action,
        'processing_log': processing_log,
        'results_dir': results_dir
    }

if __name__ == "__main__":
    # Run the pipeline
    # If you have a real EEG data file, provide its path, otherwise synthetic data will be used
    results = run_complete_pipeline(data_file=None, duration=30)  # Run for 30 seconds
    print(f"\nResults saved to: {results['results_dir']}")
