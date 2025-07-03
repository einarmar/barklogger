#!/usr/bin/env python3

import pyaudio
import numpy as np
import time
import csv
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
import signal

class SoundLevelMonitor:
    def __init__(self, sample_rate=44100, chunk_size=4096, log_interval=5):
        """
        Args:
            sample_rate: Audio sample rate (Hz)
            chunk_size: Number of samples per chunk
            log_interval: How often to log readings (seconds)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.log_interval = log_interval
        self.running = False
        
        self.audio = pyaudio.PyAudio()
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"sound_levels_{timestamp}.csv"
        
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['datetime', 'decibels_dba', 'max_db', 'min_db'])
    
    def calculate_db(self, audio_data):
        """
        Calculate decibel level from audio data
        Uses A-weighting approximation for better representation of human/dog hearing
        """
        # Convert to float and normalize
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Calculate RMS (Root Mean Square)
        rms = np.sqrt(np.mean(audio_float**2))
        
        # Avoid log(0) by setting minimum value
        if rms < 1e-10:
            rms = 1e-10
        
        # Convert to decibels (using reference level for microphone input)
        # This is calibrated for typical computer microphones
        db = 20 * np.log10(rms) + 94  # 94 dB reference for calibration
        
        # Apply A-weighting approximation (simplified)
        # A-weighting reduces low and very high frequencies
        db_a = db - 2  # Simplified A-weighting adjustment
        
        return max(0, db_a)  # Don't return negative dB values
    
    def list_audio_devices(self):
        print("Available audio input devices:")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  {i}: {info['name']} - {info['maxInputChannels']} channels")
    
    def start_monitoring(self, device_index=None):
        try:
            if device_index is None:
                device_index = self.audio.get_default_input_device_info()['index']
            
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
            
            print(f"Starting sound level monitoring...")
            print(f"Logging to: {self.csv_filename}")
            print(f"Log interval: {self.log_interval} seconds")
            print("Press Ctrl+C to stop")
            
            self.running = True
            start_time = time.time()
            readings_buffer = []
            
            while self.running:
                try:
                    audio_data = np.frombuffer(
                        stream.read(self.chunk_size, exception_on_overflow=False),
                        dtype=np.int16
                    )
                    
                    db_level = self.calculate_db(audio_data)
                    readings_buffer.append(db_level)
                    
                    if time.time() - start_time >= self.log_interval:
                        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        avg_db = np.mean(readings_buffer)
                        max_db = np.max(readings_buffer)
                        min_db = np.min(readings_buffer)
                        
                        with open(self.csv_filename, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([datetime_str, f"{avg_db:.1f}", 
                                           f"{max_db:.1f}", f"{min_db:.1f}"])
                        
                        print(f"{datetime_str} - Avg: {avg_db:.1f} dB, Max: {max_db:.1f} dB, Min: {min_db:.1f} dB")
                        
                        start_time = time.time()
                        readings_buffer = []
                
                except Exception as e:
                    print(f"Error reading audio: {e}")
                    time.sleep(0.1)
            
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.list_audio_devices()
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
    
    def stop_monitoring(self):
        self.running = False
        print(f"\nStopped monitoring. Data saved to: {self.csv_filename}")
    
    def plot_data(self, csv_file=None):
        if csv_file is None:
            csv_file = self.csv_filename
        
        try:
            # Read the CSV data
            df = pd.read_csv(csv_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            plt.plot(df['datetime'], df['decibels_dba'], label='Average dB', linewidth=1)
            plt.fill_between(df['datetime'], df['min_db'], df['max_db'], 
                           alpha=0.3, label='Min/Max Range')
            
            plt.xlabel('Time')
            plt.ylabel('Sound Level (dB)')
            plt.title('Dog Barking Sound Level Monitor')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plot_filename = csv_file.replace('.csv', '_plot.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {plot_filename}")
            
            # Show plot
            plt.show()
            
        except Exception as e:
            print(f"Error creating plot: {e}")
    
    def cleanup(self):
        self.audio.terminate()

def signal_handler(sig, frame, monitor):
    monitor.stop_monitoring()
    monitor.cleanup()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Dog Barking Sound Level Monitor')
    parser.add_argument('--list-devices', action='store_true', 
                       help='List available audio input devices')
    parser.add_argument('--device', type=int, default=None,
                       help='Audio input device index to use')
    parser.add_argument('--interval', type=int, default=5,
                       help='Logging interval in seconds (default: 5)')
    parser.add_argument('--plot', type=str, default=None,
                       help='Create plot from existing CSV file')
    
    args = parser.parse_args()
    
    monitor = SoundLevelMonitor(log_interval=args.interval)
    
    if args.list_devices:
        monitor.list_audio_devices()
        monitor.cleanup()
        return
    
    if args.plot:
        monitor.plot_data(args.plot)
        monitor.cleanup()
        return
    
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, monitor))
    
    try:
        monitor.start_monitoring(device_index=args.device)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.cleanup()

if __name__ == "__main__":
    main()
